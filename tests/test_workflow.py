import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.data import ROOT, FEATURES, NUMERIC, CLASSES, build_data, label_costs, validate_predictors
from src.classify import evaluate, pipeline_for
from src.inference import assess, load_artifacts, neighbour_vote
from src.ai_interpretation import AIInterpretationError, build_ai_context, generate_ai_interpretation, explanation_key, get_ai_settings


# These tests check mathematical and data-contract invariants, not a desired
# accuracy target. Real saved artifacts provide integration evidence; small
# synthetic inputs isolate edge cases such as invalid charges and zero distance.
class WorkflowTests(unittest.TestCase):
    # unittest runs setUpClass once for this test class. Share read-only artifacts
    # and a sample profile so individual checks avoid reloading the same files.
    @classmethod
    # Load the deployed bundle and original cleaned records to compare actual
    # application behaviour with saved split IDs, predictions and metrics.
    def setUpClass(cls):
        cls.bundle, cls.results = load_artifacts()
        cls.data, cls.audit = build_data()
        cls.profile = dict(age=35, bmi=25., children=0, sex='female', smoker='no', region='northeast')

    # Verify the supplied dataset audit and separation of training/test records.
    # Recalculate thresholds only for verification; this is not model selection.
    def test_cleaning_split_thresholds(self):
        self.assertEqual(self.audit['source_rows'], 1338)
        self.assertEqual(len(self.data), 1337)
        self.assertEqual(sum(self.audit['missing_counts_raw'].values()), 0)
        split = self.results['split']
        self.assertFalse(set(split['development_ids']) & set(split['test_ids']))
        self.assertEqual(set(self.bundle['reference'].source_row), set(split['development_ids']))
        np.testing.assert_array_equal(self.bundle['reference'].charges.quantile([1/3,2/3]), self.bundle['thresholds'])
        # np.nextafter chooses the nearest representable float above a cutoff.
        # This catches <= versus < mistakes that ordinary rounded values may miss.
        q1,q2 = self.bundle['thresholds']
        self.assertEqual(label_costs([0,q1,np.nextafter(q1,np.inf),q2,np.nextafter(q2,np.inf),1e9], [q1,q2]).tolist(), ['Low','Low','Medium','Medium','High','High'])
        with self.assertRaises(ValueError):
            label_costs([1], [2,2])

    # Exercise invalid numeric/category inputs, missing targets and an unfamiliar
    # but valid age. The correct responses differ: reject, drop/report, or flag.
    def test_input_validation_and_missing_targets(self):
        for field,value in [('children',1.2),('age',-1),('bmi',float('inf')),('region','moon')]:
            with self.assertRaises(ValueError):
                assess(self.bundle, dict(self.profile, **{field:value}))
        raw = self.data.head(5)[FEATURES+['charges']].copy()
        raw.loc[0,'charges'] = np.nan
        raw.loc[1,'charges'] = -1
        raw.loc[2,'age'] = np.nan
        # Use an isolated temporary CSV so deliberately malformed test data never
        # changes the real source file. The context manager removes it afterward.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'test.csv'; raw.to_csv(p,index=False)
            clean,audit = build_data(p)
        self.assertEqual(len(audit['invalid_target_rows_removed']), 2)
        self.assertEqual(clean.age.isna().sum(), 1)
        self.assertIn('age', assess(self.bundle, dict(self.profile, age=100))['outside_training_range'])

    # Rebuild pipelines on saved training folds and compare learned scaler means
    # to those folds, checking that validation rows were not used for preprocessing.
    def test_fold_preprocessing_and_schema(self):
        reference = self.bundle['reference'].set_index('source_row')
        for fold in self.results['split']['cv_folds']:
            self.assertFalse(set(fold['training_ids']) & set(fold['validation_ids']))
            self.assertTrue(set(fold['training_ids']).issubset(reference.index))
            train = reference.loc[fold['training_ids']]
            model = pipeline_for('k-NN').fit(train[FEATURES], train.tier)
            pre = model.named_steps['preprocess']
            np.testing.assert_allclose(pre.named_transformers_['numerical'].named_steps['scale'].mean_, train[NUMERIC].mean())
            self.assertEqual(pre.transform(train[FEATURES]).shape[1], 11)
            self.assertFalse(any('charges' in name or 'tier' in name for name in pre.get_feature_names_out()))
        # Inject one missing predictor into a copy to verify median imputation even
        # though the supplied dataset itself has no missing predictors.
        train = reference.iloc[:20].copy()
        train.loc[train.index[0], 'age'] = np.nan
        model = pipeline_for('k-NN').fit(train[FEATURES], train.tier)
        self.assertAlmostEqual(model.named_steps['preprocess'].named_transformers_['numerical'].named_steps['impute'].statistics_[0], train.age.median())

    # Reload the serialized estimator and reproduce holdout predictions. Then
    # recalculate each model's metrics from its saved prediction column to detect
    # inconsistent metrics or class ordering without refitting the benchmarks.
    def test_support_reload_and_predictions(self):
        for profile in [self.profile, self.bundle['reference'].iloc[0][FEATURES].to_dict()]:
            a = assess(self.bundle, profile)
            self.assertEqual(list(a['support']), CLASSES)
            self.assertTrue(np.isfinite(list(a['support'].values())).all())
            self.assertAlmostEqual(sum(a['support'].values()), 1.)
        reloaded = joblib.load(ROOT/'models/model.joblib')
        pred = pd.read_csv(ROOT/'outputs/evaluation_predictions.csv')
        test = self.data.set_index('source_row').loc[pred.source_row]
        np.testing.assert_array_equal(reloaded['pipeline'].predict(test[FEATURES]), pred['k-NN'])
        for name in ['k-NN','LDA','QDA']:
            self.assertEqual(evaluate(pred.tier, pred[name], pred.smoker), self.results['models'][name]['test'])

    # A small one-dimensional example makes the expected neighbours transparent.
    # Duplicate coordinates with different labels exercise equal zero-distance
    # votes; a second query exercises ordinary nonzero inverse-distance weights.
    def test_uniform_distance_zero_votes(self):
        X=np.array([[0.],[0.],[2.],[4.]])
        y=np.array(['High','Low','Medium','Low'])
        for weights in ['uniform','distance']:
            model=KNeighborsClassifier(n_neighbors=4, weights=weights, algorithm='kd_tree').fit(X,y)
            for query in [[0.],[1.]]:
                dist,idx=model.kneighbors([query])
                manual=neighbour_vote(dist[0],y[idx[0]],weights)
                proba=dict(zip(model.classes_,model.predict_proba([query])[0]))
                np.testing.assert_allclose([manual[c] for c in CLASSES],[proba[c] for c in CLASSES])

    # Create an intentionally incompatible bundle in a temporary folder and
    # require an actionable rebuild error, not silent use of old artifacts.
    def test_artifact_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'outputs').mkdir(); (root/'models').mkdir()
            bad=copy.deepcopy(self.results); bad['schema_version']=1
            (root/'outputs/results.json').write_text(json.dumps(bad))
            (root/'models/model.joblib').write_bytes(b'old')
            with self.assertRaisesRegex(ValueError,'python -m src.main'):
                load_artifacts(root)

    # Mock the Responses client so tests make no paid or network requests.
    # SimpleNamespace supplies only the response attributes used by production code.
    def test_ai_contract_and_failures(self):
        a=assess(self.bundle,self.profile)
        context=build_ai_context(self.profile,a,self.results)
        self.assertNotIn('preview',context)
        client=Mock()
        client.responses.create.return_value=SimpleNamespace(status='completed',output_text='Mock explanation')
        self.assertEqual(generate_ai_interpretation(client,'configured-model',context),'Mock explanation')
        self.assertEqual(client.responses.create.call_args.kwargs['max_output_tokens'],650)
        for response in [SimpleNamespace(status='incomplete',output_text='Partial'),SimpleNamespace(status='completed',output_text='')]:
            client.responses.create.return_value=response
            with self.assertRaises(AIInterpretationError):
                generate_ai_interpretation(client,'configured-model',context)
        client.responses.create.side_effect=TimeoutError
        with self.assertRaises(AIInterpretationError):
            generate_ai_interpretation(client,'configured-model',context)
        # Deep-copy nested dictionaries so changing the test profile does not mutate
        # the original context and accidentally invalidate the cache-key comparison.
        changed=copy.deepcopy(context); changed['submitted_profile']['age']=36
        self.assertNotEqual(explanation_key(context,'m'),explanation_key(changed,'m'))
        changed=copy.deepcopy(context); changed['run_id']='new'
        self.assertNotEqual(explanation_key(context,'m'),explanation_key(changed,'m'))
        with patch('src.ai_interpretation.load_dotenv'), patch.dict('os.environ', {'OPENAI_API_KEY':''}, clear=True):
            self.assertIsNone(get_ai_settings()[0])

    # Regression check for the earlier adjacent-string/dictionary-key bug.
    # This confirms expected keys, not correctness of the entire legacy analysis.
    def test_legacy_mardia_keys(self):
        from src.assumptions import mardia_test
        result=mardia_test(np.random.default_rng(42).normal(size=(30,3)))
        self.assertIn('skewness_p_value',result)
        self.assertIn('kurtosis_p_value',result)


# Allow direct execution while unittest discovery can also import this file.
if __name__ == '__main__':
    unittest.main()
