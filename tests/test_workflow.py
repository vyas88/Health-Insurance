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


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.results = load_artifacts()
        cls.data, cls.audit = build_data()
        cls.profile = dict(age=35, bmi=25., children=0, sex='female', smoker='no', region='northeast')

    def test_cleaning_split_thresholds(self):
        self.assertEqual(self.audit['source_rows'], 1338)
        self.assertEqual(len(self.data), 1337)
        self.assertEqual(sum(self.audit['missing_counts_raw'].values()), 0)
        split = self.results['split']
        self.assertFalse(set(split['development_ids']) & set(split['test_ids']))
        self.assertEqual(set(self.bundle['reference'].source_row), set(split['development_ids']))
        np.testing.assert_array_equal(self.bundle['reference'].charges.quantile([1/3,2/3]), self.bundle['thresholds'])
        q1,q2 = self.bundle['thresholds']
        self.assertEqual(label_costs([0,q1,np.nextafter(q1,np.inf),q2,np.nextafter(q2,np.inf),1e9], [q1,q2]).tolist(), ['Low','Low','Medium','Medium','High','High'])
        with self.assertRaises(ValueError):
            label_costs([1], [2,2])

    def test_input_validation_and_missing_targets(self):
        for field,value in [('children',1.2),('age',-1),('bmi',float('inf')),('region','moon')]:
            with self.assertRaises(ValueError):
                assess(self.bundle, dict(self.profile, **{field:value}))
        raw = self.data.head(5)[FEATURES+['charges']].copy()
        raw.loc[0,'charges'] = np.nan
        raw.loc[1,'charges'] = -1
        raw.loc[2,'age'] = np.nan
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'test.csv'; raw.to_csv(p,index=False)
            clean,audit = build_data(p)
        self.assertEqual(len(audit['invalid_target_rows_removed']), 2)
        self.assertEqual(clean.age.isna().sum(), 1)
        self.assertIn('age', assess(self.bundle, dict(self.profile, age=100))['outside_training_range'])

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
        train = reference.iloc[:20].copy()
        train.loc[train.index[0], 'age'] = np.nan
        model = pipeline_for('k-NN').fit(train[FEATURES], train.tier)
        self.assertAlmostEqual(model.named_steps['preprocess'].named_transformers_['numerical'].named_steps['impute'].statistics_[0], train.age.median())

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

    def test_artifact_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'outputs').mkdir(); (root/'models').mkdir()
            bad=copy.deepcopy(self.results); bad['schema_version']=1
            (root/'outputs/results.json').write_text(json.dumps(bad))
            (root/'models/model.joblib').write_bytes(b'old')
            with self.assertRaisesRegex(ValueError,'python -m src.main'):
                load_artifacts(root)

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
        changed=copy.deepcopy(context); changed['submitted_profile']['age']=36
        self.assertNotEqual(explanation_key(context,'m'),explanation_key(changed,'m'))
        changed=copy.deepcopy(context); changed['run_id']='new'
        self.assertNotEqual(explanation_key(context,'m'),explanation_key(changed,'m'))
        with patch('src.ai_interpretation.load_dotenv'), patch.dict('os.environ', {'OPENAI_API_KEY':''}, clear=True):
            self.assertIsNone(get_ai_settings()[0])

    def test_legacy_mardia_keys(self):
        from src.assumptions import mardia_test
        result=mardia_test(np.random.default_rng(42).normal(size=(30,3)))
        self.assertIn('skewness_p_value',result)
        self.assertIn('kurtosis_p_value',result)


if __name__ == '__main__':
    unittest.main()
