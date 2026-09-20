"""Streamlit inference reads frozen artifacts; no runtime training or diagnostics."""
import hashlib
import json
import pandas as pd
import streamlit as st
from src.data import ROOT, NUMERIC, CATEGORIES, CLASSES
from src.inference import load_artifacts, assess
from src.ai_interpretation import (AIInterpretationError, build_ai_context, create_client,
    generate_ai_interpretation, get_ai_settings, explanation_key)

PAGES = ['Cost Group Assessment', 'How the Model Works', 'Results & Model Comparison', 'Dataset & Methodology']


@st.cache_resource
def cached_artifacts(fingerprint):
    return load_artifacts()


def main():
    st.set_page_config(page_title='Medical-Cost Groups', layout='wide')
    st.markdown('''<style>
    [data-testid="stAppViewContainer"] {background:#f5f7fb;color:#102a43;}
    [data-testid="stHeader"] {background:#f5f7fb;}
    [data-testid="stSidebar"] {background:#102a43;}
    [data-testid="stSidebar"] * {color:#f7fafc;}
    h1,h2,h3 {color:#102a43;}
    [data-testid="stWidgetLabel"] p {color:#102a43 !important;}
    [data-testid="stMetric"], [data-testid="stExpander"] {background:white;border:1px solid #d9e2ec;border-radius:8px;padding:12px;}
    .stButton button {background:#1f5d9b;color:white;}
    </style>''', unsafe_allow_html=True)
    st.title('Multivariate Classification of Health-Insurance Medical-Cost Groups')
    st.caption('A university demonstration of observed medical-cost grouping. Historical similarity does not establish future expenditure, medical status or an actuarial price.')
    try:
        paths = [ROOT / 'models/model.joblib', ROOT / 'outputs/results.json', ROOT / 'outputs/figures/manifest.json']
        paths += sorted((ROOT / 'outputs/figures').glob('*.png'))
        fingerprint = hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()
        bundle, results = cached_artifacts(fingerprint)
    except Exception as error:
        st.error(str(error) + ' Regenerate with python -m src.main if files are missing.')
        st.stop()
    if st.session_state.get('artifact_fingerprint') != fingerprint:
        st.session_state.pop('assessment', None)
        st.session_state.pop('ai_cache', None)
        st.session_state['artifact_fingerprint'] = fingerprint
    page = st.sidebar.radio('Explore', PAGES)
    st.sidebar.caption(f"Run {results['run_id']} | development n={len(bundle['reference'])}")
    if page == PAGES[0]:
        show_assessment(bundle, results)
    elif page == PAGES[1]:
        st.header('How the Model Works')
        st.write('Six raw variables describe each profile jointly: age, BMI, children (a count), sex, smoker and region. Charges never enter neighbour distance. These are supervised cost labels, not unsupervised clusters.')
        st.write('Numerical values use training-fold mean and standard deviation. Full one-hot encoding retains all categories, giving 11 dimensions. A category mismatch contributes two indicator differences: 2 to Manhattan distance or 2 to squared Euclidean distance. This is a modelling choice about similarity, not a uniquely correct medical distance.')
        st.write('Full encoding keeps region distances symmetric. Its linear dependencies do not prevent k-NN because k-NN does not invert a covariance matrix. LDA and QDA use eight dimensions with reference-category encoding instead.')
        st.write('Development CV selects k, voting and distance by macro F1. The fixed development thresholds define the same task in every fold. The best tuning score is a selection estimate, not nested-CV performance.')
        with st.expander('Selected configuration and voting details'):
            st.json(results['models']['k-NN']['selected_parameters'])
            st.write('Uniform support is the fraction of all k neighbours. Distance voting normalizes inverse distances. If any distances are zero, only zero-distance neighbours vote equally. Class ties follow the estimator class order High, Low, Medium. Equal-distance neighbour membership depends on the fixed source-row training order and pinned library implementation. Preview rows are sorted by distance then source row.')
        st.image(str(ROOT / 'outputs/flowchart.png'), caption='Offline development and online inference are separate.')
    elif page == PAGES[2]:
        st.header('Results & Model Comparison')
        rows = []
        for name, report in results['models'].items():
            metric = report['test']
            rows.append({'Model': name, 'Accuracy': metric['accuracy'], 'Balanced accuracy': metric['balanced_accuracy'], 'Macro F1': metric['macro_f1'], 'High recall': metric['per_class']['High']['recall']})
        st.dataframe(pd.DataFrame(rows).set_index('Model').style.format('{:.1%}'), width='stretch')
        st.caption(f"Same held-out test records, n={results['models']['k-NN']['test']['n']}. k-NN was predeclared primary; small differences do not establish statistical superiority.")
        error = results['models']['k-NN']['test']
        subgroup = error['high_non_smoker']
        subgroup_recall = "undefined (no eligible records)" if subgroup["recall"] is None else format(subgroup["recall"], ".1%")
        st.write(f"High errors: {error['high_to_low']} classified Low and {error['high_to_medium']} classified Medium. High-cost non-smokers: n={subgroup['n']}, recall={subgroup_recall}.")
        st.image(str(ROOT / 'outputs/figures/knn_confusion.png'))
        st.image(str(ROOT / 'outputs/figures/recall_comparison.png'))
        for name, report in results['models'].items():
            with st.expander(name + ': complete errors, class metrics and development CV'):
                st.json(report)
        st.write('This dataset was examined in earlier coursework. The test records were held out during this refactor, not independently collected or never previously examined. Earlier whole-dataset-tier results are not comparable validation estimates.')
    else:
        st.header('Dataset & Methodology')
        st.write('Research question: How accurately can demographic and lifestyle characteristics classify individuals into Low, Medium and High observed medical-cost groups, and where does classification fail?')
        st.markdown('[Medical Cost Personal Datasets, Miri Choi](https://www.kaggle.com/datasets/mirichoi0218/insurance)')
        st.json(results['data'])
        st.write('One random 80/20 split (seed 42), without whole-data stratification. Development quantiles are frozen before five-fold stratified selection. Missing predictors, if present, use fold-fitted medians/modes. No records were trimmed for high charges.')
        st.write(f"Frozen exact thresholds: q1={results['thresholds'][0]!r}, q2={results['thresholds'][1]!r}. {results['boundary_rule']}")
        st.subheader('Development-data foundations')
        st.write('Raw numerical mean vector')
        st.dataframe(pd.Series(results['eda']['mean_vector'], name='Mean'))
        for key in ['covariance', 'correlation']:
            st.write('Raw numerical ' + key + ' matrix')
            st.dataframe(pd.DataFrame(results['eda'][key]))
        st.json({'category_frequencies': results['eda']['category_counts'], 'class_counts': results['split']['class_counts'], 'numerical_ranges': results['eda']['numerical_ranges']})
        st.write('Observed range coverage and CV stability do not prove independence or population representativeness. No person IDs are available; matching records are not proof of duplicate people. Unmeasured medical conditions, treatment and utilization may help explain failures.')
        st.download_button('Download methodology source', (ROOT / 'METHODOLOGY.md').read_text(), file_name='METHODOLOGY.md')


def show_assessment(bundle, results):
    st.header('Cost Group Assessment')
    st.caption('Enter all six fields, then assess. Numerical limits are data validation, not clinical rules. Submit edited fields to replace the previous assessment.')
    left, right = st.columns(2)
    with left:
        age = st.number_input('Age', value=35, step=1)
        bmi = st.number_input('BMI', value=25.0, step=.1)
        children = st.number_input('Children (count)', value=0, step=1)
    with right:
        sex = st.selectbox('Sex', CATEGORIES['sex'])
        smoker = st.selectbox('Smoker', CATEGORIES['smoker'])
        region = st.selectbox('Region', CATEGORIES['region'])
    profile = dict(age=age, bmi=bmi, children=children, sex=sex, smoker=smoker, region=region)
    if st.session_state.get('assessment', {}).get('profile') != profile:
        st.session_state.pop('assessment', None)
    if st.button('Assess cost group', type='primary'):
        st.session_state.pop('assessment', None)
        try:
            st.session_state['assessment'] = assess(bundle, profile)
        except ValueError as error:
            st.error(str(error))
    assessment = st.session_state.get('assessment')
    if not assessment:
        return
    st.subheader('Predicted observed-cost group: ' + assessment['tier'])
    st.caption('Python outputs')
    for column, label in zip(st.columns(3), CLASSES):
        column.metric(label + ' neighbour support', f"{assessment['support'][label]:.1%}")
    st.write(f"k={assessment['k']}; {assessment['weights']} voting. Support is not calibrated confidence. Even 100% means unanimous or fully weighted neighbour support, not proof of correctness.")
    if assessment['outside_training_range']:
        st.warning('Outside development-data range: ' + ', '.join(assessment['outside_training_range']) + '. Similarity is less established for this unfamiliar profile.')
    with st.expander('Historical cost context and fixed group definition', expanded=True):
        st.write(results['boundary_rule'])
        st.write(f"q1={results['thresholds'][0]:.6f}; q2={results['thresholds'][1]:.6f}")
        st.dataframe(pd.DataFrame([results['historical_development_costs'][assessment['tier']]]), hide_index=True)
        st.caption('Observed development-group charges: sample min/max, median and quartiles. These are not personalized prediction intervals or bounds on future costs.')
    with st.expander('Technical details and nearest training profiles'):
        st.write(f"{assessment['metric']} distance in 11 encoded dimensions. Preview of five of {assessment['k']} neighbours; all {assessment['k']} enter the voting rule. With zero-distance matches, distance weighting gives other neighbours zero weight.")
        st.dataframe(assessment['preview'], hide_index=True, width='stretch')
        st.caption('Anonymous source-row labels. Only development records are eligible. Equal-distance preview rows are ordered by source row.')
    st.subheader('AI-assisted interpretation')
    context = build_ai_context(profile, assessment, results)
    api_key, model = get_ai_settings()
    try:
        api_key = st.secrets.get('OPENAI_API_KEY', api_key)
        model = st.secrets.get('OPENAI_MODEL', model)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    key = explanation_key(context, model)
    cache = st.session_state.setdefault('ai_cache', {})
    if st.button('Explain my result with AI'):
        if not api_key:
            st.info('AI is not configured. Python results remain available. This availability message is not generated by OpenAI.')
        elif key not in cache:
            try:
                with st.spinner('Preparing interpretation...'):
                    cache[key] = generate_ai_interpretation(create_client(api_key), model, context)
            except AIInterpretationError as error:
                st.info(str(error) + ' Python results remain available. This availability message is not generated by OpenAI.')
    if key in cache:
        st.caption('AI-assisted prose may contain errors; the Python cards above remain authoritative.')
        st.markdown(cache[key].replace('$', r'\$'))


if __name__ == '__main__':
    main()
