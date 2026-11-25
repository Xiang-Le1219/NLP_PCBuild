"""Streamlit deployment entry point."""
import logging
import pandas as pd
import streamlit as st
import advisor as A
import pc_engine as E

st.set_page_config(page_title='PC Build Advisor | OPT', page_icon='🖥️', layout='wide')
st.title('PC Build Advisor using OPT')
st.caption('English • Bahasa Melayu • Manglish | Budget-aware PC recommendations')

@st.cache_resource
def load_engine():
    return A.load_engine()

# Cache just one model at a time to limit memory on CPU hosting.
@st.cache_resource(max_entries=1, show_spinner=False)
def load_model(mode):
    return A.ModelSession(mode)

with st.sidebar:
    st.header('NLP settings')
    mode = st.selectbox(
        'Requirement extraction', A.available_modes(),
        help="Choose how the user's natural-language PC requirements are interpreted."
    )
    with st.expander('What do these extraction modes mean?'):
        st.markdown("""
        - **Rule-based baseline:** uses keywords and regular expressions; fast and does not download a model.
        - **Pretrained OPT-125M (few-shot):** uses the original OPT-125M checkpoint with example prompts; it has not been fine-tuned for this project.
        - **Fine-tuned OPT-125M (LoRA):** uses the supplied project adapter trained for PC requirement extraction.

        **Important:** this setting changes requirement extraction only. The separate PC engine selects components, enforces its budget rules, and builds the recommendation. The current build explanation is a deterministic template, not OPT-generated text.
        """)
    if not A.adapter_ready():
        st.warning('No trained OPT adapter is included. The baseline is ready; pretrained OPT is available as an experimental option.')
    st.caption('OPT mode downloads the 125M base model on first use. CPU inference can be slow and may exceed free hosting resources.')
    st.divider()
    st.markdown('**Pricing assumption**')
    st.write(f'USD 1 = RM {E.USD_TO_MYR:.2f} (fixed dataset conversion, not live pricing).')
    st.caption('Check actual store prices and hardware clearances before purchasing.')

st.markdown('### 1. Describe your PC')
with st.form('query_form'):
    query = st.text_area('Your requirements', placeholder='Nak PC untuk gaming, bajet RM4500, Ryzen CPU, 16GB RAM dan monitor.', max_chars=1000)
    extract = st.form_submit_button('Analyse requirements', type='primary')

if extract:
    if not query.strip():
        st.warning('Enter a description first.')
    else:
        if mode == A.BASELINE:
            req, source, raw = A.baseline_extract(query)
        else:
            try:
                with st.spinner('Loading OPT and analysing your requirements…'):
                    req, source, raw = load_model(mode).extract(query)
            except Exception:
                logging.exception('OPT load or inference failed')
                req, source, raw = A.baseline_extract(query)
                source = 'Rule-based fallback: OPT could not load or run.'
                st.warning('OPT was unavailable. Continuing with the rule-based baseline. See deployment logs for details.')
        st.session_state['parsed'] = req
        st.session_state['source'] = source
        st.session_state['raw'] = raw
        st.session_state['parsed_query'] = query
        st.session_state['request_id'] = st.session_state.get('request_id', 0) + 1
        st.session_state.pop('build_result', None)

if 'parsed' in st.session_state:
    req = st.session_state['parsed']
    rid = st.session_state['request_id']
    st.markdown('### 2. Review extracted requirements')
    st.caption('Extraction method: ' + st.session_state['source'])
    st.write('Analysed request: ' + st.session_state['parsed_query'])
    st.caption('Edit any incorrect fields before generating a build. The recommendation uses the confirmed values.')
    if st.session_state['raw'] is not None:
        with st.expander('Inspect raw OPT output'):
            st.code(st.session_state['raw'], language='text')
    with st.form(f'review_{rid}'):
        a,b,c = st.columns(3)
        use_cases = E.USE_CASES
        use = a.selectbox('Use case', use_cases, index=use_cases.index(req['use_case']), format_func=lambda v: v.replace('_', ' ').title())
        budget = b.number_input('Maximum budget (required)', min_value=0, value=min(req['budget'] or 0, 1000000), max_value=1000000, step=100, help='If your message did not include a budget, enter it here. The model is not allowed to invent one.')
        currency = c.selectbox('Currency', ['MYR','USD'], index=['MYR','USD'].index(req['currency']))
        strategy_label = st.selectbox(
            'Build selection strategy',
            ['Closest under budget (minimize leftover)',
             'Best value (avoid unnecessary spending)'],
            index=0,
            help=('Closest under budget prioritizes the highest compatible total below '
                  'your budget. It may include optional performance upgrades such as a '
                  'discrete GPU. Best value prioritizes suitability for the selected use case.')
        )
        budget_strategy = ('closest_to_budget' if strategy_label.startswith('Closest')
                           else 'best_value')
        a,b = st.columns(2)
        cpu = ['any','amd','intel']; gpu = ['any','nvidia','amd','intel']
        cpu_brand = a.selectbox('CPU preference', cpu, index=cpu.index(req['cpu_brand']))
        gpu_brand = b.selectbox('GPU preference', gpu, index=gpu.index(req['gpu_brand']))
        a,b = st.columns(2)
        ram = a.number_input('Minimum RAM (GB; 0 = automatic)', min_value=0, max_value=1024, value=min(req['ram_gb'] or 0,1024), step=1)
        storage = b.number_input('Minimum SSD storage (GB; 0 = automatic)', min_value=0, max_value=100000, value=min(req['storage_gb'] or 0,100000), step=100)
        extras = st.multiselect('Peripherals', ['monitor','keyboard','mouse'], default=req['extras'])
        compact = st.checkbox('Compact PC', value=req['compact'])
        build = st.form_submit_button('Generate compatible PC build', type='primary')
    if build:
        confirmed = dict(use_case=use, budget=budget, currency=currency,
                         cpu_brand=cpu_brand, gpu_brand=gpu_brand,
                         ram_gb=ram or None, storage_gb=storage or None,
                         extras=extras, compact=compact,
                         budget_strategy=budget_strategy)
        st.session_state.pop('build_result', None)
        try:
            with st.spinner('Searching the hardware catalogue…'):
                res, violations = A.recommend(load_engine(), confirmed)
            st.session_state['build_result'] = (res, violations)
        except Exception:
            logging.exception('Recommendation failed')
            st.error('Unable to generate a build. Check the hardware files and deployment logs.')

if 'build_result' in st.session_state:
    res, violations = st.session_state['build_result']
    st.markdown('### 3. Your recommended build')
    if not res['ok']:
        st.warning(res['message'])
    else:
        a,b,c = st.columns(3)
        a.metric('Estimated total', f"RM {res['total_myr']:,.2f}")
        b.metric('Budget', f"RM {res['budget_myr']:,.2f}")
        c.metric('Remaining', f"RM {res['budget_myr']-res['total_myr']:,.2f}")
        st.caption(
            f"Build strategy: {'Closest under budget' if res.get('budget_strategy') == 'closest_to_budget' else 'Best value'} "
            f"• Budget used: {res.get('budget_utilization_pct', 0):.1f}%"
        )
        df = pd.DataFrame([{'Component': E.CAT_LABEL[it['category']], 'Product': it['name'], 'Specifications': it['spec'], 'Price (RM)': round(it['price_myr'],2)} for it in res['items']])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button('Download build CSV', df.to_csv(index=False).encode('utf-8'), 'recommended_pc.csv', 'text/csv')
        if violations:
            st.error('Compatibility checks flagged: ' + '; '.join(violations))
        else:
            st.success('Passed the implemented budget and compatibility checks.')
        st.subheader('Recommendation explanation')
        st.write(E.template_explanation(res))
        st.caption('Deterministic template explanation from catalogue facts — not OPT-generated. OPT is used for extraction when selected and successful.')
        for upgrade in res.get('upgrades', []):
            st.write('• ' + upgrade)
        for warning in res.get('warnings', []):
            st.warning(warning)

with st.expander('About this prototype and evaluation'):
    st.write('The NLP module extracts structured requirements; a separate knowledge-based engine selects compatible hardware.')
    run_info = A.ROOT / 'results' / 'run_mode.txt'
    if run_info.is_file():
        st.code(run_info.read_text(encoding='utf-8'), language='text')
    st.write('Synthetic training data and a small out-of-distribution test set are not proof of real-world accuracy. Lexical checks and hardware rules are limited.')
    for filename in ('extraction_results.csv','end_to_end.csv'):
        path = A.ROOT / 'results' / filename
        if path.is_file():
            st.caption(filename)
            st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)
