"""PC Build Advisor Streamlit app.

The NLP layer extracts all catalogue-backed attributes that are explicitly
mentioned. Attributes not mentioned by the user remain Auto/default and are
selected by the recommendation engine.
"""
from __future__ import annotations

import logging
import time
import json
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

import advisor as A
import pc_engine as E
import catalog_nlu as C

st.set_page_config(page_title="PC Build Advisor | OPT", page_icon="🖥️", layout="wide")
st.title("PC Build Advisor using OPT")
st.caption(
    "Describe your PC in natural language. The system extracts available "
    "component specifications, applies defaults for unmentioned fields, "
    "and recommends a compatible build."
)


@st.cache_resource
def load_engine():
    return A.load_engine()


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model(mode):
    return A.ModelSession(mode)


engine = load_engine()

with st.sidebar:
    st.header("NLP settings")
    mode = st.selectbox(
        "Requirement extraction model",
        A.available_modes(),
        help=(
            "Rule-based baseline uses deterministic rules. Pretrained OPT-125M "
            "uses the original checkpoint. Fine-tuned OPT-125M loads your LoRA adapter."
        ),
    )
    with st.expander("What do the three model choices mean?"):
        st.markdown(
            "- **Rule-based baseline:** Regex, aliases and catalogue values; fast and explainable.\n"
            "- **Pretrained OPT-125M:** Original model without your PC-specific fine-tuning.\n"
            "- **Fine-tuned OPT-125M:** Your PC requirement extraction LoRA adapter.\n\n"
            "These choices affect requirement extraction only. Component filtering, "
            "compatibility checks and build ranking are performed by the same PC engine."
        )
    if A.adapter_ready("extract"):
        st.success("Fine-tuned OPT-125M adapter found (weights included).")
    else:
        st.warning("No fine-tuned OPT-125M adapter found. Pretrained and rule-based modes remain available.")
    st.caption("Both OPT modes load the official facebook/opt-125m base model on first use. "
               "A network connection and enough Streamlit memory are required; CPU inference can be slow.")
    st.divider()
    strategy_label = st.selectbox(
        "Build selection strategy",
        ["Closest under budget", "Best value"],
        help=(
            "Closest under budget chooses the highest-priced valid build below the "
            "budget. Best value prioritizes the workload profile and performance."
        ),
    )
    budget_strategy = (
        "closest_to_budget" if strategy_label == "Closest under budget" else "best_value"
    )
    st.divider()
    st.markdown("**Dataset pricing assumption**")
    st.write(f"USD 1 = RM {E.USD_TO_MYR:.2f} (fixed conversion, not live pricing).")
    st.caption("Confirm actual store prices, BIOS support, cooler mounting and case clearances before purchasing.")


st.markdown("## 1. Describe your PC")
with st.form("query_form"):
    query = st.text_area(
        "Your requirements",
        placeholder=(
            "Example: Need a gaming PC around RM7000, Ryzen 7 CPU, RTX 5070, "
            "32GB DDR5 6000 RAM, 2TB NVMe SSD, white ATX case, 750W Gold PSU, "
            "27-inch IPS 144Hz monitor and wireless mouse."
        ),
        max_chars=1500,
        height=115,
    )
    analyse = st.form_submit_button("Analyse requirements", type="primary")

if analyse:
    if not query.strip():
        st.warning("Please describe the PC you want.")
    else:
        started = time.perf_counter()
        try:
            if mode == A.BASELINE:
                req, source, raw, field_report = A.analyse_requirements(
                    query, engine.d, mode=mode
                )
            else:
                with st.spinner("Loading the selected OPT model and analysing your requirements…"):
                    session = load_model(mode)
                    req, source, raw, field_report = A.analyse_requirements(
                        query, engine.d, mode=mode, model_session=session
                    )
        except Exception as exc:
            logging.exception("Requirement analysis failed")
            req, source, raw, field_report = A.analyse_requirements(
                query, engine.d, mode=A.BASELINE
            )
            source = f"Rule-based fallback (selected model failed: {type(exc).__name__})"
            st.warning(
                "The selected OPT model could not run. The system used the rule-based "
                "baseline instead. See deployment logs for details."
            )
        elapsed = time.perf_counter() - started
        st.session_state["parsed"] = req
        st.session_state["source"] = source
        st.session_state["raw"] = raw
        st.session_state["field_report"] = field_report
        st.session_state["analysis_seconds"] = elapsed
        st.session_state["parsed_query"] = query
        # Session-only history. Streamlit Cloud's filesystem is ephemeral; the
        # user must download the CSV/JSON to keep experiment predictions.
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "selected_mode": mode,
            "actual_source": source,
            "query": query,
            "elapsed_s": round(elapsed, 3),
            "model_only_prediction": req.get("model_prediction"),
            "confirmed_extraction_before_review": {k: req.get(k) for k in (
                "use_case", "budget", "currency", "cpu_brand", "gpu_brand",
                "ram_gb", "storage_gb", "extras", "compact")},
            "raw_model_output": raw,
        }
        st.session_state.setdefault("analysis_history", []).append(entry)
        st.session_state["request_id"] = st.session_state.get("request_id", 0) + 1
        st.session_state.pop("build_result", None)

if "parsed" in st.session_state:
    req = st.session_state["parsed"]
    rid = st.session_state["request_id"]
    st.markdown("## 2. Analyse requirements")
    m1, m2, m3 = st.columns(3)
    m1.metric("Analysis time", f"{st.session_state['analysis_seconds']:.2f} sec")
    m2.metric("Detected dataset attributes", int(req.get("attribute_count", 0)))
    m3.metric("Requirement extraction", st.session_state["source"].split(" + ")[0])
    st.caption("Original request: " + st.session_state["parsed_query"])

    changes = req.get("typo_corrections") or []
    if changes:
        st.info("Possible typo correction: " + req.get("corrected_text", st.session_state["parsed_query"]))
    if req.get("budget_is_default"):
        st.info(f"No budget was found in your message. Default budget RM {req['budget']:,.0f} is being used.")
    if st.session_state["raw"]:
        with st.expander("Inspect raw OPT output and model-only prediction"):
            st.code(st.session_state["raw"], language="text")
            st.markdown("**Model-only JSON (before rule-based safety checks)**")
            st.json(req.get("model_prediction"))
            st.caption("The review fields use explicit values verified from your input. "
                       "They can differ from raw OPT predictions; this is a hybrid system.")
    if st.session_state.get("analysis_history"):
        recent = st.session_state["analysis_history"][-1]
        st.download_button(
            "Download this analysis (JSON)",
            json.dumps(recent, ensure_ascii=False, indent=2, default=str),
            file_name="pc_advisor_last_nlp_prediction.json",
            mime="application/json",
            key=f"download_analysis_{rid}",
        )

    report = st.session_state["field_report"]
    stated = report[report["Source"] == "User-stated"]
    defaults = report[report["Source"] == "Default"]
    unavailable = report[report["Source"] == "Unavailable"]
    with st.expander(
        f"All dataset attributes ({len(stated)} user-stated, {len(defaults)} default/auto, {len(unavailable)} unavailable)",
        expanded=True,
    ):
        st.write(
            "Every field available in the cleaned component catalogue is listed below. "
            "User-stated values become filters. Fields not mentioned remain Auto/default "
            "and are selected using the workload profile and PC engine."
        )
        if not stated.empty:
            st.markdown("**Detected from your sentence**")
            st.dataframe(stated, hide_index=True, use_container_width=True)
        if not defaults.empty:
            st.markdown("**Not mentioned — system default / automatic selection**")
            st.dataframe(defaults, hide_index=True, use_container_width=True)
        if not unavailable.empty:
            st.markdown("**Unavailable after dataset cleaning**")
            st.dataframe(unavailable, hide_index=True, use_container_width=True)

    st.markdown("### Review core requirements")
    st.caption("You can correct the main requirements before generating the build. Dataset-specific filters detected above are also applied.")
    if st.session_state["raw"]:
        pass
    with st.form(f"review_{rid}"):
        col1, col2, col3 = st.columns(3)
        use_cases = E.USE_CASES
        use = col1.selectbox(
            "PC usage",
            use_cases,
            index=use_cases.index(req.get("use_case", "general")),
            format_func=lambda v: v.replace("_", " ").title(),
        )
        budget = col2.number_input(
            f"Maximum budget ({req.get('currency', 'MYR')})",
            min_value=100,
            max_value=100000,
            value=int(req.get("budget") or A.DEFAULT_BUDGET_MYR),
            step=100,
            help="If your sentence does not contain a budget, the default is RM4,000. You can change it here.",
        )
        currency = col3.selectbox(
            "Budget currency",
            ["MYR", "USD"],
            index=["MYR", "USD"].index(req.get("currency", "MYR")),
        )
        col1, col2, col3 = st.columns(3)
        cpu_brand = col1.selectbox(
            "CPU brand",
            ["any", "amd", "intel"],
            index=["any", "amd", "intel"].index(req.get("cpu_brand", "any")),
        )
        gpu_brand = col2.selectbox(
            "GPU brand",
            ["any", "nvidia", "amd", "intel"],
            index=["any", "nvidia", "amd", "intel"].index(req.get("gpu_brand", "any")),
        )
        ram = col3.number_input(
            "Minimum RAM (GB; 0 = use profile default)",
            min_value=0, max_value=1024, value=int(req.get("ram_gb") or 0), step=8,
        )
        col1, col2, col3 = st.columns(3)
        storage = col1.number_input(
            "Minimum SSD (GB; 0 = use profile default)",
            min_value=0, max_value=100000, value=int(req.get("storage_gb") or 0), step=250,
        )
        compact = col2.checkbox("Compact PC", value=bool(req.get("compact", False)))
        extras = col3.multiselect(
            "Include peripherals",
            ["monitor", "keyboard", "mouse"],
            default=req.get("extras", []),
        )
        build = st.form_submit_button("Generate compatible PC build", type="primary")

    if build:
        confirmed = dict(req)
        confirmed.update({
            "use_case": use,
            "budget": budget,
            "currency": currency,
            "cpu_brand": cpu_brand,
            "gpu_brand": gpu_brand,
            "ram_gb": ram or None,
            "storage_gb": storage or None,
            "extras": extras,
            "compact": compact,
            "budget_strategy": budget_strategy,
        })
        st.session_state.pop("build_result", None)
        try:
            with st.spinner("Searching the hardware catalogue and checking compatibility…"):
                result, violations = A.recommend(engine, confirmed)
            st.session_state["build_result"] = (result, violations)
        except Exception:
            logging.exception("Recommendation failed")
            st.error("Unable to generate a build. Check the hardware files and deployment logs.")

if "build_result" in st.session_state:
    result, violations = st.session_state["build_result"]
    st.markdown("## 3. Recommended PC build")
    if not result["ok"]:
        st.warning(result["message"])
    else:
        a, b, c = st.columns(3)
        a.metric("Estimated total", f"RM {result['total_myr']:,.2f}")
        b.metric("Budget", f"RM {result['budget_myr']:,.2f}")
        c.metric("Remaining", f"RM {result['budget_myr'] - result['total_myr']:,.2f}")
        display_rows = []
        for item in result["items"]:
            display_rows.append({
                "Component": E.CAT_LABEL[item["category"]],
                "Product": item["name"],
                "Key specifications": item["spec"],
                "Price (RM)": round(item["price_myr"], 2),
            })
        build_df = pd.DataFrame(display_rows)
        st.dataframe(build_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download recommended build CSV",
            build_df.to_csv(index=False).encode("utf-8"),
            "recommended_pc_build.csv",
            "text/csv",
        )
        if violations:
            st.error("Compatibility checks flagged: " + "; ".join(violations))
        else:
            st.success("Passed the budget and compatibility rules implemented in this prototype.")
        st.subheader("Full catalogue attributes of selected products")
        for item in result["items"]:
            with st.expander(f"{E.CAT_LABEL[item['category']]} — {item['name']}"):
                st.write(item["spec"])
                attrs = item.get("attributes", {})
                if attrs:
                    attr_df = pd.DataFrame(
                        [{"Attribute": k, "Dataset value": str(v)} for k, v in attrs.items()]
                    )
                    st.dataframe(attr_df, hide_index=True, use_container_width=True)
        st.subheader("Recommendation explanation")
        st.write(E.template_explanation(result))
        st.caption(
            "This explanation is generated from verified catalogue facts using a deterministic template. "
            "OPT is used for requirement extraction, not as the hardware compatibility engine."
        )
        for warning in result.get("warnings", []):
            st.warning(warning)

with st.expander("Saved OPT-125M experimental results", expanded=False):
    st.write(
        "These CSVs contain **recorded experimental outputs**, not live "
        "predictions for the message currently typed above. The pretrained "
        "base model is still downloaded separately for live inference."
    )
    saved_files = {
        "Same-run evaluation (pretrained zero/few-shot vs fine-tuned LoRA)":
            "results/opt125m_evaluation.csv",
        "Saved model predictions from training run":
            "results/opt125m_predictions.csv",
        "Recorded evaluation of current project (baseline + pretrained zero-shot)":
            "results/model_comparison.csv",
        "Recorded raw pretrained responses from current project":
            "results/opt_raw_predictions.csv",
        "Training loss and epochs": "results/opt125m_training.csv",
    }
    for label, relpath in saved_files.items():
        path = A.ROOT / relpath
        if not path.is_file():
            continue
        with st.expander(label):
            try:
                df = pd.read_csv(path)
                st.dataframe(df.head(150), hide_index=True, use_container_width=True)
                st.caption(f"Showing up to 150 of {len(df)} recorded rows.")
            except Exception as e:
                st.info(f"Preview unavailable: {e}")
            st.download_button("Download " + path.name, path.read_bytes(),
                               file_name=path.name, mime="text/csv", key="saved_" + path.name)
    st.info("The saved experimental runs used separate evaluation setups. "
            "For a rigorous comparison, rerun both models on the same fixed test set "
            "using the same prompts and measurement procedure.")

if st.session_state.get("analysis_history"):
    with st.expander("Download predictions from this Streamlit session"):
        log_df = pd.DataFrame([
            {**{k: v for k, v in record.items()
                if k not in ("model_only_prediction", "confirmed_extraction_before_review")},
             "model_only_prediction": json.dumps(record.get("model_only_prediction"), ensure_ascii=False),
             "verified_fields": json.dumps(record.get("confirmed_extraction_before_review"), ensure_ascii=False)}
            for record in st.session_state["analysis_history"]
        ])
        st.dataframe(log_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download session predictions CSV",
            log_df.to_csv(index=False).encode("utf-8"),
            file_name="streamlit_opt_predictions.csv", mime="text/csv",
        )
        st.caption("Session history is not written to GitHub or kept permanently on Streamlit Cloud.")

with st.expander("About this prototype"):
    st.markdown(
        "The system extracts explicit requirements from all component fields available in the cleaned CSV catalogue. "
        "Fields not mentioned by the user remain automatic and use the engine's workload-based defaults. "
        "The recommendation engine applies catalogue filters, budget logic and simplified compatibility rules. "
        "Check product specifications and manufacturer compatibility before purchasing."
    )
    run_info = A.ROOT / "results" / "run_mode.txt"
    if run_info.is_file():
        st.code(run_info.read_text(encoding="utf-8"), language="text")
