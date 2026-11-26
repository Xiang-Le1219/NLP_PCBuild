# PC Build Advisor using OPT

A Natural Language Processing coursework prototype that converts English, Malay and Manglish PC-building requests into structured requirements, filters the supplied component catalogue, applies compatibility rules, and recommends a budget-aware PC build.

## Main features

- **Catalogue-wide requirement analysis:** the app audits 98 raw and derived attributes from the 11 component catalogues included in `csv/` (CPU, GPU, motherboard, RAM, SSD, PSU, case, cooler, monitor, keyboard and mouse).
- **User-stated requirements first:** explicitly mentioned requirements become filters. Unmentioned attributes remain `Auto / system default` and are selected by the workload profile and build engine.
- **Typo-tolerant input:** common PC vocabulary mistakes (for example `wireles`, `motheboard`, `nvdia`, `keybord`, and `moniter`) are corrected before requirement extraction. Corrections are shown in the interface.
- **Analysis duration:** the app displays how long requirement analysis took, including model loading on the first request.
- **Three requirement-extraction approaches:** Rule-based baseline, original pretrained OPT-125M, and fine-tuned OPT-125M (LoRA).
- **Two build strategies:** Closest under budget and Best value.
- **Build validation:** checks the compatibility rules implemented in this prototype and reports when a stated preference cannot be satisfied by the catalogue.
- **Experiment notebook:** compares the three extraction approaches using the same synthetic and out-of-distribution test sets.

The model selector changes **requirement extraction only**. The component selection, budget search and compatibility checks are handled by the same deterministic PC build engine in all three modes.

## Repository structure

```text
PCBuild-OPT/
├── app.py
├── advisor.py
├── pc_engine.py
├── catalog_nlu.py
├── nlu_data.py
├── opt_nlu.py
├── requirements.txt
├── requirements-notebook.txt
├── smoke_test.py
├── test_budget_strategy.py
├── csv/                         # 11 source component CSV files
├── adapters/
│   ├── extract_125m/             # Fine-tuned requirement-extraction LoRA adapter
│   └── explain_125m/             # Explanation adapter artefact (not used by app inference)
├── notebooks/
│   └── PCBuild_OPT_Simplified.ipynb
├── results/
│   ├── attribute_audit.csv
│   ├── cleaning_report.csv
│   ├── model_comparison.csv
│   ├── budget_strategy_test.csv
│   ├── notebook_run_summary.json
│   └── run_mode.txt
└── .streamlit/
    └── config.toml
```

## Run locally

Use Python 3.11 (recommended).

```bash
python -m venv .venv
# Activate the virtual environment for your OS.
python -m pip install -r requirements.txt
python smoke_test.py
python test_budget_strategy.py
streamlit run app.py
```

Start in **Rule-based baseline** mode first. This mode does not download OPT weights and is the quickest way to verify data loading, requirement analysis and PC build generation.

## Use the notebook and compare all three models

Open `notebooks/PCBuild_OPT_Simplified.ipynb` from the project folder in Jupyter or Google Colab.

In the setup cell:

- `RUN_MODEL_EVALUATION = True` runs inference for original pretrained OPT-125M and fine-tuned OPT-125M, alongside the Rule-based baseline.
- `RETRAIN_EXTRACT_ADAPTER = False` uses the supplied `adapters/extract_125m` adapter. Change it to `True` only if you intentionally want to train the adapter again on a CUDA GPU.
- The pretrained model weights are downloaded from Hugging Face the first time they are needed. Internet access and sufficient memory are required.

The notebook compares the same Synthetic and OOD tests using:
- JSON-validity percentage
- Exact-match accuracy
- Mean field accuracy
- Per-field accuracy (use case, budget, CPU/GPU brand, RAM and storage)
- Seconds per example

The notebook always writes rows for all three methods to `results/model_comparison.csv`. If a model cannot load or run, its rows are labelled **Not evaluated** rather than given invented scores. The Streamlit app displays this comparison file under **Model performance comparison**.

Run all notebook cells in order. The notebook is simplified to a single OPT-125M extraction comparison workflow; it does not automatically train multiple model sizes.

## How defaults work

Every field in the cleaned component schema is listed in the Streamlit **All dataset attributes** report.

- **User-stated:** the system detected an explicit value and applies it as a catalogue filter.
- **Default:** the user did not mention it, so the build engine selects an appropriate value automatically according to the usage profile, compatibility rules and budget strategy.
- **Unavailable:** the field is not populated in the cleaned catalogue and cannot be used as a reliable filter.

A default is not necessarily one fixed product value. For example, an unmentioned cooler is selected according to CPU requirements and the available catalogue.

## Model choices in the Streamlit sidebar

1. **Rule-based baseline:** regex, aliases, catalogue values and typo correction. Fast and explainable.
2. **Pretrained OPT-125M:** the original OPT checkpoint, without PC-specific fine-tuning.
3. **Fine-tuned OPT-125M (LoRA):** the base OPT-125M model plus the supplied requirement-extraction adapter.

OPT can generate incorrect structured values. The application therefore gives explicit user input and catalogue-aware extraction priority over model guesses. Always review the extracted requirements before generating a build.

## Deployment on Streamlit Community Cloud

1. Upload the **contents** of this project folder to the root of a GitHub repository. Keep `app.py` and `requirements.txt` at the repository root.
2. Ensure `.streamlit/config.toml`, `csv/`, `adapters/` and all root-level Python modules are also committed. Do not upload secrets.
3. Sign in at https://share.streamlit.io/ using GitHub and create an app from the repository.
4. Select branch `main`, main file path `app.py`, and Python 3.11 in Advanced settings.
5. Deploy. Test Rule-based baseline first. Pretrained OPT downloads model weights on first use; CPU inference may be slow and can exceed free hosting resources.

The supplied LoRA adapter files are included in `adapters/`. Do not upload Hugging Face cache folders or intermediate training checkpoints.

## Known limitations

- Prices are catalogue prices supplied in the CSV files. The prototype uses the fixed USD-to-MYR assumption in `pc_engine.py`; these are not live Malaysian retail prices.
- Compatibility validation covers only the rules implemented in this prototype. It is not a substitute for checking BIOS support, cooler mounting, case clearances, PSU connectors and manufacturer specifications before purchasing.
- A recommendation may fail when an explicitly requested attribute has no matching product in the available catalogue. The system should report this rather than silently ignoring the requirement.
- OPT inference and LoRA training require additional compute and internet access. Live Streamlit hosting and full model inference should be tested in the target deployment environment.
- Verify the original CSV data licence/redistribution terms before making the GitHub repository public. Keep it private if you do not have permission to redistribute the data.

## Tests

```bash
python smoke_test.py
python test_budget_strategy.py
```

These offline tests validate data loading, baseline requirement extraction, typo correction, selected catalogue attributes, default handling, budget constraints and the encoded compatibility rules. They do not download OPT weights or verify live hosting.
