# PC Build Advisor using OPT

BMDS2123 Natural Language Processing coursework prototype. Extracts natural-language PC requirements (English, Malay and Manglish), then recommends hardware using a budget-aware compatibility engine.

## Deploy on Streamlit Community Cloud

1. Extract the delivered ZIP. Upload its CONTENTS (including hidden `.streamlit` and `.gitignore` if using Git) to the root of a **private** GitHub repository. Do not upload the ZIP itself as the app. Check dataset licensing before uploading; no licence was provided in the supplied archive.
2. Sign in at https://share.streamlit.io/ with GitHub and authorize access to that repository. Create an app from the repository.
3. Select branch `main`, main file path `app.py`, and Python **3.11** in Advanced settings. Deploy.
4. Start with the default rule-based mode. Choose pretrained OPT-125M in the sidebar to test few-shot inference (first use downloads weights; free CPU hosting may be too limited). A complete trained adapter automatically enables fine-tuned mode and becomes the default. If model loading or inference raises an exception, the app falls back to the baseline and labels it. Resource kills cannot necessarily be caught; use baseline mode or a larger host if that happens.
5. Enter a request, analyse requirements, review/edit the extracted fields, and generate the build. The web app provides template explanations, not OPT-generated explanations.

## Run locally

Use Python 3.11 in a fresh virtual environment.

```bash
python -m venv .venv
# Activate the virtual environment for your OS before installing.
python -m pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

The model dependency uses a CPU PyTorch wheel on Linux/Windows. The app does not need a GPU, API key, or model training at startup. The baseline does not load Torch or download models at runtime.

## Completing the OPT assignment experiment

The supplied `results/run_mode.txt` says **baseline**; no trained adapters were present. The CSV results are genuine supplied baseline results, not OPT results. Do not describe them as fine-tuned results.

- Open `nlpass_completed.ipynb` in Colab, enable a GPU, provide the `csv/` directory or the original `csv.zip`. In setup select `RUN_MODE = "quick"` for a small OPT-125M training run; `RUN_MODE = "full"` for the complete three-model comparison.
- Run all cells and download the newly generated `pc_advisor_results.zip`. Copy its `adapters/extract_125m/` folder into the repository `adapters/` directory, preserving the files and paths. If individual files exceed GitHub's web upload limit, use Git/Git LFS as appropriate, checking that the host retrieves the actual weights rather than pointer files.
- Replace the `results/` CSVs and `run_mode.txt` with the newly exported actual experiment results. Keep the updated notebook as well. Commit and redeploy. The notebook retains explanation training/evaluation; the Streamlit deployment intentionally uses only extraction OPT and template explanations to reduce memory.
- No GPU training is performed in the deployment. Pretrained few-shot OPT is not a substitute for trained/evaluated assignment results.

## Repository contents

- `app.py`: Streamlit interface and session state
- `advisor.py`: deployment model loading, fallback, requirement normalization and compatibility checking
- `pc_engine.py`, `nlu_data.py`, `opt_nlu.py`: original project modules
- `csv/`: 11 supplied component datasets
- `adapters/`: instructions for adding trained weights
- `results/`: unpacked baseline experiment outputs
- `nlpass_completed.ipynb`: training and evaluation notebook
- `requirements.txt`, `.streamlit/config.toml`, `.gitignore`: deployment configuration
- `smoke_test.py`: offline functional test

## Limitations and submission

Prices are catalogue estimates in USD converted using the notebook's fixed USD/MYR rate of 4.3, not current Malaysian prices. Compatibility checks cover only encoded rules; verify clearances, cooler mounting, BIOS and specifications before purchasing. Generated OPT requirements can be inaccurate; review the fields. Synthetic tests are not real-world validation. The supplied dataset source/licence is unverified; verify redistribution permission, keep coursework private and share only as authorized. Declare AI assistance, and retain your own understanding of the implementation.
