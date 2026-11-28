# PC Build Advisor using OPT-125M

A Streamlit PC recommendation prototype that extracts requirements from English, Malay and Manglish, searches a cleaned PC parts catalogue, checks simplified hardware compatibility and presents a budget-aware build.

## GitHub → Streamlit Community Cloud

1. **Extract** the distributed ZIP. Upload the **contents** (including the hidden `.streamlit/` folder) to the root of your GitHub repository, not the ZIP itself or a nested parent folder.
2. In Streamlit Community Cloud, choose that repository's `main` branch and set main file `app.py`. Choose **Python 3.11** in Advanced Settings when creating the app. The pinned CPU PyTorch requirements can fail on unsupported new Python versions.
3. Deploy and select one of the three NLP modes in the sidebar. The first use of either OPT mode may need to download the ~125-million-parameter base checkpoint from Hugging Face. **The base checkpoint is NOT stored in this ZIP**, since it is too large for ordinary GitHub file uploads.
4. If the base checkpoint cannot be downloaded or the free Streamlit instance runs out of memory, the app warns and falls back to the rule-based parser. A fallback is **not** evidence that OPT successfully ran.
5. Keep the repository private for coursework unless your tutor and data licences permit sharing. The dataset has static USD prices and the displayed RM estimates use a **fixed** 4.30 multiplier, not current Malaysian store prices.

## Included models and saved results

| Mode | Shipped in this ZIP? | How it runs |
| --- | --- | --- |
| Rule-based baseline | Yes | Local Python; no model download |
| Pretrained OPT-125M (few-shot) | Model *code* and labelled prompt examples; **not base weights** | Downloads `facebook/opt-125m`, adds 4 labelled examples, runs inference |
| Fine-tuned OPT-125M (LoRA) | **Genuine `adapters/extract_125m/` weights included**, from the user's earlier training run | Downloads the same base checkpoint, applies saved LoRA adapter, runs inference |

The earlier training generated the LoRA adapter using 2,000 synthetic examples and three epochs. Its recorded metrics are preserved, **not recreated**:

- `results/opt125m_training.csv` – saved training loss.
- `results/opt125m_evaluation.csv` – rule baseline, LoRA, pretrained zero-shot/few-shot on the earlier held-out and manually written examples.
- `results/opt125m_predictions.csv` – saved model-level responses from that run.
- `results/model_comparison.csv` and `results/opt_raw_predictions.csv` – original, separate evaluations bundled with the most recent project.
- `results/run_manifest.json` – dataset, checkpoint and provenance information.

**Do not combine metrics from separate runs as if they share an identical test setup.** If the rubric requires a strict head-to-head, evaluate both current models on a common held-out set in Colab (see `notebooks/Train_OPT125M_for_Streamlit.ipynb`). We could not run actual OPT inference in the packaging environment; only code and offline engine tests were run.

In the Streamlit UI, **Saved OPT-125M experimental results** displays/downloads these files. **Download this analysis (JSON)** and **Download session predictions CSV** save genuine live outputs for queries you submit, including model-only JSON and verified requirements. Streamlit Community Cloud session history is not persistent: download it before closing the session. No user queries are automatically pushed to GitHub.

## Requirements and budget bug fixes

- RAM/SSD: `8GB RAM 256GB SSD`, `32GB RAM 1TB NVMe`, `RAM: 32GB SSD: 512GB`, `16GB DDR5 RAM` and 24GB minimum memory requests are parsed and shown in **Review core requirements**.
- A RAM/storage request is a **minimum**; a compatible 32GB kit can satisfy a 24GB minimum. SSD selection no longer rejects larger drives arbitrarily.
- The manual review form can override previously extracted RAM and SSD constraints; stale catalogue filters are removed.
- If the selected budget cannot afford a discrete GPU and there is no explicit GPU constraint, the solver can find a compatible integrated-graphics alternative. **Gaming with an iGPU is clearly flagged as performance-limited.**
- The engine reports infeasible catalogue constraints separately from an insufficient budget, and can show the cheapest available candidate when found.
- Product compatibility is only checked against simplified rules in this dataset; real BIOS, connector, memory-QVL, PSU transient, GPU clearance and cooler mounting need independent verification.

## RAM and SSD extraction: corrected November 2026

The parser now handles separate component-labelled patterns instead of taking
unqualified numbers from nearby text. It recognizes, among others:

- `16GB RAM, 512GB SSD`
- `RAM is 32GB and SSD is 1TB`
- `RAM should be 32GB and storage should be 2TB`
- `2 x 16GB RAM` (reported as 32GB total system memory)
- `SSD 512GB, RAM: 32GB`
- `32GB 内存，1TB 固态硬盘`

It avoids treating GPU-only VRAM (for example, `RTX 4070 16GB VRAM`) as
system RAM. `1TB` is normalized to `1000GB` to match the processed catalogue.
The explicit RAM/SSD values in the user's message override the OPT model's
JSON guess in both pretrained and fine-tuned modes; the raw model-only output
is still preserved separately for inspection and evaluation. The detected
values appear in Review core requirements and are passed as minimum-capacity
filters to the build engine. Values manually changed in the review form
replace the previously extracted RAM/SSD filters.

After a successful build, the app now offers a JSON download containing the
original text, selected model, raw output, model-only prediction, rule-verified
extraction, user-reviewed values and final recommendation. It also offers a
CSV download of all completed analyses in the current Streamlit session.
Streamlit Community Cloud sessions are temporary: download the records to
keep them after the session ends.

Regression evidence is saved under `results/ram_ssd_parser_regression.csv` and
`results/ram_ssd_build_regression.csv`; rerun with `python test_ram_ssd_requirements.py`.
These are offline parser/recommendation tests, not live OPT inference tests.

## Run and test locally

Use Python 3.11:

```sh
python -m pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

`smoke_test.py` performs **offline** tests covering RAM/storage detection, tight budgets, compatibility checks, manual corrections and adapter file presence. It does **not** prove that the on-demand pretrained model downloads or that live OPT inference works on a specific hosting account.

## Project layout

- `app.py` – Streamlit interface and per-session result export.
- `advisor.py`, `nlu_data.py`, `catalog_nlu.py`, `opt_nlu.py` – NLP model and preference extraction.
- `pc_engine.py` – parts recommendation and compatibility engine.
- `data/processed/catalog/` – all 25 cleaned source categories.
- `data/processed/engine/` – eleven tables used for build generation.
- `adapters/extract_125m/` – real fine-tuned LoRA adapter, **not** a complete base model.
- `results/` – saved experiments, documentation and summaries.
- `notebooks/` – original dedicated OPT training/retraining notebook.

Please declare AI tools/assistance and comply with copyright, dataset licensing and your institution's academic integrity rules.
