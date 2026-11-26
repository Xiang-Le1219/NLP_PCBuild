# Implementation update and validation notes

This repository revision includes the requested PC Build Advisor improvements.

## What changed

1. **Catalogue-wide extraction and review**
   - `catalog_nlu.py` defines a 98-field schema over the 11 component CSVs included in this repository.
   - The Streamlit app lists every schema field as `User-stated`, `Default`, or `Unavailable`.
   - Explicit user values become component filters. Fields not mentioned remain `Auto / system default`; the build engine chooses them based on use case, compatibility and budget.
   - Core values such as CPU/GPU brand, RAM capacity and SSD capacity are also reflected in the field report.

2. **Improved typo recovery and attribute association**
   - Common PC terms are corrected before extraction.
   - Added regression checks for misspellings and values such as `32GB DDR5 6000 RAM`, `liquid cooler 240mm`, and motherboard socket notation.
   - Fixed nearby component association so `black mouse, white case` assigns black to the mouse and white to the case, rather than incorrectly applying both values to the case.

3. **Analysis duration**
   - `app.py` measures elapsed time around requirement analysis and displays it in the Results section.
   - First-time OPT model loading is included in the displayed duration.

4. **Three-method evaluation**
   - The simplified notebook compares:
     - Rule-based baseline
     - Pretrained OPT-125M
     - Fine-tuned OPT-125M (LoRA)
   - Both Synthetic and OOD test sets are used.
   - Metrics include JSON validity, exact match, mean field accuracy, per-field accuracy and time per example.
   - When a model cannot be loaded, it is explicitly marked `Not evaluated`; no synthetic score is invented.

5. **Simplified notebook**
   - The older, longer notebook has been replaced in this repository by `notebooks/PCBuild_OPT_Simplified.ipynb` (36 cells).
   - Model evaluation is enabled by default; adapter retraining is disabled by default because an extraction adapter is supplied.
   - Set `RETRAIN_EXTRACT_ADAPTER=True` only when you intentionally want to retrain on a GPU.

## Validation performed in this environment

- 8 Python files passed `py_compile`.
- `python smoke_test.py`: passed.
- `python test_budget_strategy.py`: 28 budget scenarios passed.
- The simplified notebook's non-model cells ran successfully with `RUN_MODEL_EVALUATION=False` for offline validation.
- The component audit contains 98 fields and the cleaned catalogue contains 5,273 usable products.

## Not run here

- Actual Hugging Face download/inference for pretrained OPT-125M.
- Actual inference with the supplied LoRA adapter.
- LoRA training.
- Live Streamlit server startup/deployment.

The runtime used for this validation does not have Streamlit, Transformers or PEFT installed and did not have model-download access. Run the notebook with the required dependencies and internet/GPU resources to obtain actual pretrained and fine-tuned OPT performance measurements.
