# OPT adapters (not supplied)

The uploaded results were generated in baseline mode, so no trained weights
are present. Do not create empty placeholder weights.

After a successful GPU notebook run, extract `adapters/extract_125m/` from
`pc_advisor_results.zip` into this directory. Keep all adapter/tokenizer files.
The deployed application automatically detects a complete extraction adapter.
It does not load larger models or generate OPT explanations; explanations are
template-based to keep the hosted prototype lightweight.
