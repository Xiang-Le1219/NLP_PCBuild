"""PC Build Advisor orchestration: NLU extraction + catalogue-aware preferences."""
from pathlib import Path
import threading
import pc_engine as E
import nlu_data as N
import catalog_nlu as C

ROOT = Path(__file__).resolve().parent
MODEL = "facebook/opt-125m"
DEFAULT_BUDGET_MYR = 4000
BASELINE = "Rule-based baseline"
PRETRAINED = "Pretrained OPT-125M (few-shot)"
FINETUNED = "Fine-tuned OPT-125M (LoRA)"


def adapter_ready(kind="extract"):
    folder = ROOT / "adapters" / f"{kind}_125m"
    return (folder / "adapter_config.json").is_file() and any(
        (folder / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def available_modes():
    modes = [BASELINE, PRETRAINED]
    if adapter_ready("extract"):
        modes.append(FINETUNED)
    return modes


def load_engine():
    return E.BuildEngine(E.load_processed_data(ROOT / "data" / "processed" / "engine"))


def _explicit_core_fields(text):
    """Deterministic extraction for high-impact fields from the original text."""
    corrected, _ = C.correct_typos(text)
    rules = N.normalise(N.rule_extract(corrected))
    budget = N.extract_budget_amount(corrected)
    if budget:
        rules["budget"], rules["currency"] = budget
        rules["budget_is_default"] = False
    else:
        rules["budget"] = DEFAULT_BUDGET_MYR
        rules["currency"] = "MYR"
        rules["budget_is_default"] = True
    # Ensure model-only guesses do not become user-stated constraints.
    rules["ram_gb"] = rules.get("ram_gb") if re_search(r"\b(?:ram|memory|memori|ddr[45])\b|内存", corrected) else None
    rules["storage_gb"] = rules.get("storage_gb") if re_search(r"\b(?:ssd|storage|nvme|hard drive|hard disk|solid state drive)\b|固态硬盘", corrected) else None
    return rules, corrected


def re_search(pattern, text):
    import re
    return re.search(pattern, str(text), flags=re.I) is not None


def rule_requirements(text, data=None):
    """Rule-based baseline plus catalogue-derived component attribute parsing."""
    corrected, typo_changes = C.correct_typos(text)
    rules, corrected = _explicit_core_fields(corrected)
    preferences, corrected, _ = C.extract_component_preferences(corrected, data or {})
    # The generic catalogue patterns can accidentally treat the SSD number
    # following 'RAM' as RAM (e.g. '8GB RAM 256GB SSD'). Explicitly parsed
    # core capacities are authoritative; remove false numeric constraints.
    if "ram" in preferences:
        preferences["ram"].pop("total_gb", None)
    if "ssd" in preferences:
        preferences["ssd"].pop("capacity", None)
    if rules.get("ram_gb"):
        preferences.setdefault("ram", {})["total_gb"] = {"min": rules["ram_gb"]}
    if rules.get("storage_gb"):
        preferences.setdefault("ssd", {})["capacity"] = {"min": rules["storage_gb"]}
    rules["component_preferences"] = preferences
    rules["typo_corrections"] = typo_changes
    rules["corrected_text"] = corrected
    rules["attribute_count"] = sum(len(v) for v in preferences.values())
    return rules


class ModelSession:
    """Lazy-loaded OPT-125m requirement extractor."""
    def __init__(self, mode):
        import opt_nlu as O
        import torch
        self.O = O
        self.mode = mode
        self.lock = threading.RLock()
        self.shots = None
        torch.set_num_threads(min(2, torch.get_num_threads()))
        if mode == FINETUNED:
            if not adapter_ready("extract"):
                raise FileNotFoundError("No complete adapters/extract_125m adapter was supplied.")
            self.tok, self.model = O.load_finetuned(
                MODEL, str(ROOT / "adapters" / "extract_125m")
            )
        elif mode == PRETRAINED:
            self.tok, self.model = O.load_opt(MODEL)
            # The base OPT checkpoint is not instruction tuned; a few genuine
            # labelled training examples provide the expected JSON format.
            import json
            training_file = ROOT / "data" / "nlu" / "train.jsonl"
            if training_file.is_file():
                with training_file.open(encoding="utf-8") as f:
                    examples = [(row["text"], row["label"])
                                for line in f if (row := json.loads(line))]
                self.shots = O.pick_shots(examples, k=4, seed=42)
        else:
            raise ValueError("Unknown OPT mode")

    def extract(self, text, data=None):
        corrected, typo_changes = C.correct_typos(text)
        rules = rule_requirements(corrected, data)
        with self.lock:
            preds, _, raw = self.O.run_extraction(
                self.model, self.tok, [corrected], shots=self.shots, batch_size=1,
                prompt_builder=(N.extract_prompt_training
                                if self.mode == FINETUNED else None)
            )

        model_prediction = N.normalise(preds[0]) if preds and preds[0] else {}
        # Start with model interpretation, then use the user's explicit wording
        # as authoritative. Unmentioned fields are filled with application defaults.
        req = dict(model_prediction)
        for key in ("use_case", "cpu_brand", "gpu_brand", "ram_gb",
                    "storage_gb", "extras", "compact"):
            req[key] = rules.get(key)
        req["budget"] = rules["budget"]
        req["currency"] = rules["currency"]
        req["budget_is_default"] = rules["budget_is_default"]
        req["component_preferences"] = rules["component_preferences"]
        req["typo_corrections"] = typo_changes
        req["corrected_text"] = corrected
        req["attribute_count"] = rules["attribute_count"]

        # Keep the model's use-case inference only if the deterministic baseline
        # could not identify a more specific use case, and the message contains
        # a use-case cue rather than being completely silent.
        if rules["use_case"] == "general" and model_prediction.get("use_case") in E.USE_CASES:
            if re_search(r"\b(?:gaming|stream|editing|programming|coding|office|ai|machine learning)\b", corrected):
                req["use_case"] = model_prediction["use_case"]
            else:
                req["use_case"] = "general"

        req["model_prediction"] = model_prediction if preds and preds[0] is not None else None
        has_json = bool(preds and preds[0] is not None)
        source = self.mode if has_json else self.mode + " + rule fallback"
        if not has_json:
            source += " (model returned invalid JSON)"
        return req, source, raw[0] if raw else ""


def analyse_requirements(text, data, mode=BASELINE, model_session=None):
    """One entry point used by Streamlit; returns req, source, raw, field report."""
    if mode == BASELINE:
        req = rule_requirements(text, data)
        source, raw = BASELINE, None
    else:
        if model_session is None:
            model_session = ModelSession(mode)
        req, source, raw = model_session.extract(text, data)
    report = C.requirement_field_report(data, req.get("component_preferences", {}), req)
    return req, source, raw, report


def baseline_extract(text, data=None):
    req = rule_requirements(text, data)
    return req, BASELINE, None


def recommend(engine, req):
    """Preserve extracted catalogue preferences while normalising core NLU fields."""
    original = dict(req or {})
    norm = N.normalise(original)
    norm["component_preferences"] = {
        key: dict(value) for key, value in
        (original.get("component_preferences") or {}).items()
    }
    # Review-form values take priority over the initial NLP extraction.
    for comp, col, key in (("ram", "total_gb", "ram_gb"),
                           ("ssd", "capacity", "storage_gb")):
        norm["component_preferences"].setdefault(comp, {}).pop(col, None)
        if norm.get(key):
            norm["component_preferences"][comp][col] = {"min": norm[key]}
    for comp, key in (("cpu", "cpu_brand"), ("gpu", "gpu_brand")):
        norm["component_preferences"].setdefault(comp, {}).pop("brand", None)
        if norm.get(key) in {"amd", "intel", "nvidia"}:
            norm["component_preferences"][comp]["brand"] = {"eq": norm[key]}
    norm["budget_strategy"] = original.get("budget_strategy", "closest_to_budget")
    norm["budget_is_default"] = bool(original.get("budget_is_default", False))
    if original.get("budget") is not None:
        norm["budget"] = original["budget"]
    if original.get("currency"):
        norm["currency"] = original["currency"]
    result = engine.recommend(norm)
    violations = E.validate_build(result, engine.d) if result["ok"] else []
    return result, violations
