"""Deployment layer; rule mode has no dependency on model downloads."""
from pathlib import Path
import threading
import re
import pc_engine as E
import nlu_data as N

ROOT = Path(__file__).resolve().parent
MODEL = 'facebook/opt-125m'
BASELINE = 'Rule-based baseline'
PRETRAINED = 'Pretrained OPT-125M (few-shot)'
FINETUNED = 'Fine-tuned OPT-125M (LoRA)'


def adapter_ready(kind='extract'):
    folder = ROOT / 'adapters' / f'{kind}_125m'
    return (folder / 'adapter_config.json').is_file() and any(
        (folder / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin'))


def available_modes():
    return ([FINETUNED] if adapter_ready() else []) + [BASELINE, PRETRAINED]


def load_engine():
    return E.BuildEngine(E.load_data(ROOT / 'csv'))


class ModelSession:
    def __init__(self, mode):
        import opt_nlu as O
        import torch
        self.O = O
        self.mode = mode
        self.lock = threading.RLock()
        torch.set_num_threads(min(2, torch.get_num_threads()))
        if mode == FINETUNED:
            if not adapter_ready():
                raise FileNotFoundError('No complete extract_125m adapter was supplied.')
            self.tok, self.model = O.load_finetuned(MODEL, str(ROOT / 'adapters' / 'extract_125m'))
            self.shots = None
        elif mode == PRETRAINED:
            self.tok, self.model = O.load_opt(MODEL)
            self.shots = _pretrained_budget_neutral_shots()
        else:
            raise ValueError('Unknown OPT mode')

    def extract(self, text):
        with self.lock:
            preds, _, raw = self.O.run_extraction(
                self.model, self.tok, [text], shots=self.shots, batch_size=1
            )

        # Parse model output first; deterministic user-text parsing then takes
        # precedence for high-impact numeric constraints such as the budget.
        if preds[0] is None or not any(key in preds[0] for key in N.KEYS):
            return rule_requirements(text), 'Rule-based fallback: OPT did not return usable JSON.', raw[0]

        req = N.normalise(preds[0])
        rules = rule_requirements(text)
        parsed_budget = rules.get('budget')
        changed = False

        if parsed_budget is not None:
            # Explicit amount found in the user's own sentence: always trust it.
            changed = (req['budget'], req['currency']) != (parsed_budget, rules['currency'])
            req['budget'] = parsed_budget
            req['currency'] = rules['currency']
            budget_note = 'user amount verified'
        elif self.mode == FINETUNED and N.has_budget_mention(text) and req.get('budget'):
            # Preserve a fine-tuned model's valid extraction only when the user
            # clearly mentioned a budget but used a format the parser missed.
            # The review form still lets the user verify/correct this value.
            budget_note = 'fine-tuned model budget; please verify'
        else:
            # Pretrained few-shot OPT is prone to copying demo amounts (e.g. 9700).
            # Never accept its invented budget when deterministic parsing failed.
            changed = req.get('budget') is not None
            req['budget'] = None
            req['currency'] = rules.get('currency', 'MYR')
            budget_note = 'budget not detected; enter it in the review form'

        source = self.mode
        if changed:
            source += ' + deterministic budget correction'
        if budget_note == 'fine-tuned model budget; please verify':
            source += ' + model budget fallback (verify)'
        elif budget_note == 'budget not detected; enter it in the review form':
            source += ' + budget not detected'
        return req, source, raw[0]



def _pretrained_budget_neutral_shots():
    """Few-shot examples teach fields/use cases without exposing sample prices.

    This reduces the chance that pretrained OPT copies a demonstration budget
    such as RM9,700 into a new user's request.
    """
    examples = [
        (
            "I need a gaming PC with Intel CPU, NVIDIA graphics and 32GB RAM.",
            {"use_case": "gaming", "cpu_brand": "intel", "gpu_brand": "nvidia", "ram_gb": 32},
        ),
        (
            "Please suggest a PC for video editing with an AMD processor and 64GB RAM.",
            {"use_case": "video_editing", "cpu_brand": "amd", "ram_gb": 64},
        ),
        (
            "I want a compact office computer with a monitor and keyboard.",
            {"use_case": "office", "extras": ["monitor", "keyboard"], "compact": True},
        ),
        (
            "Build a programming workstation with an Intel CPU and 1TB SSD.",
            {"use_case": "programming", "cpu_brand": "intel", "storage_gb": 1000},
        ),
    ]
    return [(text, N.normalise(label)) for text, label in examples]


def rule_requirements(text):
    """Deterministic requirements parsed directly from the user's message."""
    text = str(text or "")
    req = N.normalise(N.rule_extract(text))
    budget_info = N.extract_budget_amount(text)
    if budget_info is not None:
        req['budget'], req['currency'] = budget_info
    return req



def baseline_extract(text):
    return rule_requirements(text), BASELINE, None


def recommend(engine, req):
    req = N.normalise(req)
    if not req['budget']:
        return {'ok': False, 'message': 'Please specify a positive budget.'}, []
    result = engine.recommend(req)
    violations = E.validate_build(result, engine.d) if result['ok'] else []
    return result, violations
