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
            self.shots = O.pick_shots(N.build_dataset(100, seed=1, mode='train'), k=4)
        else:
            raise ValueError('Unknown OPT mode')

    def extract(self, text):
        with self.lock:
            preds, _, raw = self.O.run_extraction(self.model, self.tok, [text], shots=self.shots, batch_size=1)
        if preds[0] is None or not any(key in preds[0] for key in N.KEYS):
            return rule_requirements(text), 'Rule-based fallback: OPT did not return usable JSON.', raw[0]
        req = N.normalise(preds[0])
        rules = rule_requirements(text)
        # Budget is a high-impact numeric constraint: never trust a generative
        # model to invent or alter it. Use the explicitly parsed user amount.
        if rules['budget'] is not None:
            changed = (req['budget'], req['currency']) != (rules['budget'], rules['currency'])
            req['budget'], req['currency'] = rules['budget'], rules['currency']
        else:
            # If the user did not state a budget, discard any number hallucinated
            # by OPT. The review form will ask the user to enter one explicitly.
            changed = req['budget'] is not None
            req['budget'] = None
            req['currency'] = rules['currency']
        source = self.mode + (' + deterministic budget validation' if changed else '')
        return req, source, raw[0]


def rule_requirements(text):
    """Deterministic extraction for deployment, including informal budget forms."""
    text = re.sub(r"\bUSD\s*(?=\d)", "$", text, flags=re.I)
    text = re.sub(r"\bMYR\s*(?=\d)", "RM", text, flags=re.I)
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
