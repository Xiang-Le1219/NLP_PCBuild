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
        # Budget must come from an explicit user statement. OPT can otherwise
        # hallucinate a plausible-looking number (e.g. RM5,100) when none was given.
        if rules['budget'] is not None:
            changed = (req['budget'], req['currency']) != (rules['budget'], rules['currency'])
            req['budget'], req['currency'] = rules['budget'], rules['currency']
            source_suffix = ' + rule-based budget correction' if changed else ''
        else:
            changed = req['budget'] is not None
            req['budget'] = None
            req['currency'] = rules['currency']
            source_suffix = ' + unmentioned budget cleared' if changed else ''
        source = self.mode + source_suffix
        return req, source, raw[0]


def rule_requirements(text):
    # The original regex recognizes RM and $, but not an explicit USD/MYR prefix.
    # Normalize the prefix for the deployed baseline without altering experiment code.
    text = re.sub(r"\bUSD\s*(?=\d)", "$", text, flags=re.I)
    text = re.sub(r"\bMYR\s*(?=\d)", "RM", text, flags=re.I)
    return N.normalise(N.rule_extract(text))


def baseline_extract(text):
    return rule_requirements(text), BASELINE, None


def recommend(engine, req):
    # budget_strategy is a UI/build-engine preference, not an NLP entity.
    # Preserve it separately because N.normalise intentionally keeps only
    # fields extracted from the user's natural-language request.
    budget_strategy = (req.get('budget_strategy', 'closest_to_budget')
                       if isinstance(req, dict) else 'closest_to_budget')
    req = N.normalise(req)
    req['budget_strategy'] = (budget_strategy if budget_strategy in
                              ('closest_to_budget', 'best_value')
                              else 'closest_to_budget')
    if not req['budget']:
        return {'ok': False, 'message': 'Please specify a positive budget.'}, []
    result = engine.recommend(req)
    violations = E.validate_build(result, engine.d) if result['ok'] else []
    return result, violations
