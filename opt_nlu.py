"""
opt_nlu.py
==========
Everything that touches Meta's OPT models (Hugging Face port).

  load_opt()            load a base OPT model in fp16
  generate()            batched greedy generation
  run_extraction()      zero-shot / few-shot requirement extraction
  finetune_lora()       LoRA fine-tuning with prompt-masked loss
  load_finetuned()      base + adapter -> merged model for inference
  OptNLU                inference wrapper (extract + explain + fallbacks)
  check_grounded()      lexical hallucination check for explanations
"""
import gc
import json
import re
import time

import torch

import nlu_data as N

OPT_SIZES = {"125m": "facebook/opt-125m", "350m": "facebook/opt-350m",
             "1.3b": "facebook/opt-1.3b", "2.7b": "facebook/opt-2.7b"}


# ------------------------------------------------------------------ loading
def load_opt(name, dtype=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


def free():
    """Call AFTER `del model, tok` in the caller (del inside a function does
    not drop the caller's references)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=100, batch_size=8, stop=None):
    """Greedy decoding. Returns only the newly generated text per prompt."""
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        context = getattr(model.config, "max_position_embeddings", 2048)
        tok.truncation_side = "left"
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=context - max_new_tokens).to(next(model.parameters()).device)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id)
        if stop:
            kw.update(stop_strings=stop, tokenizer=tok)
        try:
            seq = model.generate(**enc, **kw)
        except (TypeError, ValueError):       # old transformers: no stop_strings
            kw.pop("stop_strings", None)
            kw.pop("tokenizer", None)
            seq = model.generate(**enc, **kw)
        new = seq[:, enc["input_ids"].shape[1]:]
        outs += [tok.decode(s, skip_special_tokens=True) for s in new]
    return outs


def pick_shots(train, k=4, seed=0):
    """k training examples with distinct use cases (for few-shot prompts)."""
    import random
    rng = random.Random(seed)
    pool = train[:]
    rng.shuffle(pool)
    shots, seen = [], set()
    for t, l in pool:
        if l["use_case"] not in seen:
            seen.add(l["use_case"])
            shots.append((t, l))
        if len(shots) == k:
            break
    return shots


def run_extraction(model, tok, texts, shots=None, batch_size=8,
                   prompt_builder=None):
    """Returns (list of dict|None, seconds_per_example, raw text)."""
    if shots:
        prompts = [N.few_shot_prompt(t, shots) for t in texts]
    else:
        build_prompt = prompt_builder or N.extract_prompt
        prompts = [build_prompt(t) for t in texts]
    t0 = time.time()
    raw = generate(model, tok, prompts, max_new_tokens=128,
                   batch_size=batch_size, stop=["}"])
    dt = (time.time() - t0) / max(len(texts), 1)
    preds = [N.parse_model_output(r) for r in raw]
    return preds, dt, raw


# ---------------------------------------------------------- fine-tuning
class _SFTData(torch.utils.data.Dataset):
    def __init__(self, tok, pairs, max_len=640):
        self.items = []
        for prompt, completion in pairs:
            p = tok(prompt, add_special_tokens=True)["input_ids"]
            c = tok(" " + completion.strip(), add_special_tokens=False)["input_ids"]
            c = c + [tok.eos_token_id]
            if len(c) >= max_len:
                raise ValueError("Completion exceeds max_len; increase the training context length")
            p = p[-(max_len - len(c)):]
            ids = p + c
            labels = [-100] * len(p) + c    # loss only on completion
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(pad_id):
    def fn(batch):
        m = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), m), -100, dtype=torch.long)
        att = torch.zeros((len(batch), m), dtype=torch.long)
        for i, (a, b) in enumerate(batch):          # right padding for training
            ids[i, :len(a)] = torch.tensor(a)
            lab[i, :len(b)] = torch.tensor(b)
            att[i, :len(a)] = 1
        return dict(input_ids=ids, labels=lab, attention_mask=att)
    return fn


def finetune_lora(name, pairs, out_dir, epochs=2, lr=2e-4, batch_size=8,
                  grad_accum=2, r=16, max_len=640, grad_ckpt=False, seed=42):
    """LoRA fine-tune OPT `name` on (prompt, completion) pairs.
    Returns (train_seconds, final_loss)."""
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              TrainingArguments, Trainer, set_seed)
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    if not pairs:
        raise ValueError("No training pairs supplied")
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if cuda else torch.float32)
    model.to("cuda" if cuda else "cpu")
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():                      # fp16 base, fp32 adapters
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    ds = _SFTData(tok, pairs, max_len)
    args = TrainingArguments(
        output_dir=out_dir + "_ckpt", per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, num_train_epochs=epochs,
        learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * epochs * ((len(ds) + batch_size * grad_accum - 1) // (batch_size * grad_accum)))),
        fp16=cuda, logging_steps=20, save_strategy="no", report_to="none",
        remove_unused_columns=False, optim="adamw_torch", seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=_collate(tok.pad_token_id))
    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    loss = float(result.training_loss)
    del model, trainer
    free()
    return secs, loss


def load_finetuned(name, adapter_dir):
    """Load the base OPT plus a real LoRA adapter for inference.

    Keeping PEFT layers unmerged is friendlier to RAM-limited CPU hosting;
    a merged model may create a second temporary copy of the base weights.
    """
    from peft import PeftModel
    tok, base = load_opt(name)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tok, model


# -------------------------------------------------------- training pairs
def extraction_pairs(dataset):
    return [(N.extract_prompt(t), N.to_json(l)) for t, l in dataset]


def explanation_pairs(engine, dataset, pc_engine):
    """(facts -> explanation) pairs from real engine output."""
    pairs = []
    for t, l in dataset:
        res = engine.recommend(l)
        if not res["ok"]:
            continue
        pairs.append((N.explain_prompt(pc_engine.facts_text(res)),
                      pc_engine.template_explanation(res)))
    return pairs


# ----------------------------------------------------------- grounding check
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MODEL = re.compile(
    r"\b(?:RTX|GTX|RX|GT|Arc|Ryzen|Core(?: Ultra)?|Xeon|i[3579]-?|Radeon|GeForce)"
    r"[\w\- ]*?\d[\w\-]*", re.I)


def _sig_numbers(text):
    out = set()
    for m in _NUM.findall(text):
        v = m.replace(",", "").rstrip(".")
        if len(v.replace(".", "")) >= 2:        # ignore 1-digit numbers
            out.add(v)
    return out


def _models(text):
    return {re.sub(r"\s+", " ", m).strip().lower() for m in _MODEL.findall(text)}


def check_grounded(text, facts):
    """Lexical grounding: every significant number and every CPU/GPU model
    mentioned in `text` must also appear in `facts`.
    Returns (is_grounded, unsupported_numbers, unsupported_models)."""
    fnums = _sig_numbers(facts)
    ftxt = facts.lower()
    bad_n = sorted(n for n in _sig_numbers(text) if n not in fnums)
    bad_m = sorted(m for m in _models(text) if m not in ftxt)
    return (not bad_n and not bad_m), bad_n, bad_m


# ----------------------------------------------------------- inference wrapper
class OptNLU:
    """Extract requirements + explain builds with fine-tuned OPT, with
    rule-based fallbacks so the demo never returns nothing."""

    def __init__(self, extract_model, extract_tok, explain_model=None,
                 explain_tok=None):
        self.em, self.et = extract_model, extract_tok
        self.xm, self.xt = explain_model or extract_model, explain_tok or extract_tok

    def extract(self, text):
        if self.em is None:
            return N.rule_extract(text), True, "OPT not loaded"
        preds, _, raw = run_extraction(self.em, self.et, [text], batch_size=1)
        p = preds[0]
        if p is None:
            return N.rule_extract(text), True, raw[0]
        p = N.normalise(p)
        fb = N.rule_extract(text)
        # Never accept a numeric budget invented by the language model.
        # Only a budget explicitly found in the user's text is authoritative.
        had_explicit_budget = fb["budget"] is not None
        changed = (p["budget"], p["currency"]) != (fb["budget"], fb["currency"])
        p["budget"] = fb["budget"]
        p["currency"] = fb["currency"]
        return p, (not had_explicit_budget or changed), raw[0]

    def explain(self, res, pc_engine):
        if self.xm is None:
            return pc_engine.template_explanation(res), False, ([], [])
        facts = pc_engine.facts_text(res)
        txt = generate(self.xm, self.xt, [N.explain_prompt(facts)],
                       max_new_tokens=260, batch_size=1)[0].split("\n\n")[0].strip()
        ok, bn, bm = check_grounded(txt, facts)
        if not ok or len(txt) < 40:
            return pc_engine.template_explanation(res), False, (bn, bm)
        return txt, True, ([], [])
