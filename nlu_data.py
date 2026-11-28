"""
nlu_data.py
===========
Natural-language understanding side of the PC Build Advisor.

* SCHEMA / canonical JSON format that OPT must produce
* Synthetic English / Malay / Manglish ("rojak") request generator
  - phrases starting with '*' are HELD OUT: they never appear in the
    training split and appear in ~50% of the test split, so the test
    set measures generalisation, not memorisation.
* Rule-based (regex / keyword) baseline extractor
* Prompt builders and evaluation metrics
"""
import json
import re
import random

KEYS = ["use_case", "budget", "currency", "cpu_brand", "gpu_brand",
        "ram_gb", "storage_gb", "extras", "compact"]

# ---------------------------------------------------------------- phrase banks
USE = {
    "gaming": [
        ("gaming", "en"), ("playing AAA games", "en"), ("esports", "en"),
        ("playing Valorant and CS2", "en"), ("*playing Cyberpunk and Elden Ring", "en"),
        ("main game", "ms"), ("game berat", "ms"), ("game online", "ms"),
        ("*main game FPS", "ms"), ("gaming lah", "mix"), ("play games one", "mix"),
        ("*game Dota", "mix")],
    "video_editing": [
        ("video editing", "en"), ("editing videos", "en"), ("3D rendering", "en"),
        ("content creation", "en"), ("*Premiere Pro and After Effects", "en"),
        ("edit video", "ms"), ("edit video YouTube", "ms"), ("*render video", "ms"),
        ("edit video lah", "mix"), ("do content creation", "mix"),
        ("*Davinci Resolve", "mix")],
    "office": [
        ("office work", "en"), ("Excel and Word", "en"), ("school assignments", "en"),
        ("*spreadsheets and emails", "en"), ("kerja pejabat", "ms"),
        ("buat assignment", "ms"), ("kerja office", "ms"), ("*kerja Excel", "ms"),
        ("office work only", "mix"), ("study one", "mix"), ("*do assignment", "mix")],
    "programming": [
        ("programming", "en"), ("coding", "en"), ("software development", "en"),
        ("*web development and Docker", "en"), ("buat coding", "ms"),
        ("develop aplikasi", "ms"), ("*program Python", "ms"),
        ("coding lah", "mix"), ("do programming", "mix"), ("*code Java", "mix")],
    "streaming": [
        ("streaming", "en"), ("live streaming on Twitch", "en"),
        ("game streaming", "en"), ("*streaming on YouTube", "en"),
        ("buat live stream", "ms"), ("streaming game", "ms"),
        ("*live stream TikTok", "ms"), ("stream game lah", "mix"),
        ("*stream Twitch", "mix")],
    "ai_ml": [
        ("machine learning", "en"), ("AI model training", "en"),
        ("deep learning", "en"), ("running local LLMs", "en"),
        ("*data science with PyTorch", "en"), ("belajar machine learning", "ms"),
        ("latih model AI", "ms"), ("*kerja deep learning", "ms"),
        ("do AI stuff", "mix"), ("run LLM", "mix"), ("*train neural network", "mix")],
    "general": [
        ("daily use", "en"), ("general use", "en"),
        ("browsing and watching Netflix", "en"), ("everyday tasks", "en"),
        ("*light everyday work", "en"), ("guna harian", "ms"), ("guna biasa", "ms"),
        ("tengok Netflix", "ms"), ("*guna rumah", "ms"), ("normal use only", "mix"),
        ("surf internet", "mix"), ("*basic use", "mix")],
}
OPEN = {
    "en": ["I want to build a PC for {u}.", "Can you recommend PC parts for {u}?",
           "Need a computer for {u}.", "Help me pick components for a {u} PC.",
           "*Looking to put together a desktop for {u}."],
    "ms": ["Saya nak build PC untuk {u}.", "Tolong cadangkan komponen komputer untuk {u}.",
           "Nak rakit PC untuk {u}.", "Boleh cadangkan PC untuk {u}?",
           "*Saya perlukan komputer baru untuk {u}."],
    "mix": ["Boss, I want build PC for {u} lah.", "Bro can recommend PC for {u} or not?",
            "Eh got any PC build for {u} ah?", "Wah need new PC for {u} la.",
            "*Hi ah, planning to build one PC for {u} leh."],
}
BUDGET = {
    ("MYR", "en"): ["My budget is RM{n}", "Budget RM{n}", "around RM{n}",
                    "under RM{n}", "*not more than RM{n}"],
    ("MYR", "ms"): ["bajet RM{n}", "bawah RM{n}", "dalam RM{n}",
                    "lebih kurang RM{n}", "*maksimum RM{n}", "bajet {n} ringgit"],
    ("MYR", "mix"): ["budget RM{n} only", "RM{n} can ah", "{n} ringgit la",
                     "budget {n} only", "*budget around RM{n} la"],
    ("USD", "en"): ["my budget is ${n}", "budget ${n}", "under ${n}",
                    "{n} USD", "*around {n} dollars"],
}
CPU = {"amd": ["AMD CPU", "Ryzen processor", "guna AMD", "*Ryzen only"],
       "intel": ["Intel CPU", "Intel Core processor", "guna Intel", "*Intel chip please"]}
GPU = {"nvidia": ["Nvidia GPU", "RTX card", "kad grafik Nvidia", "*NVIDIA graphics"],
       "amd": ["Radeon GPU", "AMD graphics card", "kad grafik Radeon", "*AMD GPU"],
       "intel": ["Intel Arc GPU", "Arc graphics"]}
BOTH = {("amd", "amd"): ["all AMD build", "semua AMD", "full AMD", "*AMD everything"],
        ("intel", "nvidia"): ["Intel and Nvidia combo", "Intel CPU with Nvidia card",
                              "*Intel + RTX"]}
RAM = ["{n}GB RAM", "RAM {n}GB", "{n} GB of memory", "ram {n}g", "*{n}gig memory"]
SSD = {500: ["500GB SSD", "storage 500GB", "SSD 500GB"],
       1000: ["1TB SSD", "storage 1TB", "SSD 1000GB", "*1 TB NVMe"],
       2000: ["2TB SSD", "2TB storage", "SSD 2000GB", "*2 TB NVMe"]}
EXTRA_WORDS = {"monitor": ["monitor", "monitor", "*display"],
               "keyboard": ["keyboard", "keyboard", "papan kekunci"],
               "mouse": ["mouse", "mouse", "tetikus"]}
EXTRA_JOIN = {"en": ["with {x}", "include {x}", "also need {x}"],
              "ms": ["dengan {x}", "nak {x} sekali"],
              "mix": ["{x} sekali", "{x} also can", "got {x} or not"]}
COMPACT = ["small form factor", "compact build", "mini ITX", "PC kecil",
           "small casing", "*as small as possible"]


def _pick(rng, items, mode, key=lambda x: x):
    """Train mode never sees '*' items; test mode sees them ~50% of the time."""
    star = [i for i in items if key(i).startswith("*")]
    normal = [i for i in items if not key(i).startswith("*")]
    if mode == "test" and star and rng.random() < 0.5:
        choice = rng.choice(star)
    else:
        choice = rng.choice(normal)
    return choice


def _clean(s):
    return s[1:] if s.startswith("*") else s


# ------------------------------------------------------------------ generator
def make_sample(rng, mode="train"):
    """Return (text, label_dict)."""
    lang = rng.choice(["en", "ms", "mix"], ) if False else rng.choice(["en", "ms", "mix"])
    uc = rng.choice(list(USE))
    uphrase, ulang = _pick(rng, USE[uc], mode, key=lambda t: t[0])
    uphrase = _clean(uphrase)
    opener = _clean(_pick(rng, OPEN[lang], mode)).format(u=uphrase)

    cur = "USD" if rng.random() < 0.15 else "MYR"
    if cur == "USD":
        n = rng.choice(range(500, 3501, 100))
        bl = "en"
    else:
        n = rng.choice(list(range(1500, 15001, 100)))
        bl = lang
    shown = str(n)
    r = rng.random()
    if n % 1000 == 0 and r < 0.35:
        shown = f"{n // 1000}k"
    elif n >= 1000 and r < 0.55:
        shown = f"{n:,}"
    elif cur == "MYR" and n % 500 == 0 and n >= 2000 and r < 0.75:
        shown = f"{n / 1000:g}k"
    if cur == "MYR" and bl != "en" and n % 1000 == 0 and rng.random() < 0.25:
        bphrase = rng.choice(["bajet {k} ribu", "dalam {k} ribu", "lebih kurang RM{k} ribu"]
                             ).format(k=n // 1000)
    else:
        bphrase = _clean(_pick(rng, BUDGET[(cur, bl)], mode)).format(n=shown)

    label = dict(use_case=uc, budget=n, currency=cur, cpu_brand="any",
                 gpu_brand="any", ram_gb=None, storage_gb=None, extras=[],
                 compact=False)
    clauses = [bphrase]

    if rng.random() < 0.40:
        t = rng.random()
        if t < 0.25:
            pair = rng.choice(list(BOTH))
            label["cpu_brand"], label["gpu_brand"] = pair
            clauses.append(_clean(_pick(rng, BOTH[pair], mode)))
        else:
            if rng.random() < 0.5:
                b = rng.choice(list(CPU))
                label["cpu_brand"] = b
                clauses.append(_clean(_pick(rng, CPU[b], mode)))
            if rng.random() < 0.7 or label["cpu_brand"] == "any":
                b = rng.choice(list(GPU))
                label["gpu_brand"] = b
                clauses.append(_clean(_pick(rng, GPU[b], mode)))
    if rng.random() < 0.35:
        g = rng.choice([8, 16, 32, 64])
        label["ram_gb"] = g
        clauses.append(_clean(_pick(rng, RAM, mode)).format(n=g))
    if rng.random() < 0.35:
        g = rng.choice([500, 1000, 2000])
        label["storage_gb"] = g
        clauses.append(_clean(_pick(rng, SSD[g], mode)))
    if rng.random() < 0.22:
        ex = [e for e in EXTRA_WORDS if rng.random() < 0.5] or [rng.choice(list(EXTRA_WORDS))]
        words = [_clean(_pick(rng, EXTRA_WORDS[e], mode)) for e in ex]
        label["extras"] = [e for e in EXTRA_WORDS if e in ex]
        x = ", ".join(words[:-1]) + (" and " if len(words) > 1 else "") + words[-1]
        clauses.append(_clean(rng.choice(EXTRA_JOIN[lang])).format(x=x))
    if rng.random() < 0.08:
        label["compact"] = True
        clauses.append(_clean(_pick(rng, COMPACT, mode)))

    rng.shuffle(clauses)
    tail = ""
    for i, c in enumerate(clauses):
        tail += ("" if i == 0 else rng.choice([", ", ". ", " and ", "; "])) + c
    text = opener + " " + tail.strip() + rng.choice([".", "", "!", " thanks."])
    if rng.random() < 0.12:
        text = text.lower()
    return text, label


def build_dataset(n, seed=0, mode="train"):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        t, l = make_sample(rng, mode)
        if t in seen:
            continue
        seen.add(t)
        out.append((t, l))
    return out


# ------------------------------------------------------- JSON / prompt format
def to_json(label):
    """Canonical compact JSON with fixed key order."""
    return json.dumps({k: label[k] for k in KEYS}, separators=(",", ":"))


def normalise(d):
    """Make a parsed dict comparable to the gold label (fill + coerce)."""
    out = dict(use_case="general", budget=None, currency="MYR", cpu_brand="any",
               gpu_brand="any", ram_gb=None, storage_gb=None, extras=[],
               compact=False)
    if not isinstance(d, dict):
        return out
    for k in KEYS:
        if k in d:
            out[k] = d[k]
    try:
        out["budget"] = int(float(out["budget"])) if out["budget"] is not None else None
    except (TypeError, ValueError, OverflowError):
        out["budget"] = None
    for k in ("ram_gb", "storage_gb"):
        try:
            out[k] = int(float(out[k])) if out[k] is not None else None
        except (TypeError, ValueError, OverflowError):
            out[k] = None
    if not isinstance(out["extras"], list):
        out["extras"] = []
    out["extras"] = sorted(set(str(e) for e in out["extras"]) & {"monitor", "keyboard", "mouse"})
    for key, choices, default in [
        ("use_case", set(USE), "general"), ("currency", {"MYR", "USD"}, "MYR"),
        ("cpu_brand", {"any", "intel", "amd"}, "any"),
        ("gpu_brand", {"any", "nvidia", "amd", "intel"}, "any")]:
        val = out[key]
        out[key] = val if isinstance(val, str) and val in choices else default
    for key in ("budget", "ram_gb", "storage_gb"):
        if out[key] is not None and out[key] <= 0:
            out[key] = None
    out["compact"] = out["compact"] is True or str(out["compact"]).lower() == "true"
    return out


def parse_model_output(text):
    """Extract the first JSON object from generated text. None if invalid."""
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


EXTRACT_HEAD = ("Extract the PC requirements from the customer message as JSON.\n"
                "Keys: use_case, budget, currency, cpu_brand, gpu_brand, ram_gb, "
                "storage_gb, extras, compact.\n"
                "IMPORTANT BUDGET RULES: Copy a budget only when the message explicitly states one. "
                "Interpret shorthand such as '5k', 'RM5k', or 'around 5k' as 5000. "
                "If no budget is explicitly stated, return budget: null. "
                "Never invent a budget or copy a numeric value from the examples.\n")


def extract_prompt(text):
    return f"{EXTRACT_HEAD}Message: {text}\nJSON:"

# The bundled LoRA adapter was trained on this shorter instruction in the
# previously supplied OPT-125M notebook. Preserve it for true adapter inference.
EXTRACT_HEAD_TRAINING = ("Extract the PC requirements from the customer message as JSON.\n"
                         "Keys: use_case, budget, currency, cpu_brand, gpu_brand, ram_gb, "
                         "storage_gb, extras, compact.\n")


def extract_prompt_training(text):
    return f"{EXTRACT_HEAD_TRAINING}Message: {text}\nJSON:"


def few_shot_prompt(text, shots):
    body = "".join(f"Message: {t}\nJSON: {to_json(l)}\n\n" for t, l in shots)
    return f"{EXTRACT_HEAD}\n{body}Message: {text}\nJSON:"


def explain_prompt(facts):
    return f"Recommended PC build:\n{facts}\nExplain why this build suits the customer:"


# ----------------------------------------------------------- rule baseline
_UC_RULES = [
    ("ai_ml", r"machine learning|deep learning|\bai\b|\bllm|neural|pytorch|data science|latih model"),
    ("streaming", r"stream|twitch|live ?stream"),
    ("video_editing", r"edit(ing)? video|video edit|render|premiere|davinci|content creation|after effects"),
    ("programming", r"coding|\bcode\b|programming|\bprogram\b|develop|docker"),
    ("gaming", r"gaming|\bgame|esports|valorant|cs2|dota|fps|cyberpunk"),
    ("office", r"office|excel|word|assignment|spreadsheet|pejabat|study"),
    ("general", r"daily|general|netflix|everyday|harian|biasa|browsing|surf|basic|rumah"),
]


def extract_budget_amount(text, default_currency="MYR"):
    """Extract a user-stated PC budget without mistaking hardware specs for it.

    Handles common English/Manglish forms: RM5500, RM 5.5k, around 5k,
    budget of 5,000, 5k budget, 5k ringgit, USD 1.5k, under RM8,000,
    and "PC build for 5k". Returns (integer_amount, currency) or None.
    """
    text = str(text or "").lower().replace("’", "'")
    number = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    unit = r"(?P<unit>k|thousand|ribu)?"
    amount = rf"(?P<number>{number})\s*{unit}(?![a-z])"

    patterns = [
        # Currency prefix: RM5500, RM 5.5k, MYR 5000, USD 1.5k, $1200.
        rf"\b(?P<currency>rm|myr|usd)\s*(?P<number>{number})\s*(?P<unit>k|thousand|ribu)?(?![a-z])",
        rf"\$\s*(?P<number>{number})\s*(?P<unit>k|thousand|ribu)?(?![a-z])",
        # Currency suffix: 5k ringgit, 5500 MYR, 1.5k USD.
        rf"(?P<number>{number})\s*(?P<unit>k|thousand|ribu)?\s*(?P<currency>ringgit|myr|usd|dollars?)\b",
        # Context phrase followed by an amount.
        rf"\b(?:budget|bajet|around|about|abt|roughly|approximately|approx|under|below|maximum|max|up to|not more than|not exceeding|spend|spending|within|price limit|spending limit)\b"
        rf"(?:\s+(?:is|of|about|around|at|only|up to))?\s*(?P<currency>rm|myr|usd)?\s*{amount}",
        # Amount followed by budget/limit wording: 5k budget / 5500 maximum.
        rf"{amount}\s*(?:myr|rm)?\s*(?:budget|bajet|maximum|max|limit)\b",
        # PC build wording: "build a PC for 5k" / "PC for around 5k".
        rf"\b(?:pc|computer|build|setup)\b.{{0,35}}?\bfor\s+(?P<currency>rm|myr|usd)?\s*(?P<number>{number})\s*(?P<unit>k|thousand|ribu)?(?![a-z])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        groups = match.groupdict()
        try:
            value = float(groups["number"].replace(",", ""))
        except (TypeError, ValueError):
            continue
        suffix = (groups.get("unit") or "").lower()
        if suffix in {"k", "thousand", "ribu"}:
            value *= 1000
        amount_value = int(round(value))
        if amount_value <= 0:
            return None

        currency = (groups.get("currency") or default_currency).upper()
        if currency == "RM" or currency == "RINGGIT":
            currency = "MYR"
        elif currency in {"DOLLAR", "DOLLARS", "$"}:
            currency = "USD"
        if currency not in {"MYR", "USD"}:
            currency = default_currency
        return amount_value, currency
    return None


def has_budget_mention(text):
    """True only when the message contains a budget-like cue plus an amount.

    Used as a cautious fallback for a fine-tuned extractor if a user's amount
    wording is unusual and the deterministic parser cannot normalize it.
    """
    text = str(text or "").lower()
    amount = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:k|thousand|ribu)?"
    patterns = [
        rf"\b(?:budget|bajet|around|about|abt|roughly|approximately|approx|under|below|maximum|max|up to|not more than|not exceeding|spend|spending|within|price limit|spending limit)\b.{{0,35}}?{amount}",
        rf"\b(?:rm|myr|usd)\s*{amount}",
        rf"\$\s*{amount}",
        rf"{amount}\s*(?:ringgit|myr|usd|dollars?)\b",
        rf"{amount}\s*(?:budget|bajet|maximum|max|limit)\b",
    ]
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def extract_memory_storage(text):
    """Extract explicit system RAM and storage capacities in GB.

    RAM and storage are parsed separately so a GPU's VRAM or the neighbouring
    SSD capacity cannot be mistaken for system RAM. A kit such as 2 x 16GB is
    returned as 32GB total capacity. TB values are normalized using 1TB=1000GB,
    matching this project's catalogue representation.
    """
    t = str(text or "").lower()
    ram = None

    # 1) A multi-module kit describes total system memory (2 x 16GB = 32GB).
    kit_patterns = [
        r"\b([1-4])\s*[x×]\s*(\d{1,3})\s*(?:gb|g|gigabytes?)\s*(?:of\s+)?(?:ddr[45](?:[- ]?\d{4,5})?\s*)?(?:ram|memory|memori)?\b",
        r"\b(?:ram|memory|memori)\s*[:=]?\s*([1-4])\s*[x×]\s*(\d{1,3})\s*(?:gb|g)\b",
    ]
    for pattern in kit_patterns:
        match = re.search(pattern, t, flags=re.I)
        if match:
            ram = int(match.group(1)) * int(match.group(2))
            break

    # 2) Explicit label connectors have highest priority. This also handles
    #    "SSD 512GB, RAM: 32GB" without misreading the SSD amount as RAM.
    if ram is None:
        ram_after_labeled = re.search(
            r"\b(?:system\s+memory|ram|memory|memori|内存)\b"
            r"\s*(?:is|should\s+be|must\s+be|at\s+least|minimum|min|capacity|size)\s*"
            r"[:=]?\s*(\d{1,3})\s*(?:gb|g|gigs?|gigabytes?)\b|"
            r"\b(?:system\s+memory|ram|memory|memori|内存)\s*[:=]\s*"
            r"(\d{1,3})\s*(?:gb|g|gigs?|gigabytes?)\b", t, flags=re.I
        )
        if ram_after_labeled:
            value = next((g for g in ram_after_labeled.groups() if g is not None), None)
            ram = int(value)

    # 3) Value followed by RAM/memory is preferred over a bare "RAM 256GB"
    #    match, avoiding capture of a later SSD value.
    if ram is None:
        ram_before = re.search(
            r"\b(\d{1,3})\s*(gb|g|gigs?|gigabytes?)\s*(?:of\s+)?"
            r"(?:ddr[45](?:[- ]?\d{4,5})?\s*)?"
            r"(?:system\s+memory|ram|memory|memori|内存)\b", t, flags=re.I
        )
        if ram_before:
            ram = int(ram_before.group(1))

    # 4) Direct label-before-value ("RAM 16GB") is the last fallback.
    if ram is None:
        ram_after = re.search(
            r"\b(?:system\s+memory|ram|memory|memori|内存)\b"
            r"\s*(\d{1,3})\s*(?:gb|g|gigs?|gigabytes?)\b", t, flags=re.I
        )
        if ram_after:
            ram = int(ram_after.group(1))

    # 4) DDR5 6000 32GB (memory amount follows the DDR generation/speed).
    if ram is None:
        ddr_first = re.search(
            r"\bddr[45](?:[- ]?\d{4,5})?\s+(\d{1,3})\s*(?:gb|g)\b", t, flags=re.I
        )
        if ddr_first:
            ram = int(ddr_first.group(1))

    storage = None
    # Prefer an amount after the named storage component. This avoids the
    # common false match in "RAM: 32GB, SSD: 512GB".
    amount = r"(\d+(?:\.\d+)?)\s*(tb|terabytes?|gb|gigabytes?)"
    storage_after = re.search(
        r"\b(?:ssd|nvme|storage|solid\s*state\s*drive|hard\s*drive|hard\s*disk|固态硬盘)\b"
        r"\s*(?:(?:capacity|size|of|is|should\s+be|must\s+be|at\s+least|minimum|min)\s*)?"
        r"[:=]?\s*" + amount + r"\b", t, flags=re.I
    )
    if storage_after:
        value, unit = float(storage_after.group(1)), storage_after.group(2).lower()
    else:
        # Or an amount followed by an explicit storage label.
        storage_before = re.search(
            r"\b" + amount + r"\s*(?:of\s+)?"
            r"(?:ssd|nvme|storage|solid\s*state\s*drive|hard\s*drive|hard\s*disk|固态硬盘)\b",
            t, flags=re.I
        )
        if storage_before:
            value, unit = float(storage_before.group(1)), storage_before.group(2).lower()
        else:
            value = unit = None

    if unit is not None:
        storage = int(round(value * (1000 if unit.startswith("tb") or unit.startswith("terabyte") else 1)))

    # Common word-number variants used in natural requests (e.g. "one TB SSD").
    if storage is None:
        word_units = {
            "one": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3,
            "half": 0.5, "half a": 0.5,
        }
        word_storage = re.search(
            r"\b(one|two|three|half(?:\s+a)?)\s*(tb|terabytes?)\s*(?:ssd|nvme|storage)\b",
            t, flags=re.I
        )
        if word_storage:
            storage = int(word_units[word_storage.group(1).lower()] * 1000)

    return ram, storage


def rule_extract(text):
    """Keyword / regex baseline (no learning)."""
    t = text.lower()
    out = dict(use_case="general", budget=None, currency="MYR", cpu_brand="any",
               gpu_brand="any", ram_gb=None, storage_gb=None, extras=[], compact=False)
    for uc, pat in _UC_RULES:
        if re.search(pat, t):
            out["use_case"] = uc
            break
    budget_info = extract_budget_amount(t)
    if budget_info is not None:
        out["budget"], out["currency"] = budget_info
    if re.search(r"all amd|semua amd|full amd|amd everything", t):
        out["cpu_brand"] = out["gpu_brand"] = "amd"
    elif re.search(r"intel and nvidia|intel cpu with nvidia|intel \+ rtx", t):
        out["cpu_brand"], out["gpu_brand"] = "intel", "nvidia"
    else:
        if re.search(r"ryzen|amd cpu|guna amd", t):
            out["cpu_brand"] = "amd"
        elif re.search(r"intel (cpu|core|chip)|guna intel", t):
            out["cpu_brand"] = "intel"
        if re.search(r"nvidia|rtx|geforce", t):
            out["gpu_brand"] = "nvidia"
        elif re.search(r"radeon|amd graphics|amd gpu", t):
            out["gpu_brand"] = "amd"
        elif re.search(r"arc", t):
            out["gpu_brand"] = "intel"
    out["ram_gb"], out["storage_gb"] = extract_memory_storage(t)
    for e, pat in (("monitor", r"monitor|display"), ("keyboard", r"keyboard|papan kekunci"),
                   ("mouse", r"mouse|tetikus")):
        if re.search(pat, t):
            out["extras"].append(e)
    out["compact"] = bool(re.search(r"small form|compact|mini itx|pc kecil|small casing|as small", t))
    return out


# ------------------------------------------------------------------ metrics
def evaluate(preds, golds):
    """preds: list of dict|None (None = invalid JSON). golds: label dicts."""
    n = len(golds)
    if n == 0 or len(preds) != n:
        raise ValueError("Predictions and gold labels must have equal nonzero length")
    valid = sum(p is not None for p in preds)
    field_ok = {k: 0 for k in KEYS}
    exact = 0
    for p, g in zip(preds, golds):
        pn, gn = normalise(p), normalise(g)
        ok_all = p is not None
        for k in KEYS:
            ok = (pn[k] == gn[k]) and p is not None
            field_ok[k] += ok
            ok_all &= ok
        exact += ok_all
    res = {"json_valid_%": 100 * valid / n, "exact_match_%": 100 * exact / n}
    for k in KEYS:
        res[f"{k}_%"] = 100 * field_ok[k] / n
    res["mean_field_acc_%"] = sum(100 * field_ok[k] / n for k in KEYS) / len(KEYS)
    return res


# ------------------------------------------------------------------ OOD set
def _L(uc, budget, cur="MYR", cpu="any", gpu="any", ram=None, ssd=None,
       extras=(), compact=False):
    return dict(use_case=uc, budget=budget, currency=cur, cpu_brand=cpu,
                gpu_brand=gpu, ram_gb=ram, storage_gb=ssd,
                extras=sorted(extras), compact=compact)


# Hand-written, natural-sounding requests that do NOT come from the generator
# templates. Used to measure out-of-distribution robustness.
OOD_SET = [
    ("my son wants a pc to play fortnite and minecraft, we can spend about rm3500", _L("gaming", 3500)),
    ("Looking for a workstation for Blender and Unreal Engine, 12k ringgit max, nvidia please", _L("video_editing", 12000, gpu="nvidia")),
    ("recommend me a build for my office staff, excel and chrome only, rm 2000", _L("office", 2000)),
    ("I'd like to train small neural nets at home, budget 1800 usd, prefer nvidia", _L("ai_ml", 1800, "USD", gpu="nvidia")),
    ("boss, rm5000 got any good gaming pc? intel cpu and 32gb ram", _L("gaming", 5000, cpu="intel", ram=32)),
    ("pc untuk edit video tiktok, bajet 4 ribu", _L("video_editing", 4000)),
    ("nak buat live stream dan main game, budget rm 6,500, nak monitor sekali", _L("streaming", 6500, extras=["monitor"])),
    ("Can I get a decent PC for web dev around $1,200? AMD CPU please", _L("programming", 1200, "USD", cpu="amd")),
    ("budget RM 3k, just for browsing youtube and zoom meetings", _L("general", 3000)),
    ("Need something small, mini itx, for gaming. rm4500.", _L("gaming", 4500, compact=True)),
    ("pls recommend pc 2tb ssd 64gb ram for data science, rm 14000", _L("ai_ml", 14000, ram=64, ssd=2000)),
    ("I have 8k to spend. want to edit 4k videos with keyboard and mouse included.", _L("video_editing", 8000, extras=["keyboard", "mouse"])),
    ("sis nak PC untuk kerja akaun, tak payah mahal, rm1800", _L("office", 1800)),
    ("gaming pc rm 2.5k, amd gpu", _L("gaming", 2500, gpu="amd")),
    ("Radeon only and ryzen, for playing Call of Duty, RM 6000", _L("gaming", 6000, cpu="amd", gpu="amd")),
    ("Trying to spec a PC to run Llama locally, 3000 USD, 64GB of RAM", _L("ai_ml", 3000, "USD", ram=64)),
    ("student pc for assignments and Zoom classes, RM 2200", _L("office", 2200)),
    ("wah my budget only 1.5k ah, can build gaming pc or not", _L("gaming", 1500)),
    ("twitch streamer here, need a build, ten thousand ringgit, 1tb ssd", _L("streaming", 10000, ssd=1000)),
    ("Recommend a PC for coding and docker containers, 16GB RAM enough, RM4000", _L("programming", 4000, ram=16)),
    ("build for editing youtube vids, budget about 5000 myr, nvidia gpu", _L("video_editing", 5000, gpu="nvidia")),
    ("cheap pc for my mom to watch netflix and facebook, below rm 1800", _L("general", 1800)),
    ("I want a high-end gaming rig, money is no object, about RM 20000", _L("gaming", 20000)),
    ("Budget: 4,500 ringgit. Use: coding. Intel CPU + RTX 4060", _L("programming", 4500, cpu="intel", gpu="nvidia")),
    ("pc game sekitar rm4k, ram 16gb, ssd 1tb", _L("gaming", 4000, ram=16, ssd=1000)),
    ("need pc for rendering in 3ds max and v-ray, budget USD 2200", _L("video_editing", 2200, "USD")),
    ("just a basic desktop for the shop counter, rm1600", _L("office", 1600)),
    ("I stream on Twitch daily, 9000 MYR, with monitor keyboard and mouse", _L("streaming", 9000, extras=["monitor", "keyboard", "mouse"])),
    ("Intel Arc GPU build for esports, 3000 ringgit", _L("gaming", 3000, gpu="intel")),
    ("Fine-tune LLMs at home, RM 15000, Nvidia only", _L("ai_ml", 15000, gpu="nvidia")),
    ("Can you suggest a quiet PC for general use? budget $900", _L("general", 900, "USD")),
    ("mak nak PC untuk tengok drama dan video call, bajet rm2k", _L("general", 2000)),
    ("program in C++ and run virtual machines, 32GB RAM, rm 5500", _L("programming", 5500, ram=32)),
    ("1TB SSD for office use, rm2300", _L("office", 2300, ssd=1000)),
    ("Gaming PC under 3k with Nvidia", _L("gaming", 3000, gpu="nvidia")),
    ("I'm into esports (Valorant), need a small PC, compact, budget RM 3500, ryzen", _L("gaming", 3500, cpu="amd", compact=True)),
    ("Computer for gaming and I also need a monitor, RM 6500", _L("gaming", 6500, extras=["monitor"])),
]
