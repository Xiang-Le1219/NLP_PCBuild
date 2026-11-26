"""
catalog_nlu.py
--------------
Dataset-aware extraction of explicit PC component preferences.

Unmentioned fields are deliberately omitted from the preference dictionary;
the PC engine then selects its normal default according to use case and
catalogue constraints. Every raw catalogue field is exposed in FIELD_SCHEMA
for transparency and review.
"""
from __future__ import annotations

import re
import difflib
from collections import OrderedDict

import pandas as pd

# component key -> source category in pc_engine.load_data()
FIELD_SCHEMA = OrderedDict({
    "cpu": [
        ("name", "CPU model/name", "name", "text"),
        ("price", "CPU maximum price (USD dataset price)", "price", "number"),
        ("core_count", "CPU core count", "cores", "number"),
        ("core_clock", "CPU base clock (GHz)", "core_clock", "number"),
        ("boost_clock", "CPU boost clock (GHz)", "boost_clock", "number"),
        ("microarchitecture", "CPU microarchitecture", "microarchitecture", "choice"),
        ("tdp", "CPU TDP (W)", "tdp", "number"),
        ("graphics", "Integrated graphics model", "graphics", "choice"),
        ("socket", "CPU socket (derived)", "socket", "choice"),
        ("brand", "CPU brand (derived)", "brand", "choice"),
        ("has_igpu", "Integrated graphics available (derived)", "has_igpu", "bool"),
    ],
    "gpu": [
        ("name", "GPU model/name", "name", "text"),
        ("price", "GPU maximum price (USD dataset price)", "price", "number"),
        ("chipset", "GPU chipset", "chipset", "choice"),
        ("memory", "GPU VRAM (GB)", "memory", "number"),
        ("core_clock", "GPU core clock (MHz)", "core_clock", "number"),
        ("boost_clock", "GPU boost clock (MHz)", "boost_clock", "number"),
        ("color", "GPU colour", "color", "choice"),
        ("length", "GPU length (mm)", "length", "number"),
        ("brand", "GPU brand (derived)", "brand", "choice"),
    ],
    "mobo": [
        ("name", "Motherboard model/name", "name", "text"),
        ("price", "Motherboard maximum price (USD dataset price)", "price", "number"),
        ("socket", "Motherboard CPU socket", "socket", "choice"),
        ("form_factor", "Motherboard form factor", "form_factor", "choice"),
        ("max_memory", "Motherboard maximum memory (GB)", "max_memory", "number"),
        ("memory_slots", "Motherboard memory slots", "memory_slots", "number"),
        ("color", "Motherboard colour", "color", "choice"),
        ("chipset_code", "Motherboard chipset (derived)", "chipset_code", "choice"),
        ("ddr", "Supported DDR generation (derived)", "ddr", "number"),
    ],
    "ram": [
        ("name", "RAM model/name", "name", "text"),
        ("price", "RAM maximum price (USD dataset price)", "price", "number"),
        ("speed", "RAM speed field", "speed", "text"),
        ("modules", "RAM module configuration", "modules", "text"),
        ("price_per_gb", "RAM price per GB", "price_per_gb", "number"),
        ("color", "RAM colour", "color", "choice"),
        ("first_word_latency", "RAM first-word latency", "first_word_latency", "number"),
        ("cas_latency", "RAM CAS latency", "cas_latency", "number"),
        ("ddr", "RAM DDR generation (derived)", "ddr", "number"),
        ("mhz", "RAM speed (MT/s, derived)", "mhz", "number"),
        ("n_mod", "RAM number of modules (derived)", "n_mod", "number"),
        ("gb_each", "RAM GB per module (derived)", "gb_each", "number"),
        ("total_gb", "RAM total capacity (derived)", "total_gb", "number"),
    ],
    "ssd": [
        ("name", "Storage model/name", "name", "text"),
        ("price", "Storage maximum price (USD dataset price)", "price", "number"),
        ("capacity", "Storage capacity (GB)", "capacity", "number"),
        ("price_per_gb", "Storage price per GB", "price_per_gb", "number"),
        ("type", "Storage type", "type", "choice"),
        ("cache", "Storage cache (MB)", "cache", "number"),
        ("form_factor", "Storage form factor", "form_factor", "choice"),
        ("interface", "Storage interface", "interface", "choice"),
    ],
    "psu": [
        ("name", "PSU model/name", "name", "text"),
        ("price", "PSU maximum price (USD dataset price)", "price", "number"),
        ("type", "PSU form/type", "type", "choice"),
        ("efficiency", "PSU efficiency rating", "efficiency", "choice"),
        ("wattage", "PSU wattage (W)", "wattage", "number"),
        ("modular", "PSU modularity", "modular", "choice"),
        ("color", "PSU colour", "color", "choice"),
        ("watt", "PSU wattage (derived)", "watt", "number"),
        ("eff", "PSU efficiency rank (derived)", "eff", "number"),
    ],
    "case": [
        ("name", "PC case model/name", "name", "text"),
        ("price", "Case maximum price (USD dataset price)", "price", "number"),
        ("type", "Case type/form factor", "type", "choice"),
        ("color", "Case colour", "color", "choice"),
        ("psu", "Included PSU wattage", "psu", "number"),
        ("side_panel", "Case side panel", "side_panel", "choice"),
        ("external_volume", "Case external volume", "external_volume", "number"),
        ("internal_35_bays", "Internal 3.5-inch bays", "internal_35_bays", "number"),
    ],
    "cooler": [
        ("name", "CPU cooler model/name", "name", "text"),
        ("price", "Cooler maximum price (USD dataset price)", "price", "number"),
        ("rpm", "Cooler fan RPM source range", "rpm", "text"),
        ("rpm_max", "Cooler maximum fan RPM (derived)", "rpm_max", "number"),
        ("noise_level", "Cooler noise source range", "noise_level", "text"),
        ("noise_max", "Cooler maximum noise level (derived)", "noise_max", "number"),
        ("color", "Cooler colour", "color", "choice"),
        ("size", "Cooler radiator size (mm)", "size", "number"),
    ],
    "monitor": [
        ("name", "Monitor model/name", "name", "text"),
        ("price", "Monitor maximum price (USD dataset price)", "price", "number"),
        ("screen_size", "Monitor screen size (inches)", "screen_size", "number"),
        ("resolution", "Monitor resolution", "resolution", "choice"),
        ("refresh_rate", "Monitor refresh rate (Hz)", "refresh_rate", "number"),
        ("response_time", "Monitor response time (ms)", "response_time", "number"),
        ("panel_type", "Monitor panel type", "panel_type", "choice"),
        ("aspect_ratio", "Monitor aspect ratio", "aspect_ratio", "choice"),
    ],
    "keyboard": [
        ("name", "Keyboard model/name", "name", "text"),
        ("price", "Keyboard maximum price (USD dataset price)", "price", "number"),
        ("style", "Keyboard style", "style", "choice"),
        ("switches", "Keyboard switch type", "switches", "choice"),
        ("backlit", "Keyboard backlight", "backlit", "choice"),
        ("tenkeyless", "Tenkeyless (TKL)", "tenkeyless", "bool"),
        ("connection_type", "Keyboard connection", "connection_type", "choice"),
        ("color", "Keyboard colour", "color", "choice"),
    ],
    "mouse": [
        ("name", "Mouse model/name", "name", "text"),
        ("price", "Mouse maximum price (USD dataset price)", "price", "number"),
        ("tracking_method", "Mouse tracking method", "tracking_method", "choice"),
        ("connection_type", "Mouse connection", "connection_type", "choice"),
        ("max_dpi", "Mouse maximum DPI", "max_dpi", "number"),
        ("hand_orientation", "Mouse hand orientation", "hand_orientation", "choice"),
        ("color", "Mouse colour", "color", "choice"),
    ],
})

COMPONENT_ALIASES = {
    "cpu": ["cpu", "processor", "ryzen", "intel core", "core ultra"],
    "gpu": ["gpu", "graphics card", "graphic card", "video card", "rtx", "gtx", "radeon", "geforce", "nvidia", "arc"],
    "mobo": ["motherboard", "mainboard", "mobo"],
    "ram": ["ram", "memory kit", "system memory", "ddr4", "ddr5"],
    "ssd": ["ssd", "storage", "nvme", "hard drive", "solid state drive"],
    "psu": ["psu", "power supply", "power-supply"],
    "case": ["pc case", "computer case", "chassis", "casing", "case"],
    "cooler": ["cpu cooler", "cooler", "liquid cooler", "aio"],
    "monitor": ["monitor", "display", "screen"],
    "keyboard": ["keyboard", "key board"],
    "mouse": ["mouse", "gaming mouse"],
}

TYPO_MAP = {
    "wirless": "wireless", "wireles": "wireless", "wirelese": "wireless",
    "motherbord": "motherboard", "mothboard": "motherboard", "motheboard": "motherboard",
    "mainbaord": "mainboard", "keybord": "keyboard", "keybaord": "keyboard",
    "moniter": "monitor", "montior": "monitor", "procesor": "processor",
    "proccessor": "processor", "grahpics": "graphics", "grapics": "graphics",
    "nvdia": "nvidia", "nvida": "nvidia", "nivdia": "nvidia",
    "intell": "intel", "ryzen": "ryzen", "moues": "mouse", "mousse": "mouse",
    "memmory": "memory", "storag": "storage", "coolling": "cooling",
    "moduler": "modular", "tenkeyles": "tenkeyless", "keyless": "keyless",
    "gamming": "gaming", "straming": "streaming", "strreaming": "streaming",
    "moniter": "monitor", "sdd": "ssd", "nvmee": "nvme",
}

FUZZY_VOCAB = sorted(set(
    ["wireless", "motherboard", "mainboard", "keyboard", "monitor", "processor",
     "graphics", "nvidia", "intel", "ryzen", "gaming", "streaming", "cooler",
     "modular", "bluetooth", "tenkeyless", "backlit", "optical", "mechanical",
     "microatx", "miniitx", "atx", "tempered", "glass", "resolution",
     "refresh", "rate", "storage", "memory", "capacity", "wattage", "efficiency",
     "chipset", "socket", "form", "factor", "latency", "response", "speed",
     "cores", "core", "boost", "clock", "colour", "color", "white", "black",
     "silver", "red", "blue", "green", "purple", "pink", "gold", "platinum",
     "nvidia", "geforce", "radeon", "rtx", "geforce", "ssd", "nvme",
     "display", "screen", "mouse", "case", "chassis", "cooling", "liquid",
     "air", "power", "supply", "headphone", "headphones"]
))

def correct_typos(text: str):
    """Correct common PC vocabulary misspellings conservatively."""
    original = str(text or "")
    corrected = original
    for wrong, right in TYPO_MAP.items():
        corrected = re.sub(rf"\b{re.escape(wrong)}\b", right, corrected, flags=re.I)
    changes = []
    # Conservative token-level fuzzy correction only for known technical words.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+-]*", corrected)
    for token in set(tokens):
        low = token.lower()
        if len(low) < 5 or low in FUZZY_VOCAB or low.isupper():
            continue
        match = difflib.get_close_matches(low, FUZZY_VOCAB, n=1, cutoff=0.88)
        if match and match[0] != low:
            corrected = re.sub(rf"\b{re.escape(token)}\b", match[0], corrected, flags=re.I)
    if corrected != original:
        # Report token replacements for transparency.
        changes.append({"original": original, "corrected": corrected})
    return corrected, changes


def _component_positions(text):
    low = text.lower()
    positions = {}
    for comp, aliases in COMPONENT_ALIASES.items():
        found = []
        for alias in aliases:
            for m in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", low):
                found.append(m.start())
        if found:
            positions[comp] = found
    # Infer CPU/GPU from common brand/product references.
    if "cpu" not in positions:
        for pat in (r"\bryzen\b", r"\bintel\s+(?:core|ultra)\b"):
            m = re.search(pat, low)
            if m: positions.setdefault("cpu", []).append(m.start())
    if "gpu" not in positions:
        for pat in (r"\brtx\s*\d", r"\bgtx\s*\d", r"\bradeon\s+rx\b", r"\barc\s+[ab]\d"):
            m = re.search(pat, low)
            if m: positions.setdefault("gpu", []).append(m.start())
    return positions


def _nearest_component_for_span(text, comp_positions, start, end=None, max_distance=75):
    """Associate an attribute/value phrase with the most plausible component.

    Handles both ``white ATX case`` (value before component) and
    ``motherboard AM5`` (component before value). Punctuation creates a clause
    boundary, so ``black mouse, white case`` assigns each colour correctly.
    """
    end = start if end is None else end
    previous = []
    following = []

    for comp, anchors in comp_positions.items():
        aliases = COMPONENT_ALIASES.get(comp, [])
        for pos in anchors:
            matched_alias = next(
                (alias for alias in sorted(aliases, key=len, reverse=True)
                 if text[pos:pos + len(alias)].lower() == alias.lower()),
                None,
            )
            alias_end = pos + len(matched_alias or "")
            # Attribute comes immediately after a component: "motherboard AM5".
            if 0 <= start - alias_end <= 25:
                between = text[alias_end:start]
                words = re.findall(r"[a-z0-9]+", between.lower())
                if not re.search(r"[,;.!?]", between) and len(words) <= 3:
                    previous.append((start - alias_end, comp))
            # Attribute comes immediately before a component: "white ATX case".
            gap = pos - end
            if 0 <= gap <= 24:
                between = text[end:pos]
                words = re.findall(r"[a-z0-9]+", between.lower())
                if not re.search(r"[,;.!?]", between) and len(words) <= 3:
                    following.append((gap, comp))

    # Prefer a directly preceding component mention when the value follows it.
    if previous:
        previous.sort()
        return previous[0][1]
    if following:
        following.sort()
        return following[0][1]

    midpoint = (start + end) / 2
    best_comp, best_distance = None, float("inf")
    for comp, anchors in comp_positions.items():
        distance = min(abs(midpoint - pos) for pos in anchors)
        if distance < best_distance:
            best_comp, best_distance = comp, distance
    return best_comp if best_distance <= max_distance else None


def _near_component(text, comp_positions, component):
    return component in comp_positions

def _options_for(data, component, col):
    frame = data.get(component)
    if frame is None or col not in frame.columns:
        return []
    vals = frame[col].dropna().astype(str).unique().tolist()
    # Exclude values which are not meaningful for a selection control.
    vals = [v for v in vals if v.strip() and v.lower() not in {"nan", "none"}]
    if col == "name":
        return vals
    return sorted(vals, key=lambda x: (len(x), x.lower()))


def _contains_value(text, value):
    low = text.lower()
    v = str(value).strip().lower()
    if not v:
        return False
    # Numeric categorical strings must match as a token; punctuation is tolerated.
    if re.fullmatch(r"[a-z0-9.+:/ -]+", v):
        return re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", low) is not None
    return v in low


def _add_choice(prefs, component, col, value, contains=False):
    prefs.setdefault(component, {})[col] = {"contains": str(value)} if contains else {"eq": value}


def _number_rules(text, prefs, component):
    low = text.lower()
    patterns = {
        "cpu": [
            ("core_count", r"\b(\d{1,2})\s*[- ]?cores?\b|\b(?:cpu|processor)\s+(?:with\s*)?(\d{1,2})\s+cores?\b", "min"),
            ("core_clock", r"\b(\d(?:\.\d+)?)\s*ghz\s*(?:base|base clock|core clock)\b", "min"),
            ("boost_clock", r"\b(\d(?:\.\d+)?)\s*ghz\s*(?:boost|turbo)(?: clock)?\b|\b(?:boost|turbo)\s*(?:clock)?\s*(\d(?:\.\d+)?)\s*ghz\b", "min"),
            ("tdp", r"\b(\d{2,4})\s*(?:w|watts?)\s*(?:tdp|cpu)\b", "max"),
            ("tdp", r"\btdp\s*(?:of|:)?\s*(\d{2,4})\s*w\b", "max"),
            ("price", r"\b(?:cpu|processor)\s+(?:under|below|max|less than)\s*rm?\s*(\d+)\b", "max"),
        ],
        "gpu": [
            ("memory", r"\b(\d{1,3})\s*gb\s*(?:vram|video memory|graphics memory)\b", "min"),
            ("length", r"\b(?:gpu|graphics card|video card)\s*(?:length)?\s*(?:under|below|max)?\s*(\d{2,3})\s*mm\b|\b(\d{2,3})\s*mm\s*(?:gpu|graphics card|video card)\b", "max"),
            ("core_clock", r"\b(?:gpu\s*)?(?:core\s*)?clock\s*(\d{3,4})\s*mhz\b", "min"),
            ("boost_clock", r"\b(?:gpu\s*)?boost\s*(?:clock)?\s*(\d{3,4})\s*mhz\b", "min"),
            ("price", r"\b(?:gpu|graphics card|video card)\s+(?:under|below|max|less than)\s*rm?\s*(\d+)\b", "max"),
        ],
        "mobo": [
            ("max_memory", r"\b(?:motherboard|mobo)\s+(?:supports?|max(?:imum)?)\s*(\d{2,4})\s*gb\b|\b(\d{2,4})\s*gb\s*(?:max memory|maximum memory)\b", "min"),
            ("memory_slots", r"\b(\d)\s*(?:ram\s*)?slots?\b", "min"),
        ],
        "ram": [
            ("total_gb", r"\b(\d{1,3})\s*gb\s*(?:ram|memory)\b", "min"),
            ("total_gb", r"\b(\d{1,3})\s*gb\s*ddr[45]\s*(?:\d{4,5}\s*)?(?:ram|memory)\b", "min"),
            ("total_gb", r"\b(?:ram|memory)\s*(?:capacity\s*)?(\d{1,3})\s*gb\b", "min"),
            ("total_gb", r"\b([124])\s*[x×]\s*(\d{1,3})\s*gb\b", "min"),
            ("mhz", r"\bddr[45]\s*[- ]?(\d{4,5})\b|\b(\d{4,5})\s*mhz\b", "min"),
            ("n_mod", r"\b(\d)\s*(?:sticks?|modules?|dimms?)\b", "min"),
            ("gb_each", r"\b\d\s*[x×]\s*(\d{1,3})\s*gb\b", "min"),
            ("cas_latency", r"\bcl\s*(\d{1,2})\b", "max"),
        ],
        "ssd": [
            ("capacity", r"\b(\d+(?:\.\d+)?)\s*(tb|gb)\s*(?:ssd|nvme|storage)\b|\b(?:ssd|nvme|storage)\s*(\d+(?:\.\d+)?)\s*(tb|gb)\b", "min"),
            ("cache", r"\b(\d{2,6})\s*mb\s*cache\b", "min"),
        ],
        "psu": [
            ("wattage", r"\b(\d{3,4})\s*w(?:att)?\b.{0,24}\b(?:psu|power supply)\b|\b(?:psu|power supply)\s*(?:of)?\s*(\d{3,4})\s*w\b", "min"),
        ],
        "case": [
            ("psu", r"\b(?:case|casing)\s*(?:with|includes?)\s*(\d{3,4})\s*w\s*(?:psu|power supply)\b", "min"),
            ("internal_35_bays", r"\b(\d{1,2})\s*(?:internal\s*)?3\.5(?:-inch)?\s*bays?\b", "min"),
            ("external_volume", r"\b(?:case|casing)\s*(?:volume\s*)?(?:under|below|max)?\s*(\d+(?:\.\d+)?)\s*l\b|\b(\d+(?:\.\d+)?)\s*l\s*(?:case|casing)\b", "max"),
        ],
        "cooler": [
            ("size", r"\b(120|140|240|280|360|420)\s*mm\s*(?:aio|radiator|liquid cooler)\b|\b(?:aio|radiator|liquid cooler)\s*(?:of\s*)?(120|140|240|280|360|420)\s*mm\b", "eq"),
            ("rpm_max", r"\b(\d{3,4})\s*rpm\b", "max"),
            ("noise_max", r"\b(\d+(?:\.\d+)?)\s*db(?:a)?\b", "max"),
        ],
        "monitor": [
            ("screen_size", r"\b(\d{2}(?:\.\d)?)\s*(?:-inch|inch|inches|in)\b", "min"),
            ("refresh_rate", r"\b(\d{2,3})\s*hz\b", "min"),
            ("response_time", r"\b(\d+(?:\.\d+)?)\s*ms\b", "max"),
        ],
        "keyboard": [],
        "mouse": [
            ("max_dpi", r"\b(\d{3,6})\s*dpi\b", "min"),
        ],
    }
    for col, pattern, op in patterns.get(component, []):
        match = re.search(pattern, low, flags=re.I)
        if not match:
            continue
        nums = [x for x in match.groups() if x is not None]
        try:
            if col == "capacity":
                if len(nums) >= 2:
                    value = float(nums[0]) * (1000 if nums[1] == "tb" else 1)
                else:
                    value = float(nums[0]) * (1000 if nums[-1] == "tb" else 1)
            elif col == "mhz":
                value = float(nums[0])
            elif col == "total_gb" and len(nums) >= 2:
                value = float(nums[0]) * float(nums[1])
            elif col == "external_volume" and len(nums) >= 2:
                value = float(nums[0])
            else:
                value = float(nums[0])
        except (ValueError, IndexError):
            continue
        if col in {"total_gb", "gb_each"}:
            value = int(value)
        prefs.setdefault(component, {})[col] = {op: value}


def extract_component_preferences(text, data):
    """Extract explicit dataset-backed component preferences from natural text."""
    corrected, typos = correct_typos(text)
    low = corrected.lower()
    positions = _component_positions(low)
    prefs = {}
    matched_fields = []

    # Numeric/structured attributes.
    for comp in positions:
        _number_rules(low, prefs, comp)

    # Common requirement aliases and exact catalogue values.
    aliases = {
        ("cpu", "brand"): {"intel": ["intel"], "amd": ["amd", "ryzen"]},
        ("gpu", "brand"): {"nvidia": ["nvidia", "geforce", "rtx", "gtx"],
                           "amd": ["amd", "radeon", "rx "], "intel": ["intel arc", "arc"]},
        ("gpu", "chipset"): {},
        ("mobo", "form_factor"): {"ATX": ["atx motherboard", "atx board"],
                                   "Micro ATX": ["micro atx motherboard", "matx motherboard"],
                                   "Mini ITX": ["mini itx motherboard", "itx motherboard"]},
        ("mobo", "ddr"): {4: ["ddr4 motherboard"], 5: ["ddr5 motherboard"]},
        ("case", "type"): {"ATX": ["atx case", "atx casing"],
                           "MicroATX": ["micro atx case", "matx case"],
                           "Mini ITX": ["mini itx case", "itx case"]},
        ("ram", "ddr"): {4: ["ddr4"], 5: ["ddr5"]},
        ("ssd", "interface"): {"M.2 PCIe": ["nvme", "m.2 pcie", "pcie ssd"]},
        ("psu", "efficiency"): {"gold": ["80+ gold", "80 plus gold", "gold efficiency", "gold psu", "gold power supply"],
                                "bronze": ["80+ bronze", "80 plus bronze", "bronze efficiency", "bronze psu"],
                                "platinum": ["80+ platinum", "platinum efficiency", "platinum psu"],
                                "titanium": ["80+ titanium", "titanium psu"],
                                "silver": ["80+ silver", "silver psu"]},
        ("psu", "modular"): {"Full": ["fully modular", "full modular"],
                             "Semi": ["semi modular"], "false": ["non-modular", "non modular"]},
        ("case", "side_panel"): {"Tempered Glass": ["tempered glass"],
                                 "Mesh": ["mesh panel", "mesh side"]},
        ("keyboard", "backlit"): {"RGB": ["rgb backlit", "rgb lighting", "rgb keyboard"],
                                  "White": ["white backlight"], "Red": ["red backlight"]},
        ("keyboard", "tenkeyless"): {"True": ["tenkeyless", "tkl"], "False": ["full size keyboard"]},
        ("keyboard", "connection_type"): {"Wireless": ["wireless keyboard", "bluetooth keyboard"],
                                           "Wired": ["wired keyboard"]},
        ("mouse", "connection_type"): {"Wireless": ["wireless mouse", "bluetooth mouse"],
                                        "Wired": ["wired mouse"]},
        ("mouse", "tracking_method"): {"Optical": ["optical mouse"], "Laser": ["laser mouse"]},
        ("mouse", "hand_orientation"): {"Left": ["left handed mouse", "left-hand mouse"],
                                         "Right": ["right handed mouse", "right-hand mouse"]},
    }

    for (comp, col), mapping in aliases.items():
        if comp not in positions:
            continue
        for value, phrases in mapping.items():
            hit = next((phrase for phrase in phrases if phrase in low), None)
            if hit:
                use_contains = ((comp, col) in {("case", "type"), ("psu", "modular"),
                                                ("psu", "efficiency"), ("ssd", "interface"),
                                                ("keyboard", "connection_type"),
                                                ("mouse", "connection_type")})
                prefs.setdefault(comp, {})[col] = (
                    {"contains": value} if use_contains else {"eq": value}
                )

    # Explicit CPU model names such as Ryzen 7 7800X3D or Core i7-14700K.
    if "cpu" in positions:
        m = re.search(r"\b(?:ryzen\s+[3579]\s+[\w-]+|core\s+(?:ultra\s+)?[3579]\s*[- ]?\d{4,5}[a-z]{0,2})\b", low, re.I)
        if m:
            prefs.setdefault("cpu", {})["name"] = {"contains": m.group(0)}
    # Explicit GPU model/chipset names such as RTX 5070 or RX 7800 XT.
    if "gpu" in positions:
        m = re.search(r"\b((?:RTX|GTX)\s*\d{3,4}(?:\s*(?:TI|SUPER))?|RX\s*\d{3,4}(?:\s*(?:XT|GRE))?)\b", low, re.I)
        if m:
            prefs.setdefault("gpu", {})["chipset"] = {"contains": m.group(1)}
    # Explicit motherboard chipset such as B650 or Z790.
    if "mobo" in positions:
        m = re.search(r"\b(A520|B450|B550|X570|A620|B650|X670|B840|B850|X870|H410|B460|H470|Z490|H510|B560|H570|Z590|H610|B660|H670|Z690|B760|H770|Z790|H810|B860|Z890)\b", low, re.I)
        if m:
            prefs.setdefault("mobo", {})["chipset_code"] = {"eq": m.group(1).upper()}
    if re.search(r"\batx\s+(?:pc\s+)?case\b|\batx\s+casing\b", low) and not re.search(r"\bmicro\s*atx\b", low):
        prefs.setdefault("case", {})["type"] = {
            "in": ["ATX Mid Tower", "ATX Full Tower", "ATX Mini Tower"]
        }
    # Resolve an exact GPU tier where the user did not specify a suffix such as Ti/SUPER.
    if "gpu" in positions:
        gpu_match = re.search(r"\b((?:RTX|GTX)\s*\d{3,4})(?:\s*(TI|SUPER))?\b", low, re.I)
        if gpu_match:
            requested = " ".join(x for x in gpu_match.groups() if x).lower()
            candidates = data.get("gpu")
            if candidates is not None and "chipset" in candidates:
                chips = candidates["chipset"].dropna().astype(str).unique()
                matched = []
                for chip in chips:
                    found = re.search(r"\b(?:RTX|GTX)\s*\d{3,4}(?:\s*(?:TI|SUPER))?\b", chip, re.I)
                    if found and " ".join(found.group(0).lower().split()) == " ".join(requested.split()):
                        matched.append(chip)
                if len(matched) == 1:
                    prefs.setdefault("gpu", {})["chipset"] = {"eq": matched[0]}
    # Avoid treating "gold PSU" as a gold-coloured PSU when it describes efficiency.
    psu_gold_efficiency = bool(re.search(r"\b(?:80\+?\s*)?(?:gold|bronze|silver|platinum|titanium)\b.{0,24}\b(?:psu|power supply)\b", low))
    # Dataset vocabulary matching. Only match actual values and associate them
    # with components explicitly mentioned by the user.
    for comp in positions:
        frame = data.get(comp)
        if frame is None:
            continue
        for field, label, data_col, kind in FIELD_SCHEMA[comp]:
            if kind != "choice" or data_col not in frame.columns:
                continue
            if comp == "psu" and data_col == "color" and psu_gold_efficiency:
                continue
            if comp == "keyboard" and data_col == "color" and re.search(
                r"cherry\s*mx\s*red\s*switch", low
            ):
                continue
            current = prefs.get(comp, {}).get(data_col)
            if current:
                continue
            vals = _options_for(data, comp, data_col)
            # Avoid broad/high-cardinality model fields unless explicitly named.
            for val in sorted(vals, key=lambda x: len(x), reverse=True):
                if len(val) < 2 or val.lower() in {"true", "false"}:
                    continue
                parts = [p.strip() for p in re.split(r"[,/]", val) if p.strip()]
                candidates = parts if len(parts) > 1 else [val]
                hit_match = None
                for part in candidates:
                    if not part:
                        continue
                    pattern = rf"(?<![a-z0-9]){re.escape(part.lower())}(?![a-z0-9])"
                    for m in re.finditer(pattern, low):
                        if _nearest_component_for_span(low, positions, m.start(), m.end()) == comp:
                            hit_match = (part, m)
                            break
                    if hit_match:
                        break
                if hit_match:
                    part, m = hit_match
                    _add_choice(prefs, comp, data_col, part, contains=(len(parts) > 1))
                    break

    # Common human-friendly resolution names map to the dataset's stored format.
    if "monitor" in positions:
        resolution_map = {
            "1080p": "1920,1080", "full hd": "1920,1080", "fhd": "1920,1080",
            "1440p": "2560,1440", "2k": "2560,1440", "qhd": "2560,1440",
            "4k": "3840,2160", "uhd": "3840,2160",
        }
        for phrase, value in resolution_map.items():
            if re.search(rf"\b{re.escape(phrase)}\b", low):
                prefs.setdefault("monitor", {})["resolution"] = {"eq": value}
                break

    # Backlit mentioned without a colour uses any available backlit option.
    if "keyboard" in positions and re.search(r"\bbacklit\b", low) and "backlit" not in prefs.get("keyboard", {}):
        prefs.setdefault("keyboard", {})["backlit"] = {"not_empty": True}

    # Boolean requirements.
    if "cpu" in positions:
        if re.search(r"\b(?:integrated|onboard)\s+(?:graphics|gpu)\b", low):
            prefs.setdefault("cpu", {})["has_igpu"] = {"eq": True}
    # Name matching: only add exact model match when a sufficiently long name appears.
    for comp in positions:
        frame = data.get(comp)
        if frame is None or "name" not in frame:
            continue
        names = frame["name"].dropna().astype(str).tolist()
        match_name = next((n for n in sorted(names, key=len, reverse=True)
                           if len(n) >= 8 and n.lower() in low), None)
        if match_name:
            prefs.setdefault(comp, {})["name"] = {"contains": match_name}

    return prefs, corrected, typos


def requirement_field_report(data, prefs, core_requirements=None):
    """Return every dataset field as user-stated, auto/default, or unavailable.

    The requirement parser stores detailed catalogue filters in ``prefs``.
    A few high-level fields are also kept in the core NLU structure, so map
    them here to the corresponding dataset columns for accurate reporting.
    """
    core_requirements = core_requirements or {}
    core_map = {
        ("cpu", "brand"): ("cpu_brand", "eq"),
        ("gpu", "brand"): ("gpu_brand", "eq"),
        ("ram", "total_gb"): ("ram_gb", "min"),
        ("ssd", "capacity"): ("storage_gb", "min"),
    }
    rows = []
    for comp, fields in FIELD_SCHEMA.items():
        frame = data.get(comp)
        for field, label, col, kind in fields:
            available = bool(
                frame is not None and col in frame.columns
                and frame[col].notna().any()
            )
            condition = (prefs.get(comp) or {}).get(col)
            if condition is None and (comp, col) in core_map:
                core_key, operation = core_map[(comp, col)]
                core_value = core_requirements.get(core_key)
                if core_value not in (None, "", "any", 0):
                    condition = {operation: core_value}
            if condition is None:
                value, source = "Auto / system default", "Default"
            else:
                value, source = str(condition), "User-stated"
            if not available:
                value, source = "Not available after dataset cleaning", "Unavailable"
            rows.append({
                "Component": comp.upper(),
                "Dataset attribute": label,
                "Dataset column": col,
                "Detected requirement": value,
                "Source": source,
                "Available in cleaned data": available,
            })
    return pd.DataFrame(rows)
