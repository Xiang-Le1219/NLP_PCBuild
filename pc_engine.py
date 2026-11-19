"""
pc_engine.py
============
Data cleaning + knowledge-based PC build engine.

The CSV files only contain raw part listings. Several things that a real
build needs are NOT in the data, so this module derives them with explicit,
documented rules (these are the "knowledge base" of the system):

  * CPU socket          -> derived from `microarchitecture` (+ name)
  * Motherboard chipset -> parsed from the board name
  * Memory type (DDR)   -> parsed from `speed` ("5,6000" = DDR5 @ 6000 MT/s);
                           for boards: from the socket, or from the board name
  * GPU performance/TDP -> small curated table (GPU_TABLE, approximate)
  * Case <-> board fit  -> mapping of case type to supported board sizes
  * PSU wattage         -> (CPU TDP + GPU TDP + 100 W) * 1.25 headroom

All prices in the CSVs are USD; they are converted to MYR for display.
"""
import re
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

USD_TO_MYR = 4.3          # change if the exchange rate changes
SUPPORTED_SOCKETS = {"AM4", "AM5", "LGA1200", "LGA1700", "LGA1851"}

# --------------------------------------------------------------------------
# Curated GPU table: chipset -> (relative performance, board power in watts)
# Relative performance is approximate (RTX 4090 = 100), compiled from public
# review aggregates. It is an ASSUMPTION of the system, not part of the CSV.
# --------------------------------------------------------------------------
GPU_TABLE = {
    # NVIDIA RTX 50
    "GeForce RTX 5090": (130, 575), "GeForce RTX 5080": (85, 360),
    "GeForce RTX 5070 Ti": (78, 300), "GeForce RTX 5070": (58, 250),
    "GeForce RTX 5060 Ti": (45, 180), "GeForce RTX 5060": (38, 145),
    "GeForce RTX 5050": (28, 130),
    # NVIDIA RTX 40
    "GeForce RTX 4090": (100, 450), "GeForce RTX 4080 SUPER": (78, 320),
    "GeForce RTX 4080": (76, 320), "GeForce RTX 4070 Ti SUPER": (67, 285),
    "GeForce RTX 4070 Ti": (61, 285), "GeForce RTX 4070 SUPER": (58, 220),
    "GeForce RTX 4070": (50, 200), "GeForce RTX 4060 Ti": (40, 160),
    "GeForce RTX 4060": (33, 115),
    # NVIDIA RTX 30
    "GeForce RTX 3090 Ti": (67, 450), "GeForce RTX 3090": (62, 350),
    "GeForce RTX 3080 Ti": (61, 350), "GeForce RTX 3080 12GB LHR": (59, 350),
    "GeForce RTX 3080 10GB": (55, 320), "GeForce RTX 3080 10GB LHR": (55, 320),
    "GeForce RTX 3070 Ti": (46, 290), "GeForce RTX 3070": (43, 220),
    "GeForce RTX 3070 LHR": (43, 220), "GeForce RTX 3060 Ti": (38, 200),
    "GeForce RTX 3060 Ti LHR": (38, 200), "GeForce RTX 3060 12GB": (28, 170),
    "GeForce RTX 3060 8GB": (26, 170), "GeForce RTX 3050 8GB": (20, 130),
    "GeForce RTX 3050 6GB": (17, 70),
    # NVIDIA RTX 20 / GTX 16 / GTX 10
    "GeForce RTX 2080 Ti": (42, 250), "GeForce RTX 2080 SUPER": (37, 250),
    "GeForce RTX 2080": (35, 215), "GeForce RTX 2070 SUPER": (33, 215),
    "GeForce RTX 2070": (29, 175), "GeForce RTX 2060 SUPER": (27, 175),
    "GeForce RTX 2060": (23, 160), "GeForce GTX 1660 Ti": (20, 120),
    "GeForce GTX 1660 SUPER": (20, 125), "GeForce GTX 1660": (18, 120),
    "GeForce GTX 1650 SUPER": (17, 100), "GeForce GTX 1650 G6": (12, 75),
    "GeForce GTX 1080 Ti": (33, 250), "GeForce GTX 1080": (25, 180),
    "GeForce GTX 1070 Ti": (22, 180), "GeForce GTX 1070": (20, 150),
    "GeForce GTX 1060 6GB": (13.5, 120), "GeForce GTX 1050 Ti": (9, 75),
    # AMD RX 9000 / 7000 / 6000 / 5000
    "Radeon RX 9070 XT": (70, 304), "Radeon RX 9070": (63, 220),
    "Radeon RX 9060 XT": (40, 160), "Radeon RX 7900 XTX": (70, 355),
    "Radeon RX 7900 XT": (61, 315), "Radeon RX 7900 GRE": (55, 260),
    "Radeon RX 7800 XT": (52, 263), "Radeon RX 7700 XT": (45, 245),
    "Radeon RX 7600 XT": (32, 190), "Radeon RX 7600": (29, 165),
    "Radeon RX 6950 XT": (55, 335), "Radeon RX 6900 XT": (52, 300),
    "Radeon RX 6800 XT": (50, 300), "Radeon RX 6800": (45, 250),
    "Radeon RX 6750 XT": (38, 250), "Radeon RX 6700 XT": (36, 230),
    "Radeon RX 6650 XT": (27, 180), "Radeon RX 6600 XT": (26, 160),
    "Radeon RX 6600": (24, 132), "Radeon RX 6500 XT": (14, 107),
    "Radeon RX 6400": (11, 53), "Radeon RX 5700 XT": (31, 225),
    "Radeon RX 5600 XT": (24, 150), "Radeon RX 5500 XT": (15, 130),
    "Radeon RX 580": (13, 185), "Radeon RX 570": (11, 150),
    # Intel Arc
    "Arc B580": (34, 190), "Arc B570": (30, 150), "Arc A770": (30, 225),
    "Arc A750": (27, 225), "Arc A580": (23, 185),
}

# CPU microarchitecture -> (socket, relative IPC/generation factor)
ARCH_INFO = {
    "Zen": ("AM4", 0.65), "Zen+": ("AM4", 0.72), "Zen 2": ("AM4", 0.85),
    "Zen 3": ("AM4", 1.00), "Zen 4": ("AM5", 1.15), "Zen 5": ("AM5", 1.28),
    "Comet Lake": ("LGA1200", 0.85), "Rocket Lake": ("LGA1200", 0.95),
    "Alder Lake": ("LGA1700", 1.05), "Raptor Lake": ("LGA1700", 1.15),
    "Raptor Lake Refresh": ("LGA1700", 1.18), "Arrow Lake": ("LGA1851", 1.20),
}

CHIPSET_RE = re.compile(
    r"(?<![A-Z0-9])(A520|B450|B550|X570|A620|B650|X670|B840|B850|X870|"
    r"H410|B460|H470|Z490|H510|B560|H570|Z590|H610|B660|H670|Z690|B760|"
    r"H770|Z790|H810|B860|Z890)(?![0-9])")
CHIPSET_TIER = {  # 1 = entry, 2 = mainstream, 3 = enthusiast
    "A520": 1, "B450": 1, "A620": 1, "H410": 1, "H510": 1, "H610": 1,
    "H810": 1, "B840": 1, "B550": 2, "B650": 2, "B460": 2, "B560": 2,
    "B660": 2, "B760": 2, "B860": 2, "H470": 2, "H570": 2, "H670": 2,
    "H770": 2, "B850": 3, "X570": 3, "X670": 3, "X870": 3, "Z490": 3,
    "Z590": 3, "Z690": 3, "Z790": 3, "Z890": 3,
}

# Which case types can hold which motherboard form factors (towers only,
# so that a discrete GPU physically fits).
_ATX_CASES = {"ATX Mid Tower", "ATX Full Tower", "ATX Mini Tower"}
_MATX_CASES = _ATX_CASES | {"MicroATX Mini Tower", "MicroATX Mid Tower"}
_ITX_CASES = _MATX_CASES | {"Mini ITX Tower"}
CASE_SUPPORT = {"ATX": _ATX_CASES, "Micro ATX": _MATX_CASES,
                "Mini ITX": _ITX_CASES}
COMPACT_CASES = {"MicroATX Mini Tower", "MicroATX Mid Tower", "Mini ITX Tower"}

EFF_RANK = {"plus": 1, "bronze": 2, "silver": 3, "gold": 4,
            "platinum": 5, "titanium": 6}

# CPU usefulness stops growing beyond this normalised score (light workloads)
CPU_SATURATION = dict(office=0.35, general=0.50, programming=0.80)

MIN_CORES = dict(gaming=6, streaming=8, video_editing=8, ai_ml=6,
                 programming=6, office=4, general=4)

USE_CASES = ["gaming", "video_editing", "office", "programming",
             "streaming", "ai_ml", "general"]

# use case -> (cpu weight, gpu weight, cpu score kind, needs discrete GPU,
#              default ram GB, default storage GB)
PROFILES = {
    "gaming":        dict(wc=0.35, wg=0.65, cpu="gaming", dgpu=True,  ram=16, ssd=1000),
    "video_editing": dict(wc=0.50, wg=0.50, cpu="multi",  dgpu=True,  ram=32, ssd=1000),
    "streaming":     dict(wc=0.50, wg=0.50, cpu="multi",  dgpu=True,  ram=32, ssd=1000),
    "ai_ml":         dict(wc=0.30, wg=0.70, cpu="multi",  dgpu=True,  ram=32, ssd=1000),
    "programming":   dict(wc=0.90, wg=0.10, cpu="multi",  dgpu=False, ram=32, ssd=1000),
    "office":        dict(wc=1.00, wg=0.00, cpu="multi",  dgpu=False, ram=16, ssd=500),
    "general":       dict(wc=0.85, wg=0.15, cpu="multi",  dgpu=False, ram=16, ssd=500),
}

NUM = lambda x: pd.to_numeric(x, errors="coerce")


# --------------------------------------------------------------------------
# 1. Loading and cleaning
# --------------------------------------------------------------------------
def _read(csv_dir, name):
    return pd.read_csv(os.path.join(csv_dir, name + ".csv"))


def _priced_unique(df):
    """Keep only rows with a price and the cheapest row per product name."""
    df = df.copy()
    df["price"] = NUM(df["price"])
    df = df[df["price"] > 0].dropna(subset=["price", "name"])
    return df.sort_values("price").drop_duplicates("name", keep="first")


def clean_cpu(df):
    df = _priced_unique(df)
    n = df["name"].str.lower()
    df = df[~n.str.contains("xeon|threadripper|epyc|pro \\d")]
    info = df["microarchitecture"].map(ARCH_INFO)
    df = df[info.notna()].copy()
    info = info[df.index]
    df["socket"] = [i[0] for i in info]
    df["arch_factor"] = [i[1] for i in info]
    df["brand"] = np.where(df["name"].str.contains("AMD"), "amd", "intel")
    df["boost"] = NUM(df["boost_clock"]).fillna(NUM(df["core_clock"]))
    df["cores"] = NUM(df["core_count"])
    df["tdp"] = NUM(df["tdp"])
    df["has_igpu"] = df["graphics"].notna()
    df["x3d"] = df["name"].str.contains("X3D")
    # hybrid Intel (12th gen+): E-cores count roughly half
    hybrid = df["microarchitecture"].isin(
        ["Alder Lake", "Raptor Lake", "Raptor Lake Refresh", "Arrow Lake"])
    df["eff_cores"] = np.where(hybrid & (df["cores"] > 8),
                               8 + (df["cores"] - 8) * 0.5, df["cores"])
    df["gaming_raw"] = (df["boost"] * df["arch_factor"]
                        * np.where(df["x3d"], 1.25, 1.0)
                        * np.minimum(df["cores"], 8) ** 0.3)
    df["multi_raw"] = df["eff_cores"] ** 0.9 * df["boost"] * df["arch_factor"]
    df["gaming_score"] = df["gaming_raw"] / df["gaming_raw"].max()
    df["multi_score"] = df["multi_raw"] / df["multi_raw"].max()
    # coolers: unlocked / X-series chips normally ship WITHOUT a cooler
    df["needs_cooler"] = (df["name"].str.contains(r"(?:K|KF|KS|X|X3D|XT)$")
                          | (df["tdp"] >= 105))
    return df.reset_index(drop=True)


def clean_gpu(df):
    df = _priced_unique(df)
    df = df[df["chipset"].isin(GPU_TABLE)].copy()
    df["perf"] = df["chipset"].map(lambda c: GPU_TABLE[c][0])
    df["tdp"] = df["chipset"].map(lambda c: GPU_TABLE[c][1])
    df["vram"] = NUM(df["memory"])
    df["brand"] = np.where(df["chipset"].str.startswith("GeForce"), "nvidia",
                   np.where(df["chipset"].str.startswith("Radeon"), "amd",
                            "intel"))
    df = df[df["vram"].notna()].copy()
    # cheapest card per (chipset, VRAM size) keeps the search space small
    df = df.sort_values("price").drop_duplicates(["chipset", "vram"], keep="first")
    # more VRAM than the base 8 GB gives a small bonus (and less a penalty)
    adj = 1 + 0.08 * np.log2(np.maximum(df["vram"], 4) / 8.0)
    df["score"] = df["perf"] * adj / 130.0          # RTX 5090 ~ 1.0
    return df.reset_index(drop=True)


def _board_ddr(row):
    """Return (ddr_generation, confirmed). None = cannot be inferred."""
    s, name, chip = row["socket"], row["name"], row["chipset_code"]
    if s in ("AM4", "LGA1200"):
        return 4, True
    if s in ("AM5", "LGA1851"):
        return 5, True
    if s == "LGA1700":   # supports both generations -> look at the name
        if re.search(r"DDR\s?4|\bD4\b|[-/ ]D4", name, re.I):
            return 4, True
        if re.search(r"DDR\s?5|\bD5\b", name, re.I):
            return 5, True
        if chip == "H610":
            return 4, False       # assumed
        if chip in ("Z790", "Z690"):
            return 5, False       # assumed
    return None, False


def clean_motherboard(df):
    df = _priced_unique(df)
    df = df[df["socket"].isin(SUPPORTED_SOCKETS)].copy()
    df["chipset_code"] = df["name"].str.upper().str.extract(CHIPSET_RE)[0]
    df = df[df["chipset_code"].notna()]
    df = df[df["form_factor"].isin(["ATX", "Micro ATX", "Mini ITX"])].copy()
    ddr = df.apply(_board_ddr, axis=1)
    df["ddr"] = [d[0] for d in ddr]
    df["ddr_confirmed"] = [d[1] for d in ddr]
    df = df[df["ddr"].notna()].copy()
    df["tier"] = df["chipset_code"].map(CHIPSET_TIER)
    df["max_mem"] = NUM(df["max_memory"])
    df["slots"] = NUM(df["memory_slots"])
    return df.reset_index(drop=True)


def clean_memory(df):
    df = _priced_unique(df)
    sp = df["speed"].astype(str).str.split(",", expand=True)
    mo = df["modules"].astype(str).str.split(",", expand=True)
    df["ddr"] = NUM(sp[0])
    df["mhz"] = NUM(sp[1])
    df["n_mod"] = NUM(mo[0])
    df["gb_each"] = NUM(mo[1])
    df["total_gb"] = df["n_mod"] * df["gb_each"]
    df = df[((df["ddr"] == 4) & (df["mhz"] >= 3200)) |
            ((df["ddr"] == 5) & (df["mhz"] >= 5600))]
    df = df[df["n_mod"].isin([1, 2])]
    df = df[(df["n_mod"] == 2) | (df["total_gb"] <= 8)]
    return df.reset_index(drop=True)


def clean_storage(df):
    df = _priced_unique(df)
    df["capacity"] = NUM(df["capacity"])
    df = df[(df["type"] == "SSD") & (df["form_factor"] == "M.2-2280")
            & df["interface"].astype(str).str.startswith("M.2 PCIe")]
    df = df[df["capacity"].between(250, 4000)]
    return df.reset_index(drop=True)


PSU_BRANDS = ("Corsair", "SeaSonic", "Seasonic", "EVGA", "be quiet!", "Thermaltake",
              "MSI", "Cooler Master", "Gigabyte", "Asus", "NZXT", "Antec", "FSP",
              "Super Flower", "Fractal Design", "Lian Li", "SilverStone",
              "Silverstone", "XPG", "DeepCool", "Phanteks", "ADATA")


def clean_psu(df):
    df = _priced_unique(df)
    df = df[df["name"].str.startswith(PSU_BRANDS)]   # reputable brands only
    df["watt"] = NUM(df["wattage"])
    df["eff"] = df["efficiency"].map(EFF_RANK)
    df = df[(df["type"] == "ATX") & (df["eff"] >= 2) & (df["watt"] >= 350)]
    return df.reset_index(drop=True)


def clean_case(df):
    df = _priced_unique(df)
    df = df[df["type"].isin(_ITX_CASES) & (df["price"] >= 30)]
    return df.reset_index(drop=True)


def clean_cooler(df):
    df = _priced_unique(df)
    df = df[df["size"].isna() & df["noise_level"].notna() & (df["price"] >= 15)]
    return df.reset_index(drop=True)


def clean_monitor(df):
    df = _priced_unique(df)
    res = df["resolution"].astype(str).str.split(",", expand=True)
    df["w"], df["h"] = NUM(res[0]), NUM(res[1])
    df["size_in"] = NUM(df["screen_size"])
    df["hz"] = NUM(df["refresh_rate"])
    df = df[df["h"].notna() & df["hz"].notna() & df["size_in"].between(21, 34)
            & (df["h"] >= 1080)]
    return df.reset_index(drop=True)


def clean_keyboard(df):
    df = _priced_unique(df)
    return df[df["price"] >= 12].reset_index(drop=True)


def clean_mouse(df):
    df = _priced_unique(df)
    df["dpi"] = NUM(df["max_dpi"])
    return df[df["price"] >= 8].reset_index(drop=True)


def load_data(csv_dir):
    """Load every CSV the engine needs and return cleaned DataFrames."""
    d = dict(
        cpu=clean_cpu(_read(csv_dir, "cpu")),
        gpu=clean_gpu(_read(csv_dir, "video-card")),
        mobo=clean_motherboard(_read(csv_dir, "motherboard")),
        ram=clean_memory(_read(csv_dir, "memory")),
        ssd=clean_storage(_read(csv_dir, "internal-hard-drive")),
        psu=clean_psu(_read(csv_dir, "power-supply")),
        case=clean_case(_read(csv_dir, "case")),
        cooler=clean_cooler(_read(csv_dir, "cpu-cooler")),
        monitor=clean_monitor(_read(csv_dir, "monitor")),
        keyboard=clean_keyboard(_read(csv_dir, "keyboard")),
        mouse=clean_mouse(_read(csv_dir, "mouse")),
    )
    return d


def cleaning_report(csv_dir, data):
    """Rows before/after cleaning for the report."""
    files = dict(cpu="cpu", gpu="video-card", mobo="motherboard",
                 ram="memory", ssd="internal-hard-drive", psu="power-supply",
                 case="case", cooler="cpu-cooler", monitor="monitor",
                 keyboard="keyboard", mouse="mouse")
    rows = []
    for k, f in files.items():
        raw = _read(csv_dir, f)
        rows.append(dict(category=k, raw_rows=len(raw),
                         missing_price_pct=round(100 * raw["price"].isna().mean(), 1),
                         usable_rows=len(data[k])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Compatibility knowledge
# --------------------------------------------------------------------------
def chipset_supports_cpu(chipset, cpu_arch, socket):
    """Out-of-the-box BIOS support rules (simplified)."""
    if socket == "AM4":
        if cpu_arch in ("Zen 3",):
            return chipset in ("B550", "X570", "A520")
        return chipset in ("B550", "X570", "A520", "B450")
    if socket == "LGA1200":
        if cpu_arch == "Rocket Lake":
            return chipset in ("B560", "H510", "H570", "Z590")
        return True
    return True


def psu_required_watts(cpu_tdp, gpu_tdp):
    raw = (cpu_tdp + gpu_tdp + 100) * 1.25
    return int(np.ceil(raw / 50.0) * 50)


# --------------------------------------------------------------------------
# 3. Build engine
# --------------------------------------------------------------------------
@dataclass
class Requirement:
    use_case: str = "general"
    budget: float = 4000.0
    currency: str = "MYR"
    cpu_brand: str = "any"
    gpu_brand: str = "any"
    ram_gb: int = None
    storage_gb: int = None
    extras: tuple = ()
    compact: bool = False

    @classmethod
    def from_dict(cls, d):
        uc = d.get("use_case") or "general"
        if uc not in USE_CASES:
            uc = "general"
        ex = d.get("extras") or []
        if isinstance(ex, str):
            ex = [ex]
        return cls(
            use_case=uc,
            budget=float(d.get("budget") or 4000),
            currency=(d.get("currency") or "MYR").upper(),
            cpu_brand=(d.get("cpu_brand") or "any").lower(),
            gpu_brand=(d.get("gpu_brand") or "any").lower(),
            ram_gb=int(d["ram_gb"]) if d.get("ram_gb") else None,
            storage_gb=int(d["storage_gb"]) if d.get("storage_gb") else None,
            extras=tuple(e for e in ex if e in ("monitor", "keyboard", "mouse")),
            compact=bool(d.get("compact")),
        )

    @property
    def budget_usd(self):
        return self.budget if self.currency == "USD" else self.budget / USD_TO_MYR


class BuildEngine:
    def __init__(self, data):
        self.d = data
        self._ram_cache = {}
        self._plat_cache = {}
        self._psu_cache = {}

    # ---- helpers ---------------------------------------------------------
    def _cheapest_ram(self, ddr, gb):
        key = (ddr, gb)
        if key not in self._ram_cache:
            r = self.d["ram"]
            r = r[(r["ddr"] == ddr) & (r["total_gb"] == gb)]
            self._ram_cache[key] = None if len(r) == 0 else \
                r.sort_values("price").iloc[0]
        return self._ram_cache[key]

    def _platform(self, cpu, req, ram_gb, min_tier):
        key = (cpu["socket"], cpu["microarchitecture"], req.compact, ram_gb, min_tier)
        if key not in self._plat_cache:
            self._plat_cache[key] = self._platform_uncached(cpu, req, ram_gb, min_tier)
        return self._plat_cache[key]

    def _platform_uncached(self, cpu, req, ram_gb, min_tier):
        """Cheapest (board, ram) pair compatible with the CPU, or None."""
        b = self.d["mobo"]
        b = b[(b["socket"] == cpu["socket"]) & (b["tier"] >= min_tier)]
        b = b.loc[b["chipset_code"].map(lambda c: chipset_supports_cpu(
            c, cpu["microarchitecture"], cpu["socket"])).astype(bool)]
        if req.compact:
            b = b[b["form_factor"].isin(["Mini ITX", "Micro ATX"])]
        b = b[(b["max_mem"] >= ram_gb) & (b["slots"] >= 2)]
        best = None
        for ddr in b["ddr"].unique():
            ram = self._cheapest_ram(int(ddr), ram_gb)
            if ram is None:
                continue
            board = b[b["ddr"] == ddr].sort_values("price").iloc[0]
            tot = board["price"] + ram["price"]
            if best is None or tot < best[0]:
                best = (tot, board, ram)
        return best

    def _case(self, ff, compact):
        c = self.d["case"]
        c = c[c["type"].isin(CASE_SUPPORT[ff])]
        if compact:
            c = c[c["type"].isin(COMPACT_CASES)]
        return None if len(c) == 0 else c.sort_values("price").iloc[0]

    def _ssd(self, gb):
        s = self.d["ssd"]
        s = s[(s["capacity"] >= gb) & (s["capacity"] <= gb * 2.1)]
        return None if len(s) == 0 else s.sort_values("price").iloc[0]

    def _psu(self, watts, min_eff):
        key = (watts, min_eff)
        if key not in self._psu_cache:
            p = self.d["psu"]
            p = p[(p["watt"] >= watts) & (p["eff"] >= min_eff)]
            self._psu_cache[key] = None if len(p) == 0 else \
                p.sort_values("price").iloc[0]
        return self._psu_cache[key]

    def _cooler(self, cpu):
        if not cpu["needs_cooler"]:
            return None
        c = self.d["cooler"]
        floor = 40 if cpu["tdp"] >= 125 else 20
        c = c[c["price"] >= floor]
        return None if len(c) == 0 else c.sort_values("price").iloc[0]

    # ---- extras ----------------------------------------------------------
    def _pick_extras(self, req, budget_usd):
        caps = dict(monitor=0.15, keyboard=0.03, mouse=0.02)
        out = {}
        for e in req.extras:
            cap = budget_usd * caps[e]
            if e == "keyboard":
                cap = min(cap, 80)
            elif e == "mouse":
                cap = min(cap, 60)
            if e == "monitor":
                m = self.d["monitor"]
                m = m[m["price"] <= cap]
                if req.use_case == "office" or req.use_case == "general":
                    m = m[m["hz"] >= 60]
                    score = (m["w"] * m["h"] / 2.0e6) + 0.2 * (m["panel_type"] == "IPS")
                elif req.use_case in ("gaming", "streaming"):
                    m = m[m["hz"] >= 100]
                    score = (m["w"] * m["h"] / 2.0e6) * 0.5 + m["hz"] / 144 \
                        + 0.2 * (m["panel_type"] == "IPS")
                else:  # editing / programming / ai
                    score = (m["w"] * m["h"] / 2.0e6) + 0.5 * (m["panel_type"] == "IPS")
                if len(m):
                    out["monitor"] = m.loc[score.idxmax()]
            elif e == "keyboard":
                k = self.d["keyboard"]
                k = k[(k["price"] <= cap) & ~k["name"].str.contains(
                    "combo|kit|set|bundle", case=False)]
                if req.use_case in ("gaming", "streaming") and \
                        (k["style"] == "Gaming").any():
                    k = k[k["style"] == "Gaming"]
                if len(k):
                    i = (k["price"] - 0.7 * cap).abs().idxmin()
                    out["keyboard"] = k.loc[i]
            elif e == "mouse":
                m = self.d["mouse"]
                m = m[(m["price"] <= cap) & m["dpi"].notna()]
                if req.use_case in ("gaming", "streaming") and (m["dpi"] >= 8000).any():
                    m = m[m["dpi"] >= 8000]
                if len(m):
                    i = (m["price"] - 0.7 * cap).abs().idxmin()
                    out["mouse"] = m.loc[i]
        return out

    # ---- main search -----------------------------------------------------
    def recommend(self, req):
        if isinstance(req, dict):
            req = Requirement.from_dict(req)
        prof = PROFILES[req.use_case]
        total_usd = req.budget_usd

        extras = self._pick_extras(req, total_usd)
        extras_cost = sum(float(r["price"]) for r in extras.values())
        core_budget = total_usd - extras_cost

        ram_gb = req.ram_gb or prof["ram"]
        if ram_gb > 64:
            return dict(ok=False, requirement=req, message="This prototype supports up to 64 GB RAM.")
        ssd_gb = req.storage_gb or prof["ssd"]

        # small budgets: scale the defaults down instead of failing
        if not req.ram_gb and core_budget < 700:
            ram_gb = 16 if core_budget >= 450 else 8
        if not req.storage_gb and core_budget < 700:
            ssd_gb = 500

        ssd = self._ssd(ssd_gb) if ssd_gb else None
        if ssd is None:
            return dict(ok=False, requirement=req, message="No SSD meets the requested capacity in this catalogue.")

        cpus = self.d["cpu"]
        if req.cpu_brand in ("amd", "intel"):
            cpus = cpus[cpus["brand"] == req.cpu_brand]

        gpus = self.d["gpu"]
        if req.gpu_brand in ("nvidia", "amd", "intel"):
            gpus = gpus[gpus["brand"] == req.gpu_brand]
        if req.use_case == "ai_ml":
            gpus = gpus[gpus["vram"] >= 12] if (gpus["vram"] >= 12).any() else gpus

        # light-use profiles run on integrated graphics unless a GPU brand
        # was explicitly requested
        wants_gpu = prof["dgpu"] or req.gpu_brand != "any"
        if wants_gpu:
            gpu_options = [g for _, g in gpus.iterrows()]
        else:
            gpu_options = [None]
            cpus = cpus[cpus["has_igpu"]]
        cpu_key = "gaming_score" if prof["cpu"] == "gaming" else "multi_score"

        strict = cpus[cpus["cores"] >= MIN_CORES[req.use_case]]
        best, cheapest = self._search(strict, gpu_options, req, prof, cpu_key,
                                      ram_gb, ssd, core_budget)
        if best is None:                      # budget too small -> relax
            best, cheapest = self._search(cpus, gpu_options, req, prof, cpu_key,
                                          ram_gb, ssd, core_budget)

        if best is None:
            mn = None if cheapest is None else (cheapest[-2] + extras_cost) * USD_TO_MYR
            return dict(ok=False, requirement=req, min_cost_myr=mn,
                        message=("Budget is too low for a compatible build"
                                 + ("" if mn is None else
                                    f"; the cheapest build in the dataset is about RM {mn:,.0f}.")))

        _, _, cpu, gpu, board, ram, case, cooler, psu, total, watts = best
        parts = dict(cpu=cpu, mobo=board, ram=ram, ssd=ssd, case=case, psu=psu)
        if gpu is not None:
            parts["gpu"] = gpu
        if cooler is not None:
            parts["cooler"] = cooler
        parts.update(extras)

        # spend what is left on a better PSU / more RAM / more storage
        left = core_budget - total
        parts, left, upgrades = self._upgrade(parts, left, req, ram_gb, ssd_gb, watts)

        return self._pack(req, parts, watts, left, upgrades)

    def _search(self, cpus, gpu_options, req, prof, cpu_key, ram_gb, ssd,
                core_budget):
        best, cheapest = None, None
        for _, cpu in cpus.iterrows():
            min_tier = 2 if (cpu["tdp"] >= 105 or cpu["cores"] >= 12) else 1
            plat = self._platform(cpu, req, ram_gb, min_tier)
            if plat is None:
                continue
            _, board, ram = plat
            case = self._case(board["form_factor"], req.compact)
            if case is None:
                continue
            cooler = self._cooler(cpu)
            if cpu["needs_cooler"] and cooler is None:
                continue
            fixed = (cpu["price"] + board["price"] + ram["price"]
                     + case["price"] + (ssd["price"] if ssd is not None else 0)
                     + (cooler["price"] if cooler is not None else 0))
            for gpu in gpu_options:
                if gpu is None and not cpu["has_igpu"]:
                    continue
                gtdp = 0 if gpu is None else gpu["tdp"]
                watts = psu_required_watts(cpu["tdp"], gtdp)
                psu = self._psu(watts, 4 if gtdp >= 250 else 2)
                if psu is None:
                    continue
                total = fixed + psu["price"] + (0 if gpu is None else gpu["price"])
                obj = self._objective(req.use_case, prof, cpu, gpu, cpu_key)
                obj *= (1 - 0.12 * total / max(core_budget, 1))   # mild thrift
                cand = (obj, -total, cpu, gpu, board, ram, case, cooler, psu,
                        total, watts)
                if cheapest is None or total < cheapest[-2]:
                    cheapest = cand
                if total <= core_budget and (best is None or cand[:2] > best[:2]):
                    best = cand

        return best, cheapest

    @staticmethod
    def _objective(use_case, prof, cpu, gpu, cpu_key):
        """Estimated usefulness of a CPU+GPU pair for the use case."""
        c = min(float(cpu[cpu_key]), CPU_SATURATION.get(use_case, 9.0))
        if gpu is None:                      # integrated graphics build
            return c
        g = float(gpu["score"])
        if prof["cpu"] == "gaming":
            # FPS is limited by the slower of CPU and GPU -> soft minimum
            c, p = c * 1.3, 3.0
            return (c ** -p + g ** -p) ** (-1.0 / p)
        if use_case == "ai_ml":
            g *= 0.5 + 0.5 * min(float(gpu["vram"]), 24) / 24
            if gpu["brand"] == "nvidia":
                g *= 1.25          # CUDA ecosystem
        wg = max(prof["wg"], 0.3)                         # user asked for a GPU
        return (c ** (1 - wg)) * (g ** wg)                # weighted geometric mean

    def _upgrade(self, parts, left, req, ram_gb, ssd_gb, watts):
        notes = []
        # 1) PSU -> gold if affordable
        cur = parts["psu"]
        if cur["eff"] < 4:
            alt = self._psu(watts, 4)
            if alt is not None and alt["price"] - cur["price"] <= left * 0.5:
                left -= alt["price"] - cur["price"]
                parts["psu"] = alt
                notes.append("PSU upgraded to 80+ Gold with leftover budget")
        # 2) storage -> next size up
        for nxt in (500, 1000, 2000, 4000):
            if nxt <= ssd_gb:
                continue
            alt = self._ssd(nxt)
            if alt is not None and alt["price"] - parts["ssd"]["price"] <= left * 0.6:
                left -= alt["price"] - parts["ssd"]["price"]
                parts["ssd"] = alt
                notes.append(f"SSD increased to {int(alt['capacity'])} GB")
            break
        # 3) RAM -> 32 GB if on 16
        if ram_gb == 16 and not req.ram_gb:
            ram = self._cheapest_ram(int(parts["ram"]["ddr"]), 32)
            if ram is not None and ram["price"] - parts["ram"]["price"] <= left * 0.8 \
                    and parts["mobo"]["max_mem"] >= 32:
                left -= ram["price"] - parts["ram"]["price"]
                parts["ram"] = ram
                notes.append("RAM increased to 32 GB")
        # 4) tempered-glass case for gaming/streaming builds
        if req.use_case in ("gaming", "streaming"):
            ff = parts["mobo"]["form_factor"]
            c = self.d["case"]
            c = c[c["type"].isin(CASE_SUPPORT[ff]) & (c["side_panel"] == "Tempered Glass")]
            if req.compact:
                c = c[c["type"].isin(COMPACT_CASES)]
            if len(c):
                alt = c.sort_values("price").iloc[0]
                if 0 < alt["price"] - parts["case"]["price"] <= left * 0.6:
                    left -= alt["price"] - parts["case"]["price"]
                    parts["case"] = alt
                    notes.append("Case changed to a tempered-glass model")
        return parts, left, notes

    # ---- output ------------------------------------------------------------
    def _pack(self, req, parts, watts, left, upgrades):
        order = ["cpu", "cooler", "mobo", "ram", "gpu", "ssd", "psu", "case",
                 "monitor", "keyboard", "mouse"]
        items, total = [], 0.0
        for k in order:
            if k not in parts:
                continue
            r = parts[k]
            total += float(r["price"])
            items.append(dict(category=k, name=r["name"],
                              price_usd=float(r["price"]),
                              price_myr=float(r["price"]) * USD_TO_MYR,
                              spec=self._spec(k, r)))
        warnings = []
        if left / max(req.budget_usd, 1) > 0.35 and req.use_case in ("office", "general", "programming"):
            warnings.append("This use case does not need the whole budget, so the "
                            "build stops early. Name a GPU brand or a higher use "
                            "case (e.g. gaming) if you want to spend more.")
        missing = sorted(set(req.extras) - set(parts))
        if missing:
            warnings.append("Requested peripherals not included within their allocated budget: " + ", ".join(missing))
        b = parts["mobo"]
        if not b["ddr_confirmed"]:
            warnings.append("Memory type of the motherboard was inferred from its "
                            "chipset; verify DDR4/DDR5 on the manufacturer page.")
        warnings.append("CPU-cooler socket support and GPU length vs case "
                        "clearance are not in the dataset - verify before buying.")
        return dict(ok=True, requirement=req, items=items,
                    total_usd=total, total_myr=total * USD_TO_MYR,
                    budget_myr=req.budget_usd * USD_TO_MYR,
                    left_myr=left * USD_TO_MYR, psu_min_watts=watts,
                    upgrades=upgrades, warnings=warnings)

    @staticmethod
    def _spec(k, r):
        f = lambda v: "" if pd.isna(v) else v
        if k == "cpu":
            return (f"{int(r['cores'])} cores, up to {r['boost']:.1f} GHz, "
                    f"{int(r['tdp'])} W, socket {r['socket']}"
                    + (", integrated graphics" if r["has_igpu"] else ""))
        if k == "gpu":
            return f"{r['chipset']}, {int(r['vram'])} GB VRAM, ~{int(r['tdp'])} W"
        if k == "mobo":
            return f"{r['chipset_code']} chipset, {r['socket']}, {r['form_factor']}, DDR{int(r['ddr'])}"
        if k == "ram":
            return f"{int(r['total_gb'])} GB ({int(r['n_mod'])}x{int(r['gb_each'])}) DDR{int(r['ddr'])}-{int(r['mhz'])}"
        if k == "ssd":
            return f"{int(r['capacity'])} GB NVMe, {r['interface']}"
        if k == "psu":
            return f"{int(r['watt'])} W, 80+ {r['efficiency']}, {f(r['modular'])} modular"
        if k == "case":
            return f"{r['type']}, {f(r['side_panel'])}"
        if k == "cooler":
            return f"air cooler, {f(r['noise_level'])} dBA"
        if k == "monitor":
            return f"{r['size_in']:.0f}\" {int(r['w'])}x{int(r['h'])} {int(r['hz'])} Hz {f(r['panel_type'])}"
        if k == "keyboard":
            return f"{f(r['style'])} keyboard, {f(r['connection_type'])}"
        if k == "mouse":
            return f"mouse, up to {int(r['dpi']) if not pd.isna(r['dpi']) else '?'} DPI"
        return ""


# --------------------------------------------------------------------------
# 4. Presentation helpers
# --------------------------------------------------------------------------
CAT_LABEL = dict(cpu="CPU", cooler="CPU cooler", mobo="Motherboard", ram="Memory",
                 gpu="Graphics card", ssd="Storage", psu="Power supply",
                 case="Case", monitor="Monitor", keyboard="Keyboard", mouse="Mouse")


def format_build(res):
    if not res["ok"]:
        return res["message"]
    lines = []
    for it in res["items"]:
        lines.append(f"{CAT_LABEL[it['category']]:<14} {it['name']}  "
                     f"[{it['spec']}]  RM {it['price_myr']:,.0f}")
    lines.append("-" * 70)
    lines.append(f"TOTAL RM {res['total_myr']:,.0f}  (budget RM {res['budget_myr']:,.0f}, "
                 f"left RM {res['left_myr']:,.0f})")
    for u in res["upgrades"]:
        lines.append("* " + u)
    for w in res["warnings"]:
        lines.append("! " + w)
    return "\n".join(lines)


def facts_text(res):
    """Compact fact sheet fed to OPT for grounded explanation generation."""
    r = res["requirement"]
    lines = [f"Use: {r.use_case.replace('_', ' ')} | Budget: RM {res['budget_myr']:,.0f} | "
             f"Total: RM {res['total_myr']:,.0f}"]
    for it in res["items"]:
        lines.append(f"{CAT_LABEL[it['category']]}: {it['name']} | {it['spec']} | "
                     f"RM {it['price_myr']:,.0f}")
    lines.append(f"Estimated power requirement: {res['psu_min_watts']} W")
    return "\n".join(lines)


_REASON = {
    "gaming": "Most of the budget goes to the graphics card because frame rate in games depends on it the most.",
    "video_editing": "Video editing needs many CPU cores and plenty of RAM, so the budget is split between a strong CPU and GPU, with as much RAM as the budget allows.",
    "streaming": "Streaming needs strong multi-core CPU power to encode while the GPU renders the game.",
    "ai_ml": "AI work depends on GPU memory (VRAM), so a graphics card with as much VRAM as the budget allows was chosen.",
    "programming": "Programming and compiling benefit from CPU cores and RAM, while a powerful GPU is not needed.",
    "office": "Office work only needs a modest CPU with integrated graphics, which keeps the cost low.",
    "general": "A balanced CPU with enough RAM and fast SSD storage suits everyday use.",
}


def template_explanation(res):
    """Rule-based explanation. Used as training target AND as safe fallback."""
    if not res["ok"]:
        return res["message"]
    r = res["requirement"]
    g = {it["category"]: it for it in res["items"]}
    s = [_REASON[r.use_case]]
    c = g["cpu"]
    s.append(f"The {c['name']} ({c['spec']}) was chosen as the processor.")
    if "gpu" in g:
        s.append(f"The {g['gpu']['name']} ({g['gpu']['spec']}) handles graphics.")
    else:
        s.append("No separate graphics card is needed because the CPU has integrated graphics.")
    s.append(f"The {g['mobo']['name']} is compatible with the CPU socket and uses "
             f"{g['ram']['spec'].split(')')[-1].strip().split('-')[0]} memory.")
    s.append(f"Storage is a {g['ssd']['spec'].split(',')[0]} and the "
             f"{g['psu']['name']} provides {g['psu']['spec'].split(',')[0]}, "
             f"above the estimated {res['psu_min_watts']} W requirement.")
    s.append(f"The total is RM {res['total_myr']:,.0f} against a budget of "
             f"RM {res['budget_myr']:,.0f}.")
    return " ".join(s)


# --------------------------------------------------------------------------
# 5. Independent validator + randomised compatibility test
# --------------------------------------------------------------------------
def validate_build(res, data):
    """Re-check a finished build against every rule. Returns violation list."""
    if not res["ok"]:
        return []
    req = res["requirement"]
    names = {it["category"]: it for it in res["items"]}
    look = lambda key, cat: data[key][data[key]["name"] == names[cat]["name"]].iloc[0]
    cpu, mb, ram = look("cpu", "cpu"), look("mobo", "mobo"), look("ram", "ram")
    psu, case = look("psu", "psu"), look("case", "case")
    v = []
    if cpu["socket"] != mb["socket"]:
        v.append("socket mismatch")
    if not chipset_supports_cpu(mb["chipset_code"], cpu["microarchitecture"], cpu["socket"]):
        v.append("chipset does not support CPU generation")
    if int(ram["ddr"]) != int(mb["ddr"]):
        v.append("DDR generation mismatch")
    if ram["total_gb"] > mb["max_mem"]:
        v.append("RAM exceeds board maximum")
    if ram["n_mod"] > mb["slots"]:
        v.append("more RAM modules than slots")
    if case["type"] not in CASE_SUPPORT[mb["form_factor"]]:
        v.append("board does not fit case")
    if req.compact and case["type"] not in COMPACT_CASES:
        v.append("compact requested but case is not compact")
    if "gpu" in names:
        gpu = look("gpu", "gpu")
        gtdp = gpu["tdp"]
        if req.gpu_brand in ("nvidia", "amd", "intel") and gpu["brand"] != req.gpu_brand:
            v.append("GPU brand preference ignored")
    else:
        gtdp = 0
        if not cpu["has_igpu"]:
            v.append("no GPU and CPU has no integrated graphics")
    if psu["watt"] < psu_required_watts(cpu["tdp"], gtdp):
        v.append("PSU too weak")
    if req.cpu_brand in ("amd", "intel") and cpu["brand"] != req.cpu_brand:
        v.append("CPU brand preference ignored")
    if res["total_myr"] > res["budget_myr"] + 0.01:
        v.append("over budget")
    if req.ram_gb and int(ram["total_gb"]) < req.ram_gb:
        v.append("less RAM than requested")
    if "ssd" not in names:
        v.append("missing SSD")
    elif req.storage_gb and look("ssd", "ssd")["capacity"] < req.storage_gb:
        v.append("less storage than requested")
    if cpu["needs_cooler"] and "cooler" not in names:
        v.append("missing required CPU cooler")
    # Optional peripherals may be omitted; _pack reports these explicitly.
    return v


def run_compat_tests(engine, n=300, seed=0):
    """Random requirements -> build -> validate. Returns summary DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        req = dict(
            use_case=str(rng.choice(USE_CASES)),
            budget=float(rng.choice(np.arange(1500, 15001, 500))),
            currency="MYR",
            cpu_brand=str(rng.choice(["any", "any", "amd", "intel"])),
            gpu_brand=str(rng.choice(["any", "any", "nvidia", "amd", "intel"])),
            ram_gb=int(rng.choice([0, 0, 16, 32, 64])) or None,
            storage_gb=int(rng.choice([0, 0, 500, 1000, 2000])) or None,
            extras=[e for e in ("monitor", "keyboard", "mouse") if rng.random() < 0.25],
            compact=bool(rng.random() < 0.15),
        )
        res = engine.recommend(req)
        viol = validate_build(res, engine.d)
        rows.append(dict(**{k: (str(v) if isinstance(v, list) else v)
                            for k, v in req.items()},
                         built=res["ok"], violations=len(viol),
                         violation_text="; ".join(viol)))
    return pd.DataFrame(rows)
