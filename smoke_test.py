"""Offline project smoke tests. No OPT download or Streamlit server required."""
import pc_engine as E
import advisor as A
import catalog_nlu as C

engine = A.load_engine()
data = engine.d

# 1. Baseline PC-build requests.
for text in [
    "Gaming PC below RM4500 with Ryzen and 16GB RAM",
    "Nak PC untuk kerja pejabat, bajet RM3000",
    "Need a coding PC, budget USD1000 with 1TB SSD",
    "Gaming lah, budget RM5000, include monitor and mouse",
]:
    req = A.rule_requirements(text, data)
    result, violations = A.recommend(engine, req)
    assert result["ok"], (text, result.get("message"))
    assert not violations, (text, violations)
    assert result["total_myr"] < result["budget_myr"]
    print(f"PASS build: {text} -> RM{result['total_myr']:.2f}")

# 2. Missing budget uses the documented default instead of an OPT guess.
req = A.rule_requirements("I need a gaming PC", data)
assert req["budget"] == A.DEFAULT_BUDGET_MYR
assert req["budget_is_default"] is True
print("PASS default budget")

# 3. Typo correction and extraction of several dataset-backed attributes.
query = (
    "Need a wireles mouse, black, 16000 DPI, white ATX case, "
    "DDR5 6000 32GB RAM, 2TB NVMe SSD, 750W gold PSU, "
    "RTX 5070 GPU and 27 inch IPS 144Hz monitor, budget RM15000."
)
req = A.rule_requirements(query, data)
assert "wireless mouse" in req["corrected_text"].lower()
prefs = req["component_preferences"]
assert prefs["mouse"]["connection_type"]["contains"] == "Wireless"
assert prefs["mouse"]["max_dpi"]["min"] == 16000
assert prefs["ram"]["total_gb"]["min"] == 32
assert prefs["ram"]["ddr"]["eq"] == 5
assert prefs["ssd"]["capacity"]["min"] == 2000
assert prefs["psu"]["wattage"]["min"] == 750
assert prefs["gpu"]["chipset"]["eq"] == "GeForce RTX 5070"
assert "case" in prefs and "monitor" in prefs
result, violations = A.recommend(engine, req)
assert result["ok"], result.get("message")
assert not violations, violations
print("PASS catalogue-aware attributes, typo correction and build filtering")

# 4. Attribute ownership, uncommon numeric phrasing and technical typos.
query = (
    "Gamming PC with a black wireles mouse, white ATX case, "
    "motheboard AM5, keybord RGB, 32GB DDR5 6000 RAM, "
    "and liquid cooler 240mm. Budget around 5k."
)
req = A.rule_requirements(query, data)
prefs = req["component_preferences"]
assert req["budget"] == 5000
assert "gaming" in req["corrected_text"].lower()
assert prefs["mouse"]["color"]["contains"] == "Black"
assert prefs["case"]["color"]["contains"] == "White"
assert prefs["mobo"]["socket"]["eq"] == "AM5"
assert prefs["keyboard"]["backlit"]["eq"] == "RGB"
assert prefs["ram"]["total_gb"]["min"] == 32
assert prefs["ram"]["mhz"]["min"] == 6000
assert prefs["cooler"]["size"]["eq"] == 240
print("PASS typo recovery, attribute ownership and numeric phrase extraction")

# 5. Every schema field is listed for reporting.
report = C.requirement_field_report(data, prefs, req)
assert len(report) == sum(len(fields) for fields in C.FIELD_SCHEMA.values())
assert (report["Source"] == "User-stated").any()
assert (report["Source"] == "Default").any()
print(f"PASS requirement report: {len(report)} attributes")

# 6. No incompatible filters should be silently ignored.
impossible = A.rule_requirements(
    "Gaming PC, white case, budget RM5000", data
)
impossible["component_preferences"]["case"]["color"] = {"eq": "__not_in_catalogue__"}
r, _ = A.recommend(engine, impossible)
assert not r["ok"]
print("PASS unavailable explicit preference is not silently ignored")

# 7. Too-low budgets are rejected.
req = A.rule_requirements("Gaming PC for RM1", data)
r, _ = A.recommend(engine, req)
assert not r["ok"]
print("PASS infeasible budget rejected")

print("All project smoke tests passed. OPT inference and live Streamlit hosting are not covered.")
