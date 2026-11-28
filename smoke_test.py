"""Fast offline checks. Run: python smoke_test.py (no model download needed)."""
import json
from pathlib import Path
import advisor as A
import nlu_data as N


def main():
    engine = A.load_engine()
    cases = [
        ("Need gaming PC RM4500 16GB RAM 1TB SSD", 16, 1000),
        ("Office PC RM1500 8GB RAM 256GB SSD", 8, 256),
        ("PC RM4000 24GB RAM 1TB SSD", 24, 1000),
        ("RAM: 32GB SSD: 512GB, gaming RM4500", 32, 512),
        ("SSD 512GB RAM:32GB gaming RM4500", 32, 512),
        ("16gb DDR5 RAM with SSD 1 TB under RM4300", 16, 1000),
        ("2x16 GB RAM and 2TB NVMe SSD gaming RM5000", 32, 2000),
        ("Nak PC untuk gaming, bajet RM4500, Ryzen CPU, 16GB RAM dan 1TB SSD", 16, 1000),
        ("Need PC 16g ram and 500gb storage for programming budget 3000", 16, 500),
        ("Office PC 2500 RM, 32 GB DDR5 RAM and 1 TB SSD", 32, 1000),
    ]
    for text, ram, ssd in cases:
        req, *_ = A.analyse_requirements(text, engine.d)
        assert req["ram_gb"] == ram, ("RAM extraction", text, req)
        assert req["storage_gb"] == ssd, ("SSD extraction", text, req)
        assert (req["component_preferences"].get("ram", {}).get("total_gb")
                == {"min": ram}), ("RAM preferences", text)
        assert (req["component_preferences"].get("ssd", {}).get("capacity")
                == {"min": ssd}), ("SSD preferences", text)
        res, violations = A.recommend(engine, req)
        assert res["ok"], ("build failed", text, res.get("message"))
        assert not violations, ("compatibility", text, violations)
        items = {i["category"]: i for i in res["items"]}
        assert items["ram"]["attributes"]["total_gb"] >= ram, text
        assert items["ssd"]["attributes"]["capacity"] >= ssd, text
        assert res["total_myr"] < res["budget_myr"], text

    # Manual review edits must override stale catalogue min-capacity filters.
    old = A.rule_requirements("PC gaming RM4500 32GB RAM 2TB SSD", engine.d)
    old["ram_gb"], old["storage_gb"] = 16, 500
    res, violations = A.recommend(engine, old)
    assert res["ok"] and not violations
    assert res["requirement"].component_preferences["ram"]["total_gb"] == {"min": 16}
    assert res["requirement"].component_preferences["ssd"]["capacity"] == {"min": 500}

    # Budget may not be the reason an impossible catalogue constraint fails.
    req = A.rule_requirements("Office PC RM9000", engine.d)
    req["component_preferences"]["cpu"] = {"name": {"contains": "THIS_CPU_DOES_NOT_EXIST"}}
    result, _ = A.recommend(engine, req)
    assert not result["ok"]
    assert "constraints" in result["message"].lower()

    # Models' raw predictions are preserved separately from confirmed fields.
    # This verifies the online extraction orchestration without downloading OPT.
    import threading
    class FakeOPT:
        def run_extraction(self, _model, _tok, texts, **kwargs):
            assert kwargs.get("prompt_builder") is N.extract_prompt_training
            prediction = {"use_case": "gaming", "budget": 999999,
                          "ram_gb": 256, "storage_gb": 500,
                          "currency": "USD"}
            return [prediction], 0.0, [json.dumps(prediction)]

    dummy = A.ModelSession.__new__(A.ModelSession)
    dummy.O = FakeOPT()
    dummy.lock = threading.RLock()
    dummy.mode = A.FINETUNED
    dummy.model = object()
    dummy.tok = object()
    dummy.shots = None
    req, source, raw = dummy.extract("Gaming PC RM4500 16GB RAM 1TB SSD", engine.d)
    assert req["ram_gb"] == 16 and req["storage_gb"] == 1000
    assert req["budget"] == 4500 and req["currency"] == "MYR"
    assert req["model_prediction"]["ram_gb"] == 256
    assert source == A.FINETUNED

    assert A.adapter_ready("extract")
    modes = A.available_modes()
    assert A.PRETRAINED in modes and A.FINETUNED in modes
    adapter_config = json.loads((A.ROOT / "adapters" / "extract_125m" / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == A.MODEL
    for name in ["opt125m_evaluation.csv", "opt125m_predictions.csv", "opt_raw_predictions.csv"]:
        assert (A.ROOT / "results" / name).is_file()
    print(f"PASS: {len(cases)} build/extraction cases, review override, diagnostics, adapters and saved results.")


if __name__ == "__main__":
    main()
