"""Regression tests for RAM/SSD extraction, review reporting, and build use."""
import json
import advisor as A
import nlu_data as N

PARSER_CASES = [
    ("Need gaming PC with 16GB RAM and 512GB SSD", 16, 512),
    ("Office PC RM1500 8GB RAM 256GB SSD", 8, 256),
    ("RAM 32GB SSD 1TB", 32, 1000),
    ("RAM: 32GB; SSD: 512GB", 32, 512),
    ("SSD 512GB, RAM: 32GB", 32, 512),
    ("RAM is 32GB and SSD is 1TB", 32, 1000),
    ("RAM should be 32GB and storage should be 2TB", 32, 2000),
    ("at least 32GB RAM and at least 2TB storage", 32, 2000),
    ("I have 16 GB of RAM and 512 GB of storage", 16, 512),
    ("Need 2x16GB RAM and 1TB NVMe SSD", 32, 1000),
    ("Need 2 x 16 GB RAM and 1 TB SSD", 32, 1000),
    ("RAM 2x16GB, SSD 1TB", 32, 1000),
    ("GPU has 16GB VRAM; system RAM 32GB; SSD 1TB", 32, 1000),
    ("RTX 4070 12GB, 32GB RAM, 1TB NVMe", 32, 1000),
    ("32 gigs memory and one terabyte SSD", 32, 1000),
    ("DDR5 6000 32GB RAM, SSD capacity 1TB", 32, 1000),
    ("32GB 内存，1TB 固态硬盘", 32, 1000),
    ("RTX 4070 has 16GB VRAM", None, None),
    ("Gaming PC with 12GB GPU memory, no RAM or SSD specified", None, None),
]

BUILD_CASES = [
    ("Office PC RM1500 8GB RAM 256GB SSD", 8, 256),
    ("Gaming PC RM4500, SSD 512GB, RAM: 32GB", 32, 512),
    ("Gaming PC RM6000 with 2 x 16GB RAM and 1TB SSD", 32, 1000),
    ("Office PC RM3500, 32GB 内存，1TB 固态硬盘", 32, 1000),
]


def main():
    engine = A.load_engine()
    parser_results = []
    for text, expected_ram, expected_ssd in PARSER_CASES:
        ram, ssd = N.extract_memory_storage(text)
        assert (ram, ssd) == (expected_ram, expected_ssd), (text, ram, ssd)
        req, source, raw, report = A.analyse_requirements(text, engine.d)
        assert req["ram_gb"] == expected_ram and req["storage_gb"] == expected_ssd
        fields = report[report["Dataset attribute"].str.contains(
            "RAM total capacity|Storage capacity", case=False, regex=True
        )]
        sources = fields.set_index("Dataset attribute")["Source"].to_dict()
        if expected_ram is not None:
            assert sources.get("RAM total capacity (derived)") == "User-stated"
        if expected_ssd is not None:
            assert sources.get("Storage capacity (GB)") == "User-stated"
        parser_results.append({
            "input": text, "expected_ram_gb": expected_ram,
            "extracted_ram_gb": ram, "expected_storage_gb": expected_ssd,
            "extracted_storage_gb": ssd, "field_report_user_stated": True,
            "passed": True,
        })

    build_results = []
    for text, expected_ram, expected_ssd in BUILD_CASES:
        req, *_ = A.analyse_requirements(text, engine.d)
        result, violations = A.recommend(engine, req)
        assert result["ok"], (text, result.get("message"))
        selected = {x["category"]: x for x in result["items"]}
        ram_selected = float(selected["ram"]["attributes"]["total_gb"])
        ssd_selected = float(selected["ssd"]["attributes"]["capacity"])
        assert ram_selected >= expected_ram, (text, ram_selected)
        assert ssd_selected >= expected_ssd, (text, ssd_selected)
        assert not violations, (text, violations)
        build_results.append({
            "input": text, "required_ram_gb": expected_ram,
            "selected_ram_gb": ram_selected, "required_storage_gb": expected_ssd,
            "selected_storage_gb": ssd_selected,
            "total_myr": result["total_myr"], "budget_myr": result["budget_myr"],
            "compatibility_violations": len(violations), "passed": True,
        })

    # Simulate values corrected by a user in the Review core requirements form.
    req = A.rule_requirements("Gaming PC RM4500 32GB RAM 2TB SSD", engine.d)
    req.update({"ram_gb": 16, "storage_gb": 500})
    result, violations = A.recommend(engine, req)
    assert result["ok"] and not violations
    assert result["requirement"].component_preferences["ram"]["total_gb"] == {"min": 16}
    assert result["requirement"].component_preferences["ssd"]["capacity"] == {"min": 500}

    import pandas as pd
    pd.DataFrame(parser_results).to_csv("results/ram_ssd_parser_regression.csv", index=False)
    pd.DataFrame(build_results).to_csv("results/ram_ssd_build_regression.csv", index=False)
    print(f"PASS: {len(parser_results)} parser/review cases, {len(build_results)} end-to-end build cases, and manual review override.")


if __name__ == "__main__":
    main()
