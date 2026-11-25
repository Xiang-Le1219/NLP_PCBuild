"""Regression tests for budget-constrained build selection."""
import advisor as A
import pc_engine as E


def main():
    engine = A.load_engine()
    scenarios = []
    for use_case in E.USE_CASES:
        for budget in (5000, 10000):
            for strategy in ("closest_to_budget", "best_value"):
                req = {
                    "use_case": use_case, "budget": budget, "currency": "MYR",
                    "cpu_brand": "any", "gpu_brand": "any",
                    "ram_gb": None, "storage_gb": None, "extras": [],
                    "compact": False, "budget_strategy": strategy,
                }
                result, violations = A.recommend(engine, req)
                assert result["ok"], (use_case, budget, strategy, result)
                assert result["total_myr"] <= result["budget_myr"] + 1e-6, (
                    use_case, budget, strategy, result["total_myr"], result["budget_myr"]
                )
                assert not violations, (use_case, budget, strategy, violations)
                scenarios.append((use_case, budget, strategy, result["total_myr"]))

    # In closest-to-budget mode, a higher gaming budget should select a more
    # expensive build when the catalogue offers suitable options.
    gaming = {(b, s): total for uc, b, s, total in scenarios if uc == "gaming"}
    assert gaming[(10000, "closest_to_budget")] > gaming[(5000, "closest_to_budget")]

    # For a light workload the two strategies should be meaningfully distinct.
    office = {(b, s): total for uc, b, s, total in scenarios if uc == "office"}
    assert office[(10000, "closest_to_budget")] > office[(10000, "best_value")]

    print(f"PASS: {len(scenarios)} budget strategy scenarios")
    print("Gaming RM5,000 closest:", round(gaming[(5000, "closest_to_budget")], 2))
    print("Gaming RM10,000 closest:", round(gaming[(10000, "closest_to_budget")], 2))
    print("Office RM10,000 closest:", round(office[(10000, "closest_to_budget")], 2))
    print("Office RM10,000 best value:", round(office[(10000, "best_value")], 2))


if __name__ == "__main__":
    main()
