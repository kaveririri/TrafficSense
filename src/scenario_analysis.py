
import pandas as pd

from .prediction import predict_traffic


def simulate_scenario(pipeline, base_input, scenarios):
    """Compare predicted traffic under different scenarios."""

    results = []

    for scenario_name, changes in scenarios.items():

        modified_input = base_input.copy()
        modified_input.update(changes)

        prediction = predict_traffic(
            pipeline,
            modified_input
        )

        results.append({
            "Scenario": scenario_name,
            "Predicted Traffic": prediction,
        })

    return pd.DataFrame(results)


def compare_scenarios(pipeline, base_input, scenarios):
    """Calculate traffic changes relative to baseline."""

    results = simulate_scenario(
        pipeline,
        base_input,
        scenarios
    )

    baseline = predict_traffic(
        pipeline,
        base_input
    )

    results["Difference"] = (
        results["Predicted Traffic"] - baseline
    )

    results["Percentage Change"] = (
        results["Difference"] / max(baseline, 1) * 100
    ).round(2)

    return baseline, results
