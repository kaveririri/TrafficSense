
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def traffic_trend_chart(df):
    """Display average traffic by hour."""

    hourly = (
        df.groupby("hour", as_index=False)["traffic_volume"]
        .mean()
    )

    fig = px.line(
        hourly,
        x="hour",
        y="traffic_volume",
        markers=True,
        title="Average Traffic by Hour",
        labels={
            "hour": "Hour of Day",
            "traffic_volume": "Average Vehicles per Hour",
        },
    )

    fig.update_layout(template="plotly_dark")

    return fig


def weather_traffic_chart(df):
    """Compare traffic volume across weather conditions."""

    weather = (
        df.groupby("weather_main", as_index=False)
        ["traffic_volume"]
        .mean()
        .sort_values("traffic_volume", ascending=False)
    )

    fig = px.bar(
        weather,
        x="weather_main",
        y="traffic_volume",
        title="Average Traffic by Weather",
        labels={
            "weather_main": "Weather",
            "traffic_volume": "Average Vehicles per Hour",
        },
    )

    fig.update_layout(template="plotly_dark")

    return fig


def scenario_comparison_chart(results):
    """Visualize scenario predictions."""

    fig = px.bar(
        results,
        x="Scenario",
        y="Predicted Traffic",
        color="Scenario",
        title="Scenario Traffic Comparison",
        text="Predicted Traffic",
    )

    fig.update_layout(template="plotly_dark")

    return fig


def actual_vs_predicted_chart(actual, predicted):
    """Compare actual and predicted traffic."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=list(actual),
        mode="lines",
        name="Actual Traffic",
    ))

    fig.add_trace(go.Scatter(
        y=list(predicted),
        mode="lines",
        name="Predicted Traffic",
    ))

    fig.update_layout(
        title="Actual vs Predicted Traffic",
        xaxis_title="Test Observations",
        yaxis_title="Vehicles per Hour",
        template="plotly_dark",
    )

    return fig
