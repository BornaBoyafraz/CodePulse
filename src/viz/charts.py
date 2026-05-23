"""Chart data builders for the CodePulse frontend.

All functions return Plotly Figure JSON strings suitable for direct use with
Plotly.js on the frontend. Backgrounds are transparent so they inherit the
site's dark theme.
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


def _shorten_path(file_path: str) -> str:
    """Return the last two path segments joined with '/'.

    Examples:
        'src/features/churn.py' → 'features/churn.py'
        'churn.py'              → 'churn.py'
    """
    parts = Path(file_path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else file_path


def _risk_color(score: float) -> str:
    if score < 40:
        return "#3fb950"
    if score < 70:
        return "#d29922"
    return "#f85149"


def build_risk_barchart(files: list[dict]) -> str:
    """Build a horizontal bar chart of the top-10 riskiest files.

    Args:
        files: List of dicts with at least 'file_path' and 'risk_score' keys.

    Returns:
        Plotly Figure JSON string.
    """
    top10 = sorted(files, key=lambda f: f["risk_score"], reverse=True)[:10]
    # Reverse so highest risk appears at the top of the horizontal chart
    top10 = list(reversed(top10))

    labels = [_shorten_path(f["file_path"]) for f in top10]
    scores = [f["risk_score"] for f in top10]
    colors = [_risk_color(f["risk_score"]) for f in top10]

    trace = go.Bar(
        orientation="h",
        x=scores,
        y=labels,
        marker=dict(color=colors, opacity=0.9, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.1f}<extra></extra>",
        hoverlabel=dict(
            bgcolor="#161b22",
            font=dict(family="JetBrains Mono", size=12, color="#e6edf3"),
            bordercolor="#30363d",
        ),
    )

    n = len(top10)
    shapes = [
        dict(
            type="line", x0=40, x1=40, y0=-0.5, y1=n - 0.5,
            line=dict(color="#d29922", width=1, dash="dot"),
        ),
        dict(
            type="line", x0=70, x1=70, y0=-0.5, y1=n - 0.5,
            line=dict(color="#f85149", width=1, dash="dot"),
        ),
    ]

    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=200, r=40, t=20, b=40),
        xaxis=dict(
            range=[0, 100],
            tickfont=dict(family="Inter", size=11, color="#8b949e"),
            gridcolor="#21262d",
            linecolor="#30363d",
            title=dict(text="Risk Score", font=dict(family="Inter", size=11, color="#8b949e")),
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(family="JetBrains Mono", size=11, color="#e6edf3"),
            linecolor="#30363d",
            fixedrange=True,
            automargin=True,
        ),
        bargap=0.35,
        height=max(300, n * 44 + 60),
        font=dict(family="Inter", color="#e6edf3"),
        shapes=shapes,
    )

    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_json(fig)


def build_risk_treemap(files: list[dict]) -> str:
    """Build a treemap where each rectangle represents a file coloured by risk.

    Args:
        files: List of dicts with at least 'file_path' and 'risk_score' keys.

    Returns:
        Plotly Figure JSON string.
    """
    labels = [_shorten_path(f["file_path"]) for f in files]
    customdata = [f["risk_score"] for f in files]

    trace = go.Treemap(
        labels=labels,
        parents=[""] * len(files),
        values=[1] * len(files),
        customdata=customdata,
        marker=dict(
            colors=customdata,
            colorscale=[
                [0.0, "#3fb950"],
                [0.4, "#d29922"],
                [0.7, "#f85149"],
                [1.0, "#8b1c13"],
            ],
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="Risk", font=dict(color="#8b949e", size=11)),
                tickfont=dict(color="#8b949e", size=10),
                thickness=12,
            ),
        ),
        texttemplate="<b>%{label}</b><br>%{customdata:.0f}",
        hovertemplate="<b>%{label}</b><br>Risk Score: %{customdata:.1f}<extra></extra>",
        hoverlabel=dict(
            bgcolor="#161b22",
            font=dict(family="JetBrains Mono", size=12, color="#e6edf3"),
        ),
    )

    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family="Inter", color="#e6edf3"),
    )

    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_json(fig)


def build_pr_curve(pr_curve_json: str) -> str:
    """Format the evaluator PR curve JSON for Plotly.js consumption.

    Ensures paper/plot bgcolor is transparent so it fits the dark theme.
    Returns the input unchanged if it is empty or the null sentinel '{}' .

    Args:
        pr_curve_json: Plotly Figure JSON string from evaluate_model.

    Returns:
        Updated Plotly Figure JSON string, or '{}' if input is empty.
    """
    if not pr_curve_json or pr_curve_json.strip() in ("{}", ""):
        return "{}"

    try:
        fig = pio.from_json(pr_curve_json)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return pio.to_json(fig)
    except Exception:
        return pr_curve_json
