import plotly.graph_objects as go

GREEN = "#2ECC71"
YELLOW = "#F1C40F"
RED = "#E74C3C"
BLUE = "#3498DB"

TRANSPARENT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def _score_color(score):
    return GREEN if score > 0.7 else YELLOW if score >= 0.5 else RED


def trust_score_bar_chart(results):
    titles = [r.get("title", "")[:40] for r in results]
    scores = [r.get("trust_score", 0.0) for r in results]
    colors = [_score_color(s) for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=titles, orientation="h",
        marker=dict(color=colors),
        text=[f"{s * 100:.1f}%" for s in scores], textposition="outside",
    ))
    fig.update_layout(
        title="Trust Score by Result",
        xaxis=dict(range=[0, 1], title="Trust Score"),
        height=max(240, 40 * len(titles)),
        margin=dict(l=20, r=20, t=50, b=20),
        **TRANSPARENT_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def shap_waterfall_chart(shap_values, feature_names, base_value):
    if isinstance(shap_values, dict):
        feature_names = list(shap_values.keys())
        shap_values = list(shap_values.values())

    order = sorted(range(len(shap_values)), key=lambda i: -abs(shap_values[i]))
    labels = ["Base value"] + [feature_names[i] for i in order] + ["Result"]
    values = [base_value] + [shap_values[i] for i in order] + [base_value + sum(shap_values)]
    measures = ["absolute"] + ["relative"] * len(order) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=BLUE)),
        connector=dict(line=dict(color="#94A3B8")),
    ))
    fig.update_layout(
        title="Feature Contribution Waterfall",
        height=380, margin=dict(l=20, r=20, t=50, b=20),
        **TRANSPARENT_LAYOUT,
    )
    return fig


def prediction_donut_chart(label, confidence):
    remainder = max(0.0, 1 - confidence)
    color = _score_color(confidence)

    fig = go.Figure(go.Pie(
        values=[confidence, remainder],
        hole=0.7,
        marker=dict(colors=[color, "#334155"]),
        textinfo="none",
        sort=False,
        direction="clockwise",
    ))
    fig.update_layout(
        title=f"Prediction: {label}",
        annotations=[dict(text=f"{confidence * 100:.1f}%", x=0.5, y=0.5, font_size=28, showarrow=False)],
        showlegend=False, height=280, margin=dict(l=20, r=20, t=50, b=20),
        **TRANSPARENT_LAYOUT,
    )
    return fig


def feature_importance_chart(importances, feature_names, top_n=8):
    pairs = sorted(zip(feature_names, importances), key=lambda p: -abs(p[1]))[:top_n]
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=BLUE)))
    fig.update_layout(
        title="Top Feature Importances",
        height=320, margin=dict(l=20, r=20, t=50, b=20),
        **TRANSPARENT_LAYOUT,
    )
    return fig


def risk_gauge(risk_pct):
    value = risk_pct * 100 if risk_pct <= 1 else risk_pct
    bar_color = GREEN if value <= 30 else YELLOW if value <= 60 else RED

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%"},
        title={"text": "Risk Level"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 30], "color": "#DCFCE7"},
                {"range": [30, 60], "color": "#FEF3C7"},
                {"range": [60, 100], "color": "#FEE2E2"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def precision_recall_chart(precision_vals, recall_vals):
    x = list(range(1, len(precision_vals) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=precision_vals, name="Precision", mode="lines",
                              line=dict(color=BLUE), fill="tozeroy"))
    fig.add_trace(go.Scatter(x=x, y=recall_vals, name="Recall", mode="lines",
                              line=dict(color=GREEN), fill="tozeroy"))
    fig.update_layout(
        title="Precision / Recall @ K",
        xaxis_title="K", yaxis=dict(range=[0, 1], title="Score"),
        height=340, margin=dict(l=20, r=20, t=50, b=20),
        **TRANSPARENT_LAYOUT,
    )
    return fig
