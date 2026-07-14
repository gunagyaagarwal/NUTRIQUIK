import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_trust_gauge(score: float, title="Composite Trust Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#0F172A"}},
        title={'text': title, 'font': {'size': 18, 'color': '#334155'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#2563EB" if score >= 0.7 else ("#F59E0B" if score >= 0.5 else "#EF4444")},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 50], 'color': '#FEE2E2'},
                {'range': [50, 75], 'color': '#FEF3C7'},
                {'range': [75, 100], 'color': '#DCFCE7'}
            ],
            'threshold': {'line': {'color': "#DC2626", 'width': 3}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def create_evidence_pyramid():
    levels = ['Level I: Meta-Analyses & RCTs', 'Level II: Cohort & Control Studies',
              'Level III: In-Vitro & Animal Models', 'Level IV: Expert Opinion & Blogs']
    values = [40, 30, 20, 10]
    colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
    fig = go.Figure(go.Funnel(
        y=levels, x=values, textinfo="value+percent initial",
        marker={"color": colors, "line": {"width": [2, 2, 2, 2], "color": ["#047857", "#1D4ED8", "#B45309", "#B91C1C"]}}
    ))
    fig.update_layout(
        title="Oxford Centre for Evidence-Based Medicine Pyramid",
        showlegend=False, height=320, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_retrieval_scatter(results):
    df = pd.DataFrame(results)
    fig = px.scatter(
        df, x="bm25_score", y="tfidf_cosine",
        size=[max(12, t * 25) for t in df["trust_score"]],
        color="trust_score", hover_name="title",
        hover_data=["doc_id", "is_rejected"],
        labels={"bm25_score": "BM25 Lexical Score", "tfidf_cosine": "TF-IDF Cosine Similarity", "trust_score": "Trust Score"},
        color_continuous_scale="RdYlGn", range_color=[0.0, 1.0],
        title="Retrieval Map: BM25 Lexical vs TF-IDF Cosine vs Trust Score"
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def create_shap_chart(shap_dict, title, chart_title="Feature Attribution"):
    features = list(shap_dict.keys())
    values = list(shap_dict.values())
    colors = ['#10B981' if v >= 0 else '#EF4444' for v in values]
    fig = go.Figure(go.Bar(x=values, y=features, orientation='h', marker=dict(color=colors)))
    fig.update_layout(
        title=f"{chart_title}: {title[:45]}",
        xaxis_title="Contribution", height=300,
        margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_ir_metrics_chart(ir_results):
    metrics = ['Precision@5', 'Recall@5', 'MAP', 'NDCG@5']
    scores = [ir_results.get('Precision@5', 0), ir_results.get('Recall@5', 0),
              ir_results.get('MAP', 0), ir_results.get('NDCG@5', 0)]
    fig = go.Figure(go.Bar(x=metrics, y=scores, marker_color='#10B981'))
    fig.update_layout(
        title="BM25 Information Retrieval Evaluation (real ground-truth queries)",
        yaxis=dict(range=[0, 1.0], title="Score"),
        height=340, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_ml_metrics_chart(ml_df):
    fig = go.Figure()
    for metric, color in [("accuracy", "#10B981"), ("f1", "#3B82F6"), ("precision", "#F59E0B"), ("recall", "#94A3B8")]:
        fig.add_trace(go.Bar(x=ml_df["model"], y=ml_df[metric], name=metric, marker_color=color))
    fig.update_layout(
        barmode='group', title="XGBoost Disease Model Evaluation (held-out test split)",
        yaxis=dict(range=[0, 1.0], title="Score"), height=380,
        margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_trust_comparison_chart(results, rejected):
    all_docs = list(results) + list(rejected)
    all_docs.sort(key=lambda d: d["trust_score"], reverse=True)

    titles = [d.get("title", "")[:40] for d in all_docs]
    scores = [d["trust_score"] for d in all_docs]
    colors = ['#10B981' if s >= 0.75 else ('#F59E0B' if s >= 0.5 else '#EF4444') for s in scores]
    opacities = [1.0 if s >= 0.5 else 0.45 for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=titles, orientation='h',
        marker=dict(color=colors, opacity=opacities),
        text=[f"{s * 100:.1f}%" for s in scores], textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#94A3B8",
                  annotation_text="Guardrail cutoff (0.50)", annotation_position="top")
    fig.update_layout(
        title="Trust Score Comparison — Passed vs Guardrail-Rejected",
        xaxis=dict(range=[0, 1], title="Trust Score"),
        showlegend=False,
        height=max(280, 40 * len(titles)),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def create_interaction_matrix_chart(interactions):
    df = pd.DataFrame(interactions)
    colors = []
    for t in df["type"]:
        if t == "Synergy":
            colors.append("#DCFCE7")
        elif t == "Contraindication":
            colors.append("#FEE2E2")
        else:
            colors.append("#FEF3C7")
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Nutrient / Agent", "Biological Target", "Interaction Type", "Clinical Mechanism"],
                    fill_color='#1E293B', font=dict(color='white', size=13), align='left'),
        cells=dict(values=[df['nutrient'], df['target'], df['type'], df['detail']],
                   fill_color=[colors, colors, colors, colors], font=dict(color='#0F172A', size=12), align='left'))
    ])
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
    return fig
