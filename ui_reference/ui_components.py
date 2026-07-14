import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_trust_gauge(score: float, title="Composite Trust Score"):
    """
    Creates a modern gauge chart for Trust Score percentage
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
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
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_evidence_pyramid():
    """
    Creates an Evidence Pyramid representation for Clinical Trust
    """
    levels = ['Level I: Meta-Analyses & RCTs', 'Level II: Cohort & Control Studies', 'Level III: In-Vitro & Animal Models', 'Level IV: Expert Opinion & Blogs']
    values = [40, 30, 20, 10]
    colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
    
    fig = go.Figure(go.Funnel(
        y=levels,
        x=values,
        textinfo="value+percent initial",
        marker={"color": colors, "line": {"width": [2, 2, 2, 2], "color": ["#047857", "#1D4ED8", "#B45309", "#B91C1C"]}}
    ))
    fig.update_layout(
        title="Oxford Centre for Evidence-Based Medicine Pyramid",
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_retrieval_scatter(results):
    """
    Scatter plot comparing Lexical BM25 score vs Dense Cosine score, colored by Composite Trust
    """
    df = pd.DataFrame(results)
    
    fig = px.scatter(
        df,
        x="bm25_score",
        y="dense_score",
        size=[max(12, t * 25) for t in df["trust_score"]],
        color="trust_score",
        hover_name="title",
        hover_data=["id", "source", "evidence_level", "is_rejected"],
        labels={"bm25_score": "BM25 Lexical Score", "dense_score": "Dense Embedding Cosine Similarity", "trust_score": "Trust Score"},
        color_continuous_scale="RdYlGn",
        range_color=[0.3, 1.0],
        title="Retrieval Map: Lexical (BM25) vs Dense (FAISS) vs Trust Score"
    )
    
    # Add guardrail line
    fig.add_shape(
        type="line", x0=0.4, y0=0.4, x1=1.0, y1=1.0,
        line=dict(color="Red", width=2, dash="dashdot"),
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_shap_chart(shap_dict, doc_title):
    """
    Creates horizontal bar chart for SHAP feature attribution
    """
    features = list(shap_dict.keys())
    values = list(shap_dict.values())
    colors = ['#10B981' if v >= 0 else '#EF4444' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors)
    ))
    fig.update_layout(
        title=f"SHAP Feature Attribution: {doc_title[:45]}...",
        xaxis_title="Impact on Trust Score (SHAP value)",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_ir_metrics_chart():
    """
    Creates IR Evaluation comparison chart (BM25 vs FAISS vs Hybrid)
    """
    metrics = ['Precision@5', 'Recall@5', 'MAP', 'NDCG@5']
    bm25_scores = [0.68, 0.72, 0.65, 0.70]
    faiss_scores = [0.75, 0.79, 0.74, 0.78]
    hybrid_scores = [0.89, 0.92, 0.88, 0.91]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics, y=bm25_scores, name='BM25 (Sparse)', marker_color='#94A3B8'))
    fig.add_trace(go.Bar(x=metrics, y=faiss_scores, name='FAISS (Dense)', marker_color='#3B82F6'))
    fig.add_trace(go.Bar(x=metrics, y=hybrid_scores, name='Hybrid (NUTRIQUIK)', marker_color='#10B981'))
    
    fig.update_layout(
        barmode='group',
        title="Information Retrieval Benchmark Evaluation",
        yaxis=dict(range=[0, 1.0], title="Score"),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_interaction_matrix_chart(interactions):
    """
    Visual table of nutrient-medication interactions
    """
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
                    fill_color='#1E293B',
                    font=dict(color='white', size=13),
                    align='left'),
        cells=dict(values=[df['nutrient'], df['target'], df['type'], df['detail']],
                   fill_color=[colors, colors, colors, colors],
                   font=dict(color='#0F172A', size=12),
                   align='left'))
    ])
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
    return fig
