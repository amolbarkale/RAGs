# Heatmap generation and display

import plotly.graph_objects as go

def create_heatmap(similarity_matrix, text_labels):
    """Create interactive heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=text_labels,
        y=text_labels,
        colorscale='RdYlBu_r',
        text=similarity_matrix,
        texttemplate='%{text}%',
        textfont={"size": 12},
        colorbar=dict(title="Similarity %")
    ))

    fig.update_layout(
        title="Similarity Matrix Heatmap",
        xaxis_title="Texts",
        yaxis_title="Texts",
        width=600,
        height=500
    )

    return fig
