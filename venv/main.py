# =============================================================
# fuzzy_relations_visual_app.py  —  v3
# -------------------------------------------------------------
# Streamlit application for interactive exploration of
# fuzzy binary relations: visualisation, algebraic operations,
# property checks, α‑срезы, Warshall–Kosko closure, independence
# of projections (type‑1 and type‑2).
# =============================================================
"""
Usage
-----
    pip install streamlit numpy pandas altair networkx plotly pandas
    streamlit run fuzzy_relations_visual_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------
# Core algebra over fuzzy relations
# -------------------------------------------------------------

def fuzzy_union(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Element‑wise maximum."""
    assert R.shape == S.shape, "Shapes mismatch"
    return np.maximum(R, S)

def fuzzy_intersection(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Element‑wise minimum."""
    assert R.shape == S.shape, "Shapes mismatch"
    return np.minimum(R, S)

def fuzzy_complement(R: np.ndarray) -> np.ndarray:
    return 1.0 - R

# ---------------- composition & closure ----------------------

def maxmin_comp(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Max–min composition."""
    return np.maximum.reduce(np.minimum(R[:, None, :], S[None, :, :]), axis=2)

def warshall_kosko_closure(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Warshall–Kosko transitive closure under max–min t‑norm."""
    T = R.copy()
    while True:
        T_next = np.maximum(T, maxmin_comp(T, T))
        if np.allclose(T, T_next, atol=eps):
            return T_next
        T = T_next

# ---------------- projections & α‑cuts -----------------------

def vertical_projection(R: np.ndarray) -> np.ndarray:
    return R.max(axis=1)

def horizontal_projection(R: np.ndarray) -> np.ndarray:
    return R.max(axis=0)

def alpha_cut(R: np.ndarray, alpha: float, crisp: bool = True) -> np.ndarray:
    """Return α‑срез. If crisp=True → 0/1 matrix, else values < α are set to 0."""
    if crisp:
        return (R >= alpha).astype(float)
    out = R.copy()
    out[out < alpha] = 0.0
    return out

# -------------------------------------------------------------
# Property checks
# -------------------------------------------------------------

TOL = 1e-6


def is_reflexive(R):
    return np.allclose(np.diag(R), 1.0, atol=TOL)

def is_antireflexive(R):
    return np.allclose(np.diag(R), 0.0, atol=TOL)

def is_symmetric(R):
    return np.allclose(R, R.T, atol=TOL)

def is_antisymmetric(R):
    diff = np.minimum(R, R.T)
    np.fill_diagonal(diff, 0.0)
    return np.all(diff <= TOL)

def is_transitive(R):
    return np.all(maxmin_comp(R, R) <= R + TOL)

def is_strongly_complete(R):
    return np.all(np.maximum(R, R.T) >= 1 - TOL)

def is_weakly_complete(R):
    n = R.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.all(np.maximum(R, R.T)[mask] > TOL)

# -------- independence of projections -----------------------

def independence_type1(R):
    """Type‑1: μ(x,y) == min(projX(x), projY(y))"""
    vx = vertical_projection(R)[:, None]
    vy = horizontal_projection(R)[None, :]
    return np.allclose(R, np.minimum(vx, vy), atol=TOL)

def independence_type2(R):
    """Type‑2: μ(x,y) == projX(x) * projY(y) (product)."""
    vx = vertical_projection(R)[:, None]
    vy = horizontal_projection(R)[None, :]
    return np.allclose(R, vx * vy, atol=TOL)

PROPERTY_FUNCS = {
    "Reflexive": is_reflexive,
    "Antireflexive": is_antireflexive,
    "Symmetric": is_symmetric,
    "Antisymmetric": is_antisymmetric,
    "Max–min transitive": is_transitive,
    "Strongly complete": is_strongly_complete,
    "Weakly complete": is_weakly_complete,
    "Independent (type‑1 min)": independence_type1,
    "Independent (type‑2 product)": independence_type2,
}

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------

def load_relation(file) -> np.ndarray:
    return pd.read_csv(file, header=None).values.astype(float)

def generate_random_relation(n, m, seed=None):
    rng = np.random.default_rng(seed)
    return rng.random((n, m)).round(2)

# -------------------------------------------------------------
# Visual helpers
# -------------------------------------------------------------

def heatmap(matrix: np.ndarray, title: str) -> alt.Chart:
    df = pd.DataFrame(matrix)
    df = df.reset_index().melt('index', var_name='col', value_name='value')
    df.rename(columns={'index': 'row'}, inplace=True)
    return alt.Chart(df, height=300, width=300).mark_rect().encode(
        x='col:O', y='row:O', color=alt.Color('value:Q', scale=alt.Scale(domain=[0, 1])),
        tooltip=['row', 'col', 'value']
    ).properties(title=title)


def relation_graph(R: np.ndarray, alpha: float = 0.0) -> go.Figure:
    G = nx.DiGraph()
    n, m = R.shape
    for i in range(n):
        for j in range(m):
            if R[i, j] >= alpha:
                G.add_edge(f'x{i}', f'y{j}', weight=R[i, j])
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    node_x = [pos[node][0] for node in G.nodes]
    node_y = [pos[node][1] for node in G.nodes]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1), hoverinfo='none')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes),
                            marker=dict(size=12, color='LightSkyBlue'), textposition='top center')
    return go.Figure([edge_trace, node_trace]).update_layout(title=f'Graph (α ≥ {alpha})', showlegend=False)

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------

def main():
    st.set_page_config('Fuzzy Relation Explorer', layout='wide')
    st.title('🔍 Fuzzy Relation Explorer')

    # ---------- Data input ----------
    st.sidebar.header('Data input')
    upload = st.sidebar.file_uploader('Upload CSV matrix', type='csv')
    if upload is not None:
        R = load_relation(upload)
    else:
        n = st.sidebar.number_input('Rows (X)', 2, 100, 6)
        m = st.sidebar.number_input('Cols (Y)', 2, 100, 6)
        seed = st.sidebar.number_input('Seed', value=42)
        if st.sidebar.button('Generate random') or 'R' not in st.session_state:
            st.session_state['R'] = generate_random_relation(n, m, int(seed))
        R = st.session_state['R']

    st.subheader('Matrix μ_R')
    st.altair_chart(heatmap(R, 'μ_R'), use_container_width=True)

    # ---------- α‑cut view ----------
    alpha_display = st.slider('α‑threshold for display & graph', 0.0, 1.0, 0.0, 0.05)
    st.altair_chart(heatmap(alpha_cut(R, alpha_display, crisp=True), f'α‑cut (α={alpha_display})'), use_container_width=True)

    # ---------- Property checks ----------
    with st.expander('Check properties'):
        for name, func in PROPERTY_FUNCS.items():
            st.write(f"{name}: {'✅' if func(R) else '❌'}")

    # ---------- Operations ----------
    st.sidebar.header('Operations')
    op_choice = st.sidebar.selectbox('Choose', ['None', 'Complement', 'Union', 'Intersection'])
    R_op: Optional[np.ndarray] = None
    if op_choice == 'Complement':
        R_op = fuzzy_complement(R)
    elif op_choice in {'Union', 'Intersection'}:
        upload2 = st.sidebar.file_uploader('Second matrix (same shape)', key='m2')
        if upload2 is not None:
            S = load_relation(upload2)
            if S.shape != R.shape:
                st.sidebar.error('Shape mismatch')
            else:
                R_op = fuzzy_union(R, S) if op_choice == 'Union' else fuzzy_intersection(R, S)
    if R_op is not None:
        st.subheader(f'Result of {op_choice}')
        st.altair_chart(heatmap(R_op, op_choice), use_container_width=True)

    # ---------- Closure ----------
    if st.checkbox('Compute Warshall–Kosko closure (square matrices only)', disabled=R.shape[0]!=R.shape[1]):
        R_star = warshall_kosko_closure(R)
        st.subheader('Transitive closure μ_R*')
        st.altair_chart(heatmap(R_star, 'μ_R*'), use_container_width=True)

    # ---------- Graph ----------
    st.subheader('Graph view')
    st.plotly_chart(relation_graph(R, alpha_display), use_container_width=True)

    # ---------- Projections ----------
    st.subheader('Projections')
    col1, col2 = st.columns(2)
    col1.bar_chart(vertical_projection(R))
    col2.bar_chart(horizontal_projection(R))

    # ---------- Export ----------
    csv_bytes = pd.DataFrame(R).to_csv(index=False, header=False).encode()
    st.sidebar.download_button('Download matrix', csv_bytes, 'relation.csv', 'text/csv')


if __name__ == '__main__':
    main()
