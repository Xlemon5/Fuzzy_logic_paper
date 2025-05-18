# =============================================================
# main.py  —  Fuzzy Relation Explorer v4.1
# -------------------------------------------------------------
# Streamlit application for interactive exploration of
# fuzzy binary relations: visualisation, algebraic operations,
# α‑cuts, Warshall–Kosko closure, projection independence.
# -------------------------------------------------------------
# 2025‑05‑18: fix max–min composition broadcasting bug for
#             rectangular matrices (m×n  ∘ n×p → m×p).
# =============================================================
"""
Run:
    pip install streamlit numpy pandas altair networkx plotly pandas
    streamlit run main.py
"""
from __future__ import annotations

from typing import Dict, Optional
import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------ algebra ----------------------

def fuzz_union(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == B.shape, "Union: shapes must match"
    return np.maximum(A, B)

def fuzz_intersection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == B.shape, "Intersection: shapes must match"
    return np.minimum(A, B)

def fuzz_complement(A: np.ndarray) -> np.ndarray:
    return 1.0 - A

# ---- correct max–min composition (m×n, n×p) -----

def maxmin_comp(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Max–min composition R ∘ S.
    A: m×n,  B: n×p  →  C: m×p
    C_{ij} = max_k min(A_{ik}, B_{kj})"""
    assert A.shape[1] == B.shape[0], "Composition: inner dimensions mismatch"
    # Broadcast to (m, n, p):
    min_tensor = np.minimum(A[:, :, None], B[None, :, :])  # (m,n,p)
    return np.maximum.reduce(min_tensor, axis=1)           # (m,p)

def warshall_kosko(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    T = A.copy()
    while True:
        T_next = np.maximum(T, maxmin_comp(T, T))
        if np.allclose(T, T_next, atol=eps):
            return T_next
        T = T_next

# ---------------- projections --------------------

def v_proj(A: np.ndarray) -> np.ndarray:
    return A.max(axis=1)

def h_proj(A: np.ndarray) -> np.ndarray:
    return A.max(axis=0)

def alpha_cut(A: np.ndarray, a: float, crisp: bool = True) -> np.ndarray:
    out = A.copy()
    out[out < a] = 0.0
    return (out > 0).astype(float) if crisp else out

# -------------- visual helpers ------------------

def heatmap(mat: np.ndarray, title: str) -> alt.Chart:
    df = pd.DataFrame(mat)
    melted = df.reset_index().melt('index', var_name='col', value_name='val')
    melted.rename(columns={'index': 'row'}, inplace=True)
    return (alt.Chart(melted, width=300, height=300)
            .mark_rect()
            .encode(x='col:O', y='row:O', color=alt.Color('val:Q', scale=alt.Scale(domain=[0, 1])),
                    tooltip=['row', 'col', 'val'])
            .properties(title=title))

def graph_view(A: np.ndarray, a: float = 0.0) -> go.Figure:
    G = nx.DiGraph()
    n, m = A.shape
    for i in range(n):
        for j in range(m):
            if A[i, j] >= a:
                G.add_edge(f'x{i}', f'y{j}', weight=A[i, j])
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    return (go.Figure([
                go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1)),
                go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes),
                           marker=dict(size=12, color='LightSkyBlue'), textposition='top center')])
            .update_layout(showlegend=False, title=f'Graph (α ≥ {a})'))

# -------------- utilities -----------------------

def parse_text_matrix(txt: str) -> np.ndarray:
    rows = [list(filter(None, line.replace(',', ' ').split())) for line in txt.strip().splitlines() if line.strip()]
    arr = np.array(rows, dtype=float)
    if not ((0 <= arr).all() and (arr <= 1).all()):
        raise ValueError('Values must be in [0,1]')
    return arr

def load_csv(upload) -> np.ndarray:
    return pd.read_csv(upload, header=None).values.astype(float)

# -------------- Streamlit UI --------------------

def init_state():
    if 'matrices' not in st.session_state:
        st.session_state['matrices']: Dict[str, np.ndarray] = {}
        st.session_state['matrices']['R1'] = np.array([[0.1, 0.7, 0.4],
                                                       [1.0, 0.5, 0.0]])
        st.session_state['matrices']['R2'] = np.array([[0.9, 0.0, 1.0, 0.2],
                                                       [0.3, 0.6, 0.0, 0.9],
                                                       [0.1, 1.0, 0.0, 0.5]])

init_state()

st.set_page_config('Fuzzy Relation Explorer', layout='wide')
st.title('🔍 Fuzzy Relation Explorer')

# ----- Sidebar matrix manager -----
st.sidebar.header('📂 Matrix manager')
mat_names = list(st.session_state['matrices'].keys())
selected = st.sidebar.selectbox('Select matrix', mat_names)
A = st.session_state['matrices'][selected]

with st.sidebar.expander('➕ Add / replace matrix'):
    new_name = st.text_input('Matrix name', value=f'M{len(mat_names)+1}')
    method = st.radio('Input method', ['Manual text', 'Upload CSV', 'Random'])
    if method == 'Manual text':
        txt = st.text_area('Enter rows (space/comma separated)', height=150)
        if st.button('Save matrix') and txt.strip():
            try:
                st.session_state['matrices'][new_name] = parse_text_matrix(txt)
                st.experimental_rerun()
            except Exception as e:
                st.error(str(e))
    elif method == 'Upload CSV':
        up = st.file_uploader('Choose CSV')
        if up and st.button('Save matrix'):
            st.session_state['matrices'][new_name] = load_csv(up)
            st.experimental_rerun()
    else:
        r_rows = st.number_input('Rows', 2, 50, 4)
        r_cols = st.number_input('Cols', 2, 50, 4)
        seed = st.number_input('Seed', value=0)
        if st.button('Generate & save'):
            st.session_state['matrices'][new_name] = np.random.default_rng(int(seed)).random((r_rows, r_cols)).round(2)
            st.experimental_rerun()

st.subheader(f'Matrix "{selected}"')
st.altair_chart(heatmap(A, selected), use_container_width=True)

# ----- α‑cut & graph -----
a_val = st.slider('α for cut / graph', 0.0, 1.0, 0.0, 0.05)
st.altair_chart(heatmap(alpha_cut(A, a_val), f'α‑cut ({a_val})'), use_container_width=True)
st.plotly_chart(graph_view(A, a_val), use_container_width=True)

# ----- Operations between matrices -----
st.header('⚙️ Operations')
col_ops = st.columns(2)
with col_ops[0]:
    op = st.selectbox('Operation', ['Complement', 'Union', 'Intersection', 'Max–min Composition'])
with col_ops[1]:
    B_name = st.selectbox('Second matrix (if needed)', mat_names)
    B = st.session_state['matrices'][B_name]

result: Optional[np.ndarray] = None
if st.button('Compute'):
    try:
        if op == 'Complement':
            result = fuzz_complement(A)
        elif op == 'Union':
            result = fuzz_union(A, B)
        elif op == 'Intersection':
            result = fuzz_intersection(A, B)
        else:  # composition
            result = maxmin_comp(A, B)
        if result is not None:
            name_res = f'{op}({selected},{B_name})' if op != 'Complement' else f'¬{selected}'
            st.session_state['matrices'][name_res] = result
            st.success(f'Result saved as "{name_res}"')
            st.altair_chart(heatmap(result, 'Result'), use_container_width=True)
    except AssertionError as e:
        st.error(str(e))

# ----- Closure -----
if A.shape[0] == A.shape[1]:
    if st.checkbox('Compute Warshall–Kosko closure for selected matrix'):
        res_cl = warshall_kosko(A)
        st.session_state['matrices'][f'{selected}*'] = res_cl
        st.altair_chart(heatmap(res_cl, f'{selected}*'), use_container_width=True)
        st.success(f'Closure saved as "{selected}*"')

# ----- Export -----
with st.sidebar.expander('💾 Export matrix'):
    export_name = st.selectbox('Choose matrix', list(st.session_state['matrices'].keys()), key='export')
    csv_bytes = pd.DataFrame(st.session_state['matrices'][export_name]).to_csv(index=False, header=False).encode()
    st.download_button('Download CSV', csv_bytes, f'{export_name}.csv', 'text/csv')
