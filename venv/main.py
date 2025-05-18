# =============================================================
# main.py — Fuzzy Relation Explorer v6.3
# -------------------------------------------------------------
# Complete Streamlit app for fuzzy binary relations: matrix
# management, visualization, α-cuts, projections, property
# checks, operations (¬, ∪, ∩, ∘), Warshall–Kosko closure, export.
# =============================================================
"""
Run with:
    pip install streamlit numpy pandas altair networkx plotly pandas
    streamlit run main.py
"""
from __future__ import annotations

from typing import Dict
import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TOL = 1e-6  # numeric tolerance

# ────────────── helpers ──────────────

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ─────────── algebraic ops ───────────

def fuzz_union(A, B):
    assert A.shape == B.shape, "Union: shape mismatch"
    return np.maximum(A, B)

def fuzz_intersection(A, B):
    assert A.shape == B.shape, "Intersection: shape mismatch"
    return np.minimum(A, B)

def fuzz_complement(A):
    return 1.0 - A

def maxmin_comp(A, B):
    assert A.shape[1] == B.shape[0], "Composition: inner dims mismatch"
    return np.maximum.reduce(np.minimum(A[:, :, None], B[None, :, :]), axis=1)

def warshall_kosko(A, eps=1e-6):
    T = A.copy()
    while True:
        Tn = np.maximum(T, maxmin_comp(T, T))
        if np.allclose(T, Tn, atol=eps):
            return Tn
        T = Tn

# ───────────── projections ───────────

def v_proj(A):
    return A.max(axis=1)

def h_proj(A):
    return A.max(axis=0)

def alpha_cut(A, a, crisp=True):
    B = A.copy(); B[B < a] = 0
    return (B > 0).astype(float) if crisp else B

# ─────────── properties ─────────────

def _square(A):
    return A.shape[0] == A.shape[1]

def is_reflexive(A):
    return _square(A) and np.allclose(np.diag(A), 1.0, atol=TOL)

def is_symmetric(A):
    return _square(A) and np.allclose(A, A.T, atol=TOL)

def is_antisymmetric(A):
    if not _square(A):
        return False
    diff = np.minimum(A, A.T); np.fill_diagonal(diff, 0)
    return np.all(diff <= TOL)

def is_transitive(A):
    return _square(A) and np.all(maxmin_comp(A, A) <= A + TOL)

PROPERTY_FUNCS = {
    "Reflexive": is_reflexive,
    "Symmetric": is_symmetric,
    "Antisymmetric": is_antisymmetric,
    "Max–min transitive": is_transitive,
}

# ─────────── visualization ──────────

def heatmap(M, title):
    df = pd.DataFrame(M)
    melt = df.reset_index().melt('index', var_name='col', value_name='val').rename(columns={'index':'row'})
    return (alt.Chart(melt, width=300, height=300)
            .mark_rect()
            .encode(x='col:O', y='row:O', color=alt.Color('val:Q', scale=alt.Scale(domain=[0,1])),
                    tooltip=['row','col','val'])
            .properties(title=title))

def graph_view(A, a=0.0):
    G = nx.DiGraph(); n,m=A.shape
    for i in range(n):
        for j in range(m):
            if A[i,j] >= a:
                G.add_edge(f'x{i}', f'y{j}', weight=A[i,j])
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u,v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    node_x = [pos[n][0] for n in G.nodes]; node_y = [pos[n][1] for n in G.nodes]
    return (go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1)),
        go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes), marker=dict(size=12,color='LightSkyBlue'), textposition='top center')])
            .update_layout(showlegend=False, title=f'Graph (α ≥ {a})'))

# ───────────── utilities ─────────────

def parse_text(txt):
    rows = [list(filter(None, r.replace(',', ' ').split())) for r in txt.strip().splitlines() if r.strip()]
    arr = np.array(rows, dtype=float)
    if not ((0<=arr).all() & (arr<=1).all()):
        raise ValueError('Values must be in [0,1]')
    return arr

def load_csv(up):
    return pd.read_csv(up, header=None).values.astype(float)

# ───────────── initial state ─────────
if 'mats' not in st.session_state:
    st.session_state['mats']: Dict[str,np.ndarray] = {
        'R1': np.array([[0.1,0.7,0.4],[1.0,0.5,0.0]]),
        'R2': np.array([[0.9,0,1,0.2],[0.3,0.6,0,0.9],[0.1,1,0,0.5]])
    }

mats = st.session_state['mats']

# ───────────── interface ─────────────
st.set_page_config('Fuzzy Relation Explorer', layout='wide')
st.title('🔍 Fuzzy Relation Explorer')

# Sidebar manager
st.sidebar.header('📂 Matrices')
sel = st.sidebar.selectbox('Current', list(mats.keys()))
A = mats[sel]

with st.sidebar.expander('➕ Add / Replace'):
    new_name = st.text_input('Name', f'M{len(mats)+1}')
    method = st.radio('Method', ['Manual', 'CSV', 'Random'])
    if method == 'Manual':
        txt = st.text_area('Rows', height=150)
        if st.button('Save') and txt.strip():
            try:
                mats[new_name] = parse_text(txt); safe_rerun()
            except Exception as e: st.error(e)
    elif method == 'CSV':
        up = st.file_uploader('CSV')
        if up and st.button('Save'):
            mats[new_name] = load_csv(up); safe_rerun()
    else:
        r = st.number_input('Rows', 2, 50, 4)
        c = st.number_input('Cols', 2, 50, 4)
        seed = st.number_input('Seed', 0)
        if st.button('Generate'):
            mats[new_name] = np.random.default_rng(int(seed)).random((r, c)).round(2); safe_rerun()

# Tabs
mat_tab, proj_tab, prop_tab, op_tab, clo_tab = st.tabs(['Matrix', 'Projections', 'Properties', 'Operations', 'Closure'])

with mat_tab:
    st.altair_chart(heatmap(A, sel), use_container_width=True)
    alpha_val = st.slider('α-cut threshold', 0.0, 1.0, 0.0, 0.05)
    st.altair_chart(heatmap(alpha_cut(A, alpha_val), f'α={alpha_val}'), use_container_width=True)
    st.plotly_chart(graph_view(A, alpha_val), use_container_width=True)

with proj_tab:
    st.subheader('Vertical projection μ_X'); st.bar_chart(v_proj(A))
    st.subheader('Horizontal projection μ_Y'); st.bar_chart(h_proj(A))

with prop_tab:
    for name, func in PROPERTY_FUNCS.items():
        st.write(f"{name}: {'✅' if func(A) else '❌'}")

with op_tab:
    op = st.selectbox('Operation', ['Complement', 'Union', 'Intersection', 'Max–min Composition'])
    B_name = st.selectbox('Second matrix', list(mats.keys()))
    B = mats[B_name]
    if st.button('Compute operation'):
        try:
            if op == 'Complement':
                res = fuzz_complement(A); res_name = f'¬{sel}'
            elif op == 'Union':
                res = fuzz_union(A, B); res_name = f'({sel}∪{B_name})'
            elif op == 'Intersection':
                res = fuzz_intersection(A, B); res_name = f'({sel}∩{B_name})'
            else:
                res = maxmin_comp(A, B); res_name = f'({sel}∘{B_name})'
            mats[res_name] = res
            st.success(f"Saved as '{res_name}'")
            st.altair_chart(heatmap(res, res_name), use_container_width=True)
        except Exception as e:
            st.error(str(e))

with clo_tab:
    if _square(A):
        if st.button('Compute Warshall–Kosko closure'):
            closure = warshall_kosko(A)
            name = f"{sel}*"
            mats[name] = closure
            st.success(f"Closure saved as '{name}'")
            st.altair_chart(heatmap(closure, name), use_container_width=True)
            st.info(f"Transitive? {'✅' if is_transitive(closure) else '❌'}")
    else:
        st.info('Matrix is not square → closure undefined')

# Export
with st.sidebar.expander('💾 Export'):
    ex_name = st.selectbox('Matrix', list(mats.keys()), key='exp')
    st.download_button('Download CSV', pd.DataFrame(mats[ex_name]).to_csv(index=False, header=False).encode(), f'{ex_name}.csv', 'text/csv')
