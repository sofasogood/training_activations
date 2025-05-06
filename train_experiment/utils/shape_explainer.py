# utils/shape_explain.py  ─────────────────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math

def explain_tensor_shape(tensor, axis_labels=None, max_grid=8, name="Tensor", key=None):
    """
    Visual, intuitive explanation of a tensor's shape (à la your screenshot).

    Args
    ----
    tensor : torch.Tensor | np.ndarray
        The tensor whose shape you want to explain.
    axis_labels : list[str] | None
        Human-readable names for each dimension.
        If None we fall back to ['dim 0', 'dim 1', ...].
    max_grid : int
        The size of the *illustrative* grid (max_grid × max_grid).
        We sample / aggregate values so the plot stays tiny.
    name : str
        Title prefix (e.g. "PyTorch Tensor" or "GPT-2 Activation").
    key : str
        Unique key for the Streamlit element to prevent duplicate IDs.

    How it works
    ------------
    • Always shows the exact shape.
    • Builds a bullet list: **1st dimension (B)**: Batch size – …
    • Makes a Plotly heat-map:
        – 4-D image tensor  →  channels collapsed, show H×W grid
        – 3-D (batch, seq, hidden) →  grid of seq × hidden (sampled)
        – 2-D (batch, feats) →  bar strip
    -------------------------------------------------------------------------
    """
    import torch  # inside function so utils has no hard deps on torch

    shape = list(tensor.shape)
    rank  = len(shape)

    # 1. TITLE ----------------------------------------------------------------
    st.markdown(f"### {name} with shape `{tuple(shape)}`")

    # 2. AXIS LABELS -----------------------------------------------------------
    if axis_labels is None:
        axis_labels = {
            4: ["Batch", "Channels", "Height", "Width"],
            3: ["Batch", "Seq len", "Hidden"],
            2: ["Batch", "Features"],
        }.get(rank, [f"dim {i}" for i in range(rank)])
    # Make sure we have the right length
    if len(axis_labels) != rank:
        axis_labels = [f"dim {i}" for i in range(rank)]

    st.markdown("#### Understanding the dimensions:")
    for idx, (label, size) in enumerate(zip(axis_labels, shape), start=1):
        st.markdown(f"* **{idx}{ordinal(idx)} dimension ({size})** – {friendly_axis(label, size)}")

    # # 3. GRID / BAR VISUAL -----------------------------------------------------
    # st.markdown("#### What the tensor looks like:")
    # fig = build_shape_figure(tensor, axis_labels, max_grid)
    # st.plotly_chart(fig, use_container_width=False, key=f"shape_plot_{key}")

    # 4. TOTAL VALUES ----------------------------------------------------------
    total_vals = int(np.prod(shape))
    st.markdown(
        f"""\n<div style='padding:0.6em;background:#fffcea;
            border-radius:6px; text-align:center; font-weight:600'>
            Total number of values: { ' × '.join(map(str,shape)) } = {total_vals:,}
        </div>""",
        unsafe_allow_html=True,
    )

# ───────────────────────────────────────── helpers ───────────────────────────
def ordinal(n):  # 1→'st', 2→'nd', 3→'rd', else 'th'
    return "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4][:2] or "th"

def friendly_axis(label, size):
    lookup = {
        "Batch": "number of items processed together",
        "Channels": "RGB or feature maps",
        "Height": f"{size} pixels tall",
        "Width": f"{size} pixels wide",
        "Seq len": "tokens per input",
        "Hidden": "neuron dimensions",
        "Features": "learned embeddings / features",
    }
    return lookup.get(label, "")

def build_shape_figure(tensor, labels, max_grid):
    rank = len(tensor.shape)
    # Use dummy data – we only care about the layout
    if rank == 4:  # (B, C, H, W) – collapse channels; show H×W
        h, w = tensor.shape[-2:]
        grid = np.zeros((h, w))
    elif rank == 3:  # (B, S, H) – show S×H grid
        s, h = tensor.shape[1:]
        grid = np.zeros((s, h))
    elif rank == 2:  # (B, F) – draw a single row
        grid = np.zeros((1, tensor.shape[-1]))
    else:  # fallback 1-D
        grid = np.zeros((1, tensor.shape[-1]))

    # down-sample so the grid is at most max_grid × max_grid
    step0 = max(1, math.ceil(grid.shape[0]/max_grid))
    step1 = max(1, math.ceil(grid.shape[1]/max_grid))
    grid  = grid[::step0, ::step1]

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        showscale=False,
        colorscale="Greys",
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=200, width=200,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_visible=False, yaxis_visible=False,
    )
    return fig
