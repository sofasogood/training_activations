import streamlit as st
import torch
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils.shape_explainer import explain_tensor_shape

def load_activation(step):
    """Load activation data for a given step."""
    act_path = Path("activations") / f"act_block4_output_step_{step}.pt"
    if not act_path.exists():
        return None
    return torch.load(act_path)

def get_available_steps():
    """Get list of available activation steps."""
    act_dir = Path("activations")
    steps = []
    for file in act_dir.glob("act_block4_output_step_*.pt"):
        step = int(file.stem.split("_")[-1])
        steps.append(step)
    return sorted(steps)

def plot_activation_heatmap(activation, title):
    """Plot activation heatmap using plotly."""
    # Take mean across sequence dimension
    act_mean = activation.mean(dim=1).numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=act_mean,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Hidden Dimension",
        yaxis_title="Batch Size"
    )
    return fig

def plot_activation_stats(activation, title):
    """Plot activation statistics."""
    # Take mean across sequence and batch dimensions
    act_mean = activation.mean(dim=(0, 1)).numpy()
    act_std = activation.std(dim=(0, 1)).numpy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=act_mean,
        name='Mean',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        y=act_std,
        name='Std',
        line=dict(color='red')
    ))
    fig.update_layout(
        title=f"{title} - Activation Statistics",
        xaxis_title="Hidden Dimension",
        yaxis_title="Value"
    )
    return fig

def compute_activation_diff(act1, act2):
    """Compute difference between activations, handling size mismatches."""
    # Take mean across sequence dimension for both tensors
    act1_mean = act1.mean(dim=1)  # [batch_size, hidden_dim]
    act2_mean = act2.mean(dim=1)  # [batch_size, hidden_dim]
    
    # Ensure same batch size by taking minimum
    min_batch = min(act1_mean.size(0), act2_mean.size(0))
    act1_mean = act1_mean[:min_batch]
    act2_mean = act2_mean[:min_batch]
    
    return act1_mean - act2_mean

# New function for dimension explorer
def plot_single_dimension(activation, dim_index, title):
    """Plot activation values for a single dimension across sequence length."""
    # Handle different activation shapes
    if len(activation.shape) == 3:  # [batch_size, seq_len, hidden_dim]
        dim_values = activation[:, :, dim_index].numpy()
    elif len(activation.shape) == 2:  # [batch_size, hidden_dim]
        # For 2D tensors, we don't have sequence information
        dim_values = activation[:, dim_index].unsqueeze(1).numpy()
    else:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")
    
    fig = go.Figure()
    for i in range(dim_values.shape[0]):
        fig.add_trace(go.Scatter(
            y=dim_values[i],
            mode='lines',
            name=f'Batch item {i}'
        ))
    
    fig.update_layout(
        title=f"{title} - Dimension {dim_index}",
        xaxis_title="Sequence Position",
        yaxis_title="Activation Value"
    )
    return fig

def plot_enhanced_dimension(activation, input_ids, tokenizer, dim_index, step, title=None):
    """
    Create an enhanced visualization of dimension activations with token annotations
    and additional contextual information.
    
    Args:
        activation: Tensor of shape [batch_size, seq_len, hidden_dim]
        input_ids: Tensor of input token IDs [batch_size, seq_len]
        tokenizer: The tokenizer to decode input_ids
        dim_index: Index of dimension to visualize
        step: Training step number
        title: Optional custom title
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Extract values for the selected dimension
    if len(activation.shape) == 3:  # [batch_size, seq_len, hidden_dim]
        dim_values = activation[:, :, dim_index].numpy()
    elif len(activation.shape) == 2:  # [batch_size, hidden_dim]
        dim_values = activation[:, dim_index].unsqueeze(1).numpy()
    else:
        raise ValueError(f"Unexpected shape: {activation.shape}")
    
    batch_size, seq_len = dim_values.shape
    
    # Create subplot with main plot and token visualization area
    fig = make_subplots(
        rows=2, cols=1, 
        row_heights=[0.8, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    # Plot activation lines for each batch item
    colors = ['blue', 'lightblue', 'teal', 'cyan']  # Colors for different batch items
    
    for i in range(batch_size):
        fig.add_trace(
            go.Scatter(
                x=list(range(seq_len)),
                y=dim_values[i],
                mode='lines',
                name=f'Batch item {i}',
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=1, col=1
        )
    
    # Add vertical regions for high activation areas
    # Highlight positions where activation is in top 10% for any batch item
    threshold = np.percentile(np.abs(dim_values), 90)
    highlighted_positions = set()
    
    for i in range(batch_size):
        for pos in range(seq_len):
            if abs(dim_values[i, pos]) > threshold:
                highlighted_positions.add(pos)
    
    # Sort positions for drawing regions
    highlighted_positions = sorted(list(highlighted_positions))
    
    # Group adjacent positions
    regions = []
    if highlighted_positions:
        region_start = highlighted_positions[0]
        for i in range(1, len(highlighted_positions)):
            if highlighted_positions[i] > highlighted_positions[i-1] + 1:
                regions.append((region_start, highlighted_positions[i-1]))
                region_start = highlighted_positions[i]
        regions.append((region_start, highlighted_positions[-1]))
    
    # Add highlighted regions
    for start, end in regions:
        fig.add_shape(
            type="rect",
            x0=start, x1=end, y0=-6, y1=6,
            fillcolor="yellow", opacity=0.2, layer="below", line_width=0,
            row=1, col=1
        )
    
    # Add token visualization for first batch item
    if input_ids is not None and tokenizer is not None:
        # Get tokens for the first batch item
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
        
        # Add token markers at the bottom
        token_positions = []
        token_labels = []
        
        # Sample tokens to avoid overcrowding (show every nth token)
        sample_rate = max(1, seq_len // 30)
        
        for pos in range(0, min(seq_len, len(tokens)), sample_rate):
            token_text = tokens[pos]
            if token_text.strip():  # Skip empty tokens
                token_positions.append(pos)
                token_labels.append(token_text)
        
        # Add token markers
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=[0] * len(token_positions),
                mode='markers+text',
                text=token_labels,
                textposition="bottom center",
                marker=dict(size=5, color='black'),
                showlegend=False,
                hoverinfo='text',
                hovertext=[f"Position {pos}: {label}" for pos, label in zip(token_positions, token_labels)]
            ),
            row=2, col=1
        )
    
    # Add statistical properties in annotations
    mean_val = float(dim_values.mean())
    std_val = float(dim_values.std())
    max_val = float(dim_values.max())
    min_val = float(dim_values.min())
    
    stats_text = (
        f"Mean: {mean_val:.2f}<br>"
        f"Std Dev: {std_val:.2f}<br>"
        f"Range: [{min_val:.2f}, {max_val:.2f}]"
    )
    
    fig.add_annotation(
        x=0.01, y=0.98,
        xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Add neuron interpretation hypothesis (placeholder - you'd replace with actual analysis)
    interp_text = (
        "<b>Possible Function:</b><br>"
        "Responds to narrative transitions<br>"
        "and sentence boundaries"
    )
    
    fig.add_annotation(
        x=0.99, y=0.98,
        xref="paper", yref="paper",
        text=interp_text,
        showarrow=False,
        font=dict(size=10),
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Set the title
    if title is None:
        title = f"Step {step} - Dimension {dim_index} Activation Analysis"
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        xaxis_title="",
        yaxis_title="Activation Value",
        xaxis2_title="Sequence Position",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Configure y-axis ranges
    fig.update_yaxes(range=[-6, 6], row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    
    return fig

# New function for training progress visualization
def extract_loss_from_checkpoint(step):
    """Extract loss value from a checkpoint file."""
    checkpoint_path = Path("checkpoints") / f"checkpoint_step_{step}.pt"
    if not checkpoint_path.exists():
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Try different possible structures
            if 'loss' in checkpoint:
                return checkpoint['loss']
            elif 'model_state_dict' in checkpoint and 'loss' in checkpoint:
                return checkpoint['loss']
            # Look for the loss in nested dicts as well
            for key, value in checkpoint.items():
                if isinstance(value, dict) and 'loss' in value:
                    return value['loss']
        return None  # No loss found
    except Exception as e:
        print(f"Error loading checkpoint at {checkpoint_path}: {str(e)}")
        return None

def plot_training_progress():
    """Plot training progress using data from checkpoints."""
    checkpoints_dir = Path("checkpoints")
    steps = []
    losses = []
    
    for file in checkpoints_dir.glob("checkpoint_step_*.pt"):
        step = int(file.stem.split("_")[-1])
        loss = extract_loss_from_checkpoint(step)
        
        if loss is not None:
            steps.append(step)
            losses.append(loss)
    
    if not steps:
        return None
    
    # Sort by step
    step_loss = sorted(zip(steps, losses))
    steps = [s for s, _ in step_loss]
    losses = [l for _, l in step_loss]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=losses,
        mode='lines+markers',
        name='Training Loss'
    ))
    
    fig.update_layout(
        title="Training Loss Over Time",
        xaxis_title="Step",
        yaxis_title="Loss",
        yaxis=dict(
            type="log" if st.checkbox("Log Scale for Loss", False) else "linear"
        )
    )
    return fig

def main():
    st.title("GPT-2 Block 4 Activation Inspector")

    # ------------------------------------------------------------------ #
    #  Shared controls (sidebar)                                          #
    # ------------------------------------------------------------------ #
    steps = get_available_steps()
    if not steps:
        st.error("No activation files found. Please run training first.")
        return

    st.sidebar.header("Step Selection")
    selected_step = st.sidebar.selectbox(
        "Training step", steps, index=len(steps) - 1
    )

    # Try to load activations & tokens once; reuse across tabs
    act_data = load_activation(selected_step)
    if act_data is None:
        st.error(f"No activation data found for step {selected_step}")
        return

    input_ids, tokenizer = None, None
    try:
        from transformers import AutoTokenizer

        input_path = Path("train_experiment/activations") / f"input_ids_step_{selected_step}.pt"
        if input_path.exists():
            input_ids = torch.load(input_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        st.warning(f"Could not load input tokens: {e}")

    # ------------------------------------------------------------------ #
    #  Tabs                                                              #
    # ------------------------------------------------------------------ #
    tab_overview, tab_heat, tab_dims, tab_enh, tab_diff = st.tabs(
        ["Training ⏳", "Heat-maps 🔥", "Dim Explorer 📈", "Enhanced 🧠", "Δ vs Prev 📊"]
    )

    # 1️⃣  TRAINING OVERVIEW
    with tab_overview:
        st.header("Training Progress")
        if st.checkbox("Show training loss"):
            fig = plot_training_progress()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No checkpoints or loss values found.")

        with st.expander("Tensor Shape Explained", expanded=True):
            # st.metric("Input shape", str(act_data["input"].shape))
            explain_tensor_shape(act_data["input"],
                         axis_labels=["Batch", "Seq len", "Hidden"],
                         name="GPT-2 Block-4 Input",
                         key="input_shape")
        with st.expander("Tensor Shape Explained", expanded=True):
            # st.metric("Output shape", str(act_data["output"].shape))
            explain_tensor_shape(act_data["output"],
                         axis_labels=["Batch", "Seq len", "Hidden"],
                         name="GPT-2 Block-4 Output",
                         key="output_shape")


    # 2️⃣  HEAT-MAPS
    with tab_heat:
        st.header(f"Step {selected_step} – Activation Heat-maps")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.plotly_chart(
                plot_activation_heatmap(act_data["input"], f"Input – step {selected_step}"),
                use_container_width=True,
            )
        with col2:
            st.subheader("Output")
            st.plotly_chart(
                plot_activation_heatmap(act_data["output"], f"Output – step {selected_step}"),
                use_container_width=True,
            )

        if st.checkbox("Show per-dimension stats"):
            st.plotly_chart(
                plot_activation_stats(act_data["output"], f"Step {selected_step}"),
                use_container_width=True,
            )

    # 3️⃣  BASIC DIMENSION EXPLORER
    with tab_dims:
        st.header("Dimension Explorer")

        out_shape = act_data["output"].shape
        num_dims = out_shape[2] if len(out_shape) == 3 else out_shape[1]
        dim_idx = st.slider("Pick a dimension", 0, num_dims - 1, 0)

        fig = plot_single_dimension(act_data["output"], dim_idx, f"Step {selected_step}")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Stats for this dimension"):
            dim_vals = (
                act_data["output"][:, :, dim_idx]
                if len(out_shape) == 3
                else act_data["output"][:, dim_idx]
            )
            st.write(
                {
                    "mean": float(dim_vals.mean()),
                    "std": float(dim_vals.std()),
                    "min": float(dim_vals.min()),
                    "max": float(dim_vals.max()),
                }
            )

    # 4️⃣  ENHANCED DIMENSION VIEW
    with tab_enh:
        st.header("Enhanced Dimension Analysis (illustrative)")
        enh_idx = st.slider("Dimension", 0, num_dims - 1, 0, key="enh_dim_slider")
        st.plotly_chart(
            plot_enhanced_dimension(
                act_data["output"], input_ids, tokenizer, enh_idx, selected_step
            ),
            use_container_width=True,
        )

    # 5️⃣  DIFFERENCE VS PREVIOUS STEP
    with tab_diff:
        st.header("Change from Previous Step")
        if selected_step > min(steps):
            prev_step = steps[steps.index(selected_step) - 1]
            prev_act = load_activation(prev_step)
            if prev_act is not None:
                diff = compute_activation_diff(act_data["output"], prev_act["output"])
                st.plotly_chart(
                    plot_activation_heatmap(
                        diff, f"Δ (step {selected_step} − step {prev_step})"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Previous-step activations not found.")
        else:
            st.info("No earlier step to compare against.")


if __name__ == "__main__":
    main()
