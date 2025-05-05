import streamlit as st
import torch
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_activation(step):
    """Load activation data for a given step."""
    act_path = Path("train_experiment/activations") / f"act_block4_output_step_{step}.pt"
    if not act_path.exists():
        return None
    return torch.load(act_path)

def get_available_steps():
    """Get list of available activation steps."""
    act_dir = Path("train_experiment/activations")
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

def main():
    st.title("GPT-2 Block 4 Activation Inspector")
    
    # Get available steps
    steps = get_available_steps()
    if not steps:
        st.error("No activation files found. Please run training first.")
        return
    
    # Step selection
    st.sidebar.header("Step Selection")
    selected_step = st.sidebar.selectbox(
        "Select training step",
        steps,
        index=len(steps)-1
    )
    
    # Load activation data
    act_data = load_activation(selected_step)
    if act_data is None:
        st.error(f"No activation data found for step {selected_step}")
        return
    
    # Display activation information
    st.header(f"Step {selected_step} Activations")
    
    # Show basic stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Input Shape", str(act_data['input'].shape))
        st.metric("Output Shape", str(act_data['output'].shape))
    
    # Plot activations
    st.subheader("Activation Visualizations")
    
    # Input activations
    st.plotly_chart(plot_activation_heatmap(
        act_data['input'],
        f"Input Activations - Step {selected_step}"
    ))
    
    # Output activations
    st.plotly_chart(plot_activation_heatmap(
        act_data['output'],
        f"Output Activations - Step {selected_step}"
    ))
    
    # Activation statistics
    st.subheader("Activation Statistics")
    st.plotly_chart(plot_activation_stats(
        act_data['output'],
        f"Step {selected_step}"
    ))
    
    # Compare with previous step if available
    if selected_step > min(steps):
        prev_step = steps[steps.index(selected_step) - 1]
        prev_act = load_activation(prev_step)
        
        if prev_act is not None:
            st.subheader(f"Change from Step {prev_step}")
            
            try:
                # Compute difference using the new function
                diff = compute_activation_diff(act_data['output'], prev_act['output'])
                st.plotly_chart(plot_activation_heatmap(
                    diff,
                    f"Activation Change (Step {selected_step} - Step {prev_step})"
                ))
            except Exception as e:
                st.error(f"Could not compute difference: {str(e)}")
                st.info("This might be due to incompatible tensor shapes between steps.")

if __name__ == "__main__":
    main() 