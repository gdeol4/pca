import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
import matplotlib.pyplot as plt

# Add this at the very beginning of your app
st.set_page_config(layout="centered")

# Adjust the container width to be slightly wider
st.markdown("""
    <style>
        .stApp > header {
            background-color: transparent;
        }
        .block-container {
            max-width: 1200px;  # Increased from 1000px
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# Increase default figure DPI and size
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Load sample data (same as before)
standard_data = pd.DataFrame({
    'RT (min)': [16.726, 37.816, 40.178, 43.777, 69.108, 76.498, 78.998, 86.744, 90.931, 102.283],
    'Area%': [10.255, 12.666, 13.680, 11.218, 12.490, 9.977, 13.158, 8.470, 1.879, 6.205],
    'Height%': [9.12, 9.95, 7.97, 6.06, 11.44, 11.95, 13.59, 14.70, 4.44, 10.77]
})

sample_data = pd.DataFrame({
    'RT (min)': [40.225, 43.767, 50.345, 69.228, 72.961, 76.714, 86.891, 87.313],
    'Area%': [56.204, 17.919, 1.851, 2.208, 2.521, 6.509, 5.648, 7.140],
    'Height%': [36.70, 11.39, 1.54, 3.01, 4.31, 11.47, 15.15, 16.44]
})

def scale_data(data, features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])
    else:
        scaled_data = scaler.transform(data[features])
    return scaled_data, scaler

def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca_data, pca

def generate_pca_plot(standard_df, sample_df):
    features = ['RT (min)', 'Area%', 'Height%']
    
    # Scale data
    standard_scaled, scaler = scale_data(standard_df, features)
    sample_scaled, _ = scale_data(sample_df, features, scaler)
    
    # Perform PCA
    standard_pca, pca = perform_pca(standard_scaled)
    sample_pca = pca.transform(sample_scaled)
    
    # Create larger plot with higher DPI
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot points
    ax.scatter(standard_pca[:, 0], standard_pca[:, 1], 
              c='#6A0DAD', marker='o', s=150, label='Standard',
              alpha=0.8, edgecolor='white', linewidth=1, zorder=2)
    ax.scatter(sample_pca[:, 0], sample_pca[:, 1], 
              c='#228B22', marker='s', s=150, label='Sample',
              alpha=0.8, edgecolor='white', linewidth=1, zorder=2)
    
    # Add labels with adjusted text positions
    texts = []
    for i in range(len(standard_pca)):
        texts.append(plt.text(standard_pca[i,0], standard_pca[i,1],
                    f'RT={standard_df["RT (min)"].iloc[i]:.1f}',
                    fontsize=11, fontweight='bold', color='#4A4A4A'))
    
    for i in range(len(sample_pca)):
        texts.append(plt.text(sample_pca[i,0], sample_pca[i,1],
                    f'RT={sample_df["RT (min)"].iloc[i]:.1f}',
                    fontsize=11, fontweight='bold', color='#4A4A4A'))
    
    adjust_text(texts,
           arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
           expand_points=(4, 4),     # Increased from 3 to 4 for more spacing
           force_points=(2.5, 2.5),  # Increased from 2 to 2.5 for stronger repulsion
           force_text=(2.5, 2.5),    # Increased to match force_points
           ha='center',
           va='bottom',
           only_move={'points':'y', 'texts':'xy'},
           add_objects=[],
           lim=2000,                 # Increased from 1000 for more iterations
           autoalign='xy'            # Added to help with alignment
    )
    
    # Remove scientific notation from axes
    ax.ticklabel_format(style='plain')
    
    # Add larger margins to the axes limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    margin = 0.15  # Increased margin to 15%
    ax.set_xlim(x_min - abs(x_min*margin), x_max + abs(x_max*margin))
    ax.set_ylim(y_min - abs(y_min*margin), y_max + abs(y_max*margin))
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)', 
              fontsize=13, labelpad=10)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)', 
              fontsize=13, labelpad=10)
    plt.title('Mass Spec Data PCA Analysis\nBatch Comparison', 
              fontsize=16, pad=20)
    
    # Move legend to upper left to avoid overlap
    legend = plt.legend(title='', loc='upper left', 
                       frameon=True, 
                       framealpha=0.95,
                       edgecolor='none',
                       fontsize=12)
    legend.get_frame().set_facecolor('white')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    return fig


# Streamlit app
st.title('Mass Spec Data Analysis')

# Single file upload button
uploaded_file = st.file_uploader("Upload Data File (XLSX)", type=['xlsx'])

# Initialize session state
if 'show_data' not in st.session_state:
    st.session_state.show_data = False

# Button to use sample data
if st.button('Use Sample Data') or uploaded_file is not None:
    st.session_state.show_data = True

# Show data tables if button is clicked
if st.session_state.show_data:
    with st.expander("Standard Dataset"):
        st.dataframe(standard_data)
    
    with st.expander("Sample Dataset"):
        st.dataframe(sample_data)
    
    # Button to generate PCA plot
    if st.button('Generate PCA Plot'):
        fig = generate_pca_plot(standard_data, sample_data)
        st.pyplot(fig)
