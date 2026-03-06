"""
LUNG CANCER DETECTION - STREAMLIT APP (Deployment Version)
Handles missing models gracefully
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .high-risk { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .medium-risk { background: linear-gradient(135deg, #f39c12, #e67e22); }
    .low-risk { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .info-box {
        padding: 1rem;
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🫁 Lung Cancer Detection System</h1><p>Powered by Deep Learning (CNN)</p></div>', unsafe_allow_html=True)

# Try to import TensorFlow (with error handling)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import pickle
    TENSORFLOW_AVAILABLE = True
    st.sidebar.success("✅ TensorFlow loaded")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    st.sidebar.warning("⚠️ TensorFlow not available - using demo mode")

# Load models with caching and error handling
@st.cache_resource
def load_models():
    """Load trained models with error handling"""
    ct_model = None
    histo_model = None
    ct_classes = None
    histo_classes = None
    
    if not TENSORFLOW_AVAILABLE:
        return ct_model, histo_model, ct_classes, histo_classes
    
    try:
        # Load CT model
        if os.path.exists('models/ct_model_final.h5'):
            ct_model = load_model('models/ct_model_final.h5')
            with open('models/ct_class_indices.pkl', 'rb') as f:
                ct_classes = pickle.load(f)
            st.success("✅ CT model loaded")
        else:
            st.warning("⚠️ CT model file not found")
        
        # Load Histopathology model
        if os.path.exists('models/histo_model_final.h5'):
            histo_model = load_model('models/histo_model_final.h5')
            with open('models/histo_class_indices.pkl', 'rb') as f:
                histo_classes = pickle.load(f)
            st.success("✅ Histopathology model loaded")
        else:
            st.warning("⚠️ Histopathology model file not found")
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    return ct_model, histo_model, ct_classes, histo_classes

# Load models
with st.spinner("Loading models..."):
    ct_model, histo_model, ct_classes, histo_classes = load_models()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Analysis Type",
    ["Home", "CT Scan Analysis", "Histopathology Analysis", "Ensemble Analysis", "About"]
)

# Demo mode warning
if not TENSORFLOW_AVAILABLE:
    st.sidebar.info("""
    **📱 Demo Mode Active**
    
    TensorFlow is not installed. 
    This is a preview of the UI.
    
    To enable predictions:
    1. Add tensorflow to requirements.txt
    2. Redeploy the app
    """)

# Image preprocessing function
def preprocess_image(uploaded_file):
    """Preprocess uploaded image for prediction"""
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    if TENSORFLOW_AVAILABLE:
        img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# Plot probabilities (demo version)
def plot_probabilities_demo():
    """Create demo probability chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    classes = ['Adenocarcinoma', 'Normal/Benign', 'Squamous Cell']
    probs = [0.45, 0.35, 0.20]  # Demo probabilities
    bars = ax.bar(classes, probs, color=['#e74c3c', '#2ecc71', '#3498db'])
    ax.set_title('Demo: Class Probabilities (No Model)', fontsize=14)
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.0%}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_probabilities_real(probs, class_names):
    """Create real probability chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, probs, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_title('Class Probabilities', fontsize=14)
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.2%}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# HOME PAGE
if page == "Home":
    st.markdown("""
    ### Welcome to the Lung Cancer Detection System
    
    This application uses Deep Learning to analyze medical images for lung cancer detection.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🖥️ CT Scan Analysis")
        st.write("Upload chest CT scans for lung nodule detection and classification")
    
    with col2:
        st.markdown("### 🔬 Histopathology")
        st.write("Analyze tissue samples for cancer type classification")
    
    with col3:
        st.markdown("### 🧠 Ensemble Analysis")
        st.write("Combine both methods for comprehensive diagnosis")
    
    st.markdown("""
    <div class="info-box">
    <strong>⚠️ Note:</strong> This system is for research and educational purposes only. 
    Always consult healthcare professionals for medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# CT SCAN PAGE
elif page == "CT Scan Analysis":
    st.header("📊 CT Scan Analysis")
    
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not installed. Showing demo mode with sample predictions.")
    
    uploaded_file = st.file_uploader(
        "Upload CT Scan Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest CT scan image"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded CT Scan", use_container_width=True)
        
        with col2:
            if not TENSORFLOW_AVAILABLE:
                # Demo mode - show sample prediction
                st.markdown("""
                <div class="prediction-box medium-risk">
                    <h3>Prediction: Adenocarcinoma</h3>
                    <h1>87% Confidence</h1>
                    <p style="font-size: 0.9rem;">Demo Mode - Install TensorFlow for real predictions</p>
                </div>
                """, unsafe_allow_html=True)
                st.pyplot(plot_probabilities_demo())
            else:
                if ct_model is None:
                    st.error("CT Model not loaded. Please train first or check model files.")
                else:
                    with st.spinner("Analyzing image..."):
                        # Real prediction code here
                        img, img_array = preprocess_image(uploaded_file)
                        
                        # This is where your actual prediction would go
                        # predictions = ct_model.predict(img_array, verbose=0)[0]
                        # For now, using demo
                        st.info("Model loaded but prediction code needs to be completed")

# HISTOPATHOLOGY PAGE
elif page == "Histopathology Analysis":
    st.header("🔬 Histopathology Analysis")
    
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not installed. Showing demo mode with sample predictions.")
    
    uploaded_file = st.file_uploader(
        "Upload Histopathology Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a tissue sample image"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Tissue Sample", use_container_width=True)
        
        with col2:
            if not TENSORFLOW_AVAILABLE:
                st.markdown("""
                <div class="prediction-box high-risk">
                    <h3>Prediction: Squamous Cell Carcinoma</h3>
                    <h1>78% Confidence</h1>
                    <p style="font-size: 0.9rem;">Demo Mode - Install TensorFlow for real predictions</p>
                </div>
                """, unsafe_allow_html=True)
                st.pyplot(plot_probabilities_demo())

# ENSEMBLE PAGE
elif page == "Ensemble Analysis":
    st.header("🧠 Ensemble Analysis (CT + Histopathology)")
    
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not installed. Showing demo mode with sample predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CT Scan")
        ct_file = st.file_uploader("Upload CT Scan", type=['jpg', 'jpeg', 'png'], key="ct_ensemble")
    
    with col2:
        st.subheader("Histopathology")
        histo_file = st.file_uploader("Upload Histopathology", type=['jpg', 'jpeg', 'png'], key="histo_ensemble")
    
    if ct_file is not None and histo_file is not None:
        if not TENSORFLOW_AVAILABLE:
            st.markdown("""
            <div class="prediction-box low-risk">
                <h3>Ensemble Prediction: Normal Tissue</h3>
                <h1>92% Confidence</h1>
                <p style="font-size: 0.9rem;">Demo Mode - Install TensorFlow for real predictions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show comparison plot
            fig, ax = plt.subplots(figsize=(10, 5))
            classes = ['Adenocarcinoma', 'Normal', 'Squamous']
            x = np.arange(len(classes))
            width = 0.25
            
            ax.bar(x - width, [0.4, 0.5, 0.1], width, label='CT Scan', color='#3498db')
            ax.bar(x, [0.3, 0.6, 0.1], width, label='Histopathology', color='#2ecc71')
            ax.bar(x + width, [0.35, 0.55, 0.1], width, label='Ensemble', color='#9b59b6')
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Probability')
            ax.set_title('Demo: Model Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)

# ABOUT PAGE
else:
    st.header("About This Project")
    
    st.markdown("""
    ### 🫁 Lung Cancer Detection using Deep Learning
    
    This project uses Convolutional Neural Networks (CNNs) to detect lung cancer from medical images.
    
    **Features:**
    - CT Scan analysis for lung nodule detection
    - Histopathology image analysis for tissue classification
    - Ensemble learning combining both modalities
    
    **Technologies:**
    - TensorFlow / Keras for deep learning
    - Streamlit for web interface
    - Python for backend processing
    
    **Deployment Status:** """ + ("✅ TensorFlow Installed" if TENSORFLOW_AVAILABLE else "⚠️ Running in Demo Mode") + """
    
    """)
    
    if not TENSORFLOW_AVAILABLE:
        st.info("""
        ### To Enable Real Predictions:
        
        1. Make sure tensorflow is in your requirements.txt
        2. Redeploy the app
        3. Train your models and add them to the repository
        """)

# Footer
st.markdown("---")
st.markdown("© 2024 Lung Cancer Detection System | " + 
            ("🔴 Demo Mode" if not TENSORFLOW_AVAILABLE else "🟢 Production Mode"))
