"""
LUNG CANCER DETECTION - STREAMLIT APP
One file, no HTML/CSS/JavaScript needed!
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS (optional, but Streamlit makes it easy)
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🫁 Lung Cancer Detection System</h1><p>Powered by Deep Learning (CNN)</p></div>', unsafe_allow_html=True)

# Load models with caching (Streamlit's magic!)
@st.cache_resource
def load_models():
    """Load trained models (cached for performance)"""
    ct_model = None
    histo_model = None
    ct_classes = None
    histo_classes = None
    
    # Load CT model
    if os.path.exists('models/ct_model_final.h5'):
        ct_model = tf.keras.models.load_model('models/ct_model_final.h5')
        with open('models/ct_class_indices.pkl', 'rb') as f:
            ct_classes = pickle.load(f)
        st.success("✅ CT model loaded")
    
    # Load Histopathology model
    if os.path.exists('models/histo_model_final.h5'):
        histo_model = tf.keras.models.load_model('models/histo_model_final.h5')
        with open('models/histo_class_indices.pkl', 'rb') as f:
            histo_classes = pickle.load(f)
        st.success("✅ Histopathology model loaded")
    
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

# Image preprocessing function
def preprocess_image(uploaded_file):
    """Preprocess uploaded image for prediction"""
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# Plot probabilities
def plot_probabilities(probs, class_names):
    """Create probability bar chart"""
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🖥️ CT Scan Analysis")
        st.write("Upload chest CT scans for lung nodule detection and classification")
        if st.button("Go to CT Scan →", key="home_ct"):
            st.session_state.page = "CT Scan Analysis"
            st.experimental_rerun()
    
    with col2:
        st.markdown("### 🔬 Histopathology")
        st.write("Analyze tissue samples for cancer type classification")
        if st.button("Go to Histopathology →", key="home_histo"):
            st.session_state.page = "Histopathology Analysis"
            st.experimental_rerun()
    
    with col3:
        st.markdown("### 🧠 Ensemble Analysis")
        st.write("Combine both methods for comprehensive diagnosis")
        if st.button("Go to Ensemble →", key="home_ensemble"):
            st.session_state.page = "Ensemble Analysis"
            st.experimental_rerun()
    
    st.info("⚠️ **Note:** This system is for research and educational purposes only. Always consult healthcare professionals.")

# CT SCAN PAGE
elif page == "CT Scan Analysis":
    st.header("📊 CT Scan Analysis")
    
    # File uploader (built into Streamlit!)
    uploaded_file = st.file_uploader(
        "Upload CT Scan Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest CT scan image"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded CT Scan", use_container_width=True)
        
        with col2:
            if ct_model is None:
                st.error("Model not loaded. Please train first.")
            else:
                with st.spinner("Analyzing image..."):
                    # Preprocess and predict
                    img, img_array = preprocess_image(uploaded_file)
                    
                    # Get prediction
                    predictions = ct_model.predict(img_array, verbose=0)[0]
                    pred_class_idx = np.argmax(predictions)
                    confidence = np.max(predictions)
                    
                    # Get class name
                    idx_to_class = {v: k for k, v in ct_classes.items()}
                    pred_class = idx_to_class[pred_class_idx]
                    
                    # Show result
                    confidence_pct = confidence * 100
                    
                    # Color-coded result box
                    risk_class = "low-risk" if confidence_pct > 80 else "medium-risk" if confidence_pct > 60 else "high-risk"
                    st.markdown(f'<div class="prediction-box {risk_class}">'
                               f'<h3>Prediction: {pred_class.replace("_", " ").title()}</h3>'
                               f'<h1>{confidence_pct:.1f}% Confidence</h1>'
                               f'</div>', unsafe_allow_html=True)
                    
                    # Show probability plot
                    st.pyplot(plot_probabilities(predictions, list(ct_classes.keys())))

# HISTOPATHOLOGY PAGE
elif page == "Histopathology Analysis":
    st.header("🔬 Histopathology Analysis")
    
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
            if histo_model is None:
                st.error("Model not loaded. Please train first.")
            else:
                with st.spinner("Analyzing tissue sample..."):
                    img, img_array = preprocess_image(uploaded_file)
                    
                    predictions = histo_model.predict(img_array, verbose=0)[0]
                    pred_class_idx = np.argmax(predictions)
                    confidence = np.max(predictions)
                    
                    idx_to_class = {v: k for k, v in histo_classes.items()}
                    pred_class = idx_to_class[pred_class_idx]
                    
                    confidence_pct = confidence * 100
                    
                    risk_class = "low-risk" if confidence_pct > 80 else "medium-risk" if confidence_pct > 60 else "high-risk"
                    st.markdown(f'<div class="prediction-box {risk_class}">'
                               f'<h3>Prediction: {pred_class.replace("_", " ").title()}</h3>'
                               f'<h1>{confidence_pct:.1f}% Confidence</h1>'
                               f'</div>', unsafe_allow_html=True)
                    
                    st.pyplot(plot_probabilities(predictions, list(histo_classes.keys())))

# ENSEMBLE PAGE
elif page == "Ensemble Analysis":
    st.header("🧠 Ensemble Analysis (CT + Histopathology)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CT Scan")
        ct_file = st.file_uploader("Upload CT Scan", type=['jpg', 'jpeg', 'png'], key="ct_ensemble")
    
    with col2:
        st.subheader("Histopathology")
        histo_file = st.file_uploader("Upload Histopathology", type=['jpg', 'jpeg', 'png'], key="histo_ensemble")
    
    if ct_file is not None and histo_file is not None:
        if ct_model is None or histo_model is None:
            st.error("Models not loaded. Please train first.")
        else:
            with st.spinner("Analyzing both images..."):
                # CT prediction
                _, ct_array = preprocess_image(ct_file)
                ct_probs = ct_model.predict(ct_array, verbose=0)[0]
                ct_class_idx = np.argmax(ct_probs)
                ct_confidence = np.max(ct_probs)
                
                # Histopathology prediction
                _, histo_array = preprocess_image(histo_file)
                histo_probs = histo_model.predict(histo_array, verbose=0)[0]
                histo_class_idx = np.argmax(histo_probs)
                histo_confidence = np.max(histo_probs)
                
                # Ensemble (average)
                ensemble_probs = (ct_probs + histo_probs) / 2
                ensemble_class_idx = np.argmax(ensemble_probs)
                ensemble_confidence = np.max(ensemble_probs)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                idx_to_class_ct = {v: k for k, v in ct_classes.items()}
                idx_to_class_histo = {v: k for k, v in histo_classes.items()}
                
                with col1:
                    st.markdown("### CT Scan")
                    st.markdown(f"**Prediction:** {idx_to_class_ct[ct_class_idx].replace('_', ' ').title()}")
                    st.markdown(f"**Confidence:** {ct_confidence*100:.1f}%")
                
                with col2:
                    st.markdown("### Histopathology")
                    st.markdown(f"**Prediction:** {idx_to_class_histo[histo_class_idx].replace('_', ' ').title()}")
                    st.markdown(f"**Confidence:** {histo_confidence*100:.1f}%")
                
                with col3:
                    st.markdown("### Ensemble")
                    # Use CT class names for ensemble (assuming they match)
                    ensemble_class = list(ct_classes.keys())[ensemble_class_idx]
                    st.markdown(f"**Prediction:** {ensemble_class.replace('_', ' ').title()}")
                    st.markdown(f"**Confidence:** {ensemble_confidence*100:.1f}%")
                
                # Show comparison plot
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(ct_classes))
                width = 0.25
                
                ax.bar(x - width, ct_probs, width, label='CT Scan', color='#3498db')
                ax.bar(x, histo_probs, width, label='Histopathology', color='#2ecc71')
                ax.bar(x + width, ensemble_probs, width, label='Ensemble', color='#9b59b6')
                
                ax.set_xlabel('Classes')
                ax.set_ylabel('Probability')
                ax.set_title('Model Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(list(ct_classes.keys()), rotation=45)
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
    
    **Model Architecture:**
    - CT Scan: VGG16-based transfer learning
    - Histopathology: ResNet50-based transfer learning
    
    **Technologies:**
    - TensorFlow / Keras for deep learning
    - Streamlit for web interface
    - Python for backend processing
    
    ⚠️ **Disclaimer:** This system is for research and educational purposes only. 
    Always consult with qualified healthcare professionals for medical diagnosis.
    """)

# Footer
st.markdown("---")
st.markdown("© 2024 Lung Cancer Detection System | Powered by Deep Learning")
