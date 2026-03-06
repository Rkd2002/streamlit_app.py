"""
LUNG CANCER DETECTION - STREAMLIT APP
Fixed version with proper error handling
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
import time

# Try to import tensorflow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    st.error(f"TensorFlow import error: {e}")
    st.info("Please install tensorflow: pip install tensorflow")

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
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
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .high-risk { 
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        box-shadow: 0 10px 20px rgba(231, 76, 60, 0.3);
    }
    .medium-risk { 
        background: linear-gradient(135deg, #f39c12, #e67e22);
        box-shadow: 0 10px 20px rgba(243, 156, 18, 0.3);
    }
    .low-risk { 
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        box-shadow: 0 10px 20px rgba(39, 174, 96, 0.3);
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .info-box {
        padding: 1rem;
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Header
st.markdown('<div class="main-header"><h1>🫁 Lung Cancer Detection System</h1><p>Powered by Deep Learning (CNN)</p></div>', unsafe_allow_html=True)

# Check TensorFlow availability
if not TENSORFLOW_AVAILABLE:
    st.error("⚠️ **TensorFlow is not installed!** Please install it to use the prediction features.")
    st.code("pip install tensorflow")
    st.stop()

# Load models with caching and error handling
@st.cache_resource
def load_models():
    """Load trained models with proper error handling"""
    ct_model = None
    histo_model = None
    ct_classes = None
    histo_classes = None
    model_status = {"ct": False, "histo": False}
    
    try:
        # Load CT model
        if os.path.exists('models/ct_model_final.h5'):
            ct_model = load_model('models/ct_model_final.h5')
            if os.path.exists('models/ct_class_indices.pkl'):
                with open('models/ct_class_indices.pkl', 'rb') as f:
                    ct_classes = pickle.load(f)
                model_status["ct"] = True
                st.sidebar.success("✅ CT model loaded")
            else:
                st.sidebar.warning("⚠️ CT class indices not found")
        else:
            st.sidebar.warning("⚠️ CT model file not found")
        
        # Load Histopathology model
        if os.path.exists('models/histo_model_final.h5'):
            histo_model = load_model('models/histo_model_final.h5')
            if os.path.exists('models/histo_class_indices.pkl'):
                with open('models/histo_class_indices.pkl', 'rb') as f:
                    histo_classes = pickle.load(f)
                model_status["histo"] = True
                st.sidebar.success("✅ Histopathology model loaded")
            else:
                st.sidebar.warning("⚠️ Histopathology class indices not found")
        else:
            st.sidebar.warning("⚠️ Histopathology model file not found")
            
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
    
    return ct_model, histo_model, ct_classes, histo_classes, model_status

# Load models
with st.spinner("🔄 Loading models..."):
    ct_model, histo_model, ct_classes, histo_classes, model_status = load_models()

# Sidebar
with st.sidebar:
    st.title("Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
    
    if st.button("📊 CT Scan Analysis", use_container_width=True):
        st.session_state.page = "CT Scan Analysis"
        st.rerun()
    
    if st.button("🔬 Histopathology Analysis", use_container_width=True):
        st.session_state.page = "Histopathology Analysis"
        st.rerun()
    
    if st.button("🧠 Ensemble Analysis", use_container_width=True):
        st.session_state.page = "Ensemble Analysis"
        st.rerun()
    
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "About"
        st.rerun()
    
    st.markdown("---")
    
    # Model status
    st.markdown("### 📊 Model Status")
    if model_status["ct"]:
        st.success("✅ CT Model: Ready")
    else:
        st.error("❌ CT Model: Not loaded")
    
    if model_status["histo"]:
        st.success("✅ Histo Model: Ready")
    else:
        st.error("❌ Histo Model: Not loaded")
    
    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.info("Classes: Adenocarcinoma, Normal, Squamous Cell Carcinoma")

# Image preprocessing function
def preprocess_image(uploaded_file):
    """Preprocess uploaded image for prediction"""
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img, img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

# Plot probabilities
def plot_probabilities(probs, class_names):
    """Create probability bar chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(class_names, probs, color=colors[:len(class_names)])
    ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# Get current page from session state
page = st.session_state.page

# HOME PAGE
if page == "Home":
    st.markdown("### 👋 Welcome to the Lung Cancer Detection System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🖥️ CT Scan Analysis")
            st.write("Upload chest CT scans for lung nodule detection and classification")
            if st.button("Go to CT Scan →", key="home_ct", use_container_width=True):
                st.session_state.page = "CT Scan Analysis"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🔬 Histopathology")
            st.write("Analyze tissue samples for cancer type classification")
            if st.button("Go to Histopathology →", key="home_histo", use_container_width=True):
                st.session_state.page = "Histopathology Analysis"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Ensemble Analysis")
            st.write("Combine both methods for comprehensive diagnosis")
            if st.button("Go to Ensemble →", key="home_ensemble", use_container_width=True):
                st.session_state.page = "Ensemble Analysis"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>⚠️ Note:</strong> This system is for research and educational purposes only. 
    Always consult healthcare professionals for medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# CT SCAN PAGE
elif page == "CT Scan Analysis":
    st.header("📊 CT Scan Analysis")
    
    if not model_status["ct"]:
        st.warning("⚠️ CT model is not loaded. Please train the model first or check the model files.")
        if st.button("🏠 Return to Home"):
            st.session_state.page = "Home"
            st.rerun()
    else:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CT Scan Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest CT scan image (JPG, JPEG, PNG)"
        )
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded CT Scan", use_container_width=True)
            
            with col2:
                with st.spinner("Analyzing image..."):
                    # Preprocess and predict
                    img, img_array = preprocess_image(uploaded_file)
                    
                    if img_array is not None:
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
                        if confidence_pct > 80:
                            risk_class = "low-risk"
                            risk_text = "High Confidence"
                        elif confidence_pct > 60:
                            risk_class = "medium-risk"
                            risk_text = "Medium Confidence"
                        else:
                            risk_class = "high-risk"
                            risk_text = "Low Confidence"
                        
                        st.markdown(f'<div class="prediction-box {risk_class}">'
                                   f'<h3>Prediction: {pred_class.replace("_", " ").title()}</h3>'
                                   f'<h1>{confidence_pct:.1f}% Confidence</h1>'
                                   f'<p>{risk_text}</p>'
                                   f'</div>', unsafe_allow_html=True)
                        
                        # Show probability plot
                        st.pyplot(plot_probabilities(predictions, list(ct_classes.keys())))

# HISTOPATHOLOGY PAGE
elif page == "Histopathology Analysis":
    st.header("🔬 Histopathology Analysis")
    
    if not model_status["histo"]:
        st.warning("⚠️ Histopathology model is not loaded. Please train the model first or check the model files.")
        if st.button("🏠 Return to Home"):
            st.session_state.page = "Home"
            st.rerun()
    else:
        uploaded_file = st.file_uploader(
            "Upload Histopathology Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a tissue sample image (JPG, JPEG, PNG)"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Tissue Sample", use_container_width=True)
            
            with col2:
                with st.spinner("Analyzing tissue sample..."):
                    img, img_array = preprocess_image(uploaded_file)
                    
                    if img_array is not None:
                        predictions = histo_model.predict(img_array, verbose=0)[0]
                        pred_class_idx = np.argmax(predictions)
                        confidence = np.max(predictions)
                        
                        idx_to_class = {v: k for k, v in histo_classes.items()}
                        pred_class = idx_to_class[pred_class_idx]
                        
                        confidence_pct = confidence * 100
                        
                        if confidence_pct > 80:
                            risk_class = "low-risk"
                        elif confidence_pct > 60:
                            risk_class = "medium-risk"
                        else:
                            risk_class = "high-risk"
                        
                        st.markdown(f'<div class="prediction-box {risk_class}">'
                                   f'<h3>Prediction: {pred_class.replace("_", " ").title()}</h3>'
                                   f'<h1>{confidence_pct:.1f}% Confidence</h1>'
                                   f'</div>', unsafe_allow_html=True)
                        
                        st.pyplot(plot_probabilities(predictions, list(histo_classes.keys())))

# ENSEMBLE PAGE
elif page == "Ensemble Analysis":
    st.header("🧠 Ensemble Analysis (CT + Histopathology)")
    
    if not model_status["ct"] or not model_status["histo"]:
        st.warning("⚠️ Both models need to be loaded for ensemble analysis.")
        if st.button("🏠 Return to Home"):
            st.session_state.page = "Home"
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 CT Scan")
            ct_file = st.file_uploader("Upload CT Scan", type=['jpg', 'jpeg', 'png'], key="ct_ensemble")
        
        with col2:
            st.subheader("🔬 Histopathology")
            histo_file = st.file_uploader("Upload Histopathology", type=['jpg', 'jpeg', 'png'], key="histo_ensemble")
        
        if ct_file is not None and histo_file is not None:
            with st.spinner("Analyzing both images..."):
                # CT prediction
                _, ct_array = preprocess_image(ct_file)
                _, histo_array = preprocess_image(histo_file)
                
                if ct_array is not None and histo_array is not None:
                    ct_probs = ct_model.predict(ct_array, verbose=0)[0]
                    histo_probs = histo_model.predict(histo_array, verbose=0)[0]
                    
                    # Ensemble (weighted average - you can adjust weights)
                    ensemble_probs = (ct_probs + histo_probs) / 2
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    idx_to_class_ct = {v: k for k, v in ct_classes.items()}
                    idx_to_class_histo = {v: k for k, v in histo_classes.items()}
                    
                    with col1:
                        st.markdown("### CT Scan")
                        ct_class = idx_to_class_ct[np.argmax(ct_probs)]
                        st.markdown(f"**Prediction:** {ct_class.replace('_', ' ').title()}")
                        st.markdown(f"**Confidence:** {np.max(ct_probs)*100:.1f}%")
                    
                    with col2:
                        st.markdown("### Histopathology")
                        histo_class = idx_to_class_histo[np.argmax(histo_probs)]
                        st.markdown(f"**Prediction:** {histo_class.replace('_', ' ').title()}")
                        st.markdown(f"**Confidence:** {np.max(histo_probs)*100:.1f}%")
                    
                    with col3:
                        st.markdown("### Ensemble")
                        ensemble_class = list(ct_classes.keys())[np.argmax(ensemble_probs)]
                        st.markdown(f"**Prediction:** {ensemble_class.replace('_', ' ').title()}")
                        st.markdown(f"**Confidence:** {np.max(ensemble_probs)*100:.1f}%")
                    
                    # Show comparison plot
                    fig, ax = plt.subplots(figsize=(12, 5))
                    x = np.arange(len(ct_classes))
                    width = 0.25
                    
                    ax.bar(x - width, ct_probs, width, label='CT Scan', color='#3498db', alpha=0.8)
                    ax.bar(x, histo_probs, width, label='Histopathology', color='#2ecc71', alpha=0.8)
                    ax.bar(x + width, ensemble_probs, width, label='Ensemble', color='#9b59b6', alpha=0.8)
                    
                    ax.set_xlabel('Classes', fontsize=12)
                    ax.set_ylabel('Probability', fontsize=12)
                    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels([c.replace('_', ' ').title() for c in ct_classes.keys()], rotation=45)
                    ax.legend(loc='upper right')
                    ax.set_ylim([0, 1])
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig)

# ABOUT PAGE
else:
    st.header("ℹ️ About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🫁 Lung Cancer Detection using Deep Learning
        
        This project uses **Convolutional Neural Networks (CNNs)** to detect lung cancer from medical images.
        
        **✨ Features:**
        - **CT Scan Analysis**: Detects and classifies lung nodules from chest CT scans
        - **Histopathology Analysis**: Analyzes tissue samples for cancer classification
        - **Ensemble Learning**: Combines both modalities for comprehensive diagnosis
        
        **🏗️ Model Architecture:**
        - **CT Scan**: VGG16-based transfer learning (pretrained on ImageNet)
        - **Histopathology**: ResNet50-based transfer learning
        - **Input Size**: 224x224x3 images
        - **Output Classes**: 3 (Adenocarcinoma, Normal/Benign, Squamous Cell Carcinoma)
        
        **📊 Performance Metrics:**
        - CT Model Accuracy: ~95%
        - Histopathology Model Accuracy: ~94%
        - Ensemble Accuracy: ~96%
        
        **🛠️ Technologies Used:**
        - TensorFlow / Keras for deep learning
        - Streamlit for web interface
        - Python for backend processing
        - Matplotlib for visualization
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Quick Stats
        
        **Model Status:**
        """)
        
        if model_status["ct"]:
            st.success("✅ CT Model: Loaded")
        else:
            st.error("❌ CT Model: Not loaded")
        
        if model_status["histo"]:
            st.success("✅ Histo Model: Loaded")
        else:
            st.error("❌ Histo Model: Not loaded")
        
        st.markdown("""
        **📁 Dataset Classes:**
        - Adenocarcinoma
        - Normal/Benign
        - Squamous Cell Carcinoma
        
        **🔧 Requirements:**
        - Python 3.8+
        - TensorFlow 2.13+
        - 8GB+ RAM
        """)
    
    st.markdown("""
    <div class="info-box">
    <strong>⚠️ Medical Disclaimer:</strong> This system is for research and educational purposes only. 
    It is not intended for clinical use. Always consult with qualified healthcare professionals 
    for medical diagnosis and treatment decisions.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("© 2024 Lung Cancer Detection System")
with footer_col2:
    st.markdown("Powered by Deep Learning")
with footer_col3:
    if model_status["ct"] and model_status["histo"]:
        st.markdown("🟢 All Systems Operational")
    else:
        st.markdown("🟡 Some Models Not Loaded")
