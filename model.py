import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

st.set_page_config(
    page_title="NeuroScan AI", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    /* Clean Minimalist Styling */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        animation: fadeIn 0.8s ease-in;
        padding-top: 2rem;
        padding-bottom: 100px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Custom Headers for a premium feel */
    .main-title {
        text-align: center;
        color: #F8FAFC;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 5px;
        margin-bottom: 40px;
    }

    /* Medical-grade Footer */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-top: 1px solid rgba(255,255,255,0.05);
        text-align: center;
        padding: 14px;
        color: #CBD5E1;
        font-size: 0.95rem;
        z-index: 1000;
    }
    
    .heart {
        color: #EF4444;
        display: inline-block;
        animation: heartbeat 1.5s infinite;
    }
    
    @keyframes heartbeat {
        0%, 28%, 70% { transform: scale(1); }
        14%, 42% { transform: scale(1.3); }
    }
    
    /* Enhance the file uploader area */
    [data-testid="stFileUploadDropzone"] {
        border: 1px dashed #334155;
        border-radius: 10px;
        background-color: rgba(30, 41, 59, 0.5);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

footer_html = """
<div class="custom-footer">
    Made with love and <span class="heart">❤️</span> by <strong>Rohit, Aayush, Yogesh & Team</strong>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=60) 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### **NeuroScan AI**")
    st.markdown("<p style='color: #94A3B8;'>Diagnostic support tool utilizing deep learning for early-stage Alzheimer's detection via MRI analysis.</p>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("#### **System Info**")
    st.markdown("- **Engine:** TensorFlow 2.15")
    st.markdown("- **Architecture:** InceptionV3")
    st.markdown("- **Input Format:** 224x224 RGB")
    
    st.divider()
    st.caption("© 2026 | Educational Purposes Only.")

st.markdown('<h1 class="main-title">NeuroScan AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced MRI Classification & Analysis System</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('inceptionv3_model.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model Initialization Failed. Ensure 'inceptionv3_model.keras' is present. Details: {e}")
    st.stop()

class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

uploaded_file = st.file_uploader("Upload a top-down brain MRI scan for processing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("") 
    
    col1, col_space, col2 = st.columns([1, 0.1, 1.2]) 
    
    with col1:
        st.markdown("<h4 style='color: #E2E8F0; margin-bottom: 15px;'>Imaging Preview</h4>", unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True, output_format="PNG")

    with col2:
        st.markdown("<h4 style='color: #E2E8F0; margin-bottom: 15px;'>Diagnostic Results</h4>", unsafe_allow_html=True)
        
        with st.spinner('Running neural network inferences...'):
            time.sleep(0.8) 
            
            size = (224, 224) 
            image_resized = image.resize(size)
            image_array = np.array(image_resized)
            tensor_img = tf.convert_to_tensor(image_array, dtype=tf.float32)
            tensor_img = tf.image.resize(tensor_img, [224, 224])
            image_reshaped = tf.expand_dims(tensor_img, axis=0)
            
            prediction = model.predict(image_reshaped)
            confidence = np.max(prediction)
            predicted_class = class_names[np.argmax(prediction)]
            
        st.write("")
        
        if predicted_class == 'Non Demented':
            st.success(f"Classification: **{predicted_class}**")
            st.info("Analysis shows normal brain structure with no significant indicators of dementia.")
        elif predicted_class == 'Very Mild Demented':
            st.warning(f"Classification: **{predicted_class}**")
            st.info("Analysis detected slight anomalies. Clinical correlation recommended.")
        else:
            st.error(f"Classification: **{predicted_class}**")
            st.error("Analysis detected significant structural markers associated with Alzheimer's.")
        
        st.write("")
        st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")
        st.progress(float(confidence)) 
        
        st.write("")
        with st.expander("View Raw Output Tensors"):
            st.code(f"""
[Class 0] Mild Demented:      {prediction[0][0]*100:.4f}%
[Class 1] Moderate Demented:  {prediction[0][1]*100:.4f}%
[Class 2] Non Demented:       {prediction[0][2]*100:.4f}%
[Class 3] Very Mild Demented: {prediction[0][3]*100:.4f}%
            """, language="text")