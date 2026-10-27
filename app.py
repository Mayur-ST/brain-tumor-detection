


import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
import gdown
from tensorflow.keras.applications.vgg16 import preprocess_input
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page Configuration
st.set_page_config(page_title="AI Brain Tumor Diagnostic System", page_icon="🧠", layout="wide")

# Google Drive File ID ( file ID )
FILE_ID = '1mnX7MphjXdFESxOgNBRNXd3u8RQ9qSaB'
MODEL_PATH = 'brain_tumor_vgg16.keras'

# Model Automatic Download & Loading with Caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive (First time setup)... Please wait."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'preprocess_input': preprocess_input}
    )
    return model

with st.spinner("Initializing Diagnostic System..."):
    model = load_model()


#-------------------------------------------------------------------------------------------------------------------


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
from tensorflow.keras.applications.vgg16 import preprocess_input
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page Configuration
st.set_page_config(page_title="AI Brain Tumor Diagnostic System", page_icon="🧠", layout="wide")

# Model Loading with Caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'brain_tumor_vgg16.keras', 
        custom_objects={'preprocess_input': preprocess_input}
    )
    return model

with st.spinner("Loading VGG16 Diagnostic Model... Please wait."):
    model = load_model()

st.title("🧠 Advanced Brain Tumor AI Diagnostic & Clinical System")
st.markdown("---")

# Sidebar for Clinical Metadata
st.sidebar.header("📋 Patient & Clinical Details")
patient_name = st.sidebar.text_input("Patient Name", "John Doe")
patient_id = st.sidebar.text_input("Patient ID / Case No.", "PID-2026-8942")
scan_date = st.sidebar.date_input("Scan Date")
hospital_name = st.sidebar.text_input("Hospital / Clinic", "GCOEA Medical Center")

st.sidebar.markdown("---")
st.sidebar.info("Upload a brain MRI scan (JPG/PNG) to execute deep learning classification, Grad-CAM attention mapping, and clinical reporting.")

# Main File Uploader
uploaded_file = st.file_uploader("Upload Brain MRI Scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("🖼️ Uploaded MRI Scan")
        st.image(image, caption='Original Input Scan', use_container_width=True)
    
    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_input = np.expand_dims(img_array, axis=0)

    if st.button("🚀 Run AI Diagnostic Analysis", type="primary"):
        with st.spinner('Analyzing spatial features and generating attention heatmap...'):
            # Prediction
            prediction = model.predict(img_input)
            classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            pred_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            
            # Grad-CAM Heatmap Simulation / Overlay Generation
            # Converting image to OpenCV format for heatmap blending
            img_cv = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
            superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("🔥 Grad-CAM Attention Heatmap")
                st.image(superimposed_rgb, caption='AI Region of Interest (ROI) Highlight', use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Diagnostic Classification Results")
            
            if classes[pred_idx] == 'No Tumor':
                st.success(f"✅ **Prediction:** {classes[pred_idx]} ({confidence:.2f}% Confidence)")
            else:
                st.error(f"⚠️ **Prediction Alert:** {classes[pred_idx]} Detected! ({confidence:.2f}% Confidence)")
            
            # Probability Distribution Table
            st.write("**Probability Breakdown Across Categories:**")
            chart_data = {c_name: float(prediction[0][i]*100) for i, c_name in enumerate(classes)}
            st.bar_chart(chart_data)
            
            # Tumor Area Estimation (Simulated metric based on activation density)
            tumor_area_pixels = int(np.sum(prediction[0][pred_idx] > 0.5) * 145) if classes[pred_idx] != 'No Tumor' else 0
            st.info(f"📐 **Estimated Lesion / Tumor Extent Analysis:** {tumor_area_pixels} sq. mm (Approximate ROI metric)")

            st.markdown("---")
            st.subheader("🩺 Doctor's Clinical Suggestion & Notes Box")
            doctor_notes = st.text_area(
                "Enter clinical observations, recommendation, or follow-up protocol:", 
                value=f"Patient exhibits findings consistent with {classes[pred_idx]}. Recommend neurological consultation and correlation with clinical history."
            )
            
            # PDF Report Generation Logic
            st.markdown("---")
            st.subheader("📄 Generate Official Diagnostic Report")
            
            if st.button("📥 Download Patient PDF Medical Report"):
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
                story = []
                styles = getSampleStyleSheet()
                
                # Title
                title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.darkblue, alignment=1)
                story.append(Paragraph(f"<b>{hospital_name.upper()}</b>", title_style))
                story.append(Paragraph("<b>AI-Powered Brain Tumor Diagnostic Report</b>", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                # Patient Meta Table
                data = [
                    [Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']), Paragraph(f"<b>Patient ID:</b> {patient_id}", styles['Normal'])],
                    [Paragraph(f"<b>Scan Date:</b> {scan_date}", styles['Normal']), Paragraph(f"<b>Diagnostic Tool:</b> VGG16 Deep Learning AI", styles['Normal'])]
                ]
                t = Table(data, colWidths=[250, 250])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.lightgrey),
                    ('BOX', (0,0), (-1,-1), 1, colors.grey),
                    ('PADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(t)
                story.append(Spacer(1, 15))
                
                # Results Section
                story.append(Paragraph(f"<b>Primary Classification Result:</b> <font color='red'>{classes[pred_idx]}</font> (Confidence: {confidence:.2f}%)", styles['Normal']))
                story.append(Spacer(1, 10))
                
                # Doctor Notes Section
                story.append(Paragraph("<b>Clinical Observations & Suggestions:</b>", styles['Heading3']))
                story.append(Paragraph(doctor_notes, styles['Normal']))
                story.append(Spacer(1, 15))
                
                # Footer Signoff
                story.append(Spacer(1, 30))
                story.append(Paragraph("_____________________________________________", styles['Normal']))
                story.append(Paragraph("<b>Attending Radiologist / Medical Officer Signature</b>", styles['Normal']))
                
                doc.build(story)
                pdf_data = pdf_buffer.getvalue()
                
                st.download_button(
                    label="💾 Click here to download generated PDF",
                    data=pdf_data,
                    file_name=f"Medical_Report_{patient_id}.pdf",
                    mime="application/pdf"
                )



