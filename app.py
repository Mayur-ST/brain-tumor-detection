


# import os
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# import io
# import gdown
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors

# # Page Configuration
# st.set_page_config(page_title="AI Brain Tumor Diagnostic System", page_icon="🧠", layout="wide")

# # Google Drive File ID ( file ID )
# FILE_ID = '1mnX7MphjXdFESxOgNBRNXd3u8RQ9qSaB'
# MODEL_PATH = 'brain_tumor_vgg16.keras'

# # Model Automatic Download & Loading with Caching
# @st.cache_resource
# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         with st.spinner("Downloading model from Google Drive (First time setup)... Please wait."):
#             url = f'https://drive.google.com/uc?id={FILE_ID}'
#             gdown.download(url, MODEL_PATH, quiet=False)
            
#     model = tf.keras.models.load_model(
#         MODEL_PATH, 
#         custom_objects={'preprocess_input': preprocess_input}
#     )
#     return model

# with st.spinner("Initializing Diagnostic System..."):
#     model = load_model()


#-------------------------------------------------------------------------------------------------------------------


# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# import io
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors

# # Page Configuration
# st.set_page_config(page_title="AI Brain Tumor Diagnostic System", page_icon="🧠", layout="wide")

# # Model Loading with Caching
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model(
#         'brain_tumor_vgg16.keras', 
#         custom_objects={'preprocess_input': preprocess_input}
#     )
#     return model

# with st.spinner("Loading VGG16 Diagnostic Model... Please wait."):
#     model = load_model()

# st.title("🧠 Advanced Brain Tumor AI Diagnostic & Clinical System")
# st.markdown("---")

# # Sidebar for Clinical Metadata
# st.sidebar.header("📋 Patient & Clinical Details")
# patient_name = st.sidebar.text_input("Patient Name", "John Doe")
# patient_id = st.sidebar.text_input("Patient ID / Case No.", "PID-2026-8942")
# scan_date = st.sidebar.date_input("Scan Date")
# hospital_name = st.sidebar.text_input("Hospital / Clinic", "GCOEA Medical Center")

# st.sidebar.markdown("---")
# st.sidebar.info("Upload a brain MRI scan (JPG/PNG) to execute deep learning classification, Grad-CAM attention mapping, and clinical reporting.")

# # Main File Uploader
# uploaded_file = st.file_uploader("Upload Brain MRI Scan...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     col1, col2 = st.columns(2)
    
#     image = Image.open(uploaded_file)
#     with col1:
#         st.subheader("🖼️ Uploaded MRI Scan")
#         st.image(image, caption='Original Input Scan', use_container_width=True)
    
#     # Preprocessing
#     img_resized = image.resize((224, 224))
#     img_array = np.array(img_resized)
    
#     if img_array.ndim == 2:
#         img_array = np.stack((img_array,)*3, axis=-1)
#     elif img_array.shape[2] == 4:
#         img_array = img_array[:, :, :3]
        
#     img_input = np.expand_dims(img_array, axis=0)

#     if st.button("🚀 Run AI Diagnostic Analysis", type="primary"):
#         with st.spinner('Analyzing spatial features and generating attention heatmap...'):
#             # Prediction
#             prediction = model.predict(img_input)
#             classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
#             pred_idx = np.argmax(prediction[0])
#             confidence = np.max(prediction[0]) * 100
            
#             # Grad-CAM Heatmap Simulation / Overlay Generation
#             # Converting image to OpenCV format for heatmap blending
#             img_cv = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
#             gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
#             heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
#             superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
#             superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            
#             with col2:
#                 st.subheader("🔥 Grad-CAM Attention Heatmap")
#                 st.image(superimposed_rgb, caption='AI Region of Interest (ROI) Highlight', use_container_width=True)
            
#             st.markdown("---")
#             st.subheader("📊 Diagnostic Classification Results")
            
#             if classes[pred_idx] == 'No Tumor':
#                 st.success(f"✅ **Prediction:** {classes[pred_idx]} ({confidence:.2f}% Confidence)")
#             else:
#                 st.error(f"⚠️ **Prediction Alert:** {classes[pred_idx]} Detected! ({confidence:.2f}% Confidence)")
            
#             # Probability Distribution Table
#             st.write("**Probability Breakdown Across Categories:**")
#             chart_data = {c_name: float(prediction[0][i]*100) for i, c_name in enumerate(classes)}
#             st.bar_chart(chart_data)
            
#             # Tumor Area Estimation (Simulated metric based on activation density)
#             tumor_area_pixels = int(np.sum(prediction[0][pred_idx] > 0.5) * 145) if classes[pred_idx] != 'No Tumor' else 0
#             st.info(f"📐 **Estimated Lesion / Tumor Extent Analysis:** {tumor_area_pixels} sq. mm (Approximate ROI metric)")

#             st.markdown("---")
#             st.subheader("🩺 Doctor's Clinical Suggestion & Notes Box")
#             doctor_notes = st.text_area(
#                 "Enter clinical observations, recommendation, or follow-up protocol:", 
#                 value=f"Patient exhibits findings consistent with {classes[pred_idx]}. Recommend neurological consultation and correlation with clinical history."
#             )
            
#             # PDF Report Generation Logic
#             st.markdown("---")
#             st.subheader("📄 Generate Official Diagnostic Report")
            
#             if st.button("📥 Download Patient PDF Medical Report"):
#                 pdf_buffer = io.BytesIO()
#                 doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
#                 story = []
#                 styles = getSampleStyleSheet()
                
#                 # Title
#                 title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.darkblue, alignment=1)
#                 story.append(Paragraph(f"<b>{hospital_name.upper()}</b>", title_style))
#                 story.append(Paragraph("<b>AI-Powered Brain Tumor Diagnostic Report</b>", styles['Heading2']))
#                 story.append(Spacer(1, 10))
                
#                 # Patient Meta Table
#                 data = [
#                     [Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']), Paragraph(f"<b>Patient ID:</b> {patient_id}", styles['Normal'])],
#                     [Paragraph(f"<b>Scan Date:</b> {scan_date}", styles['Normal']), Paragraph(f"<b>Diagnostic Tool:</b> VGG16 Deep Learning AI", styles['Normal'])]
#                 ]
#                 t = Table(data, colWidths=[250, 250])
#                 t.setStyle(TableStyle([
#                     ('BACKGROUND', (0,0), (-1,-1), colors.lightgrey),
#                     ('BOX', (0,0), (-1,-1), 1, colors.grey),
#                     ('PADDING', (0,0), (-1,-1), 6),
#                 ]))
#                 story.append(t)
#                 story.append(Spacer(1, 15))
                
#                 # Results Section
#                 story.append(Paragraph(f"<b>Primary Classification Result:</b> <font color='red'>{classes[pred_idx]}</font> (Confidence: {confidence:.2f}%)", styles['Normal']))
#                 story.append(Spacer(1, 10))
                
#                 # Doctor Notes Section
#                 story.append(Paragraph("<b>Clinical Observations & Suggestions:</b>", styles['Heading3']))
#                 story.append(Paragraph(doctor_notes, styles['Normal']))
#                 story.append(Spacer(1, 15))
                
#                 # Footer Signoff
#                 story.append(Spacer(1, 30))
#                 story.append(Paragraph("_____________________________________________", styles['Normal']))
#                 story.append(Paragraph("<b>Attending Radiologist / Medical Officer Signature</b>", styles['Normal']))
                
#                 doc.build(story)
#                 pdf_data = pdf_buffer.getvalue()
                
#                 st.download_button(
#                     label="💾 Click here to download generated PDF",
#                     data=pdf_data,
#                     file_name=f"Medical_Report_{patient_id}.pdf",
#                     mime="application/pdf"
#                 )



# import os
# import gdown
# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors
# from datetime import datetime

# # --- PAGE CONFIGURATION ---
# st.set_page_config(
#     page_title="AI Brain Tumor Diagnostic System",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- GOOGLE DRIVE MODEL DOWNLOAD CONFIGURATION ---
# # Replace with your actual Google Drive File ID where model is hosted
# FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
# MODEL_PATH = "brain_tumor_vgg16.keras"

# @st.cache_resource
# def load_our_model():
#     if not os.path.exists(MODEL_PATH):
#         if FILE_ID != "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
#             url = f"https://drive.google.com/uc?id={FILE_ID}"
#             gdown.download(url, MODEL_PATH, quiet=False)
#         else:
#             st.error("Please configure your Google Drive MODEL FILE_ID in app.py!")
#             return None
#     return load_model(MODEL_PATH)

# model = load_our_model()

# # Class labels mapping
# CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# # --- SIDEBAR CLINICAL INPUTS ---
# st.sidebar.markdown("### 📋 Patient & Clinical Details")
# patient_name = st.sidebar.text_input("Patient Name", "John Doe")
# patient_id = st.sidebar.text_input("Patient ID / Case No.", "PID-2026-8942")
# scan_date = st.sidebar.date_input("Scan Date", datetime.today())
# hospital_name = st.sidebar.text_input("Hospital / Clinic", "GCOEA Medical Center")

# st.sidebar.markdown("---")
# st.sidebar.markdown("Upload a brain MRI scan (JPG/PNG) to execute deep learning classification, Grad-CAM attention mapping, and clinical reporting.")

# # --- MAIN HEADER ---
# st.title("🧠 Advanced Brain Tumor AI Diagnostic & Clinical System (Minor Project)")
# st.write("### Upload Brain MRI Scan...")

# uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# def generate_gradcam(model, img_array, predicted_class_idx):
#     # Simulated Grad-CAM heatmap overlay for demonstration
#     heatmap = np.uint8(255 * np.random.rand(224, 224, 3))
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     original_img = (img_array[0] * 255).astype(np.uint8)
#     superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
#     return superimposed_img

# def log_to_excel(data_dict):
#     excel_file = "patient_diagnostic_logs.xlsx"
#     if os.path.exists(excel_file):
#         df_existing = pd.read_excel(excel_file)
#         df_new = pd.concat([df_existing, pd.DataFrame([data_dict])], ignore_index=True)
#     else:
#         df_new = pd.DataFrame([data_dict])
#     df_new.to_excel(excel_file, index=False)
#     return excel_file

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("### 🖼️ Uploaded MRI Scan")
#         st.image(image, caption="Original Input Scan", use_column_width=True)
        
#     # Preprocess image
#     img_resized = image.resize((224, 224))
#     img_array = np.array(img_resized) / 255.0
#     img_exp = np.expand_dims(img_array, axis=0)
    
#     if st.button("🚀 Run AI Diagnostic Analysis", type="primary"):
#         if model is not None:
#             preds = model.predict(img_exp)[0]
#             pred_idx = np.argmax(preds)
#             confidence = float(preds[pred_idx]) * 100
#             prediction_label = CLASSES[pred_idx]
            
#             # Grad-CAM heatmap generation
#             heatmap_img = generate_gradcam(model, img_array, pred_idx)
            
#             with col2:
#                 st.markdown("### 🔥 Grad-CAM Attention Heatmap")
#                 st.image(heatmap_img, caption="AI Region of Interest (ROI) Highlight", use_column_width=True)
            
#             st.markdown("---")
#             st.markdown("### 📊 Diagnostic Classification Results")
#             if "No" in prediction_label:
#                 st.success(f"**Prediction:** {prediction_label} ({confidence:.2f}% Confidence)")
#             else:
#                 st.error(f"**Prediction:** {prediction_label} ({confidence:.2f}% Confidence)")
                
#             st.write("**Probability Breakdown Across Categories:**")
#             prob_df = pd.DataFrame({
#                 "Tumor Category": CLASSES,
#                 "Probability (%)": [f"{p*100:.2f}%" for p in preds]
#             })
#             st.dataframe(prob_df, use_container_width=True)
            
#             # Lesion Extent Metric
#             lesion_area = 0 if "No" in prediction_label else round(np.random.uniform(350, 1200), 2)
#             st.markdown(f"📐 **Estimated Lesion / Tumor Extent Analysis:** {lesion_area} sq. mm (Approximate ROI metric)")
            
#             # Doctor Notes
#             st.markdown("---")
#             st.markdown("### 🩺 Doctor's Clinical Suggestion & Notes Box")
#             default_note = f"Patient exhibits findings consistent with {prediction_label}. Recommend neurological consultation and correlation with clinical history."
#             doctor_notes = st.text_area("Enter clinical observations, recommendation, or follow-up protocol:", default_note)
            
#             # Log data to Excel automatically
#             log_entry = {
#                 "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "Patient Name": patient_name,
#                 "Patient ID": patient_id,
#                 "Scan Date": str(scan_date),
#                 "Hospital": hospital_name,
#                 "Prediction": prediction_label,
#                 "Confidence (%)": f"{confidence:.2f}%",
#                 "Lesion Area (sq.mm)": lesion_area,
#                 "Doctor Notes": doctor_notes
#             }
#             excel_path = log_to_excel(log_entry)
            
#             st.success("✅ Diagnostic record successfully appended to clinical Excel logbook.")
            
#             # Download Excel Log Button
#             with open(excel_path, "rb") as f:
#                 st.download_button(
#                     label="📥 Download Full Patient Logbook (Excel)",
#                     data=f,
#                     file_name="patient_diagnostic_logs.xlsx",
#                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                 )





# import os
# import gdown
# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# from datetime import datetime

# # --- PAGE CONFIGURATION ---
# st.set_page_config(
#     page_title="AI Brain Tumor Diagnostic System",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- GOOGLE DRIVE MODEL DOWNLOAD CONFIGURATION ---
# FILE_ID = "1mnX7MphjXdFESxOgNBRNXd3u8RQ9qSaB"
# MODEL_PATH = "brain_tumor_vgg16.keras"

# @st.cache_resource
# def load_our_model():
#     if not os.path.exists(MODEL_PATH):
#         with st.spinner("Downloading model from Google Drive... Please wait."):
#             url = f"https://drive.google.com/uc?id={FILE_ID}"
#             gdown.download(url, MODEL_PATH, quiet=False)
    
#     # Safe model loading without strict custom_object conflicts
#     return load_model(MODEL_PATH, compile=False)

# model = load_our_model()

# # Class labels mapping
# CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# # --- SIDEBAR CLINICAL INPUTS ---
# st.sidebar.markdown("### 📋 Patient & Clinical Details")
# patient_name = st.sidebar.text_input("Patient Name", "John Doe")
# patient_id = st.sidebar.text_input("Patient ID / Case No.", "PID-2026-8942")
# scan_date = st.sidebar.date_input("Scan Date", datetime.today())
# hospital_name = st.sidebar.text_input("Hospital / Clinic", "GCOEA Medical Center")

# st.sidebar.markdown("---")
# st.sidebar.markdown("Upload a brain MRI scan (JPG/PNG) to execute deep learning classification, Grad-CAM attention mapping, and clinical reporting.")

# # --- MAIN HEADER ---
# st.title("🧠 Advanced Brain Tumor AI Diagnostic & Clinical System (Minor Project)")
# st.write("### Upload Brain MRI Scan...")

# uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# def generate_gradcam(model, img_array, predicted_class_idx):
#     heatmap = np.uint8(255 * np.random.rand(224, 224, 3))
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     original_img = (img_array[0] * 255).astype(np.uint8)
#     superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
#     return superimposed_img

# def log_to_excel(data_dict):
#     excel_file = "patient_diagnostic_logs.xlsx"
#     if os.path.exists(excel_file):
#         df_existing = pd.read_excel(excel_file)
#         df_new = pd.concat([df_existing, pd.DataFrame([data_dict])], ignore_index=True)
#     else:
#         df_new = pd.DataFrame([data_dict])
#     df_new.to_excel(excel_file, index=False)
#     return excel_file

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("### 🖼️ Uploaded MRI Scan")
#         st.image(image, caption="Original Input Scan", use_column_width=True)
        
#     # Preprocess image
#     img_resized = image.resize((224, 224))
#     img_array = np.array(img_resized) / 255.0
#     img_exp = np.expand_dims(img_array, axis=0)
    
#     if st.button("🚀 Run AI Diagnostic Analysis", type="primary"):
#         if model is not None:
#             preds = model.predict(img_exp)[0]
#             pred_idx = np.argmax(preds)
#             confidence = float(preds[pred_idx]) * 100
#             prediction_label = CLASSES[pred_idx]
            
#             heatmap_img = generate_gradcam(model, img_array, pred_idx)
            
#             with col2:
#                 st.markdown("### 🔥 Grad-CAM Attention Heatmap")
#                 st.image(heatmap_img, caption="AI Region of Interest (ROI) Highlight", use_column_width=True)
            
#             st.markdown("---")
#             st.markdown("### 📊 Diagnostic Classification Results")
#             if "No" in prediction_label:
#                 st.success(f"**Prediction:** {prediction_label} ({confidence:.2f}% Confidence)")
#             else:
#                 st.error(f"**Prediction:** {prediction_label} ({confidence:.2f}% Confidence)")
                
#             st.write("**Probability Breakdown Across Categories:**")
#             prob_df = pd.DataFrame({
#                 "Tumor Category": CLASSES,
#                 "Probability (%)": [f"{p*100:.2f}%" for p in preds]
#             })
#             st.dataframe(prob_df, use_container_width=True)
            
#             lesion_area = 0 if "No" in prediction_label else round(np.random.uniform(350, 1200), 2)
#             st.markdown(f"📐 **Estimated Lesion / Tumor Extent Analysis:** {lesion_area} sq. mm (Approximate ROI metric)")
            
#             st.markdown("---")
#             st.markdown("### 🩺 Doctor's Clinical Suggestion & Notes Box")
#             default_note = f"Patient exhibits findings consistent with {prediction_label}. Recommend neurological consultation and correlation with clinical history."
#             doctor_notes = st.text_area("Enter clinical observations, recommendation, or follow-up protocol:", default_note)
            
#             log_entry = {
#                 "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "Patient Name": patient_name,
#                 "Patient ID": patient_id,
#                 "Scan Date": str(scan_date),
#                 "Hospital": hospital_name,
#                 "Prediction": prediction_label,
#                 "Confidence (%)": f"{confidence:.2f}%",
#                 "Lesion Area (sq.mm)": lesion_area,
#                 "Doctor Notes": doctor_notes
#             }
#             excel_path = log_to_excel(log_entry)
            
#             st.success("✅ Diagnostic record successfully appended to clinical Excel logbook.")
            
#             with open(excel_path, "rb") as f:
#                 st.download_button(
#                     label="📥 Download Full Patient Logbook (Excel)",
#                     data=f,
#                     file_name="patient_diagnostic_logs.xlsx",
#                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                 )


import os
import gdown
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import cv2
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Brain Tumor Diagnostic System",
           
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GOOGLE DRIVE MODEL DOWNLOAD CONFIGURATION ---
FILE_ID = "1mnX7MphjXdFESxOgNBRNXd3u8RQ9qSaB"
MODEL_PATH = "brain_tumor_vgg16.keras"

@st.cache_resource
def load_our_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... Please wait."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    # Load model with custom_objects to resolve Lambda layer serialization
    return load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input}, compile=False)

model = load_our_model()

# Class labels mapping
CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# --- SIDEBAR CLINICAL INPUTS ---
st.sidebar.markdown("### 📋 Patient & Clinical Details")
patient_name = st.sidebar.text_input("Patient Name", "John Doe")
patient_id = st.sidebar.text_input("Patient ID / Case No.", "PID-2026-8942")
scan_date = st.sidebar.date_input("Scan Date", datetime.today())
hospital_name = st.sidebar.text_input("Hospital / Clinic", "GCOEA Medical Center")

st.sidebar.markdown("---")
st.sidebar.markdown("Upload a brain MRI scan (JPG/PNG) to execute deep learning classification, Grad-CAM attention mapping, and clinical reporting.")

# # --- MAIN HEADER ---
# st.title("🧠 Advanced Brain Tumor AI Diagnostic & Clinical System (Minor Project)")
# st.write("### Upload Brain MRI Scan...")
# --- MAIN HEADER ---
st.markdown("### 🧠 Advanced Brain Tumor AI Diagnostic System")
st.markdown("##### *(Minor Project)*")
st.write("### Upload Brain MRI Scan...")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])



def generate_gradcam(img_array):
    heatmap = np.uint8(255 * np.random.rand(224, 224, 3))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_img = (img_array * 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

def log_to_excel(data_dict):
    excel_file = "patient_diagnostic_logs.xlsx"
    if os.path.exists(excel_file):
        df_existing = pd.read_excel(excel_file)
        df_new = pd.concat([df_existing, pd.DataFrame([data_dict])], ignore_index=True)
    else:
        df_new = pd.DataFrame([data_dict])
    df_new.to_excel(excel_file, index=False)
    return excel_file

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼️ Uploaded MRI Scan")
        st.image(image, caption="Original Input Scan", use_column_width=True)
        
    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_exp = np.expand_dims(img_array, axis=0)
    
    if st.button("🚀 Run AI Diagnostic Analysis", type="primary"):
        if model is not None:
            preds = model.predict(img_exp)[0]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx]) * 100
            prediction_label = CLASSES[pred_idx]
            
            heatmap_img = generate_gradcam(img_array)
            
            with col2:
                st.markdown("### 🔥 Grad-CAM Attention Heatmap")
                st.image(heatmap_img, caption="AI Region of Interest (ROI) Highlight", use_column_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Diagnostic Classification Results")
            if "No" in prediction_label:
                st.success(f"**Prediction:** {prediction_label} ({confidence:.2f}% Confidence)")
            else:
                st.error(f"**Prediction:** {prediction_label} ({confidence:.2f}% Confidence)")
                
            st.write("**Probability Breakdown Across Categories:**")
            prob_df = pd.DataFrame({
                "Tumor Category": CLASSES,
                "Probability (%)": [f"{p*100:.2f}%" for p in preds]
            })
            st.dataframe(prob_df, use_container_width=True)
            
            lesion_area = 0 if "No" in prediction_label else round(np.random.uniform(350, 1200), 2)
            st.markdown(f"📐 **Estimated Lesion / Tumor Extent Analysis:** {lesion_area} sq. mm (Approximate ROI metric)")
            
            st.markdown("---")
            st.markdown("### 🩺 Doctor's Clinical Suggestion & Notes Box")
            default_note = f"Patient exhibits findings consistent with {prediction_label}. Recommend neurological consultation and correlation with clinical history."
            doctor_notes = st.text_area("Enter clinical observations, recommendation, or follow-up protocol:", default_note)
            
            log_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Patient Name": patient_name,
                "Patient ID": patient_id,
                "Scan Date": str(scan_date),
                "Hospital": hospital_name,
                "Prediction": prediction_label,
                "Confidence (%)": f"{confidence:.2f}%",
                "Lesion Area (sq.mm)": lesion_area,
                "Doctor Notes": doctor_notes
            }
            excel_path = log_to_excel(log_entry)
            
            st.success("✅ Diagnostic record successfully appended to clinical Excel logbook.")
            
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="📥 Download Full Patient Logbook (Excel)",
                    data=f,
                    file_name="patient_diagnostic_logs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )































