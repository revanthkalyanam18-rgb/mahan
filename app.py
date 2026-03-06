import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from groq import Groq

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="AI Malaria Detection", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Custom CSS (Modern Medical Theme)
# -------------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f4f7f6;
        color: #333;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #0056b3 0%, #007bff 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 5px;
    }

    /* Card Containers */
    .stMarkdown {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 5px;
        border: 1px solid #ddd;
    }

    /* Result Badges */
    .result-badge {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        color: white;
        margin-bottom: 20px;
    }
    .bg-success { background-color: #28a745; }
    .bg-danger { background-color: #dc3545; }
    .bg-warning { background-color: #ffc107; color: #333; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 0.9rem;
        margin-top: 40px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header Section
# -------------------------
st.markdown('<div class="main-header"><h1>🧬 AI Malaria Detection System</h1><p>Advanced Microscopic Blood Cell Analysis</p></div>', unsafe_allow_html=True)

# -------------------------
# Sidebar / Model Loading
# -------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    st.info("Model: ResNet18 (Malaria Dataset)")
    st.info("Status: Ready")
    
    # Use cache_resource for model loading to prevent reloading on every interaction
    @st.cache_resource
    def load_model():
        try:
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            # Check if file exists to prevent crash
            if os.path.exists("training/malaria_model.pth"):
                model.load_state_dict(torch.load("training/malaria_model.pth", map_location="cpu"))
            else:
                st.error("Model file not found! Please ensure 'training/malaria_model.pth' exists.")
                return None
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_model()
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.write("This system uses Deep Learning to detect Malaria parasites in blood smear images.")

# -------------------------
# Patient Information
# -------------------------
st.subheader("👤 Patient Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    name = st.text_input("Patient Name", placeholder="e.g. John Doe")
with col2:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
with col3:
    blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
with col4:
    test_date = st.date_input("Test Date", value=datetime.date.today())

# -------------------------
# Image Upload
# -------------------------
st.subheader("🔬 Sample Upload")

uploaded_file = st.file_uploader("Upload Blood Cell Image", type=["jpg", "png", "jpeg"])

client = GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#------------------------
def generate_llama_analysis(name, age, blood_group, test_date, result, conf, severity):

    prompt = f"""
You are a medical AI assistant generating a malaria diagnostic report.

Patient Information
Name: {name}
Age: {age}
Blood Group: {blood_group}
Test Date: {test_date}

AI Detection Result: {result}
Confidence: {conf}%
Estimated Severity: {severity}

Based on the result, determine the most likely malaria parasite stage
from the following options:

• Ring Stage
• Trophozoite Stage
• Schizont Stage
• Gametocyte Stage

Explain why that stage might occur in this infection.

Generate a structured clinical report with sections:

1. Diagnosis Summary
2. Estimated Parasite Stage
3. Infection Severity Interpretation
4. Possible Symptoms
5. Medical Precautions
6. Recommended Clinical Action

Write about 12–15 lines in a professional medical tone.
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

# -------------------------
# Prediction Logic
# -------------------------
if uploaded_file and model is not None:
    
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Blood Cell Image", use_container_width=True)

    # Processing State
    with st.spinner("🔍 Analyzing cellular structures..."):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

    # Results Processing
    classes = ["Parasitized", "Uninfected"]
    result = classes[pred.item()]
    conf = round(confidence.item() * 100, 2)

    # Severity Logic
    if conf < 60:
        severity = "Low Risk"
        severity_color = "bg-warning"
    elif conf < 85:
        severity = "Moderate Infection"
        severity_color = "bg-warning"
    else:
        severity = "High Infection Risk"
        severity_color = "bg-danger"

    if result == "Parasitized":
        status_color = "bg-danger"
        status_text = "Malaria Parasite Detected"
    else:
        status_color = "bg-success"
        status_text = "No Malaria Detected"
        severity = "None"

    # -------------------------
    # Results Display
    # -------------------------
    st.markdown("---")
    st.subheader("📋 Diagnostic Results")

    # Top Row: Status & Confidence
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.markdown(f"""
        <div class="result-badge {status_color}">
            {status_text}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="result-badge {severity_color}">
            Severity: {severity}
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.metric(label="Confidence Score", value=f"{conf}%")
        st.progress(int(conf))

    # -------------------------
    # Visualizations
    # -------------------------
    st.subheader("📊 Analysis Visualization")
    viz_col1, viz_col2 = st.columns(2)

    # Heatmap Function
    def generate_heatmap(img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        return overlay

    with viz_col1:
        st.markdown("**Probability Distribution**")
        fig = plt.figure(figsize=(5, 4))
        labels = ["Parasitized", "Uninfected"]
        values = probs.numpy()[0]
        plt.bar(labels, values, color=['#dc3545', '#28a745'])
        plt.ylabel("Probability")
        plt.title("Prediction Confidence")
        plt.ylim(0, 1)
        st.pyplot(fig)

    with viz_col2:
        st.markdown("**Attention Heatmap**")
        heatmap_img = generate_heatmap(image)
        st.image(heatmap_img, caption="Highlighted Cell Regions", use_container_width=True)
    # -------------------------
    # Detailed Analysis (Collapsible)
    # -------------------------
    with st.expander("📖 View Detailed Medical Analysis"):

        with st.spinner("Generating AI medical report..."):

            analysis_text = generate_llama_analysis(
                name,
                age,
                blood_group,
                test_date,
                result,
                conf,
                severity
            )

        st.write(analysis_text)
    # -------------------------
    # PDF Generation
    # -------------------------
    st.markdown("---")
    st.subheader("📄 Generate Report")

    def create_pdf(name, age, blood_group, test_date, result, conf, severity):
        filename = "malaria_report.pdf"
        c = canvas.Canvas(filename)
        
        # Logo Handling
        try:
            logo = ImageReader("assets/logo.png")
            c.drawImage(logo, 40, 760, width=80, height=60)
        except:
            pass

        c.setFont("Helvetica-Bold", 22)
        c.drawString(140, 800, "AI MALARIA DIAGNOSTIC REPORT")
        c.setFont("Helvetica", 12)

        c.drawString(50, 740, f"Patient Name: {name}")
        c.drawString(50, 720, f"Age: {age}")
        c.drawString(50, 700, f"Blood Group: {blood_group}")
        c.drawString(50, 680, f"Test Date: {test_date}")

        c.drawString(50, 640, f"Diagnosis Result: {result}")
        c.drawString(50, 620, f"Confidence Level: {conf}%")
        c.drawString(50, 600, f"Infection Severity: {severity}")

        c.drawString(50, 560, "Malaria Overview:")
        c.drawString(50, 540, "Malaria is a life-threatening disease caused by Plasmodium parasites")
        c.drawString(50, 520, "transmitted to humans through the bites of infected mosquitoes.")

        c.drawString(50, 480, "Parasite Development Stages:")
        c.drawString(50, 460, "• Ring Stage – Early infection stage")
        c.drawString(50, 440, "• Trophozoite – Parasite grows inside RBC")
        c.drawString(50, 420, "• Schizont – Parasite multiplies")
        c.drawString(50, 400, "• Gametocyte – Transmissible stage")

        c.drawString(50, 360, "Precautions:")
        c.drawString(50, 340, "• Use mosquito nets and repellents")
        c.drawString(50, 320, "• Seek medical treatment immediately")
        c.drawString(50, 300, "• Avoid stagnant water areas")

        c.drawString(50, 260, "Note:")
        c.drawString(50, 240, "This AI diagnostic system assists in malaria detection.")
        c.drawString(50, 220, "Results should be confirmed by a medical professional.")

        c.save()
        return filename

    if st.button("📥 Download Medical Report (PDF)", type="primary", use_container_width=True):
        if not name:
            st.warning("Please enter the Patient Name before generating the report.")
        else:
            try:
                pdf = create_pdf(name, age, blood_group, test_date, result, conf, severity)
                with open(pdf, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name="malaria_report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown('<div class="footer">© 2023 AI Malaria Detection System | For Educational & Diagnostic Assistance Only</div>', unsafe_allow_html=True)