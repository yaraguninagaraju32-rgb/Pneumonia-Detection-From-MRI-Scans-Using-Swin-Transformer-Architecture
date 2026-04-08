import streamlit as st
import torch
import timm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------------------------------------------
# GRAD-CAM: Windows + Linux compatible import
# -----------------------------------------------------------
try:
    from grad_cam import GradCAM
    from grad_cam.utils.image import show_cam_on_image
    from grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection - Swin Transformer",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------------------------------
# CUSTOM CSS FOR ENHANCED UI
# -----------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .stCard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Prediction result cards */
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-normal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    
    .prediction-pneumonia {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Image containers */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    
    /* Metric boxes */
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 5px;
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        margin: 20px 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# -----------------------------------------------------------
# SWIN TRANSFORMER COMPATIBLE GRAD-CAM
# -----------------------------------------------------------
def reshape_transform(tensor, height=7, width=7):
    """
    Reshape transform for Swin Transformer.
    Swin outputs (B, H*W, C) - we need to reshape to (B, C, H, W) for Grad-CAM
    """
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
    # Bring channels to first dimension (B, C, H, W)
    result = result.permute(0, 3, 1, 2)
    return result


def generate_gradcam(model, tensor, img_resized, target_class=1):
    """
    Generate Grad-CAM heatmap for Swin Transformer
    """
    # For Swin Transformer, we need to target the last layer's normalization
    # and use the reshape transform
    try:
        target_layer = model.layers[-1].blocks[-1].norm2
    except:
        try:
            target_layer = model.layers[-1].blocks[-1].norm1
        except:
            target_layer = model.norm

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]

    # Create heatmap overlay
    heatmap = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

    return grayscale_cam, heatmap


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model(
        "swin_small_patch4_window7_224",
        pretrained=False,
        num_classes=2
    )
    ckpt = torch.load("best_swin_small.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
CLASSES = ["Normal", "Pneumonia"]


# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.markdown("## 🩺 **Pneumonia AI**")
st.sidebar.markdown("---")
st.sidebar.title("📌 Navigation")
tab_choice = st.sidebar.radio(
    "Choose a section:",
    ["🏥 Prediction", "🔥 Grad-CAM Visualization",
        "📊 Metrics Dashboard", "📘 About Model"]
)

# Show device info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Info")
st.sidebar.info(f"**Device:** {DEVICE.upper()}")
st.sidebar.info(f"**Model:** Swin-Small")


# -----------------------------------------------------------
# ---------------------- PREDICTION TAB ----------------------
# -----------------------------------------------------------
if tab_choice == "🏥 Prediction":

    st.title("🏥 Pneumonia Detection – Swin Transformer")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.write("Upload one or more chest X-rays for AI-powered pneumonia detection with visual explanations.")

    uploaded_files = st.file_uploader(
        "📤 Upload Chest X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_files:
        for idx, file in enumerate(uploaded_files):

            st.markdown(f"### 📋 Result for: `{file.name}`")
            st.markdown('<div class="custom-divider"></div>',
                        unsafe_allow_html=True)

            img = Image.open(file).convert("RGB")
            img_np = np.array(img)

            tensor = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = np.argmax(probs)
            pred = CLASSES[pred_idx]
            confidence = probs[pred_idx] * 100

            # Prepare image for Grad-CAM
            img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

            # Generate Grad-CAM (target the predicted class)
            with st.spinner("🔍 Generating Grad-CAM visualization..."):
                grayscale_cam, heatmap = generate_gradcam(
                    model, tensor, img_resized, target_class=pred_idx)

            # Layout: 3 columns for Original, Heatmap, Results
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.markdown("#### 🖼️ Original X-ray")
                st.image(img, width="stretch")

            with col2:
                st.markdown("#### 🔥 AI Focus Area (Grad-CAM)")
                st.image(heatmap, width="stretch")
                st.caption("Red/Yellow areas = High attention regions")

            with col3:
                st.markdown("#### 🩺 Diagnosis Result")

                # Styled prediction result
                if pred == "Normal":
                    st.markdown(f"""
                    <div class="prediction-normal">
                        ✅ {pred}<br>
                        <span style="font-size: 16px;">Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-pneumonia">
                        ⚠️ {pred}<br>
                        <span style="font-size: 16px;">Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("##### 📊 Confidence Breakdown")

                # Progress bars for confidence
                st.markdown("**Normal:**")
                st.progress(float(probs[0]))
                st.caption(f"{probs[0]*100:.2f}%")

                st.markdown("**Pneumonia:**")
                st.progress(float(probs[1]))
                st.caption(f"{probs[1]*100:.2f}%")

            st.markdown("---")

            # Expandable details
            with st.expander("📝 Detailed Analysis"):
                det_col1, det_col2 = st.columns(2)
                with det_col1:
                    st.markdown("**Model Confidence Scores:**")
                    st.json({
                        "Normal": f"{probs[0]*100:.4f}%",
                        "Pneumonia": f"{probs[1]*100:.4f}%",
                        "Predicted Class": pred,
                        "Raw Logits": logits.cpu().numpy().tolist()[0]
                    })
                with det_col2:
                    st.markdown("**Interpretation:**")
                    if pred == "Pneumonia":
                        st.warning("""
                        🔴 The AI has detected patterns consistent with pneumonia.
                        The heatmap shows regions where the model found concerning features.
                        **Please consult a medical professional for proper diagnosis.**
                        """)
                    else:
                        st.success("""
                        🟢 The AI did not detect strong pneumonia indicators.
                        The X-ray appears normal based on the model's analysis.
                        **This is not a medical diagnosis - consult a doctor if concerned.**
                        """)

            st.markdown("<br>", unsafe_allow_html=True)
    else:
        # Show placeholder when no image uploaded
        st.info(
            "👆 Upload chest X-ray images above to get started with AI-powered diagnosis.")


# -----------------------------------------------------------
# ----------------------- GRAD-CAM TAB -----------------------
# -----------------------------------------------------------
if tab_choice == "🔥 Grad-CAM Visualization":

    st.title("🔥 Grad-CAM Heatmap Visualization")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.write("Explore what the Swin Transformer focuses on when detecting pneumonia.")

    uploaded_file = st.file_uploader(
        "📤 Upload an X-ray image",
        type=["jpg", "jpeg", "png"],
        key="gradcam"
    )

    if uploaded_file:

        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        tensor = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

        # Get prediction first
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = np.argmax(probs)
        pred = CLASSES[pred_idx]

        # Resize image for visualization
        img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

        # Option to select which class to visualize
        st.markdown("### 🎯 Visualization Options")
        viz_col1, viz_col2 = st.columns([1, 2])

        with viz_col1:
            target_class = st.radio(
                "Select class to visualize:",
                ["Predicted Class", "Normal (0)", "Pneumonia (1)"],
                index=0
            )

        if target_class == "Predicted Class":
            class_idx = pred_idx
        elif target_class == "Normal (0)":
            class_idx = 0
        else:
            class_idx = 1

        with st.spinner("🔄 Generating Grad-CAM heatmap..."):
            grayscale_cam, heatmap = generate_gradcam(
                model, tensor, img_resized, target_class=class_idx)

        st.markdown("---")
        st.markdown("### 📊 Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🖼️ Original X-ray")
            st.image(img, width="stretch")

        with col2:
            st.markdown("#### 🔥 Grad-CAM Heatmap")
            st.image(heatmap, width="stretch")

        with col3:
            st.markdown("#### 🌡️ Raw Attention Map")
            # Create a colored version of the grayscale cam
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(grayscale_cam, cmap='jet')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Prediction info
        st.markdown("---")
        st.markdown("### 🩺 Model Prediction")

        pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1])

        with pred_col1:
            if pred == "Normal":
                st.markdown("""
                <div class="prediction-normal">
                    ✅ Normal
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-pneumonia">
                    ⚠️ Pneumonia
                </div>
                """, unsafe_allow_html=True)

        with pred_col2:
            st.metric("Normal Confidence", f"{probs[0]*100:.2f}%")

        with pred_col3:
            st.metric("Pneumonia Confidence", f"{probs[1]*100:.2f}%")

        # Explanation
        st.markdown("---")
        with st.expander("ℹ️ How to interpret Grad-CAM"):
            st.markdown("""
            **Understanding the Heatmap:**
            
            - 🔴 **Red/Yellow regions**: High attention - the model strongly focuses on these areas
            - 🔵 **Blue/Purple regions**: Low attention - less important for the prediction
            - 🟢 **Green regions**: Moderate attention
            
            **For Pneumonia Detection:**
            - Look for red/yellow areas in the lung fields
            - Pneumonia typically shows infiltrates, consolidations, or opacities
            - The model should focus on the lung regions, not edges or markers
            
            **Note:** This visualization helps understand model decisions but should not replace professional medical diagnosis.
            """)
    else:
        st.info(
            "👆 Upload a chest X-ray image to visualize where the AI focuses its attention.")


# -----------------------------------------------------------
# ---------------------- METRICS DASHBOARD -------------------
# -----------------------------------------------------------
if tab_choice == "📊 Metrics Dashboard":

    st.title("📊 Model Metrics Dashboard")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.write("Comprehensive training metrics and performance visualization.")

    # Model Summary
    st.markdown("### 🎯 Model Performance Summary")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Accuracy", "90%+", delta="High")
    with metric_col2:
        st.metric("Model", "Swin-Small")
    with metric_col3:
        st.metric("Parameters", "~50M")
    with metric_col4:
        st.metric("Training Time", "~15 min")

    st.markdown("---")

    colA, colB = st.columns(2)

    try:
        with colA:
            st.markdown("### 📈 Training Curves")
            st.image("paper_figures/fig_training_curves.png",
                     width="stretch")
            st.caption("Loss and accuracy progression during training epochs")

        with colB:
            st.markdown("### 📊 Confusion Matrix")
            st.image("paper_figures/fig_confusion.png", width="stretch")
            st.caption("Model predictions vs actual labels")

        st.markdown("---")
        st.markdown("### 📉 ROC Curve")
        roc_col1, roc_col2, roc_col3 = st.columns([1, 2, 1])
        with roc_col2:
            st.image("paper_figures/fig_roc.png", width="stretch")
            st.caption(
                "Receiver Operating Characteristic curve showing model discrimination ability")

    except:
        st.error(
            "⚠️ Metric images not found. Please ensure the 'paper_figures' folder exists with the following files:")
        st.code("""
paper_figures/
├── fig_training_curves.png
├── fig_confusion.png
└── fig_roc.png
        """)


# -----------------------------------------------------------
# -------------------------- ABOUT TAB ------------------------
# -----------------------------------------------------------
if tab_choice == "📘 About Model":

    st.title("📘 About This Project")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
### 🩺 Pneumonia Detection with Swin Transformer

This advanced AI system uses state-of-the-art deep learning for automated pneumonia detection from chest X-rays.

---

### 🏗️ Technical Architecture

| Component | Details |
|-----------|---------|
| **Model** | Swin-Small Transformer (Shifted Window) |
| **Input Size** | 224 × 224 pixels |
| **Dataset** | Chest X-ray Pneumonia Dataset (Kaggle) |
| **Classes** | Normal, Pneumonia |
| **Framework** | PyTorch + timm |

---

### 🔧 Training Details

- **Augmentations**: Albumentations (rotation, flip, brightness, contrast)
- **Sampling**: Weighted sampling for class imbalance
- **Loss Function**: Focal Loss for hard example mining
- **Optimizer**: AdamW with cosine annealing
- **Training Time**: ~15-17 minutes on Google Colab GPU

---

### 🔥 Key Features

| Feature | Description |
|---------|-------------|
| ✅ **High Accuracy** | 90%+ validation accuracy |
| 🔥 **Grad-CAM** | Visual explanations of model focus areas |
| 📊 **Metrics** | Comprehensive training visualizations |
| 🌐 **Cross-Platform** | Windows, Linux, Streamlit Cloud |
| ⚡ **Real-time** | Fast inference with GPU support |

---

### 📚 How Grad-CAM Works

**Gradient-weighted Class Activation Mapping (Grad-CAM)** uses the gradients 
flowing into the final convolutional layer to produce a localization map 
highlighting important regions for prediction.

For **Swin Transformers**, we apply a special reshape transform since the 
architecture uses shifted windows and outputs feature maps in a different 
format than traditional CNNs.

---

### ⚠️ Disclaimer

This tool is for **educational and research purposes only**. 
It is NOT a substitute for professional medical diagnosis. 
Always consult qualified healthcare professionals for medical advice.

---

### 👥 Project

 Advanced Medical Image Analysis with Transformers
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        Built with ❤️ using Streamlit, PyTorch, and Swin Transformer
    </div>
    """, unsafe_allow_html=True)
