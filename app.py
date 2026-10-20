import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import html


st.set_page_config(
    page_title="COSMOS | UncertaintyAI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

THRESHOLD = 0.70
MC_PASSES = 30
MODEL_PATH = "resnet18_cifar100.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle", "bowl",
    "boy", "bridge", "bus", "butterfly", "camel", "can",
    "castle", "caterpillar", "cattle", "chair", "chimpanzee",
    "clock", "cloud", "cockroach", "couch", "crab", "crocodile",
    "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest",
    "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster",
    "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate",
    "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray",
    "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree",
    "wolf", "woman", "worm"
]


st.markdown("""
<style>
.stApp {
    background: #F3F7FC;
    color: #172B4D;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    margin-left: auto;
    margin-right: auto;
}

h1, h2, h3, h4, p, label {
    color: #172B4D;
}

h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 27px !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 21px !important;
    font-weight: 700 !important;
}

/* Main title and team details */
.hero {
    text-align: center;
    background: linear-gradient(135deg, #FFFFFF, #E5F0FF);
    border: 1px solid #D5E4F7;
    border-radius: 24px;
    padding: 32px 22px;
    margin-bottom: 24px;
    box-shadow: 0 6px 22px rgba(36, 77, 130, 0.07);
}

.eyebrow {
    color: #2864B4;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero-title {
    color: #153B70;
    font-size: 44px;
    font-weight: 850;
    margin: 12px 0;
}

.hero-subtitle {
    color: #435A78;
    font-size: 18px;
    line-height: 1.6;
}

.team-pill {
    display: inline-block;
    margin-top: 12px;
    padding: 8px 22px;
    border-radius: 30px;
    background: #DCEBFF;
    color: #174A8B;
    font-weight: 800;
    font-size: 17px;
}

.team-members {
    margin-top: 15px;
    color: #334E70;
    font-size: 15px;
    line-height: 2;
}

/* White content panels */
.panel {
    background: #FFFFFF;
    border: 1px solid #DCE6F2;
    border-radius: 18px;
    padding: 23px;
    margin: 12px 0;
    box-shadow: 0 4px 15px rgba(36, 77, 130, 0.045);
}

.section-heading {
    text-align: center;
    color: #174A8B;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 12px;
}

/* Metric cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #D6E4F5;
    border-top: 4px solid #4385D9;
    border-radius: 15px;
    padding: 20px 12px;
    text-align: center;
    min-height: 125px;
    box-shadow: 0 4px 12px rgba(36, 77, 130, 0.05);
}

.metric-label {
    color: #536985;
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 0.6px;
}

.metric-value {
    color: #153B70;
    font-size: 25px;
    font-weight: 850;
    margin-top: 12px;
    overflow-wrap: anywhere;
}

/* Centre uploader and buttons */
[data-testid="stFileUploader"] {
    max-width: 720px;
    margin-left: auto;
    margin-right: auto;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F8FBFF;
    border: 2px dashed #8CB5E8;
    border-radius: 14px;
}

.stButton > button {
    display: block;
    margin: auto;
    background: #245CA6;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: 750;
}

.stButton > button:hover {
    background: #17447F;
    color: white;
    border: none;
}

div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

div[data-testid="stAlert"] {
    border-radius: 13px;
}

div[data-testid="stProgress"] {
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}

div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"],
div[data-testid="stBarChart"] {
    width: 100%;
}

.footer {
    text-align: center;
    color: #60758F;
    font-size: 13px;
    padding: 24px 0 8px;
    line-height: 1.8;
}

hr {
    border-color: #D8E4F2;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="eyebrow">RELIABLE AI • HACKATHON PROJECT</div>
    <div class="hero-title">🔬 UncertaintyAI</div>
    <div class="hero-subtitle">
        Measuring when an AI model is unsure.<br>
        Building more responsible and reliable AI predictions.
    </div>
    <div class="team-pill">✦ TEAM COSMOS ✦</div>
    <div class="team-members">
        <b>Team Members</b><br>
        A S KRITHIKHA &nbsp; • &nbsp; S ADVIKAA<br>
        P SOLEHA SNEGHAL
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, 100)
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model



def predict_image(image, model):
    image_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    image_tensor = image_transform(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    probability_runs = []

    
    model.eval()
    model.fc[0].train()

    try:
        with torch.no_grad():
            for _ in range(MC_PASSES):
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                probability_runs.append(probabilities.cpu())
    finally:
        model.eval()

    stacked = torch.stack(probability_runs)
    mean_probs = stacked.mean(dim=0)[0]

    variance = stacked.var(
        dim=0, unbiased=False
    ).mean().item()

    confidence, predicted_index = mean_probs.max(dim=0)

    entropy = -(
        mean_probs * mean_probs.clamp_min(1e-12).log()
    ).sum().item()

    return (
        CLASS_NAMES[predicted_index.item()],
        confidence.item(),
        entropy,
        variance,
        mean_probs
    )


st.markdown(
    '<div class="section-heading">📷 Image Analysis</div>',
    unsafe_allow_html=True
)

st.markdown("""
<p style="text-align:center;color:#536985;font-size:16px;">
Upload an image to receive a prediction and an uncertainty assessment.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, PNG or WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="visible"
)

st.markdown("""
<div style="
    text-align:center;
    background:#E5F0FF;
    border-radius:12px;
    padding:12px;
    color:#254C7D;
    font-size:14px;
    margin:12px 0 22px;">
    <b>Review policy:</b> Confidence ≥ 70% means prediction accepted;
    confidence &lt; 70% means human review required.
    <br>
    This threshold was selected using validation data from CIFAR-100.
</div>
""", unsafe_allow_html=True)


if uploaded_file is None:
    st.markdown("""
    <div class="panel" style="text-align:center;padding:35px;">
        <div style="font-size:42px;">🧠</div>
        <h3>Ready to analyse</h3>
        <p style="color:#536985;">
            Your prediction, confidence score and uncertainty indicators
            will appear here after you upload an image.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    image = Image.open(uploaded_file).convert("RGB")


    image_left, image_center, image_right = st.columns([1, 2, 1])

    with image_center:
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

    try:
        with st.spinner("Analysing image with Monte Carlo Dropout..."):
            model = load_model()

            predicted_class, confidence, entropy, variance, probs = (
                predict_image(image, model)
            )

        class_display = predicted_class.replace("_", " ").title()

        st.markdown(
            '<div class="section-heading">📊 Prediction Results</div>',
            unsafe_allow_html=True
        )

        
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PREDICTED CLASS</div>
                <div class="metric-value">{html.escape(class_display)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MODEL CONFIDENCE</div>
                <div class="metric-value">{confidence:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            decision = (
                "ACCEPTED"
                if confidence >= THRESHOLD
                else "HUMAN REVIEW"
            )

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SYSTEM DECISION</div>
                <div class="metric-value" style="font-size:20px;">
                    {decision}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        if confidence >= THRESHOLD:
            st.success(
                "✅ Prediction accepted: confidence meets the 70% threshold. "
                "Acceptance does not guarantee that the prediction is correct."
            )
        else:
            st.warning(
                "🧑‍⚖️ Human review required: confidence is below 70%. "
                "The system abstains from automatically accepting this prediction."
            )

        st.markdown(
            "<p style='text-align:center;font-weight:700;'>"
            "Confidence level</p>",
            unsafe_allow_html=True
        )

        st.progress(float(confidence))

        
        st.markdown(
            '<div class="section-heading">🔍 Uncertainty Indicators</div>',
            unsafe_allow_html=True
        )

        u1, u2 = st.columns(2, gap="medium")

        with u1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PREDICTIVE ENTROPY</div>
                <div class="metric-value">{entropy:.4f}</div>
                <p style="color:#536985;font-size:13px;">
                    Higher values indicate more spread-out class probabilities.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with u2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MEAN CLASS VARIANCE</div>
                <div class="metric-value">{variance:.6f}</div>
                <p style="color:#536985;font-size:13px;">
                    Variation across repeated dropout predictions.
                </p>
            </div>
            """, unsafe_allow_html=True)

        
        top_values, top_indices = torch.topk(probs, 5)

        chart_data = pd.DataFrame({
            "Class": [
                CLASS_NAMES[i].replace("_", " ").title()
                for i in top_indices.tolist()
            ],
            "Probability (%)": [
                float(value) * 100
                for value in top_values.tolist()
            ]
        })

        st.markdown(
            '<div class="section-heading">📈 Top 5 Class Probabilities</div>',
            unsafe_allow_html=True
        )

        st.bar_chart(
            chart_data.set_index("Class"),
            horizontal=True,
            height=320,
            use_container_width=True
        )

        st.markdown("""
        <div class="panel" style="text-align:center;">
            <b>How to interpret this result</b>
            <p style="color:#536985;">
                Confidence is the highest average class probability across
                repeated predictions. Entropy and dropout variance provide
                additional uncertainty indicators, but none guarantees
                that a prediction is correct.
            </p>
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(
            "Model file not found. Ensure "
            "'resnet18_cifar100.pth' is in the same folder as app.py."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.markdown("""
<hr>
<div class="footer">
    <b style="color:#174A8B;">COSMOS | UncertaintyAI</b><br>
    ResNet18 • CIFAR-100 • Monte Carlo Dropout<br>
    Reliable AI starts by knowing what it does not know.
</div>
""", unsafe_allow_html=True)