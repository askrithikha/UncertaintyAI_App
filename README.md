# COSMOS | UncertaintyAI
### Uncertainty-Aware Image Classification with Deep Learning

**COSMOS | UncertaintyAI** is an AI-powered image classification project designed to explore a critical challenge in artificial intelligence: a model can make a prediction with high confidence and still be wrong.

By combining **ResNet18** with **Monte Carlo Dropout**, the system estimates predictive uncertainty alongside classification confidence. This information can help identify predictions that may require human review rather than relying solely on a model's predicted label.

---

## 1. Problem Statement

Traditional image classification systems typically return a predicted class and a confidence score. However, confidence alone does not guarantee that a prediction is correct.

In situations where reliability matters, users need additional information to understand when a model's prediction may be uncertain.

## 2. Proposed Solution

COSMOS | UncertaintyAI extends image classification with uncertainty estimation.

The application processes an uploaded image, generates a prediction, and presents confidence and uncertainty-related measurements through an interactive Streamlit interface.

When a prediction falls below a configurable confidence threshold, the system flags it for human review.

The objective is to make model outputs more transparent and support informed human oversight.

---

## 3. Key Features

- **Image Classification:** Classifies images into categories from the CIFAR-100 label set using ResNet18.
- **Monte Carlo Dropout:** Repeats predictions with dropout enabled to estimate variability across model outputs.
- **Confidence Estimation:** Displays the average predicted probability for the most likely class.
- **Uncertainty Indicators:** Presents entropy and prediction variance as complementary indicators of uncertainty.
- **Top-5 Predictions:** Displays the five highest-ranked class probabilities for further inspection.
- **Human-Review Flagging:** Identifies low-confidence predictions using a configurable threshold.
- **Interactive Dashboard:** Provides prediction results and uncertainty information through a Streamlit web interface.

---

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Classification Architecture | ResNet18 |
| Uncertainty Estimation | Monte Carlo Dropout |
| Dataset | CIFAR-100 |
| User Interface | Streamlit |
| Image Processing | Pillow |
| Data Handling | NumPy, Pandas |

---

## 5. How It Works

1. **Image Input:** The user uploads an image through the Streamlit interface.
2. **Preprocessing:** The image is resized, converted to a tensor, and normalized.
3. **Repeated Inference:** The model generates multiple predictions with dropout enabled.
4. **Prediction Aggregation:** The probability outputs are averaged to obtain the final predicted class and confidence.
5. **Uncertainty Estimation:** Entropy and variance are calculated to provide additional information about prediction uncertainty.
6. **Decision Support:** The system compares the prediction confidence with a configurable threshold and indicates whether human review is recommended.
7. **Result Visualization:** The dashboard displays the predicted class, confidence, uncertainty indicators, and top-5 predictions.

---

## 6. Project Structure

```text
COSMOS-UncertaintyAI/
├── app.py
├── requirements.txt
├── README.md
├── resnet18_cifar100.pth
└── notebooks/
    └── model_training.ipynb
```

**Note:** The filenames above are illustrative. Replace them with the actual filenames in your repository. The trained model file must be available at the path expected by the application.

---

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip package manager
- Git
- The trained model weights file

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd COSMOS-UncertaintyAI
```

Replace `<YOUR_REPOSITORY_URL>` with the URL of your GitHub repository.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

Replace `app.py` with your actual Streamlit application filename if necessary.

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

---

## 8. Model Training

The model-training notebook is maintained separately in the `notebooks/` directory.

The project uses ResNet18 for image classification on CIFAR-100. The trained model weights are loaded by the Streamlit application for inference.

The notebook can document the training workflow, model configuration, evaluation, and generation of the saved weights.

---

## 9. Limitations and Considerations

- The model's predictions are limited by its training data, learned representations, and generalization capability.
- Monte Carlo Dropout provides an estimate of predictive variability; it does not guarantee that uncertainty is accurately calibrated.
- Confidence scores should not be interpreted as guarantees of correctness.
- The human-review threshold is configurable and should be evaluated using suitable validation data.
- Performance on arbitrary uploaded images may differ from performance on CIFAR-100 images because of differences in image content and distribution.
- Human review is a decision-support mechanism, not a guarantee that every incorrect prediction will be detected.

---

## 10. Future Enhancements

- Evaluate uncertainty estimation using calibration metrics and selective-classification measures.
- Compare different uncertainty estimation techniques.
- Add batch image evaluation and downloadable reports.
- Improve inference speed through batch processing.
- Introduce model monitoring to identify changes in input data distributions.
- Extend the system to additional datasets and classification tasks.

---

## 11. Team COSMOS

- A S KRITHIKHA
- S ADVIKAA
- P SOLEHA SNEGHAL

---

## 12. Project Objective

COSMOS | UncertaintyAI explores how uncertainty estimation can complement conventional image classification and help users recognize when an AI prediction deserves closer inspection.

**Our principle: Reliable AI should communicate not only what it predicts, but also when its predictions may need further review.**

---

*Developed as an AI/ML project exploring uncertainty-aware prediction and human-in-the-loop decision support.*