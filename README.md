# Multimodal Explainable AI System for Early Parkinson's Disease Identification 🧤🔬

A comprehensive, explainable diagnostic system for early-stage Parkinson's Disease identification using multimodal analysis of Voice and Handwriting data. This project outperforms benchmark metrics from several published studies, reaching **93.33% accuracy** on the UCI voice dataset.

## 🚀 Performance Highlights

- **ML Voice Analysis (UCI Dataset)**: **93.33% Accuracy** (Optimized SVM architecture, exceeding the 89.5% benchmark).
- **Handwriting Analysis**: **99.08% Accuracy** (Fine-tuned ResNet18 with ImageNet weights and data augmentation).
- **DL Voice Analysis**: Neural Network with MC Dropout for uncertainty estimation.

## 🧠 Explainability Features

- **GradCAM 2.0 (Handwriting)**: Targeting the deepest ResNet18 `layer4` for high-resolution diagnostic feature mapping. Includes Gaussian smoothing and medical-grade attention intensity scales.
- **Uncertainty Estimation (DL Voice)**: Utilizes Monte Carlo Dropout to provide probability distributions and confidence scores for neural predictions.
- **Hybrid Feature Interpretability**: Supports both tree-based importance (Random Forest) and variance-based discriminative markers (SVM) for voice diagnostics.
- **Platt Calibration**: Calibrates fused multimodal probabilities for better clinical alignment and risk categorization (Low/Moderate/High).

## 📊 Dashboard Modules

1. **Tab 1: ML Voice Analysis**: Upload a WAV file (sustained vowel sound) for instant SVM assessment with feature interpretability.
2. **Tab 2: DL Voice Analysis**: Neural Network detection with detailed uncertainty and probability distribution charts.
3. **Tab 3: Handwriting CNN**: Upload spiral drawing images for ResNet18 analysis with integrated GradCAM overlays.
4. **Tab 4: Fusion & Report**: Weighted multimodal fusion and automated PDF diagnostic report generation.

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.9+ installed. Install the core dependencies:
```bash
pip install streamlit pandas numpy matplotlib librosa torch torchvision scikit-learn joblib Pillow opencv-python reportlab
```

### 2. Model Initialization
Train and stabilize all three modality models (Voice and Handwriting) by running the training script once:
```bash
python train_all_models.py
```

### 3. Launch the Application
Start the Streamlit dashboard:
```bash
streamlit run parkinsons_app.py
```

## 📂 Project Structure

- `parkinsons_app.py`: The main Streamlit dashboard and inference logic.
- `train_all_models.py`: Optimized training script for all 3 AI models.
- `parkinsons1.csv`: UCI Voice dataset features.
- `sample of voice/`: Healthy and PD audio samples for DL training.
- `Handwritten Samples/`: Spiral drawing imagery for CNN training.
- `*.pth / *.pkl`: Stabilized model artifacts (Auto-generated).

## 📄 License
This project is intended for research and educational purposes in the field of early medical diagnostics and explainable AI.
