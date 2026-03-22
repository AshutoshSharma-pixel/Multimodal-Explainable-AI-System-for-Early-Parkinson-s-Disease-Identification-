import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import io
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Constants
CSV_PATH = "/Users/ashutoshsharma/Desktop/Pakinsons/parkinsons1.csv"
DL_MODEL_PATH = "dl_voice_model.pth"
ML_MODEL_PATH = "ml_voice_model.pkl"
HW_MODEL_PATH = "handwriting_model.pth"

# --- Model Classes ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.network(x)

class HandwritingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        self.gradients = None
        self.activations = None

    def forward(self, x):
        return self.model(x)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

# --- Feature Extraction ---
def extract_22_features(y, sr):
    f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=600)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if len(voiced_f0) < 2:
        return None
    f0_mean = np.mean(voiced_f0)
    f0_max = np.max(voiced_f0)
    f0_min = np.min(voiced_f0)
    periods = 1.0 / (voiced_f0 + 1e-8)
    avg_period = np.mean(periods)
    jitter_pct = np.mean(np.abs(np.diff(periods))) / avg_period * 100
    jitter_abs = np.mean(np.abs(np.diff(periods)))
    rap = np.mean([abs(periods[i] - np.mean(periods[i-1:i+2])) for i in range(1, len(periods)-1)]) / avg_period if len(periods) > 2 else 0
    ppq = np.mean([abs(periods[i] - np.mean(periods[i-2:i+3])) for i in range(2, len(periods)-2)]) / avg_period if len(periods) > 4 else 0
    ddp = rap * 3
    rms = librosa.feature.rms(y=y)[0]
    shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)
    shimmer_db = 20 * np.log10(shimmer + 1e-8)
    apq3 = np.mean([abs(rms[i] - np.mean(rms[max(0,i-1):i+2])) for i in range(1, len(rms)-1)]) / (np.mean(rms) + 1e-8) if len(rms) > 2 else 0
    apq5 = np.mean([abs(rms[i] - np.mean(rms[max(0,i-2):i+3])) for i in range(2, len(rms)-2)]) / (np.mean(rms) + 1e-8) if len(rms) > 4 else 0
    apq11 = np.mean([abs(rms[i] - np.mean(rms[max(0,i-5):i+6])) for i in range(5, len(rms)-5)]) / (np.mean(rms) + 1e-8) if len(rms) > 10 else 0
    dda = apq3 * 3
    nhr = np.std(y) / (np.mean(np.abs(y)) + 1e-8)
    hnr = 20 * np.log10((np.mean(np.abs(y)) + 1e-8) / (np.std(y) + 1e-8))
    ac = np.correlate(y, y, mode='full')[len(y)-1:]
    ac = ac / (ac[0] + 1e-8)
    ac_pos = np.abs(ac[:100]) + 1e-8
    rpde = float(-np.sum(ac_pos * np.log2(ac_pos)) / 100)
    cs = np.cumsum(y - y.mean())
    seg = max(len(y)//20, 1)
    fluct = [np.sqrt(np.mean((cs[i:i+seg] - np.polyval(np.polyfit(np.arange(seg), cs[i:i+seg], 1), np.arange(seg)))**2)) for i in range(0, len(cs)-seg, seg)]
    dfa = float(np.mean(fluct)) if fluct else 0.0
    spread1 = float(np.log(np.std(voiced_f0) + 1e-8))
    spread2 = float(np.var(voiced_f0) / (f0_mean**2 + 1e-8))
    d2 = float(np.log1p(np.sum(np.abs(np.diff(y)) > np.std(y))) / np.log1p(len(y)))
    hist, _ = np.histogram(voiced_f0, bins=30, density=True)
    hist = hist[hist > 0]
    ppe = float(-np.sum(hist * np.log(hist + 1e-8)))
    return [f0_mean, f0_max, f0_min, jitter_pct, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq11, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

def extract_38_features(y, sr):
    mfc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    f0, flag, _ = librosa.pyin(y, fmin=75, fmax=600)
    vf0 = f0[flag]
    f0m, f0s = (np.mean(vf0), np.std(vf0)) if len(vf0) > 0 else (0, 0)
    jit = np.mean(np.abs(np.diff(1.0/(vf0+1e-8)))) / (np.mean(1.0/(vf0+1e-8)) + 1e-8) if len(vf0) > 1 else 0
    shim = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8) if np.mean(rms) > 0 else 0
    hnr = 20*np.log10(np.mean(rms)/(np.std(rms)+1e-8)) if np.mean(rms) > 0 else 0
    d1 = librosa.feature.delta(mfc)
    d2 = librosa.feature.delta(mfc, order=2)
    return np.concatenate([np.mean(mfc,1), np.std(mfc,1), [np.mean(rms), np.mean(zcr), np.mean(cent), np.mean(bw), np.mean(roll), f0m, f0s, jit, shim, hnr, np.mean(d1), np.mean(d2)]])

# --- GUI Application ---
def main():
    st.set_page_config(
        page_title="Parkinson's Detection System", 
        page_icon="🧠", 
        layout="wide"
    )
    st.title("🧠 Multi-Modal Parkinson's Disease Detection System")

    # Disclaimer
    disclaimer = "⚠️ This system is a research prototype. Not for clinical use. Always consult a qualified neurologist."

    # Load Models
    if 'ml_model' not in st.session_state:
        with st.spinner("Loading models..."):
            try:
                # ML
                ml_data = joblib.load(ML_MODEL_PATH)
                st.session_state['ml_model'] = ml_data['model']
                st.session_state['ml_scaler'] = ml_data['scaler']
                st.session_state['ml_accuracy'] = ml_data['accuracy']
                st.session_state['ml_model_name'] = ml_data.get('model_name', 'SVM')
                # DL
                dl_data = torch.load(DL_MODEL_PATH, map_location='cpu', weights_only=False)
                dl_model = MLP(38)
                dl_model.load_state_dict(dl_data['model_state'])
                dl_model.eval()
                st.session_state['dl_model'] = dl_model
                st.session_state['dl_mean'] = dl_data['feature_mean']
                st.session_state['dl_std'] = dl_data['feature_std']
                # HW
                hw_model = HandwritingCNN()
                state_dict = torch.load(HW_MODEL_PATH, map_location='cpu', weights_only=False)
                
                # Fix key mismatch — saved model has raw ResNet keys, wrapper expects 'model.' prefix
                if not any(k.startswith('model.') for k in state_dict.keys()):
                    state_dict = {'model.' + k: v for k, v in state_dict.items()}
                
                hw_model.load_state_dict(state_dict)
                hw_model.eval()
                st.session_state['hw_model'] = hw_model
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.stop()

    # Initialization of probability states
    for key in ['ml_pd_prob', 'dl_pd_prob', 'hw_pd_prob']:
        if key not in st.session_state: st.session_state[key] = None

    tabs = st.tabs(["Tab 1: ML Voice Analysis", "Tab 2: DL Voice Analysis", "Tab 3: Handwriting CNN", "Tab 4: Fusion & Report"])

    # --- TAB 1: ML Voice ---
    with tabs[0]:
        st.header("ML Voice Analysis (UCI Dataset)")
        model_name = st.session_state.get('ml_model_name', 'ML Model')
        st.success(f"✅ {model_name} loaded — CV Accuracy: {st.session_state['ml_accuracy']:.2%}")
        
        up_ml = st.file_uploader("Upload a voice WAV file", type=["wav"], key="up_ml")
        if up_ml:
            if st.button("🔍 Analyze Voice (ML)", key="ml_btn"):
                y, sr = librosa.load(up_ml, sr=None)
                feats = extract_22_features(y, sr)
                if feats is None:
                    st.error("Could not extract F0 from this audio. Please use a sustained vowel sound recording.")
                else:
                    feats_s = st.session_state['ml_scaler'].transform([feats])
                    prob = st.session_state['ml_model'].predict_proba(feats_s)[0]
                    st.session_state['ml_pd_prob'] = prob[1]
                    
                    label = "Parkinson's Detected" if prob[1] > 0.5 else "Healthy"
                    st.markdown(f"### Prediction: {label}")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.barh(["Healthy", "PD"], prob, color=['green', 'red'])
                        ax.set_title("Prediction Probability")
                        st.pyplot(fig)
                        plt.close(fig)
                    with col2:
                        # Feature importance — works for both RF and SVM
                        model = st.session_state['ml_model']
                        feature_names = pd.read_csv(CSV_PATH).drop(['name', 'status'], axis=1).columns.tolist()
                        
                        if hasattr(model, 'feature_importances_'):
                            # Random Forest — use built-in importances
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1][:10]
                            fig2, ax2 = plt.subplots(figsize=(5, 3))
                            ax2.barh([feature_names[i] for i in indices], importances[indices], color='skyblue')
                            ax2.invert_yaxis()
                            ax2.set_title('Top 10 Feature Importances')
                            st.pyplot(fig2)
                            plt.close(fig2)
                        else:
                            # SVM — use permutation importance approximation
                            # Show coefficient-based or just show top features from domain knowledge
                            st.markdown("**Top Discriminative Features (SVM)**")
                            # Use the absolute values of features weighted by their variance
                            scaler = st.session_state['ml_scaler']
                            feature_vars = np.var(scaler.transform(
                                pd.read_csv(CSV_PATH).drop(['name','status'], axis=1).values
                            ), axis=0)
                            top_idx = np.argsort(feature_vars)[::-1][:10]
                            fig2, ax2 = plt.subplots(figsize=(5, 3))
                            ax2.barh([feature_names[top_idx[i]] for i in range(len(top_idx))], feature_vars[top_idx], color='skyblue')
                            ax2.invert_yaxis()
                            ax2.set_title('Top 10 Most Variable Features')
                            st.pyplot(fig2)
                            plt.close(fig2)

        with st.expander("📋 Demo Mode — UCI Dataset Patients"):
            df_demo = pd.read_csv(CSV_PATH)
            p_idx = st.selectbox("Select Patient Index", df_demo.index.tolist())
            row = df_demo.iloc[p_idx]
            st.table(pd.DataFrame(row.drop(['name', 'status'])).T)
            if st.button("🔍 Analyze Voice (Demo ML)"):
                feats = row.drop(['name', 'status']).values
                feats_s = st.session_state['ml_scaler'].transform([feats])
                prob = st.session_state['ml_model'].predict_proba(feats_s)[0]
                st.session_state['ml_pd_prob'] = prob[1]
                label = "Parkinson's Detected" if prob[1] > 0.5 else "Healthy"
                st.markdown(f"### Prediction: {label}")

        st.info(disclaimer)

    # --- TAB 2: DL Voice ---
    with tabs[1]:
        st.header("DL Voice Analysis (Neural Network)")
        st.success("✅ Neural Network loaded — trained on HC_AH + PD_AH samples")
        
        up_dl = st.file_uploader("Upload WAV", type=["wav"], key="up_dl")
        if up_dl:
            if st.button("🔍 Analyze Voice (DL)", key="dl_btn"):
                y, sr = librosa.load(up_dl, sr=22050)
                feats = extract_38_features(y, sr)
                feats_s = (feats - st.session_state['dl_mean']) / (st.session_state['dl_std'] + 1e-8)
                
                # MC Dropout
                model = st.session_state['dl_model']
                model.train() # Enable Dropout
                probs = []
                with torch.no_grad():
                    for _ in range(20):
                        out = torch.softmax(model(torch.FloatTensor(feats_s).unsqueeze(0)), dim=1)
                        probs.append(out.numpy()[0])
                
                mean_prob = np.mean(probs, axis=0)
                std_prob = np.std(probs, axis=0)
                st.session_state['dl_pd_prob'] = mean_prob[1]
                
                label = "Parkinson's Detected" if mean_prob[1] > 0.5 else "Healthy"
                st.markdown(f"### Prediction: {label}")
                st.write(f"Uncertainty: {np.mean(std_prob)*100:.2f}%")
                conf = "High" if np.mean(std_prob) < 0.05 else "Medium" if np.mean(std_prob) < 0.15 else "Low"
                st.write(f"Confidence Level: **{conf}**")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    bars = ax.bar(['Healthy', "Parkinson's"], mean_prob, 
                                  yerr=std_prob, 
                                  color=['#2ecc71', '#e74c3c'], 
                                  capsize=5, width=0.4)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Probability')
                    ax.set_title('DL Voice Prediction Probabilities')
                    for bar, val in zip(bars, mean_prob):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                                f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
                    st.pyplot(fig)
                    plt.close(fig)
        
        st.info(disclaimer)

    # --- TAB 3: Handwriting CNN ---
    with tabs[2]:
        st.header("Handwriting CNN")
        st.success("✅ Handwriting CNN loaded")
        
        up_hw = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="up_hw")
        if up_hw:
            img = Image.open(up_hw).convert("RGB")
            st.image(img, caption="Uploaded Image", width=300)
            
            if st.button("🔍 Analyze Handwriting", key="hw_btn"):
                tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                tensor = tf(img).unsqueeze(0)
                
                # GradCAM implementation
                model = st.session_state['hw_model']
                model.eval()
                target_layer = model.model.layer4[-1]
                
                activations = []
                gradients = []
                
                def forward_hook(module, input, output):
                    activations.append(output.detach())
                
                def backward_hook(module, grad_input, grad_output):
                    gradients.append(grad_output[0].detach())
                
                h1 = target_layer.register_forward_hook(forward_hook)
                h2 = target_layer.register_full_backward_hook(backward_hook)
                
                # Forward pass
                inp_grad = tensor.clone().requires_grad_(True)
                output = model(inp_grad)
                pred_class = output.argmax(dim=1).item()
                prob = torch.softmax(output, dim=1)[0]
                
                # Backward pass on predicted class
                model.zero_grad()
                output[0, pred_class].backward()
                
                h1.remove()
                h2.remove()
                
                # Generate heatmap
                acts = activations[0]  # shape: (1, C, H, W)
                grads = gradients[0]   # shape: (1, C, H, W)
                
                # Global average pool gradients
                weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
                
                # Weighted sum of activations
                cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
                cam = torch.relu(cam)
                cam = cam.squeeze().numpy()
                
                # Normalize
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                
                # Resize to original image size WITHOUT heavy blurring
                original_w, original_h = img.size
                img_display = np.array(img.resize((original_w, original_h)))
                cam_resized = cv2.resize(cam, (original_w, original_h))
                
                # Apply only light smoothing
                cam_resized = cv2.GaussianBlur(cam_resized, (7, 7), 0)
                cam_resized = cam_resized - cam_resized.min()
                cam_resized = cam_resized / (cam_resized.max() + 1e-8)
                
                # Apply jet colormap
                hm_uint8 = np.uint8(255 * cam_resized)
                hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
                
                # Create overlay
                overlay = cv2.addWeighted(img_display, 0.5, hm_color, 0.5, 0)
                
                # Display
                st.session_state['hw_pd_prob'] = prob[1].item()
                label = "Parkinson's" if pred_class == 1 else "Healthy"
                st.markdown(f"### Prediction: {label}")
                st.markdown(f"**Confidence: {prob[pred_class].item():.2%}**")
                
                c1, c2, c3 = st.columns(3)
                c1.image(img_display, caption=f"Original ({original_w}x{original_h})", use_container_width=True)
                c2.image(hm_color, caption="GradCAM Heatmap", use_container_width=True)
                c3.image(overlay, caption="Overlay", use_container_width=True)
                
                # Attention scale bar
                st.markdown("**🎨 Attention Intensity Scale**")
                col_scale, _ = st.columns([1, 1])
                with col_scale:
                    fig_scale, ax_scale = plt.subplots(figsize=(4, 0.35))
                    gradient = np.linspace(0, 1, 256).reshape(1, -1)
                    gradient = np.vstack([gradient, gradient])
                    ax_scale.imshow(gradient, aspect='auto', cmap='jet')
                    ax_scale.set_xticks([0, 128, 255])
                    ax_scale.set_xticklabels(['Low Attention', 'Medium', 'High Attention'])
                    ax_scale.set_yticks([])
                    st.pyplot(fig_scale)
                    plt.close(fig_scale)
                
                col_m, _ = st.columns([1, 2])
                with col_m:
                    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Max Attention | {cam_resized.max():.3f} |
| Mean Attention | {cam_resized.mean():.3f} |
| High Focus Coverage | {(np.sum(cam_resized > 0.5) / cam_resized.size * 100):.1f}% |
""")
        
        st.info(disclaimer)

    # --- TAB 4: Fusion & Report ---
    with tabs[3]:
        st.header("Fusion & Diagnostic Report")
        
        active = []
        if st.session_state['ml_pd_prob'] is not None: active.append(("ML Voice", st.session_state['ml_pd_prob']))
        if st.session_state['dl_pd_prob'] is not None: active.append(("DL Voice", st.session_state['dl_pd_prob']))
        if st.session_state['hw_pd_prob'] is not None: active.append(("Handwriting", st.session_state['hw_pd_prob']))
        
        for name, _ in active: st.write(f"✅ {name} analysis complete")
        
        if len(active) < 2:
            st.info("Complete at least 2 analyses above to see fusion result")
        else:
            # Weighted Fusion
            if len(active) == 3:
                fused = 0.35*st.session_state['ml_pd_prob'] + 0.35*st.session_state['dl_pd_prob'] + 0.30*st.session_state['hw_pd_prob']
            else:
                fused = np.mean([p for _, p in active])
            
            # Platt Calibration
            cal = 1 / (1 + np.exp(-(fused - 0.5) * 4))
            # Entropy
            H = -(cal * np.log2(cal+1e-8) + (1-cal) * np.log2(1-cal+1e-8))
            # Risk
            risk = "Low" if cal < 0.4 else "Moderate" if cal < 0.7 else "High"
            color = "green" if risk == "Low" else "orange" if risk == "Moderate" else "red"
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Prob (Calibrated)", f"{cal:.1%}")
            col2.markdown(f"Risk Level: <span style='color:{color}; font-size:24px; font-weight:bold;'>{risk}</span>", unsafe_allow_html=True)
            col3.metric("System Entropy", f"{H:.3f}")
            
            col1, _ = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                modality_names = [n for n, _ in active] + ["Fused"]
                modality_values = [p for _, p in active] + [cal]
                colors = ['#e74c3c' if v > 0.5 else '#2ecc71' for v in modality_values]
                bars = ax.bar(modality_names, modality_values, color=colors, width=0.4)
                ax.set_ylim(0, 1)
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Decision boundary')
                ax.set_ylabel('PD Probability')
                ax.set_title('Modality Comparison')
                for bar, val in zip(bars, modality_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
                st.pyplot(fig)
                plt.close(fig)
            
            
            p_name = st.text_input("Patient Name", "John Doe")
            if st.button("Generate PDF Report"):
                buf = io.BytesIO()
                c = canvas.Canvas(buf, pagesize=letter)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(100, 750, "Parkinson's Disease Detection Report")
                c.setFont("Helvetica", 12)
                c.drawString(100, 730, f"Patient: {p_name}")
                c.drawString(100, 715, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                y = 680
                c.drawString(100, y, "Analysis Results:")
                for name, prob in active:
                    y -= 20
                    c.drawString(120, y, f"- {name}: {prob*100:.1f}% PD Probability")
                
                y -= 40
                c.setFont("Helvetica-Bold", 12)
                c.drawString(100, y, f"Fused Probability: {cal:.1%}")
                y -= 20
                c.drawString(100, y, f"Risk Level: {risk}")
                y -= 20
                c.drawString(100, y, f"System Entropy: {H:.3f}")
                
                y -= 40
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(100, y, "Clinical Recommendations:")
                y -= 15
                if risk == "High": c.drawString(120, y, "Immediate clinical evaluation and DAT scan recommended.")
                elif risk == "Moderate": c.drawString(120, y, "Sustained clinical monitoring and follow-up in 3 months.")
                else: c.drawString(120, y, "Baseline established. Routine annual checkups.")
                
                c.setFont("Helvetica-Oblique", 8)
                c.drawString(100, 50, "Disclaimer: This report is for research purposes only. Not a medical diagnosis.")
                c.save()
                st.download_button("Download PDF", buf.getvalue(), f"Report_{p_name}.pdf", "application/pdf")

        st.info(disclaimer)

if __name__ == "__main__":
    main()
