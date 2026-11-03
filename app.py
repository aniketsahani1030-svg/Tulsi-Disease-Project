import os
import io
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 1. Device and class configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['bacterial', 'fungal', 'healthy', 'pests']

# 2. Re-instantiate exact matching MobileNetV2 architecture
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(128, len(class_names))
)

# 3. Load trained weights
MODEL_PATH = "tulsi_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully onto:", device)
else:
    print(f"[ERROR] Could not find {MODEL_PATH} in current directory.")

# 4. Standard validation/eval transform
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Agronomic remedies mapping
REMEDIES = {
    'bacterial': {
        'title': 'Bacterial Leaf Spot',
        'desc': 'Caused by Pseudomonas/Xanthomonas. Characterized by dark brown-black water-soaked lesions with yellow chlorotic halos.',
        'organic': 'Spray cold-pressed Neem Oil (0.5%) emulsion or Panchagavya solution. Prune and destroy heavily infected lower foliage.',
        'chemical': 'Apply Copper Hydroxide (2g/L water) or Streptocycline formulation at early symptom onset. Avoid overhead watering.'
    },
    'fungal': {
        'title': 'Fungal Blight / Mildew',
        'desc': 'Caused by Peronospora/Alternaria. Shows chlorotic leaf patches with dark or greyish sporulation spores under leaf surface.',
        'organic': 'Apply bio-fungicide Trichoderma viride or spray diluted milk/potassium bicarbonate solution. Increase canopy airflow.',
        'chemical': 'Apply Mancozeb (2.5g/L) or Carbendazim (1g/L) foliar spray. Ensure morning watering so leaves dry quickly.'
    },
    'healthy': {
        'title': 'Healthy Foliage',
        'desc': 'Clean, vibrant green foliage with uncompromised vein vascularization. No foliar pathogen lesions detected.',
        'organic': 'Maintain routine organic compost mulching and regular vermicompost nourishment.',
        'chemical': 'No chemical fungicide or bactericide intervention needed.'
    },
    'pests': {
        'title': 'Pest Infestation (Leaf Miners / Aphids)',
        'desc': 'Visible serpentine leaf mines, foliage holes, or leaf curling caused by insect feeding.',
        'organic': 'Install yellow sticky traps near canopy level. Spray Azadirachtin (Neem seed extract) or Beauveria bassiana.',
        'chemical': 'Use systemic foliar insecticides like Imidacloprid (0.3ml/L) only if severe infestation threatens yield.'
    }
}

def validate_leaf_image(pil_img):
    """Automated input validation filter to detect corrupted/blank/non-leaf frames"""
    img_rgb = np.array(pil_img)
    # Check minimum contrast/standard deviation
    if np.std(img_rgb) < 15.0:
        return False, "Image appears completely blank, dark, or uniformly flat."
    return True, "Valid leaf frame"

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        start_time = time.time()
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Quality pre-filter
        is_valid, reason = validate_leaf_image(pil_img)
        if not is_valid:
            return jsonify({
                'valid': False,
                'error': f"Input Validation Rejection: {reason}"
            }), 200

        # Run model inference
        tensor_input = eval_transforms(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor_input)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy() * 100

        latency_ms = round((time.time() - start_time) * 1000, 1)
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        pred_conf = round(float(probs[pred_idx]), 2)

        # Build response payload
        class_probs = {cls_name: round(float(probs[i]), 2) for i, cls_name in enumerate(class_names)}
        
        return jsonify({
            'valid': True,
            'prediction': pred_label,
            'confidence': pred_conf,
            'latency_ms': latency_ms,
            'probabilities': class_probs,
            'remedy': REMEDIES[pred_label]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Starts server on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)