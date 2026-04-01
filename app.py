from flask import Flask, render_template, request, jsonify, url_for, session
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ---------------- CONFIG ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (150, 150)

MODEL_PATH = os.path.join(MODELS_FOLDER, 'skin_cancer_best.keras')
LABELS_PATH = os.path.join(MODELS_FOLDER, 'class_labels.npy')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ---------------- #
model = None
class_labels = None

try:
    print("📦 Loading model from:", MODEL_PATH)
    print("📦 Model exists:", os.path.exists(MODEL_PATH))

    print("📦 Loading labels from:", LABELS_PATH)
    print("📦 Labels exist:", os.path.exists(LABELS_PATH))

    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
    class_labels = np.load(LABELS_PATH, allow_pickle=True)

    print("✅ Model and labels loaded successfully!")

except Exception as e:
    print("❌ Model load error:", e)
    model = None
    class_labels = None


# ---------------- HELPERS ---------------- #
def allowed_file(filename):
    return (
        bool(filename)
        and '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def get_disease_info(disease_name):
    info = {
        'melanoma': {
            'severity': 'High',
            'description': 'A serious form of skin cancer.',
            'symptoms': ['Asymmetrical moles', 'Irregular borders', 'Color variation'],
            'treatment': 'Consult a dermatologist immediately for further evaluation and treatment.',
            'prevention': 'Use sunscreen, avoid UV exposure, and do regular skin checks.'
        },
        'basal_cell_carcinoma': {
            'severity': 'Medium',
            'description': 'A common and usually treatable type of skin cancer.',
            'symptoms': ['Pearly bump', 'Flat lesion', 'Bleeding sore'],
            'treatment': 'Usually treated with minor surgical or dermatological procedures.',
            'prevention': 'Sun protection and regular skin monitoring.'
        },
        'Acne': {
            'severity': 'Low',
            'description': 'A common skin condition that causes pimples and inflammation.',
            'symptoms': ['Whiteheads', 'Blackheads', 'Pimples'],
            'treatment': 'Can be managed with skincare, topical medicines, or dermatologist advice.',
            'prevention': 'Keep skin clean and avoid irritating products.'
        },
        'Normal': {
            'severity': 'None',
            'description': 'Healthy skin with no major detected abnormality.',
            'symptoms': ['Clear skin', 'Even texture'],
            'treatment': 'No treatment needed.',
            'prevention': 'Maintain a healthy skincare routine.'
        }
    }

    if disease_name in info:
        return info[disease_name]

    normalized = str(disease_name).lower().replace(" ", "_")
    for key, value in info.items():
        if key.lower() == normalized:
            return value

    return {
        'severity': 'Unknown',
        'description': f'Detected condition: {disease_name}',
        'symptoms': ['Please consult a dermatologist for accurate interpretation.'],
        'treatment': 'Professional medical consultation recommended.',
        'prevention': 'Monitor the skin regularly and seek expert advice if needed.'
    }


# ---------------- ROUTES ---------------- #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or class_labels is None:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG only.'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        predictions = model.predict(img, verbose=0)

        if predictions is None or len(predictions) == 0:
            return jsonify({'error': 'Prediction failed. Empty model output.'}), 500

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = str(class_labels[predicted_index])
        confidence = float(predictions[0][predicted_index]) * 100.0

        top_indices = np.argsort(predictions[0])[::-1][:7]
        top_predictions = [
            {
                'class': str(class_labels[int(idx)]),
                'probability': float(predictions[0][int(idx)]) * 100.0
            }
            for idx in top_indices
        ]

        disease_info = get_disease_info(predicted_class)

        session['prediction_result'] = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'image_path': url_for('static', filename=f'uploads/{unique_filename}'),
            'disease_info': {
                'severity': str(disease_info.get('severity', 'Unknown')),
                'description': str(disease_info.get('description', 'No description available.')),
                'symptoms': [str(x) for x in disease_info.get('symptoms', [])],
                'treatment': str(disease_info.get('treatment', 'Consult a medical professional.')),
                'prevention': str(disease_info.get('prevention', 'Regular monitoring recommended.'))
            },
            'top_predictions': [
                {
                    'class': str(item['class']),
                    'probability': float(item['probability'])
                }
                for item in top_predictions
            ]
        }

        return jsonify({
            'success': True,
            'redirect': url_for('result')
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/result')
def result():
    prediction_result = session.get('prediction_result')

    if not prediction_result:
        return render_template('result.html', no_result=True)

    # Safety defaults so template never crashes
    prediction_result.setdefault('predicted_class', 'Unknown')
    prediction_result.setdefault('confidence', 0.0)
    prediction_result.setdefault('image_path', '')
    prediction_result.setdefault('top_predictions', [])
    prediction_result.setdefault('disease_info', {
        'severity': 'Unknown',
        'description': 'No description available.',
        'symptoms': [],
        'treatment': 'Consult a medical professional.',
        'prevention': 'Regular monitoring recommended.'
    })

    return render_template('result.html', result=prediction_result)


@app.route('/contact-submit', methods=['POST'])
def contact_submit():
    try:
        data = request.get_json(silent=True) or {}
        print("Contact:", data)
        return jsonify({'success': True, 'message': 'Thank you for contacting us!'})
    except Exception as e:
        print("❌ Contact submit error:", e)
        return jsonify({'success': False, 'error': 'Failed to submit contact form.'}), 500


# ---------------- RUN ---------------- #
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)