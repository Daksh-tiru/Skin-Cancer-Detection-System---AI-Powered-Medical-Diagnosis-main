from flask import Flask, render_template, request, jsonify, url_for, session
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime
import secrets

app = Flask(__name__)
app.secret_key = 'skincare_ai_super_secret_production_key_123'

# ---------------- CONFIG ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (224, 224)

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
    # EfficientNet naturally expects input in the [0, 255] range, no scaling needed
    return img_array


def get_disease_info(disease_name):
    info = {
        'melanoma': {
            'severity': 'High',
            'description': 'A serious form of skin cancer that begins in melanocytes.',
            'symptoms': ['Asymmetrical moles', 'Irregular borders', 'Color variation', 'Evolving size'],
            'treatment': 'Consult a dermatologist immediately for biopsy and possible excision.',
            'prevention': 'Use sunscreen, avoid UV exposure, and do regular skin checks.'
        },
        'basal_cell_carcinoma': {
            'severity': 'Low to Medium',
            'description': 'A common and usually treatable type of skin cancer.',
            'symptoms': ['Pearly bump', 'Flat flesh-colored lesion', 'Bleeding sore that heals and returns'],
            'treatment': 'Minor surgical or dermatological procedures.',
            'prevention': 'Sun protection and regular skin monitoring.'
        },
        'squamous_cell_carcinoma': {
            'severity': 'High',
            'description': 'A common form of skin cancer that develops in the squamous cells.',
            'symptoms': ['Firm, red nodule', 'Flat sore with a scaly crust'],
            'treatment': 'Surgical excision, Mohs surgery, or radiation therapy. Consult a doctor.',
            'prevention': 'Limit sun exposure, wear protective clothing, use broad-spectrum sunscreen.'
        },
        'Acne': {
            'severity': 'Low',
            'description': 'A common skin condition that causes pimples and inflammation.',
            'symptoms': ['Whiteheads', 'Blackheads', 'Pimples', 'Cysts'],
            'treatment': 'Over-the-counter creams, proper hygiene, or dermatologist-prescribed medication.',
            'prevention': 'Keep skin clean and avoid irritating products.'
        },
        'Normal': {
            'severity': 'None',
            'description': 'Healthy skin with no major detected abnormality.',
            'symptoms': ['Clear skin', 'Even texture'],
            'treatment': 'No treatment needed.',
            'prevention': 'Maintain a healthy skincare routine.'
        },
        'Benign_tumors': {
            'severity': 'Low',
            'description': 'Non-cancerous growths on the skin.',
            'symptoms': ['Painless lump', 'Slow-growing mass'],
            'treatment': 'Often no treatment needed unless bothersome; can be surgically removed.',
            'prevention': 'Generally cannot be prevented; monitor for rapid changes.'
        },
        'Eczema': {
            'severity': 'Low',
            'description': 'A condition that makes your skin red and itchy.',
            'symptoms': ['Dry skin', 'Itching', 'Red to brownish-gray patches'],
            'treatment': 'Moisturizers, topical corticosteroids, and avoiding triggers.',
            'prevention': 'Moisturize regularly, wear soft fabrics, avoid harsh soaps.'
        },
        'Tinea': {
            'severity': 'Low',
            'description': 'A highly contagious fungal infection of the skin (like Ringworm).',
            'symptoms': ['Ring-shaped red rash', 'Itching', 'Scaly skin'],
            'treatment': 'Antifungal creams or oral medication.',
            'prevention': 'Keep skin clean and dry, do not share personal items.'
        },
        'Psoriasis': {
            'severity': 'Low to Medium',
            'description': 'A skin disease that causes red, itchy scaly patches.',
            'symptoms': ['Red patches of skin covered with thick, silvery scales', 'Dry, cracked skin'],
            'treatment': 'Topical treatments, light therapy, and systemic medications.',
            'prevention': 'Manage stress, avoid smoking, and moisturize daily.'
        },
        'Actinic_Keratosis': {
            'severity': 'Medium',
            'description': 'A rough, scaly patch on the skin that develops from years of sun exposure (Precancerous).',
            'symptoms': ['Rough, scaly patch', 'Itching or burning', 'Varies in color (pink, red, or brown)'],
            'treatment': 'Cryotherapy, topical creams, or laser therapy to remove it.',
            'prevention': 'Strict sun protection to prevent progression to cancer.'
        },
        'Vitiligo': {
            'severity': 'Low',
            'description': 'A disease that causes loss of skin color in patches.',
            'symptoms': ['Patchy loss of skin color', 'Premature whitening of hair'],
            'treatment': 'Light therapy, topical corticosteroids, or depigmentation.',
            'prevention': 'Cannot be prevented, but sun protection helps protect the lighter skin.'
        },
        'Warts': {
            'severity': 'Low',
            'description': 'Small, fleshy bumps on the skin or mucous membranes caused by HPV.',
            'symptoms': ['Small, fleshy, grainy bumps', 'Flesh-colored, white, pink or tan'],
            'treatment': 'Salicylic acid, freezing (cryotherapy), or minor surgery.',
            'prevention': 'Avoid direct contact with warts, do not share personal items.'
        },
        'Lichen': {
            'severity': 'Low',
            'description': 'An inflammatory skin condition triggered by the immune system.',
            'symptoms': ['Purplish, itchy, flat-topped bumps', 'Lacy white patches'],
            'treatment': 'Corticosteroid creams, antihistamines, or light therapy.',
            'prevention': 'Exact cause is unknown; manage stress.'
        },
        'DrugEruption': {
            'severity': 'High',
            'description': 'An adverse skin reaction to a drug or medication.',
            'symptoms': ['Red rash', 'Hives', 'Itching', 'Skin blistering in severe cases'],
            'treatment': 'Stop the offending drug immediately and consult a doctor. Antihistamines.',
            'prevention': 'Avoid medications you are allergic to.'
        },
        'Vascular_Tumors': {
            'severity': 'Low',
            'description': 'Abnormal overgrowth of blood vessels (e.g., hemangiomas).',
            'symptoms': ['Red or purplish lump', 'May bleed easily if scratched'],
            'treatment': 'Often left alone if small; laser therapy or surgical removal if problematic.',
            'prevention': 'Generally cannot be prevented.'
        },
        'Infestations_Bites': {
            'severity': 'Low',
            'description': 'Skin irritation caused by insects, mites, or ticks.',
            'symptoms': ['Redness', 'Swelling', 'Itching', 'Puncture marks'],
            'treatment': 'Wash area, apply anti-itch cream or antihistamines.',
            'prevention': 'Use insect repellent, wear protective clothing outdoors.'
        },
        'Bullous': {
            'severity': 'High',
            'description': 'A group of rare diseases that cause fluid-filled blisters on the skin.',
            'symptoms': ['Large, fluid-filled blisters', 'Itchy, red skin'],
            'treatment': 'Corticosteroids, immunosuppressants. Consult a dermatologist.',
            'prevention': 'Cannot be prevented; avoiding trauma to the skin helps.'
        },
        'Vasculitis': {
            'severity': 'High',
            'description': 'Inflammation of the blood vessels causing changes in the skin.',
            'symptoms': ['Purple or red spots (petechiae)', 'Skin ulcers', 'Painful nodules'],
            'treatment': 'Corticosteroids to reduce inflammation. Requires medical attention.',
            'prevention': 'Prompt treatment of underlying infections or autoimmune conditions.'
        },
        'Seborrh_Keratoses': {
            'severity': 'Low',
            'description': 'A common noncancerous skin growth that often appears as you age.',
            'symptoms': ['Waxy, scaly, slightly elevated appearance', 'Brown, black, or light tan'],
            'treatment': 'Usually requires no treatment; can be frozen or scraped off if irritated.',
            'prevention': 'Cannot be entirely prevented; part of natural aging.'
        },
        'Moles': {
            'severity': 'Low',
            'description': 'Common skin growths caused by clusters of pigmented cells.',
            'symptoms': ['Brown, black or skin-colored spots', 'Usually round or oval'],
            'treatment': 'Usually none needed unless they change in size, shape, or color.',
            'prevention': 'Monitor for changes that could indicate melanoma.'
        },
        'Sun_Sunlight_Damage': {
            'severity': 'Low',
            'description': 'Skin damage caused by chronic exposure to UV rays.',
            'symptoms': ['Wrinkles', 'Sunspots', 'Uneven pigmentation', 'Leathery skin'],
            'treatment': 'Topical retinoids, laser therapy, chemical peels.',
            'prevention': 'Use sunscreen daily, wear hats, and avoid peak sun hours.'
        },
        'Lupus': {
            'severity': 'High',
            'description': 'An autoimmune disease that can cause skin rashes, especially a butterfly rash.',
            'symptoms': ['Butterfly-shaped rash on the face', 'Sensitivity to sunlight', 'Red, scaly patches'],
            'treatment': 'Immunosuppressants, antimalarial drugs, and strict sun protection.',
            'prevention': 'Cannot be prevented; avoid UV light to prevent flare-ups.'
        },
        'Rosacea': {
            'severity': 'Low',
            'description': 'A common skin condition that causes redness and visible blood vessels in your face.',
            'symptoms': ['Facial redness', 'Swollen red bumps', 'Eye problems'],
            'treatment': 'Topical drugs that reduce redness, oral antibiotics, laser therapy.',
            'prevention': 'Avoid triggers like spicy foods, hot drinks, and extreme temperatures.'
        },
        'Candidiasis': {
            'severity': 'Low',
            'description': 'A fungal infection caused by a yeast (a type of fungus) called Candida.',
            'symptoms': ['Red rash', 'Itching', 'Small blisters or pustules'],
            'treatment': 'Antifungal creams, ointments, or oral medications.',
            'prevention': 'Keep skin dry and clean, wear breathable clothing.'
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

        top_indices = np.argsort(predictions[0])[::-1][:8]
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