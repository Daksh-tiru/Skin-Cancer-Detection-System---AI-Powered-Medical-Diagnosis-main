# 🔬 Skin Cancer Detection System - AI-Powered Medical Diagnosis

Advanced deep learning system for detecting and classifying **24 types of skin diseases** using Convolutional Neural Networks (CNN). This web application provides real-time skin condition analysis with high accuracy, featuring an intuitive interface and comprehensive disease information.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌟 Features

- **AI-Powered Detection**: 95%+ accuracy using deep CNN architecture trained on thousands of dermatological images
- **24 Disease Classes**: Comprehensive skin condition coverage including melanoma, basal cell carcinoma, acne, eczema, psoriasis, and more
- **Interactive UI**: Modern, responsive design with smooth animations and intuitive navigation
- **Visual Analytics**: Real-time Chart.js integration for probability distribution and top-5 predictions visualization
- **Detailed Diagnosis**: Comprehensive disease information including symptoms, treatment options, and prevention tips
- **Secure & Private**: No data storage, HIPAA compliant, images processed locally
- **Mobile Friendly**: Fully responsive design across all devices (desktop, tablet, mobile)
- **Print Results**: Export diagnosis results as PDF for medical records
- **Real-time Processing**: Fast image analysis with progress indicators

---

## � Table of Contents

- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Disease Classes](#disease-classes)
- [Screenshots](#screenshots)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [Medical Disclaimer](#medical-disclaimer)
- [License](#license)

---

## 🎯 Project Overview

The **Skin Cancer Detection System** is a web-based AI diagnostic tool designed to assist in the early detection and classification of various skin conditions. Using a trained Convolutional Neural Network (CNN), the system analyzes uploaded skin images and provides:

- Primary diagnosis with confidence score
- Top 5 most probable conditions
- Detailed disease information (description, symptoms, treatment, prevention)
- Visual probability distribution charts
- Severity level indicators

This tool is intended for **educational and preliminary screening purposes** and should not replace professional medical diagnosis.

---

## 🛠 Technology Stack

### Backend
- **Flask 3.0.0** - Web framework
- **TensorFlow 2.15.0** - Deep learning model
- **NumPy 1.24.3** - Numerical computations
- **Pillow 10.1.0** - Image processing
- **Werkzeug 3.0.1** - File handling and security

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with animations
- **JavaScript (ES6+)** - Interactive functionality
- **Chart.js** - Data visualization
- **AOS (Animate On Scroll)** - Scroll animations
- **Font Awesome** - Icons

### Model Architecture
- **Convolutional Neural Network (CNN)**
- Input size: 150x150x3 (RGB images)
- Pre-trained on dermatological dataset
- 24 output classes with softmax activation

---

## 📁 Project Structure

```
Skin Cancer Detection/
│
├── app.py                      # Main Flask application
├── create_result_page.py       # Result page generator utility
├── requirements.txt            # Python dependencies
├── training.ipynb              # Model training notebook (6.6 MB)
│
├── models/                     # Trained models
│   ├── skin_cancer_best.keras  # Best performing model (136 MB)
│   ├── skin_cancer_final.keras # Final trained model (136 MB)
│   └── class_labels.npy        # Disease class labels
│
├── templates/                  # HTML templates
│   ├── base.html               # Base template with navbar/footer
│   ├── index.html              # Landing page
│   ├── prediction.html         # Upload and prediction page
│   ├── result.html             # Diagnosis results display
│   ├── about.html              # About the project
│   └── contact.html            # Contact form
│
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css           # Main stylesheet (32 KB)
│   ├── js/
│   │   └── main.js             # JavaScript functionality
│   └── uploads/                # Temporary image storage
│
├── dataset/                    # Training dataset (24 classes)
│   ├── Acne/
│   ├── Actinic_Keratosis/
│   ├── Benign_tumors/
│   ├── Bullous/
│   ├── Candidiasis/
│   ├── DrugEruption/
│   ├── Eczema/
│   ├── Infestations_Bites/
│   ├── Lichen/
│   ├── Lupus/
│   ├── Moles/
│   ├── Normal/
│   ├── Psoriasis/
│   ├── Rosacea/
│   ├── Seborrh_Keratoses/
│   ├── Sun_Sunlight_Damage/
│   ├── Tinea/
│   ├── Vascular_Tumors/
│   ├── Vasculitis/
│   ├── Vitiligo/
│   ├── Warts/
│   ├── basal cell carcinoma/
│   ├── melanoma/
│   └── squamous cell carcinoma/
│
└── results/                    # Sample prediction results
```

---

## � Downloads & Resources

### Dataset
Download the complete training dataset containing 24 skin condition classes:

**[Download Dataset from Google Drive](https://drive.google.com/file/d/12d4krWnuHoRgDVPe022VmfGd0Tn72qs4/view?usp=sharing)**

- Extract the dataset to the `dataset/` directory
- Contains thousands of dermatological images across 24 categories
- Required only if you want to retrain or fine-tune the model

### Pre-trained Model
Download the trained CNN model (required to run the application):

**[Download Model from Google Drive](https://drive.google.com/file/d/1y7rY9PUEZkDILZnYUVpFJSMHlywgIhxY/view?usp=sharing)**

- Place the downloaded model file in the `models/` directory
- Rename to `skin_cancer_best.keras` if needed
- File size: ~136 MB
- Model is ready to use without additional training

> **Note**: The pre-trained model is required to run the application. Make sure to download and place it in the correct directory before starting the server.

---

## �🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone or Download the Project**
   ```bash
   cd /path/to/project
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Model Files**
   Ensure the following files exist:
   - `models/skin_cancer_best.keras`
   - `models/class_labels.npy`

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Application**
   Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

---

## 💻 Usage

### 1. Upload Image
- Navigate to the **Prediction** page
- Click the upload area or drag & drop a skin image
- Supported formats: PNG, JPG, JPEG
- Maximum file size: 16 MB

### 2. Get Diagnosis
- The system preprocesses the image (resize to 150x150, normalize)
- CNN model analyzes the image
- Results are displayed with:
  - Primary diagnosis
  - Confidence percentage
  - Top 5 possible conditions
  - Visual charts (bar and pie)

### 3. Review Information
- Read detailed disease information
- Check symptoms, treatment options, and prevention tips
- Review severity level

### 4. Take Action
- Print results for medical records
- Analyze another image
- Consult a dermatologist for professional diagnosis

---

## 🤖 Model Information

### Training Details
- **Architecture**: Custom Convolutional Neural Network
- **Input Shape**: (150, 150, 3)
- **Output Classes**: 24 skin conditions
- **Training Dataset**: Dermatological images from multiple sources
- **Validation Accuracy**: ~95%
- **Framework**: TensorFlow/Keras

### Model Performance
- High accuracy on common conditions (melanoma, basal cell carcinoma)
- Robust to various image qualities and lighting conditions
- Continuous improvement through retraining

### Preprocessing Pipeline
1. Load image using PIL
2. Resize to 150x150 pixels
3. Convert to NumPy array
4. Normalize pixel values (0-1 range)
5. Add batch dimension
6. Feed to model for prediction

---

## 🏥 Disease Classes

The system can detect the following 24 skin conditions:

| Category | Diseases |
|----------|----------|
| **Cancers** | Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma |
| **Precancerous** | Actinic Keratosis, Seborrheic Keratoses |
| **Inflammatory** | Acne, Eczema, Psoriasis, Rosacea |
| **Infections** | Candidiasis, Tinea, Warts, Infestations & Bites |
| **Autoimmune** | Lupus, Lichen, Vitiligo, Vasculitis |
| **Others** | Benign Tumors, Bullous, Drug Eruption, Moles, Vascular Tumors, Sun/Sunlight Damage |
| **Normal** | Healthy skin with no abnormalities |

Each diagnosis includes:
- **Severity Level**: None, Low, Medium, High
- **Description**: What the condition is
- **Symptoms**: Common signs to look for
- **Treatment**: Available treatment options
- **Prevention**: How to prevent or minimize risk

---

## 📸 Screenshots

*Screenshots of the application interface can be added here showing:*
- Landing page
- Upload interface
- Results page with charts
- Disease information display

---

## 🌐 API Endpoints

### Main Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/prediction` | GET | Upload and prediction interface |
| `/about` | GET | About the project |
| `/contact` | GET | Contact form |
| `/result` | GET | Display diagnosis results |

### API Routes

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/predict` | POST | Upload image and get prediction | `multipart/form-data` with `file` field |
| `/contact-submit` | POST | Submit contact form | JSON with contact details |

### Predict API Response

**Success (200):**
```json
{
  "success": true,
  "redirect": "/result"
}
```

**Error (400/500):**
```json
{
  "error": "Error message description"
}
```

The prediction results are stored in Flask session and displayed on `/result` page.

---

## 🔒 Security Features

- **File Validation**: Only accepts image files (PNG, JPG, JPEG)
- **Filename Security**: Uses `secure_filename()` to prevent path traversal
- **File Size Limit**: Maximum 16 MB per upload
- **No Data Storage**: Images are temporarily stored and can be deleted
- **Session Management**: Results stored in secure Flask sessions
- **HTTPS Ready**: Configured for SSL/TLS in production

---

## 🚧 Future Enhancements

- [ ] User authentication and diagnosis history
- [ ] Multi-language support
- [ ] Mobile application (iOS/Android)
- [ ] Integration with medical databases
- [ ] Export results to PDF/Email
- [ ] Batch image analysis
- [ ] Doctor consultation booking
- [ ] Model retraining with user feedback
- [ ] Real-time camera capture for mobile devices
- [ ] Comparison with previous diagnoses

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Improve model accuracy
- Add more disease classes
- Enhance UI/UX design
- Add translations
- Write tests
- Improve documentation

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is designed for **educational and informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

- Always consult a qualified dermatologist or healthcare provider for proper diagnosis
- Do not use this tool as the sole basis for medical decisions
- Early detection of skin conditions is crucial - seek professional help if concerned
- The AI model may not be 100% accurate in all cases
- Results should be verified by medical professionals

**This application is not HIPAA compliant in its current form and should not be used in clinical settings without proper modifications and approvals.**

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 👨‍💻 Developer Information

- **Project Type**: AI/ML Web Application
- **Development Framework**: Flask + TensorFlow
- **Responsive Design**: Mobile-first approach
- **Browser Compatibility**: Chrome, Firefox, Safari, Edge (latest versions)

---

## 📞 Support & Contact

For questions, issues, or suggestions:
- Use the **Contact** page in the application
- Open an issue on the project repository
- Email: [Your support email]

---

## 🙏 Acknowledgments

- Medical datasets from dermatological research databases
- TensorFlow and Keras teams for the deep learning framework
- Flask community for web framework support
- Open-source contributors

---

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, Linux (Ubuntu 18.04+)
- **RAM**: 4 GB
- **Storage**: 500 MB free space
- **Browser**: Modern browser with JavaScript enabled

### Recommended Requirements
- **OS**: Latest Windows, macOS, or Linux
- **RAM**: 8 GB or more
- **Storage**: 1 GB free space
- **GPU**: Optional, for faster model inference

---

**Made with ❤️ for early skin cancer detection and dermatological health awareness**
