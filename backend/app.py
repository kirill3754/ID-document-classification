# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'jpg'}
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://model-service:8000/predict')

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Image saved to {filepath}")
        
        # Send to model service
        with open(filepath, 'rb') as img_file:
            files = {'image': (filename, img_file, 'image/jpg')}
            response = requests.post(MODEL_SERVICE_URL, files=files)
        
        # Clean up the temporary file
        os.remove(filepath)
        
        if response.status_code != 200:
            logger.error(f"Model service returned error: {response.text}")
            return jsonify({"error": "Model service error"}), 500
        
        return jsonify(response.json())
    
    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
