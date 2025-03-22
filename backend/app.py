# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'jpg'}
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://localhost:8000/predict')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/test', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f'Image saved to {filepath}')
        
        with open(filepath, 'rb') as img_file:
            files = {'image': (filename, img_file, 'image/jpg')}
            response = requests.post(MODEL_SERVICE_URL, files=files)
        
        if response.status_code != 200:
            logger.error(f'Model service returned error: {response.text}')
            return jsonify({'error': 'Model service error'}), 500
        
        result = response.json()
        # Return the combined predictions
        return jsonify({
            'llm_prediction': {
                'class_name': result['llm']['class_name'], 
                'confidence': result['llm']['confidence'] * 100
            },
            'mobilenet_prediction': {
                'class_name': result['mobilenet']['class_name'],
                'confidence': result['mobilenet']['confidence'] * 100
            }
        })
    
    except Exception as e:
        logger.exception('Error during prediction')
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
