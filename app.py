from flask import Flask, render_template, request, jsonify
import os
import sys
from werkzeug.utils import secure_filename

sys.path.append('src')
from predict import FoodPredictor

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
predictor = FoodPredictor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            predictions, original_img = predictor.predict(filepath, top_k=3)
            return jsonify({'success': True, 'predictions': predictions, 'image_url': f'/static/uploads/{filename}'})
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'})
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    print('Starting Food Vision Web App...')
    print('Open http://localhost:5000 in your browser')
    app.run(debug=True)
