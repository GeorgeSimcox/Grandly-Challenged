import os
from flask import Flask, render_template, request, jsonify
import base64
from model.model import predict

app = Flask(__name__)

# Update this path to match your model location
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model_fold_0.pth')

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    """
    Process image from camera and return predictions
    
    Expects a JSON with a base64-encoded image
    Returns a JSON with prediction results
    """
    if request.method == 'POST':
        # Get the image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        try:
            # Extract the base64 encoded image
            image_data = data['image']
            
            # Make prediction
            result = predict(image_data, MODEL_PATH, is_base64=True)
            
            return jsonify(result)
        
        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Please place your model file in the 'model' directory")
    
    # Run the Flask app in debug mode (change to False for production)
    app.run(debug=True, host='0.0.0.0', port=5002)