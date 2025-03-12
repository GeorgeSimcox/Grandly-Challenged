// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const switchCameraBtn = document.getElementById('switch-camera-btn');
const loadingIndicator = document.getElementById('loading');
const resultsContainer = document.getElementById('results');

// Prediction Result Elements
const plasticTypeEl = document.getElementById('plastic-type');
const confidenceEl = document.getElementById('confidence');
const commonUsesEl = document.getElementById('common-uses');
const recyclabilityEl = document.getElementById('recyclability');
const disposalEl = document.getElementById('disposal');
const environmentalImpactEl = document.getElementById('environmental-impact');
const otherPredictionsList = document.getElementById('other-predictions-list');

// Camera variables
let stream = null;
let facingMode = 'environment'; // Start with back camera by default
let currentStream = null;

// Initialize the application
function init() {
    // Start camera
    startCamera();
    
    // Set up event listeners
    captureBtn.addEventListener('click', captureImage);
    switchCameraBtn.addEventListener('click', toggleCamera);
}

// Start the device camera
async function startCamera() {
    if (currentStream) {
        stopCurrentStream();
    }
    
    try {
        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };
        
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        
        // Wait for video to be ready
        video.onloadedmetadata = () => {
            video.play();
        };
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Error accessing camera. Please make sure you have granted camera permissions and are using a compatible browser.');
    }
}

// Stop the current camera stream
function stopCurrentStream() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
}

// Toggle between front and back cameras
function toggleCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    startCamera();
}

// Capture an image from the video stream
function captureImage() {
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the current video frame on the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert the canvas image to base64
    const imageBase64 = canvas.toDataURL('image/jpeg');
    
    // Show loading indicator
    loadingIndicator.style.display = 'flex';
    
    // Send to the server for prediction
    sendImageForPrediction(imageBase64);
}

// Send the captured image to the server for prediction
async function sendImageForPrediction(imageBase64) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageBase64
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error during prediction:', error);
        loadingIndicator.style.display = 'none';
        alert('Error during prediction. Please try again.');
    }
}

// Display the prediction results
function displayResults(result) {
    // Show results container
    resultsContainer.style.display = 'block';
    
    // Display top prediction
    const topPrediction = result.top_prediction;
    plasticTypeEl.textContent = topPrediction.class;
    confidenceEl.textContent = `${topPrediction.confidence.toFixed(1)}%`;
    
    // Display plastic information
    commonUsesEl.textContent = topPrediction.info.common_uses;
    recyclabilityEl.textContent = topPrediction.info.recyclable;
    disposalEl.textContent = topPrediction.info.disposal;
    environmentalImpactEl.textContent = topPrediction.info.environmental_impact;
    
    // Display other predictions
    otherPredictionsList.innerHTML = '';
    
    // Only show other predictions (skip the top one that's already displayed)
    const otherPredictions = result.top_3_predictions.slice(1);
    
    if (otherPredictions.length > 0) {
        otherPredictions.forEach(prediction => {
            const predictionEl = document.createElement('div');
            predictionEl.className = 'prediction-item';
            predictionEl.innerHTML = `
                <span>${prediction.class}</span>
                <span>${prediction.confidence.toFixed(1)}%</span>
            `;
            
            // Make prediction clickable to view details
            predictionEl.addEventListener('click', () => {
                displayPredictionDetails(prediction);
            });
            
            otherPredictionsList.appendChild(predictionEl);
        });
    } else {
        otherPredictionsList.innerHTML = '<p>No other significant predictions</p>';
    }
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Update the display with details of a selected prediction
function displayPredictionDetails(prediction) {
    plasticTypeEl.textContent = prediction.class;
    confidenceEl.textContent = `${prediction.confidence.toFixed(1)}%`;
    
    commonUsesEl.textContent = prediction.info.common_uses;
    recyclabilityEl.textContent = prediction.info.recyclable;
    disposalEl.textContent = prediction.info.disposal;
    environmentalImpactEl.textContent = prediction.info.environmental_impact;
}

// Initialize the app when the document is loaded
document.addEventListener('DOMContentLoaded', init);

// Clean up when the page is unloaded
window.addEventListener('beforeunload', () => {
    stopCurrentStream();
});