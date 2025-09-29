"""
Prediction script for Food Recognition Model
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import json
from config import config
from calorie_estimation import CalorieEstimator

class FoodPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.MODEL_SAVE_PATH
        
        # Load class names
        with open(config.CLASS_NAMES_SAVE_PATH, 'r') as f:
            self.class_names = json.load(f)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Initialize calorie estimator
        self.calorie_estimator = CalorieEstimator()
    
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        img = image.load_img(img_path, target_size=config.IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        return img_array, img
    
    def predict(self, img_path, top_k=5):
        """Predict food class from image"""
        print(f"🔍 Analyzing image: {os.path.basename(img_path)}")
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(img_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_probs = predictions[0]
        
        # Get top K predictions
        top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
        top_predictions = []
        
        print(f"🏆 Top {top_k} Predictions:")
        print("-" * 50)
        
        for i, idx in enumerate(top_indices):
            class_name = self.class_names[idx]
            confidence = predicted_probs[idx]
            
            # Get calorie estimation
            nutrition_info = self.calorie_estimator.get_nutrition_info(
                class_name, confidence
            )
            
            prediction_info = {
                'rank': i + 1,
                'class_name': class_name,
                'display_name': class_name.replace('_', ' ').title(),
                'confidence': float(confidence),
                'calories': nutrition_info['calories']['estimated'],
                'calorie_range': nutrition_info['calories']['range']
            }
            
            top_predictions.append(prediction_info)
            
            # Print results
            print(f"{i+1}. {prediction_info['display_name']}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Estimated Calories: {prediction_info['calories']} ({prediction_info['calorie_range']})")
            print()
        
        return top_predictions, original_img

def main():
    """Test the predictor"""
    predictor = FoodPredictor()
    
    # Test with sample images from your dataset
    # You can use images from the food-101 dataset for testing
    test_images = [
        # Example: "data/food-101/food-101/images/pizza/123.jpg"
        # Add paths to actual images from your dataset
    ]
    
    if test_images:
        for img_path in test_images:
            if os.path.exists(img_path):
                predictor.predict(img_path)
            else:
                print(f"❌ Image not found: {img_path}")
    else:
        print("📝 No test images specified.")
        print("💡 To test, find some images in your food-101 dataset and add their paths here")

if __name__ == "__main__":
    main()