"""
Configuration file for Food Vision Calorie Estimator
Contains all the settings and hyperparameters
"""

class Config:
    # Data paths
    DATA_PATH = "data/food-101/food-101"
    IMAGES_PATH = "data/food-101/food-101/images"
    META_PATH = "data/food-101/food-101/meta"
    
    # Model parameters
    IMAGE_SIZE = (224, 224)  # EfficientNet default size
    BATCH_SIZE = 32
    NUM_CLASSES = 101  # Food-101 has 101 classes
    
    # Training parameters
    EPOCHS = 10  # Start with 10, you can increase later
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Model saving
    MODEL_SAVE_PATH = "models/best_food_model.h5"
    CLASS_NAMES_SAVE_PATH = "models/class_names.json"
    
    # Calorie estimation
    NUTRITION_DATA_PATH = "data/nutrition_data.csv"

# Create config instance
config = Config()