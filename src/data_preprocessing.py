"""
Data preprocessing utilities for Food-101 dataset
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import config

class FoodDataPreprocessor:
    def __init__(self):
        self.data_path = config.DATA_PATH
        self.img_size = config.IMAGE_SIZE
        self.classes = []
        
    def load_classes(self):
        """Load food classes from meta file"""
        try:
            classes_path = os.path.join(config.META_PATH, 'classes.txt')
            print(f"Loading classes from: {classes_path}")
            
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            print(f"Successfully loaded {len(self.classes)} classes")
            return self.classes
            
        except Exception as e:
            print(f"Error loading classes: {e}")
            return []
    
    def create_data_generators(self, batch_size=None, validation_split=None):
        """Create data generators for training and validation"""
        
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        if validation_split is None:
            validation_split = config.VALIDATION_SPLIT
            
        print("Creating data generators...")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {batch_size}")
        print(f"Validation split: {validation_split}")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            config.IMAGES_PATH,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            config.IMAGES_PATH,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        
        return train_generator, validation_generator

# Test function
def test_preprocessor():
    print("🧪 Testing data preprocessor...")
    preprocessor = FoodDataPreprocessor()
    classes = preprocessor.load_classes()
    print("First 10 classes:", classes[:10])
    return preprocessor

if __name__ == "__main__":
    test_preprocessor()