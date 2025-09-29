"""
Model architecture for Food Recognition using Transfer Learning
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import config

class FoodRecognitionModel:
    def __init__(self, num_classes=None):
        if num_classes is None:
            num_classes = config.NUM_CLASSES
        self.num_classes = num_classes
        self.img_size = config.IMAGE_SIZE
        self.model = None
        
    def build_model(self, use_pretrained=False):
        """Build the food recognition model"""
        
        print("Building model...")
        print(f"Number of classes: {self.num_classes}")
        print(f"Image size: {self.img_size}")
        
        if use_pretrained:
            print("⚠️  Pretrained models having compatibility issues. Training from scratch...")
        
        # Build custom CNN from scratch (more reliable)
        inputs = Input(shape=self.img_size + (3,))
        
        # First Conv Block
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Second Conv Block
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Third Conv Block
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Fourth Conv Block
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Classification Head
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built and compiled successfully!")
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")

# Example usage
if __name__ == "__main__":
    # Test the model
    model_builder = FoodRecognitionModel(num_classes=101)
    model = model_builder.build_model()
    model_builder.summary()