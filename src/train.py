"""
Training script for Food Recognition Model with resume capability
"""
import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Import our custom modules
from config import config
from data_preprocessing import FoodDataPreprocessor
from model import FoodRecognitionModel

def setup_directories():
    """Create necessary directories"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    print("Directories setup complete!")

def find_latest_checkpoint():
    """Find the latest checkpoint to resume from"""
    checkpoint_dir = "models"
    if os.path.exists(config.MODEL_SAVE_PATH):
        return config.MODEL_SAVE_PATH
    
    # Look for other checkpoint files
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".h5"):
            return os.path.join(checkpoint_dir, file)
    
    return None

def load_model_and_history(model_builder, checkpoint_path):
    """Load model from checkpoint"""
    print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
    
    # Load the model architecture first
    model = model_builder.build_model()
    
    # Load the weights
    model.load_weights(checkpoint_path)
    print("✅ Checkpoint loaded successfully!")
    
    return model

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_logs/training_history.png')
    plt.show()

def save_class_names(class_names):
    """Save class names to JSON file"""
    with open(config.CLASS_NAMES_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {config.CLASS_NAMES_SAVE_PATH}")

def main():
    """Main training function"""
    print("🚀 Starting Food Recognition Model Training...")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Check if we can resume from checkpoint
    initial_epoch = 0
    checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path:
        print(f"📁 Found existing checkpoint: {checkpoint_path}")
        resume = input("Do you want to resume training from checkpoint? (y/n): ")
        if resume.lower() == 'y':
            # We'll load the model after building the architecture
            initial_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
            print(f"🔄 Resuming from epoch {initial_epoch}")
        else:
            print("🆕 Starting fresh training...")
    
    # Initialize data preprocessor
    print("\n📊 Loading and preprocessing data...")
    preprocessor = FoodDataPreprocessor()
    
    # Load classes
    classes = preprocessor.load_classes()
    if not classes:
        print("❌ Failed to load classes. Exiting.")
        return
    
    print(f"📁 Loaded {len(classes)} food classes")
    
    # Create data generators
    train_generator, validation_generator = preprocessor.create_data_generators()
    
    # Get actual class names from generator
    class_names = list(train_generator.class_indices.keys())
    print(f"🎯 Training on {len(class_names)} classes")
    
    # Save class names for later use
    save_class_names(class_names)
    
    # Build model
    print("\n🤖 Building model...")
    model_builder = FoodRecognitionModel(num_classes=len(class_names))
    
    # Load from checkpoint if resuming
    if checkpoint_path and initial_epoch > 0:
        model = load_model_and_history(model_builder, checkpoint_path)
    else:
        model = model_builder.build_model()
    
    # Display model architecture
    print("\n📋 Model Architecture:")
    model_builder.summary()
    
    # Enhanced callbacks for better checkpointing
    callbacks = [
        # Save best model
        ModelCheckpoint(
            config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Save periodic checkpoints
        ModelCheckpoint(
            'models/checkpoint_epoch_{epoch:02d}.h5',
            save_freq='epoch',
            verbose=0
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n🎯 Starting training...")
    print(f"📈 Training for {config.EPOCHS} epochs (starting from epoch {initial_epoch})")
    print(f"📦 Batch size: {config.BATCH_SIZE}")
    
    try:
        history = model.fit(
            train_generator,
            epochs=config.EPOCHS,
            initial_epoch=initial_epoch,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        print("\n📊 Plotting training history...")
        plot_training_history(history)
        
        print("\n✅ Training completed successfully!")
        print(f"💾 Best model saved to: {config.MODEL_SAVE_PATH}")
        
        # Clean up checkpoint files
        for file in os.listdir("models"):
            if file.startswith("checkpoint_epoch_") and file.endswith(".h5"):
                os.remove(os.path.join("models", file))
        print("🧹 Temporary checkpoint files cleaned up!")
        
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user. Model checkpoint saved.")
        print("💡 You can resume later by running this script again!")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("💾 Model checkpoint saved. You can resume training!")

if __name__ == "__main__":
    main()