import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configuration
DATASET_PATH = 'dataset'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

# Create output directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("🚀 Starting improved training pipeline using EfficientNetB0...")
    
    # 1. Data Generators
    # EfficientNet expects inputs in range [0, 255] and handles its own rescaling internally
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    num_classes = len(train_generator.class_indices)
    class_labels = list(train_generator.class_indices.keys())
    
    # Save class labels
    np.save(os.path.join(MODELS_DIR, 'class_labels.npy'), class_labels)
    print(f"✓ Saved {num_classes} class labels to {MODELS_DIR}/class_labels.npy")

    # 3. Build Transfer Learning Model
    print("\n🏗️ Building Transfer Learning Model (EfficientNetB0)...")
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=IMAGE_SIZE + (3,)
    )

    # Freeze the base model
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 4. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, 'skin_cancer_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Train the Model
    print("\n⏳ Starting Training Phase 1: Training top layers only...")
    history1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Fine-tuning Phase (Optional but highly recommended)
    print("\n🔥 Starting Training Phase 2: Fine-tuning the entire model...")
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze the first 100 layers and unfreeze the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Recompile with a much lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=15, # Continue training
        callbacks=callbacks,
        verbose=1
    )

    print("\n🎉 Training Complete! Evaluating...")

    # 7. Evaluate
    validation_generator.reset()
    preds = model.predict(validation_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = validation_generator.classes

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)

    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(16, 14))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Skin Cancer Detection (EfficientNetB0)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    print(f"\n✅ All done! The improved model is saved as '{MODELS_DIR}/skin_cancer_best.keras'.")

if __name__ == '__main__':
    main()
