import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')
---
# Configuration
DATASET_PATH = 'dataset'
IMAGE_SIZE = (150, 150)  # Increased from 128
BATCH_SIZE = 16  # Reduced for better gradient updates
EPOCHS = 50
LEARNING_RATE = 0.0001

# Create output directory
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
---
# Count images per class
class_counts = {}
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[class_name] = count

# Sort and display
class_counts = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
print(f"Total Classes: {len(class_counts)}")
print(f"Total Images: {sum(class_counts.values())}")
print("\nClass Distribution:")
for cls, cnt in class_counts.items():
    print(f"{cls}: {cnt} images")
---
# Visualize class distribution
plt.figure(figsize=(14, 6))
plt.bar(range(len(class_counts)), list(class_counts.values()), color='steelblue')
plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=90)
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Dataset')
plt.tight_layout()
plt.savefig('results/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
---
# Display sample images from each class
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
axes = axes.ravel()

for idx, (class_name, count) in enumerate(list(class_counts.items())[:24]):
    class_path = os.path.join(DATASET_PATH, class_name)
    sample_img = os.listdir(class_path)[0]
    img_path = os.path.join(class_path, sample_img)
    img = load_img(img_path, target_size=IMAGE_SIZE)
    axes[idx].imshow(img)
    axes[idx].set_title(f"{class_name}\n({count} imgs)", fontsize=9)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
---
# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Validation data (only rescaling)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
---
# Create generators - FIX: Don't use steps_per_epoch in flow_from_directory
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

# Get class information
num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Number of classes: {num_classes}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps per epoch (train): {train_generator.samples // BATCH_SIZE}")
print(f"Validation steps: {validation_generator.samples // BATCH_SIZE}")
---
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Build model
model = build_cnn_model(IMAGE_SIZE + (3,), num_classes)

# Compile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
---
# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/skin_cancer_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
---
# Train the model - CRITICAL FIX: Let Keras calculate steps automatically
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
---
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Print best scores
best_epoch = np.argmax(history.history['val_accuracy'])
print(f"\nBest Validation Accuracy: {max(history.history['val_accuracy']):.4f} at epoch {best_epoch + 1}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
---
# Load best model
best_model = load_model('models/skin_cancer_best.keras')

# Get predictions
validation_generator.reset()
predictions = best_model.predict(validation_generator, steps=len(validation_generator), verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

# Classification Report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Save report
with open('results/classification_report.txt', 'w') as f:
    f.write(report)
---
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - Skin Cancer Detection', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate overall accuracy
accuracy = np.trace(cm) / np.sum(cm)
print(f"\nOverall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
---
# Per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
class_performance = pd.DataFrame({
    'Class': class_labels,
    'Accuracy': class_accuracy,
    'Total_Samples': cm.sum(axis=1)
}).sort_values('Accuracy', ascending=False)

print("\nPer-Class Performance:")
print(class_performance.to_string(index=False))

# Visualize per-class accuracy
plt.figure(figsize=(14, 6))
colors = ['green' if acc > 0.7 else 'orange' if acc > 0.5 else 'red' for acc in class_performance['Accuracy']]
plt.barh(class_performance['Class'], class_performance['Accuracy'], color=colors)
plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Class', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good (>70%)')
plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (>50%)')
plt.legend()
plt.tight_layout()
plt.savefig('results/per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
---
# Display sample predictions
validation_generator.reset()
x_batch, y_batch = next(validation_generator)
predictions = best_model.predict(x_batch)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.ravel()

for i in range(16):
    axes[i].imshow(x_batch[i])
    true_label = class_labels[np.argmax(y_batch[i])]
    pred_label = class_labels[np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100
    
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%", 
                     color=color, fontsize=10, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
---
# Save final model
model.save('models/skin_cancer_final.keras')
print("✓ Model saved successfully!")

# Save class labels
np.save('models/class_labels.npy', class_labels)
print("✓ Class labels saved!")

# Save training history
import json
with open('results/training_history.json', 'w') as f:
    json.dump(history.history, f)
print("✓ Training history saved!")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Best model saved at: models/skin_cancer_best.keras")
print(f"Final model saved at: models/skin_cancer_final.keras")
print(f"Results saved in: results/ directory")
---
# Test loading saved model
loaded_model = load_model('models/skin_cancer_best.keras')
loaded_labels = np.load('models/class_labels.npy', allow_pickle=True)

print("✓ Model loaded successfully!")
print(f"Model input shape: {loaded_model.input_shape}")
print(f"Number of classes: {len(loaded_labels)}")
print(f"\nClass labels: {loaded_labels}")
---
