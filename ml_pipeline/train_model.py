import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data"
MODEL_SAVE_PATH = "crop_disease_model.h5"
TFLITE_SAVE_PATH = "crop_disease_model.tflite"
LABELS_SAVE_PATH = "labels.txt"

def prepare_dummy_data():
    """Creates a dummy dataset if no real data is found for testing the pipeline."""
    print("Generating dummy data for testing the pipeline...")
    os.makedirs(os.path.join(DATA_DIR, 'train', 'maize_healthy'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'train', 'maize_blight'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'train', 'cassava_mosaic'), exist_ok=True)
    
    # Create some random images
    for cls in ['maize_healthy', 'maize_blight', 'cassava_mosaic']:
        for i in range(5):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            tf.keras.utils.save_img(os.path.join(DATA_DIR, 'train', cls, f'dummy_{i}.jpg'), img)

def load_data():
    train_dir = os.path.join(DATA_DIR, 'train')
    if not os.path.exists(train_dir):
        prepare_dummy_data()

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        validation_split=0.2,
        subset="training",
        seed=123
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=123
    )

    class_names = train_dataset.class_names
    with open(LABELS_SAVE_PATH, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"Saved labels: {class_names}")

    return train_dataset, validation_dataset, len(class_names)

def build_model(num_classes):
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model for transfer learning

    # Add custom classification head
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def convert_to_tflite(model):
    print("Converting model to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Apply post-training quantization to reduce size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {TFLITE_SAVE_PATH}")

def main():
    print("Loading data...")
    train_dataset, validation_dataset, num_classes = load_data()

    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    print("Building model...")
    model = build_model(num_classes)

    print("Training model...")
    model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

    print(f"Saving Keras model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    convert_to_tflite(model)
    print("ML Pipeline completed successfully.")

if __name__ == '__main__':
    main()
