import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ------------------ Animal Class Labels ------------------ #
class_labels = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin",
    "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
    "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse",
    "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster",
    "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster",
    "panda", "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat",
    "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid",
    "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
]
class_indices = {i: label for i, label in enumerate(class_labels)}
num_classes = len(class_labels)

# ------------------ Settings ------------------ #
IMAGE_SIZE = (224, 224)
CHECKPOINT_PATH = r"C:\Users\poorn\OneDrive\Desktop\IDP-AnimalModel\checkpoint_stage1.weights.h5"
INPUT_FOLDER = r"C:\Users\poorn\Downloads\Wild Animals.v1i.yolov11\test\images"
OUTPUT_FOLDER = r"C:\Users\poorn\OneDrive\Desktop\IDP-AnimalModel\predictions_output"
CONFIDENCE_THRESHOLD = 0.6  # Only save if confidence > 60%

# ------------------ Load Model ------------------ #
def load_model_from_checkpoint(checkpoint_path):
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Activation("softmax", dtype=tf.float32)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(checkpoint_path)
    print("[INFO] Model loaded successfully.")
    return model

# ------------------ Predict and Save ------------------ #
def predict_and_save(model, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = [".jpg", ".jpeg", ".png"]

    for img_file in os.listdir(input_folder):
        if not any(img_file.lower().endswith(ext) for ext in image_extensions):
            continue

        img_path = os.path.join(input_folder, img_file)
        try:
            img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        except Exception as e:
            print(f"[ERROR] Skipping {img_file}: {e}")
            continue

        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions)
        pred_label = class_indices[pred_idx]
        confidence = predictions[0][pred_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"[SKIP] {img_file}: confidence too low ({confidence:.2f})")
            continue

        # Save to class-specific subfolder
        class_folder = os.path.join(output_folder, pred_label)
        os.makedirs(class_folder, exist_ok=True)

        # Annotate and save image
        fig, ax = plt.subplots()
        ax.imshow(Image.open(img_path))
        ax.axis('off')
        ax.set_title(f"{img_file}\nPrediction: {pred_label}\nConfidence: {confidence:.2f}", fontsize=10)
        output_path = os.path.join(class_folder, f"{Path(img_file).stem}_pred.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"[SAVE] {img_file} -> {pred_label} ({confidence:.2f})")

# ------------------ Main ------------------ #
if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
    elif not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder not found at {INPUT_FOLDER}")
    else:
        model = load_model_from_checkpoint(CHECKPOINT_PATH)
        predict_and_save(model, INPUT_FOLDER, OUTPUT_FOLDER)
        print(f"\n✅ All confident results saved in: {OUTPUT_FOLDER}")
