import os
import cv2
import json
import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import tensorflow as tf

# ------------------ Class Labels ------------------ #
class_labels = [ "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin",
    "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
    "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse",
    "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster",
    "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster",
    "panda", "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat",
    "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid",
    "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra" ]
class_indices = {i: label for i, label in enumerate(class_labels)}
num_classes = len(class_labels)

# ------------------ Paths ------------------ #
CHECKPOINT_PATH = r"C:\Users\poorn\OneDrive\Desktop\IDP-AnimalModel\checkpoint_stage1.weights.h5"
DETECTOR_PATH = r"C:\Users\poorn\OneDrive\Desktop\IDP-AnimalModel\best100.pt"
INPUT_FOLDER = r"C:\Users\poorn\Downloads\Wild Animals.v1i.yolov11\test\images"
OUTPUT_FOLDER = r"C:\Users\poorn\OneDrive\Desktop\IDP-AnimalModel\predictions_output"
CROPPED_DIR = os.path.join(OUTPUT_FOLDER, "cropped_detections")

# ------------------ Load Classifier ------------------ #
def load_model(checkpoint_path):
    base = tf.keras.applications.EfficientNetB3(include_top=False, input_shape=(224,224,3), weights='imagenet', pooling='max')
    base.trainable = False
    x = tf.keras.Input(shape=(224,224,3))
    y = base(x, training=False)
    y = tf.keras.layers.Dense(256, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dropout(0.45)(y)
    y = tf.keras.layers.Dense(num_classes)(y)
    output = tf.keras.layers.Activation("softmax", dtype=tf.float32)(y)
    model = tf.keras.Model(inputs=x, outputs=output)
    model.load_weights(checkpoint_path)
    return model

# ------------------ Adaptive Threshold ------------------ #
def get_adaptive_threshold(img_path):
    img = Image.open(img_path).convert('L')
    stat = ImageStat.Stat(img)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    threshold = 0.7
    if brightness < 60: threshold -= 0.1
    if contrast < 30: threshold -= 0.05
    return max(threshold, 0.5)

# ------------------ Process ------------------ #
def run_pipeline(detector_path, classifier_path, input_folder, output_folder):
    model = load_model(classifier_path)
    detector = YOLO(detector_path)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(CROPPED_DIR, exist_ok=True)

    stats = {
        "total_images": 0,
        "total_detections": 0,
        "classifier_fallback": 0,
        "undetected": 0,
        "average_confidences": [],
        "class_counts": {}
    }

    for img_file in os.listdir(input_folder):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        stats["total_images"] += 1
        image_path = os.path.join(input_folder, img_file)
        image = Image.open(image_path).convert("RGB")
        img_cv = cv2.imread(image_path)

        results = detector(image_path)
        detections = results[0].boxes
        adaptive_thresh = get_adaptive_threshold(image_path)

        if detections and any(float(box.conf[0]) > adaptive_thresh for box in detections):
            for i, box in enumerate(detections):
                conf = float(box.conf[0])
                if conf < adaptive_thresh:
                    continue
                cls_id = int(box.cls[0])
                label = detector.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = img_cv[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

                save_dir = os.path.join(CROPPED_DIR, label)
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"{Path(img_file).stem}_det_{i}.jpg"), crop)

                stats["total_detections"] += 1
                stats["average_confidences"].append(conf)
                stats["class_counts"][label] = stats["class_counts"].get(label, 0) + 1
        else:
            # Fallback to classifier
            image_resized = image.resize((224,224))
            img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions)
            pred_label = class_indices[pred_idx]
            confidence = float(predictions[0][pred_idx])

            save_dir = os.path.join(output_folder, pred_label)
            os.makedirs(save_dir, exist_ok=True)

            image.save(os.path.join(save_dir, f"{Path(img_file).stem}_cls_{confidence:.2f}.jpg"))
            stats["classifier_fallback"] += 1
            stats["average_confidences"].append(confidence)
            stats["class_counts"][pred_label] = stats["class_counts"].get(pred_label, 0) + 1

            if confidence < 0.5:
                stats["undetected"] += 1

    # ------------------ Save Report ------------------ #
    avg_conf = sum(stats["average_confidences"]) / len(stats["average_confidences"]) if stats["average_confidences"] else 0
    sorted_classes = sorted(stats["class_counts"].items(), key=lambda x: x[1], reverse=True)
    report = {
        "total_images": stats["total_images"],
        "total_detections": stats["total_detections"],
        "classifier_fallback": stats["classifier_fallback"],
        "undetected": stats["undetected"],
        "top_5_classes": sorted_classes[:5],
        "average_confidence": round(avg_conf, 3)
    }
    with open(os.path.join(output_folder, "report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Pipeline complete.")
    print("📁 Output saved to:", output_folder)
    print("📄 Report summary saved to:", os.path.join(output_folder, "report.json"))

# ------------------ Run ------------------ #
if __name__ == "__main__":
    run_pipeline(DETECTOR_PATH, CHECKPOINT_PATH, INPUT_FOLDER, OUTPUT_FOLDER)
