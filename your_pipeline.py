# your_pipeline.py
import io
import time
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageStat
import tensorflow as tf
from ultralytics import YOLO
import os
import shutil # Import shutil for copying files
from typing import List, Tuple

# ------------------ Class Labels ------------------ #
class_labels = [
    "antelope","badger","bat","bear","bee","beetle","bison","boar","butterfly","cat",
    "caterpillar","chimpanzee","cockroach","cow","coyote","crab","crow","deer","dog","dolphin",
    "donkey","dragonfly","duck","eagle","elephant","flamingo","fly","fox","goat","goldfish",
    "goose","gorilla","grasshopper","hamster","hare","hedgehog","hippopotamus","hornbill","horse",
    "hummingbird","hyena","jellyfish","kangaroo","koala","ladybugs","leopard","lion","lizard","lobster",
    "mosquito","moth","mouse","octopus","okapi","orangutan","otter","owl","ox","oyster",
    "panda","parrot","pelecaniformes","penguin","pig","pigeon","porcupine","possum","raccoon","rat",
    "reindeer","rhinoceros","sandpiper","seahorse","seal","shark","sheep","snake","sparrow","squid",
    "squirrel","starfish","swan","tiger","turkey","turtle","whale","wolf","wombat","woodpecker","zebra"
]
class_indices = {i: label for i, label in enumerate(class_labels)}
num_classes = len(class_labels)

# ------------------ Checkpoints (Use relative paths or paths accessible to your FastAPI app) ------------------ #
CHECKPOINT_PATH = Path("checkpoint_stage2_finetuned.weights.h5")
DETECTOR_PATH   = Path("best100.pt")

# Define a base directory for the *final classified output*
# This will be created parallel to 'media' or inside it, depending on your preference.
# For now, let's put it directly in the project root.
CLASSIFIED_OUTPUT_BASE_DIR = Path("classified_images_output")
CLASSIFIED_OUTPUT_BASE_DIR.mkdir(exist_ok=True) # Ensure this base directory exists on server startup

# ------------------ Load Models Once (Singleton Pattern for efficiency) ------------------ #
_classifier_model = None
_detector_model = None

def _load_classifier():
    global _classifier_model
    if _classifier_model is None:
        base = tf.keras.applications.EfficientNetB3(
            include_top=False,
            input_shape=(224,224,3),
            weights="imagenet",
            pooling="max"
        )
        base.trainable = False

        inp = tf.keras.Input(shape=(224,224,3))
        x = base(inp, training=False)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.45)(x)
        x = tf.keras.layers.Dense(len(class_labels))(x)
        out = tf.keras.layers.Activation("softmax", dtype=tf.float32)(x)

        _classifier_model = tf.keras.Model(inputs=inp, outputs=out)
        try:
            _classifier_model.load_weights(str(CHECKPOINT_PATH))
        except Exception as e:
            print(f"Error loading classifier weights from {CHECKPOINT_PATH}: {e}")
            raise
    return _classifier_model

def _load_detector():
    global _detector_model
    if _detector_model is None:
        try:
            _detector_model = YOLO(str(DETECTOR_PATH))
        except Exception as e:
            print(f"Error loading detector model from {DETECTOR_PATH}: {e}")
            raise
    return _detector_model

# Initialize models on module import
classifier = _load_classifier()
detector = _load_detector()


# ------------------ Helpers ------------------ #
def _adaptive_threshold(pil_img: Image.Image) -> float:
    stat = ImageStat.Stat(pil_img.convert("L"))
    brightness, contrast = stat.mean[0], stat.stddev[0]
    thr = 0.7
    if brightness < 60: thr -= 0.1
    if contrast   < 30: thr -= 0.05
    return max(thr, 0.5)

# ------------------ Main Batch Pipeline ------------------ #
async def process_batch_with_reports(
    image_streams_with_names: List[Tuple[io.BytesIO, str]], # List of (BytesIO, filename)
    base_output_dir: Path # This is your MEDIA_DIR
) -> dict:
    """
    Processes a list of image byte streams and generates a comprehensive report
    and saves classified/detected images similar to the local script.
    It also copies/moves images to a new 'classified_images_output' folder,
    organized into 'yolo_detections' and 'classifier_fallbacks' subfolders.

    image_streams_with_names: List of tuples, where each tuple contains
                              (io.BytesIO object of the image, original filename string).
    base_output_dir:          Base directory for web-accessible outputs (e.g., your MEDIA_DIR).
    Returns a dict containing batch results for frontend display and report summary.
    """
    batch_start_time = time.time()

    # --- Setup output directories ---
    # 1. Directory for web-accessible results (under /media)
    current_batch_web_output_folder = base_output_dir / f"batch_results_{int(time.time())}"
    cropped_detections_web_dir = current_batch_web_output_folder / "cropped_detections"
    classifier_fallback_web_dir = current_batch_web_output_folder / "classifier_fallback"

    os.makedirs(current_batch_web_output_folder, exist_ok=True)
    os.makedirs(cropped_detections_web_dir, exist_ok=True)
    os.makedirs(classifier_fallback_web_dir, exist_ok=True)

    # 2. Directory for the new, organized 'classified_images_output'
    # Create a timestamped subfolder within CLASSIFIED_OUTPUT_BASE_DIR for this batch
    current_batch_classified_dir = CLASSIFIED_OUTPUT_BASE_DIR / f"batch_{int(time.time())}"
    os.makedirs(current_batch_classified_dir, exist_ok=True)

    # Create the two sets of subclasses within the batch classified directory
    final_yolo_output_dir = current_batch_classified_dir / "yolo_detections"
    final_classifier_output_dir = current_batch_classified_dir / "classifier_fallbacks"
    os.makedirs(final_yolo_output_dir, exist_ok=True)
    os.makedirs(final_classifier_output_dir, exist_ok=True)


    stats = {
        "total_images": 0,
        "total_detections": 0,
        "classifier_fallback_count": 0,
        "undetected_by_threshold": 0,
        "average_confidences": [],
        "class_counts": {},
        "per_image_results": []
    }

    for image_stream, original_filename in image_streams_with_names:
        stats["total_images"] += 1
        processing_start_time = time.time()
        
        file_result_for_frontend = {
            "fileName": original_filename,
            "status": "success",
            "error": None,
            "detections": [],
            "fallbackClassification": None,
            "adaptiveThreshold": 0,
            "imageUrl": None, # For frontend display, pointing to web_output_folder
            "classifiedPath": None # New field to indicate where it was classified in the separate folder
        }

        try:
            image_stream.seek(0)
            pil_img = Image.open(image_stream).convert("RGB")
            
            image_stream.seek(0)
            arr = np.frombuffer(image_stream.read(), dtype=np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if cv_img is None:
                raise ValueError("Could not decode image with OpenCV. Invalid image format?")

            adaptive_thresh = _adaptive_threshold(pil_img)
            file_result_for_frontend["adaptiveThreshold"] = round(adaptive_thresh, 3)

            results = detector(cv_img)[0]
            boxes = results.boxes

            has_valid_detections = False
            if boxes:
                for b in boxes:
                    if float(b.conf[0]) > adaptive_thresh:
                        has_valid_detections = True
                        break

            if has_valid_detections:
                for i, b in enumerate(boxes):
                    conf = float(b.conf[0])
                    if conf < adaptive_thresh:
                        continue

                    cls_id = int(b.cls[0])
                    label = detector.names[cls_id]
                    x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())

                    h, w, _ = cv_img.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        crop = cv_img[y1:y2, x1:x2]

                        # Save to web-accessible folder (for frontend display)
                        web_save_dir = cropped_detections_web_dir / label
                        os.makedirs(web_save_dir, exist_ok=True)
                        web_cropped_fname = f"{Path(original_filename).stem}_det_{i}_{conf:.3f}.jpg"
                        cv2.imwrite(str(web_save_dir / web_cropped_fname), crop)
                        
                        web_cropped_image_url_relative = str(web_save_dir.relative_to(base_output_dir) / web_cropped_fname).replace("\\", "/")

                        # Save to the new classified_images_output/yolo_detections folder
                        classified_save_dir = final_yolo_output_dir / label # Separate by source, then class
                        os.makedirs(classified_save_dir, exist_ok=True)
                        classified_cropped_fname = f"{Path(original_filename).stem}_det_{i}_{conf:.3f}.jpg"
                        # Copy the image to the new classified folder
                        shutil.copy(str(web_save_dir / web_cropped_fname), str(classified_save_dir / classified_cropped_fname))

                        file_result_for_frontend["detections"].append({
                            "animal": label,
                            "confidence": round(conf, 3),
                            "bbox": {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1},
                            "image_url": f"/media/{web_cropped_image_url_relative}"
                        })
                        # Path for the classified output, relative to CLASSIFIED_OUTPUT_BASE_DIR
                        file_result_for_frontend["classifiedPath"] = str(classified_save_dir.relative_to(CLASSIFIED_OUTPUT_BASE_DIR) / classified_cropped_fname).replace("\\", "/")

                        stats["total_detections"] += 1
                        stats["average_confidences"].append(conf)
                        stats["class_counts"][label] = stats["class_counts"].get(label, 0) + 1
                    else:
                        print(f"Warning: Invalid crop for {original_filename} - {label} at ({x1},{y1},{x2},{y2})")

                # If there were valid detections, the main image URL points to the first detection's image
                if file_result_for_frontend["detections"]:
                    file_result_for_frontend["imageUrl"] = file_result_for_frontend["detections"][0]["image_url"]
                else:
                    # If detections were found but none passed the confidence threshold, it's considered 'undetected' by YOLO.
                    stats["undetected_by_threshold"] += 1
                    # Fall through to classifier fallback if no valid YOLO detection.
                    pass # Continue to the else block if no valid detections
            
            # Fallback to classifier if no (valid) detections were made by YOLO
            if not has_valid_detections or not file_result_for_frontend["detections"]:
                stats["classifier_fallback_count"] += 1
                
                img224 = pil_img.resize((224,224))
                x = tf.keras.preprocessing.image.img_to_array(img224)
                x = tf.keras.applications.efficientnet.preprocess_input(x)
                x = np.expand_dims(x, axis=0)

                preds = classifier.predict(x, verbose=0)[0]
                idx = int(np.argmax(preds))
                label = class_indices[idx]
                conf = float(preds[idx])

                # Save original image to the batch-specific classifier_fallback web folder
                web_fallback_fname = f"{Path(original_filename).stem}_cls_{conf:.3f}.jpg"
                pil_img.save(str(classifier_fallback_web_dir / web_fallback_fname))
                
                web_fallback_image_url_relative = str(classifier_fallback_web_dir.relative_to(base_output_dir) / web_fallback_fname).replace("\\", "/")

                # Save to the new classified_images_output/classifier_fallbacks folder
                classified_save_dir = final_classifier_output_dir / label # Separate by source, then class
                os.makedirs(classified_save_dir, exist_ok=True)
                classified_fallback_fname = f"{Path(original_filename).stem}_cls_{conf:.3f}.jpg"
                # Copy the image to the new classified folder
                shutil.copy(str(classifier_fallback_web_dir / web_fallback_fname), str(classified_save_dir / classified_fallback_fname))


                file_result_for_frontend["fallbackClassification"] = {
                    "animal": label,
                    "confidence": round(conf, 3),
                    "image_url": f"/media/{web_fallback_image_url_relative}"
                }
                # For fallback, the main image URL is the original image as classified by fallback
                file_result_for_frontend["imageUrl"] = file_result_for_frontend["fallbackClassification"]["image_url"]
                # Path for the classified output, relative to CLASSIFIED_OUTPUT_BASE_DIR
                file_result_for_frontend["classifiedPath"] = str(classified_save_dir.relative_to(CLASSIFIED_OUTPUT_BASE_DIR) / classified_fallback_fname).replace("\\", "/")


                stats["average_confidences"].append(conf)
                stats["class_counts"][label] = stats["class_counts"].get(label, 0) + 1

                if conf < 0.5:
                    stats["undetected_by_threshold"] += 1 # If fallback confidence is also low


        except Exception as e:
            file_result_for_frontend["status"] = "error"
            file_result_for_frontend["error"] = str(e)
            print(f"Error processing {original_filename}: {e}")

        file_result_for_frontend["processingTime"] = round(time.time() - processing_start_time, 3)
        stats["per_image_results"].append(file_result_for_frontend)

    # ------------------ Save Report ------------------ #
    avg_conf = sum(stats["average_confidences"]) / len(stats["average_confidences"]) if stats["average_confidences"] else 0
    sorted_classes = sorted(stats["class_counts"].items(), key=lambda x: x[1], reverse=True)
    report = {
        "total_images": stats["total_images"],
        "total_detections_yolo": stats["total_detections"],
        "classifier_fallback_count": stats["classifier_fallback_count"],
        "undetected_by_threshold": stats["undetected_by_threshold"],
        "top_5_classes": sorted_classes[:5],
        "average_confidence_across_batch": round(avg_conf, 3),
        "batch_processing_time_total": round(time.time() - batch_start_time, 3),
        "classified_output_directory_relative": str(current_batch_classified_dir.relative_to(CLASSIFIED_OUTPUT_BASE_DIR)).replace("\\", "/") # Relative path to the classified output
    }
    
    report_file_path = current_batch_web_output_folder / "batch_report.json"
    with open(str(report_file_path), "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n✅ Batch pipeline complete. Report saved to: {report_file_path}")
    print(f"📁 Classified images also saved to: {current_batch_classified_dir}")

    return {
        "results": stats["per_image_results"],
        "report_summary": report,
        "output_folder_url": f"/media/{current_batch_web_output_folder.relative_to(base_output_dir).as_posix()}",
        "classified_output_folder": str(current_batch_classified_dir.relative_to(CLASSIFIED_OUTPUT_BASE_DIR)).replace("\\", "/") # Return this for frontend info
    }

# The `process_image` function for single file uploads can also be updated
# if you want single uploads to also go into the new CLASSIFIED_OUTPUT_BASE_DIR
# For now, I'll only modify the batch processing.
async def process_image(image_bytes: io.BytesIO, media_dir: Path) -> dict:
    """
    Processes a single image byte stream (used for /process/single endpoint).
    """
    start_time = time.time()

    image_bytes.seek(0)
    pil_img = Image.open(image_bytes).convert("RGB")
    
    image_bytes.seek(0)
    arr = np.frombuffer(image_bytes.read(), dtype=np.uint8)
    cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if cv_img is None:
        raise ValueError("Could not decode image with OpenCV. Ensure it's a valid image format.")

    results = detector(cv_img)[0]
    boxes = results.boxes
    thr = _adaptive_threshold(pil_img)

    detections = []
    fallback = None

    has_valid_detections = False
    if boxes:
        for b in boxes:
            if float(b.conf[0]) > thr:
                has_valid_detections = True
                break

    if has_valid_detections:
        for b in boxes:
            conf = float(b.conf[0])
            if conf < thr: 
                continue

            cls_id = int(b.cls[0])
            label = detector.names[cls_id]
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())

            h, w, _ = cv_img.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                crop = cv_img[y1:y2, x1:x2]
                out_dir = media_dir / "single_detections"
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{time.time_ns()}_{label}_{conf:.3f}.jpg"
                cv2.imwrite(str(out_dir/fname), crop)
                
                detections.append({
                    "animal":     label,
                    "confidence": round(conf, 3),
                    "bbox":       {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1},
                    "image_url":  f"/media/single_detections/{fname}"
                })
            else:
                print(f"Skipping invalid crop coordinates for single upload: ({x1},{y1},{x2},{y2})")

    if not detections and not has_valid_detections:
        img224 = pil_img.resize((224,224))
        x = tf.keras.preprocessing.image.img_to_array(img224)
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = classifier.predict(x, verbose=0)[0]
        idx   = int(np.argmax(preds))
        label = class_indices[idx]
        conf  = float(preds[idx])

        out_dir = media_dir / "single_fallback"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{time.time_ns()}_{label}_{conf:.3f}.jpg"
        pil_img.save(str(out_dir/fname))

        fallback = {
            "animal":     label,
            "confidence": round(conf, 3),
            "image_url":  f"/media/single_fallback/{fname}"
        }

    return {
        "detections":           detections,
        "fallbackClassification": fallback,
        "adaptiveThreshold":    round(thr, 3),
        "processingTime": round(time.time() - start_time, 3)
    }