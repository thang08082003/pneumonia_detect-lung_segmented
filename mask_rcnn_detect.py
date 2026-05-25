"""
Mask R-CNN RSNA Pneumonia Detection - Correct Inference Script
=============================================================
Usage:
  python mask_rcnn_detect_rsna.py --image path/to/xray.png
  python mask_rcnn_detect_rsna.py --input-dir images --output-dir outputs
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import warnings

# ------------------------------------------------------------------
# Silence TF
# ------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1.keras import backend as K

from mrcnn import model as modellib
from mrcnn.config import Config

# ------------------------------------------------------------------
# PATH TO MODEL
# ------------------------------------------------------------------
MODEL_PATH = "mask_rcnn_pneumonia_0015.h5"   # <-- sửa nếu tên khác


# ------------------------------------------------------------------
# CONFIG (MATCH TRAINING)
# ------------------------------------------------------------------
class PneumoniaConfig(Config):
    NAME = "pneumonia"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 2  # background + pneumonia

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    DETECTION_MIN_CONFIDENCE = 0.05


class InferenceConfig(PneumoniaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# ------------------------------------------------------------------
# LOAD MODEL (SAFE GRAPH/SESSION)
# ------------------------------------------------------------------
def load_model():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        with sess.as_default():
            K.set_session(sess)
            config = InferenceConfig()
            model = modellib.MaskRCNN(
                mode="inference",
                config=config,
                model_dir="."
            )
            model.load_weights(MODEL_PATH, by_name=True)
    return {"graph": graph, "sess": sess, "model": model}


# ------------------------------------------------------------------
# IMAGE LOADER (PNG / JPG / DICOM)
# ------------------------------------------------------------------
def load_image(path):
    path = str(path)
    if path.lower().endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img = img / img.max() * 255.0
        img = img.astype(np.uint8)
    else:
        img = np.array(Image.open(path).convert("L"))
    return img


# ------------------------------------------------------------------
# DETECTION
# ------------------------------------------------------------------
def detect(holder, image, conf=0.05, force_top1=False):
    sess = holder["sess"]
    graph = holder["graph"]
    model = holder["model"]

    # IMPORTANT: DO NOT RESIZE
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image

    with graph.as_default():
        with sess.as_default():
            K.set_session(sess)
            result = model.detect([rgb], verbose=0)[0]

    detections = []

    for i in range(len(result["rois"])):
        score = float(result["scores"][i])
        if score < conf:
            continue

        y1, x1, y2, x2 = result["rois"][i]
        mask = result["masks"][:, :, i]

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "mask": mask
        })

    # Force highest score if nothing passed threshold
    if force_top1 and len(detections) == 0 and len(result["scores"]) > 0:
        idx = int(np.argmax(result["scores"]))
        y1, x1, y2, x2 = result["rois"][idx]
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": float(result["scores"][idx]),
            "mask": result["masks"][:, :, idx]
        })

    return detections


# ------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------
def visualize(image, detections, alpha=0.4):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        mask = det["mask"]

        if mask is not None:
            overlay = out.copy()
            overlay[mask > 0] = (255, 50, 50)
            out = cv2.addWeighted(out, 1, overlay, alpha, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = f"Pneumonia {score:.2f}"
        cv2.putText(out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--force-top1", action="store_true")
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        raise ValueError("Provide --image or --input-dir")

    holder = load_model()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        img = load_image(args.image)
        dets = detect(holder, img, args.conf, args.force_top1)
        vis = visualize(img, dets)
        out = Path(args.output_dir) / (Path(args.image).stem + "_result.png")
        cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out} | detections: {len(dets)}")

    else:
        for p in Path(args.input_dir).glob("*"):
            if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".dcm"]:
                continue
            img = load_image(p)
            dets = detect(holder, img, args.conf, args.force_top1)
            vis = visualize(img, dets)
            out = Path(args.output_dir) / (p.stem + "_result.png")
            cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"{p.name}: {len(dets)} detections")


if __name__ == "__main__":
    main()
