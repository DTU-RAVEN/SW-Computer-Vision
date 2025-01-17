# object_detection_shared.py

import torch
from ultralytics import YOLO

def initialize_model(model_name: str, device: torch.device) -> YOLO:
    """
    Loads a YOLO model (Ultralytics) given a model name/path and returns it.
    Does not explicitly move the model to the device; 
    Ultralytics internally handles it if 'device' is passed during predict.
    """
    model = YOLO(model_name)
    return model

def parse_detection_result(result, target_class_ids, custom_labels, model):
    """
    Given a single YOLO 'result' object (from model.predict), extract bounding boxes, 
    class IDs, confidence, etc. Filters out detections not in 'target_class_ids'.
    
    Returns a list of tuples: (class_id, conf, x1, y1, x2, y2, display_label)
    where 'display_label' is either from custom_labels or model.names.
    """
    detections = []
    
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            # Skip if not in the desired set of classes
            if class_id not in target_class_ids:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Pick a nice display label
            default_label = model.names.get(class_id, f"class_{class_id}")
            display_label = custom_labels.get(class_id, default_label)

            detections.append((class_id, conf, x1, y1, x2, y2, display_label))
    
    return detections
