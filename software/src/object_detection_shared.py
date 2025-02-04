# object_detection_shared.py

import torch
from ultralytics import YOLO

def initialize_model(model_name: str, device: torch.device) -> YOLO:
    """
    Initialize and load a YOLO model for object detection.

    Args:
        model_name (str): Path to the model weights or name of a pretrained model.
        device (torch.device): The computing device (CPU/GPU) to use for inference.
                             Note: Currently used for documentation only as Ultralytics
                             handles device placement internally.

    Returns:
        YOLO: Initialized YOLO model ready for inference.

    Note:
        The function doesn't explicitly move the model to the specified device
        as Ultralytics handles device placement internally when calling predict().
    """
    model = YOLO(model_name)
    return model

def parse_detection_result(result, target_class_ids, custom_labels, model):
    """
    Process and filter YOLO detection results into a standardized format.

    Args:
        result: Single result object from YOLO model's predict() method containing
               detection information (boxes, classes, confidence scores).
        target_class_ids (list): List of class IDs to keep in the results. 
                                Detections of other classes will be filtered out.
        custom_labels (dict): Mapping of class IDs to custom display labels.
                            Used to override default model labels.
        model (YOLO): The YOLO model object containing the class names mapping.

    Returns:
        list[tuple]: List of processed detections, where each detection is a tuple:
                    (class_id, confidence, x1, y1, x2, y2, display_label)
                    - class_id (int): The predicted class ID
                    - confidence (float): Detection confidence score
                    - x1, y1 (int): Top-left corner coordinates of bounding box
                    - x2, y2 (int): Bottom-right corner coordinates of bounding box
                    - display_label (str): Human-readable class label

    Note:
        The function performs several key operations:
        1. Filters detections based on target_class_ids
        2. Converts bounding box coordinates to integers
        3. Resolves human-readable labels using custom_labels or model defaults
    """
    detections = []
    
    # Process results only if valid detections exist
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # Extract and convert class ID to integer
            class_id = int(box.cls[0])
            
            # Skip non-target classes
            if class_id not in target_class_ids:
                continue

            # Extract confidence score and bounding box coordinates
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Resolve display label: prefer custom label, fall back to model's default
            default_label = model.names.get(class_id, f"class_{class_id}")
            display_label = custom_labels.get(class_id, default_label)

            detections.append((class_id, conf, x1, y1, x2, y2, display_label))
    
    return detections
