import json
import os
from pathlib import Path
from typing import Optional, Tuple

import keras_cv
import tensorflow as tf

from vesselattributeprediction.constants import (
    DEFAULT_IMAGE_BBOX_LOSS,
    DEFAULT_IMAGE_DETECTOR_LOSS,
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_MODEL_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_OUTPUT_CLASSES,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_TRAINING_DIR,
    DEFAULT_YOLO_BBOX_FORMAT,
    DEFAULT_YOLO_MODEL,
)
from vesselattributeprediction.utils import get_abs_path


class VesselDetectorModel:
    """
    Model utilizing YOLOv8 architecture to extract visual embeddings from input images for vessel detection.
    """

    def __init__(
        self,
        img_input_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
        img_backbone_model: str = DEFAULT_YOLO_MODEL,
        bounding_box_format: str = DEFAULT_YOLO_BBOX_FORMAT,
        num_classes: int = DEFAULT_NUM_OUTPUT_CLASSES,
        learning_rate: float = DEFAULT_MODEL_LEARNING_RATE,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
    ) -> None:
        self.img_input_shape = img_input_shape
        self.num_epoch = num_epochs
        self.img_backbone_model = img_backbone_model
        self.bounding_box_format = bounding_box_format
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.training_output_directory: Path = get_abs_path(DEFAULT_PROJECT_ROOT, DEFAULT_TRAINING_DIR)
        self.image_classification_loss = DEFAULT_IMAGE_DETECTOR_LOSS
        self.image_bbox_loss = DEFAULT_IMAGE_BBOX_LOSS

        self.model: Optional[tf.keras.Model] = None
        self.training_history: Optional[tf.keras.callbacks.History] = None

        self.initialize_vessel_detector_model()

    def initialize_vessel_detector_model(self) -> None:
        """
        Initializes and compiles the vessel detection model.
        """
        # Freeze weights first for transfer learning
        img_branch = keras_cv.models.YOLOV8Backbone.from_preset(
            self.img_backbone_model,
            input_shape=self.img_input_shape,
            bounding_box_format=self.bounding_box_format,
            include_rescaling=False,
        )
        img_branch.trainable = False

        # Downgrade fpn depth to focus on small object detection
        img_detector_model = keras_cv.models.YOLOV8Detector(
            num_classes=self.num_classes,
            bounding_box_format=self.bounding_box_format,
            backbone=img_branch,
            fpn_depth=1,
        )

        img_detector_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            classification_loss=self.image_classification_loss,
            box_loss=self.image_bbox_loss,
        )

        self.model = img_detector_model

    def train_model_with_transfer_learning(
        self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset
    ) -> None:
        """
        Runs model training using train dataset.
        """
        if not self.model:
            raise ValueError(
                "No vessel detector model has been initialized. Run initialize_vessel_detector_model() first."
            )

        # First round training for head
        self.model.fit(train_dataset, validation_data=validation_dataset, epochs=15)

        # Second round training unfreeze earlier layers
        num_unfreeze_layers = 10
        for layer in self.model.backbone.layers[-num_unfreeze_layers:]:
            layer.trainable = True

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            classification_loss=self.image_classification_loss,
            box_loss=self.image_bbox_loss,
        )
        training_log = self.model.fit(train_dataset, validation_data=validation_dataset, epochs=50)
        self.training_history = training_log.history

        # Save training history/model
        training_history_filepath = get_abs_path(
            self.training_output_directory,
            "detector_training_history.json",
        )
        best_model_filepath = get_abs_path(
            self.training_output_directory,
            "vessel_detector_model.keras",
        )
        os.makedirs(self.training_output_directory, exist_ok=True)
        with training_history_filepath.open("w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2)
        self.model.save(best_model_filepath)
