from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sahi.slicing import slice_image
from sahi.utils.coco import CocoAnnotation

from vesselattributeprediction.constants import (
    CLASS_ID_TO_VESSEL_TYPES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_SEED,
    DEFAULT_TILE_MIN_AREA_RATIO,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILED_IMAGE_DIR,
    DEFAULT_XVIEW_ANNOTATION_FILE,
    DEFAULT_XVIEW_DATASET_ROOT,
    DEFAULT_XVIEW_IMAGE_DIR,
    DEFAULT_XVIEW_SPLIT_RATIOS,
    VESSEL_TYPES_TO_CLASS_ID,
)
from vesselattributeprediction.data.dataclasses import ImageAnnotationDatasets, XViewObjectAnnotation
from vesselattributeprediction.utils import (
    collect_img_files,
    get_abs_path,
    load_geojson_file,
    load_tiff_image,
    map_type_id_to_class_id_and_vessel,
)
from vesselattributeprediction.data.data_utils import convert_bbox_dim,load_and_scale_image


class xViewDataset:
    def __init__(
        self,
        dataset_root: Union[str, Path] = DEFAULT_XVIEW_DATASET_ROOT,
        annotation_file: Union[str, Path] = DEFAULT_XVIEW_ANNOTATION_FILE,
        image_dir: Union[str, Path] = DEFAULT_XVIEW_IMAGE_DIR,
        split_ratios: Tuple[float, float, float] = DEFAULT_XVIEW_SPLIT_RATIOS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.annotation_file = get_abs_path(self.dataset_root, annotation_file)
        self.image_dir = get_abs_path(self.dataset_root, image_dir)
        self.split_ratios = split_ratios
        self.seed = seed
        self.batch_size = DEFAULT_BATCH_SIZE

        self._ori_datasets: Optional[ImageAnnotationDatasets] = None
        self._processed_datasets: Optional[ImageAnnotationDatasets] = None

    def _parse_geojson_annotations(self, image_filepaths: List[Union[str, Path]]) -> List[XViewObjectAnnotation]:
        """
        Loads geojson file and returns parsed object annotations as a List of XViewObjectAnnotation.
        """
        annotations_dict = load_geojson_file(self.annotation_file)["features"]
        object_annotations = []
        for annotation in annotations_dict:
            image_filepath = get_abs_path(self.image_dir, annotation["properties"]["image_id"])
            feature_id = int(annotation["properties"]["feature_id"])
            object_id = int(annotation["properties"]["type_id"])
            class_id, vessel_type = map_type_id_to_class_id_and_vessel(object_id)
            bbox_dim = convert_bbox_dim(annotation["properties"]["bounds_imcoords"])
            geometry = annotation["geometry"]["coordinates"][0]

            # Skip adding this annotation if bbox is invalid, not a vessel annotation or image file is missing
            if bbox_dim is None:
                continue

            if class_id == -1:
                continue

            if image_filepath not in image_filepaths:
                continue

            object_annotations.append(
                XViewObjectAnnotation(
                    image_path=image_filepath,
                    feature_id=feature_id,
                    object_id=object_id,
                    vessel_type=vessel_type,
                    bbox_dim=bbox_dim,
                    geometry=geometry,
                    detector_class_id=class_id,
                )
            )

        return object_annotations

    def collect_object_annotations(self) -> List[XViewObjectAnnotation]:
        """
        Collects all valid object annotations and returns as a list.
        """
        image_filepaths = collect_img_files(self.image_dir)
        object_annotations = self._parse_geojson_annotations(image_filepaths)
        # object_annotations = self._rebalance_vessel_annotations(object_annotations)

        return object_annotations

    def collect_image_annotations(
        self,
        object_annotations: List[XViewObjectAnnotation],
    ) -> Dict:
        """
        Collects all object annotations and groups them by image.
        """
        img_annotations = {}

        for annotation in object_annotations:
            img_annotation = img_annotations.setdefault(
                annotation.image_path,
                {
                    "boxes": [],
                    "classes": [],
                },
            )

            img_annotation["boxes"].append(annotation.bbox_dim)
            img_annotation["classes"].append(annotation.detector_class_id)

        return img_annotations

    def stratified_dataset_split(self, img_annotations: Dict) -> ImageAnnotationDatasets:
        """
        Runs stratified splitting of image annotations into training, validation and testing datasets.
        """
        random.seed(self.seed)

        annotations_list = list(img_annotations.items())
        random.shuffle(annotations_list)
        num_images = len(annotations_list)
        num_classes = len(VESSEL_TYPES_TO_CLASS_ID.keys())

        # Records if a class is present in an image
        indexes = np.arange(num_images).reshape(-1, 1)
        class_status = np.zeros((num_images, num_classes), dtype=np.int8)

        for index, (_, annotation) in enumerate(annotations_list):
            class_ids_in_img = set(annotation["classes"])
            for class_id in class_ids_in_img:
                class_status[index, int(class_id)] = 1

        # First split: train vs val + test
        split_ratio = self.split_ratios[1] + self.split_ratios[2]
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=self.seed)
        train_indexes, second_split_indexes = next(msss.split(indexes, class_status))

        # Second split: val vs test
        remaining_indexes = indexes[second_split_indexes]
        remaining_class_status = class_status[second_split_indexes]
        split_ratio = self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2])
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=self.seed)
        val_indexes, test_indexes = next(msss.split(remaining_indexes, remaining_class_status))

        # Obtain dictionaries of datasets
        train_dataset = {annotations_list[i][0]: annotations_list[i][1] for i in train_indexes}
        val_dataset = {annotations_list[i][0]: annotations_list[i][1] for i in second_split_indexes[val_indexes]}
        test_dataset = {annotations_list[i][0]: annotations_list[i][1] for i in second_split_indexes[test_indexes]}

        return ImageAnnotationDatasets(training=train_dataset, validation=val_dataset, testing=test_dataset)

    def _slice_image_annotations(
        self,
        image_path: Path,
        img_annotations: Dict,
        output_dir: Path,
    ) -> Dict:
        """
        Slices one image and returns tile-local annotations in xyxy format.
        """
        # Convert annotations to COCO form to use SAHI's tiling API
        coco_annotations = []
        for bbox, class_id in zip(img_annotations["boxes"], img_annotations["classes"]):
            x_min, y_min, x_max, y_max = bbox
            coco_annotations.append(
                CocoAnnotation.from_coco_bbox(
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    category_id=int(class_id),
                    category_name=CLASS_ID_TO_VESSEL_TYPES[int(class_id)],
                )
            )

        sliced_result = slice_image(
            image=load_tiff_image(image_path),
            coco_annotation_list=coco_annotations,
            output_file_name=image_path.stem,
            output_dir=str(output_dir),
            slice_height=DEFAULT_TILE_SIZE,
            slice_width=DEFAULT_TILE_SIZE,
            overlap_height_ratio=DEFAULT_TILE_OVERLAP,
            overlap_width_ratio=DEFAULT_TILE_OVERLAP,
            auto_slice_resolution=False,
            min_area_ratio=DEFAULT_TILE_MIN_AREA_RATIO,
            out_ext=".tif",
            verbose=False,
        )

        # Convert COCO annotations back to xyxy bboxes
        tiled_annotations = {}
        for sliced_image in sliced_result.sliced_image_list:
            tile_boxes = []
            tile_classes = []
            for sliced_annotation in sliced_image.coco_image.annotations:
                x_min, y_min, width, height = sliced_annotation.bbox
                tile_boxes.append([x_min, y_min, x_min + width, y_min + height])
                tile_classes.append(sliced_annotation.category_id)

            tile_path = output_dir / sliced_image.coco_image.file_name
            tiled_annotations[tile_path] = {
                "boxes": tile_boxes,
                "classes": tile_classes,
            }

        return tiled_annotations

    def slice_dataset(
        self,
        dataset: Dict,
        split_name: str,
        sample_negative_tiles: bool = False,
    ) -> Dict:
        """
        Slices all source images in one dataset split.
        """
        output_dir = get_abs_path(
            self.dataset_root,
            Path(DEFAULT_TILED_IMAGE_DIR) / split_name,
        )
        tiled_dataset = {}
        for image_path, annotation in dataset.items():
            tiled_dataset.update(
                self._slice_image_annotations(
                    image_path=Path(image_path),
                    img_annotations=annotation,
                    output_dir=output_dir,
                )
            )

        if not sample_negative_tiles:
            return tiled_dataset

        positive_tiles = {
            image_path: annotation for image_path, annotation in tiled_dataset.items() if annotation["boxes"]
        }
        negative_tiles = [image_path for image_path, annotation in tiled_dataset.items() if not annotation["boxes"]]
        sampled_negative_tiles = random.Random(self.seed).sample(
            negative_tiles,
            k=min(len(positive_tiles), len(negative_tiles)),
        )

        return {
            **positive_tiles,
            **{image_path: tiled_dataset[image_path] for image_path in sampled_negative_tiles},
        }

    def process_image_data(
        self,
        image_path: tf.Tensor,
        bbox_dims: tf.Tensor,
        class_ids: tf.Tensor,
    ) -> Dict:
        """
        Loads and formats one image annotation dataset.
        """
        # Convert tensors into numpy
        scaled_image, scaled_bboxes, class_ids = tf.py_function(
            func=load_and_scale_image,
            inp=[image_path, bbox_dims, class_ids],
            Tout=[tf.float32, tf.float32, tf.float32],
        )
        scaled_image.set_shape(DEFAULT_IMAGE_SHAPE)
        scaled_bboxes.set_shape((None, 4))
        class_ids.set_shape((None,))

        return {
            "images": scaled_image,
            "bounding_boxes": {
                "boxes": scaled_bboxes,
                "classes": class_ids,
            },
        }

    def process_dataset(self, dataset: Dict, training: bool = False) -> tf.data.Dataset:
        """
        Processes one dataset into a tensorflow batched dataset for model training/validation/testing.
        """
        img_annotations = list(dataset.values())
        image_paths = tf.constant(
            [str(image_path) for image_path in dataset],
            dtype=tf.string,
        )
        bboxes = tf.ragged.constant(
            [annotation["boxes"] for annotation in img_annotations],
            ragged_rank=1,
            inner_shape=(4,),
            dtype=tf.float32,
        )
        classes = tf.ragged.constant(
            [annotation["classes"] for annotation in img_annotations],
            ragged_rank=1,
            dtype=tf.float32,
        )

        # Process into slices
        tf_dataset = tf.data.Dataset.from_tensor_slices((image_paths, bboxes, classes))
        if training:
            tf_dataset = tf_dataset.shuffle(
                buffer_size=len(image_paths),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        # Map to loading/process function
        tf_dataset = tf_dataset.map(self.process_image_data, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch dataset to avoid memory crash
        tf_dataset = tf_dataset.ragged_batch(self.batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        return tf_dataset

    def process_xviewdataset_workflow(self) -> None:
        """
        Workflow to run preparation of images and annotations for model training/validation/testing.
        """
        object_annotations = self.collect_object_annotations()
        img_annotaions = self.collect_image_annotations(object_annotations)
        stratified_split_datasets = self.stratified_dataset_split(img_annotaions)

        tiled_datasets = ImageAnnotationDatasets(
            training=self.slice_dataset(
                stratified_split_datasets.training,
                "training",
                sample_negative_tiles=True,
            ),
            validation=self.slice_dataset(
                stratified_split_datasets.validation,
                "validation",
            ),
            testing=self.slice_dataset(
                stratified_split_datasets.testing,
                "testing",
            ),
        )

        train_dataset = self.process_dataset(tiled_datasets.training, training=True)
        val_dataset = self.process_dataset(tiled_datasets.validation)
        test_dataset = self.process_dataset(tiled_datasets.testing)

        self._ori_datasets = stratified_split_datasets
        self._processed_datasets = ImageAnnotationDatasets(
            training=train_dataset, validation=val_dataset, testing=test_dataset
        )
