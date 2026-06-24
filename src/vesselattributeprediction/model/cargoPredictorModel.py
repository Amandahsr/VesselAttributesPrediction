from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from keras_hub.models import CLIPBackbone, CLIPPreprocessor

from vesselattributeprediction.constants import DEFAULT_CLIP_MODEL


class CargoPredictorModel:
    """
    Model using CLIP to predict cargo status based on vessel image and vessel type prompts.
    """

    def __init__(
        self,
        clip_model: str = DEFAULT_CLIP_MODEL,
    ) -> None:
        self.clip_model_name: str = clip_model
        self.clip_preprocessor: Optional[CLIPPreprocessor] = None
        self.clip_model: Optional[CLIPBackbone] = None

        self.initialize_clip_model()

    def initialize_clip_model(self) -> None:
        """
        Initializes a pretrained CLIP model.
        """
        self.clip_preprocessor = CLIPPreprocessor.from_preset(self.clip_model_name)
        self.clip_model = CLIPBackbone.from_preset(self.clip_model_name)

    def _generate_cargo_status_prompts(self, vessel_type: str) -> List[str]:
        """
        Dynamically generates cargo status CLIP prompts for the detected vessel type.
        """
        return [
            f"a satellite optical image crop of a {vessel_type} carrying visible cargo",
            f"a satellite optical image crop of a {vessel_type} not carrying visible cargo",
        ]

    def _map_prompt_to_cargo_status(self, prompt: str) -> bool:
        """
        Maps a sentence prompt to a boolean cargo status.
        """
        if "not carrying visible cargo" in prompt:
            return False

        return True

    def predict_vessel_cargo_status(
        self,
        vessel_image: np.ndarray,
        vessel_type: str,
    ) -> Tuple[bool, float]:
        """
        Runs CLIP scoring and returns cargo status and confidence.
        """
        # Unnormalize images since CLIP model does normalization
        candidate_prompts = self._generate_cargo_status_prompts(vessel_type)
        candidate_prompts_tf = tf.constant(
            candidate_prompts,
            dtype=tf.string,
        )
        image_batch = tf.expand_dims(
            tf.convert_to_tensor(vessel_image, dtype=tf.float32) * 255.0,
            axis=0,
        )
        processed_inputs = self.clip_preprocessor(
            {
                "images": image_batch,
                "prompts": candidate_prompts_tf,
            }
        )

        # Run CLIP model
        model_inputs = {
            "images": processed_inputs["images"],
            "token_ids": processed_inputs["token_ids"],
        }
        output_features = self.clip_model(model_inputs, training=False)

        probabilities = tf.nn.softmax(
            output_features["vision_logits"][0],
            axis=-1,
        )
        predicted_index = int(tf.argmax(probabilities).numpy())
        confidence = float(probabilities[predicted_index].numpy())
        prompt = candidate_prompts[predicted_index]

        return self._map_prompt_to_cargo_status(prompt), confidence
