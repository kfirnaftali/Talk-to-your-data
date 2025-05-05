#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Custom model handlers to be used with RunInference, using keras_hub.
"""

from typing import Sequence, Optional, Any, Iterable
import logging

import keras_hub # Using keras_hub as per user confirmation
from apache_beam.ml.inference.base import ModelHandler, PredictionResult
try:
    from keras_hub.models import GemmaCausalLM # Using keras_hub as per user confirmation
except ImportError:
    logging.error("Could not import GemmaCausalLM from keras_hub.models. Ensure your environment provides keras_hub.")
    # Define a dummy class for type hinting if import fails
    class GemmaCausalLM:
        @staticmethod
        def from_preset(name): raise NotImplementedError("Import failed")
        def generate(self, text, max_length): raise NotImplementedError("Import failed")


class GemmaModelHandler(ModelHandler[str, PredictionResult, GemmaCausalLM]):
  """
  A RunInference model handler for the Gemma model using keras_hub.
  """

  def __init__(self, model_name: str = "gemma_2b", max_length: int = 64):
    """ Implementation of the ModelHandler interface for Gemma using text as input.

    Args:
      model_name: The Gemma model name for keras_hub.from_preset (e.g., "gemma_2b").
      max_length: The maximum length for the generated sequence.
    """
    super().__init__()
    self._model_name = model_name
    self._max_length = max_length
    self._model = None
    self._env_vars = {}
    logging.info(f"Initializing GemmaModelHandler (keras_hub) with model: {self._model_name}, max_length: {self._max_length}")

  def share_model_across_processes(self) -> bool:
    return True

  def load_model(self) -> GemmaCausalLM:
    if self._model is None:
        logging.info(f"Loading keras_hub model: {self._model_name}")
        try:
            self._model = keras_hub.models.GemmaCausalLM.from_preset(self._model_name)
            logging.info(f"keras_hub model {self._model_name} loaded successfully.")
        except Exception as e:
            logging.exception(f"Failed to load keras_hub model {self._model_name}: {e}")
            raise
    return self._model

  def run_inference(
      self,
      batch: Sequence[str],
      model: GemmaCausalLM,
      inference_args: Optional[dict[str, Any]] = None) -> Iterable[PredictionResult]: # Changed unused to inference_args
    logging.debug(f"Running keras_hub inference: batch_size={len(batch)}, max_length={self._max_length}")
    results = []
    try:
        for one_text in batch:
          result_text = model.generate(one_text, max_length=self._max_length)
          # Using model_id in PredictionResult for better tracking if multiple models were used
          results.append(PredictionResult(example=one_text, inference=result_text, model_id=self._model_name))
    except Exception as e:
        logging.exception(f"Error during keras_hub inference: {e}")
        results = [PredictionResult(example=prompt, inference=f"Inference Error: {e}", model_id=self._model_name) for prompt in batch]
    return results