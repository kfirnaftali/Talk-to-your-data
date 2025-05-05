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
A machine learning streaming inference example for the Dataflow Solution Guides.
"""

import time
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
from ml_ai_pipeline.pipeline import create_pipeline


class MyPipelineOptions(PipelineOptions):
  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument("--input_gaming_sub", required=True, type=str,
                        help="Pub/Sub subscription for input gaming data.")
    parser.add_argument("--input_questions_sub", required=True, type=str,
                        help="Pub/Sub subscription for input questions.")
    parser.add_argument("--output_sql_topic", required=True, type=str,
                        help="Pub/Sub topic for generated SQL query output.")
    parser.add_argument("--sql_gen_model_name", type=str, default="gemma_2b",
                        help="Gemma model name/path for keras_hub.models.GemmaCausalLM.from_preset (e.g., 'gemma_2b').")
    parser.add_argument("--sql_gen_max_length", type=int, default=64,
                        help="Max generation length for the SQL generation model.")


if __name__ == "__main__":


    pipeline_options = PipelineOptions(save_main_session=True)
    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)

    current_time_suffix = int(time.time())
    google_cloud_options.job_name = f"gemma-inference-pipeline-{current_time_suffix}"

    # Directly get custom options and run
    custom_options = pipeline_options.view_as(MyPipelineOptions)


    pipeline = create_pipeline(custom_options)
    result = pipeline.run()
