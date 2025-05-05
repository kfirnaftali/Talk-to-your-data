#  Copyright 2024 Google LLC
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
#  limitations under the License.   --save_main_session \

python main.py \
  --runner=PrismRunner \
  --project=$PROJECT \
  --temp_location=gs://dataflow-gemma-weights/tmp \
  --region=$REGION \
  --machine_type=$MACHINE_TYPE \
  --num_workers=1 \
  --disk_size_gb=$DISK_SIZE_GB \
  --max_num_workers=$MAX_DATAFLOW_WORKERS \
  --no_use_public_ip \
  --subnetwork="regions/us-central1/subnetworks/dataflow-vpc" \
  --sdk_container_image=$CONTAINER_URI \
  --dataflow_service_options="worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver:5xx" \
  --input_gaming_sub=projects/demos-337108/subscriptions/gaming-data-sub \
  --output_sql_topic=projects/demos-337108/topics/gaming-data-output \
  --input_questions_sub=projects/demos-337108/subscriptions/cfo-questions-sub \
  --max_length=128 \
  --model_path="gemma_2B" \
  --sql_gen_model_path="gemma_2B" \
  --setup_file=./setup.py

