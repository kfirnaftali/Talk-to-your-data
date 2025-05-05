# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Beam pipeline for SQL generation using Gemma model via keras_hub.
"""

import logging
import json
import pandas as pd
from typing import List, Dict, Tuple, Any
import time

import apache_beam as beam
from apache_beam import Pipeline, PCollection, pvalue
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
from apache_beam.ml.inference import RunInference
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.io.gcp import pubsub
from apache_beam.transforms import window, core
from apache_beam.transforms.combiners import ToListCombineFn

from .model_handlers import GemmaModelHandler # Uses keras_hub internally


class ParseJsonDoFn(beam.DoFn):
    def process(self, element: str):
        try:
            yield json.loads(element)
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error: {element[:200]}... Error: {e}")
        except Exception as e:
            logging.error(f"Failed to parse JSON: {element[:200]}... Error: {e}")

class FormatSQLPromptFromDataFrame(beam.DoFn):
    def process(self, element: Tuple[Any, List[Dict[str, Any]]], question_str: str, window_param=beam.DoFn.WindowParam):
        user_id, gaming_events_list = element
        window_str = str(window_param)
        if not question_str:
            if gaming_events_list:
                logging.warning(f"Win {window_str} User {user_id}: Data received for user but no question available.")
            return
        if not gaming_events_list: # Check if the list is empty
            if question_str:
                logging.warning(f"Win {window_str} User {user_id}: Question '{question_str[:50]}...' received but no gaming data for user.")
            return

        logging.info(f"Win {window_str}: Formatting prompt for question: '{question_str}' from {len(gaming_events_list)} records.")
        try:
            df_window = pd.DataFrame(gaming_events_list)
            if df_window.empty:
                logging.warning(f"Win {window_str} User {user_id}: DataFrame is empty for user after creating from gaming_events_list.")
                sample_data_md = "No data in window for this user."
                # Depending on requirements, you might want to return here or proceed with an empty data prompt
            else:
                sample_data_md = df_window.head(3).to_markdown(index=False)

            schema_str = ", ".join(df_window.columns.astype(str)) if not df_window.empty else "N/A (no data)"

            input_prompt = f"""Context: Based *only* on the schema and sample data from a 1-minute window of 'gaming_events', generate a SQL query for the user's question. Table name: 'gaming_events'.
Schema: {schema_str}
Sample Data (first 3 rows):
{sample_data_md}
User Question: "{question_str}"
SQL Query:"""
            yield input_prompt
        except Exception as e:
            logging.exception(f"Win {window_str}: Error formatting prompt for question '{question_str}': {e}")

@beam.ptransform_fn
def ReadAndDecodePubSub(p: Pipeline, subscription: str, name: str) -> PCollection[str]:
    logging.info(f"PTransform '{name}': Reading from Pub/Sub: '{subscription}'")
    if not subscription: raise ValueError(f"PTransform '{name}': Invalid subscription: '{subscription}'")
    messages = p | f"ReadPubSub_{name}" >> pubsub.ReadFromPubSub(subscription=subscription)
    return messages | f"Decode_{name}" >> beam.Map(lambda x: x.decode("utf-8")).with_output_types(str)

def create_pipeline(options) -> Pipeline:
    logging.info("Creating SQL Generation pipeline (keras_hub) with options: %s", options)
    pipeline = beam.Pipeline(options=options)
    WINDOW_DURATION_SECONDS = 60

    is_local_run = not isinstance(options, GoogleCloudOptions) or "PrismRunner" in options.view_as(
        PipelineOptions).runner

    if is_local_run:
        logging.info("Local run (e.g., PrismRunner) detected: Using beam.Create for I/O.")
        sample_gaming_data_json_strings = [
            json.dumps({
                "user_id": "1", "event_type": "join_game", "datetime": "Tue, 29 Apr 2025 07:53:46 UTC",
                "coins_awarded": "66", "coins_purchased": 245, "activity_type": "trading",
                "region": "EU-Central", "is_new_user": False, "platform": "android"
            }),
            json.dumps({
                "user_id": "2", "event_type": "collect_reward", "datetime": "Tue, 29 Apr 2025 07:54:00 UTC",
                "coins_awarded": "10", "coins_purchased": 0, "activity_type": "daily_bonus",
                "region": "NA-West", "is_new_user": True, "platform": "ios"
            }),
            json.dumps({
                "user_id": "1", "event_type": "purchase_item", "datetime": "Tue, 29 Apr 2025 07:55:12 UTC",
                "coins_awarded": "0", "coins_purchased": 100, "activity_type": "in-app-purchase",
                "region": "EU-Central", "is_new_user": False, "platform": "android"
            })
        ]
        sample_question_strings = [
            "What are the total coins awarded for user 1?"
        ]
        gaming_data_msgs = pipeline | "CreateGamingData" >> beam.Create(sample_gaming_data_json_strings)
        question_msgs = pipeline | "CreateQuestions" >> beam.Create(sample_question_strings)
    else:
        gaming_data_msgs = pipeline | "ReadGamingData" >> ReadAndDecodePubSub(options.input_gaming_sub, "GamingData")
        question_msgs = pipeline | "ReadQuestions" >> ReadAndDecodePubSub(options.input_questions_sub, "Questions")

    parsed_gaming_data = gaming_data_msgs | "ParseGamingJson" >> beam.ParDo(ParseJsonDoFn())

    # create a tuple of key value by user_id, (user_id: 111, {data})
    keyed_gaming_data = parsed_gaming_data | "KeyByUserId" >> beam.Map(lambda x: (x.get('user_id'), x))
    windowed_data = keyed_gaming_data | "WindowGamingData" >> beam.WindowInto(window.FixedWindows(WINDOW_DURATION_SECONDS))

    window_data_list = windowed_data | "CollectDataInWindow" >> core.CombinePerKey(ToListCombineFn())
    # window_data_list = ("user123", [{"event_type": "login", ...}, {"event_type": "view_item", ...}])
    question_side_input = beam.pvalue.AsSingleton(question_msgs)

    sql_prompts = window_data_list | "FormatSQLPrompt" >> beam.ParDo(
        FormatSQLPromptFromDataFrame(), question_side_input
    )

    # Use model_name and max_length from options for GemmaModelHandler
    sql_gen_handler = GemmaModelHandler(
        model_name=options.sql_gen_model_name,
        max_length=options.sql_gen_max_length
    )
    sql_predictions = sql_prompts | "RunSQLGemmaInference" >> RunInference(sql_gen_handler)
    generated_sql = sql_predictions | "ExtractGeneratedSQL" >> beam.Map(lambda r: str(r.inference).strip())

    (generated_sql
        | "EncodeSQLToBytes" >> beam.Map(lambda s: s.encode('utf-8')).with_output_types(bytes)
        | "PublishSQLToPubSub" >> pubsub.WriteToPubSub(topic=options.output_sql_topic)
    )
    return pipeline

