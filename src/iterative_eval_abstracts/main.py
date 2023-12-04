import logging
import os
from pathlib import Path

import pandas as pd
from fondant.pipeline import ComponentOp, Pipeline
from fondant.pipeline.compiler import DockerCompiler
from fondant.pipeline.runner import DockerRunner

import weaviate
from iterative_eval_abstracts.read_output_data import read_latest_data

logger = logging.getLogger(__name__)

class IterativeEvaluation():
    def __init__(
            self,
            pipeline_dir: str,
            embed_model_provider: str,
            embed_model: str,
            weaviate_url: str,
            weaviate_class_name: str
    ):
        self.base_path = pipeline_dir
        self.embed_model_provider = embed_model_provider
        self.embed_model = embed_model
        self.weaviate_url = weaviate_url
        self.weaviate_class_name = weaviate_class_name

    def run_indexing_pipeline(
            self,
            hf_dataset_name: str,
            data_column_name: str,
            n_rows_to_load: int,
            chunk_size: int,
            chunk_overlap: int,
    ) -> dict:

        Path(self.base_path).mkdir(parents=True, exist_ok=True)

        pipeline = Pipeline(
            pipeline_name="ingestion-pipeline",  # Add a unique pipeline name to easily track your progress and data
            pipeline_description="Pipeline to prepare and process data for building a RAG solution",
            base_path=self.base_path, # The demo pipelines uses a local directory to store the data.
        )

        load_from_hf_hub = ComponentOp(
        component_dir="components/load_from_hf_hub",
        arguments={
            # Add arguments
            "dataset_name": hf_dataset_name,
            # Define the column mapping between the huggingface dataset and the Fondant dataframe
            "column_name_mapping": {
                data_column_name: "text"
            },
            "n_rows_to_load": n_rows_to_load
            }
        )
        pipeline.add_op(load_from_hf_hub)

        chunk_text_op = ComponentOp.from_registry(
            name="chunk_text",
            arguments={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )
        pipeline.add_op(chunk_text_op, dependencies=load_from_hf_hub)

        embed_text_op = ComponentOp.from_registry(
            name="embed_text",
            arguments={
                "model_provider": self.embed_model_provider,
                "model": self.embed_model,
            }
        )
        pipeline.add_op(embed_text_op, dependencies=chunk_text_op)

        index_weaviate_op = ComponentOp.from_registry(
            name="index_weaviate",
            arguments={
                "weaviate_url": self.weaviate_url,
                "class_name": self.weaviate_class_name,  # Add a unique class name to show up on the leaderboard
            }
        )
        pipeline.add_op(index_weaviate_op, dependencies=embed_text_op)

        DockerCompiler().compile(pipeline, output_path="docker-compose.yaml")
        DockerRunner().run("docker-compose.yaml")

        weaviate_client = weaviate.Client(self.weaviate_url)

        return weaviate_client.schema.get(self.weaviate_class_name)
    
    def run_evaluation_pipeline(
            self,
            csv_dataset_uri: str,
            csv_column_separator: str,
            question_column_name: str,
            top_k: int,
            llm_name: str,
            llm_kwargs: dict,
            metrics: list
    ) -> pd.DataFrame:

        Path(self.base_path).mkdir(parents=True, exist_ok=True)

        pipeline_eval = Pipeline(
            pipeline_name="evaluation-pipeline",  # Add a unique pipeline name to easily track your progress and data
            pipeline_description="Pipeline to evaluate \
            a RAG solution",
            base_path=self.base_path, # The demo pipelines uses a local directory to store the data.
        )

        load_from_csv = ComponentOp(
            component_dir="components/load_from_csv",
            arguments={
                # Add arguments
                "dataset_uri": csv_dataset_uri,
                "column_separator": csv_column_separator,
                "column_name_mapping": {question_column_name: "text"},
            },
        )

        pipeline_eval.add_op(load_from_csv)

        embed_text_op = ComponentOp.from_registry(
            name="embed_text",
            arguments={
                "model_provider": self.embed_model_provider,
                "model": self.embed_model,
            },
        )
        pipeline_eval.add_op(embed_text_op, dependencies=load_from_csv)

        #TODO: modify when components are in the registry
        retrieve_chunks = ComponentOp(
            component_dir="components/retrieve_from_weaviate",
            arguments={
                "weaviate_url": self.weaviate_url,
                "class_name": self.weaviate_class_name,
                "top_k": top_k,
            },
        )
        pipeline_eval.add_op(retrieve_chunks, dependencies=embed_text_op)

        retriever_eval = ComponentOp(
            component_dir="components/retriever_eval",
            arguments={
                "llm_name": llm_name,
                "llm_kwargs": llm_kwargs,
                "metrics": metrics,
            },
        )
        pipeline_eval.add_op(retriever_eval, dependencies=retrieve_chunks)

        aggregate_results = ComponentOp(
            component_dir="components/aggregate_eval_results",
        )
        pipeline_eval.add_op(aggregate_results, dependencies=retriever_eval)

        DockerCompiler().compile(pipeline_eval, output_path="docker-compose.yaml")
        DockerRunner().run("docker-compose.yaml")

        return read_latest_data(
            base_path=self.base_path,
            pipeline_name="evaluation-pipeline",
            component_name="aggregate_eval_results",
        )
        