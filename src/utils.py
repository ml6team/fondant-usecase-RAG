import glob
import itertools
import json
import logging
import os
import socket
from datetime import datetime
from pathlib import Path

import pandas as pd
import pipeline_eval
import pipeline_index
import weaviate
from fondant.pipeline.runner import DockerRunner

COMPONENT_NAME = "aggregate_eval_results"


def get_host_ip():
    try:
        # Create a socket object and connect to an external server
        # This step is done to get the local machine's IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error while retrieving host IP address: {e}")
        host_ip = None
    finally:
        s.close()

    return host_ip


def create_directory_if_not_exists(path):
    p_base_path = Path(path).resolve()
    p_base_path.mkdir(parents=True, exist_ok=True)
    return str(p_base_path)


def store_results(pipeline_name, **kwargs):
    base_path = kwargs.pop("base_path")

    del kwargs["weaviate_url"]
    del kwargs["embed_api_key"]  # API key

    run_dir = get_latest_run(base_path, pipeline_name)
    param_file = os.path.join(run_dir, "params.json")
    with open(param_file, "w") as f:
        json.dump(kwargs, f)


def read_results(
    pipeline_name,
    base_path,
):
    runs = get_runs(base_path, pipeline_name)
    dfs = []
    for run in runs:
        component_path = os.path.join(base_path, pipeline_name, run, COMPONENT_NAME)
        params_file = os.path.join(component_path, "params.json")
        if not os.path.exists(params_file):
            continue
        with open(params_file) as f:
            params = json.load(f)

        # Read params
        params_df = pd.DataFrame(params, index=[0]).reset_index(drop=True)

        # Read metrics
        parquet_path = glob.glob(os.path.join(component_path, "*.parquet"))
        metrics_df = pd.read_parquet(parquet_path).reset_index(drop=True)
        metrics_df = pd.DataFrame(
            dict(zip(metrics_df["metric"], metrics_df["score"])),
            index=[0],
        )

        # Join params and metrics
        results_df = params_df.join(metrics_df)

        dfs.append(results_df)

    return pd.concat(dfs).reset_index(drop=True)


def get_runs(base_path: str, pipeline_name: str):
    # Specify the path to the 'data' directory
    data_directory = f"{base_path}/{pipeline_name}"

    # Get a list of all subdirectories in the 'data' directory
    subdirectories = [
        d
        for d in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, d))
    ]

    # keep pipeline directories
    valid_entries = [
        entry for entry in subdirectories if entry.startswith(pipeline_name)
    ]
    # keep pipeline folders containing a parquet file in the component folder

    return [
        folder
        for folder in valid_entries
        if has_parquet_file(data_directory, folder, COMPONENT_NAME)
    ]


def get_latest_run(base_path: str, pipeline_name: str):
    runs = get_runs(base_path, pipeline_name)

    # keep the latest folder
    latest_run = sorted(runs, key=extract_timestamp, reverse=True)[0]
    return os.path.join(base_path, pipeline_name, latest_run, COMPONENT_NAME)


def read_latest_data(base_path: str, pipeline_name: str):
    component_folder = get_latest_run(base_path, pipeline_name)

    # If a valid folder is found, proceed to read all Parquet files in the component folder
    if component_folder:
        # Get a list of all Parquet files in the component folder
        parquet_files = [
            f for f in os.listdir(component_folder) if f.endswith(".parquet")
        ]

        if parquet_files:
            # Read all Parquet files and concatenate them into a single DataFrame
            dfs = [
                pd.read_parquet(os.path.join(component_folder, file))
                for file in parquet_files
            ]
            return pd.concat(dfs, ignore_index=True)
        return None
    return None


def has_parquet_file(data_directory, entry, component_name):
    component_folder = os.path.join(data_directory, entry, component_name)
    # Check if the component exists
    if not os.path.exists(component_folder) or not os.path.isdir(component_folder):
        return False
    parquet_files = [
        file for file in os.listdir(component_folder) if file.endswith(".parquet")
    ]
    return bool(parquet_files)


def extract_timestamp(folder_name):
    # Extract the timestamp part from the folder name
    timestamp_str = folder_name.split("-")[-1]
    # Convert the timestamp string to a datetime object
    return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")


# Output pipelines evaluations results dataframe
def output_results(results):
    flat_results = []

    for entry in results:
        flat_entry = entry.copy()

        for key, value in entry.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_entry[sub_key] = sub_value
                del flat_entry[key]

            elif isinstance(value, pd.DataFrame):
                for sub_key, sub_value in zip(value["metric"], value["score"]):
                    flat_entry[sub_key] = sub_value
                del flat_entry[key]

        flat_results.append(flat_entry)

    return pd.DataFrame(flat_results)


def run_parameters_search(  # noqa: PLR0913
    extra_volumes,
    fixed_args,
    fixed_index_args,
    fixed_eval_args,
    chunk_sizes,
    chunk_overlaps,
    embed_models,
    top_ks,
):
    # Define pipeline runner
    runner = DockerRunner()

    # Results dictionary to store results for each iteration
    results_dict = {}

    # Perform grid search
    indexes = []
    for i, (chunk_size, chunk_overlap, embed_model) in enumerate(
        itertools.product(chunk_sizes, chunk_overlaps, embed_models),
        start=1,
    ):
        index_config_class_name = f"IndexConfig{i}"
        index_pipeline_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(
            f"Running indexing for {index_config_class_name} \
            with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}\
            , embed_model={embed_model}",
        )

        # Store Indexing configuration
        index_dict = {}
        index_dict["index_name"] = index_config_class_name
        index_dict["indexing_datetime"] = index_pipeline_datetime
        index_dict["chunk_size"] = chunk_size
        index_dict["chunk_overlap"] = chunk_overlap
        index_dict["embed_model"] = embed_model
        indexes.append(index_dict)

        # Create and Run the indexing pipeline
        indexing_pipeline = pipeline_index.create_pipeline(
            **fixed_args,
            **fixed_index_args,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model_provider=embed_model[0],
            embed_model=embed_model[1],
            weaviate_class=index_config_class_name,
        )
        run_indexing_pipeline(
            runner=runner,
            index_pipeline=indexing_pipeline,
            weaviate_url=fixed_args["weaviate_url"],
            weaviate_class=index_config_class_name,
        )

    parameters_search_results = []
    for i, (index_dict, top_k) in enumerate(
        itertools.product(indexes, top_ks),
        start=1,
    ):
        rag_config_name = f"RAGConfig{i}"
        eval_pipeline_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(
            f"Running evaluation for {rag_config_name} \
            with {index_dict['index_name']} and {top_k} retrieved chunks",
        )

        # Store RAG pipeline configuration
        results_dict = {}
        results_dict["rag_config_name"] = rag_config_name
        results_dict["evaluation_datetime"] = eval_pipeline_datetime
        results_dict.update(index_dict)
        results_dict["top_k"] = top_k

        # Create and Run the evaluation pipeline
        evaluation_pipeline = pipeline_eval.create_pipeline(
            **fixed_args,
            **fixed_eval_args,
            embed_model_provider=index_dict["embed_model"][0],
            embed_model=index_dict["embed_model"][1],
            weaviate_class=index_dict["index_name"],
            retrieval_top_k=top_k,
        )
        run_evaluation_pipeline(
            runner=runner,
            eval_pipeline=evaluation_pipeline,
            extra_volumes=extra_volumes,
        )

        # Save the evaluation results in the dictionary
        results_dict[f"agg_results_{rag_config_name}"] = read_latest_data(
            base_path=fixed_args["base_path"],
            pipeline_name="evaluation-pipeline",
        )
        # Add fixed arguments
        results_dict.update(fixed_args)
        results_dict.update(fixed_index_args)
        results_dict.update(fixed_eval_args)

        parameters_search_results.append(results_dict)

    return parameters_search_results


# index pipeline runner
def run_indexing_pipeline(runner, index_pipeline, weaviate_url, weaviate_class):
    runner.run(index_pipeline)
    docker_weaviate_client = weaviate.Client(weaviate_url)
    return docker_weaviate_client.schema.get(weaviate_class)


# eval pipeline runner
def run_evaluation_pipeline(runner, eval_pipeline, extra_volumes):
    runner.run(input=eval_pipeline, extra_volumes=extra_volumes)
