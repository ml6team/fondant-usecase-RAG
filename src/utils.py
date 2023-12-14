import glob
import json
import os
import socket
from datetime import datetime
from pathlib import Path

import pandas as pd

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
