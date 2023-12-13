import os
import socket
from datetime import datetime
from pathlib import Path

import pandas as pd


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

# Store pipeline results
def store_results(
        rag_results,
        shared_args,
        indexing_args,
        evaluation_args,
        index_pipeline_datetime,
        eval_pipeline_datetime
        ):

    pipeline_dir = shared_args["pipeline_dir"]
    pipeline_name = "evaluation-pipeline"
    component_name = "aggregate_eval_results"

    results_dict = {}
    results_dict["shared_args"] = shared_args
    results_dict["indexing_datetime"] = index_pipeline_datetime
    results_dict["indexing_args"] = indexing_args
    results_dict["evaluation_args"] = evaluation_args
    results_dict["evaluation_datetime"] = eval_pipeline_datetime
    results_dict["agg_metrics"] = read_latest_data(
                base_path=pipeline_dir,
                pipeline_name=pipeline_name,
                component_name=component_name,
            )

    rag_results.append(results_dict)

    return rag_results


def read_latest_data(base_path: str, pipeline_name: str, component_name: str):
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
    valid_entries = [
        folder
        for folder in valid_entries
        if has_parquet_file(data_directory, folder, component_name)
    ]
    # keep the latest folder
    latest_folder = sorted(valid_entries, key=extract_timestamp, reverse=True)[0]

    # If a valid folder is found, proceed to read all Parquet files in the component folder
    if latest_folder:
        # Find the path to the component folder
        component_folder = os.path.join(data_directory, latest_folder, component_name)

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
def output_results(rag_results):
    flat_data = []
    for results in rag_results:
        flat_results = results.copy()

        # Flatten shared_args
        for key, value in results['shared_args'].items():
            flat_results[key] = value

        # Flatten indexing_args
        for key, value in results['indexing_args'].items():
            flat_results[key] = value

        # Flatten evaluation_args
        for key, value in results['evaluation_args'].items():
            if key == 'llm_kwargs':
                for k, v in value.items():
                    flat_results[f'evaluation_args_{k}'] = v
            else:
                flat_results[f'evaluation_args_{key}'] = value

        # Flatten agg_metrics
        agg_metrics_df = results['agg_metrics']
        for metric, score in zip(agg_metrics_df['metric'], agg_metrics_df['score']):
            flat_results[metric] = score

        # Remove nested dictionaries
        del flat_results['shared_args']
        del flat_results['indexing_args']
        del flat_results['evaluation_args']
        del flat_results['agg_metrics']

        flat_data.append(flat_results)

        return pd.DataFrame(flat_data)