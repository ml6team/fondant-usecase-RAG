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


# Read latest chosen component
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
