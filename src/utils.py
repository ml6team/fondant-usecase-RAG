import glob
import itertools
import json
import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from itertools import product

import pandas as pd
import pipeline_eval
import pipeline_index
import weaviate
from fondant.pipeline.runner import DockerRunner


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


def cartesian_product(input_dict):
                return (dict(zip(input_dict.keys(), values)) for values in product(*input_dict.values()))


def extract_timestamp(folder_name):
    timestamp_str = folder_name.split("-")[-1]
    return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")


def has_parquet_file(data_directory, entry, component_name):
    component_folder = os.path.join(data_directory, entry, component_name)
    # Check if the component exists
    if not os.path.exists(component_folder) or not os.path.isdir(component_folder):
        return False
    parquet_files = [file for file in os.listdir(component_folder) if file.endswith(".parquet")]
    return bool(parquet_files)


def get_metrics_latest_run(base_path,
                      pipeline_name="evaluation-pipeline",
                      component_name = "aggregate_eval_results"):

    data_directory = f"{base_path}/{pipeline_name}"
    
    # keep data folders that belong to pipeline and contain parquet file
    valid_entries = [d
        for d in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, d))
        and d.startswith(pipeline_name)
        and has_parquet_file(data_directory, d, component_name)]

    # keep the latest folder
    latest_run = sorted(valid_entries, key=extract_timestamp, reverse=True)[0]

    # read all Parquet files and concatenate them into a single DataFrame
    component_folder = os.path.join(data_directory, latest_run, component_name)
    parquet_files = [f for f in os.listdir(component_folder) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(component_folder, file)) for file in parquet_files]

    return pd.concat(dfs, ignore_index=True).set_index('metric')['score'].to_dict()

class ParameterSearch:
    """RAG parameter search"""
    
    def __init__(self,
                search_method,
                searchable_index_params,
                searchable_eval_params,
                shared_args,
                index_args,
                eval_args):
        self.search_method = search_method
        self.searchable_index_params = searchable_index_params
        self.searchable_eval_params = searchable_eval_params
        self.shared_args = shared_args
        self.index_args = index_args
        self.search_method = search_method
        self.eval_args = eval_args

        # list of dicts to store all params & results
        self.results = []

        # create directory for pipeline output data
        self.base_path = create_directory_if_not_exists(shared_args['base_path'])

        # mount directory of pipeline output data from docker
        self.extra_volumes = [str(os.path.join(os.path.abspath(eval_args['evaluation_set_path']))) + ":/evaldata"]

        # access Weaviate vector store using local ip
        self.weaviate_url = f"http://{get_host_ip()}:8080"

        # define pipeline runner
        self.runner = DockerRunner()

    
    def run(self):
        
        runcount = 0
        
        while True:

            pipelines = self.create_pipelines(runcount)

            if pipelines is None:
                break
            
            # create pipelines
            indexing_config, indexing_pipeline, evaluation_config, evaluation_pipeline = pipelines
    
            # run indexing pipeline
            self.run_indexing_pipeline(runcount, indexing_config, indexing_pipeline)
   
            # run evaluation pipeline
            self.run_evaluation_pipeline(runcount, evaluation_config, evaluation_pipeline)

            # read metrics from pipeline output
            metrics = get_metrics_latest_run(self.base_path)

            metadata = {
                'run_number' : runcount,
                'date_time' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
           
            # collect results
            self.results.append({**metadata, **indexing_config, **evaluation_config, **metrics})

            runcount += 1

        return pd.DataFrame(self.results)

    
    def create_pipelines(self, runcount):
        if self.search_method == 'grid_search':

            # all possible combinations of parameters
            all_combinations = list(cartesian_product(self.searchable_index_params|self.searchable_eval_params))

            # when all combinations have been tried, stop searching
            if runcount > len(all_combinations) - 1:
                return None

            # create config for indexing pipeline
            indexing_config = all_combinations[runcount]       
            indexing_config = {key: indexing_config[key] for key in self.searchable_index_params}
            indexing_config['embed_model_provider'] = indexing_config['embed_model'][0]
            indexing_config['embed_model'] = indexing_config['embed_model'][1]

            # create indexing pipeline
            indexing_pipeline = pipeline_index.create_pipeline(
                **self.shared_args,
                **self.index_args,
                **indexing_config
            )

            # create config for evaluation pipeline
            evaluation_config = all_combinations[runcount]
            evaluation_config = {key: evaluation_config[key] for key in self.searchable_eval_params}

             # create evaluation pipeline
            evaluation_pipeline = pipeline_eval.create_pipeline(
                **self.shared_args,
                **self.eval_args,
                **evaluation_config
            )
                        
            return indexing_config, indexing_pipeline, evaluation_config, evaluation_pipeline

        elif self.search_method == 'progressive_search':
            pass

    def run_indexing_pipeline(self, runcount, indexing_config, indexing_pipeline):
        logging.info(f'Starting indexing pipeline of run #{runcount} with {indexing_config}')       
        self.runner.run(indexing_pipeline)

    def run_evaluation_pipeline(self, runcount, evaluation_config, evaluation_pipeline):
        logging.info(f'Starting evaluation pipeline of run #{runcount} with {evaluation_config}')
        self.runner.run(input=evaluation_pipeline, extra_volumes=self.extra_volumes)

def add_embed_model_numerical_column(df):
    df['embed_model_numerical'] = pd.factorize(df['embed_model'])[0] + 1
    return df

def show_legend_embed_models(df):
    columns_to_show = ['embed_model','embed_model_numerical']
    df = df[columns_to_show].drop_duplicates().set_index('embed_model_numerical')
    df.index.name = ''
    return df