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

def add_embed_model_numerical_column(df):
    df['embed_model_numerical'] = pd.factorize(df['embed_model'])[0] + 1
    return df

def show_legend_embed_models(df):
    columns_to_show = ['embed_model','embed_model_numerical']
    df = df[columns_to_show].drop_duplicates().set_index('embed_model_numerical')
    df.index.name = ''
    return df


class ParameterSearch:
    """RAG parameter search"""
    
    def __init__(self,
                 searchable_index_params,
                 searchable_eval_params,
                 searchable_shared_params,
                 index_args,
                 eval_args,
                 shared_args,
                 search_method = 'progressive_search',
                 target_metric = 'context_precision'):
        self.search_method = search_method
        self.target_metric = target_metric
        self.searchable_index_params = searchable_index_params
        self.searchable_shared_params = searchable_shared_params
        self.searchable_eval_params = searchable_eval_params
        self.shared_args = shared_args
        self.searchable_params = {**searchable_index_params, **searchable_eval_params, **searchable_shared_params}
        self.index_args = index_args
        self.search_method = search_method
        self.eval_args = eval_args

        # create directory for pipeline output data
        self.base_path = create_directory_if_not_exists(shared_args['base_path'])

        # mount directory of pipeline output data from docker
        self.extra_volumes = [str(os.path.join(os.path.abspath(eval_args['evaluation_set_path']))) + ":/evaldata"]

        # define pipeline runner
        self.runner = DockerRunner()

        # add url to shared arguments to access Weaviate from within Docker
        self.shared_args["weaviate_url"] = f"http://{get_host_ip()}:8080" # IP address

        # list of dicts to store all params & results
        self.results = []
    
    def run(self):
        
        runcount = 0
        
        while True:

            configs = self.create_configs(runcount)

            if configs is None:
                break
            
            # create configs
            indexing_config, evaluation_config = configs

            # create pipeline objects
            indexing_pipeline, evaluation_pipeline = self.create_pipelines(indexing_config, evaluation_config)
    
            # run indexing pipeline
            self.run_indexing_pipeline(runcount, indexing_config, indexing_pipeline)
   
            # run evaluation pipeline
            self.run_evaluation_pipeline(runcount, evaluation_config, evaluation_pipeline)

            # read metrics from pipeline output
            metrics={}
            metrics = get_metrics_latest_run(self.base_path)

            metadata = {
                'run_number' : runcount,
                'date_time' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
           
            # collect results
            self.results.append({**metadata, **indexing_config, **evaluation_config, **metrics})

            runcount += 1

        return pd.DataFrame(self.results)
    
    def create_configs(self, runcount):

        if self.search_method == 'grid_search':

            # all possible combinations of parameters
            all_combinations = list(cartesian_product(self.searchable_params))

            # when all combinations have been tried, stop searching
            if runcount > len(all_combinations) - 1:
                return None

            # create base config for indexing pipeline
            pipeline_config = all_combinations[runcount]

        elif self.search_method == 'progressive_search':

            # initialize pipeline config with middle values for each parameter
            pipeline_config = {}
            for key, value in self.searchable_params.items():
                middle_index = int((len(value) - 1)/2)
                pipeline_config.update({key:value[middle_index]})
            
            # make a list of variations to try
            keys_to_try = []
            values_to_try = []
            for key, values in self.searchable_params.items():
                if len(values) > 1: # only variations to try when more than one option
                    for option in values:
                        if not (key in pipeline_config and option == pipeline_config[key]): # should not be default option
                            keys_to_try.append(key)
                            values_to_try.append(option)
            variations_to_try = [{keys_to_try[i]:values_to_try[i]} for i in range(len(keys_to_try))]

            # if there are no variations to try, just schedule one run
            if len(variations_to_try) == 0:
                variations_to_try = list(self.searchable_params.items())[0]

            # when all variations have been tried, stop searching
            if runcount > len(keys_to_try) - 1:
                return None

            # update with best performing params
            results_ext = pd.DataFrame(self.results)
            results_ext['parameter_tested'] = keys_to_try[:runcount]

            print(results_ext)

            best_config = {}
            if len(results_ext) > 0:
                best_config = results_ext.groupby('parameter_tested', sort=False).idxmax()[self.target_metric].to_list()
                best_config = {keys_to_try[vtw]:values_to_try[vtw] for vtw in best_config}
                logging.info(f'Best configuration so far: {best_config}')

            # print(pipeline_config)
            pipeline_config.update(best_config)
            logging.info(f'Trying: {variations_to_try[runcount]}')
            pipeline_config.update(variations_to_try[runcount])

        else:
            raise ValueError('Please provide a valid search method')

        # filter out indexing & evaluation parameters
        indexing_config = {key: pipeline_config[key] for key in {**self.searchable_index_params, **self.searchable_shared_params}}
        evaluation_config = {key: pipeline_config[key] for key in {**self.searchable_eval_params, **self.searchable_shared_params}}

        # More shared parameters
        indexing_config['weaviate_class'] = evaluation_config['weaviate_class'] = f'Run{runcount}'
        indexing_config['embed_model_provider'] = evaluation_config['embed_model_provider'] = indexing_config['embed_model'][0]
        indexing_config['embed_model'] = evaluation_config['embed_model'] = indexing_config['embed_model'][1]
        
        return indexing_config, evaluation_config

    def create_pipelines(self, indexing_config, evaluation_config):
        # create indexing pipeline

        indexing_pipeline = pipeline_index.create_pipeline(
            **self.shared_args,
            **self.index_args,
            **indexing_config
        )

        # create evaluation pipeline
        evaluation_pipeline = pipeline_eval.create_pipeline(
            **self.shared_args,
            **self.eval_args,
            **evaluation_config
        )

        print(f'RUN # {indexing_config["weaviate_class"]}')
        print('{**self.shared_args, **self.index_args, **indexing_config}')
        print({**self.shared_args, **self.index_args, **indexing_config})
        print('{**self.shared_args, **self.eval_args, **evaluation_config}')
        print({**self.shared_args, **self.eval_args, **evaluation_config})
                    
        return indexing_pipeline, evaluation_pipeline

    def run_indexing_pipeline(self, runcount, indexing_config, indexing_pipeline):
        logging.info(f'Starting indexing pipeline of run #{runcount} with {indexing_config}')       
        self.runner.run(indexing_pipeline)

    def run_evaluation_pipeline(self, runcount, evaluation_config, evaluation_pipeline):
        logging.info(f'Starting evaluation pipeline of run #{runcount} with {evaluation_config}')
        self.runner.run(input=evaluation_pipeline, extra_volumes=self.extra_volumes)