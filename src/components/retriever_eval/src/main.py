import pandas as pd
from datasets import Dataset
from fondant.component import PandasTransformComponent
from langchain.llms import OpenAI
from ragas import evaluate
from ragas.llms import LangchainLLM


class RetrieverEval(PandasTransformComponent):
    def __init__(self, *_, openai_key, metrics) -> None:
        """
        Args:
            openai_key: OpenAI key
            metrics: RAGAS metrics to compute
        """
        llm = OpenAI(openai_api_key=openai_key)
        self.gpt_wrapper = LangchainLLM(llm=llm)
        self.metric_functions = self.extract_metric_functions(metrics=metrics)
        self.set_llm(self.metric_functions)

    #import the metric functions selected
    @staticmethod
    def import_from(module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)
    
    def extract_metric_functions(self, metrics: list):
        functions = []
        for metric in metrics:
            functions.append(self.import_from("ragas.metrics", metric))
        return functions
    
    def set_llm(self, metric_functions: list):
        for metric_function in metric_functions:
            metric_function.llm = self.gpt_wrapper

    #evaluate the retriever
    @staticmethod
    def create_hf_ds(dataframe: pd.DataFrame):
        dataframe.rename(
            columns={"data": "question", "retrieved+chunks": "contexts"}, inplace=True
        )
        return Dataset.from_pandas(dataframe)

    def ragas_eval(self, dataset):
        result = evaluate(
            dataset=dataset, metrics=self.metric_functions
        )
        return result

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        hf_dataset = self.create_hf_ds(
            dataframe=dataframe["text"][["data", "retrieved+chunks"]]
        )
        if "id" in hf_dataset.column_names:
            hf_dataset = hf_dataset.remove_columns("id")
        print(hf_dataset)
        if hf_dataset.num_rows != 0:
            result = self.ragas_eval(dataset=hf_dataset)
            results_df = result.to_pandas()
            results_df.index = results_df.index.astype(str)
            # rename columns to avoid issues with following component
            results_df.columns = results_df.columns.str.replace("_", "+")
            # Set multi-index column for the expected subset and field
            results_df.columns = pd.MultiIndex.from_product(
                [["text"], results_df.columns],
            )

            return results_df