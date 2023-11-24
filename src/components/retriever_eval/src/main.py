import pandas as pd
from datasets import Dataset
from fondant.component import PandasTransformComponent
from langchain.llms import OpenAI
from ragas import evaluate
from ragas.llms import LangchainLLM
from ragas.metrics import context_precision, context_relevancy


class RetrieverEval(PandasTransformComponent):
    def __init__(self, *_, openai_key) -> None:
        """
        Args:
            openai_key: OpenAI key
        """
        llm = OpenAI(openai_api_key=openai_key)
        gpt_wrapper = LangchainLLM(llm=llm)
        context_precision.llm = gpt_wrapper
        context_relevancy.llm = gpt_wrapper

    @staticmethod
    def create_hf_ds(dataframe: pd.DataFrame):
        dataframe.rename(
            columns={"data": "question", "retrieved+chunks": "contexts"}, inplace=True
        )
        return Dataset.from_pandas(dataframe)

    @staticmethod
    def ragas_eval(dataset):
        result = evaluate(
            dataset=dataset, metrics=[context_precision, context_relevancy]
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

        # in case empty parquet files, create dummy dataframe
        else:
            empty = pd.DataFrame(
                {
                    "question": "",
                    "contexts": [" "],
                    "context+precision": 0.01,
                    "context+relevancy": 0.01,
                }
            )
            empty.index = empty.index.astype(str)
            empty.columns = pd.MultiIndex.from_product([["text"], empty.columns])
            return empty
