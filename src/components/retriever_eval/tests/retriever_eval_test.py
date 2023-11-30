import pandas as pd
from main import RetrieverEval


def test_transform():
    input_dataframe = pd.DataFrame(
        {
            "data": [
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit?",
                "Sed massa massa, interdum a porttitor sit amet, semper eget nunc?",
            ],
            "retrieved+chunks": [
                [
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                        Quisque ut efficitur neque. Aenean mollis eleifend est, \
                        eu laoreet magna egestas quis. Cras id sagittis erat. \
                        Aliquam vel blandit arcu. Morbi ac nulla ullamcorper, \
                        rutrum neque nec, pellentesque diam. Nulla nec tempor \
                        enim. Suspendisse a volutpat leo, quis varius dolor.",
                    "Curabitur placerat ultrices mauris et lobortis. Maecenas \
                        laoreet tristique sagittis. Integer facilisis eleifend \
                        dolor, quis fringilla orci eleifend ac. Vestibulum nunc \
                        odio, tincidunt ut augue et, ornare vehicula sapien. Orci \
                        varius natoque penatibus et magnis dis parturient montes, \
                        nascetur ridiculus mus. Sed auctor felis lacus, rutrum \
                        tempus ligula viverra ac. Curabitur pharetra mauris et \
                        ornare pulvinar. Suspendisse a ultricies nisl. Mauris \
                        sit amet odio condimentum, venenatis orci vitae, \
                        tincidunt purus. Ut ullamcorper convallis ligula ac \
                        posuere. In efficitur enim ac lacus dignissim congue. \
                        Nam turpis augue, aliquam et velit sit amet, varius \
                        euismod ante. Duis volutpat nisl sit amet auctor tempus.\
                            Vivamus in eros ex.",
                ],
                [
                    "am leo massa, ultricies eu viverra ac, commodo non sapien. \
                        Mauris et mauris sollicitudin, ultricies ex ac, luctus \
                        nulla.",
                    "Cras tincidunt facilisis mi, ac eleifend justo lobortis ut. \
                        In lobortis cursus ante et faucibus. Vestibulum auctor \
                        felis at odio varius, ac vulputate leo dictum. \
                        Phasellus in augue ante. Aliquam aliquam mauris \
                        sed tellus egestas fermentum.",
                ],
            ],
        }
    )

    input_dataframe.columns = pd.MultiIndex.from_product(
        [["text"], input_dataframe.columns],
    )

    component = RetrieverEval(
        module="langchain.llms",
        llm_name="OpenAI",
        llm_kwargs={"openai_api_key": ""},
        metrics=["context_precision", "context_relevancy"],
    )

    output_dataframe = component.transform(input_dataframe)

    expected_output_dataframe = pd.DataFrame(
        {
            "question": [
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit?",
                "Sed massa massa, interdum a porttitor sit amet, semper eget nunc?",
            ],
            "contexts": [
                [
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                        Quisque ut efficitur neque. Aenean mollis eleifend est, \
                        eu laoreet magna egestas quis. Cras id sagittis erat. \
                        Aliquam vel blandit arcu. Morbi ac nulla ullamcorper, \
                        rutrum neque nec, pellentesque diam. Nulla nec tempor \
                        enim. Suspendisse a volutpat leo, quis varius dolor.",
                    "Curabitur placerat ultrices mauris et lobortis. Maecenas \
                        laoreet tristique sagittis. Integer facilisis eleifend \
                        dolor, quis fringilla orci eleifend ac. Vestibulum nunc \
                        odio, tincidunt ut augue et, ornare vehicula sapien. Orci \
                        varius natoque penatibus et magnis dis parturient montes, \
                        nascetur ridiculus mus. Sed auctor felis lacus, rutrum \
                        tempus ligula viverra ac. Curabitur pharetra mauris et \
                        ornare pulvinar. Suspendisse a ultricies nisl. Mauris \
                        sit amet odio condimentum, venenatis orci vitae, \
                        tincidunt purus. Ut ullamcorper convallis ligula ac \
                        posuere. In efficitur enim ac lacus dignissim congue. \
                        Nam turpis augue, aliquam et velit sit amet, varius \
                        euismod ante. Duis volutpat nisl sit amet auctor tempus.\
                            Vivamus in eros ex.",
                ],
                [
                    "am leo massa, ultricies eu viverra ac, commodo non sapien. \
                        Mauris et mauris sollicitudin, ultricies ex ac, luctus \
                        nulla.",
                    "Cras tincidunt facilisis mi, ac eleifend justo lobortis ut. \
                        In lobortis cursus ante et faucibus. Vestibulum auctor \
                        felis at odio varius, ac vulputate leo dictum. \
                        Phasellus in augue ante. Aliquam aliquam mauris \
                        sed tellus egestas fermentum.",
                ],
            ],
            "context+precision": 0.15,
            "context+relevancy": 0.35,
        }
    )

    expected_output_dataframe.columns = pd.MultiIndex.from_product(
        [["text"], expected_output_dataframe.columns],
    )

    # Check if columns are the same
    columns_equal = expected_output_dataframe.columns.equals(output_dataframe.columns)

    # Check if data types within each column match
    dtypes_match = expected_output_dataframe.dtypes.equals(output_dataframe.dtypes)

    # Check if both conditions are met
    assert columns_equal and dtypes_match