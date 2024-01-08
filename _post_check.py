import os

import pandas as pd


def check_result():
    save_path = os.getenv("ENV_RESULT")
    no = os.getenv("ENV_NO")
    num_examples = int(os.getenv("ENV_NUM_EXAMPLES", 200))

    result_path = os.path.join(save_path, no + "-text.csv")
    assert os.path.exists(result_path)

    df_result = pd.read_csv(result_path)
    assert len(df_result) == num_examples


if __name__ == "__main__":
    check_result()
