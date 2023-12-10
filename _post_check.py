import os

import pandas as pd


def check_result():
    save_path = os.getenv("ENV_RESULT")  # 中间结果存储路径
    no = os.getenv("ENV_NO")  # 结果文件的no
    num_examples = int(os.getenv("ENV_NUM_EXAMPLES", 200))  # 评测样本数

    result_path = os.path.join(save_path, no + "-text.csv")
    assert os.path.exists(result_path)

    df_result = pd.read_csv(result_path)
    assert len(df_result) == num_examples


if __name__ == "__main__":
    check_result()
