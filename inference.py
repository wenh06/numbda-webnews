import csv
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

dataset = os.getenv("ENV_DATASET")  # 基础数据集路径
c_dataset = os.getenv("ENV_CHILDDATASET")  # 子数据集名称
save_path = os.getenv("ENV_RESULT")  # 中间结果存储路径
no = os.getenv("ENV_NO")  # 结果文件的No.
assert "default" not in [dataset, c_dataset, save_path, no]

num_examples = int(os.getenv("ENV_NUM_EXAMPLES", 200))  # 评测样本数
model_batch_size = int(os.getenv("ENV_MODEL_BATCH_SIZE", 32))  # 模型评测时的batch_size

print(f"dataset: {dataset}")
print(f"sub-dataset: {c_dataset}")
print(f"result file path: {save_path}")
print(f"result No.: {no}")
print(f"eval sample number: {num_examples}")
print(f"eval batch_size: {model_batch_size}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BUILTIN_DATASETS = {
    "amazon": "AmazonReviewsZH",
    "dianping": "DianPingTiny",
    "imdb": "IMDBReviewsTiny",
    "jd_binary": "JDBinaryTiny",
    "jd_full": "JDFullTiny",
    "sst": "SST",
    "ifeng": "Ifeng",
    "chinanews": "Chinanews",
}


df_dataset = pd.read_csv(dataset)  # columns: "text", "label", "dataset", "adv_text", "adv_output"
df_dataset = df_dataset[df_dataset["dataset"] == BUILTIN_DATASETS[c_dataset.lower()]].reset_index(drop=True)
df_dataset = df_dataset.sample(n=min(int(num_examples), len(df_dataset)), random_state=42)


@torch.no_grad()
def get_inference(model):
    if hasattr(model, "model"):
        model.model.eval()
        model.model.to(device)
    else:
        model.eval()
        model.to(device)

    adv_inputs = df_dataset["adv_text"].tolist()
    inputs = df_dataset["text"].tolist()

    scores, adv_scores = [], []
    pbar = tqdm(
        total=len(inputs) + len(adv_inputs),
        dynamic_ncols=True,
        mininterval=1.0,
        desc="Text Static Attack",
    )
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + model_batch_size]
        batch_preds = model(batch)

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        if isinstance(batch_preds, list):
            scores.extend(batch_preds)
        elif isinstance(batch_preds, np.ndarray):
            scores.append(torch.tensor(batch_preds))
        else:
            scores.append(batch_preds)
        i += model_batch_size

        pbar.update(len(batch))

    i = 0
    while i < len(adv_inputs):
        batch = adv_inputs[i : i + model_batch_size]
        batch_preds = model(batch)

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        if isinstance(batch_preds, list):
            adv_scores.extend(batch_preds)
        elif isinstance(batch_preds, np.ndarray):
            adv_scores.append(torch.tensor(batch_preds))
        else:
            adv_scores.append(batch_preds)
        i += model_batch_size

        pbar.update(len(batch))

    pbar.close()

    if isinstance(scores[0], torch.Tensor):
        scores = torch.cat(scores, dim=0)

    # apply softmax to scores if they are not probabilities
    if (scores.max() > 1.0) or (scores.min() < 0.0):
        scores = torch.nn.functional.softmax(scores, dim=1)

    assert len(inputs) == len(scores), f"Got {len(scores)} outputs for {len(inputs)} inputs"

    predictions = scores.argmax(dim=1)
    scores = scores.max(dim=1).values

    scores = scores.cpu().numpy().tolist()
    predictions = predictions.cpu().numpy().tolist()

    if isinstance(adv_scores[0], torch.Tensor):
        adv_scores = torch.cat(adv_scores, dim=0)

    # apply softmax to scores if they are not probabilities
    if (adv_scores.max() > 1.0) or (adv_scores.min() < 0.0):
        adv_scores = torch.nn.functional.softmax(adv_scores, dim=1)

    assert len(adv_inputs) == len(adv_scores), f"Got {len(adv_scores)} outputs for {len(adv_inputs)} inputs"

    adv_predictions = adv_scores.argmax(dim=1)
    adv_scores = adv_scores.max(dim=1).values

    adv_scores = adv_scores.cpu().numpy().tolist()
    adv_predictions = adv_predictions.cpu().numpy().tolist()

    df_results = pd.DataFrame(
        {
            "ground_truth_output": df_dataset["label"].tolist(),
            "num_queries": [0 for _ in range(len(inputs))],
            "original_output": predictions,
            "original_score": scores,
            "original_text": inputs,
            "perturbed_output": adv_predictions,
            "perturbed_score": adv_scores,
            "perturbed_text": adv_inputs,
        }
    )
    df_results["result_type"] = df_results.apply(
        lambda row: "Skipped"
        if row["ground_truth_output"] != row["original_output"]
        else "Failed"
        if row["perturbed_output"] == row["original_output"]
        else "Successful",
        axis=1,
    )

    # save results
    df_results.to_csv(
        os.path.join(save_path, no + "-text.csv"),
        quoting=csv.QUOTE_NONNUMERIC,
        index=False,
    )

    return df_results
