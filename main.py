import sys

from _post_check import check_result
from inference import get_inference
from model import getModel

if __name__ == "__main__":
    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    else:
        model_path = None
    model = getModel(model_path)
    get_inference(model)
    check_result()
