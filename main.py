from _post_check import check_result
from inference import get_inference
from model import getModel

if __name__ == "__main__":
    model = getModel()
    get_inference(model)
    check_result()
