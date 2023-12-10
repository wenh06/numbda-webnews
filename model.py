import os
import re
import tarfile
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch_ecg.utils.misc import ReprMixin

__all__ = [
    "NLPVictimModel",
]


class NLPVictimModel(ReprMixin, ABC):
    """
    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    __name__ = "NLPVictimModel"

    @abstractmethod
    def __call__(self, text_input_list: Sequence[str], **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_grad(self, text_input: str) -> Any:
        """Get gradient of loss with respect to input tokens."""
        raise NotImplementedError

    def _tokenize(self, inputs: Sequence[str]) -> List[List[str]]:
        """Helper method for `tokenize`"""
        raise NotImplementedError

    def tokenize(self, inputs: Sequence[str], strip_prefix: bool = False) -> List[List[str]]:
        """Helper method that tokenizes input strings
        Args:
            inputs: list of input strings
            strip_prefix: If `True`, we strip auxiliary characters added to tokens as prefixes (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens: List of list of tokens as strings
        """
        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            strip_pattern = f"^({'|'.join(strip_chars)})"
            tokens = [[re.sub(strip_pattern, "", t) for t in x] for x in tokens]
        return tokens


class PyTorchNLPVictimModel(NLPVictimModel):
    """ """

    __name__ = "PyTorchNLPVictimModel"

    def __init__(self, model: nn.Module, tokenizer: Callable[[Sequence[str]], np.ndarray]) -> None:
        """ """
        self.model = model
        self.tokenizer = tokenizer

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def __call__(self, text_input_list: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer(text_input_list)
        ids = torch.tensor(ids).to(model_device)

        outputs = []
        i = 0
        while i < len(ids):
            batch = ids[i : i + batch_size]
            with torch.no_grad():
                batch_preds = self.model(batch)

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]
            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu().detach().numpy()
            outputs.append(batch_preds)
            i += batch_size
        outputs = np.concatenate(outputs, axis=0)

        return outputs

    def get_grad(
        self,
        text_input: str,
        loss_fn: Union[nn.Module, Callable] = nn.CrossEntropyLoss(),
    ) -> Dict[str, torch.Tensor]:
        """Get gradient of loss with respect to input text.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer([text_input])
        ids = torch.tensor(ids).to(model_device)

        predictions = self.model(ids)

        output = predictions.argmax(dim=1)
        loss = loss_fn(predictions, output)
        loss.backward()

        # grad w.r.t to word embeddings
        grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad}

        return output

    def _tokenize(self, inputs: Sequence[str]) -> List[List[str]]:
        """Helper method that for `tokenize`
        Args:
            inputs: list of input strings
        Returns:
            tokens: List of list of tokens as strings
        """
        tokens = [self.tokenizer.convert_ids_to_tokens(self.tokenizer(x)) for x in inputs]
        return tokens


class HuggingFaceNLPVictimModel(PyTorchNLPVictimModel):
    """ """

    __name__ = "HuggingFaceNLPVictimModel"

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    ) -> None:
        """ """
        self.model = model
        self.tokenizer = tokenizer
        self._pipeline = None

    def __call__(self, text_input_list: Sequence[str]) -> Union[List[str], torch.Tensor]:
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as positional arguments.)
        """
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input: str) -> Dict[str, torch.Tensor]:
        """Get gradient of loss with respect to input text.

        Args:
            text_input: input string
        Returns:
            Dict of ids, and gradient as numpy array.
        """
        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiated your model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs: Sequence[str]) -> List[List[str]]:
        """Helper method that for `tokenize`
        Args:
            inputs: list of input strings
        Returns:
            tokens: List of list of tokens as strings
        """
        return [self.tokenizer.convert_ids_to_tokens(self.tokenizer([x], truncation=True)["input_ids"][0]) for x in inputs]

    @property
    def id2label(self) -> dict:
        """ """
        return self.model.config.id2label

    @property
    def max_length(self) -> int:
        """ """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        return 512 if self.tokenizer.model_max_length == int(1e30) else self.tokenizer.model_max_length


def _unzip_file(path_to_zip_file: str, unzipped_folder_path: str) -> None:
    """Unzips a .zip file to folder path."""
    print(f"Extracting file {path_to_zip_file} to {unzipped_folder_path}.")
    ufp = Path(unzipped_folder_path)
    with zipfile.ZipFile(path_to_zip_file) as zip_ref:
        if os.path.dirname(zip_ref.namelist()[0]).startswith(ufp.name):
            ufp = ufp.parent
        zip_ref.extractall(str(ufp))


def _untar_file(path_to_tar_file: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """
    Decompress a .tar.xx file to folder path.

    Parameters
    ----------
    path_to_tar_file: str or Path,
        path to the .tar.xx file
    dst_dir: str or Path,
        path to the destination folder

    """
    print(f"Extracting file {path_to_tar_file} to {dst_dir}.")
    mode = Path(path_to_tar_file).suffix.replace(".", "r:").replace("tar", "")
    # print(f"mode: {mode}")
    with tarfile.open(str(path_to_tar_file), mode) as tar_ref:
        # tar_ref.extractall(str(dst_dir))
        # CVE-2007-4559 (related to  CVE-2001-1267):
        # directory traversal vulnerability in `extract` and `extractall` in `tarfile` module
        _safe_tar_extract(tar_ref, str(dst_dir))


def _is_within_directory(directory: Union[str, Path], target: Union[str, Path]) -> bool:
    """
    check if the target is within the directory

    Parameters
    ----------
    directory: str or Path,
        path to the directory
    target: str or Path,
        path to the target

    Returns
    -------
    bool,
        True if the target is within the directory, False otherwise.

    """
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def _safe_tar_extract(
    tar: tarfile.TarFile,
    dst_dir: Union[str, Path],
    members: Optional[Iterable[tarfile.TarInfo]] = None,
    *,
    numeric_owner: bool = False,
) -> None:
    """
    Extract members from a tarfile **safely** to a destination directory.

    Parameters
    ----------
    tar: tarfile.TarFile,
        the tarfile to extract from
    dst_dir: str or Path,
        the destination directory
    members: Iterable[tarfile.TarInfo], optional,
        the members to extract,
        if None, extract all members,
        if not None, must be a subset of the list returned by `tar.getmembers()`
    numeric_owner: bool, default False,
        if True, only the numbers for user/group names are used and not the names.

    """
    for member in members or tar.getmembers():
        member_path = os.path.join(dst_dir, member.name)
        if not _is_within_directory(dst_dir, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(dst_dir, members, numeric_owner=numeric_owner)


# DEFAULT_MODEL = "wenh06/numbda-webnews"
DEFAULT_MODEL = "http://218.245.5.12/NLP/models/numbda-webnews.zip"


def getModel(path=None):
    """
    path: directory or compressed file
    """
    print(f"Loading model from {path}...")
    if path is not None:
        path = Path(path).expanduser().resolve()
        if path.is_file():
            # assert is compressed file
            assert path.suffix in [".zip", ".tar.gz", ".tar.bz2", ".tar"]
            # extract to temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"Extracting {path} to {tmpdir}...")
                if path.suffix == ".zip":
                    _unzip_file(path, tmpdir)
                else:
                    # mode = path.suffix.replace(".", "r:").replace("tar", "")
                    _untar_file(path, tmpdir)
                if len(os.listdir(tmpdir)) == 1 and os.path.isdir(os.path.join(tmpdir, os.listdir(tmpdir)[0])):
                    tmpdir = os.path.join(tmpdir, os.listdir(tmpdir)[0])
                model = transformers.AutoModelForSequenceClassification.from_pretrained(tmpdir)
                tokenizer = transformers.AutoTokenizer.from_pretrained(tmpdir)
            return HuggingFaceNLPVictimModel(model, tokenizer)
        else:
            tmpdir = None
            # assert path.is_dir(), f"Path {path} is not a file or directory"

    model = transformers.AutoModelForSequenceClassification.from_pretrained(path or DEFAULT_MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path or DEFAULT_MODEL)

    return HuggingFaceNLPVictimModel(model, tokenizer)
