from data.base_dataset import BaseDataset
import logging
import json
from sklearn.model_selection import train_test_split

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from collections import defaultdict
from pathlib import Path
import os

def generate_splits(data_path, save_path, name="mnist_splits.json", val_size=5000):
    save_path = Path(save_path) / name
    inr_path = Path(data_path)
    data_split = defaultdict(lambda: defaultdict(list))
    for p in list(inr_path.glob("mnist_png_*/**/*.pth")):

        if "training" in p.as_posix():
            s = "train"
        else:
            s = "test"
        logging.info

        data_split[s]["path"].append((os.getcwd() / p).as_posix())
        data_split[s]["label"].append(p.parent.parent.stem.split("_")[-2])

    # val split
    val_size = val_size
    train_indices, val_indices = train_test_split(
        range(len(data_split["train"]["path"])), test_size=val_size
    )
    data_split["val"]["path"] = [data_split["train"]["path"][v] for v in val_indices]
    data_split["val"]["label"] = [data_split["train"]["label"][v] for v in val_indices]

    data_split["train"]["path"] = [
        data_split["train"]["path"][v] for v in train_indices
    ]
    data_split["train"]["label"] = [
        data_split["train"]["label"][v] for v in train_indices
    ]

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, test size: {len(data_split['test']['path'])}"
    )

    print(save_path)
    with open(save_path, "w") as file:
        json.dump(data_split, file)


# Taken from neural-field-arena 
class DWSNetsDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        start_idx: Union[int, float] = 0.0,
        end_idx: Union[int, float] = 1.0,
        data_prefix: str = "",
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
        download_url: str =  None,
        force_download: bool =  False,
    ):
        """Initialize theDataset object.

        Args:
            path (Union[str, Path]): The path to the directory containing the dataset files.
            start_idx (Union[int, float], optional): The starting index or fraction of the dataset to use. Defaults to 0.0.
            end_idx (Union[int, float], optional): The ending index or fraction of the dataset to use. Defaults to 1.0.
            data_prefix (str, optional): The prefix of the dataset files. Defaults to "".
            data_keys (List[str], optional): The list of keys to load from the dataset files. Defaults to None, which loads all keys.
            transform (Optional[Union[Callable, Dict[str, Callable]]], optional): The transformation function to apply to the loaded data.
                Defaults to None.
        """

        super().__init__(path, download_url = download_url, force_download = force_download)

        if isinstance(path, str):
            path = Path(path)
        

        assert path.exists(), f"Path {path.absolute()} does not exist"
        assert path.is_dir(), f"Path {path.absolute()} is not a directory"
        """
        file_pattern = f"{data_prefix}*.pth"
        paths = list(path.glob(f"{data_prefix}*.pth"))

        if len(paths) == 0:
            raise ValueError(
                f"No files found at `{path.absolute()}` with pattern `{file_pattern}`"
            )
        """



        generate_splits(path, path, name="mnist_splits.json")

    """
    def __len__(self):
        return self.data[self.data_keys[0]].shape[0]

    def __getitem__(self, idx):
        batch_item = {key: self.data[key][idx] for key in self.data_keys}
        if self.transform is not None:
            batch_item, self.rng = self.transform(batch_item, self.rng)
        return batch_item
    """
    
