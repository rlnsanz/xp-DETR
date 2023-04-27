import numpy as np
import os
from PIL import Image, ImageDraw

from datasets import load_dataset, DatasetDict


cppe5 = load_dataset("cppe-5")
assert isinstance(cppe5, DatasetDict)
categories = cppe5["train"].features["objects"].feature["category"].names  # type: ignore
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

remove_idx = [590, 821, 822, 875, 876, 878, 879]
keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
cppe5["train"] = cppe5["train"].select(keep)
