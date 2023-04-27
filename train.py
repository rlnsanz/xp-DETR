import albumentations
import numpy as np
import os
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils import data as torchdata

from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import flor
from flor import MTK as Flor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cppe5 = load_dataset("cppe-5")
assert isinstance(cppe5, DatasetDict)
categories = cppe5["train"].features["objects"].feature["category"].names  # type: ignore
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

remove_idx = [590, 821, 822, 875, 876, 878, 879]
keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
cppe5["train"] = cppe5["train"].select(keep)

model_name = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to(device)
Flor.checkpoints(model)

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item["labels"].to(device) for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"].to(device)
    batch["pixel_mask"] = encoding["pixel_mask"].to(device)
    batch["labels"] = labels
    return batch


batch_size = 8
num_epochs = 10
learning_rate = 1e-5
weight_decay = 1e-4

train_loader = torchdata.DataLoader(
    dataset=cppe5["train"].with_transform(transform_aug_ann),  # type: ignore
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
Flor.checkpoints(optimizer)

total_step = len(train_loader)
for epoch in Flor.loop(range(num_epochs)):
    model.train()
    for i, batch in Flor.loop(enumerate(train_loader)):
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i, total_step, flor.log("loss", loss.item())
            )
        )
        if i > 9:
            break

print("Model TEST")
