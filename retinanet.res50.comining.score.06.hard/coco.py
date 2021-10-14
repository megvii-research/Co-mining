#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii Inc. and its affiliates.

from PIL import Image
import torchvision.transforms as tfs

from cvpods.data.datasets import COCODataset
from cvpods.data.registry import DATASETS, PATH_ROUTES

_UPDATE_DICT = {
    "coco_2017_train_full":
        ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_train_easy":
        ("coco/train2017", "coco/annotations/instances_train2017_easy.json"),
    "coco_2017_train_hard":
        ("coco/train2017", "coco/annotations/instances_train2017_hard.json"),
    "coco_2017_train_extreme":
        ("coco/train2017", "coco/annotations/instances_train2017_extreme.json"),
    "coco_missing_50p":
        ("coco/train2017", "coco/annotations/coco_missing_50p.json"),
}


PATH_ROUTES.get("COCO")["coco"].update(_UPDATE_DICT)
PATH_ROUTES.get("COCO")["dataset_type"] = "COCOPartial"


@DATASETS.register()
class COCOPartial(COCODataset):
    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        dataset_dict = super().__getitem__(index)
        image = dataset_dict["image"]
        image = image.permute(1, 2, 0).numpy()

        s = 1
        color_jitter = tfs.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        color_jitter_img = tfs.RandomApply([color_jitter], p=0.8)(Image.fromarray(image))
        dataset_dict["image_color"] = tfs.ToTensor()(color_jitter_img) * 255

        return dataset_dict