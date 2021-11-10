import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig
from coco import COCOPartial  # noqa

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        FCOS=dict(
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            PSEUDO_SCORE_THRES=0.5,
            MATCHING_IOU_THRES=0.4
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train_easy",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
            LR_SCHEDULER=dict(
                MAX_ITER=90000,
                STEPS=(60000, 80000),
            ),
            OPTIMIZER=dict(
                BASE_LR=0.01,
            ),
            IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(800,), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
