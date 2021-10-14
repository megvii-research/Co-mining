import os.path as osp

from cvpods.configs.retinanet_config import RetinaNetConfig
from coco import COCOPartial  # noqa


_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        RESNETS=dict(DEPTH=101),
        RETINANET=dict(
            IOU_THRESHOLDS=[0.4, 0.5],
            IOU_LABELS=[0, -1, 1],
            NMS_THRESH_TEST=0.5,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SMOOTH_L1_LOSS_BETA=0.1,
            PSEUDO_SCORE_THRES=0.6,
            MATCHING_IOU_THRES=0.4
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_missing_50p",),
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


class CustomRetinaNetConfig(RetinaNetConfig):
    def __init__(self):
        super(CustomRetinaNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomRetinaNetConfig()
