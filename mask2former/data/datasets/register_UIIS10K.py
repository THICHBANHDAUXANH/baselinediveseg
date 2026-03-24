import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json

# Dynamic data root: set DIVESEG_DATA_ROOT env var or defaults to data/UIIS10K/ relative to CWD
DATA_ROOT = os.environ.get("DIVESEG_DATA_ROOT", os.path.join(os.getcwd(), "data", "UIIS10K"))
DATA_ROOT = os.path.join(DATA_ROOT, "")  # ensure trailing slash

def custom_dataset_loader(json_file, image_root):
    """Load COCO-format UIIS10K dataset and remap category IDs to 0-indexed."""
    dataset_dicts = load_coco_json(json_file, image_root)
    for record in dataset_dicts:
        for ann in record.get("annotations", []):
            ann["category_id"] = ann["category_id"] - 1
    return dataset_dicts

def register_UIIS10K_instance_dataset():
    classes = [
        'fish',
        'reptiles',
        'arthropoda',
        'corals',
        'mollusk',
        'plants',
        'ruins',
        'garbage',
        'human',
        'robots',
    ]

    train_json     = DATA_ROOT + "annotations/multiclass_train.json"
    train_img_root = DATA_ROOT + "img"
    val_json       = DATA_ROOT + "annotations/multiclass_test.json"
    val_img_root   = DATA_ROOT + "img"

    DatasetCatalog.register(
        "UIIS10K_train",
        lambda: custom_dataset_loader(train_json, train_img_root)
    )
    MetadataCatalog.get("UIIS10K_train").set(
        json_file=train_json,
        image_root=train_img_root,
        evaluator_type="coco",
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id={i+1: i for i in range(len(classes))}
    )

    DatasetCatalog.register(
        "UIIS10K_val",
        lambda: custom_dataset_loader(val_json, val_img_root)
    )
    MetadataCatalog.get("UIIS10K_val").set(
        json_file=val_json,
        image_root=val_img_root,
        evaluator_type="coco",
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id={i+1: i for i in range(len(classes))}
    )

# Execute registration on import
register_UIIS10K_instance_dataset()
