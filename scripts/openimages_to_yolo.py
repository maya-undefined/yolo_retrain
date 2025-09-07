"""
Download a filtered Open Images V7 subset via FiftyOne and export to YOLO format.
This version avoids `max_samples` as a dict for compatibility with older FiftyOne
by loading each split separately.


Usage:
python scripts/openimages_to_yolo.py \
--classes Tank Truck Car SUV Aircraft Helicopter "Armored fighting vehicle" \
--train-samples 1200 --val-samples 300 --out data
"""
from __future__ import annotations
import argparse
import fiftyone as fo
import fiftyone.zoo as foz


def find_detections_field(dataset: fo.Dataset) -> str:
    schema = dataset.get_field_schema()
    # Prefer a field whose document_type is Detections
    for name, field in schema.items():
        if getattr(field, "document_type", None) is not None:
            if getattr(field.document_type, "__name__", "") == "Detections":
                return name
    # Common fallbacks
    for cand in ("ground_truth", "detections"):
        if cand in schema:
            return cand
    raise RuntimeError("Could not find a Detections label field in the dataset")


def load_split(split: str, classes, max_samples: int) -> fo.Dataset:
    print(f"Loading Open Images V7 split '{split}' with max_samples={max_samples}...")
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        shuffle=True,
        only_matching=True,
    )
    return ds


def export_view(view: fo.DatasetView, out_dir: str, split: str, classes):
    label_field = find_detections_field(view._dataset)
    print(f"Exporting split='{split}' using label_field='{label_field}'...")
    view.export(
        export_dir=out_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data", help="Export directory")
    ap.add_argument("--classes", nargs="+", required=True, help="Open Images class names")
    ap.add_argument("--train-samples", type=int, default=1500)
    ap.add_argument("--val-samples", type=int, default=300)
    args = ap.parse_args()

    classes = args.classes

    # Load each split separately for compatibility
    train_ds = load_split("train", classes, args.train_samples)
    val_ds   = load_split("validation", classes, args.val_samples)

    # Get views tagged by split (FiftyOne tags items with the split name)
    train_view = train_ds.view()
    val_view   = val_ds.view()

    export_view(train_view, args.out, "train", classes)
    export_view(val_view, args.out, "val", classes)

    print(f"written to {args.out}/images/{{train,val}} and {args.out}/labels/{{train,val}}")


if __name__ == "__main__":
    main()