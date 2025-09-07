from __future__ import annotations
import typer
from pathlib import Path
from ultralytics import YOLO

app = typer.Typer(add_completion=False)

@app.command()
def main(
    run: str = typer.Option(..., help="Path to a Ultralytics run folder (e.g., runs/detect/train)"),
    split: str = typer.Option("val", help="val or test"),
    export: str | None = typer.Option(None, help="Optional export format: onnx, engine, torchscript, openvino"),
):
    weights = Path(run) / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    print("Evaluating...")
    model.val(split=split)

    if export:
        print(f"Exporting to {export}...")
        model.export(format=export, dynamic=True)

if __name__ == "__main__":
    app()