from __future__ import annotations
import typer
from ultralytics import YOLO

app = typer.Typer(add_completion=False)

@app.command()
def main(
    weights: str = typer.Option(..., help="Path to .pt or exported model"),
    source: str = typer.Option("demo/samples", help="Image/video/webcam(0) path"),
    imgsz: int = typer.Option(992),
    conf: float = typer.Option(0.25),
    save: bool = typer.Option(True, help="Save visualizations under runs/detect/predict"),
):
    model = YOLO(weights)
    res = model.predict(source=source, imgsz=imgsz, conf=conf, save=save)
    print(res)

if __name__ == "__main__":
    app()