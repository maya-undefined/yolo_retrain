# Yolo trained on OpenImages military craft

## Pipeline
```bash
# preliminaries
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download images via fiftyone of these classes
python scripts/openimages_to_yolo.py --out data --classes Tank Truck Car Aircraft Helicopter "Combat vehicle" "Military vehicle" "Military helicopter"

# train the new model with the new data
python src/train_dual.py --data milveh.yaml --model yolov8n.pt --img-min 640 --img-max 768 --epochs 300  --batch 32

# get the performance results and deposit them into runs/detect/val<next_highest_number>
python src/eval_export.py --run runs/detect/train_dual<highest_number>

# copy some pictures into demo/sample and the results will go into runs/detect/predict<next_highest_number>
python src/infer.py --weights runs/detect/train<highest_number>/weights/best.pt --source demo/sample


# gradio fancy web demo
python app.py --weights runs/detect/train_dual<highest_number>/weights/best.pt
```

## Preliminary EDA

I can see that the 'car' class is over-represented 

![labels and their counts](./img/labels.jpg)

I also don't really have enough of each military craft so I'll have to get more data from other data sets.
