# NASOA

This repository is for our paper "**Faster Task-oriented Online Fine-tuning**".

## Efficient Training (ET-NAS) Model-Zoo 

| Model Name | MParam | GMac  | Top-1 | Inference Time  (ms) | Training Step Time (ms) |
| ---------- | ------ | ----- | ----- | -------------------- | ----------------------- |
| ET-NAS-A   | 2.6    | 0.23  | 62.06 | 5.30                 | 14.74                   |
| ET-NAS-B   | 3.9    | 0.39  | 66.92 | 5.92                 | 15.78                   |
| ET-NAS-C   | 7.1    | 0.58  | 71.29 | 8.94                 | 26.28                   |
| ET-NAS-D   | 15.2   | 1.55  | 74.46 | 14.54                | 36.30                   |
| ET-NAS-E   | 21.4   | 2.61  | 76.87 | 25.34                | 61.95                   |
| ET-NAS-F   | 28.4   | 2.31  | 78.80 | 33.83                | 93.04                   |
| ET-NAS-G   | 49.3   | 5.68  | 80.41 | 53.08                | 133.97                  |
| ET-NAS-H   | 44.0   | 5.33  | 80.92 | 76.80                | 193.40                  |
| ET-NAS-I   | 72.4   | 13.13 | 81.38 | 94.60                | 265.13                  |
| ET-NAS-J   | 103.0  | 18.16 | 82.08 | 131.92               | 370.28                  |
| ET-NAS-K   | 87.3   | 27.51 | 82.42 | 185.75               | 505.00                  |
| ET-NAS-L   | 130.4  | 23.46 | 82.65 | 191.89               | 542.52                  |



### How to use ET-NAS 

#### Use ET-NAS in your own project

```python
from etnas import ETNas, MODEL_MAPPINGS
model_name = "ET-NAS-A"

# construct a ET-NAS network
network = ETNas(MODEL_MAPPINGS[model_name])

# load pre-trained model weights
network.load_state_dict(torch.load("{}.pth".format(MODEL_MAPPINGS[model_name])))
```

All pre-trained model can be download from:

#### Example: Running test on Imagenet

```python
python test.py \
       --model-name ET-NAS-A \ 
       --data-dir imagenet \ # path to dataset, imagenet for example
       --model-zoo-dir model_zoo \ # path to pre-trained models, modelz_zoo for example
       --batch-size 256 \ # batch size on each card
       --num-workers 4 \ # number of workers
       --device cuda:0 # device to run the network, set "cuda" for gpus and "cpu" for cpu
```

