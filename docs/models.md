# Models

pihome-vision finds people and vehicles with an object detection model that you supply.
None is included in this repository, for two reasons: a model file is tens of
megabytes, and the models that work well here are published under licences of their
own, which this project cannot relicense and which you should read before running one.

## What it runs

Any ONNX file that OpenCV's DNN module can load and that has one of the two output
heads Ultralytics exports, with the 80 COCO classes:

| Head | Output shape | Exported by |
| --- | --- | --- |
| NMS-free | `(1, N, 6)`: corners, confidence, class | YOLO26 |
| Classic | `(1, 84, N)`: centre box and a score per class | YOLO11 and earlier |

Of the 80 classes, only `person` and the vehicles (bicycle, car, motorcycle, bus, truck)
are used. Everything else is dropped as soon as it is decoded.

The input size is fixed when a model is exported, and pihome-vision reads it from the
file. `model.input_size` in `vision.yaml` is needed only for a model exported to take
any size (`dynamic=True`), and is 640 if left out. Set for a model of one size, it must
be that size, and pihome-vision refuses to start if it is not.

The model runs on as many threads as there are CPUs, up to 8. More make a 640 model no
faster, and only keep more cores busy: on a 6-core, 12-thread desktop, 8 threads ran a
frame in 35 ms and 12 in 36, but 12 used a third more CPU time. `model.threads` sets
another number, fewer to leave room on a machine that does other work.

## Licences

The code in this repository is MIT. A model is not code from this repository, and its
own terms apply to you when you download and run it:

- **Ultralytics YOLO** (YOLO11, YOLO26): the weights and anything exported from them
  are [AGPL-3.0](https://www.ultralytics.com/license), or an enterprise licence from
  Ultralytics. Read what AGPL-3.0 asks of you before running one, particularly if you
  change it or offer it to others over a network.

Any other model with one of the heads above works the same way. Check its licence too.

## Exporting a YOLO model

Do this in an environment of its own, not in pihome-vision's: Ultralytics installs
PyTorch and its own copy of OpenCV, and neither is needed once the file exists.

```sh
python3 -m venv ~/yolo-export
# The CPU build of PyTorch, a fraction of the size of the default one.
~/yolo-export/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/yolo-export/bin/pip install ultralytics

cd "$(mktemp -d)"
~/yolo-export/bin/yolo export model=yolo26n.pt format=onnx imgsz=640 opset=12 simplify=True dynamic=False
```

`yolo26n.pt` is downloaded on first use. Copy the result to where `vision.yaml` points,
and record its hash so that any other file is refused:

```sh
install -D -m 644 yolo26n.onnx /path/to/pihome-vision/models/detector.onnx
sha256sum /path/to/pihome-vision/models/detector.onnx   # → model.sha256 in vision.yaml
```

## Choosing a size

The `n` (nano) models are the ones to start with: on a desktop CPU from a few years ago
a 640 input takes about 40 ms a frame. That is far more than one camera at five frames a
second needs. On a smaller machine, export at 480 or 416 instead. This is faster, and
it misses small or distant people sooner.

Try a model on a photo before pointing it at the camera:

```sh
pihome-vision detect photo.jpg
```

It prints how long the model took and what it found, with confidence and position.
