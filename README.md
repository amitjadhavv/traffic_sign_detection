# Real-Time Traffic Sign Recognition

**Bachelor's final-year project — B.E. Computer Science & Engineering, VTU (The Oxford College of Engineering, Bengaluru), 2021**

A convolutional neural network that classifies **27 types of traffic signs** from a live webcam feed and sends the recognised sign to a client application over a socket, as a prototype of a driver-assistance alert system.

![Python](https://img.shields.io/badge/Python-3-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-real--time-green)

## Results

| | |
|---|---|
| Dataset | 23,249 labelled sign images, 32×32 RGB, 27 classes |
| Split | 80 % train (18,599) / 20 % validation (4,650) |
| Model | 4-layer CNN + dense head, 608 k parameters |
| Validation accuracy | **99.7 %** (training stopped early once train > 99 % and val > 98.5 %) |

## How it works

```mermaid
flowchart LR
    A[Webcam frame<br/>640×480] -->|threaded capture| B[Sliding window<br/>40×40, 50 % overlap]
    B --> C[CNN classifier<br/>softmax over 27 classes]
    C -->|confidence > 0.99| D[Recognised sign]
    D -->|TCP socket| E[Client app]
```

| File | Purpose |
|------|---------|
| `Model_Training.ipynb` | Loads and resizes the images with OpenCV, one-hot encodes labels, trains the Keras CNN (four Conv2D layers, 16→32→64→64 filters, max-pooling, dropout, Dense 512, softmax), with Adam and an early-stopping callback. |
| `traffic.h5` | Trained model weights. |
| `camera.py` | Threaded webcam stream; scans each frame with a sliding window and reports a sign when the classifier is more than 99 % confident. |
| `server.py` | TCP socket server (`jpysocket`) that sends the recognised sign name to a Java client. |

## Run it

```bash
pip install -r requirements.txt
# camera.py loads "model2.h5": rename traffic.h5 or change the path in the script
python camera.py --resolution 640x480
```

`server.py` binds to a hard-coded LAN address; change `host` to your machine's IP before running it.

## Limitations and what I would do today

The sliding-window search is slow on a CPU and the random train/validation split can overstate real-world accuracy. A modern version would use a single-shot detector (e.g. YOLO) for localisation, evaluate on a separate test set such as GTSRB's official test split, and export to TensorFlow Lite for an embedded device (the TFLite path is already sketched in `camera.py`).
