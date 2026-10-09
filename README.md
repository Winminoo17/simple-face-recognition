# CNN Face Recognition System

A local computer-vision system built with **TensorFlow/Keras** to classify face identities using deep Convolutional Neural Networks (CNNs). This repository handles data pipeline extraction, custom model architecture compilation, checkpoint validation tracking, and rapid localized image prediction.

---

## 📁 Project Directory Structure

```text
face-regonization/
├── models/
│   ├── best_cnn_model.keras       # Automatically saved optimal validation checkpoint
│   └── cnn_face_id_model.keras    # Finalized training state structural array weights
├── train/                         # Structured training class identity label images
│   ├── s01/
│   └── s02/
├── val/                           # Structured validation validation dataset splits
│   └── s01/
│       └── 11_jpg.rf.eb44e0f3d6c1b9b5e6ce087a0a5504b5.jpg
├── .gitignore                     # Git untracked parameters restriction configuration
├── face-regonization.ipynb       # Main executable notebook script application
└── README.md                      # Documentation breakdown reference guide
```

---

## 🛠️ Architectural Configuration Details

* **Image Input Dimensions:** `128x128 pixels` (RGB multi-channel normalization matrix mapping via scale adjustments).
* **Batch Sizing Hyperparameters:** `32 inputs per processing stack initialization`.
* **Deep Neural Feature Extraction:** 3 cascaded convolutional grouping blocks (`32 -> 64 -> 128` filtering pathways accompanied by max-pooling downsizings).
* **Regularization Parameters:** High-dropout normalization adjustments (`Dropout(0.6)`) mapped against deep multi-connected fully dense dense layers to prevent localized sample overfitting.

---

## 🚀 Execution & Setup Lifecycle

### 1. Environmental Core Requirements
Ensure your active environment contains the foundational Python configurations:
```bash
pip install tensorflow matplotlib numpy
```

### 2. Neural Execution and Model Training Loop
Open the workspace directory inside your text editor tool or runtime dashboard and launch `face-regonization.ipynb`. 

Running the workflow cells will automatically initialize:
1. Local directory loading via `image_dataset_from_directory`.
2. Model compilation monitoring validation thresholds.
3. Optimal parameter verification exporting directly to the `models/` caching folder.

### 3. Running Prediction Testing Pipelines
The prediction block inside the notebook points directly to your localized path array structures:
```python
input_image_path = "./val/s01/11_jpg.rf.eb44e0f3d6c1b9b5e6ce087a0a5504b5.jpg"
```
Execute the inference execution component inside the notebook workspace to test input matrix weights, calculate dynamic softmax metrics distributions, and plot visual confirmation frames with accuracy scales.

---

## 🔒 Local Storage Tracking Precautions

Large structural neural models cannot be uploaded to remote source servers due to tracking bounds limits. This repository applies automated isolation mechanics inside the **`.gitignore`** configuration profile:

```text
models/
*.keras
__pycache__/
.ipynb_checkpoints/
```
*Note: While the code tracks structure layouts cleanly across your local hard drives, remote repositories will only save source files and dataset index rules.*
