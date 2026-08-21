# Manga Cleaner 🧹

An experimental full-stack project demonstrating how to integrate a deep learning segmentation model with a FastAPI backend and a modern web frontend to automatically detect and erase text inside manga speech bubbles while preserving the outer bubble borders and artwork. [demo](docs/assets/)

## 💡 Overview

This project serves as an end-to-end experiment combining computer vision, deep learning inference, and web technologies:
1. **AI Segmentation**: Uses a YOLOv8 instance segmentation model fine-tuned on manga speech bubbles to identify balloon locations and polygons.
2. **Text Isolation & Border Protection**: Applies adaptive thresholding and Connected Component Analysis (CCA) inside each bubble to isolate text strokes while keeping the bubble frame intact.
3. **Telea Inpainting**: Seamlessly restores the text background using OpenCV inpainting.
4. **FastAPI Backend**: Provides a fast, lightweight REST API for image processing.
5. **Web UI**: A clean dark-mode web application for drag-and-drop uploading and before/after comparisons.

---

## ✨ Features

- 🎯 **Accurate Balloon Detection**: Detects speech bubbles of varying shapes and orientations.
- 🛡️ **Frame & Border Protection**: Separates outer frame strokes from text so bubble outlines are never accidentally deleted.
- 🧹 **Complete Text Removal**: Uses contrast analysis and morphological closing to avoid leaving hollow or half-erased glyphs.
- 🎨 **Inpainting Reconstruction**: Inpaints text regions smoothly with matching surrounding background tones.
- 🖥️ **Interactive Web Interface**: Split-view and single-view modes with instant download options.

---

## 🛠️ Tech Stack

- **Backend / API**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Computer Vision & AI**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/)
- **Frontend**: Vanilla HTML5, CSS3 (Modern Dark Theme), JavaScript (Fetch API)
- **Package Management**: [uv](https://github.com/astral-sh/uv) / Pip

---

## 🚀 Getting Started

### Prerequisites

- Python `>= 3.11`
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 1. Clone the Repository

```bash
git clone https://github.com/Teemo4621/manga-cleaner.git
cd manga-cleaner
```

### 2. Install Dependencies

Using `uv`:
```bash
uv sync
```

Or using `pip`:
```bash
pip install -r requirements.txt
# Or: pip install fastapi uvicorn opencv-python numpy ultralytics python-multipart
```

### 3. Model Setup

Make sure your fine-tuned model weight file `best.pt` is placed in the root directory:
```
manga_cleaner/
├── best.pt              <-- YOLO segmentation model weights
├── server.py
├── static/
│   └── index.html
└── ...
```

### 4. Run the Application

Start the FastAPI server:
```bash
uv run server.py
# Or: python server.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```


## 🎖️ Credits & Acknowledgements

- **Speech Bubble Segmentation Model**: Fine-tuned model by [@huyvux3005](https://huggingface.co/huyvux3005) on the [Manga109](http://www.manga109.org/) dataset.
  - Model weights & details: [huyvux3005/manga109-segmentation-bubble on Hugging Face](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- **Dataset**: Manga109 dataset for research and academic manga analysis.
- **Frameworks**: Ultralytics YOLOv8 & FastAPI.