# DermAnnotate

**ML-Assisted Medical Image Annotation Platform for Dermatology**

DermAnnotate is a full-stack annotation tool that combines a fine-tuned deep learning classifier with an expert annotation interface to accelerate the labeling of dermatoscopy images. Upload a skin lesion image, get instant ML predictions with confidence scores, review or override the diagnosis, draw a segmentation polygon over the lesion, and export the full annotated dataset in COCO JSON format.

Built with **FastAPI + SQLite** on the backend and **Next.js** on the frontend. The classifier is a fine-tuned **Swin Transformer** trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) and can detect seven skin conditions.

---

## Features

- **Image upload** — drag-and-drop or click to upload JPEG/PNG dermatoscopy images
- **ML classification** — top-3 predictions with confidence scores from a Swin-Tiny model fine-tuned on HAM10000
- **Accept predictions** — one-click to copy a predicted diagnosis into the annotation form
- **Segmentation canvas** — draw polygon boundaries over lesions directly on the image
- **Clinical notes** — free-text notes per annotation
- **Status tracking** — images progress through `pending → predicted → annotated → reviewed`
- **Dataset export** — download all annotations as a COCO-format JSON file
- **Delete images** — remove images and all associated annotations

---

## Detected Conditions

| Label                | Description                     |
| -------------------- | ------------------------------- |
| Melanocytic Nevi     | Common benign mole              |
| Melanoma             | Malignant skin cancer           |
| Benign Keratosis     | Seborrheic / actinic keratosis  |
| Basal Cell Carcinoma | Most common skin cancer         |
| Actinic Keratosis    | UV-induced pre-cancerous lesion |
| Vascular Lesion      | Blood vessel abnormality        |
| Dermatofibroma       | Benign fibrous nodule           |

---

## Architecture

```
┌─────────────────┐        ┌──────────────────────┐        ┌──────────────────┐
│   Next.js UI    │ ──────▶│   FastAPI Backend     │ ──────▶│  ML Classifier   │
│   Port 3000     │ ◀───── │   Port 8000           │ ◀───── │  Swin-Tiny       │
└─────────────────┘        └──────────────────────┘        └──────────────────┘
                                      │
                               ┌──────┴──────┐
                               │   SQLite    │
                               └─────────────┘
```

---

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- (Optional) Apple Silicon Mac or CUDA GPU for faster ML inference

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AmitHemantJadhav/Dermannote.git
cd Dermannotate
```

### 2. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Copy the example env file
cp .env.example .env
```

The default `.env` has `USE_MOCK=true`, which runs a mock classifier that returns random predictions — no model download needed to get the app running.

#### To use the real ML classifier

1. Install the ML dependencies (they are commented out in `requirements.txt` to keep the default install light):

```bash
pip install "transformers>=4.38.0" "torch>=2.1.0"
```

2. Set `USE_MOCK=false` in your `.env` file.

3. Download the base model weights (~350 MB):

```bash
python scripts/download_models.py
```

4. _(Optional)_ Train a class-balanced classifier on HAM10000 for best accuracy (~94% top-1):

```bash
# Fine-tune the classifier (downloads ~1.7 GB training data on first run)
# Uses MPS on Apple Silicon, CUDA if available, otherwise CPU
python scripts/train_classifier.py

# Point the app at your trained model
# Add this line to your .env:
# SKIN_MODEL_PATH=./models/skin_classifier
```

#### Start the backend server

```bash
uvicorn app.main:app --reload --port 8000
```

---

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Directory Structure

```
Dermannotate/
├── backend/
│   ├── app/                    # FastAPI application (routes, models, DB)
│   ├── ml/                     # ML classifier (Swin-Tiny wrapper)
│   ├── scripts/
│   │   ├── download_models.py       # Downloads HuggingFace model weights
│   │   ├── train_classifier.py      # Fine-tunes classifier on HAM10000
│   │   ├── download_ham10000_test.py  # Downloads HAM10000 test images
│   │   └── evaluate.py              # Evaluates classifier accuracy
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── components/         # React UI components
│       └── lib/api.ts          # API client
└── README.md
```

---

## Running an Evaluation

After training, measure accuracy against the HAM10000 test set:

```bash
cd backend

# Download test images (25 per class, 175 total)
python scripts/download_ham10000_test.py

# Run evaluation
python scripts/evaluate.py --images dataset/ham10000_test --out results/eval.csv
```

Sample results after fine-tuning:

| Metric         | Score |
| -------------- | ----- |
| Top-1 accuracy | 94.3% |
| Top-3 accuracy | 100%  |

---

## Team

| Name             | Role                              |
| ---------------- | --------------------------------- |
| Amit Jadhav      | ML Pipeline & Model Integration   |
| Shaurya Beriwala | ML Pipeline & System Architecture |
| Shruti S         | Frontend & Annotation UI          |
| Sudarshan K      | Frontend & Data Visualization     |
| Arnav M          | Backend API & Export Logic        |

**Sponsor:** Dr. Hajiarbabi — Purdue University Fort Wayne
