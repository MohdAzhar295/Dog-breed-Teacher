# Dog Breed Recognition + Breed Profile

Web app that identifies **30 common dog breeds** from an uploaded or camera-captured image and returns breed characteristics, temperament, environment needs, and food priorities.

## Features

- Image upload and browser camera capture
- 30-breed classifier trained with transfer learning on Stanford Dogs
- Offline breed profile database (`data/breed_profiles.json`) with origin and regional presence
- Optional live enrichment via [TheDogAPI](https://thedogapi.com/) when `THEDOGAPI_KEY` is set
- Confidence threshold with low-confidence handling

## Project structure

```
dog-breed-recognizer/
  app/                  FastAPI application
  data/                 Breed config and profile data
  models/               Trained checkpoint and reports
  scripts/              Dataset download, prep, train, evaluate
  static/               Web UI
```

## Setup

```bash
cd dog-breed-recognizer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

1. Download Stanford Dogs:

```bash
python scripts/download_dataset.py
```

2. Prepare the 30-breed subset:

```bash
python scripts/prepare_dataset.py
```

3. Train:

```bash
python scripts/train.py --epochs 8
```

4. Evaluate:

```bash
python scripts/evaluate.py
```

## Run the web app

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` (or another port if `8000` is already in use).

## Model performance

After training on the prepared 30-breed subset:

- Validation accuracy: **83.3%**
- Test accuracy: **83.9%**

Optional API enrichment:

```bash
set THEDOGAPI_KEY=your_key_here
uvicorn app.main:app --reload
```

## Dataset

- **Images**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
- **Breed traits**: Curated offline profiles with AKC references; optional enrichment from TheDogAPI

## Disclaimer

Temperament and behavior information reflects **typical breed traits**, not the mood or personality of the individual dog in the photo.
