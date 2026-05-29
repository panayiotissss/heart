# Disease Prediction API

Binary classifier that predicts the likelihood of disease from patient health indicators. Built with scikit-learn (Random Forest) and served via FastAPI.

## Stack

- **Model**: Random Forest — 93% accuracy, 0.98 ROC-AUC on held-out test set
- **API**: FastAPI
- **Frontend**: Streamlit *(coming soon)*

## Setup

```bash
pip install -r requirements.txt
```

Train the model (generates `model.pkl` and `metrics.json`):

```bash
python model.py
```

Start the API:

```bash
uvicorn main:app --reload
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/model/info` | Last training metrics |
| POST | `/predict` | Predict disease from patient features |
| POST | `/train` | Retrain the model *(requires `X-API-Key` header)* |

Interactive docs available at `http://localhost:8000/docs`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_API_KEY` | API key for the `/train` endpoint | `dev-key` |
