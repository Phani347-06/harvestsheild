# HarvestShield

Starter scaffold for HarvestShield web app.

## Backend

Path: `backend/`

- `app.py`: Flask API for image detection and recommendations.
- `model_utils.py`: placeholder model loading and prediction utilities.
- `requirements.txt`: backend Python dependencies.

### Run backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The API will run at `http://localhost:5000`.

## Frontend

Path: `frontend/`

- `package.json`: React/Vite dependencies.
- `src/App.jsx`: main UI for photo upload, sensor values, location, and detection.
- `src/style.css`: basic styling.

### Run frontend

```bash
cd frontend
npm install
npm run dev
```

Open the displayed Vite URL, typically `http://localhost:5173`.

## Notes

- The backend currently uses a mock CNN placeholder. Replace `harvest_model.h5` and the placeholder logic in `backend/model_utils.py` with your real model.
- The frontend sends image, sensor and location data to the backend endpoint `/api/detect`.
- If needed, update `frontend/src/App.jsx` to point to a deployed backend URL instead of `http://localhost:5000`.
