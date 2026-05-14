import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || `${window.location.protocol}//${window.location.hostname}:5001/api`;

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [temperature, setTemperature] = useState(32);
  const [soilMoisture, setSoilMoisture] = useState(45);
  const [location, setLocation] = useState({ latitude: null, longitude: null });
  const [error, setError] = useState('');

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const fetchLocation = () => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by this browser.');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocation({ latitude: pos.coords.latitude, longitude: pos.coords.longitude });
        setError('');
      },
      () => {
        setError('Unable to access location. Please allow location access.');
      }
    );
  };

  const handleSubmit = async () => {
    if (!imageFile) {
      setError('Please upload a photo first.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const data = new FormData();
      data.append('image', imageFile);
      data.append('sensors', JSON.stringify({ temperature, soilMoisture }));
      data.append('location', JSON.stringify(location));

      const response = await axios.post(`${API_URL}/detect`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setResult(response.data);
    } catch (err) {
      setError('Detection failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-shell">
      <header>
        <h1>HarvestShield</h1>
        <p>Smart crop disease detection with image, sensors, and location.</p>
      </header>

      <main>
        <section className="panel">
          <h2>Step 1: Capture a Photo</h2>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          {previewUrl && <img className="preview" src={previewUrl} alt="Preview" />}
        </section>

        <section className="panel">
          <h2>Step 2: Sensor Data</h2>
          <div className="input-row">
            <label>Temperature (°C)</label>
            <input
              type="range"
              min="10"
              max="50"
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
            />
            <span>{temperature}°C</span>
          </div>
          <div className="input-row">
            <label>Soil Moisture (%)</label>
            <input
              type="range"
              min="0"
              max="100"
              value={soilMoisture}
              onChange={(event) => setSoilMoisture(Number(event.target.value))}
            />
            <span>{soilMoisture}%</span>
          </div>
        </section>

        <section className="panel">
          <h2>Step 3: Fetch Location</h2>
          <button type="button" className="primary" onClick={fetchLocation}>
            Get Field Location
          </button>
          <div className="location-info">
            <strong>Latitude:</strong> {location.latitude ?? '---'}<br />
            <strong>Longitude:</strong> {location.longitude ?? '---'}
          </div>
        </section>

        <section className="panel action-panel">
          <h2>Step 4: Run Disease Detection</h2>
          <button type="button" className="primary" onClick={handleSubmit} disabled={loading}>
            {loading ? 'Detecting...' : 'Detect Disease'}
          </button>
          {error && <div className="error-box">{error}</div>}
        </section>

        {result && (
          <section className="panel result-panel">
            <h2>Detection Result</h2>
            <div className="result-grid">
              <div><strong>Disease:</strong> {result.disease}</div>
              <div><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</div>
              <div><strong>Temperature:</strong> {result.sensors.temperature}°C</div>
              <div><strong>Moisture:</strong> {result.sensors.soilMoisture}%</div>
              <div><strong>Location:</strong> {result.location.latitude ?? 'N/A'}, {result.location.longitude ?? 'N/A'}</div>
            </div>
            <div className="recommendations">
              <h3>Recommended Actions</h3>
              <ul>
                {result.recommendations.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
