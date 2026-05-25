import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import BLEManager from './utils/bleManager';

const API_URL = import.meta.env.VITE_API_URL || `${window.location.protocol}//${window.location.hostname}:5005/api`;

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [temperature, setTemperature] = useState(32);
  const [humidity, setHumidity] = useState(60);
  const [soilMoisture, setSoilMoisture] = useState(45);
  const [location, setLocation] = useState({ latitude: null, longitude: null });
  const [error, setError] = useState('');
  const [bleError, setBleError] = useState('');

  const [bleStatus, setBleStatus] = useState('disconnected');
  const [bleDebugPacket, setBleDebugPacket] = useState('');
  const [bleDebugTime, setBleDebugTime] = useState('');
  const [showDetails, setShowDetails] = useState(false);

  const bleManagerRef = useRef(null);

  useEffect(() => {
    bleManagerRef.current = new BLEManager(
      (data) => {
        setTemperature(data.temperature);
        setHumidity(data.humidity);
        setSoilMoisture(data.soilMoisture);
      },
      () => setBleStatus('disconnected')
    );

    bleManagerRef.current.onDebug = (rawText) => {
      setBleDebugPacket(rawText);
      setBleDebugTime(new Date().toLocaleTimeString());
    };

    return () => {
      if (bleManagerRef.current) {
        bleManagerRef.current.disconnect();
      }
    };
  }, []);

  const connectBLE = async () => {
    try {
      setBleStatus('connecting');
      setBleError('');
      await bleManagerRef.current.connect();
      setBleStatus('connected');
    } catch (err) {
      console.error(err);
      setBleStatus('disconnected');
      setBleError(err.message || 'BLE Connection failed. Check console for details.');
    }
  };

  const disconnectBLE = () => {
    if (bleManagerRef.current) {
      bleManagerRef.current.disconnect();
    }
    setBleStatus('disconnected');
  };

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
      data.append('sensors', JSON.stringify({ temperature, humidity, soilMoisture }));
      data.append('location', JSON.stringify(location));

      console.log("[DEBUG-FRONTEND] Submitting to AI Analysis...");
      console.log("[DEBUG-FRONTEND] Sensor Payload:", { temperature, humidity, soilMoisture });
      console.log("[DEBUG-FRONTEND] Location Payload:", location);

      const response = await axios.post(`${API_URL}/detect`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log("[DEBUG-FRONTEND] AI Response Received:", response.data);
      console.log("[DEBUG-FRONTEND] Received Sensor Metrics:", response.data.raw_metrics?.sensor_readings);
      setResult(response.data);
    } catch (err) {
      console.error('Detection Error:', err);
      const errorMsg = err.response?.data?.error || 'Detection failed. Please check backend connection and try again.';
      setError(errorMsg);
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
          <h2>Wireless Sensors (BLE)</h2>
          <div className="ble-controls">
            {bleStatus === 'disconnected' ? (
              <button type="button" className="primary" onClick={connectBLE}>
                Scan & Connect ESP32
              </button>
            ) : (
              <button type="button" className="secondary" onClick={disconnectBLE}>
                Disconnect ESP32
              </button>
            )}
            <span className={`status-badge ${bleStatus}`}>
              Status: {bleStatus}
            </span>
          </div>
          <p className="ble-note">Requires compatible browser (e.g., Chrome/Edge). Ensure device is named "PlantCare_ESP32".</p>
          {bleError && <div className="error-box">{bleError}</div>}

          {/* Debug UI */}
          <div className="debug-box" style={{ marginTop: '16px', padding: '12px', background: '#1e293b', color: '#38bdf8', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.85rem' }}>
            <div style={{ marginBottom: '8px', color: '#f8fafc' }}><strong>Temporary BLE Debug Panel</strong></div>
            <div><strong>State:</strong> {bleStatus}</div>
            <div><strong>Last Packet:</strong> {bleDebugPacket || 'None'}</div>
            <div><strong>Timestamp:</strong> {bleDebugTime || '---'}</div>
          </div>
        </section>

        <section className="panel">
          <h2>Step 2: Live Sensor Data</h2>
          <div className="input-row">
            <label>Temperature (°C)</label>
            <span>{temperature}°C</span>
          </div>
          <div className="input-row">
            <label>Humidity (%)</label>
            <span>{humidity}%</span>
          </div>
          <div className="input-row">
            <label>Soil Moisture</label>
            <span>{soilMoisture}</span>
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
            <div className="farmer-summary-banner" style={{
              background: result.banner_color === 'green' ? 'linear-gradient(135deg, #065f46 0%, #064e3b 100%)' :
                result.banner_color === 'amber' ? 'linear-gradient(135deg, #92400e 0%, #78350f 100%)' :
                  result.banner_color === 'orange' ? 'linear-gradient(135deg, #9a3412 0%, #7c2d12 100%)' :
                    'linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%)',
              padding: '20px',
              borderRadius: '12px',
              marginBottom: '24px',
              borderLeft: `8px solid ${result.banner_color === 'green' ? '#34d399' :
                  result.banner_color === 'amber' ? '#fbbf24' :
                    result.banner_color === 'orange' ? '#fb923c' : '#f87171'
                }`
            }}>
              <h2 style={{
                margin: 0,
                color: result.banner_color === 'green' ? '#34d399' :
                  result.banner_color === 'amber' ? '#fbbf24' :
                    result.banner_color === 'orange' ? '#fb923c' : '#f87171',
                fontSize: '0.9rem',
                textTransform: 'uppercase',
                letterSpacing: '1px'
              }}>
                Farmer Summary
              </h2>
              <p style={{ margin: '8px 0 0 0', color: '#f8fafc', fontSize: '1.25rem', fontWeight: '600' }}>
                {result.farmer_summary}
              </p>
            </div>

            <div className="result-grid">
              <div style={{ gridColumn: '1 / -1', marginBottom: '16px' }}>
                <h3 style={{ color: '#f8fafc', fontSize: '1.1rem', marginBottom: '8px' }}>Diagnosis Reasoning</h3>
                <p style={{ color: '#cbd5e1', lineHeight: '1.6', background: 'rgba(30, 41, 59, 0.5)', padding: '12px', borderRadius: '8px', border: '1px solid #334155' }}>
                  {result.concise_explanation}
                </p>
              </div>

              <div><strong>Diagnosis:</strong> {(result.prediction || result.diagnosis || 'Unknown').toString().replace(/_/g, ' ')}</div>
              <div><strong>Reliability:</strong> <span style={{
                color: (result.banner_color === 'green' ? '#34d399' :
                  result.banner_color === 'amber' ? '#fbbf24' :
                    result.banner_color === 'orange' ? '#fb923c' : '#f87171'),
                fontWeight: 'bold'
              }}>{result.reliability || 'Unknown'}</span></div>
              <div><strong>Final Confidence:</strong> {result.overall_confidence ?? result.confidence ?? 0}%</div>
              <div><strong>Image Quality:</strong> {result.image_reliability || 'Unknown'}</div>

              {/* Severity Badge */}
              <div style={{ gridColumn: '1 / -1', borderTop: '1px solid #334155', margin: '12px 0', paddingTop: '12px' }}>
                <small style={{ color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Disease Severity Assessment</small>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <strong>Visual Severity:</strong>
                <span style={{
                  padding: '4px 12px',
                  borderRadius: '12px',
                  fontSize: '0.8rem',
                  fontWeight: '700',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  background: result.visual_severity === 'severe' ? 'rgba(239, 68, 68, 0.2)' :
                    result.visual_severity === 'moderate' ? 'rgba(249, 115, 22, 0.2)' :
                      result.visual_severity === 'mild' ? 'rgba(234, 179, 8, 0.2)' :
                        'rgba(34, 197, 94, 0.2)',
                  color: result.visual_severity === 'severe' ? '#f87171' :
                    result.visual_severity === 'moderate' ? '#fb923c' :
                      result.visual_severity === 'mild' ? '#fbbf24' :
                        '#4ade80',
                  border: `1px solid ${result.visual_severity === 'severe' ? 'rgba(239, 68, 68, 0.3)' :
                      result.visual_severity === 'moderate' ? 'rgba(249, 115, 22, 0.3)' :
                        result.visual_severity === 'mild' ? 'rgba(234, 179, 8, 0.3)' :
                          'rgba(34, 197, 94, 0.3)'
                    }`
                }}>
                  {result.visual_severity === 'healthy_appearing' ? 'Healthy' : result.visual_severity}
                </span>
              </div>

              <div><strong>Realism Score:</strong> {result.realism_score != null ? `${(result.realism_score * 100).toFixed(0)}%` : 'N/A'}</div>

              {result.disease_presence_confidence != null && (
                <div><strong>Disease Presence:</strong> {result.disease_presence_confidence}%
                  {!result.disease_gate_passed && <span style={{ color: '#fbbf24', marginLeft: '8px', fontSize: '0.8rem' }}>⚠ Below threshold</span>}
                </div>
              )}

              {result.disease_visibility_strength != null && result.disease_visibility_strength > 0 && (
                <div><strong>Disease Visibility:</strong> {(result.disease_visibility_strength * 100).toFixed(0)}%</div>
              )}

              {result.strong_evidence_override && (
                <div style={{ gridColumn: '1 / -1', padding: '8px 12px', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: '8px', fontSize: '0.85rem', color: '#86efac' }}>
                  ✓ Strong visual evidence confirmed — confidence maintained despite environmental uncertainty
                </div>
              )}

              <div style={{ gridColumn: '1 / -1', borderTop: '1px solid #334155', margin: '12px 0', paddingTop: '12px' }}>
                <small style={{ color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Evidence Detail</small>
              </div>

              <div><strong>Visual Match:</strong> {result.visual_detection_confidence}</div>
              <div><strong>Env Support:</strong> {result.environmental_support}</div>
              <div><strong>Zone:</strong> {result.raw_metrics?.climate_context?.zone || 'N/A'}</div>
              <div><strong>Source:</strong> {result.raw_metrics?.climate_context?.weather_source || 'Offline'}</div>

              {result.warnings && result.warnings.length > 0 && (
                <div style={{ gridColumn: '1 / -1', marginTop: '16px', padding: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '8px' }}>
                  <small style={{ color: '#f87171', fontWeight: 'bold', display: 'block', marginBottom: '4px' }}>ATTENTION:</small>
                  <ul style={{ margin: 0, paddingLeft: '20px', color: '#fca5a5', fontSize: '0.9rem' }}>
                    {Array.isArray(result.warnings) && result.warnings.map((w, i) => <li key={i}>{w}</li>)}
                  </ul>
                </div>
              )}
            </div>
            <div className="recommendations" style={{ marginTop: '24px' }}>
              <h3 style={{ color: '#f8fafc', marginBottom: '12px' }}>Recommended Actions</h3>
              <div className="recommendation-text" style={{
                padding: '20px',
                background: '#ffffff',
                borderRadius: '12px',
                borderLeft: `6px solid ${result.visual_severity === 'severe' ? '#ef4444' :
                    result.visual_severity === 'moderate' ? '#f97316' :
                      result.visual_severity === 'mild' ? '#eab308' :
                        '#10b981'
                  }`,
                color: '#000000',
                fontWeight: '600',
                lineHeight: '1.6',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}>
                {result.recommendation}
              </div>
            </div>

            <div style={{ marginTop: '24px' }}>
              <button
                onClick={() => setShowDetails(!showDetails)}
                style={{
                  background: 'none',
                  border: '1px solid #334155',
                  color: '#38bdf8',
                  fontSize: '0.85rem',
                  padding: '10px 16px',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                {showDetails ? 'Hide' : 'Show'} Additional Technical Details
              </button>
            </div>

            {showDetails && (
              <div className="detailed-logic" style={{
                marginTop: '16px',
                padding: '20px',
                background: 'rgba(15, 23, 42, 0.8)',
                borderRadius: '12px',
                border: '1px solid #1e293b'
              }}>
                <div style={{ marginBottom: '20px' }}>
                  <h4 style={{ margin: '0 0 12px 0', color: '#94a3b8', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Machine Reasoning Chain</h4>
                  <ul style={{ margin: 0, paddingLeft: '20px', color: '#cbd5e1', fontSize: '0.85rem', lineHeight: '1.7' }}>
                    {Array.isArray(result.technical_chain) && result.technical_chain.map((line, i) => <li key={i}>{line}</li>)}
                  </ul>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', borderTop: '1px solid #1e293b', paddingTop: '20px' }}>
                  <div>
                    <h4 style={{ color: '#94a3b8', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Raw Sensor Data</h4>
                    {console.log("[DEBUG-FRONTEND] Rendering Sensor Data Section:", result.raw_metrics?.sensor_readings)}
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#94a3b8', fontSize: '0.8rem' }}>
                      <li>Temperature: {result.raw_metrics?.sensor_readings?.temperature !== null ? `${result.raw_metrics.sensor_readings.temperature}°C` : 'N/A'}</li>
                      <li>Humidity: {result.raw_metrics?.sensor_readings?.humidity !== null ? `${result.raw_metrics.sensor_readings.humidity}%` : 'N/A'}</li>
                      <li>Soil Moisture: {result.raw_metrics?.sensor_readings?.soil_moisture !== null ? `${result.raw_metrics.sensor_readings.soil_moisture}%` : 'N/A'}</li>
                      <li>Rainfall: {result.raw_metrics?.sensor_readings?.rainfall !== undefined ? `${result.raw_metrics.sensor_readings.rainfall} mm` : '0 mm'}</li>
                    </ul>
                  </div>
                  <div>
                    <h4 style={{ color: '#94a3b8', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Quality Diagnostics</h4>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#94a3b8', fontSize: '0.8rem' }}>
                      <li>Blur: {result.raw_metrics?.quality_report?.blur_score}</li>
                      <li>Brightness: {result.raw_metrics?.quality_report?.brightness_score}</li>
                      <li>Leaf Visibility: {result.raw_metrics?.quality_report?.leaf_visibility}</li>
                      <li>Texture: {result.raw_metrics?.quality_report?.texture_clarity}</li>
                    </ul>
                  </div>
                </div>

                {/* Severity Report */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', borderTop: '1px solid #1e293b', paddingTop: '20px', marginTop: '16px' }}>
                  <div>
                    <h4 style={{ color: '#94a3b8', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Severity Analysis</h4>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#94a3b8', fontSize: '0.8rem' }}>
                      <li>Lesion Area: {result.raw_metrics?.severity_report?.lesion_area_pct ?? 'N/A'}%</li>
                      <li>Discoloration: {result.raw_metrics?.severity_report?.discoloration ?? 'N/A'}</li>
                      <li>Edge Damage: {result.raw_metrics?.severity_report?.edge_damage ?? 'N/A'}</li>
                      <li>Symptom Density: {result.raw_metrics?.severity_report?.symptom_density ?? 'N/A'}</li>
                      <li>Realism: {result.raw_metrics?.severity_report?.realism_score ?? 'N/A'}</li>
                    </ul>
                  </div>
                  <div>
                    <h4 style={{ color: '#94a3b8', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Confidence Calibration</h4>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#94a3b8', fontSize: '0.8rem' }}>
                      <li>Raw CNN: {result.raw_metrics?.cnn_confidence}%</li>
                      <li>Calibrated: {result.raw_metrics?.calibrated_confidence ?? result.raw_metrics?.cnn_confidence}%</li>
                      <li>Entropy: {result.raw_metrics?.calibration_report?.entropy ?? 'N/A'}</li>
                      <li>Margin: {result.raw_metrics?.calibration_report?.margin ?? 'N/A'}</li>
                      <li>Disease Presence: {result.raw_metrics?.calibration_report?.disease_presence ?? 'N/A'}%</li>
                    </ul>
                  </div>
                </div>

                <div style={{ marginTop: '16px', borderTop: '1px solid #1e293b', paddingTop: '12px' }}>
                  <h4 style={{ color: '#94a3b8', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Fusion Context</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', color: '#cbd5e1', fontSize: '0.8rem' }}>
                    <div>Climate Zone: {result.raw_metrics?.climate_context?.zone}</div>
                    <div>Weather Source: {result.raw_metrics?.climate_context?.weather_source}</div>
                    <div>CNN Conf: {result.raw_metrics?.cnn_confidence}%</div>
                    <div>Env Match: {result.raw_metrics?.env_score}%</div>
                  </div>
                </div>

                {/* New Morphology & Scene Reasoning */}
                {result.technical_details && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', borderTop: '1px solid #1e293b', paddingTop: '20px', marginTop: '16px' }}>
                    <div>
                      <h4 style={{ color: '#38bdf8', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Morphology Verification</h4>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#94a3b8', fontSize: '0.8rem' }}>
                        <li>Method: {result.technical_details.morphology?.method}</li>
                        <li>Signature Match: {(result.technical_details.morphology?.match_score * 100).toFixed(0)}%</li>
                        <li>Signatures: {result.technical_details.morphology?.signatures?.join(', ') || 'None'}</li>
                      </ul>
                    </div>
                    <div>
                      <h4 style={{ color: '#38bdf8', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Scene Context</h4>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#94a3b8', fontSize: '0.8rem' }}>
                        <li>Rain Detected: {result.technical_details.scene?.rain ? 'Yes' : 'No'}</li>
                        <li>Fog/Haze: {result.technical_details.scene?.fog ? 'Yes' : 'No'}</li>
                        <li>Wetness Score: {(result.technical_details.scene?.wetness * 100).toFixed(0)}%</li>
                        <li>Blur Type: {result.technical_details.scene?.blur_type}</li>
                      </ul>
                    </div>
                  </div>
                )}

                {result.technical_details?.consistency && (
                  <div style={{ marginTop: '16px', borderTop: '1px solid #1e293b', paddingTop: '12px' }}>
                    <h4 style={{ color: '#10b981', margin: '0 0 8px 0', fontSize: '0.7rem', textTransform: 'uppercase' }}>Consistency Engine</h4>
                    <div style={{ color: '#cbd5e1', fontSize: '0.8rem' }}>
                      <div style={{ marginBottom: '4px' }}>Status: <strong>{result.technical_details.consistency.status}</strong></div>
                      <ul style={{ margin: 0, paddingLeft: '20px', color: '#94a3b8', fontSize: '0.75rem' }}>
                        {Array.isArray(result.technical_details?.consistency?.logs) && result.technical_details.consistency.logs.map((log, i) => <li key={i}>{log}</li>)}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        )}

      </main>


    </div>
  );
}

export default App;
