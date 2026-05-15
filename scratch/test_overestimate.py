
import sys
import os
from dataclasses import dataclass

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from fusion_engine import ProbabilisticFusionEngine, SensorReading, ClimateZone
from confidence_calibration import ConfidenceCalibrator

@dataclass
class MockHealthySeverity:
    severity_level: str = "healthy_appearing"
    realism_score: float = 0.20
    lesion_area_pct: float = 0.5
    discoloration_score: float = 0.1
    symptom_density: float = 0.05
    texture_abnormality: float = 0.1
    edge_damage_score: float = 0.05
    
    def get(self, key, default=0.0):
        return getattr(self, key, default)

def test_visual_overestimate():
    print("Testing Visual Overestimate Logic (CNN High, Visual Healthy)...")
    
    engine = ProbabilisticFusionEngine()
    calibrator = ConfidenceCalibrator()
    
    label = "Tomato_Late_blight"
    cnn_conf = 88.0
    full_probs = [0.05, 0.88, 0.07]
    
    # Healthy-appearing leaf
    severity = MockHealthySeverity()
    
    calibration = calibrator.calibrate(
        predicted_label=label,
        raw_confidence=cnn_conf,
        full_probabilities=full_probs,
        image_quality=0.90,
        severity_report=severity
    )
    
    sensor = SensorReading(temperature=25.0, humidity=50.0, soil_moisture=40.0)
    
    result = engine.fuse(
        label=label,
        cnn_conf=cnn_conf,
        sensor=sensor,
        quality_report={"aggregate_quality": 0.90, "leaf_visibility": 0.90},
        calibration=calibration,
        severity=severity
    )
    
    print(f"Final Confidence: {result['final_confidence']}%")
    print(f"Reliability: {result['reliability']}")
    
    assert result['reliability'] == "Visual Overestimate (Leaf Appears Healthy)"
    print("SUCCESS: Visual Overestimate verified.")

if __name__ == "__main__":
    test_visual_overestimate()
