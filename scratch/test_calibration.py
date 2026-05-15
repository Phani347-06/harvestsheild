
import sys
import os
import json
from dataclasses import dataclass

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from fusion_engine import ProbabilisticFusionEngine, SensorReading, ClimateZone
from confidence_calibration import ConfidenceCalibrator, CalibratedPrediction

@dataclass
class MockSeverity:
    severity_level: str = "severe"
    realism_score: float = 0.85
    lesion_area_pct: float = 20.0
    discoloration_score: float = 0.90
    symptom_density: float = 0.85
    texture_abnormality: float = 0.80
    edge_damage_score: float = 0.75
    
    def get(self, key, default=0.0):
        return getattr(self, key, default)

def test_skepticism_override_logic():
    print("Testing Fusion Logic Override (High Visual Evidence, Low Env)...")
    
    engine = ProbabilisticFusionEngine()
    
    # 1. Setup Inputs
    label = "Tomato_Late_blight"
    cnn_conf = 95.0
    
    # Weak Environment (Low humidity, high temp for Late Blight)
    sensor = SensorReading(
        temperature=38.0, 
        humidity=15.0, 
        soil_moisture=10.0, 
        climate_zone=ClimateZone.TEMPERATE
    )
    
    # Image Quality
    quality_report = {"aggregate_quality": 0.85, "leaf_visibility": 0.9}
    
    # Mocked Severe Disease
    severity = MockSeverity()
    
    # Calibration (High Disease Presence)
    calibrator = ConfidenceCalibrator()
    full_probs = [0.01, 0.95, 0.04]
    calibration = calibrator.calibrate(
        predicted_label=label,
        raw_confidence=cnn_conf,
        full_probabilities=full_probs,
        image_quality=0.85,
        severity_report=severity
    )
    
    # 2. Run Fusion
    result = engine.fuse(
        label=label,
        cnn_conf=cnn_conf,
        sensor=sensor,
        quality_report=quality_report,
        calibration=calibration,
        severity=severity
    )
    
    reasoning = result['reasoning']
    print(f"CNN Confidence: {result['cnn_confidence']}%")
    print(f"Calibrated Confidence: {result['calibrated_confidence']}%")
    print(f"Env Score: {result['env_score']}%")
    print(f"Final Confidence: {result['final_confidence']}%")
    print(f"Reliability: {result['reliability']}")
    print(f"Strong Evidence Override: {result['strong_evidence_override']}")
    print(f"Visual Severity: {result['visual_severity']}")
    print(f"Contradiction Level: {reasoning.contradiction_level}")
    print(f"Visual Strength: {reasoning.visual_strength}")
    print(f"Evidence Recovery Factor: {calibration.calibration_factors.get('visual_evidence_recovery', 0)}")
    
    # Assertions
    assert result['strong_evidence_override'] == True
    assert result['final_confidence'] >= 80.0
    assert result['reliability'] == "Visually Obvious (Env Divergence)"
    
    print("\nSUCCESS: Override logic and recovery verified.")

if __name__ == "__main__":
    test_skepticism_override_logic()
