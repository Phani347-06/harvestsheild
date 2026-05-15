
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from reasoning_orchestrator import HierarchicalOrchestrator
from image_context import ImageContext
from dataclasses import dataclass

def test_response_structure():
    orchestrator = HierarchicalOrchestrator(model=None)
    
    @dataclass
    class MockSeverity:
        lesion_area_pct: float
        severity_level: str
        discoloration_score: float
        edge_damage_score: float
        symptom_density: float
        texture_abnormality: float
        severity_score: float
        realism_score: float

    class MockObj:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    fusion_result = {
        "prediction": "Potato Early Blight",
        "final_confidence": 0.85,
        "calibrated_confidence": 0.82,
        "env_score": 70.0, # 0-100
        "reliability_score": 0.9,
        "visual_severity": "moderate",
        "realism_score": 0.88,
        "disease_presence_confidence": 0.95,
        "disease_visibility_strength": 0.8,
        "strong_evidence_override": True
    }
    
    quality_report = {"blur_score": 0.1, "brightness_score": 0.8}
    
    severity_report = MockSeverity(
        lesion_area_pct=15.5, 
        severity_level="moderate",
        discoloration_score=0.4,
        edge_damage_score=0.2,
        symptom_density=0.3,
        texture_abnormality=0.1,
        severity_score=0.5,
        realism_score=0.8
    )
    
    calibration = MockObj(
        entropy=0.2,
        margin=0.6,
        disease_presence_confidence=0.95,
        calibration_factors={"cnn": 1.0}
    )
    
    sensor_reading = MockObj(
        temperature=25.5,
        humidity=60.0,
        soil_moisture=45.0,
        rainfall_mm=0.0
    )
    
    climate_context = {"zone": "tropical", "is_online": True}
    
    morphology_report = MockObj(
        method_name="contour_analysis",
        morphology_score=0.75,
        signatures_found=["irregular_lesions"]
    )
    
    consistency_report = MockObj(
        overall_status="Verified",
        logs=[],
        scene_context=MockObj(
            rain_detected=False,
            fog_detected=False,
            scene_wetness_score=0.1,
            blur_type="none"
        )
    )
    
    explanation = {
        "reliability": "High",
        "severity": "Moderate",
        "farmer_summary": "Test summary",
        "concise_explanation": "Test concise",
        "technical_chain": ["Step 1"],
        "recommendation": "Spray water",
        "recommendations": ["Spray water"],
        "visual_detection_confidence": "Strong",
        "environmental_support": "High",
        "image_reliability": "Good",
        "warnings": []
    }
    
    cnn_results = {"confidence_score": 88.0}
    
    response = orchestrator._build_final_response(
        fusion_result=fusion_result,
        quality_report=quality_report,
        severity_report=severity_report,
        calibration=calibration,
        sensor_reading=sensor_reading,
        climate_context=climate_context,
        morphology_report=morphology_report,
        consistency_report=consistency_report,
        explanation=explanation,
        cnn_results=cnn_results,
        start_time=0
    )
    
    import json
    print("\n--- RAW METRICS ---")
    print(json.dumps(response['raw_metrics'], indent=2))
    
    # Check for mapped fields
    sr = response['raw_metrics']['severity_report']
    if sr.get('discoloration') == 0.4 and sr.get('edge_damage') == 0.2:
        print("\n[PASS] Severity aliases mapped correctly.")
    else:
        print(f"\n[FAIL] Severity aliases missing or incorrect: {sr}")

if __name__ == "__main__":
    test_response_structure()
