"""
HarvestShield — Hierarchical Reasoning Orchestrator
===================================================
The centralized controller for the Morphology-Aware Intelligence pipeline.
Coordinates the multi-stage reasoning flow from raw image to XAI.

Execution Flow:
  1. ImageContext (Cache initialization)
  2. Scene Analysis (Rain/Fog/Field detection)
  3. CNN Inference (Raw classification)
  4. Visual Severity (Morphology + Realism)
  5. Morphology Routing (Signature verification)
  6. Fusion Engine (Probabilistic integration)
  7. Consistency Check (Semantic validation)
  8. Explanation Engine (Farmer-friendly XAI)

Design:
  - Performant: Single-pass image analysis using cached context.
  - Fail-safe: Optional stages are isolated in try-except blocks.
  - Consistent: Returns same structure as old fusion for backward compatibility.
"""

import time
import traceback
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict

# Core utils
from image_context import ImageContext
from scene_context_analyzer import SceneContextAnalyzer
from model_utils import predict_image
from visual_severity_utils import VisualSeverityAnalyzer
from disease_morphology_router import DiseaseAnalyzerRouter
from fusion_engine import ProbabilisticFusionEngine, SensorReading, ClimateZone
from consistency_engine import ConsistencyEngine
from explanation_engine import ExplanationEngine
from image_quality_utils import analyze_image_quality
from climate_utils import get_climate_context
from confidence_calibration import ConfidenceCalibrator


class HierarchicalOrchestrator:
    """
    Orchestrates the full HarvestShield reasoning pipeline.
    
    This replaces the direct call to fusion_engine in app.py.
    """
    
    def __init__(self, model):
        self.model = model
        self.scene_analyzer = SceneContextAnalyzer()
        self.severity_analyzer = VisualSeverityAnalyzer()
        self.morphology_router = DiseaseAnalyzerRouter()
        self.calibrator = ConfidenceCalibrator()
        self.fusion_engine = ProbabilisticFusionEngine()
        self.consistency_engine = ConsistencyEngine()
        self.explanation_engine = ExplanationEngine()

    def process_request(self, 
                        image_path: str, 
                        sensor_data: Dict[str, Any], 
                        location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs the full reasoning pipeline on an image.
        """
        start_time = time.time()
        print(f"\n[DEBUG-ORCHESTRATOR] === Starting Pipeline === ")
        print(f"[DEBUG-ORCHESTRATOR] Input: {image_path}")
        print(f"[DEBUG-ORCHESTRATOR] Sensors: {sensor_data}")
        
        try:
            # ── Step 0: Context Initialization ──
            print("[DEBUG-ORCHESTRATOR] 1/8 Initializing ImageContext...")
            ctx = ImageContext.from_path(image_path)
            print("[DEBUG-ORCHESTRATOR] Context initialized.")
            
            # ── Step 1: Scene Analysis (Pre-CNN) ──
            print("[DEBUG-ORCHESTRATOR] 2/8 Running Scene Analysis...")
            scene_context = self.scene_analyzer.analyze(ctx)
            print(f"[DEBUG-ORCHESTRATOR] Scene Analysis Complete: Rain={scene_context.rain_detected}")
            
            # ── Step 2: CNN Inference ──
            print("[DEBUG-ORCHESTRATOR] 3/8 Running CNN Inference...")
            cnn_results = predict_image(self.model, ctx.resized_224)
            prediction = cnn_results['prediction']
            print(f"[DEBUG-ORCHESTRATOR] CNN Result: {prediction}")
            
            # ── Step 3: Visual Severity & Realism ──
            print("[DEBUG-ORCHESTRATOR] 4/8 Running Visual Severity Analysis...")
            severity_report = self.severity_analyzer.analyze_severity(image_ctx=ctx)
            print(f"[DEBUG-ORCHESTRATOR] Severity: {severity_report['severity_level']}")
            
            # ── Step 4: Morphology Routing ──
            print("[DEBUG-ORCHESTRATOR] 5/8 Running Morphology Routing...")
            morphology_report = self.morphology_router.route(prediction, ctx)
            print(f"[DEBUG-ORCHESTRATOR] Morphology: {morphology_report.method_name}")
            
            # ── Step 5: External Context ──
            print("[DEBUG-ORCHESTRATOR] 6/8 Running Climate Context & Sensor Mapping...")
            climate_context = get_climate_context(location.get('latitude'), location.get('longitude')) if location else {}
            
            zone_str = (climate_context.get('climate_zone') or climate_context.get('zone') or 'temperate').lower()
            if "tropical" in zone_str:
                zone_enum = ClimateZone.TROPICAL
            elif "monsoon" in zone_str:
                zone_enum = ClimateZone.MONSOON
            elif "humid" in zone_str:
                zone_enum = ClimateZone.HUMID
            elif "dry" in zone_str or "arid" in zone_str:
                zone_enum = ClimateZone.DRY
            else:
                zone_enum = ClimateZone.TEMPERATE

            sensor_reading = SensorReading(
                temperature=float(sensor_data.get('temperature', 25)),
                humidity=float(sensor_data.get('humidity', 50)),
                soil_moisture=float(sensor_data.get('soil_moisture', sensor_data.get('soil', 30))),
                rainfall_mm=float(sensor_data.get('rainfall', 0)),
                climate_zone=zone_enum
            )
            print(f"[DEBUG-ORCHESTRATOR] Sensors Mapped: {sensor_reading}")
            
            quality_report = analyze_image_quality(ctx)
            
            # Run Calibration
            print("[DEBUG-ORCHESTRATOR] 7/8 Running Confidence Calibration...")
            calibration = self.calibrator.calibrate(
                predicted_label=prediction,
                raw_confidence=cnn_results['confidence_score'],
                full_probabilities=cnn_results['full_probabilities'],
                image_quality=quality_report['aggregate_quality'],
                severity_report=severity_report
            )
            print(f"[DEBUG-ORCHESTRATOR] Calibration: Final Conf={calibration.calibrated_confidence:.2f}")
            
            # ── Step 6: Multi-Factor Fusion ──
            fusion_result = self.fusion_engine.fuse(
                label=prediction,
                cnn_conf=cnn_results['confidence_score'],
                sensor=sensor_reading,
                quality_report=quality_report,
                calibration=calibration,
                severity=severity_report
            )
            print(f"[DEBUG-ORCHESTRATOR] Fusion Result: {fusion_result.get('prediction')} (Final Conf={fusion_result.get('final_confidence')})")
            
            # Defensive injection for reliability_score
            if "reliability_score" not in fusion_result:
                self.logger.warning("Fusion result missing reliability_score. Injecting default.")
                fusion_result["reliability_score"] = fusion_result.get("final_confidence", 50.0) / 100.0
            
            # ── Step 7: Consistency Validation ──
            consistency_report = self.consistency_engine.validate(
                fusion_output=fusion_result, 
                scene_context=scene_context, 
                morphology_report=morphology_report,
                severity_report=severity_report
            )
            print(f"[DEBUG-ORCHESTRATOR] Consistency: Status={consistency_report.overall_status}")
            
            # Apply overrides from consistency engine
            if consistency_report.reliability_override >= 0:
                fusion_result["reliability_score"] = consistency_report.reliability_override
                fusion_result["reliability_label"] = self._get_rel_label(consistency_report.reliability_override)
                
            if consistency_report.confidence_override >= 0:
                fusion_result["calibrated_confidence"] = consistency_report.confidence_override

            # ── Step 8: Explanation Generation (XAI) ──
            print("[DEBUG-ORCHESTRATOR] 8/8 Generating XAI Explanation...")
            explanation = self.explanation_engine.generate_explanation(
                prediction_label=fusion_result["prediction"],
                confidence=fusion_result["final_confidence"],
                reasoning=fusion_result["reasoning"],
                raw_metrics=fusion_result,
                severity=severity_report
            )
            print("[DEBUG-ORCHESTRATOR] Pipeline Complete. Assembling Response.")
            
            # ── Final Package (Authoritative Assembly) ──
            return self._build_final_response(
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
                scene_context=scene_context,
                start_time=start_time
            )
            
        except Exception as e:
            import traceback
            print(f"[CRITICAL-ORCHESTRATOR] Reasoning Pipeline Failed: {str(e)}")
            traceback.print_exc()
            return self._fallback_response(image_path, sensor_data, location)

    def _build_final_response(self, 
                             fusion_result, 
                             quality_report, 
                             severity_report, 
                             calibration, 
                             sensor_reading, 
                             climate_context, 
                             morphology_report, 
                             consistency_report, 
                             explanation,
                             cnn_results,
                             scene_context,
                             start_time) -> Dict[str, Any]:
        """
        ONE authoritative place where JSON responses are constructed.
        Ensures strict separation between root, raw_metrics, and technical_details.
        """
        # 1. Construct authoritative raw_metrics
        raw_metrics = {
            "cnn_confidence": round(cnn_results['confidence_score'], 2),
            "calibrated_confidence": round(fusion_result.get("calibrated_confidence", cnn_results['confidence_score']), 2),
            "env_score": round(fusion_result.get("env_score", 0), 2),
            "quality_report": quality_report,
            "severity_report": (
                severity_report if isinstance(severity_report, dict) 
                else {**asdict(severity_report), "discoloration": severity_report.discoloration_score, "edge_damage": severity_report.edge_damage_score}
            ),
            "calibration_report": {
                "entropy": round(calibration.entropy, 4) if calibration else None,
                "margin": round(calibration.margin, 4) if calibration else None,
                "disease_presence": round(calibration.disease_presence_confidence, 2) if calibration else None,
                "factors": calibration.calibration_factors if calibration else {}
            },
            "sensor_readings": {
                "temperature": sensor_reading.temperature,
                "humidity": sensor_reading.humidity,
                "soil_moisture": sensor_reading.soil_moisture,
                "rainfall": sensor_reading.rainfall_mm
            },
            "climate_context": {
                "zone": climate_context.get("zone", "temperate"),
                "weather_source": "Online API" if climate_context.get("is_online") else "Offline Inference"
            }
        }

        # 2. Construct authoritative technical_details
        tech_details = {
            "cnn_raw": round(cnn_results['confidence_score'], 2),
            "morphology": {
                "method": morphology_report.method_name,
                "match_score": round(morphology_report.morphology_score, 3),
                "signatures": morphology_report.signatures_found
            },
            "scene": {
                "rain": scene_context.rain_detected if scene_context else False,
                "fog": scene_context.fog_detected if scene_context else False,
                "wetness": round(scene_context.scene_wetness_score, 2) if scene_context else 0,
                "blur_type": scene_context.blur_type if scene_context else "none"
            },
            "consistency": {
                "status": consistency_report.overall_status,
                "logs": [f"{l.check_name}: {l.adjustment}" for l in consistency_report.logs]
            },
            "inference_time_ms": int((time.time() - start_time) * 1000)
        }

        # 3. Assemble Root Response (Deriving from raw_metrics/tech_details where possible)
        response = {
            "status": "success",
            "prediction": fusion_result["prediction"],
            "diagnosis": fusion_result["prediction"],
            "confidence": fusion_result["final_confidence"],
            "overall_confidence": fusion_result["final_confidence"],
            "reliability": explanation.get("reliability", "Unknown"),
            "reliability_score": fusion_result.get("reliability_score", 0.5),
            "is_healthy": "healthy" in fusion_result["prediction"].lower(),
            
            # Severity mapping
            "severity": explanation.get("severity", "Unknown"),
            "visual_severity": explanation.get("severity_classification", fusion_result.get("visual_severity", "unknown")),
            "severity_index": raw_metrics["severity_report"].get("lesion_area_pct", 0) / 100.0,
            "realism_score": round(fusion_result.get("realism_score", 0), 2),
            
            # XAI Content
            "farmer_summary": explanation.get("farmer_summary", ""),
            "concise_explanation": explanation.get("concise_explanation", ""),
            "technical_chain": explanation.get("technical_chain", []),
            "recommendation": explanation.get("recommendation", ""),
            "recommendations": explanation.get("recommendations", [explanation.get("recommendation", "")]),
            
            # Evidence Indicators
            "visual_match": explanation.get("visual_detection_confidence", "Unknown"),
            "visual_detection_confidence": explanation.get("visual_detection_confidence", "Unknown"),
            "env_support": explanation.get("environmental_support", "Unknown"),
            "environmental_support": explanation.get("environmental_support", "Unknown"),
            "image_reliability": explanation.get("image_reliability", "Unknown"),
            "warnings": explanation.get("warnings", []),
            
            # Source data blocks
            "raw_metrics": raw_metrics,
            "technical_details": tech_details,
            
            # Override Flags
            "strong_evidence_override": fusion_result.get("strong_evidence_override", False),
            "disease_presence_confidence": fusion_result.get("disease_presence_confidence", 0),
            "disease_visibility_strength": fusion_result.get("disease_visibility_strength", 0)
        }

        return self.normalize_response_schema(response)


    def _get_rel_label(self, score: float) -> str:
        if score > 0.8: return "High"
        if score > 0.5: return "Moderate"
        return "Low"

    def normalize_response_schema(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final gatekeeper: Ensures every response strictly adheres to the 
        frontend's expected JSON schema, preventing 'undefined' crashes.
        Acts as a validator/guard rather than a primary populator.
        """
        print(f"[DEBUG-ORCHESTRATOR] Validating response contract for: {response.get('prediction', 'No Prediction')}")
        
        # Define the strict contract expected by App.jsx
        REQUIRED_FIELDS = [
            "prediction", "diagnosis", "confidence", "overall_confidence",
            "reliability", "reliability_score", "severity", "visual_severity",
            "severity_index", "realism_score", "is_healthy", "farmer_summary",
            "technical_chain", "recommendation", "recommendations",
            "visual_match", "visual_detection_confidence", "env_support", 
            "environmental_support", "image_reliability", "warnings",
            "raw_metrics", "technical_details"
        ]

        # Ensure root fields exist
        for field in REQUIRED_FIELDS:
            if field not in response:
                print(f"[WARNING-ORCHESTRATOR] Missing required field: {field}. Applying default.")
                if field in ["technical_chain", "recommendations", "warnings"]:
                    response[field] = []
                elif field in ["confidence", "overall_confidence", "reliability_score", "severity_index", "realism_score"]:
                    response[field] = 0.0
                elif field == "is_healthy":
                    response[field] = False
                elif field in ["raw_metrics", "technical_details"]:
                    response[field] = {}
                else:
                    response[field] = "Unknown"

        # Special handling for raw_metrics nesting to avoid frontend 'undefined' property access
        rm = response.get("raw_metrics", {})
        if "sensor_readings" not in rm: rm["sensor_readings"] = {"temperature": None, "humidity": None, "soil_moisture": None, "rainfall": None}
        if "quality_report" not in rm: rm["quality_report"] = {"blur_score": 0, "brightness_score": 0, "leaf_visibility": 0, "texture_clarity": 0}
        if "severity_report" not in rm: rm["severity_report"] = {"lesion_area_pct": 0, "severity_level": "healthy_appearing", "realism_score": 0}
        if "climate_context" not in rm: rm["climate_context"] = {"zone": "Unknown", "weather_source": "Offline"}
        if "calibration_report" not in rm: rm["calibration_report"] = {"entropy": 0, "margin": 0, "disease_presence": 0}
        
        response["raw_metrics"] = rm

        # Ensure technical_details nesting
        td = response.get("technical_details", {})
        if "morphology" not in td: td["morphology"] = {"method": "None", "match_score": 0, "signatures": []}
        if "scene" not in td: td["scene"] = {"rain": False, "fog": False, "wetness": 0, "blur_type": "none"}
        if "consistency" not in td: td["consistency"] = {"status": "Unverified", "logs": []}
        
        response["technical_details"] = td

        return response

    def _fallback_response(self, image_path: str, sensor_data: Dict[str, Any] = None, location: Dict[str, Any] = None) -> Dict[str, Any]:
        """Emergency fallback to basic CNN prediction."""
        sensor_data = sensor_data or {}
        location = location or {}
        
        try:
            from model_utils import predict_image
            import cv2
            import numpy as np
            img = cv2.imread(image_path)
            # Resize and convert to RGB for consistency with ImageContext
            resized = cv2.resize(img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            batched = np.expand_dims(normalized, axis=0)
            res = predict_image(self.model, batched)
            
            label = res['prediction']
            conf = res['confidence_score']
            
            # Reconstruct basic sensor readings for fallback
            sensor_readings = {
                "temperature": float(sensor_data.get('temperature', 25)),
                "humidity": float(sensor_data.get('humidity', 50)),
                "soil_moisture": float(sensor_data.get('soil', sensor_data.get('soil_moisture', sensor_data.get('soilMoisture', 30)))),
                "rainfall": float(sensor_data.get('rainfall', 0))
            }
            
            return self.normalize_response_schema({
                "status": "success",
                "prediction": label,
                "diagnosis": label,
                "overall_confidence": round(conf, 2),
                "confidence": round(conf, 2),
                "reliability": "Direct CNN Output (Fallback)",
                "banner_color": "amber",
                "farmer_summary": f"Direct analysis suggests {label.replace('_', ' ')}. System reasoning was limited.",
                "concise_explanation": "Reasoning engine encountered an error. Falling back to raw neural network classification.",
                "technical_chain": ["Orchestrator fallback triggered due to internal error"],
                "image_reliability": "Unknown",
                "environmental_support": "None (Fallback)",
                "visual_detection_confidence": round(conf, 2),
                "warnings": ["Reasoning system error - results may be less reliable"],
                "recommendation": "Consult an expert for confirmation as the reasoning logic was bypassed.",
                "raw_metrics": {
                    "sensor_readings": sensor_readings
                },
                "fallback_mode": True
            })
        except Exception as e:
            return self.normalize_response_schema({
                "status": "error",
                "error": f"Critical system failure: {str(e)}",
                "prediction": "Unknown",
                "diagnosis": "Unknown",
                "overall_confidence": 0,
                "reliability": "Error",
                "banner_color": "red",
                "farmer_summary": "System error occurred during diagnosis.",
                "concise_explanation": "Critical failure in the reasoning pipeline.",
                "fallback_mode": True,
                "raw_metrics": {
                    "sensor_readings": {
                        "temperature": float(sensor_data.get('temperature', 25)),
                        "humidity": float(sensor_data.get('humidity', 50)),
                        "soil_moisture": float(sensor_data.get('soil', sensor_data.get('soil_moisture', sensor_data.get('soilMoisture', 30)))),
                        "rainfall": 0
                    }
                }
            })
