"""
HarvestShield — Probabilistic Climate-Adaptive Fusion Engine
============================================================
The central intelligence layer for HarvestShield.
Integrates CNN, BLE Sensors, Geo-Climate Context, and Advanced Image Diagnostics.
"""

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

from image_quality_utils import ImageDiagnostics
from explanation_engine import ExplanationEngine
from confidence_calibration import ConfidenceCalibrator, CalibratedPrediction
from visual_severity_utils import VisualSeverityAnalyzer, SeverityReport

# ---------------------------------------------------------------------------
# 1. Enums & Data containers
# ---------------------------------------------------------------------------

class ClimateZone(Enum):
    TROPICAL  = "tropical"   # India, SE Asia — hot, wet, high humidity
    MONSOON   = "monsoon"    # seasonal heavy rain + heat cycles
    HUMID     = "humid"      # subtropical humid (eastern coastal India)
    TEMPERATE = "temperate"  # Europe, northern US — default textbook baseline
    DRY       = "dry"        # semi-arid, low rainfall
    UNKNOWN   = "unknown"

# Zones that get tropical-adjusted scoring
TROPICAL_ZONES = {ClimateZone.TROPICAL, ClimateZone.MONSOON, ClimateZone.HUMID}


@dataclass
class SensorReading:
    """BLE sensor data + geo-climate context from the IoT layer."""
    temperature:   float                    # °C
    humidity:      float                    # %
    soil_moisture: float                    # %
    rainfall_mm:   Optional[float] = None   # mm
    climate_zone:  ClimateZone = ClimateZone.TEMPERATE

@dataclass
class ReasoningState:
    """Internal semantic state for explaining fusion logic."""
    visual_strength: str      # strong, moderate, weak
    env_support: str         # supportive, moderate, weak, negative
    temp_effect: str         # ideal, supportive, stressful, restrictive
    humidity_effect: str     # high_risk, supportive, neutral, negative
    soil_effect: str         # supportive, neutral, dry
    rainfall_effect: str     # high_risk, supportive, none
    climate_adjustment: str  # supportive, neutral
    image_reliability: str   # high, moderate, low
    weather_support: str     # high, supportive, none
    contradiction_level: str # none, minor, major
    # New severity-aware fields
    severity_level: str = "unknown"       # healthy_appearing, mild, moderate, severe
    realism_assessment: str = "unknown"   # strong, moderate, weak, minimal
    disease_gate: str = "passed"          # passed, failed
    calibration_applied: str = "none"     # none, minor, major
    prediction_label: str = "unknown"     # The disease/healthy label


@dataclass
class DiseaseEnvProfile:
    """Probabilistic environmental parameters for one disease."""
    temp_optimal:          float
    temp_spread:           float
    tropical_temp_shift:   float
    tropical_spread_boost: float
    humidity_threshold:    float
    humidity_steepness:    float
    soil_threshold:        float
    soil_steepness:        float
    weight_temp:           float
    weight_humidity:       float
    weight_soil:           float
    humidity_temp_compensation: float
    rainfall_threshold_mm: float = 5.0
    rainfall_bonus:        float = 0.08
    floor_score: float = 0.06


# ---------------------------------------------------------------------------
# 2. Disease profile registry
# ---------------------------------------------------------------------------

DISEASE_PROFILES: Dict[str, DiseaseEnvProfile] = {
    "Tomato_Late_blight": DiseaseEnvProfile(
        temp_optimal=18.0, temp_spread=6.0,
        tropical_temp_shift=6.0, tropical_spread_boost=10.0,
        humidity_threshold=75.0, humidity_steepness=0.12,
        soil_threshold=55.0, soil_steepness=0.10,
        weight_temp=0.40, weight_humidity=0.40, weight_soil=0.20,
        humidity_temp_compensation=0.40,
        rainfall_bonus=0.10,
    ),
    "Tomato_Early_blight": DiseaseEnvProfile(
        temp_optimal=26.0, temp_spread=7.0,
        tropical_temp_shift=3.0, tropical_spread_boost=6.0,
        humidity_threshold=60.0, humidity_steepness=0.10,
        soil_threshold=40.0, soil_steepness=0.08,
        weight_temp=0.35, weight_humidity=0.35, weight_soil=0.30,
        humidity_temp_compensation=0.25,
        rainfall_bonus=0.06,
    ),
    "Tomato_healthy": DiseaseEnvProfile(
        temp_optimal=26.0, temp_spread=8.0,
        tropical_temp_shift=2.0, tropical_spread_boost=4.0,
        humidity_threshold=50.0, humidity_steepness=0.08,
        soil_threshold=35.0, soil_steepness=0.07,
        weight_temp=0.40, weight_humidity=0.30, weight_soil=0.30,
        humidity_temp_compensation=0.10,
        floor_score=0.10,
    ),
}

_DEFAULT_PROFILE = DiseaseEnvProfile(
    temp_optimal=24.0, temp_spread=8.0,
    tropical_temp_shift=3.0, tropical_spread_boost=6.0,
    humidity_threshold=65.0, humidity_steepness=0.10,
    soil_threshold=45.0, soil_steepness=0.08,
    weight_temp=0.35, weight_humidity=0.40, weight_soil=0.25,
    humidity_temp_compensation=0.25,
    floor_score=0.07,
)

# ---------------------------------------------------------------------------
# 3. Core scoring functions
# ---------------------------------------------------------------------------

def _gaussian(value: float, center: float, spread: float) -> float:
    z = (value - center) / spread
    return math.exp(-0.5 * z * z)

def _sigmoid(value: float, threshold: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (value - threshold)))

def _score_temperature(profile: DiseaseEnvProfile, temp: float, humidity: float, zone: ClimateZone) -> float:
    is_tropical = zone in TROPICAL_ZONES
    center = profile.temp_optimal + (profile.tropical_temp_shift if is_tropical else 0)
    spread = profile.temp_spread + (profile.tropical_spread_boost if is_tropical else 0)
    base_score = _gaussian(temp, center, spread)
    
    sigma_deviation = abs(temp - center) / spread
    if sigma_deviation > 1.0:
        hum_score = _sigmoid(humidity, profile.humidity_threshold, profile.humidity_steepness)
        max_rescue = profile.humidity_temp_compensation * (1.0 - base_score)
        base_score = min(base_score + hum_score * max_rescue, 0.92)
    return max(base_score, profile.floor_score)

# ---------------------------------------------------------------------------
# 4. Probabilistic Fusion Engine
# ---------------------------------------------------------------------------

class ProbabilisticFusionEngine:
    CNN_WEIGHT_BASE = 0.60
    ENV_WEIGHT_BASE = 0.40
    FP_THRESHOLD = 25.0
    FP_CNN_MIN = 65.0
    IQ_FLIP_THRESHOLD = 0.40
    IQ_HIGH_THRESHOLD = 0.85

    def _get_semantic_level(self, score: float, thresholds: Dict[str, float]) -> str:
        """Maps a numeric score (0-1) to a semantic label based on custom thresholds."""
        # Sort thresholds descending by value
        sorted_keys = sorted(thresholds.keys(), key=lambda k: thresholds[k], reverse=True)
        for key in sorted_keys:
            if score >= thresholds[key]:
                return key
        return "negative" if "negative" in thresholds else "weak"

    def get_profile(self, label: str) -> DiseaseEnvProfile:
        return DISEASE_PROFILES.get(label, _DEFAULT_PROFILE)

    def fuse(self, label: str, cnn_conf: float, sensor: SensorReading, quality_report: Dict[str, Any],
             calibration: CalibratedPrediction = None, severity: SeverityReport = None) -> Dict[str, Any]:
        profile = self.get_profile(label)
        
        # ── Use calibrated confidence if available ──
        effective_conf = calibration.calibrated_confidence if calibration else cnn_conf

        # Environmental scoring
        t_score = _score_temperature(profile, sensor.temperature, sensor.humidity, sensor.climate_zone)
        h_score = _sigmoid(sensor.humidity, profile.humidity_threshold, profile.humidity_steepness)
        s_score = _sigmoid(sensor.soil_moisture, profile.soil_threshold, profile.soil_steepness)
        
        print(f"[DEBUG-FUSION] Profile: {label}")
        print(f"[DEBUG-FUSION] Temp Score: {t_score:.2f} (Input: {sensor.temperature})")
        print(f"[DEBUG-FUSION] Hum Score: {h_score:.2f} (Input: {sensor.humidity})")
        print(f"[DEBUG-FUSION] Soil Score: {s_score:.2f} (Input: {sensor.soil_moisture})")

        r_bonus = 0.0
        if sensor.rainfall_mm and sensor.rainfall_mm >= profile.rainfall_threshold_mm:
            r_bonus = profile.rainfall_bonus * min(sensor.rainfall_mm / (profile.rainfall_threshold_mm * 3), 1.0)
            print(f"[DEBUG-FUSION] Rainfall Bonus: {r_bonus:.2f} (Input: {sensor.rainfall_mm})")
            
        env_score_pct = min((profile.weight_temp * t_score + profile.weight_humidity * h_score + profile.weight_soil * s_score) + r_bonus, 1.0) * 100.0
        print(f"[DEBUG-FUSION] Final Env Score: {env_score_pct:.2f}%")

        # Multi-dimensional Image Quality Integration
        iq = quality_report["aggregate_quality"]
        
        # Reliability check: if leaf visibility is low, penalize confidence regardless of blur
        if quality_report["leaf_visibility"] < 0.3:
            iq *= 0.5 

        # ── Context-Aware Image Quality (Part 7): Separate technical quality from disease visibility ──
        # If disease is visually obvious, moderate image quality issues shouldn't dominate
        dvs = calibration.disease_visibility_strength if calibration else 0.0
        effective_iq = iq
        if dvs > 0.50 and iq < self.IQ_HIGH_THRESHOLD:
            # Lift effective IQ toward moderate when disease evidence is strong
            boost = min((dvs - 0.50) * 0.8, 0.3)  # Up to +0.3 IQ boost
            effective_iq = min(iq + boost, self.IQ_HIGH_THRESHOLD)

        # Adaptive weighting (uses effective_iq which accounts for disease visibility)
        if effective_iq < self.IQ_FLIP_THRESHOLD:
            cnn_w, env_w = 0.30, 0.70
        elif effective_iq > self.IQ_HIGH_THRESHOLD:
            cnn_w, env_w = 0.75, 0.25
        else:
            t = (effective_iq - self.IQ_FLIP_THRESHOLD) / (self.IQ_HIGH_THRESHOLD - self.IQ_FLIP_THRESHOLD)
            cnn_w = self.CNN_WEIGHT_BASE + t * (0.75 - self.CNN_WEIGHT_BASE)
            env_w = self.ENV_WEIGHT_BASE + (1 - t) * (0.70 - self.ENV_WEIGHT_BASE)

        final_conf = cnn_w * effective_conf + env_w * env_score_pct
        possible_fp = (env_score_pct < self.FP_THRESHOLD and effective_conf >= self.FP_CNN_MIN)

        print(f"[DEBUG-FUSION] Effective Conf: {effective_conf:.2f}, Env Score: {env_score_pct:.2f}, IQ: {effective_iq:.2f}")
        print(f"[DEBUG-FUSION] CNN Weight: {cnn_w:.2f}, Env Weight: {env_w:.2f}")

        # ── Contradiction-Aware CNN Skepticism ──
        # ONLY fires when evidence is genuinely weak — not for visually obvious disease
        skepticism_applied = False
        strong_evidence_override = False
        if severity and calibration:
            is_overconfident = cnn_conf > 80
            is_env_weak = env_score_pct < 40
            
            # Support both dataclass and dict for orchestrator flexibility
            s_level = severity.get('severity_level', 'unknown') if isinstance(severity, dict) else (severity.severity_level if severity else "unknown")
            s_realism = severity.get('realism_score', 0) if isinstance(severity, dict) else (severity.realism_score if severity else 0)
            
            is_severity_low = s_level in ("healthy_appearing", "mild")
            is_severity_high = s_level in ("moderate", "severe")
            is_realism_low = s_realism < 0.3
            is_realism_high = s_realism > 0.50
            disease_pres = calibration.disease_presence_confidence

            # ── Strong Evidence Override (Part 2) ──
            # If disease is visually undeniable, SKIP skepticism and boost trust
            if is_severity_high and is_realism_high and disease_pres > 85:
                strong_evidence_override = True
                # Set a confidence floor: obvious disease shouldn't go below 70% (increased from 65%)
                evidence_floor = 70.0 + (dvs * 15.0) 
                # Push cnn_w even higher to trust visual more in undeniable cases
                if final_conf >= 80:
                    cnn_w = min(cnn_w + 0.1, 0.85)
                    env_w = 1.0 - cnn_w
                    final_conf = cnn_w * effective_conf + env_w * env_score_pct
                
                # Final floor check: undeniably diseased should be at least at the evidence floor
                if final_conf < evidence_floor:
                    final_conf = evidence_floor
            else:
                # Original skepticism logic — only for weak/ambiguous evidence
                # Triple contradiction: CNN high + env weak + visually mild
                if is_overconfident and is_env_weak and is_severity_low:
                    penalty = 0.55 if s_level == "healthy_appearing" else 0.70
                    final_conf *= penalty
                    skepticism_applied = True

                # Realism gate: CNN says disease but image shows almost nothing
                if is_overconfident and is_realism_low:
                    final_conf *= 0.65
                    skepticism_applied = True

            # Disease gate: healthy probability was high (applies only to disease predictions)
            if not calibration.disease_gate_passed and "healthy" not in label.lower():
                final_conf = min(final_conf, 45.0)
                skepticism_applied = True

        # Determine semantic levels for severity/realism
        sev_level = s_level if severity else "unknown"
        realism_label = "unknown"
        if severity:
            realism_label = self._get_semantic_level(s_realism, {"strong": 0.70, "moderate": 0.40, "weak": 0.15, "minimal": 0.0})

        cal_label = "none"
        if calibration:
            reduction = (cnn_conf - calibration.calibrated_confidence) / max(cnn_conf, 1)
            cal_label = "major" if reduction > 0.25 else ("minor" if reduction > 0.05 else "none")

        # ── Visual strength now considers BOTH calibrated conf AND severity (Part 7) ──
        # Prevents "weak visual match" when severity is clearly moderate/severe
        if "healthy" in label.lower():
            # For healthy, "strong" means the model is VERY sure it's healthy
            visual_str = self._get_semantic_level(effective_conf/100.0, {"strong": 0.90, "moderate": 0.70, "weak": 0.40})
        elif severity and s_level in ("moderate", "severe") and s_realism > 0.4:
            visual_str = "strong" if s_level == "severe" else "moderate"
        else:
            visual_str = self._get_semantic_level(effective_conf/100.0, {"strong": 0.85, "moderate": 0.60, "weak": 0.40})

        # Generate Reasoning State
        reasoning = ReasoningState(
            visual_strength=visual_str,
            env_support=self._get_semantic_level(env_score_pct/100.0, {"supportive": 0.75, "moderate": 0.50, "weak": 0.30, "negative": 0.0}),
            temp_effect=self._get_semantic_level(t_score, {"ideal": 0.90, "supportive": 0.70, "stressful": 0.40, "restrictive": 0.0}),
            humidity_effect=self._get_semantic_level(h_score, {"high_risk": 0.85, "supportive": 0.65, "neutral": 0.40, "negative": 0.0}),
            soil_effect=self._get_semantic_level(s_score, {"supportive": 0.70, "neutral": 0.40, "dry": 0.0}),
            rainfall_effect="high_risk" if r_bonus > 0.05 else ("supportive" if r_bonus > 0 else "none"),
            climate_adjustment="supportive" if sensor.climate_zone in TROPICAL_ZONES else "neutral",
            image_reliability=self._get_semantic_level(effective_iq, {"high": 0.85, "moderate": 0.50, "low": 0.0}),
            weather_support="high" if (sensor.rainfall_mm and sensor.rainfall_mm > 5) else ("supportive" if sensor.rainfall_mm else "none"),
            contradiction_level="major" if (effective_conf > 80 and env_score_pct < 20) else ("minor" if possible_fp else "none"),
            severity_level=sev_level,
            realism_assessment=realism_label,
            disease_gate="passed" if (not calibration or calibration.disease_gate_passed) else "failed",
            calibration_applied=cal_label,
            prediction_label=label
        )

        return {
            "prediction": label,
            "cnn_confidence": round(cnn_conf, 2),
            "calibrated_confidence": round(effective_conf, 2),
            "env_score": round(env_score_pct, 2),
            "final_confidence": round(final_conf, 2),
            "reliability": self._get_nuanced_reliability(final_conf, reasoning),
            "reliability_score": round(final_conf / 100.0, 3),
            "possible_false_positive": possible_fp,
            "skepticism_applied": skepticism_applied,
            "strong_evidence_override": strong_evidence_override,
            "disease_visibility_strength": round(dvs, 3),
            "visual_severity": s_level,
            "realism_score": s_realism,
            "disease_presence_confidence": calibration.disease_presence_confidence if calibration else None,
            "quality_report": quality_report,
            "reasoning": reasoning
        }

    def _get_nuanced_reliability(self, conf: float, reasoning: ReasoningState) -> str:
        # 0. Handle Healthy Leaves specifically
        if "healthy" in reasoning.severity_level or "healthy" in reasoning.prediction_label.lower():
            if conf >= 85: return "Highly Reliable (Healthy)"
            if conf >= 70: return "Moderately Reliable (Healthy)"
            return "Limited Confidence (Healthy)"

        # 1. Handle major contradictions first
        if reasoning.contradiction_level == "major" and reasoning.visual_strength == "strong":
            return "Visually Obvious (Env Divergence)"
        
        # 2. Handle high confidence + environmental support
        if conf >= 80 and reasoning.env_support in ("supportive", "moderate"):
            return "Contextually Confirmed"
            
        # 3. Handle high visual strength but weak env
        if reasoning.visual_strength == "strong" and conf >= 70:
            return "Strong Visual Evidence"
            
        # 4. Handle "Healthy Appearing" leaves with high raw confidence (Skepticism Layer)
        if reasoning.severity_level == "healthy_appearing" and conf > 50:
            return "Visual Overestimate (Leaf Appears Healthy)"
            
        # 5. Handle low reliability/skepticism cases
        # For disease, low realism means the visual evidence doesn't match the label
        if reasoning.image_reliability == "low" or reasoning.realism_assessment in ("weak", "minimal"):
            return "Limited Visibility Match"
            
        # 6. Predictive uncertainty
        if conf < 55:
            return "Predictive Caution (Low Confidence)"
            
        # 7. Default fallback
        if conf < 75: return "Moderately Reliable"
        return "Highly Reliable"

# ---------------------------------------------------------------------------
# 5. Backward Compatibility Wrapper
# ---------------------------------------------------------------------------

def recommendation(label):
    recommendations = {
        "Tomato_Early_blight": "Remove infected leaves and apply fungicide. Avoid excessive moisture.",
        "Tomato_Late_blight": "Reduce humidity and avoid overhead irrigation. Improve airflow.",
        "Tomato_healthy": "Maintain current environmental conditions and continue regular monitoring."
    }
    return recommendations.get(label, "No recommendation available.")

def advanced_fusion_engine(image_path, predicted_class, cnn_confidence, temperature, humidity, soil, climate_context=None, full_probabilities=None):
    # Prepare inputs
    zone_raw = (climate_context.get("climate_zone", "temperate") if climate_context else "temperate").lower()
    
    # Map descriptive strings to Enum
    if "tropical" in zone_raw:
        zone = ClimateZone.TROPICAL
    elif "monsoon" in zone_raw:
        zone = ClimateZone.MONSOON
    elif "humid" in zone_raw:
        zone = ClimateZone.HUMID
    elif "dry" in zone_raw or "arid" in zone_raw:
        zone = ClimateZone.DRY
    else:
        zone = ClimateZone.TEMPERATE

        
    weather_data = climate_context.get("weather") if climate_context else None
    rainfall = weather_data.get("rain", 0) if weather_data else None
    
    sensor = SensorReading(
        temperature=temperature,
        humidity=humidity,
        soil_moisture=soil,
        rainfall_mm=rainfall,
        climate_zone=zone
    )

    # Image Diagnostics
    diagnostics = ImageDiagnostics(image_path)
    quality_report = diagnostics.get_quality_report()

    # ── Visual Severity Analysis (run FIRST so calibrator can use it) ──
    severity_analyzer = VisualSeverityAnalyzer(image_path)
    severity = severity_analyzer.analyze()

    # ── CNN Confidence Calibration (now context-aware via severity) ──
    calibration = None
    if full_probabilities:
        calibrator = ConfidenceCalibrator()
        calibration = calibrator.calibrate(
            predicted_label=predicted_class,
            raw_confidence=cnn_confidence,
            full_probabilities=full_probabilities,
            image_quality=quality_report["aggregate_quality"],
            severity_report=severity
        )

    # ── Fusion with Skepticism + Strong Evidence Override ──
    engine = ProbabilisticFusionEngine()
    fusion_result = engine.fuse(predicted_class, cnn_confidence, sensor, quality_report, calibration, severity)

    # ── Explainable AI Integration (Part 7) ──
    explainer = ExplanationEngine()
    xai_report = explainer.generate_explanation(
        fusion_result["prediction"], 
        fusion_result["final_confidence"], 
        fusion_result["reasoning"],
        fusion_result,
        severity
    )

    # ── Final response assembly (Part 8) ──
    return {
        "prediction": xai_report["prediction"],
        "overall_confidence": xai_report["overall_confidence"],
        "reliability": xai_report["reliability"],
        "banner_color": xai_report["banner_color"],
        
        "concise_explanation": xai_report["concise_explanation"],
        "detailed_explanation": xai_report["detailed_explanation"],
        "technical_chain": xai_report["technical_chain"],
        "farmer_summary": xai_report["farmer_summary"],

        "environmental_support": xai_report["environmental_support"],
        "visual_detection_confidence": xai_report["visual_detection_confidence"],
        "image_reliability": xai_report["image_reliability"],

        "warnings": xai_report["warnings"],
        "recommendation": xai_report["recommendation"],

        # Severity-aware fields
        "visual_severity": fusion_result["visual_severity"],
        "severity_score": severity.get('severity_score', 0) if isinstance(severity, dict) else (severity.severity_score if severity else 0),
        "realism_score": fusion_result["realism_score"],
        "disease_presence_confidence": fusion_result["disease_presence_confidence"],
        "calibrated_confidence": fusion_result["calibrated_confidence"],
        "disease_gate_passed": fusion_result["reasoning"].disease_gate == "passed",
        "disease_visibility_strength": fusion_result["disease_visibility_strength"],
        "contradiction_level": fusion_result["reasoning"].contradiction_level if hasattr(fusion_result["reasoning"], "contradiction_level") else "none",
        "strong_evidence_override": fusion_result.get("strong_evidence_override", False),
        "reliability_score": fusion_result["reliability_score"],

        # Keep raw metrics for UI charts/debug
        "raw_metrics": {
            "cnn_confidence": fusion_result["cnn_confidence"],
            "calibrated_confidence": fusion_result["calibrated_confidence"],
            "env_score": fusion_result["env_score"],
            "quality_report": quality_report,
            "severity_report": severity if isinstance(severity, dict) else {
                "lesion_area_pct": severity.lesion_area_pct if severity else 0,
                "discoloration": severity.discoloration_score if severity else 0,
                "edge_damage": severity.edge_damage_score if severity else 0,
                "symptom_density": severity.symptom_density if severity else 0,
                "texture_abnormality": severity.texture_abnormality if severity else 0,
                "severity_score": severity.severity_score if severity else 0,
                "realism_score": severity.realism_score if severity else 0
            },
            "calibration_report": {
                "entropy": calibration.entropy if calibration else None,
                "margin": calibration.margin if calibration else None,
                "factors": calibration.calibration_factors if calibration else {},
                "disease_presence": calibration.disease_presence_confidence if calibration else None
            },
            "sensor_readings": {
                "temperature": sensor.temperature,
                "humidity": sensor.humidity,
                "soil_moisture": sensor.soil_moisture,
                "rainfall": sensor.rainfall_mm
            },
            "climate_context": {
                "zone": zone.value.capitalize(),
                "weather_source": "Online API" if (climate_context and climate_context.get("is_online")) else "Offline Inference",
                "weather_data": weather_data
            }
        }

    }