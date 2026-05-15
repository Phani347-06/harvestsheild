"""
HarvestShield — Explainable AI (XAI) Reasoning Layer
====================================================
Translates probabilistic fusion outputs into human-understandable reasoning.
Now includes severity-aware explanations, realism commentary, and 
calibration-informed uncertainty communication.

v2: Semantic consistency rules prevent contradictory outputs like
    "Disease Presence: 99% + Reliability: Low". Context-aware warnings
    distinguish between image quality and disease visibility.
"""

from typing import Dict, Any, List, Optional


class ExplanationEngine:
    def __init__(self):
        # Human-friendly terminology mapping
        self.term_mapping = {
            "prediction": "Diagnosis",
            "cnn_confidence": "Visual Detection Confidence",
            "env_score": "Environmental Support",
            "reliability": "System Reliability",
            "quality": "Image Reliability"
        }

    def _get_sev_attr(self, severity: Any, attr: str, default: Any = None) -> Any:
        """Helper to safely get attributes from either SeverityReport object or dict."""
        if not severity:
            return default
        if isinstance(severity, dict):
            return severity.get(attr, default)
        return getattr(severity, attr, default)

    def generate_explanation(self, prediction_label: str, confidence: float, reasoning: Any, raw_metrics: Dict[str, Any], severity=None) -> Dict[str, Any]:
        """
        Generates structured explanations from reasoning state.
        reasoning: instance of ReasoningState from fusion_engine.py
        severity: optional SeverityReport from visual_severity_utils.py
        """
        r = reasoning
        
        # 1. Multi-factor Reliability Calibration (Part 3 & 5)
        reliability_level = self._get_reliability_label(confidence, r, prediction_label, severity)
        
        # 2. Fragment Generation (Humanized)
        fragments = []
        tech_chain = []
        
        disease_name = prediction_label.replace("_", " ")
        is_healthy = "healthy" in prediction_label.lower()

        # ── Visual logic with severity awareness ──
        cnn_conf = raw_metrics.get("cnn_confidence", 0)
        calibrated_conf = raw_metrics.get("calibrated_confidence", cnn_conf)
        
        tech_chain.append(f"Raw CNN confidence: {cnn_conf}%.")
        if calibrated_conf != cnn_conf:
            tech_chain.append(f"Calibrated confidence: {calibrated_conf}% (after temperature scaling, entropy, and margin analysis).")
        
        # Severity-informed visual fragments
        if severity and not is_healthy:
            sev = self._get_sev_attr(severity, 'severity_level', r.severity_level)
            
            if sev == "severe":
                fragments.append(f"Clear and widespread symptoms of {disease_name} are visible on the leaf.")
            elif sev == "moderate":
                fragments.append(f"Noticeable symptoms consistent with {disease_name} are present.")
            elif sev == "mild":
                fragments.append(f"Minor visual patterns resembling {disease_name} were detected, but symptom spread appears limited.")
            else:  # healthy_appearing
                fragments.append(f"The leaf appears largely healthy. Very faint patterns may resemble {disease_name}, but visible evidence is minimal.")
                
            # Lesion detail
            lesion_pct = self._get_sev_attr(severity, 'lesion_area_pct', 0)
            if lesion_pct > 1:
                tech_chain.append(f"Approximately {lesion_pct:.1f}% of leaf area shows abnormal coloration.")
            else:
                tech_chain.append("Less than 1% of leaf area shows abnormal coloration.")
        elif is_healthy:
            fragments.append("The leaf exhibits healthy coloration and texture with no significant lesions detected.")
            if severity:
                lesion_pct = self._get_sev_attr(severity, 'lesion_area_pct', 0)
                tech_chain.append(f"Leaf health verified: minimal lesion area ({lesion_pct:.2f}%).")
            else:
                tech_chain.append("Leaf health verified through visual pattern consistency.")
        else:
            # Fallback: use visual_strength from reasoning (no severity data)
            if r.visual_strength == "strong":
                if r.contradiction_level == "major":
                    fragments.append(f"Leaf symptoms strongly resemble {disease_name}, though other data is inconsistent.")
                else:
                    fragments.append(f"The uploaded leaf strongly resembles {disease_name}.")
            elif r.visual_strength == "moderate":
                fragments.append(f"The image shows moderate signs of {disease_name}.")
            else:
                fragments.append(f"Visual evidence for {disease_name} is currently weak.")

        # ── Environmental logic (Farmer-friendly) ──
        env_score = raw_metrics.get("env_score", 0)
        tech_chain.append(f"Environmental support calculated at {env_score}%.")

        if r.env_support == "supportive":
            fragments.append(f"Current weather conditions are highly conducive to this condition.")
        elif r.env_support == "moderate":
            fragments.append(f"Field conditions are partially favorable for disease development.")
        elif r.env_support == "weak":
            fragments.append(f"Current weather conditions are not ideal for rapid disease spread.")
        elif r.env_support == "negative":
            fragments.append(f"Dry or restrictive conditions may slow down any potential spread.")
        
        if is_healthy:
            fragments.append("Current environmental factors are within safe bounds for crop health.")
        
        # Specific effects
        if r.temp_effect == "ideal":
            fragments.append("Temperatures are currently in the range where this pathogen thrives.")
        elif r.temp_effect == "restrictive":
            fragments.append("Current temperatures may naturally inhibit rapid spread.")
            tech_chain.append("Temperature score reduced due to deviation from optimal spread range.")

        if r.humidity_effect == "high_risk":
            fragments.append("High humidity levels are increasing the risk of development.")
        
        if r.rainfall_effect == "high_risk":
            fragments.append("Recent rainfall has created a high-risk environment for moisture-loving diseases.")

        # Climate adaptation (Humanized)
        if r.climate_adjustment == "supportive":
            fragments.append("Local tropical conditions may still support disease persistence even in drier spells.")
            tech_chain.append("Tropical climate rules prevented environmental rejection of the visual match.")

        # ── Quality/Uncertainty (Part 6 & 7: Context-Aware Warnings) ──
        iq = raw_metrics.get("quality_report", {}).get("aggregate_quality", 1.0)
        tech_chain.append(f"Image quality score: {round(iq, 2)}. Weighting adjusted accordingly.")
        
        # Part 6: Differentiate quality warning based on disease visibility
        if r.image_reliability == "low":
            s_level = self._get_sev_attr(severity, 'severity_level')
            s_realism = self._get_sev_attr(severity, 'realism_score', 0)
            if severity and s_level in ("moderate", "severe") and s_realism > 0.4:
                # Disease is clear despite poor image quality
                fragments.append("Image quality has minor imperfections, but disease symptoms remain clearly distinguishable.")
            else:
                fragments.append("Note: Image quality is low, which reduces our prediction certainty.")

        # ── Severity and Realism technical chain ──
        if severity:
            realism = self._get_sev_attr(severity, 'realism_score', 0)
            s_level = self._get_sev_attr(severity, 'severity_level', r.severity_level)
            tech_chain.append(f"Visual realism score: {realism:.2f} ({r.realism_assessment}).")
            tech_chain.append(f"Severity classification: {s_level}.")

        # Disease visibility strength
        dvs = raw_metrics.get("disease_visibility_strength", 0)
        if dvs > 0:
            tech_chain.append(f"Disease visibility strength: {dvs:.2f}.")

        # ── Calibration technical chain ──
        if r.calibration_applied != "none":
            tech_chain.append(f"Confidence calibration applied: {r.calibration_applied} adjustment.")
        
        if r.disease_gate == "failed" and not is_healthy:
            tech_chain.append("DISEASE GATE: Healthy class probability was significant - disease evidence is weak.")
            fragments.append("The system has low certainty that the leaf is actually diseased.")

        # ── Contradiction handling ──
        if r.contradiction_level == "major":
            tech_chain.append("MAJOR CONTRADICTION: Visual match is strong but environmental data is negative.")
        
        # Strong evidence override note
        strong_override = raw_metrics.get("strong_evidence_override", False)
        if strong_override:
            tech_chain.append("STRONG EVIDENCE OVERRIDE: Visual severity confirmed disease - skepticism reduced.")

        # Skepticism note
        skepticism = raw_metrics.get("skepticism_applied", False)
        if skepticism:
            tech_chain.append("CNN skepticism engaged: confidence reduced due to low severity + environmental mismatch.")

        # 3. Assembly
        concise = self._assemble_concise(fragments, r, severity)
        farmer_summary = self._generate_farmer_summary(prediction_label, reliability_level, r, severity)
        banner_color = self._get_banner_color(confidence, r, severity)
        recommendation = self._get_calibrated_recommendation(prediction_label, confidence, r, severity)

        return {
            "prediction": prediction_label,
            "overall_confidence": round(confidence, 2),
            "reliability": reliability_level,
            "banner_color": banner_color,
            "concise_explanation": concise,
            "detailed_explanation": fragments,
            "technical_chain": tech_chain,
            "farmer_summary": farmer_summary,
            "recommendation": recommendation,
            "environmental_support": r.env_support.capitalize(),
            "visual_detection_confidence": r.visual_strength.capitalize(),
            "image_reliability": r.image_reliability.capitalize(),
            "warnings": self._get_warnings(r, severity)
        }

    def _get_reliability_label(self, conf: float, r: Any, prediction_label: str, severity=None) -> str:
        # ── 1. Context-Aware Nuanced Labels (Part 5) ──
        # Check for major contradictions first (High visual, Low env)
        if hasattr(r, "contradiction_level") and r.contradiction_level == "major" and r.visual_strength == "strong":
            return "Visually Obvious (Env Divergence)"
        
        # ── 2. Standard reliability cascade with semantic floors ──
        is_healthy = "healthy" in prediction_label.lower()
        s_level = self._get_sev_attr(severity, 'severity_level', r.severity_level)
        s_realism = self._get_sev_attr(severity, 'realism_score', 0)
        
        has_strong_visual = (severity and s_level in ("moderate", "severe") and s_realism > 0.40)
        has_strong_match = r.visual_strength in ("strong", "moderate")
        
        # Semantic floor
        semantic_floor = None
        if is_healthy:
            if conf >= 85: return "Highly Reliable (Healthy)"
            if conf >= 70: return "Moderately Reliable (Healthy)"
            return "Limited Confidence (Healthy)"
            
        if severity and s_level == "severe" and s_realism > 0.50:
            semantic_floor = "Moderate to High Confidence"
        elif has_strong_visual and has_strong_match:
            semantic_floor = "Moderate Confidence"
        
        # Determine base label
        if r.disease_gate == "failed":
            label = "Low Confidence (Weak Disease Evidence)"
        elif r.contradiction_level == "major" and conf > 70:
            label = "Visually Obvious (Env Divergence)"
        elif r.visual_strength == "strong" and conf >= 70:
            label = "Strong Visual Evidence"
        elif r.severity_level == "healthy_appearing" and conf > 60:
            label = "Visual Overestimate (Leaf Appears Healthy)"
        elif r.image_reliability == "low":
            label = "Moderate Confidence (Clear Symptoms Despite Noise)" if has_strong_visual else "Low Reliability"
        elif r.realism_assessment == "minimal" and conf > 50:
            label = "Moderate Confidence" if has_strong_visual else "Low Realism"
        elif conf >= 90:
            label = "Highly Reliable"
        elif conf >= 75:
            label = "Moderate to High Confidence"
        elif conf >= 55:
            label = "Moderate Confidence"
        else:
            label = "Initial Assessment (Low Confidence)"
        
        # Apply semantic floor
        rank = {
            "Low Reliability": 0, "Low Realism": 1, "Initial Assessment (Low Confidence)": 1,
            "Low Confidence (Weak Disease Evidence)": 0, "Visual Overestimate (Leaf Appears Healthy)": 2,
            "Moderate Confidence": 3, "Moderate Confidence (Clear Symptoms Despite Noise)": 3,
            "Visually Obvious (Env Divergence)": 4, "Strong Visual Evidence": 4,
            "Moderate to High Confidence": 5, "Contextually Confirmed": 6, "Highly Reliable": 7
        }
        
        if semantic_floor:
            floor_rank = rank.get(semantic_floor, 3)
            current_rank = rank.get(label, 0)
            if current_rank < floor_rank:
                label = semantic_floor
        
        return label

    def _get_banner_color(self, conf: float, r: Any, severity=None) -> str:
        # Disease gate failure -> always amber
        if r.disease_gate == "failed":
            return "amber"

        if r.contradiction_level == "major":
            return "amber"
        
        # Healthy-appearing with high confidence -> skeptical amber
        if r.severity_level == "healthy_appearing" and conf > 50:
            return "amber"
        
        # Severe + reasonable confidence -> alert (regardless of minor quality issues)
        s_level = self._get_sev_attr(severity, 'severity_level')
        s_realism = self._get_sev_attr(severity, 'realism_score', 0)
        if severity and s_level == "severe" and s_realism > 0.4 and conf >= 60:
            return "green"
        
        # Moderate severity + decent confidence -> confident amber/green
        if severity and s_level == "moderate" and s_realism > 0.4 and conf >= 65:
            return "green"
        
        if r.image_reliability == "low" and not (severity and s_level in ("moderate", "severe")):
            return "orange" if conf > 40 else "red"
        
        if conf < 55:
            return "orange" if conf > 40 else "red"
            
        if conf >= 80 and r.env_support in ["supportive", "moderate"]:
            return "green"
        
        if conf >= 60:
            return "amber"
            
        return "red"

    def _assemble_concise(self, fragments: List[str], r: Any, severity=None) -> str:
        if not fragments: return "Diagnosis based on balanced visual and sensor data."
        
        # Contradiction + low severity -> express strong skepticism
        if r.contradiction_level == "major" and r.severity_level in ("healthy_appearing", "mild"):
            return f"{fragments[0]} Environmental conditions and visible symptoms do not support rapid progression."
        
        if r.contradiction_level == "major":
            return f"{fragments[0]} However, current conditions are not ideal for its spread."
        
        if r.disease_gate == "failed":
            return f"{fragments[0]} The system has low certainty this represents actual disease."
            
        if len(fragments) >= 2:
            return f"{fragments[0]} {fragments[1]}"
            
        return fragments[0]

    def _generate_farmer_summary(self, label: str, reliability: str, r: Any, severity=None) -> str:
        disease = label.replace("_", " ")
        if "healthy" in label.lower():
            return f"Your crop appears healthy. Keep monitoring regularly."
        
        sev = r.severity_level
        
        # Healthy-appearing leaf
        if sev == "healthy_appearing":
            return f"The leaf looks mostly healthy. Very faint resemblance to {disease} detected - likely nothing to worry about yet."
        
        # Disease gate failed
        if r.disease_gate == "failed":
            return f"Weak visual indicators of {disease}. Not enough evidence for a confident diagnosis."
        
        # Major contradiction
        if r.contradiction_level == "major":
            if sev == "severe":
                return f"Significant symptoms of {disease} detected, although current environmental conditions are unfavorable for its rapid spread."
            if sev == "moderate":
                return f"Moderate signs of {disease} observed. Note: The environment currently seems to inhibit rapid progression."
            if sev == "mild":
                return f"Minor signs of {disease} detected, but conditions don't strongly support it. Keep watching."
            return f"Possible {disease} detected. Note: Environment does not fully support this spread."
        
        # Severity-driven summaries
        if sev == "severe":
            return f"Significant symptoms of {disease} detected. Conditions are {r.env_support}. Action may be needed."
        
        if sev == "moderate":
            return f"Moderate signs of {disease} observed. Monitor the plant and nearby leaves."
        
        if sev == "mild":
            return f"Mild symptoms suggesting {disease}. Continue observation - early intervention may help if symptoms grow."
            
        # Fallback to reliability-based
        if "High" in reliability:
            return f"Risk of {disease} detected. Conditions are favorable."
        
        return f"Potential signs of {disease} observed. Monitor closely."

    def _get_calibrated_recommendation(self, label: str, conf: float, r: Any, severity=None) -> str:
        if "healthy" in label.lower():
            return "Continue regular scouting and maintain current care."
        
        sev = r.severity_level
        
        recommendations = {
            "Tomato_Early_blight": {
                "severe": "Immediate fungal management is recommended. Remove heavily infected leaves and apply appropriate fungicide.",
                "moderate": "Monitor nearby leaves closely. If spots spread, consider organic copper spray as a preventive measure.",
                "mild": "Symptoms appear limited. Continue observation - take another photo in 2-3 days to track progression.",
                "healthy_appearing": "No clear disease symptoms visible. Ensure good airflow and drainage as a precaution."
            },
            "Tomato_Late_blight": {
                "severe": "Urgent: Apply protective fungicide and improve airflow immediately. Remove severely affected foliage.",
                "moderate": "Monitor weather closely. Avoid overhead watering and ensure plant spacing allows ventilation.",
                "mild": "Minor signs detected. Watch for spreading over the next few days before taking action.",
                "healthy_appearing": "Leaf appears healthy. If concerned, retake a clearer photo and recheck in a few days."
            }
        }
        
        category = recommendations.get(label, {})
        
        # Map severity level to recommendation tier
        if sev == "severe" and conf > 60 and r.contradiction_level != "major":
            tier = "severe"
        elif sev == "moderate" or (sev == "severe" and r.contradiction_level == "major"):
            tier = "moderate"
        elif sev == "mild" or r.disease_gate == "failed":
            tier = "mild"
        elif sev == "healthy_appearing":
            tier = "healthy_appearing"
        else:
            # Fallback to confidence-based
            tier = "severe" if conf > 80 and r.contradiction_level != "major" else ("moderate" if conf > 55 else "mild")
        
        return category.get(tier, "Consult an agricultural expert if symptoms persist.")

    def _get_warnings(self, r: Any, severity=None) -> List[str]:
        """
        Context-aware warnings (Part 6).
        
        Key improvement: warnings now distinguish between "image quality" 
        and "disease visibility". A noisy image with clear lesions generates
        a different warning than a noisy image with no symptoms.
        """
        warnings = []
        
        # Part 6 & 7: Context-aware image quality warning
        if r.image_reliability == "low":
            s_level = self._get_sev_attr(severity, 'severity_level')
            s_realism = self._get_sev_attr(severity, 'realism_score', 0)
            if severity and s_level in ("moderate", "severe") and s_realism > 0.4:
                warnings.append("Image has some noise or blur, but disease symptoms are still clearly visible.")
            else:
                warnings.append("Low image quality - try taking a clearer photo for better accuracy.")
        
        if r.contradiction_level == "major":
            warnings.append("Environmental mismatch - sensors don't fully support visual diagnosis.")
        if r.rainfall_effect == "high_risk":
            warnings.append("High moisture risk due to recent rainfall.")
        if r.disease_gate == "failed":
            warnings.append("Disease evidence is weak - the leaf may be healthy.")
        if r.severity_level == "healthy_appearing" and r.visual_strength in ("strong", "moderate"):
            warnings.append("CNN prediction appears overconfident - visible symptoms are minimal.")
        
        # Only warn about low realism when disease ALSO looks weak
        s_realism = self._get_sev_attr(severity, 'realism_score', 1.0)
        s_level = self._get_sev_attr(severity, 'severity_level')
        if severity and s_realism < 0.2:
            if s_level not in ("moderate", "severe"):
                warnings.append("Very low visual realism - prediction reliability is reduced.")
        
        return warnings
