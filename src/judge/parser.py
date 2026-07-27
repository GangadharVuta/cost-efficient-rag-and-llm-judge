import re
import json
from typing import Dict, Any, Optional

class StructuredVerdictParser:
    """
    Robustly parses and repairs raw LLM judge text into a structured dictionary.
    Handles triple-backtick markdown blocks, trailing commas, missing quotes,
    regex JSON extraction, and schema auto-repair.
    """

    @staticmethod
    def parse(raw_text: str, default_mode: str = "pointwise") -> Dict[str, Any]:
        if not raw_text or not raw_text.strip():
            return StructuredVerdictParser._build_fallback(default_mode, "Empty response from judge.")

        cleaned = raw_text.strip()

        # Step 1: Strip markdown code fence syntax ```json ... ```
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
        if json_match:
            cleaned = json_match.group(1).strip()

        # Step 2: Extract first `{ ... }` substring
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            brace_match = re.search(r'(\{[\s\S]*\})', cleaned)
            if brace_match:
                cleaned = brace_match.group(1).strip()

        # Step 3: Attempt direct json.loads
        try:
            parsed = json.loads(cleaned)
            return StructuredVerdictParser._validate_and_fix_schema(parsed, default_mode)
        except json.JSONDecodeError:
            pass

        # Step 4: Fix trailing commas and unquoted key repairs
        repaired = re.sub(r',\s*([\}\]])', r'\1', cleaned) # Remove trailing commas
        repaired = re.sub(r'(?<=[\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', repaired) # Quote unquoted keys

        try:
            parsed = json.loads(repaired)
            return StructuredVerdictParser._validate_and_fix_schema(parsed, default_mode)
        except json.JSONDecodeError:
            pass

        # Step 5: Heuristic extraction if parsing completely failed
        return StructuredVerdictParser._extract_heuristically(raw_text, default_mode)

    @staticmethod
    def _validate_and_fix_schema(data: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Ensures all required fields exist and scores are within 1.0 - 5.0 range."""
        if not isinstance(data, dict):
            return StructuredVerdictParser._build_fallback(mode, "Parsed data is not a JSON object.")

        if mode == "pointwise":
            scores = data.get("criteria_scores", {})
            if not isinstance(scores, dict):
                scores = {}

            req_criteria = ["correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety"]
            valid_scores = {}
            score_list = []

            for c in req_criteria:
                val = scores.get(c, 3.0)
                try:
                    val = float(val)
                    val = max(1.0, min(5.0, val))
                except (ValueError, TypeError):
                    val = 3.0
                valid_scores[c] = val
                score_list.append(val)

            overall = data.get("overall_score")
            try:
                overall = float(overall) if overall is not None else sum(score_list) / len(score_list)
            except (ValueError, TypeError):
                overall = sum(score_list) / len(score_list)

            return {
                "criteria_scores": valid_scores,
                "rationale": str(data.get("rationale", "No rationale provided.")),
                "overall_score": round(overall, 2),
                "pass_verdict": data.get("pass_verdict", overall >= 3.5)
            }

        elif mode == "pairwise":
            winner = str(data.get("winner", "TIE")).upper()
            if winner not in ["A", "B", "TIE"]:
                if "MODEL A" in winner or "OUTPUT A" in winner:
                    winner = "A"
                elif "MODEL B" in winner or "OUTPUT B" in winner:
                    winner = "B"
                else:
                    winner = "TIE"

            return {
                "winner": winner,
                "confidence": float(data.get("confidence", 0.8)),
                "rationale": str(data.get("rationale", "No rationale provided.")),
                "criteria_comparison": data.get("criteria_comparison", {})
            }

        return data

    @staticmethod
    def _extract_heuristically(text: str, mode: str) -> Dict[str, Any]:
        """Regex-based fallback extraction when JSON parsing fails."""
        if mode == "pairwise":
            w_match = re.search(r'winner["\s:]+([AB]|TIE)', text, re.IGNORECASE)
            winner = w_match.group(1).upper() if w_match else "TIE"
            return {
                "winner": winner,
                "confidence": 0.5,
                "rationale": f"Extracted via heuristic fallback from text: {text[:200]}...",
                "criteria_comparison": {}
            }

        scores = {}
        for c in ["correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety"]:
            m = re.search(rf'{c}["\s:]+([1-5](?:\.\d+)?)', text, re.IGNORECASE)
            scores[c] = float(m.group(1)) if m else 3.0

        overall = sum(scores.values()) / len(scores)
        return {
            "criteria_scores": scores,
            "rationale": f"Regex heuristic parsing used on text: {text[:200]}...",
            "overall_score": round(overall, 2),
            "pass_verdict": overall >= 3.5
        }

    @staticmethod
    def _build_fallback(mode: str, reason: str) -> Dict[str, Any]:
        if mode == "pairwise":
            return {
                "winner": "TIE",
                "confidence": 0.0,
                "rationale": f"Fallback due to parsing error: {reason}",
                "criteria_comparison": {}
            }

        default_scores = {c: 3.0 for c in ["correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety"]}
        return {
            "criteria_scores": default_scores,
            "rationale": f"Fallback due to parsing error: {reason}",
            "overall_score": 3.0,
            "pass_verdict": False
        }
