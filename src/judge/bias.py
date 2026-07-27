import numpy as np
from typing import List, Dict, Any
from src.judge.evaluator import LLMJudgeEvaluator

class BiasMitigationEngine:
    """
    Implements active detection, mitigation, and reporting for LLM Judge Biases:
    1. Position Bias (A/B order swap & flip rate)
    2. Verbosity Bias (Length correlation & padded response probe)
    3. Sycophancy / Style Bias (Per-criterion grounding & confidently-wrong probe)
    """

    def __init__(self, evaluator: LLMJudgeEvaluator):
        self.evaluator = evaluator

    def test_position_bias(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs each pairwise test case in BOTH orders:
        - Order 1: Output A first, Output B second
        - Order 2: Output B first, Output A second
        Calculates position Flip Rate and agreement rate.
        """
        pairwise_cases = [c for c in test_cases if "output_a" in c and "output_b" in c]
        if not pairwise_cases:
            return {
                "total_cases_evaluated": 0,
                "position_flips_count": 0,
                "flip_rate_percent": 0.0,
                "agreement_rate_percent": 100.0,
                "detailed_case_results": []
            }

        flips = 0
        total_cases = len(pairwise_cases)
        results = []

        for case in pairwise_cases:
            inp = case["input"]
            out_a = case["output_a"]
            out_b = case["output_b"]
            ctx = case.get("context", case.get("expected_output", ""))

            # Order 1: A vs B
            res_1 = self.evaluator.evaluate_pairwise(inp, out_a, out_b, context=ctx)
            win_1 = res_1.get("winner", "TIE")

            # Order 2: Swapped order (B vs A)
            res_2 = self.evaluator.evaluate_pairwise(inp, out_b, out_a, context=ctx)
            win_2_raw = res_2.get("winner", "TIE")

            # Map win_2_raw back to original A/B identifiers
            if win_2_raw == "A":
                win_2 = "B"
            elif win_2_raw == "B":
                win_2 = "A"
            else:
                win_2 = "TIE"

            is_flip = (win_1 != win_2) and (win_1 != "TIE") and (win_2 != "TIE")
            if is_flip:
                flips += 1

            results.append({
                "case_id": case.get("id", "unknown"),
                "order_1_winner": win_1,
                "order_2_winner": win_2,
                "consensus_winner": win_1 if win_1 == win_2 else "TIE",
                "position_flipped": is_flip
            })

        flip_rate = (flips / total_cases) * 100.0 if total_cases > 0 else 0.0
        agreement_rate = 100.0 - flip_rate

        return {
            "total_cases_evaluated": total_cases,
            "position_flips_count": flips,
            "flip_rate_percent": round(flip_rate, 2),
            "agreement_rate_percent": round(agreement_rate, 2),
            "detailed_case_results": results
        }

    def test_verbosity_bias(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Probes judge for verbosity bias by comparing standard answers with padded/fluffy answers.
        Computes score-length correlation.
        """
        pointwise_cases = [c for c in test_cases if "model_output" in c]
        if not pointwise_cases:
            return {
                "length_score_correlation": 0.0,
                "fluff_rewarded_rate_percent": 0.0,
                "verbosity_bias_detected": False,
                "padded_probe_results": []
            }

        scores = []
        lengths = []
        padded_probe_results = []

        for case in pointwise_cases:
            inp = case["input"]
            out = case["model_output"]

            eval_std = self.evaluator.evaluate_pointwise(inp, out, context=case.get("context"))
            score_std = eval_std.get("overall_score", 3.0)

            scores.append(score_std)
            lengths.append(len(out))

            # Create padded answer probe
            padded_out = out + "\n\nIn summary, furthermore, as previously noted, it is essential to emphasize that comprehensive detailed analysis remains crucial across all operational domains and secondary considerations." * 4
            eval_padded = self.evaluator.evaluate_pointwise(inp, padded_out, custom_criteria="Penalize unnecessary padding and ungrounded fluff.")
            score_padded = eval_padded.get("overall_score", 3.0)

            fluff_rewarded = score_padded > score_std

            padded_probe_results.append({
                "case_id": case.get("id", "unknown"),
                "standard_length": len(out),
                "standard_score": score_std,
                "padded_length": len(padded_out),
                "padded_score": score_padded,
                "fluff_rewarded": fluff_rewarded
            })

        if len(scores) > 1 and np.std(lengths) > 0 and np.std(scores) > 0:
            corr = float(np.corrcoef(lengths, scores)[0, 1])
        else:
            corr = 0.0

        rewarded_fluff_count = sum(1 for r in padded_probe_results if r["fluff_rewarded"])
        total_p = len(pointwise_cases)
        verbosity_bias_detected = rewarded_fluff_count > (total_p * 0.25) or corr > 0.4

        return {
            "length_score_correlation": round(corr, 3),
            "fluff_rewarded_rate_percent": round((rewarded_fluff_count / total_p) * 100.0, 2) if total_p > 0 else 0.0,
            "verbosity_bias_detected": verbosity_bias_detected,
            "padded_probe_results": padded_probe_results
        }
