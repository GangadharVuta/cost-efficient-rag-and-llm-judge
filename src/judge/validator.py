import numpy as np
from typing import List, Dict, Any
from src.judge.evaluator import LLMJudgeEvaluator

class JudgeValidator:
    """
    Validates LLM Judge reliability through:
    1. Agreement with human/gold ground truth labels
    2. Test-Retest consistency on identical inputs
    3. Adversarial probe set evaluation (verbose-wrong, terse-correct, hallucinated citations)
    """

    def __init__(self, evaluator: LLMJudgeEvaluator):
        self.evaluator = evaluator

    def validate_human_agreement(self, dataset_with_gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates agreement rate and correlation between Judge scores and Human Gold labels."""
        gold_cases = [c for c in dataset_with_gold if "gold_human_score" in c and "model_output" in c]
        if not gold_cases:
            return {
                "total_test_cases": 0,
                "human_judge_agreements": 0,
                "accuracy_percent": 100.0,
                "correlation": 1.0
            }

        judge_scores = []
        human_scores = []
        agreements = 0

        for case in gold_cases:
            inp = case["input"]
            out = case["model_output"]
            gold_score = float(case["gold_human_score"])
            gold_pass = case.get("gold_pass_verdict", gold_score >= 3.5)

            verdict = self.evaluator.evaluate_pointwise(inp, out, context=case.get("context"))
            j_score = verdict.get("overall_score", 3.0)
            j_pass = verdict.get("pass_verdict", j_score >= 3.5)

            judge_scores.append(j_score)
            human_scores.append(gold_score)

            if j_pass == gold_pass:
                agreements += 1

        total = len(gold_cases)
        agreement_rate = (agreements / total) * 100.0 if total > 0 else 0.0

        if total > 1 and np.std(judge_scores) > 0 and np.std(human_scores) > 0:
            corr = float(np.corrcoef(judge_scores, human_scores)[0, 1])
        else:
            corr = 1.0

        return {
            "total_test_cases": total,
            "human_judge_agreements": agreements,
            "accuracy_percent": round(agreement_rate, 2),
            "correlation": round(corr, 3)
        }

    def validate_test_retest_consistency(self, test_suite: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs identical test suite twice to evaluate test-retest verdict stability."""
        pointwise_cases = [c for c in test_suite if "model_output" in c]
        if not pointwise_cases:
            return {
                "total_cases_tested": 0,
                "identical_verdicts_count": 0,
                "consistency_rate_percent": 100.0,
                "mean_score_variance": 0.0
            }

        matches = 0
        score_diffs = []
        total = len(pointwise_cases)

        for case in pointwise_cases:
            inp = case["input"]
            out = case["model_output"]
            ctx = case.get("context")

            v1 = self.evaluator.evaluate_pointwise(inp, out, context=ctx)
            v2 = self.evaluator.evaluate_pointwise(inp, out, context=ctx)

            s1 = v1.get("overall_score", 3.0)
            s2 = v2.get("overall_score", 3.0)
            diff = abs(s1 - s2)
            score_diffs.append(diff)

            p1 = v1.get("pass_verdict")
            p2 = v2.get("pass_verdict")

            if p1 == p2 and diff <= 0.2:
                matches += 1

        consistency_rate = (matches / total) * 100.0 if total > 0 else 0.0
        mean_diff = float(np.mean(score_diffs)) if score_diffs else 0.0

        return {
            "total_cases_tested": total,
            "identical_verdicts_count": matches,
            "consistency_rate_percent": round(consistency_rate, 2),
            "mean_score_variance": round(mean_diff, 4)
        }

    def run_adversarial_probes(self, adversarial_suite: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tests judge against 4 adversarial probes:
        1. verbose_but_wrong
        2. terse_but_correct
        3. hallucinated_citations
        4. confident_misinformation
        """
        probe_cases = [c for c in adversarial_suite if "probe_type" in c]
        if not probe_cases:
            return {
                "total_adversarial_probes": 0,
                "times_judge_fooled": 0,
                "judge_robustness_percent": 100.0,
                "probe_details": []
            }

        probe_results = []
        fooled_count = 0
        total_probes = len(probe_cases)

        for probe in probe_cases:
            probe_type = probe["probe_type"]
            expected_pass = probe["expected_pass"]

            verdict = self.evaluator.evaluate_pointwise(
                input_prompt=probe["input"],
                model_output=probe["model_output"],
                context=probe.get("context")
            )

            actual_pass = verdict.get("pass_verdict")
            is_fooled = (actual_pass != expected_pass)

            if is_fooled:
                fooled_count += 1

            probe_results.append({
                "probe_id": probe.get("id", "unknown"),
                "probe_type": probe_type,
                "expected_pass": expected_pass,
                "judge_pass": actual_pass,
                "judge_score": verdict.get("overall_score"),
                "judge_fooled": is_fooled,
                "rationale_snippet": verdict.get("rationale")[:150]
            })

        robustness_rate = ((total_probes - fooled_count) / total_probes) * 100.0 if total_probes > 0 else 0.0

        return {
            "total_adversarial_probes": total_probes,
            "times_judge_fooled": fooled_count,
            "judge_robustness_percent": round(robustness_rate, 2),
            "probe_details": probe_results
        }
