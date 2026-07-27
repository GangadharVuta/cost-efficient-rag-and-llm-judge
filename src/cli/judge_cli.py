import sys
import argparse
import json
from pathlib import Path
from src.judge.evaluator import LLMJudgeEvaluator
from src.judge.bias import BiasMitigationEngine
from src.judge.validator import JudgeValidator

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Judge CLI command")

    # Evaluate suite command
    eval_parser = subparsers.add_parser("eval-suite", help="Evaluate a test suite JSON file")
    eval_parser.add_argument("suite_path", type=str, help="Path to test suite JSON")
    eval_parser.add_argument("--mode", type=str, default="pointwise", choices=["pointwise", "pairwise"], help="Evaluation mode")

    # Bias test command
    bias_parser = subparsers.add_parser("test-bias", help="Run position and verbosity bias tests")
    bias_parser.add_argument("suite_path", type=str, help="Path to test suite JSON")

    # Validate judge command
    val_parser = subparsers.add_parser("validate-judge", help="Run human agreement, test-retest, and adversarial probes")
    val_parser.add_argument("suite_path", type=str, help="Path to evaluation dataset JSON")

    # A/B compare command
    ab_parser = subparsers.add_parser("ab-compare", help="Compare Prompt Config A vs Config B and declare winner")
    ab_parser.add_argument("suite_path", type=str, help="Path to pairwise test suite JSON")

    args = parser.parse_args()
    evaluator = LLMJudgeEvaluator()

    if args.command == "eval-suite":
        suite = json.loads(Path(args.suite_path).read_text(encoding="utf-8"))
        results = []
        for case in suite:
            if args.mode == "pointwise":
                res = evaluator.evaluate_pointwise(
                    input_prompt=case["input"],
                    model_output=case["model_output"],
                    expected_output=case.get("expected_output"),
                    context=case.get("context")
                )
            else:
                res = evaluator.evaluate_pairwise(
                    input_prompt=case["input"],
                    output_a=case["output_a"],
                    output_b=case["output_b"],
                    context=case.get("context")
                )
            results.append({"id": case.get("id"), "verdict": res})

        print("\n--- Suite Evaluation Report ---")
        print(json.dumps(results, indent=2))

    elif args.command == "test-bias":
        suite = json.loads(Path(args.suite_path).read_text(encoding="utf-8"))
        bias_engine = BiasMitigationEngine(evaluator)
        pos_results = bias_engine.test_position_bias(suite)
        verb_results = bias_engine.test_verbosity_bias(suite)

        print("\n=== Position Bias (A/B Order Swap) Report ===")
        print(f"Total Cases: {pos_results['total_cases_evaluated']}")
        print(f"Position Flips: {pos_results['position_flips_count']}")
        print(f"Flip Rate: {pos_results['flip_rate_percent']}%")
        print(f"Order Agreement Rate: {pos_results['agreement_rate_percent']}%")

        print("\n=== Verbosity Bias Report ===")
        print(f"Length-Score Correlation: {verb_results['length_score_correlation']}")
        print(f"Fluff Rewarded Rate: {verb_results['fluff_rewarded_rate_percent']}%")
        print(f"Verbosity Bias Detected: {verb_results['verbosity_bias_detected']}")

    elif args.command == "validate-judge":
        suite = json.loads(Path(args.suite_path).read_text(encoding="utf-8"))
        validator = JudgeValidator(evaluator)

        gold_cases = [c for c in suite if "gold_human_score" in c]
        adv_cases = [c for c in suite if "probe_type" in c]

        print("\n=== Judge Validation Artifact ===")
        if gold_cases:
            human_res = validator.validate_human_agreement(gold_cases)
            print(f"Human Gold Agreement Accuracy: {human_res['accuracy_percent']}%")
            print(f"Human-Judge Correlation: {human_res['correlation']}")

        consistency_res = validator.validate_test_retest_consistency(suite[:10])
        print(f"Test-Retest Stability Rate: {consistency_res['consistency_rate_percent']}%")

        if adv_cases:
            adv_res = validator.run_adversarial_probes(adv_cases)
            print(f"Adversarial Robustness Rate: {adv_res['judge_robustness_percent']}%")
            print(f"Times Judge Fooled: {adv_res['times_judge_fooled']} / {adv_res['total_adversarial_probes']}")

    elif args.command == "ab-compare":
        suite = json.loads(Path(args.suite_path).read_text(encoding="utf-8"))
        bias_engine = BiasMitigationEngine(evaluator)
        pos_results = bias_engine.test_position_bias(suite)

        wins_a = sum(1 for r in pos_results["detailed_case_results"] if r["consensus_winner"] == "A")
        wins_b = sum(1 for r in pos_results["detailed_case_results"] if r["consensus_winner"] == "B")
        ties = sum(1 for r in pos_results["detailed_case_results"] if r["consensus_winner"] == "TIE")

        winner = "CONFIG A" if wins_a > wins_b else ("CONFIG B" if wins_b > wins_a else "TIE (NO CLEAR WINNER)")

        print("\n=== A/B Configuration Comparison Report ===")
        print(f"Config A Wins: {wins_a}")
        print(f"Config B Wins: {wins_b}")
        print(f"Ties: {ties}")
        print(f"Position Flip Rate: {pos_results['flip_rate_percent']}%")
        print(f"DECLARED WINNER: {winner}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
