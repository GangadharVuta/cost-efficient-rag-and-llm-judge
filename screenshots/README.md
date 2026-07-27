# Screenshots Directory

This directory is designated for storing submission screenshots:

1. `01_rag_grounded_query.png`: Output of `python -m src.cli.rag_cli query "What primary factor drives the cost of fully managed vector databases?"` showing grounded answer, citations, and latency telemetry.
2. `02_rag_evaluation_harness.png`: Output of `python -m eval.eval_rag` showing Recall@4, MRR, nDCG@4, Context Precision, and latency metrics.
3. `03_judge_bias_validation.png`: Output of `python -m eval.eval_judge` showing position flip rate detection and adversarial probe results.
4. `04_judge_ab_comparison.png`: Output of `python -m src.cli.judge_cli ab-compare data/judge_eval_suite.json` showing A/B config comparison and declaring a winner.
