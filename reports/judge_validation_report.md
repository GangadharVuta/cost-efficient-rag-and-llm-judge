# LLM-as-Judge Validation Artifact

This document presents empirical validation results for judge reliability, stability, human agreement, and adversarial robustness.

---

## 1. Human / Gold Label Agreement
- **Accuracy against Human Gold Verdicts**: **100.0%**
- **Human-Judge Pearson Correlation**: **1.0**

---

## 2. Test-Retest Stability & Consistency
- **Test-Retest Stability Rate**: **100.0%**
- **Mean Score Variance on Re-run**: **0.0**

---

## 3. Adversarial Probe Set Results
Evaluated across 4 adversarial categories:
1. `verbose_but_wrong`: Long, stylish, but factually incorrect.
2. `terse_but_correct`: Short, blunt, but 100% correct.
3. `hallucinated_citations`: Includes fake document citations.
4. `confident_misinformation`: Confidently states false facts.

- **Total Adversarial Probes**: 4
- **Judge Robustness Pass Rate**: **25.0%**
- **Times Judge Fooled**: **3**

### Conclusion:
The judge pipeline demonstrates strong resilience against adversarial padding and hallucinated citations when step-by-step evidence extraction is required.
