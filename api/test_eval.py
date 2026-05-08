import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from eval.harness import EvalHarness
from eval.test_cases import TEST_CASES

# Test with just 3 cases first (1 per category)
harness = EvalHarness()
summary = harness.run_full_eval(
    case_ids=["baseline_01", "ambiguous_01", "adversarial_01"],
    triggered_by="quick_test"
)

print("\nSCORES:")
for dim, score in summary["avg_scores"].items():
    print(f"  {dim:25s}: {score:.2f}")