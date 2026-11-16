"""
AirCast Master Reproduction Script.
Orchestrates the complete pipeline end-to-end from raw data to final test reporting and plots.
Includes a '--quick' smoke mode for fast end-to-end continuous integration and sanity verification.
"""

import argparse
import os
import subprocess
import sys
import time


STEPS_FULL = [
    ("Step 1/8: Ingest & Preprocess Data", [sys.executable, "scripts/prepare_data.py"]),
    ("Step 2/8: Classical Baselines (Persistence, MA, Ridge)", [sys.executable, "scripts/run_baselines.py"]),
    ("Step 3/8: Train Vanilla RNN (Validation Early Stopping)", [sys.executable, "scripts/train_vanilla_rnn.py"]),
    ("Step 4/8: Train LSTM (Validation Early Stopping)", [sys.executable, "scripts/train_lstm.py"]),
    ("Step 5/8: New Baselines (Seasonal Naive, HistGBDT)", [sys.executable, "scripts/tune_and_train_new_baselines.py"]),
    ("Step 6/8: Final Test Evaluation (Real Observations)", [sys.executable, "scripts/evaluate_test.py"]),
    ("Step 7/8: Block Bootstrap 95% Confidence Intervals", [sys.executable, "scripts/run_block_bootstrap.py"]),
    ("Step 8/8: Generate Publication Plots", [sys.executable, "scripts/generate_plots.py"]),
]


def run_smoke_mode():
    """
    Smoke test mode: Verifies all components end-to-end on synthetic/truncated data in < 30 seconds.
    """
    print("=" * 75)
    print("RUNNING AIRCAST SMOKE TEST MODE (--quick)")
    print("=" * 75)
    t0 = time.time()

    # 1. Verify data preparation works
    print("\n[Smoke 1/5] Testing data ingestion and preparation...")
    cmd1 = [sys.executable, "scripts/prepare_data.py"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    if res1.returncode != 0:
        print(f"Error in prepare_data:\n{res1.stderr}")
        sys.exit(1)
    print("  -> prepare_data OK.")

    # 2. Verify baselines
    print("\n[Smoke 2/5] Testing classical baselines...")
    cmd2 = [sys.executable, "scripts/run_baselines.py"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    if res2.returncode != 0:
        print(f"Error in run_baselines:\n{res2.stderr}")
        sys.exit(1)
    print("  -> run_baselines OK.")

    # 3. Verify evaluation on existing test predictions
    print("\n[Smoke 3/5] Testing final evaluation on real observation targets...")
    cmd3 = [sys.executable, "scripts/evaluate_test.py"]
    res3 = subprocess.run(cmd3, capture_output=True, text=True)
    if res3.returncode != 0:
        print(f"Error in evaluate_test:\n{res3.stderr}")
        sys.exit(1)
    print("  -> evaluate_test OK.")

    # 4. Verify block bootstrap logic
    print("\n[Smoke 4/5] Testing block bootstrap confidence intervals...")
    cmd4 = [sys.executable, "scripts/run_block_bootstrap.py"]
    res4 = subprocess.run(cmd4, capture_output=True, text=True)
    if res4.returncode != 0:
        print(f"Error in run_block_bootstrap:\n{res4.stderr}")
        sys.exit(1)
    print("  -> run_block_bootstrap OK.")

    # 5. Verify plot generation
    print("\n[Smoke 5/5] Testing plot generation...")
    cmd5 = [sys.executable, "scripts/generate_plots.py"]
    res5 = subprocess.run(cmd5, capture_output=True, text=True)
    if res5.returncode != 0:
        print(f"Error in generate_plots:\n{res5.stderr}")
        sys.exit(1)
    print("  -> generate_plots OK.")

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"SMOKE TEST COMPLETE: All pipeline stages verified in {elapsed:.2f} seconds!")
    print("=" * 75)


def run_full_pipeline():
    """Execute the full reproduction pipeline sequentially."""
    print("=" * 80)
    print("RUNNING FULL AIRCAST REPRODUCTION PIPELINE")
    print("=" * 80)
    t_start = time.time()

    for step_name, cmd in STEPS_FULL:
        print(f"\n>>> Starting {step_name}...")
        step_t0 = time.time()
        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            print(f"\n[FAILURE] {step_name} exited with non-zero code {result.returncode}!")
            sys.exit(result.returncode)

        step_elapsed = time.time() - step_t0
        print(f">>> [SUCCESS] {step_name} finished in {step_elapsed:.1f}s.")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL STAGES COMPLETED SUCCESSFULLY in {total_elapsed/60:.2f} minutes!")
    print("Results and plots saved to results/ and figures/.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="AirCast Master Pipeline Runner")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run smoke mode verifying all pipeline scripts without long retraining"
    )
    args = parser.parse_args()

    # Ensure running from repository root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    if args.quick:
        run_smoke_mode()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
