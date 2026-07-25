import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each stage is run as its own separate process (not imported) - this
# matches exactly what happens when you run these files individually, so
# there's no separate "import version" of the code path to keep in sync.
STAGES = [
    ("Cleaning raw data -> final feature set", ROOT / "Data" / "clean_data.py"),
    ("Supervised learning (5 models + figures)", ROOT / "supervised_learning" / "supervised.py"),
    ("Unsupervised learning (PCA/K-Means/etc + figures)", ROOT / "unsupervised_learning" / "unsupervised.py"),
]


def run_stage(label, script_path):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    # cwd=script_path.parent means the script runs as if you'd opened a
    # terminal inside its own folder - matters because evaluation.py is
    # imported by name, not by full path, from inside supervised.py.
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent))

    if result.returncode != 0:
        print("\nStopped: {} exited with an error (code {}).".format(
            script_path.name, result.returncode))
        sys.exit(result.returncode)


def main():
    print("F1 Pit Stop Prediction - full pipeline")
    print("Raw data -> clean_data.py -> supervised.py -> unsupervised.py")

    for label, script_path in STAGES:
        run_stage(label, script_path)

    print("\n" + "=" * 60)
    print("Pipeline complete. Figures are in the output/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()