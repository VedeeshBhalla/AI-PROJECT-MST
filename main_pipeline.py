"""Run the reproducible experiment from the repository directory."""
import os
import subprocess
import sys
import config


def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for script in ("step1_eda_preprocessing.py", "step2_feature_engineering.py",
                   "step3_train_models.py", "step4_evaluate_report.py"):
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(config.BASE_DIR / script)],
                       cwd=config.BASE_DIR, env=env, check=True)
    print(f"Complete. Reports: {config.REPORT_DIR}")


if __name__ == "__main__":
    main()
