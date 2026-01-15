import subprocess
import sys

def run_step(script_name):
    print(f"\nRunning {script_name}...")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"Error while running {script_name}")
        exit(1)


def main():
    run_step("preprocessing.py")
    run_step("train.py")
    run_step("evaluate.py")

    print("\n Full pipeline executed successfully!")

if __name__ == "__main__":
    main()