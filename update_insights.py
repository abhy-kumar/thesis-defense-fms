import sys
import os
import subprocess

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "survey_responses.csv"
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
        
    print(f"Processing survey dataset: {csv_file}")
    
    cmd_env = sys.executable
    subprocess.run([cmd_env, "run_full_analysis.py"], check=True)
    subprocess.run([cmd_env, "hypothesis_testing.py"], check=True)
    subprocess.run([cmd_env, "build_classic_academic_dashboard.py"], check=True)
    
    print("\nSuccessfully updated all statistical models, hypotheses tests, and dashboard.html!")
    print(f"Open 'd:/Dissertation/dashboard.html' in your web browser to view the dashboard.")

if __name__ == "__main__":
    main()
