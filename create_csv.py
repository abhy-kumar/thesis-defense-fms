import json
import os
import re

transcript_path = r"C:\Users\abhik\.gemini\antigravity\brain\80c2f34a-dfcb-45d4-b9e4-c3dddb5646e5\.system_generated\logs\transcript_full.jsonl"
if not os.path.exists(transcript_path):
    transcript_path = r"C:\Users\abhik\.gemini\antigravity\brain\80c2f34a-dfcb-45d4-b9e4-c3dddb5646e5\.system_generated\logs\transcript.jsonl"

with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            content = data.get("content", "")
            if "Timestamp,Which of the following" in content:
                # Find start of CSV
                idx = content.find("Timestamp,Which of the following")
                csv_text = content[idx:].strip()
                # If there is trailing pdf markers like ==Start of PDF==
                if "==Start of PDF==" in csv_text:
                    csv_text = csv_text[:csv_text.find("==Start of PDF==")].strip()
                with open("survey_responses.csv", "w", encoding="utf-8") as out:
                    out.write(csv_text + "\n")
                print(f"Successfully extracted CSV with {len(csv_text.splitlines())} lines.")
                break
