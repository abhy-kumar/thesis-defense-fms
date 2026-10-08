# Employee Psychological Outcomes of AI-Enabled Workplace Monitoring

Faculty of Management Studies, University of Delhi, Delhi - 110007  
MBA Dissertation Research Project (Academic Year 2026-2027)  
Candidate: Abhishek Kumar (Roll No. FT-25-202) / Supervisor: Prof. Tanuja Agarwala  

## Research Overview

This repository contains the empirical dataset, statistical analysis pipelines, and the interactive research dashboard for the MBA dissertation examining the psychological and behavioural outcomes of electronic performance monitoring (EPM) in Indian knowledge-intensive sectors (Information Technology, ITES/BPO, Management Consulting, and Corporate Manufacturing).

The research tests a moderated mediation framework based on Hayes (2018) Model 7:
1. Direct effect of AI Monitoring Intensity (X) on Team Psychological Safety (M) [Hypothesis 1: Supported, b = -0.382, p < 0.001].
2. Direct effect of Psychological Safety (M) on Promotive Voice (Y1) [Hypothesis 2a: Supported, b = +0.532, p < 0.001].
3. Direct effect of Psychological Safety (M) on Work Engagement (Y2) [Hypothesis 2b: Supported, b = +0.638, p < 0.001].
4. Full mediation of Psychological Safety between Monitoring and Voice [Hypothesis 3a: Supported, direct effect c' = -0.001, p = 0.966; indirect effect = -0.183, p < 0.001].
5. Partial mediation of Psychological Safety between Monitoring and Engagement [Hypothesis 3b: Supported, direct effect c' = -0.114, p = 0.002; indirect effect = -0.213, p < 0.001].
6. Moderation of Monitoring Transparency (W) on the Monitoring to Safety path [Hypothesis 4: Not supported as an interaction buffer, b = -0.035, p = 0.276; however, Transparency exerts a large positive direct effect, b = +0.594, p < 0.001].

## Live Interactive Dashboard

The interactive dashboard is hosted via GitHub Pages at:  
https://abhy-kumar.github.io/thesis-defense-fms/

Key capabilities of the dashboard:
1. Dynamic Data Ingestion: Accepts any updated CSV file export matching the survey schema and recalculates all means, regressions, correlations, and conditional slopes in the browser in real time.
2. Demographic and Sector Filtering: Allows interactive filtering by industry sector, job role level, tenure, and gender.
3. APA Formatted Tables: Full regression output, simple slopes, 9-item monitoring indicator rankings, industry comparisons, and psychometric reliability matrices.
4. Spoken Plain-Language Guide: Clear, human-readable explanations of all statistical parameters.
5. Thesis Defense Ready Reckoner: Spoken-word answers to the 12 most probable viva and review questions.

## Repository Contents

1. index.html: Standalone dashboard application deployed to GitHub Pages root.
2. dashboard.html: Mirror copy of the academic dashboard interface.
3. survey_responses.csv: Primary raw survey dataset (N = 985 submissions, 981 verified via attention check).
4. processed_survey_data.csv: Processed dataset including compute composite construct scores.
5. hypothesis_testing.py: Python script estimating Hayes Model 7 regressions and bootstrap indirect effects.
6. run_full_analysis.py: Psychometric analysis script computing Cronbach alpha reliabilities, Harman single-factor CMB test, and item correlations.
7. additional_insights.py: Exploratory subgroup regressions across industries and role hierarchies.
8. build_classic_academic_dashboard.py: Reusable compiler script generating the standalone dashboard.
9. update_insights.py: End-to-end pipeline script to process fresh data exports and update the dashboard.

## Running Locally

To run the dashboard locally:
1. Simply double-click index.html or open it with any modern web browser. No server or build step is required.
2. To run the statistical Python models, install the requirements and run:
   python hypothesis_testing.py
