import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("processed_survey_data.csv")

# 1. Full regression model with control variables
print("=======================================================")
print("  REGRESSION MODELS WITH COVARIATES / CONTROLS")
print("=======================================================")

# Model M with controls
m_ctrl = smf.ols("""
    M_mean ~ X_mean + W_mean + Privacy_Concern + Autonomy + Trust + 
             C(Q_Gender) + C(Q_Role) + C(Q_Exp) + C(Q_Sector)
""", data=data.rename(columns={
    "Which gender do you identify with?": "Q_Gender",
    "Which best describes your current role level?": "Q_Role",
    "How many years of work experience do you have?": "Q_Exp",
    "Which best describes your industry sector?": "Q_Sector"
})).fit()

print("\n--- Psychological Safety (M) with Controls ---")
print(m_ctrl.summary().tables[1])
print(f"R2 = {m_ctrl.rsquared:.4f}, Adj R2 = {m_ctrl.rsquared_adj:.4f}")

# 2. Breakdown by Role Level
print("\n=======================================================")
print("  BREAKDOWN BY ROLE LEVEL")
print("=======================================================")
roles = data["Which best describes your current role level?"].value_counts().index
for role in roles:
    sub = data[data["Which best describes your current role level?"] == role]
    m_h1_sub = smf.ols("M_mean ~ X_mean", data=sub).fit()
    m_w_sub = smf.ols("M_mean ~ W_mean", data=sub).fit()
    print(f"\nRole: {role} (N = {len(sub)})")
    print(f"  X_mean: {sub['X_mean'].mean():.2f}, W_mean: {sub['W_mean'].mean():.2f}, M_mean: {sub['M_mean'].mean():.2f}, Y1: {sub['Y1_mean'].mean():.2f}, Y2: {sub['Y2_mean'].mean():.2f}")
    print(f"  X -> M slope: {m_h1_sub.params['X_mean']:.4f} (p={m_h1_sub.pvalues['X_mean']:.4e})")
    print(f"  W -> M slope: {m_w_sub.params['W_mean']:.4f} (p={m_w_sub.pvalues['W_mean']:.4e})")

# 3. Breakdown by Industry Sector
print("\n=======================================================")
print("  BREAKDOWN BY INDUSTRY SECTOR")
print("=======================================================")
sectors = data["Which best describes your industry sector?"].value_counts().index
for sec in sectors:
    sub = data[data["Which best describes your industry sector?"] == sec]
    m_h1_sub = smf.ols("M_mean ~ X_mean", data=sub).fit()
    print(f"\nSector: {sec} (N = {len(sub)})")
    print(f"  X_mean: {sub['X_mean'].mean():.2f}, W_mean: {sub['W_mean'].mean():.2f}, M_mean: {sub['M_mean'].mean():.2f}, Y1: {sub['Y1_mean'].mean():.2f}, Y2: {sub['Y2_mean'].mean():.2f}")
    print(f"  X -> M slope: {m_h1_sub.params['X_mean']:.4f} (p={m_h1_sub.pvalues['X_mean']:.4e})")

# 4. Breakdown by Monitoring Type
print("\n=======================================================")
print("  BREAKDOWN BY MONITORING TYPE")
print("=======================================================")
mon_types = data["Which of the following best describes how your work is currently monitored by your organisation?"].value_counts().index
for mt in mon_types:
    sub = data[data["Which of the following best describes how your work is currently monitored by your organisation?"] == mt]
    print(f"\nType: {mt} (N = {len(sub)})")
    print(f"  X_mean: {sub['X_mean'].mean():.2f}, W_mean: {sub['W_mean'].mean():.2f}, M_mean: {sub['M_mean'].mean():.2f}, Y1: {sub['Y1_mean'].mean():.2f}, Y2: {sub['Y2_mean'].mean():.2f}")

# 5. Breakdown by Gender
print("\n=======================================================")
print("  BREAKDOWN BY GENDER")
print("=======================================================")
genders = ["Male", "Female"]
for gen in genders:
    sub = data[data["Which gender do you identify with?"] == gen]
    m_h1_sub = smf.ols("M_mean ~ X_mean", data=sub).fit()
    print(f"\nGender: {gen} (N = {len(sub)})")
    print(f"  X_mean: {sub['X_mean'].mean():.2f}, W_mean: {sub['W_mean'].mean():.2f}, M_mean: {sub['M_mean'].mean():.2f}, Y1: {sub['Y1_mean'].mean():.2f}, Y2: {sub['Y2_mean'].mean():.2f}")
    print(f"  X -> M slope: {m_h1_sub.params['X_mean']:.4f} (p={m_h1_sub.pvalues['X_mean']:.4e})")

# 6. Specific Monitoring Features impact on Psychological Safety & Engagement
print("\n=======================================================")
print("  CORRELATIONS OF SPECIFIC MONITORING PRACTICES WITH M AND ENGAGEMENT")
print("=======================================================")
x_cols = [
    ("Active Time Tracking", "My organisation tracks how much time I spend actively working._num"),
    ("Keystroke/Screen Logging", "My organisation uses software to record my digital activity, such as keystrokes, mouse movement, screen content or application use._num"),
    ("CCTV/Biometric Tracking", "My organisation uses cameras, biometric access or similar systems to record where I am during working hours._num"),
    ("Continuous Monitoring", "Monitoring at my workplace runs continuously rather than at occasional checkpoints._num"),
    ("Real-Time Availability", "Information about my work activity is available to my organisation in real time._num"),
    ("Beyond Job Requirements", "The monitoring systems at my workplace capture information that goes beyond what my job actually requires._num"),
    ("Intrusion into Breaks/Personal Time", "Monitoring at my workplace extends into periods I would consider personal, such as breaks or time after hours._num"),
    ("Automated Scoring/Ratings", "Automated systems, rather than a person, generate performance scores or ratings from my work activity._num"),
    ("Algorithmic Outlier Flagging", "Software at my workplace automatically flags employees whose activity falls outside an expected pattern._num")
]

for label, col in x_cols:
    r_m = data[col].corr(data["M_mean"])
    r_y1 = data[col].corr(data["Y1_mean"])
    r_y2 = data[col].corr(data["Y2_mean"])
    print(f"{label:35s} | r(M) = {r_m:+.3f} | r(Y1) = {r_y1:+.3f} | r(Y2) = {r_y2:+.3f}")
