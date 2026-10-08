import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

# Load dataset
df = pd.read_csv('survey_responses.csv')

print(f"Total respondents in dataset: {len(df)}")

# 1. Attention Check
attn_col = "To confirm that you are reading each question, please select Disagree for this item."
attn_counts = df[attn_col].value_counts()
print("\n--- Attention Check Distribution ---")
print(attn_counts)

passed_attn = df[df[attn_col] == "Disagree"].copy()
print(f"Passed attention check: {len(passed_attn)} ({len(passed_attn)/len(df)*100:.1f}%)")

# Let's inspect screening qualifier
screening_col = "Which of the following best describes how your work is currently monitored by your organisation?"
print("\n--- Screening Qualifier (Monitoring Type) ---")
print(passed_attn[screening_col].value_counts())

# Demographic breakdown
demo_cols = [
    "Which best describes your industry sector?",
    "How many years of work experience do you have?",
    "Approximately how large is your organisation?",
    "Which gender do you identify with?",
    "Which best describes your current role level?"
]

print("\n--- Demographic Summaries (Passed Attention Check) ---")
for col in demo_cols:
    print(f"\n{col}:")
    print(passed_attn[col].value_counts(normalize=True)*100)

# Likert mapping
likert_map = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neither Agree nor Disagree": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

freq_map = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}

# Construct item mappings
x_cols = [
    "My organisation tracks how much time I spend actively working.",
    "My organisation uses software to record my digital activity, such as keystrokes, mouse movement, screen content or application use.",
    "My organisation uses cameras, biometric access or similar systems to record where I am during working hours.",
    "Monitoring at my workplace runs continuously rather than at occasional checkpoints.",
    "Information about my work activity is available to my organisation in real time.",
    "The monitoring systems at my workplace capture information that goes beyond what my job actually requires.",
    "Monitoring at my workplace extends into periods I would consider personal, such as breaks or time after hours.",
    "Automated systems, rather than a person, generate performance scores or ratings from my work activity.",
    "Software at my workplace automatically flags employees whose activity falls outside an expected pattern."
]

w_cols = [
    "My organisation explains the procedures behind its monitoring systems thoroughly.",
    "The reasons my organisation gives for monitoring employees are reasonable.",
    "My organisation communicates details about monitoring in a timely manner.",
    "My organisation is candid and open when it communicates about monitoring.",
    "My organisation explains what its monitoring practices mean for someone in my specific role."
]

m_cols_raw = [
    "If I make a mistake at work, it is held against me.",
    "I am able to bring up problems and tough issues with the people I work with.",
    "People where I work sometimes reject others for being different.",
    "It is safe to take a risk at my workplace.",
    "It is difficult to ask the people I work with for help.",
    "No one at my workplace would deliberately act in a way that undermines my efforts.",
    "My unique skills and talents are valued and put to use at my workplace."
]

# In Edmondson (1999), items 1, 3, 5 are reverse-scored:
# Item 1: Mistake held against me (R)
# Item 3: Reject others for being different (R)
# Item 5: Difficult to ask for help (R)

y1_cols = [
    "I develop and make recommendations about issues that affect my work group.",
    "I speak up and encourage others in my work group to get involved in issues that affect the group.",
    "I communicate my opinions about work issues even when my view differs and others disagree.",
    "I keep myself well informed about issues where my opinion might be useful.",
    "I get involved in issues that affect the quality of working life in my group.",
    "I speak up with ideas for new projects or changes in procedure."
]

y2_cols = [
    "At my work, I feel bursting with energy.",
    "I am enthusiastic about my job.",
    "I am immersed in my work."
]

control_cols = [
    "It bothers me when organisations collect personal information about me.",
    "I prefer to decide for myself how I carry out my work.",
    "I generally assume organisations will handle employee information responsibly unless they give me reason to think otherwise."
]

# Convert dataset
data = passed_attn.copy()

for col in x_cols + w_cols + m_cols_raw + y1_cols + control_cols:
    data[col + "_num"] = data[col].map(likert_map)

for col in y2_cols:
    data[col + "_num"] = data[col].map(freq_map)

# Psychological safety reverse coding
# M1: mistake held against me -> 6 - val
data["M1_rev"] = 6 - data["If I make a mistake at work, it is held against me._num"]
data["M2"] = data["I am able to bring up problems and tough issues with the people I work with._num"]
data["M3_rev"] = 6 - data["People where I work sometimes reject others for being different._num"]
data["M4"] = data["It is safe to take a risk at my workplace._num"]
data["M5_rev"] = 6 - data["It is difficult to ask the people I work with for help._num"]
data["M6"] = data["No one at my workplace would deliberately act in a way that undermines my efforts._num"]
data["M7"] = data["My unique skills and talents are valued and put to use at my workplace._num"]

# Function for Cronbach's Alpha
def cronbach_alpha(df_items):
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    k = df_items.shape[1]
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

# Check reliability
x_df = data[[c + "_num" for c in x_cols]]
w_df = data[[c + "_num" for c in w_cols]]
m_df = data[["M1_rev", "M2", "M3_rev", "M4", "M5_rev", "M6", "M7"]]
m_df_unrev = data[[c + "_num" for c in m_cols_raw]]
y1_df = data[[c + "_num" for c in y1_cols]]
y2_df = data[[c + "_num" for c in y2_cols]]

print("\n--- Scale Reliabilities (Cronbach's Alpha) ---")
print(f"Monitoring Intensity (all 9 items): {cronbach_alpha(x_df):.3f}")
# Also check subset of 6 items from Ravid et al. (items 14-19 or 11,12,14,15,18,19)
x_6item = data[[x_cols[i] + "_num" for i in [0, 1, 3, 4, 7, 8]]] # 11, 12, 14, 15, 18, 19
print(f"Monitoring Intensity (core digital/AI 6-items): {cronbach_alpha(x_6item):.3f}")
ai_specific = data[[x_cols[i] + "_num" for i in [7, 8]]] # items 18, 19
print(f"AI/Automated specific items (items 18, 19 correlation): {ai_specific.corr().iloc[0,1]:.3f}")

print(f"Monitoring Transparency (5 items): {cronbach_alpha(w_df):.3f}")
print(f"Psychological Safety (7 items reverse-coded): {cronbach_alpha(m_df):.3f}")
print(f"Psychological Safety (7 items unreversed): {cronbach_alpha(m_df_unrev):.3f}")
print(f"Voice Behaviour (6 items): {cronbach_alpha(y1_df):.3f}")
print(f"Work Engagement (3 items): {cronbach_alpha(y2_df):.3f}")

# Construct scores
data["X_mean"] = x_df.mean(axis=1)
data["X_core"] = x_6item.mean(axis=1)
data["X_ai"] = ai_specific.mean(axis=1)
data["W_mean"] = w_df.mean(axis=1)
data["M_mean"] = m_df.mean(axis=1)
data["Y1_mean"] = y1_df.mean(axis=1)
data["Y2_mean"] = y2_df.mean(axis=1)

data["Privacy_Concern"] = data[control_cols[0] + "_num"]
data["Autonomy"] = data[control_cols[1] + "_num"]
data["Trust"] = data[control_cols[2] + "_num"]

# Summary statistics
print("\n--- Construct Descriptive Statistics ---")
desc = data[["X_mean", "X_core", "X_ai", "W_mean", "M_mean", "Y1_mean", "Y2_mean", "Privacy_Concern", "Autonomy", "Trust"]].describe()
print(desc.round(3))

# Correlation matrix
print("\n--- Pearson Correlation Matrix ---")
corr = data[["X_mean", "X_core", "X_ai", "W_mean", "M_mean", "Y1_mean", "Y2_mean", "Privacy_Concern", "Autonomy", "Trust"]].corr()
print(corr.round(3))

# Save processed dataframe for deeper analysis
data.to_csv("processed_survey_data.csv", index=False)
print("\nSaved processed_survey_data.csv successfully.")
