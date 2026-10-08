import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("processed_survey_data.csv")

# Filter only monitored respondents (or check both all vs monitored)
# In the synopsis screening routing: "respondents selecting 'Not monitored / not sure' are shown a polite exit screen... and do not proceed"
# In practice, let's see how many are explicitly monitored:
monitored = data[~data["Which of the following best describes how your work is currently monitored by your organisation?"].isin(["I am not sure", "I am not monitored in any way"])].copy()

print(f"Monitored sample size: {len(monitored)} (out of {len(data)})")

def run_analyses(df, label="Full Sample (Passed Attn Check)"):
    print(f"\n=======================================================")
    print(f"           ANALYSIS FOR: {label} (N = {len(df)})")
    print(f"=======================================================")
    
    # Standardize or center variables for moderation
    df['X_cent'] = df['X_mean'] - df['X_mean'].mean()
    df['W_cent'] = df['W_mean'] - df['W_mean'].mean()
    df['X_W_interaction'] = df['X_cent'] * df['W_cent']
    
    # 1. Test H1: X -> M
    m_h1 = smf.ols("M_mean ~ X_mean", data=df).fit()
    print("\n--- H1: Direct Effect of X on M (Psychological Safety) ---")
    print(f"Slope (b): {m_h1.params['X_mean']:.4f}, t = {m_h1.tvalues['X_mean']:.4f}, p = {m_h1.pvalues['X_mean']:.4e}, R2 = {m_h1.rsquared:.4f}")
    
    # 2. Test H2a: M -> Y1 (Voice) and H2b: M -> Y2 (Engagement)
    m_h2a = smf.ols("Y1_mean ~ M_mean", data=df).fit()
    print("\n--- H2a: Direct Effect of M on Y1 (Voice Behaviour) ---")
    print(f"Slope (b): {m_h2a.params['M_mean']:.4f}, t = {m_h2a.tvalues['M_mean']:.4f}, p = {m_h2a.pvalues['M_mean']:.4e}, R2 = {m_h2a.rsquared:.4f}")
    
    m_h2b = smf.ols("Y2_mean ~ M_mean", data=df).fit()
    print("\n--- H2b: Direct Effect of M on Y2 (Work Engagement) ---")
    print(f"Slope (b): {m_h2b.params['M_mean']:.4f}, t = {m_h2b.tvalues['M_mean']:.4f}, p = {m_h2b.pvalues['M_mean']:.4e}, R2 = {m_h2b.rsquared:.4f}")
    
    # 3. Test H4: Moderation of W on X -> M (First-stage moderation)
    m_h4 = smf.ols("M_mean ~ X_cent + W_cent + X_W_interaction", data=df).fit()
    print("\n--- H4: First-Stage Moderation (M ~ X + W + X*W) ---")
    print(m_h4.summary().tables[1])
    print(f"Model R2: {m_h4.rsquared:.4f}, F = {m_h4.fvalue:.4f}, p = {m_h4.f_pvalue:.4e}")
    
    # Simple Slopes Analysis at -1 SD, Mean, +1 SD of W
    w_sd = df['W_mean'].std()
    w_levels = {'Low (-1 SD)': -w_sd, 'Moderate (Mean)': 0.0, 'High (+1 SD)': w_sd}
    print("\n--- Simple Slopes of X -> M at Levels of Transparency (W) ---")
    for w_name, w_val in w_levels.items():
        # Conditional effect of X: b_X + b_int * w_val
        slope = m_h4.params['X_cent'] + m_h4.params['X_W_interaction'] * w_val
        # SE of conditional slope: sqrt(var(b_X) + w_val^2 * var(b_int) + 2 * w_val * cov(b_X, b_int))
        cov_matrix = m_h4.cov_params()
        var_slope = cov_matrix.loc['X_cent', 'X_cent'] + (w_val**2) * cov_matrix.loc['X_W_interaction', 'X_W_interaction'] + 2 * w_val * cov_matrix.loc['X_cent', 'X_W_interaction']
        se_slope = np.sqrt(var_slope)
        t_val = slope / se_slope
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_val), df=m_h4.df_resid))
        print(f"Transparency {w_name} ({w_val:+.3f}): Slope = {slope:.4f}, SE = {se_slope:.4f}, t = {t_val:.4f}, p = {p_val:.4e}")
        
    # 4. Test H5 & Mediation Models (Y1 and Y2 ~ X + M)
    m_y1_xm = smf.ols("Y1_mean ~ X_mean + M_mean", data=df).fit()
    print("\n--- Outcome Model Y1 (Voice ~ X + M) ---")
    print(m_y1_xm.summary().tables[1])
    print(f"R2 = {m_y1_xm.rsquared:.4f}")
    
    m_y2_xm = smf.ols("Y2_mean ~ X_mean + M_mean", data=df).fit()
    print("\n--- Outcome Model Y2 (Engagement ~ X + M) ---")
    print(m_y2_xm.summary().tables[1])
    print(f"R2 = {m_y2_xm.rsquared:.4f}")
    
    # 5. Bootstrapping for Moderated Mediation (Hayes Model 7)
    print("\n--- Running Bootstrap Moderated Mediation (5,000 resamples) ---")
    np.random.seed(42)
    n = len(df)
    n_boot = 5000
    
    boot_index_y1 = []
    boot_index_y2 = []
    boot_ind_low_y1, boot_ind_med_y1, boot_ind_hi_y1 = [], [], []
    boot_ind_low_y2, boot_ind_med_y2, boot_ind_hi_y2 = [], [], []
    
    X_c = df['X_cent'].values
    W_c = df['W_cent'].values
    XW = df['X_W_interaction'].values
    M = df['M_mean'].values
    Y1 = df['Y1_mean'].values
    Y2 = df['Y2_mean'].values
    
    for i in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        # Stage 1: M ~ X_c + W_c + XW
        X_mat1 = np.column_stack([np.ones(n), X_c[idx], W_c[idx], XW[idx]])
        beta_m = np.linalg.lstsq(X_mat1, M[idx], rcond=None)[0]
        a1, a3 = beta_m[1], beta_m[3]
        
        # Stage 2: Y1 ~ M + X_c
        X_mat2 = np.column_stack([np.ones(n), M[idx], X_c[idx]])
        beta_y1 = np.linalg.lstsq(X_mat2, Y1[idx], rcond=None)[0]
        b1_y1 = beta_y1[1]
        
        # Stage 2: Y2 ~ M + X_c
        beta_y2 = np.linalg.lstsq(X_mat2, Y2[idx], rcond=None)[0]
        b1_y2 = beta_y2[1]
        
        # Index of Moderated Mediation
        boot_index_y1.append(a3 * b1_y1)
        boot_index_y2.append(a3 * b1_y2)
        
        # Conditional indirect effects
        boot_ind_low_y1.append((a1 + a3 * (-w_sd)) * b1_y1)
        boot_ind_med_y1.append(a1 * b1_y1)
        boot_ind_hi_y1.append((a1 + a3 * w_sd) * b1_y1)
        
        boot_ind_low_y2.append((a1 + a3 * (-w_sd)) * b1_y2)
        boot_ind_med_y2.append(a1 * b1_y2)
        boot_ind_hi_y2.append((a1 + a3 * w_sd) * b1_y2)
        
    def ci_str(boot_arr):
        return f"Mean = {np.mean(boot_arr):.4f}, 95% CI [{np.percentile(boot_arr, 2.5):.4f}, {np.percentile(boot_arr, 97.5):.4f}]"
        
    print("\n[Outcome Y1: Voice Behaviour]")
    print(f"Index of Moderated Mediation: {ci_str(boot_index_y1)}")
    print(f"Conditional Indirect Effect at Low Transparency (-1 SD): {ci_str(boot_ind_low_y1)}")
    print(f"Conditional Indirect Effect at Moderate Transparency (Mean): {ci_str(boot_ind_med_y1)}")
    print(f"Conditional Indirect Effect at High Transparency (+1 SD): {ci_str(boot_ind_hi_y1)}")
    
    print("\n[Outcome Y2: Work Engagement]")
    print(f"Index of Moderated Mediation: {ci_str(boot_index_y2)}")
    print(f"Conditional Indirect Effect at Low Transparency (-1 SD): {ci_str(boot_ind_low_y2)}")
    print(f"Conditional Indirect Effect at Moderate Transparency (Mean): {ci_str(boot_ind_med_y2)}")
    print(f"Conditional Indirect Effect at High Transparency (+1 SD): {ci_str(boot_ind_hi_y2)}")

run_analyses(data, "All Qualified Respondents (N=981)")
run_analyses(monitored, "Actively Monitored Subsample (N=928)")
