"""
Statistical Analysis for Cat Robot Drop Test Experiment
========================================================

This script performs comprehensive statistical analysis on the drop test data,
including:
1. Diagnostic checks: Does morphology systematically affect initial conditions?
2. Main analysis: Linear Mixed Models (LMM) to test morphology effect while
   controlling for initial condition variability
3. Visualization of results

Author: Kunwoo
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load data
df = pd.read_csv('/Users/kunwoomac/CodeSpace/MeowMentum/data_analysis/prelim/report_0423.csv')

print("="*80)
print("STATISTICAL ANALYSIS: Cat Robot Drop Test Experiment")
print("="*80)
print(f"\nDataset: {len(df)} drops across {df['morphology'].nunique()} morphologies")
print(f"Morphologies: {df['morphology'].unique()}")
print(f"Trials: {df['trial'].unique()}")
print(f"Reps per trial: {df.groupby(['morphology', 'trial'])['rep'].count().values}")

# Create average metrics from front (F) and back (B) measurements
# Use FK-based rear (B) orientation as it's the ground truth per your code
df['avg_roll_2p5m'] = (df['F_roll_2p5m'] + df['B_roll_2p5m']) / 2
df['avg_rollrate_2p5m'] = (df['F_rollrate_2p5m'] + df['B_rollrate_2p5m']) / 2
df['avg_pitch_2p5m'] = (df['F_pitch_2p5m'] + df['B_pitch_2p5m']) / 2
df['avg_pitchrate_2p5m'] = (df['F_pitchrate_2p5m'] + df['B_pitchrate_2p5m']) / 2

# Initial conditions (sanity checks from release)
df['avg_roll_initial'] = (df['F_roll_initial'] + df['B_roll_initial']) / 2
df['avg_rollrate_initial'] = (df['F_rollrate_initial'] + df['B_rollrate_initial']) / 2
df['avg_pitch_initial'] = (df['F_pitch_initial'] + df['B_pitch_initial']) / 2
df['avg_pitchrate_initial'] = (df['F_pitchrate_initial'] + df['B_pitchrate_initial']) / 2

print("\n" + "="*80)
print("STEP 1: DIAGNOSTIC CHECKS")
print("="*80)
print("\nQuestion: Does morphology systematically affect initial conditions?")
print("(If yes, we have a confound. If no, initial variation is random.)\n")

# Define initial conditions to check
initial_vars = [
    ('avg_roll_initial', 'Initial Roll Angle (deg)'),
    ('avg_pitch_initial', 'Initial Pitch Angle (deg)'),
    ('avg_rollrate_initial', 'Initial Roll Rate (deg/s)'),
    ('avg_pitchrate_initial', 'Initial Pitch Rate (deg/s)')
]

diagnostic_results = []

for var, label in initial_vars:
    print(f"\n{label}")
    print("-" * 60)
    
    # Descriptive stats by morphology
    desc = df.groupby('morphology')[var].agg(['mean', 'std', 'count'])
    print(desc)
    
    # One-way ANOVA: initial_condition ~ morphology
    groups = [group[var].values for name, group in df.groupby('morphology')]
    f_stat, p_val = stats.f_oneway(*groups)
    
    print(f"\nOne-Way ANOVA: F({len(groups)-1}, {len(df)-len(groups)}) = {f_stat:.3f}, p = {p_val:.4f}")
    
    if p_val < 0.05:
        print("⚠️  SIGNIFICANT: Initial conditions differ by morphology!")
        result = "CONFOUNDED"
    else:
        print("✅ NOT SIGNIFICANT: Initial conditions are random across morphologies")
        result = "RANDOM"
    
    diagnostic_results.append({
        'Variable': label,
        'F-statistic': f_stat,
        'p-value': p_val,
        'Result': result
    })

# Summary table
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
diag_df = pd.DataFrame(diagnostic_results)
print(diag_df.to_string(index=False))

confounded = any(r['Result'] == 'CONFOUNDED' for r in diagnostic_results)
if confounded:
    print("\n⚠️  INTERPRETATION: Some initial conditions differ by morphology.")
    print("   LMM will control for these differences, but interpretation is limited.")
    print("   The morphology effect may be partially due to different starting conditions.")
else:
    print("\n✅ INTERPRETATION: Initial conditions are random across morphologies.")
    print("   LMM will control for nuisance variability.")
    print("   Morphology effects can be interpreted causally.")


print("\n" + "="*80)
print("STEP 2: DESCRIPTIVE STATISTICS (Outcomes at 2.5m)")
print("="*80)

outcome_vars = [
    ('avg_roll_2p5m', 'Impact Roll Angle (deg)'),
    ('avg_rollrate_2p5m', 'Impact Roll Rate (deg/s)')
]

for var, label in outcome_vars:
    print(f"\n{label}")
    print("-" * 60)
    desc = df.groupby('morphology')[var].agg(['mean', 'std', 'count'])
    print(desc)


print("\n" + "="*80)
print("STEP 3: LINEAR MIXED MODEL ANALYSIS")
print("="*80)
print("\nModel: Outcome ~ Morphology + Initial_Roll + Initial_Pitch + Initial_Roll_Vel + Initial_Pitch_Vel")
print("This controls for initial condition variability while testing morphology effect.\n")

# We'll use OLS with robust standard errors since we don't have repeated measures on same robot
# (all drops are from the same robot, so no random effect needed)
# But we'll cluster standard errors by trial to account for within-trial correlation

lmm_results = []

for var, label in outcome_vars:
    print(f"\n{'='*80}")
    print(f"Outcome: {label}")
    print('='*80)
    
    # Fit model
    formula = f"{var} ~ C(morphology) + avg_roll_initial + avg_pitch_initial + avg_rollrate_initial + avg_pitchrate_initial"
    
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['trial']})
    
    print(model.summary())
    
    # Extract morphology effect
    morphology_coef = model.params['C(morphology)[T.spine-only]']
    morphology_pval = model.pvalues['C(morphology)[T.spine-only]']
    morphology_ci = model.conf_int().loc['C(morphology)[T.spine-only]']
    
    print(f"\n{'='*60}")
    print(f"MORPHOLOGY EFFECT SUMMARY")
    print('='*60)
    print(f"Coefficient (spine-only vs spine+tail): {morphology_coef:.2f}")
    print(f"95% CI: [{morphology_ci[0]:.2f}, {morphology_ci[1]:.2f}]")
    print(f"p-value: {morphology_pval:.4f}")
    
    if morphology_pval < 0.05:
        print(f"✅ SIGNIFICANT: Morphology affects {label} (p < 0.05)")
    else:
        print(f"⚠️  NOT SIGNIFICANT: No evidence of morphology effect (p ≥ 0.05)")
    
    lmm_results.append({
        'Outcome': label,
        'Coefficient': morphology_coef,
        'CI_lower': morphology_ci[0],
        'CI_upper': morphology_ci[1],
        'p-value': morphology_pval,
        'R-squared': model.rsquared
    })

print("\n" + "="*80)
print("STEP 4: ASSUMPTION CHECKS")
print("="*80)

# Check normality of residuals for each model
for var, label in outcome_vars:
    print(f"\n{label}")
    print("-" * 60)
    
    formula = f"{var} ~ C(morphology) + avg_roll_initial + avg_pitch_initial + avg_rollrate_initial + avg_pitchrate_initial"
    model = smf.ols(formula, data=df).fit()
    
    # Shapiro-Wilk test on residuals
    stat, p_val = stats.shapiro(model.resid)
    print(f"Shapiro-Wilk test for normality of residuals: W = {stat:.4f}, p = {p_val:.4f}")
    
    if p_val < 0.05:
        print("⚠️  Residuals are not normally distributed (consider non-parametric alternatives)")
    else:
        print("✅ Residuals are approximately normal")
    
    # Homoscedasticity check (Breusch-Pagan test)
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan test for homoscedasticity: LM = {bp_stat:.4f}, p = {bp_pval:.4f}")
    
    if bp_pval < 0.05:
        print("⚠️  Heteroscedasticity detected (robust standard errors recommended)")
    else:
        print("✅ Homoscedasticity assumption satisfied")


print("\n" + "="*80)
print("STEP 5: VISUALIZATION")
print("="*80)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Drop Test Analysis: Morphology Effect', fontsize=16, y=1.00)

# Row 1: Outcomes at 2.5m
for idx, (var, label) in enumerate(outcome_vars):
    ax = axes[0, idx]
    
    # Box plot
    df.boxplot(column=var, by='morphology', ax=ax)
    ax.set_title(label)
    ax.set_xlabel('Morphology')
    ax.set_ylabel(label)
    ax.get_figure().suptitle('')  # Remove default title
    
    # Add individual points
    for morph in df['morphology'].unique():
        data = df[df['morphology'] == morph][var]
        x = np.random.normal(list(df['morphology'].unique()).index(morph) + 1, 0.04, size=len(data))
        ax.plot(x, data, 'o', alpha=0.3, markersize=4)

# Row 1, Col 3: LMM effect sizes
ax = axes[0, 2]
lmm_df = pd.DataFrame(lmm_results)
y_pos = np.arange(len(lmm_df))
ax.barh(y_pos, lmm_df['Coefficient'], xerr=[lmm_df['Coefficient'] - lmm_df['CI_lower'], 
                                                lmm_df['CI_upper'] - lmm_df['Coefficient']], 
        alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(lmm_df['Outcome'])
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Coefficient (spine-only vs spine+tail)')
ax.set_title('LMM Effect Sizes (95% CI)')
ax.grid(axis='x', alpha=0.3)

# Row 2: Initial conditions diagnostic
for idx, (var, label) in enumerate(initial_vars[:3]):  # First 3 initial vars
    ax = axes[1, idx]
    
    df.boxplot(column=var, by='morphology', ax=ax)
    ax.set_title(f'Initial: {label}')
    ax.set_xlabel('Morphology')
    ax.set_ylabel(label)
    ax.get_figure().suptitle('')
    
    # Add individual points
    for morph in df['morphology'].unique():
        data = df[df['morphology'] == morph][var]
        x = np.random.normal(list(df['morphology'].unique()).index(morph) + 1, 0.04, size=len(data))
        ax.plot(x, data, 'o', alpha=0.3, markersize=4)

plt.tight_layout()
plt.savefig('/home/claude/statistical_analysis_summary.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved to: statistical_analysis_summary.png")

print("\n" + "="*80)
print("FINAL SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\n1. DIAGNOSTIC RESULTS:")
if confounded:
    print("   ⚠️  Initial conditions differ by morphology (confounded)")
    print("   → Interpretation: Morphology effect is partially due to different starting conditions")
    print("   → Recommendation: Report both raw differences AND LMM-controlled effects")
else:
    print("   ✅ Initial conditions are random (not confounded)")
    print("   → Interpretation: Morphology effect can be interpreted causally")
    print("   → Recommendation: Focus on LMM results as primary analysis")

print("\n2. MAIN FINDINGS:")
for result in lmm_results:
    sig = "SIGNIFICANT ✅" if result['p-value'] < 0.05 else "NOT SIGNIFICANT ⚠️"
    print(f"   {result['Outcome']}: {result['Coefficient']:.2f} [{result['CI_lower']:.2f}, {result['CI_upper']:.2f}], p={result['p-value']:.4f} → {sig}")

print("\n3. NEXT STEPS:")
print("   • If confounded: Consider restricting analysis to drops with similar initial conditions")
print("   • If not confounded: Proceed with LMM as primary analysis")
print("   • Consider non-parametric alternatives if normality assumptions violated")
print("   • Expand analysis to pitch outcomes and combined roll-pitch effects")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)