"""
Distribution-Based Feature Selection for Phishing Detection
Selects features based on how well they discriminate between classes
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("DISTRIBUTION-BASED FEATURE SELECTION")
print("="*80)
print("\nThis script selects features based on distribution differences")
print("between phishing and legitimate senders, not correlations.\n")

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("📂 Loading data...")
df = pd.read_csv('notebooks/sender_temporal_features_final.csv')

# Separate features and labels
feature_cols = [col for col in df.columns if col not in ['sender', 'label']]
X = df[feature_cols]
y = df['label']

# Separate phishing and legitimate
phishing_data = X[y == 1]
legitimate_data = X[y == 0]

print(f"✓ Loaded {len(df):,} senders")
print(f"✓ Phishing: {len(phishing_data):,} ({len(phishing_data)/len(df)*100:.1f}%)")
print(f"✓ Legitimate: {len(legitimate_data):,} ({len(legitimate_data)/len(df)*100:.1f}%)")
print(f"✓ Features to evaluate: {len(feature_cols)}")

# ============================================================================
# STEP 2: Define Distribution Metrics Functions
# ============================================================================

def calculate_distribution_metrics(feature_name, phishing_data, legitimate_data):
    """
    Calculate comprehensive distribution difference metrics
    """
    phish = phishing_data[feature_name].dropna()
    legit = legitimate_data[feature_name].dropna()
    
    # Skip if not enough data
    if len(phish) < 10 or len(legit) < 10:
        return None
    
    metrics = {}
    
    # 1. Kolmogorov-Smirnov Test (distribution difference)
    try:
        ks_stat, ks_pval = stats.ks_2samp(phish, legit)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pval
    except:
        metrics['ks_statistic'] = 0
        metrics['ks_pvalue'] = 1
    
    # 2. Mann-Whitney U Test (median difference)
    try:
        mw_stat, mw_pval = stats.mannwhitneyu(phish, legit, alternative='two-sided')
        metrics['mw_statistic'] = mw_stat
        metrics['mw_pvalue'] = mw_pval
    except:
        metrics['mw_statistic'] = 0
        metrics['mw_pvalue'] = 1
    
    # 3. Cohen's D (effect size)
    try:
        pooled_std = np.sqrt((phish.std()**2 + legit.std()**2) / 2)
        if pooled_std > 0:
            metrics['cohens_d'] = abs((phish.mean() - legit.mean()) / pooled_std)
        else:
            metrics['cohens_d'] = 0
    except:
        metrics['cohens_d'] = 0
    
    # 4. Distribution Overlap (using histogram overlap)
    try:
        # Create normalized histograms
        combined = pd.concat([phish, legit])
        bins = np.histogram_bin_edges(combined, bins=30)
        hist_phish, _ = np.histogram(phish, bins=bins, density=True)
        hist_legit, _ = np.histogram(legit, bins=bins, density=True)
        
        # Normalize to probability distributions
        hist_phish = hist_phish / hist_phish.sum()
        hist_legit = hist_legit / hist_legit.sum()
        
        # Calculate overlap (lower is better)
        overlap = np.minimum(hist_phish, hist_legit).sum()
        metrics['distribution_overlap'] = overlap
    except:
        metrics['distribution_overlap'] = 1.0
    
    # 5. Jensen-Shannon Divergence (distribution similarity)
    try:
        js_divergence = jensenshannon(hist_phish, hist_legit)
        metrics['js_divergence'] = js_divergence
    except:
        metrics['js_divergence'] = 0
    
    # 6. Means and Medians
    metrics['phish_mean'] = phish.mean()
    metrics['legit_mean'] = legit.mean()
    metrics['phish_median'] = phish.median()
    metrics['legit_median'] = legit.median()
    
    # 7. Percentile differences
    metrics['phish_p25'] = phish.quantile(0.25)
    metrics['phish_p75'] = phish.quantile(0.75)
    metrics['legit_p25'] = legit.quantile(0.25)
    metrics['legit_p75'] = legit.quantile(0.75)
    
    # 8. Variance ratio
    if legit.var() > 0:
        metrics['variance_ratio'] = phish.var() / legit.var()
    else:
        metrics['variance_ratio'] = 1
    
    return metrics

def calculate_discrimination_score(metrics):
    """
    Calculate composite discrimination score
    """
    if not metrics:
        return 0
    
    # Weight different aspects
    score = 0
    
    # KS statistic (0-1, higher is better)
    score += metrics['ks_statistic'] * 30
    
    # Cohen's D (effect size, higher is better)
    score += min(metrics['cohens_d'], 3) * 25  # Cap at 3
    
    # Distribution overlap (0-1, lower is better)
    score += (1 - metrics['distribution_overlap']) * 20
    
    # JS divergence (0-1, higher is better)
    score += metrics['js_divergence'] * 15
    
    # Statistical significance (convert p-value)
    if metrics['ks_pvalue'] > 0 and metrics['ks_pvalue'] < 1:
        score += min(-np.log10(metrics['ks_pvalue']), 10) * 10  # Cap at 10
    
    return score

# ============================================================================
# STEP 3: Evaluate All Features
# ============================================================================
print("\n" + "="*80)
print("STEP 1: EVALUATING DISTRIBUTION DIFFERENCES")
print("="*80)

feature_metrics = {}
for feature in feature_cols:
    print(f"Evaluating: {feature}...", end='\r')
    metrics = calculate_distribution_metrics(feature, phishing_data, legitimate_data)
    if metrics:
        metrics['discrimination_score'] = calculate_discrimination_score(metrics)
        feature_metrics[feature] = metrics

print(" " * 50, end='\r')  # Clear line

# Create summary dataframe
summary_data = []
for feature, metrics in feature_metrics.items():
    summary_data.append({
        'Feature': feature,
        'KS_Statistic': metrics['ks_statistic'],
        'Cohens_D': metrics['cohens_d'],
        'Overlap': metrics['distribution_overlap'],
        'JS_Divergence': metrics['js_divergence'],
        'P_Value': metrics['ks_pvalue'],
        'Score': metrics['discrimination_score']
    })

summary_df = pd.DataFrame(summary_data).sort_values('Score', ascending=False)

print("\n📊 Top 15 Features by Discrimination Power:")
print("="*90)
print(f"{'Rank':<6} {'Feature':<30} {'KS Stat':<10} {'Cohen D':<10} {'Overlap':<10} {'Score':<10}")
print("-"*90)

for i, row in enumerate(summary_df.head(15).itertuples(), 1):
    sig = "***" if row.P_Value < 0.001 else "**" if row.P_Value < 0.01 else "*" if row.P_Value < 0.05 else ""
    print(f"{i:<6} {row.Feature:<30} {row.KS_Statistic:<10.3f} {row.Cohens_D:<10.3f} "
          f"{row.Overlap:<10.3f} {row.Score:<10.2f} {sig}")

# ============================================================================
# STEP 4: Group Features and Select Best Discriminators
# ============================================================================
print("\n" + "="*80)
print("STEP 2: SELECTING BEST DISCRIMINATOR FROM EACH GROUP")
print("="*80)

# Define feature groups (adjust based on your features)
feature_groups = {
    'entropy_features': ['hour_entropy', 'time_consistency_score'],
    'diversity_features': ['unique_hours', 'unique_days'],
    'concentration_features': ['concentration_1hr', 'activity_concentration', 
                               'percent_emails_in_first_hour', 'campaign_intensity'],
    'burst_features': ['rapid_succession_ratio', 'avg_burst_size', 'mean_velocity'],
    'frequency_features': ['emails_per_minute_p90', 'emails_per_minute_p95', 
                           'emails_per_minute_p99', 'emails_per_5min_p90', 
                           'emails_per_5min_p95', 'emails_per_5min_median',
                           'emails_per_15min_p90', 'emails_per_15min_p95'],
    'session_features': ['single_session', 'front_loading_score', 
                         'inter_email_cv', 'session_length_cv'],
    'pattern_features': ['day_entropy']
}

selected_features = {}
selection_details = {}

for group_name, features in feature_groups.items():
    # Filter to features that exist
    available_features = [f for f in features if f in feature_metrics]
    
    if not available_features:
        continue
    
    print(f"\n{group_name.upper()}:")
    print("-"*60)
    
    # Find best discriminator in group
    best_feature = None
    best_score = -1
    
    group_scores = []
    for feature in available_features:
        score = feature_metrics[feature]['discrimination_score']
        ks_stat = feature_metrics[feature]['ks_statistic']
        cohens_d = feature_metrics[feature]['cohens_d']
        overlap = feature_metrics[feature]['distribution_overlap']
        
        group_scores.append((feature, score, ks_stat, cohens_d, overlap))
        
        if score > best_score:
            best_score = score
            best_feature = feature
    
    # Sort by score
    group_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Display comparison
    for feature, score, ks, cd, overlap in group_scores[:3]:  # Show top 3
        selected = "✓" if feature == best_feature else " "
        print(f"  {selected} {feature:<35} Score: {score:>6.2f}  KS: {ks:.3f}  CD: {cd:.3f}")
    
    if best_feature:
        selected_features[group_name] = best_feature
        selection_details[best_feature] = {
            'group': group_name,
            'score': best_score,
            'metrics': feature_metrics[best_feature]
        }
        print(f"\n  → Selected: {best_feature} (Score: {best_score:.2f})")

# ============================================================================
# STEP 5: Visualize Distribution Differences
# ============================================================================
print("\n" + "="*80)
print("STEP 3: VISUALIZING DISTRIBUTION DIFFERENCES")
print("="*80)

# Create visualization for selected features
n_selected = len(selected_features)
fig, axes = plt.subplots(n_selected, 2, figsize=(14, 4*n_selected))

if n_selected == 1:
    axes = axes.reshape(1, -1)

for idx, (group, feature) in enumerate(selected_features.items()):
    phish = phishing_data[feature].dropna()
    legit = legitimate_data[feature].dropna()
    
    # Density plot
    ax1 = axes[idx, 0]
    ax1.set_title(f'{feature} - Density Plot', fontweight='bold')
    
    # Plot densities
    phish_density = phish.plot.density(ax=ax1, color='red', label='Phishing', alpha=0.6)
    legit_density = legit.plot.density(ax=ax1, color='green', label='Legitimate', alpha=0.6)
    
    # Add means
    ax1.axvline(phish.mean(), color='darkred', linestyle='--', alpha=0.8, label=f'Phish μ={phish.mean():.2f}')
    ax1.axvline(legit.mean(), color='darkgreen', linestyle='--', alpha=0.8, label=f'Legit μ={legit.mean():.2f}')
    
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[idx, 1]
    ax2.set_title(f'{feature} - Box Plot Comparison', fontweight='bold')
    
    box_data = pd.DataFrame({
        'Phishing': phish,
        'Legitimate': legit
    })
    box_data.boxplot(ax=ax2, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    # Add metrics text
    metrics = feature_metrics[feature]
    stats_text = f"KS: {metrics['ks_statistic']:.3f}\n"
    stats_text += f"Cohen's D: {metrics['cohens_d']:.2f}\n"
    stats_text += f"Overlap: {metrics['distribution_overlap']:.3f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_ylabel(feature)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Distribution plots saved to 'notebooks/distribution_comparison.png'")

# ============================================================================
# STEP 6: Check for Complementary Discrimination
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CHECKING COMPLEMENTARY DISCRIMINATION PATTERNS")
print("="*80)

# Check if features identify different subsets of phishing
selected_feature_list = list(selected_features.values())
n_features = len(selected_feature_list)

if n_features > 1:
    # Create binary classification for each feature (above/below median)
    phishing_patterns = pd.DataFrame()
    
    for feature in selected_feature_list:
        threshold = phishing_data[feature].median()
        phishing_patterns[f'{feature}_high'] = (phishing_data[feature] > threshold).astype(int)
    
    # Check overlap in detection
    pattern_corr = phishing_patterns.corr()
    
    print("\n📊 Detection Pattern Overlap (lower = more complementary):")
    print("(Shows correlation between features' ability to identify phishing)\n")
    
    # Find features that identify different phishing subsets
    complementary_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr = pattern_corr.iloc[i, j]
            if abs(corr) < 0.5:  # Low correlation = complementary
                complementary_pairs.append((
                    selected_feature_list[i], 
                    selected_feature_list[j], 
                    corr
                ))
    
    if complementary_pairs:
        print("✅ Complementary Feature Pairs (identify different phishing patterns):")
        for feat1, feat2, corr in sorted(complementary_pairs, key=lambda x: abs(x[2])):
            print(f"   {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("⚠️  Features may be identifying similar phishing patterns")

# ============================================================================
# STEP 7: Final Feature Selection
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FINAL FEATURE SELECTION")
print("="*80)

# Option 1: Use selected best from each group
final_features_grouped = selected_feature_list

# Option 2: Use top N by discrimination score regardless of groups
top_n = 8
final_features_top = summary_df.head(top_n)['Feature'].tolist()

print("\n🎯 RECOMMENDATION 1: Best from Each Group (Diverse)")
print("-"*60)
for i, feature in enumerate(final_features_grouped, 1):
    metrics = feature_metrics[feature]
    print(f"{i}. {feature:<30} (Score: {metrics['discrimination_score']:.2f})")

print(f"\nTotal: {len(final_features_grouped)} features")

print("\n🎯 RECOMMENDATION 2: Top {top_n} by Discrimination (Optimal)")
print("-"*60)
for i, feature in enumerate(final_features_top, 1):
    metrics = feature_metrics[feature]
    group = next((g for g, f in selected_features.items() if f == feature), "other")
    print(f"{i}. {feature:<30} (Score: {metrics['discrimination_score']:.2f}, Group: {group})")

# ============================================================================
# STEP 8: Export Final Selection
# ============================================================================
print("\n" + "="*80)
print("STEP 6: EXPORTING RESULTS")
print("="*80)

# Create final dataset with selected features
final_features = final_features_grouped  # or final_features_top

final_df = df[['sender'] + final_features + ['label']]
# CSV output removed - only merged dataset will be saved at the end

# Create detailed report
report_data = []
for feature in final_features:
    metrics = feature_metrics[feature]
    report_data.append({
        'Feature': feature,
        'Discrimination_Score': metrics['discrimination_score'],
        'KS_Statistic': metrics['ks_statistic'],
        'Cohens_D': metrics['cohens_d'],
        'Overlap': metrics['distribution_overlap'],
        'P_Value': metrics['ks_pvalue'],
        'Phishing_Mean': metrics['phish_mean'],
        'Legitimate_Mean': metrics['legit_mean'],
        'Phishing_Median': metrics['phish_median'],
        'Legitimate_Median': metrics['legit_median']
    })

report_df = pd.DataFrame(report_data)
# CSV output removed - only merged dataset will be saved at the end

# Create summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['Number of Features', 'Avg Discrimination Score', 'Avg KS Statistic', 
               'Avg Cohen D', 'Avg Overlap', 'Min P-Value'],
    'Value': [
        len(final_features),
        report_df['Discrimination_Score'].mean(),
        report_df['KS_Statistic'].mean(),
        report_df['Cohens_D'].mean(),
        report_df['Overlap'].mean(),
        report_df['P_Value'].min()
    ]
})
# CSV output removed - only merged dataset will be saved at the end

print(f"\n✓ Distribution plots saved: 'notebooks/distribution_comparison.png'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ DISTRIBUTION-BASED SELECTION COMPLETE!")
print("="*80)

print(f"""
📊 Selection Summary:
--------------------
• Total features evaluated: {len(feature_cols)}
• Features with significant differences: {sum(1 for f in feature_metrics.values() if f['ks_pvalue'] < 0.05)}
• Final features selected: {len(final_features)}
• Average discrimination score: {report_df['Discrimination_Score'].mean():.2f}
• All p-values < 0.05: {all(report_df['P_Value'] < 0.05)}

🎯 Key Insights:
----------------""")

# Show top 3 discriminators
for i, row in report_df.head(3).iterrows():
    print(f"{i+1}. {row['Feature']}:")
    print(f"   - Phishing median: {row['Phishing_Median']:.2f}")
    print(f"   - Legitimate median: {row['Legitimate_Median']:.2f}")
    print(f"   - Discrimination power: {row['Discrimination_Score']:.2f}")

print(f"""
📋 Final Feature List for Your Notebook:
-----------------------------------------""")
print("FINAL_FEATURES = [")
for feature in final_features:
    print(f"    '{feature}',")
print("]")

# Export final chosen features with detailed explanations
final_features_export = []
for feature in final_features:
    metrics = feature_metrics[feature]
    # Find which group this feature belongs to
    feature_group = 'other'
    for group_name, group_features in feature_groups.items():
        if feature in group_features:
            feature_group = group_name
            break

    final_features_export.append({
        'Feature_Name': feature,
        'Feature_Group': feature_group,
        'Discrimination_Score': metrics['discrimination_score'],
        'KS_Statistic': metrics['ks_statistic'],
        'Cohens_D': metrics['cohens_d'],
        'Distribution_Overlap': metrics['distribution_overlap'],
        'JS_Divergence': metrics['js_divergence'],
        'P_Value': metrics['ks_pvalue'],
        'Phishing_Mean': metrics['phish_mean'],
        'Legitimate_Mean': metrics['legit_mean'],
        'Phishing_Median': metrics['phish_median'],
        'Legitimate_Median': metrics['legit_median'],
        'Phishing_P25': metrics['phish_p25'],
        'Phishing_P75': metrics['phish_p75'],
        'Legitimate_P25': metrics['legit_p25'],
        'Legitimate_P75': metrics['legit_p75'],
        'Variance_Ratio': metrics['variance_ratio']
    })

final_features_df = pd.DataFrame(final_features_export)
# CSV output removed - only merged dataset will be saved at the end

# ============================================================================
# Merge selected features back to cleaned_date_merge.csv
# ============================================================================
print("\n" + "="*80)
print("MERGING SELECTED FEATURES TO ORIGINAL DATASET")
print("="*80)

# Load original dataset
print("📂 Loading original dataset...")
original_df = pd.read_csv('data/processed/cleaned_date_merge.csv')
print(f"✓ Original dataset loaded: {len(original_df):,} rows")

# Load the selected features dataset (sender-level features)
selected_features_df = df[['sender'] + final_features + ['label']]

# Merge on sender
print(f"🔗 Merging {len(final_features)} selected features to original dataset...")
merged_df = original_df.merge(selected_features_df, on='sender', how='left')

# Save merged dataset
output_path = 'notebooks/final_time_series_dataset.csv'
merged_df.to_csv(output_path, index=False)

print(f"✅ Merged dataset saved to '{output_path}'")
print(f"   - Total rows: {len(merged_df):,}")
print(f"   - Total columns: {len(merged_df.columns)}")
print(f"   - Added features: {', '.join(final_features)}")

print(f"""
💡 These features were selected because they show the strongest
   distribution differences between phishing and legitimate senders,
   regardless of correlation with each other.

📁 Output Files Generated:
   1. {output_path} - Final merged dataset with selected features
   2. notebooks/distribution_comparison.png - Distribution visualizations
""")