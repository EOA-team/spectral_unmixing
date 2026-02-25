import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.patches as mpatches
import ast
from sklearn.metrics import mean_squared_error
from scipy.stats import sem, t, ttest_rel, ttest_ind, wilcoxon


def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return 0  # Cannot compute CI with 1 value
    m = np.mean(data)
    se = sem(data)  # standard error
    h = se * t.ppf((1 + confidence) / 2., n-1)  # margin of error
    return h


def convert_pval_to_start(p_val):
    if p_val is not None:
        if p_val < 0.001:
            star = '***'
        elif p_val < 0.01:
            star = '**'
        elif p_val < 0.05:
            star = '*'
        else:
            star = ''
    
    return star


# === Settings ===
model_type = "NN"
metric = "RMSE"
metric = metric.lower()

df_scores = pd.read_pickle(f"../results/{model_type}_full_test_scores.pkl")

# Keep only global or matching soil-specific models
filtered_df = df_scores[
    (df_scores['soil_group'] == 0) | (df_scores['soil_group'] == df_scores['test_soil'])
].copy()

filtered_df['Model'] = filtered_df['soil_group'].apply(
    lambda x: 'Global' if x == 0 else 'Soil-specific'
)

# === Compute mean ± CI per class type ===
summary = (
    filtered_df
    .groupby(['class_type', 'Model'])[metric]
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
summary['ci'] = 1.96 * summary['std'] / np.sqrt(summary['count'])

# === Compute statistical significance (t-test Global vs Soil-specific) ===
p_values = {}
for class_type in filtered_df['class_type'].unique():
    global_vals = filtered_df[(filtered_df['Model'] == 'Global') & 
                              (filtered_df['class_type'] == class_type)][metric].values
    soil_vals = filtered_df[(filtered_df['Model'] == 'Soil-specific') & 
                            (filtered_df['class_type'] == class_type)][metric].values
    if len(global_vals) > 1 and len(soil_vals) > 1:
        _, p_val = ttest_ind(global_vals, soil_vals, equal_var=False)
        p_values[class_type] = p_val
    else:
        p_values[class_type] = None


"""
# === Plot simplified comparison ===
sns.set(style="whitegrid")
palette = {'Global': 'saddlebrown', 'Soil-specific': 'skyblue'}

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

for ax, class_type in zip(axes, ['NPV', 'PV', 'SOIL']):
    sub = summary[summary['class_type'] == class_type]    
    # Bar plot
    ax.bar(
        sub['Model'],
        sub['mean'],
        yerr=sub['ci'],
        capsize=4,
        color=[palette[m] for m in sub['Model']],
        edgecolor='black',
        #hatch=[hatch_map[m] for m in sub['Model']],
        width=1
    )
    ax.set_xticklabels([])

    # Add text labels and significance star
    for i, row in sub.reset_index(drop=True).iterrows():
        ax.text(i, row['mean'] + row['ci'] * 1.05, f"{row['mean']:.3f}",
                ha='center', va='bottom', fontsize=9)
    
    star = convert_pval_to_start(p_values.get(class_type))
    if star:
        ymax = sub['mean'].max() + sub['ci'].max() * 1.5
        ax.text(0.5, ymax, star, ha='center', va='bottom', fontsize=14)
    
    ax.set_title(class_type)
    ax.set_xlabel('')
    if class_type == 'NPV':
        ax.set_ylabel(f"{metric.upper()} [-]")
    ax.set_ylim(0, summary['mean'].max() * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

# Legend
handles = [
    mpatches.Patch(facecolor='saddlebrown', edgecolor='black', label='Global'),
    mpatches.Patch(facecolor='skyblue', edgecolor='black', label='Soil-specific')
]
axes[-1].legend(handles=handles, title='Model', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig(f"{model_type}_{metric}_barplot_global_vs_specific.png", dpi=300)
"""

"""
# ======= PLOT GLOBAL vs SOIL SPECIFIC SCORES ===========
# WITH STATISTICAL SINGIFICANCE OF DIFFERENCE BETWEEN GLOVAL/SPECIFIC MODELS

model_type = "NN"
metric = "RMSE"
metric = metric.lower()

df_scores = pd.read_pickle(f"../results/{model_type}_full_test_scores.pkl")
df_preds = pd.read_pickle(f"../results/{model_type}_full_test_predictions.pkl")

# Prepare data for plotting: keep only rows where model was either:
# - global (soil_group == 0), OR
# - matched the test soil (soil_group == _test_soil)
filtered_df = df_scores[
    (df_scores['soil_group'] == 0) | (df_scores['soil_group'] == df_scores['test_soil'])
].copy()

# Add model type
filtered_df['Model'] = filtered_df['soil_group'].apply(lambda x: 'Global' if x == 0 else 'Soil-specific')
filtered_df['test_soil'] = filtered_df['test_soil'].astype(str)  # For consistent plotting
filtered_df = filtered_df.sort_values(by='test_soil')

# First, compute statistical significance between golbal/soil specific model with t-test (p<0.05 then reject Ho of no difference)
significance_results = {}

for class_type in filtered_df['class_type'].unique():
    for test_soil in filtered_df['test_soil'].unique():
        global_vals = filtered_df[
            (filtered_df['soil_group'] == 0) &
            (filtered_df['test_soil'] == test_soil) &
            (filtered_df['class_type'] == class_type)
        ][metric].values

        if test_soil == "0":
            # Soil-specific models tested on soil 0 but not global
            soil_spec_vals = df_scores[
                (df_scores['soil_group'] != 0) &
                (df_scores['test_soil'] == 0) &
                (df_scores['class_type'] == class_type)  # Make sure to filter by class_type too if needed
            ][[metric, 'soil_group']].groupby('soil_group').mean()[metric].values
        else:
            soil_spec_vals = filtered_df[
                (filtered_df['soil_group'] == int(test_soil)) &
                (filtered_df['test_soil'] == test_soil) &
                (filtered_df['class_type'] == class_type)
            ][metric].values
            
            
        if len(global_vals) > 1 and len(soil_spec_vals) > 1:
            t_stat, p_val = ttest_ind(global_vals, soil_spec_vals, equal_var=False)
            significance_results[(class_type, test_soil)] = p_val
        else:
            significance_results[(class_type, test_soil)] = None
        
         

# Color map by test soil
color_map = {
    "0": "saddlebrown",
    "1": "teal",
    "2": "orange",
    "3": "rebeccapurple",
    "4": "palevioletred",
    "5": "olivedrab"
}

# Hatch pattern map
hatch_map = {
    'Global': '//',
    'Soil-specific': ''
}

# Plot
sns.set(style='white')
palette = {'Global': 'lightgray', 'Soil-specific': 'saddlebrown'}
hatch_map = {'Global': '//', 'Soil-specific': '...'}

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for ax, class_type in zip(axes, ['NPV', 'PV', 'SOIL']):
    subset = filtered_df[filtered_df['class_type'] == class_type]
    
    # Plot bars manually
    bar_width = 0.4
    test_soils = sorted(subset['test_soil'].unique())
    x_locs = range(len(test_soils))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, test_soil in enumerate(test_soils):
        star_positions = []
        for j, model in enumerate(['Global', 'Soil-specific']):
            vals = subset[
                (subset['test_soil'] == test_soil) &
                (subset['Model'] == model)
            ][metric].values
            if len(vals) > 0:
                mean_val = np.mean(vals)
                ci = compute_confidence_interval(vals)
                color = color_map[test_soil]
                hatch = hatch_map[model]
                ax.bar(
                    i + (j - 0.5) * bar_width,
                    mean_val,
                    width=bar_width,
                    color=color,
                    edgecolor='black',
                    hatch=hatch,
                    label=model if i == 0 else None,  # only add label once for legend
                    yerr=ci,
                    capsize=4,
                    error_kw={
                        'ecolor': 'dimgray',
                        'elinewidth': 1.5,
                        'capthick': 1.5,
                        'linestyle': 'dashed',
                    }
                )
                # Add label above CI bar
                upper_y = mean_val + ci if isinstance(ci, (int, float)) else mean_val + ci[1]
                text_y = upper_y + upper_y*0.015
                star_positions.append(text_y)
                ax.text(i + (j - 0.5) * bar_width, text_y, f'{mean_val:.2f}', 
                        ha='center', va='bottom', fontsize=8)

        # Add mean soil-specific bar for test_soil == "0"
        if test_soil == "0":
            # Get soil specific models on global soils
            soil_spec_vals = df_scores[
                (df_scores['soil_group'] != 0) &  # Only soil-specific models
                (df_scores['test_soil'] == 0) &   # Tested on the global dataset
                (df_scores['class_type'] == class_type)
            ][metric].values

            if len(soil_spec_vals) > 0:
                mean_val = np.mean(soil_spec_vals)
                ci = compute_confidence_interval(soil_spec_vals)
                color = color_map[test_soil]
                ax.bar(
                    i + 0.5* bar_width,  # right-most bar
                    mean_val,
                    width=bar_width,
                    color=color,
                    edgecolor='black',
                    yerr=ci,  # Add error bar
                    hatch=hatch_map['Soil-specific'],
                    capsize=4,
                    error_kw={
                        'ecolor': 'dimgray',        # error bar color
                        'elinewidth': 1.5,        # thickness of error bar lines
                        'capthick': 1.5,          # thickness of the caps
                        'linestyle': 'dashed',
                    },
                )
                upper_y = mean_val + ci if isinstance(ci, (int, float)) else mean_val + ci[1]
                text_y = upper_y + upper_y*0.015
                star_positions.append(text_y)
                ax.text(i + 0.5*bar_width, text_y, f'{mean_val:.2f}', 
                        ha='center', va='bottom', fontsize=8)


        # Add star to indicate significance (* if p<0.05, ** if p<0.01, *** if p<0.001)
        star = convert_pval_to_start(significance_results.get((class_type, test_soil)))
        star_y = max(star_positions)*1.04
        ax.text(i, star_y, star, ha='center', va='bottom', fontsize=14, color='black')


    ax.set_title(class_type, pad=15)
    ax.set_xlabel("Soil Group")
    ax.set_xticks(x_locs)
    #ax.set_xticklabels(['all' if ts == "0" else str(ts) for ts in test_soils])
    # Add test set size to the x labels       
    xtick_labels = []
    for ts in test_soils:
        ts_int = int(ts)
        # number of samples in test set
        n_test = df_preds[
            (df_preds['class_type'] == class_type) &
            (df_preds['test_soil'] == ts_int) &
            (df_preds['soil_group'] == ts_int)
        ].groupby('iteration')['y_test'].apply(lambda x: np.mean([len(arr) for arr in x])).mean()
        
        label = f"{'all' if ts == '0' else ts}\nn={int(n_test)}"
        xtick_labels.append(label)
    ax.set_xticklabels(xtick_labels)

    if class_type == 'NPV':
        ax.set_ylabel(f"{metric.upper()} [-]")

# Create custom legend for hatch
handles = [
    mpatches.Patch(facecolor='gray', edgecolor='black', hatch='//', label='Global'),
    mpatches.Patch(facecolor='gray', edgecolor='black', hatch='..', label='Soil-specific')
]
axes[-1].legend(handles=handles, title='Model', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig(f"{model_type}_{metric}_barplot_ci.png")
"""

"""
# ======= PLOT GLOBAL vs SOIL SPECIFIC SCORES ===========

model_type = "NN"
metric = "R2"
metric = metric.lower()

df_scores = pd.read_pickle(f"../results/{model_type}_full_test_scores.pkl")
df_preds = pd.read_pickle(f"../results/{model_type}_full_test_predictions.pkl")

# Average error per model per test soil over iterations
grouped = df_scores.groupby(['class_type', 'soil_group', 'test_soil'])[metric].mean().reset_index()

# Prepare data for plotting: keep only rows where model was either:
# - global (soil_group == 0), OR
# - matched the test soil (soil_group == _test_soil)
plot_df = grouped[(grouped['soil_group'] == 0) | (grouped['soil_group'] == grouped['test_soil'])].copy()

# Add a model type column for legend
plot_df['Model'] = plot_df['soil_group'].apply(lambda x: 'Global' if x == 0 else 'Soil-specific')

# Sort by test soil for consistency
plot_df['test_soil'] = plot_df['test_soil'].astype(str)  # for plotting categories
plot_df = plot_df.sort_values(by='test_soil')

# Color map by test soil
color_map = {
    "0": "saddlebrown",
    "1": "teal",
    "2": "orange",
    "3": "rebeccapurple",
    "4": "palevioletred",
    "5": "olivedrab"
}

# Hatch pattern map
hatch_map = {
    'Global': '//',
    'Soil-specific': ''
}

# Plot
sns.set(style='white')
palette = {'Global': 'lightgray', 'Soil-specific': 'saddlebrown'}
hatch_map = {'Global': '//', 'Soil-specific': '...'}

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for ax, class_type in zip(axes, ['NPV', 'PV', 'SOIL']):
    subset = plot_df[plot_df['class_type'] == class_type]
    
    # Plot bars manually
    bar_width = 0.4
    test_soils = sorted(subset['test_soil'].unique())
    x_locs = range(len(test_soils))

    for i, test_soil in enumerate(test_soils):
        for j, model in enumerate(['Global', 'Soil-specific']):
            value = subset[(subset['test_soil'] == test_soil) & (subset['Model'] == model)]
            if not value.empty:
                val = value[metric].values[0]
                color = color_map[test_soil]
                hatch = hatch_map[model]
                ax.bar(
                    i + (j - 0.5) * bar_width,
                    val,
                    width=bar_width,
                    color=color,
                    edgecolor='black',
                    hatch=hatch,
                    label=model if i == 0 else None  # only add label once for legend
                )
                # Add label
                ax.text(i + (j - 0.5) * bar_width, val+0.001, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=8)

        # Add mean soil-specific bar for test_soil == "0"
        if test_soil == "0":
            # Get avg of soil specific models on global soils
            avg_soil_df = grouped[(grouped['soil_group'] != 0) & (grouped['test_soil'] == 0)].copy()
            avg_soil_df['Model'] = ['Soil-specific']*len(avg_soil_df)
            soil_spec_vals = avg_soil_df[
                (avg_soil_df['class_type'] == class_type)][metric].values

            if len(soil_spec_vals) > 0:
                mean_val = soil_spec_vals.mean()
                std_val = soil_spec_vals.std()
                color = color_map[test_soil]
                ax.bar(
                    i + 0.5* bar_width,  # right-most bar
                    mean_val,
                    width=bar_width,
                    color=color,
                    edgecolor='black',
                    yerr=std_val,  # Add error bar
                    hatch=hatch_map['Soil-specific'],
                    capsize=4,
                    error_kw={
                        'ecolor': 'dimgray',        # error bar color
                        'elinewidth': 1.5,        # thickness of error bar lines
                        'capthick': 1.5,          # thickness of the caps
                        'linestyle': 'dashed',
                    },
                )

                ax.text(i + 0.5*bar_width, mean_val+0.001, f'{mean_val:.2f}', 
                        ha='center', va='bottom', fontsize=8)

    ax.set_title(class_type)
    ax.set_xlabel("Soil Group")
    ax.set_xticks(x_locs)
    #ax.set_xticklabels(['all' if ts == "0" else str(ts) for ts in test_soils])
    # Add test set size to the x labels       
    xtick_labels = []
    for ts in test_soils:
        ts_int = int(ts)
        # number of samples in test set
        n_test = df_preds[
            (df_preds['class_type'] == class_type) &
            (df_preds['test_soil'] == ts_int) &
            (df_preds['soil_group'] == ts_int)
        ].groupby('iteration')['y_test'].apply(lambda x: np.mean([len(arr) for arr in x])).mean()
        
        label = f"{'all' if ts == '0' else ts}\nn={int(n_test)}"
        xtick_labels.append(label)
    ax.set_xticklabels(xtick_labels)

    if class_type == 'NPV':
        ax.set_ylabel(f"{metric.upper()} [-]")

# Create custom legend for hatch
handles = [
    mpatches.Patch(facecolor='gray', edgecolor='black', hatch='//', label='Global'),
    mpatches.Patch(facecolor='gray', edgecolor='black', hatch='..', label='Soil-specific')
]
axes[-1].legend(handles=handles, title='Model', loc='upper right')

plt.tight_layout()
plt.savefig(f"{model_type}_{metric}_barplot.png")
"""

"""
# ======= SCATTER PLOT GLOBAL VS SOIL SPECIFIC ON SOIL TEST SETS ===========

model_type = "NN"

df_scores = pd.read_pickle(f"../results/{model_type}_full_test_scores.pkl")
df_preds = pd.read_pickle(f"../results/{model_type}_full_test_predictions.pkl")

soils = np.arange(1,6)

color_map = {
    1: "teal",
    2: "orange",
    3: "rebeccapurple",
    4: "palevioletred",
    5: "olivedrab"
}

fig, ax = plt.subplots(len(soils), 2, figsize=(12, 20), sharey=True, sharex=True)

for soil_group in soils:
    # Preds and scores of soil-specific model
    preds_soil = df_preds[(df_preds['class_type'] == 'SOIL') & (df_preds['soil_group'] == soil_group) & (df_preds['test_soil'] == soil_group)].copy()
    scores_soil = df_scores[(df_scores['class_type'] == 'SOIL') &(df_scores['soil_group'] == soil_group) & (df_scores['test_soil'] == soil_group)].copy()
    spec_rmse, spec_mae, spec_r2 = scores_soil['rmse'].mean(), scores_soil['mae'].mean(), scores_soil['r2'].mean()
 
    # Preds and scores of global model
    preds_global = df_preds[(df_preds['class_type'] == 'SOIL') &(df_preds['soil_group'] == 0) & (df_preds['test_soil'] == soil_group)].copy()
    scores_global = df_scores[(df_scores['class_type'] == 'SOIL') &(df_scores['soil_group'] == 0) & (df_scores['test_soil'] == soil_group)].copy()
    global_rmse, global_mae, global_r2 = scores_global['rmse'].mean(), scores_global['mae'].mean(), scores_global['r2'].mean()

    # Scatter plot - global model
    mean_y_test = np.mean(np.stack(preds_global['y_test']), axis=0)
    mean_y_pred = np.mean(np.stack(preds_global['y_pred']), axis=0)
    cv_global = np.nanmean(np.std(np.stack(preds_global['y_pred']), axis=0) / np.abs(mean_y_pred))
    ax[soil_group-1, 0].scatter(mean_y_test, mean_y_pred, alpha=0.5, color=color_map[soil_group])
    ax[soil_group-1, 0].set_xlim(0,1)
    ax[soil_group-1, 0].set_ylim(0,1)

    ax[soil_group-1, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    
    slope, intercept = np.polyfit(mean_y_test, mean_y_pred, 1)
    x_vals = np.array([0,1])
    regression_line = slope * x_vals + intercept
    ax[soil_group-1, 0].plot(x_vals, regression_line, 'k-', linewidth=1)

    ax[soil_group-1, 0].text(0.05, 0.8, f'RMSE: {global_rmse:.3f}\nMAE: {global_mae:.3f}\nR2: {global_r2:.3f}\nCV: {cv_global:.3f}',
                            fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # Scatter plot - soil-specific model
    mean_y_test = np.mean(np.stack(preds_soil['y_test']), axis=0)
    mean_y_pred = np.mean(np.stack(preds_soil['y_pred']), axis=0)
    cv_soil = np.nanmean(np.std(np.stack(preds_soil['y_pred']), axis=0) / np.abs(mean_y_pred))
    ax[soil_group-1, 1].scatter(mean_y_test, mean_y_pred, alpha=0.5, color=color_map[soil_group])
    ax[soil_group-1, 1].set_xlim(0,1)
    ax[soil_group-1, 1].set_ylim(0,1)

    ax[soil_group-1, 1].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    
    slope, intercept = np.polyfit(mean_y_test, mean_y_pred, 1)
    x_vals = np.array([0,1])
    regression_line = slope * x_vals + intercept
    ax[soil_group-1, 1].plot(x_vals, regression_line, 'k-', linewidth=1)

    ax[soil_group-1, 1].text(0.05, 0.8, f'RMSE: {spec_rmse:.3f}\nMAE: {spec_mae:.3f}\nR2: {spec_r2:.3f}\nCV: {cv_soil:.3f}',
                            fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))


    # Add a label on the right of each row to indicate soil group
    ax[soil_group-1, 1].text(1.05, 0.5, f'Soil Group {soil_group}', va='center', ha='left',
                  transform=ax[soil_group-1, 1].transAxes, fontsize=12, rotation=-90)

ax[0, 0].set_title('Global model', fontsize=16)
ax[0, 1].set_title('Soil specific models', fontsize=16)
fig.supxlabel('Reference FC [-]', fontsize=18)
fig.supylabel('Predicted FC [-]', fontsize=18)
plt.tight_layout()

plt.savefig(f'{model_type}_soiltest_scatters.png')
"""



# ======= PLOT SCATTER PLOTS PER CLASS AND SOIL GROUP ON TEST SET ===========