import numpy as np
import pandas as pd
import os
import numpy as np
from pathlib import Path
import paths as pt
matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import seaborn as sns
from utility.model import map_model_name

plt.rcParams.update({'axes.labelsize': 'small',
                     'axes.titlesize': 'small',
                     'font.size': 14.0})

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def get_y_label(metric_name):
    if "Loss" in metric_name:
        return r'Loss $\mathcal{L}(\theta)$'
    elif "CTD" in metric_name:
        return 'CTD'
    elif "IBS" in metric_name:
        return "IBS"
    else:
        return "INBLL"

def plot_calibration_curves(percentiles, pred_obs, predictions, model_names, dataset_name, save_dir=None):
    if save_dir is None:
        save_dir = pt.RESULTS_DIR
    n_percentiles = len(percentiles.keys())
    fig, axes = plt.subplots(n_percentiles, 2, figsize=(12, 12))
    labels = list()
    for i, (q, pctl) in enumerate(percentiles.items()):
        for model_idx, model_name in enumerate(model_names):
            pred = pred_obs[pctl][model_name][0]
            obs = pred_obs[pctl][model_name][1]
            preds = predictions[pctl][model_name]
            data = pd.DataFrame({'Pred': pred, 'Obs': obs})
            axes[i][0].set_xlabel("Predicted probability")
            axes[i][1].set_xlabel("Predicted probability")
            axes[i][0].set_ylabel("Observed probability")
            axes[i][0].set_title(f"Calibration at the {q}th percentile of survival time")
            axes[i][1].set_title(f"Probabilities at the {q}th percentile")
            axes[i][0].grid(True)
            axes[i][1].grid(True)
            sns.lineplot(data, x='Pred', y='Obs', color=TFColor[model_idx], ax=axes[i][0], legend=False, label=map_model_name(model_name))
            sns.kdeplot(preds, fill=True, common_norm=True, alpha=.5, cut=0, linewidth=1, color=TFColor[model_idx], ax=axes[i][1])
        ax=axes[i][0].plot([0, 1], [0, 1], c="k", ls="--", linewidth=1.5)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(Path.joinpath(Path(save_dir), f"{dataset_name.lower()}_calibration.pdf"),
                format='pdf', bbox_inches="tight")
    plt.close()

def plot_credible_interval(event_times, surv_ensemble, actual_time, actual_event,
                           model_name, dataset_name, sample_idx, save_dir):
    """Plot individual survival curve with 90% credible interval (Figure 2 from paper).
    Generic for VI, MCD, BayCox, BayMTLR. surv_ensemble: (n_mc_samples, n_test_samples, n_times)."""
    event_times = np.asarray(event_times)
    surv_sample = surv_ensemble[:, sample_idx, :]  # (n_mc, n_times)
    mean_surv = np.mean(surv_sample, axis=0)
    lower = np.percentile(surv_sample, 5, axis=0)
    upper = np.percentile(surv_sample, 95, axis=0)
    median_time_idx = np.searchsorted(-mean_surv, -0.5)
    median_time = event_times[min(median_time_idx, len(event_times) - 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(event_times, mean_surv, 'b-', linewidth=2, label='Mean S(t)')
    ax1.fill_between(event_times, lower, upper, alpha=0.3, color='blue', label='90% CrI')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=median_time, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_time:.0f}')
    if actual_event == 1:
        ax1.axvline(x=actual_time, color='green', linestyle='-', alpha=0.7, label=f'Actual: {actual_time:.0f}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Survival Probability')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'{model_name.upper()} - Individual Survival Function')
    ax1.legend(fontsize=8)

    median_times = []
    for s in range(surv_sample.shape[0]):
        idx = np.searchsorted(-surv_sample[s], -0.5)
        median_times.append(event_times[min(idx, len(event_times) - 1)])
    ax2.hist(median_times, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(x=np.mean(median_times), color='red', linestyle='--', label=f'Mean: {np.mean(median_times):.0f}')
    ax2.set_xlabel('Predicted Survival Time')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{model_name.upper()} - Predicted Time Distribution')
    ax2.legend(fontsize=8)

    fig.suptitle(f'{dataset_name} - Sample #{sample_idx}', fontsize=12)
    plt.tight_layout()
    save_path = Path(save_dir) / f"{dataset_name.lower()}_{model_name}_cri_sample{sample_idx}.pdf"
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path

def plot_training_curves(results, dataset_name, model_names, metric_names, save_dir=None):
    if save_dir is None:
        save_dir = pt.RESULTS_DIR
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for (j, metric_name) in enumerate(metric_names):
        for (k, model_name) in enumerate(model_names):
            model_results = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            metric_results = model_results[metric_name]
            n_epochs = len(model_results)
            axes[j].plot(range(n_epochs), metric_results, label=map_model_name(model_name),
                            marker="o", color=TFColor[k], linewidth=1)
        axes[j].set_xlabel('Epoch', fontsize="medium")
        axes[j].set_ylabel(metric_name, fontsize="medium")
        axes[j].grid()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    plt.savefig(Path.joinpath(Path(save_dir), f"{dataset_name.lower()}_training_curves.pdf"),
                format='pdf', bbox_inches="tight")
    plt.close()