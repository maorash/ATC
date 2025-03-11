# Adapted from https://github.com/eric-mitchell/detect-gpt/blob/main/run.py

import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, precision_recall_curve, auc


def remove_nan(predictions, exp_name, verbose=False):
    result = []
    nans = 0
    for x in predictions:
        if np.isnan(x):
            nans += 1
        elif np.isinf(x):
            nans += 1
        else:
            result.append(x)
    if nans > 0 and verbose:
        print(f"{exp_name}: Warning, got {nans} NaNs/INFs out of {len(predictions)}")
    return result, nans


# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
          "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
          "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def get_roc_metrics(real_preds, sample_preds, exp_name):
    real_preds, _ = remove_nan(real_preds, exp_name + ' Human')
    sample_preds, nans = remove_nan(sample_preds, exp_name + ' AIGC')
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc), nans


def get_precision_recall_metrics(real_preds, sample_preds, exp_name):
    real_preds, _ = remove_nan(real_preds, exp_name + ' Human')
    sample_preds, nans = remove_nan(sample_preds, exp_name + ' AIGC')
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc), nans


def analyze(human, aigc, exp_name):
    predictions = {
        'real': human,
        'samples': aigc
    }

    fpr, tpr, roc_auc, _ = get_roc_metrics(predictions['real'], predictions['samples'], exp_name)
    p, r, pr_auc, nans = get_precision_recall_metrics(predictions['real'], predictions['samples'], exp_name)

    # print(f"threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': exp_name,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
        'num_nans': nans
    }


# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
          "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
          "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def save_pr_curves_plotly(experiments):
    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each experiment
    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["pr_metrics"]
        fig.add_trace(
            go.Scatter(
                x=metrics["recall"],
                y=metrics["precision"],
                mode="lines",
                line=dict(color=color),
                name=f"{experiment['name']}, PR AUC={metrics['pr_auc']:.3f}"
            )
        )
        print(f"{experiment['name']} PR AUC: {metrics['pr_auc']:.3f}")

    # Add a diagonal line for reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[1, 0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Baseline"
        )
    )

    # Customize layout
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor="lightgray"),
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="lightgray"),
        legend=dict(title="Experiments", font=dict(size=10)),
        template="plotly_white"
    )

    # Show the plot
    fig.show()

    # Save to an HTML file (optional)
    # fig.write_html("pr_curves.html")


def save_roc_curves_plotly(experiments):
    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each experiment
    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        fig.add_trace(
            go.Scatter(
                x=metrics["fpr"],
                y=metrics["tpr"],
                mode="lines",
                line=dict(color=color),
                name=f"{experiment['name']}, ROC AUC={metrics['roc_auc']:.3f}"
            )
        )
        print(f"{experiment['name']} ROC AUC: {metrics['roc_auc']:.3f}")

    # Add a diagonal line for reference
    fig.add_trace(
        go.Scatter(
            x=[1, 0],
            y=[1, 0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Baseline"
        )
    )

    # Customize layout
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False-Positive Rate",
        yaxis_title="True-Positive Rate",
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor="lightgray"),
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="lightgray"),
        legend=dict(title="Experiments", font=dict(size=10)),
        template="plotly_white"
    )

    # Show the plot
    fig.show()

    # Save to an HTML file (optional)
    # fig.write_html("pr_curves.html")