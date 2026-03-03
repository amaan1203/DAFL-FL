import os
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .divergence import *

HEATMAP_FIG_SIZE = (30, 26)
FIG_SIZE = (15, 12)
FONT_SIZE = 50
BAR_WIDTH = 0.14
LARGE_BAR_WIDTH = 0.4
LEGEND_SIZE = 25
TICKS_SIZE = 15
LINE_WIDTH = 7
MARKER_SIZE = 10
MARKER_WIDTH = 5

MARKERS = {
    "fast": "x",
    "adafed": "d",
    "markov": "h",
    "unbiased": "s"
}

LINE_STYLES = {
    "fast": "dotted",
    "adafed": "dashed",
    "markov": "solid",
    "unbiased": "dashdot"
}

METHODS = ["unbiased", "fast", "adafed", "markov"]

COLORS = {
    "fast": "tab:blue",
    "adafed": "tab:green",
    "markov": "tab:orange",
    "unbiased": "tab:red",
    "true": "tab:pink",
    "expected": "tab:blue",
    "observed": "tab:orange"
}

HATCHES = {
    "fast": "/",
    "adafed": "\\",
    "markov": "*",
    "unbiased": "-",
    "true": "O",
    "expected": "/",
    "observed": "-"
}

BAR_POSITION = {
    "fast": -2,
    "adafed": 2,
    "markov": 1,
    "unbiased": -1,
    "true": 0,
    "expected": 0,
    "observed": -1
}

LEGEND = {
    "fast": "F3AST",
    "adafed": "AdaFed",
    "markov": "Our Proposed FL",
    "unbiased": "Unbiased",
    "true": r"Target",
    "expected": "Expected",
    "observed": "Observed"
}

FILES_NAMES = {
    "Train/Loss": "train-loss.pdf",
    "Train/Metric": "train-acc.pdf",
    "Test/Loss": "test-loss.pdf",
    "Test/Metric": "test-acc.pdf"
}

AXE_LABELS = {
    "Train/Loss": "Train loss",
    "Train/Metric": "Train accuracy",
    "Test/Loss": "Test loss",
    "Test/Metric": "Test accuracy"
}

AXE_LABELS_AVG = {
    "Train/Loss": "Time-average train loss",
    "Train/Metric": "Time-average train accuracy",
    "Test/Loss": "Time-average test loss",
    "Test/Metric": "Time-average test accuracy"
}


def parse_tf_events_file(events_path, tag):
    r"""Parser of tf events files

    Extracts the steps and the values corresponding to 'tag' from tfevents file

    Parameters
    ----------
    events_path: str

    tag: str
        name of the tag to extract, possible are
        {"Train/Loss", "Train/Metric", "Val/Loss", "Val/Metric", "Test/Loss", "Test/Metric"

    Returns
    -------
        * Tuple(List, List): list of steps and the corresponding values

    """
    try:
        ea = EventAccumulator(events_path).Reload()
        tag_values = []
        steps = []
        for event in ea.Scalars(tag):
            tag_values.append(event.value)
            steps.append(event.step)
        return steps, tag_values
    except Exception as e:
        print(f"Warning: Could not read tf events file for tag '{tag}' at path '{events_path}'. Error: {e}")
        return [], []


def parse_history_file(history_path):
    """parse history file

    Parameters
    ----------
    history_path: str
        path to .json file storing the history of the clients sampler;

    Returns
    -------
        * 2-D numpy.array: clients_activities of shape (n_steps, n_clients)
        * 2-D numpy.array: clients_weights of shape (n_steps, n_clients)
        * 1-D numpy.array: true_weights of shape (n_clients, )
        * 1-D numpy.array: true_availabilities (n_clients, )
        * 1-D numpy.array: clients_ids of shape (n_clients, )

    """
    with open(history_path, "r") as f:
        sampler_history = json.load(f)

    clients_ids = sampler_history["clients_ids"]
    true_weights_dict = sampler_history["clients_true_weights"]
    availability_dict = sampler_history["clients_true_availability"]

    sampler_history = sampler_history["history"]

    time_steps = list(sampler_history.keys())

    n_clients = len(clients_ids)
    n_steps = len(time_steps)

    clients_weights = np.zeros((n_steps, n_clients), dtype=np.float32)
    clients_activities = np.zeros((n_steps, n_clients), dtype=int)

    for time_step in time_steps:
        current_state = sampler_history[time_step]
        time_step = int(time_step)

        for client_id in current_state["active_clients"]:
            clients_activities[time_step, client_id] = 1

        for client_id, weight in \
                zip(
                    current_state["sampled_clients_ids"],
                    current_state["sampled_clients_weights"]
                ):

            clients_weights[time_step, client_id] = weight

    true_weights = np.zeros(n_clients, dtype=np.float32)
    for client_id, weight in true_weights_dict.items():
        true_weights[int(client_id)] = weight

    true_availability = np.zeros(n_clients, dtype=np.float32)
    for client_id, availability in availability_dict.items():
        true_availability[int(client_id)] = availability

    return clients_activities, clients_weights, true_weights, true_availability, clients_ids


def gather_history(history_dir):
    """
    Gathers history from JSON files in a directory structure.

    Parameters
    ----------
    history_dir: str
        The root directory containing experiment history.

    Returns
    -------
    Dict[str, Dict[int, Tuple]]
        A dictionary mapping activity type to seed to history data.
    """
    history_dict = dict()
    if not os.path.exists(history_dir):
        print(f"Warning: History directory not found at '{history_dir}'")
        return history_dict

    for activity_item in os.listdir(history_dir):
        activity_path = os.path.join(history_dir, activity_item)
        if not os.path.isdir(activity_path):
            continue

        activity = activity_item.split("_")[-1]
        history_dict[activity] = dict()

        for seed_item in os.listdir(activity_path):
            seed_path = os.path.join(activity_path, seed_item)
            if not os.path.isfile(seed_path) or not seed_item.endswith(".json"):
                continue
            
            try:
                seed = int(seed_item.split(".")[0].split("_")[-1])
                clients_activities, clients_weights, true_weights, true_availability, clients_ids = \
                    parse_history_file(seed_path)
                history_dict[activity][seed] = \
                    (clients_activities, clients_weights, true_weights, true_availability, clients_ids)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse seed from filename '{seed_item}'. Skipping.")
                continue

    return history_dict


def bar_plots(ax, method, weights, ids, width):
    """

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot

    method:str

    weights: numpy.array
        shape=(n_steps, n_clients)

    ids: 1-D numpy.array
        shape=(n_clients)

    width: float
        width of the bar


    Returns
    -------
        None
    """
    ax.bar(
        np.array(ids) - BAR_POSITION[method] * width,
        weights,
        width=width,
        label=LEGEND[method],
        color=COLORS[method],
        hatch=HATCHES[method]
    )


def plot_divergence(ax, method, cumulative_weights, true_weights, divergence_metric):
    """

    Parameters
    ----------
    ax:  matplotlib.axes._subplots.AxesSubplot

    method: str

    cumulative_weights: numpy.array
        shape = (n_steps, n_clients)

    true_weights: 1-D numpy.array
        shape = (n_clients)

    divergence_metric: fn

    Returns
    -------
        None
    """
    n_steps = len(cumulative_weights)

    divergence_list = []
    for t in range(n_steps):
        divergence = divergence_metric(true_weights, cumulative_weights[t])

        divergence_list.append(divergence)

    ax.plot(
        divergence_list,
        linewidth=LINE_WIDTH,
        label=f"{LEGEND[method]}",
        linestyle=f"{LINE_STYLES[method]}",
        color=COLORS[method]
    )


def set_divergence_ax_options(ax, y_label):
    """Set ax parameters for divergence plot


    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot


    y_label

    Returns
    -------
        None
    """

    ax.set_xlabel("Time step", fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_SIZE)

    ax.legend(fontsize=LEGEND_SIZE)

    ax.grid(True, linewidth=2)


def set_history_ax_options(ax, ids, y_label):
    """Set ax parameters of history bar plot


    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot

    ids: 1-D numpy.array
        shape=(n_clients)

    y_label

    Returns
    -------
        None
    """
    ax.set_xticks(ids)
    ax.set_xlabel("Clients", fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_SIZE)

    ax.legend(fontsize=LEGEND_SIZE)


def plot_all_history(history_dir, save_dir):
    """generates all plots related to history

    Plots the following:
        * cumulative weights
        * estimated activity
        * total variation distance
        * chi-square distance


    Parameters
    ----------

    history_dir: str
        directory containing all history; see `gather_history`

    save_dir: str
        directory to save the plots; will be created if does not exist


    Returns
    -------
        None

    """
    os.makedirs(save_dir, exist_ok=True)

    history_dict = gather_history(history_dir)
    if not history_dict:
        print("No history found to plot.")
        return

    cumulative_weights_fig, cumulative_weights_ax = plt.subplots(figsize=FIG_SIZE)
    observed_activities_fig, observed_activities_ax = plt.subplots(figsize=FIG_SIZE)
    tv_divergence_fig, tv_divergence_ax = plt.subplots(figsize=FIG_SIZE)
    chi2_divergence_fig, chi2_divergence_ax = plt.subplots(figsize=FIG_SIZE)

    true_weights = []
    true_availability = []
    clients_ids = []

    for method in METHODS:
        if method not in history_dict:
            continue
            
        clients_activities_array = []
        clients_weights_array = []
        true_weights_array = []
        true_availability_array = []
        clients_ids_array = []

        for seed in history_dict[method]:
            clients_activities, clients_weights, true_weights, true_availability, clients_ids = \
                history_dict[method][seed]

            clients_activities_array.append(clients_activities)
            clients_weights_array.append(clients_weights)
            true_weights_array.append(true_weights)
            true_availability_array.append(true_availability)
            clients_ids_array.append(clients_ids)

        if not clients_ids_array:
            continue

        clients_activities = np.array(clients_activities_array)
        clients_weights = np.array(clients_weights_array)
        true_weights = np.array(true_weights_array)
        true_availability = np.array(true_availability_array)
        clients_ids = np.array(clients_ids_array)

        bar_plots(
            ax=cumulative_weights_ax,
            method=method,
            weights=clients_weights.mean(axis=0).sum(axis=0) / clients_weights.mean(axis=0).sum(),
            ids=clients_ids[0],
            width=BAR_WIDTH
        )

        if method == "unbiased":
            bar_plots(
                ax=observed_activities_ax,
                method="observed",
                weights=clients_activities.mean(axis=0).sum(axis=0) / len(clients_activities.mean(axis=0)),
                ids=clients_ids[0],
                width=LARGE_BAR_WIDTH
            )

        with np.errstate(divide='ignore', invalid='ignore'):
            cumulative_weights = \
                clients_weights.mean(axis=0).cumsum(axis=0) / \
                clients_weights.mean(axis=0).cumsum(axis=0).sum(axis=1, keepdims=True)

        cumulative_weights = np.nan_to_num(cumulative_weights)

        plot_divergence(
            ax=tv_divergence_ax,
            method=method,
            cumulative_weights=cumulative_weights,
            true_weights=true_weights.mean(axis=0),
            divergence_metric=tv_distance
        )

        plot_divergence(
            ax=chi2_divergence_ax,
            method=method,
            cumulative_weights=cumulative_weights,
            true_weights=true_weights.mean(axis=0),
            divergence_metric=chi2_distance
        )

    if true_weights.size == 0 or clients_ids.size == 0:
        print("Not enough data to plot summary bars.")
        return
        
    bar_plots(
        ax=cumulative_weights_ax,
        method="true",
        weights=true_weights.mean(axis=0) / true_weights.mean(axis=0).sum(),
        ids=clients_ids[0],
        width=BAR_WIDTH
    )

    bar_plots(
        ax=observed_activities_ax,
        method="expected",
        weights=true_availability.mean(axis=0),
        ids=clients_ids[0],
        width=LARGE_BAR_WIDTH
    )

    set_history_ax_options(
        ax=cumulative_weights_ax,
        ids=clients_ids[0].tolist(),
        y_label="Cumulative weight"
    )

    set_history_ax_options(
        ax=observed_activities_ax,
        ids=clients_ids[0].tolist(),
        y_label="Observed availability"
    )

    set_divergence_ax_options(
        ax=tv_divergence_ax,
        y_label=r"$TV(\alpha, p^{(t)})$"
    )

    set_divergence_ax_options(
        ax=chi2_divergence_ax,
        y_label=r"$\chi^{2}(\alpha||p^{(t)})$"
    )

    cumulative_weights_fig.savefig(
        os.path.join(save_dir, "cumulative-weights.pdf"), bbox_inches='tight'
    )

    observed_activities_fig.savefig(
        os.path.join(save_dir, "estimated-activities.pdf"), bbox_inches='tight'
    )

    tv_divergence_fig.savefig(
        os.path.join(save_dir, "tv-divergence.pdf"), bbox_inches='tight'
    )

    chi2_divergence_fig.savefig(
        os.path.join(save_dir, "chi2-divergence.pdf"), bbox_inches='tight'
    )

    plt.close()


def plot_participation_heatmap(history_dir, save_dir):
    """PLot participation/availability heatmap

    Shows the availability of each client during a training run for different seed values.
    A figure is generated for each SEED_VALUE, and saved to "save_dir/SEED_VALUE"

    Parameters
    ----------
    history_dir: str
        directory containing all history; see `gather_history`

    save_dir: str
        directory to save the plots; will be created if does not exist


    Returns
    -------
        None

    """
    os.makedirs(save_dir, exist_ok=True)

    history_dict = gather_history(history_dir)

    if "markov" not in history_dict:
        warnings.warn("'markov' is not found in history, no plot is generated", RuntimeWarning)
        return

    for seed in history_dict["markov"]:
        clients_activities, clients_weights, true_weights, true_availability, clients_ids = \
            history_dict["markov"][seed]

        clients_selections = (clients_weights > 1e-6).astype(int)

        heatmap = clients_selections + clients_activities

        fig, ax = plt.subplots(figsize=HEATMAP_FIG_SIZE)

        temp = ax.imshow(heatmap.T, cmap=plt.get_cmap('YlGn', 3))
        cbar = fig.colorbar(temp, ax=ax, shrink=0.18, ticks=[0, 1, 2])

        cbar.ax.set_yticklabels(
            ["Inactive", "Active,\nbut not \nselected", "Selected"],
            fontsize=FONT_SIZE // 2
        )

        ax.set_ylabel("Clients", fontsize=FONT_SIZE)
        ax.set_xlabel("Communication round", fontsize=FONT_SIZE)

        ax.tick_params(axis='both', labelsize=TICKS_SIZE)

        save_path = os.path.join(save_dir, f"{seed}.pdf")
        plt.savefig(save_path, bbox_inches='tight')

        plt.close()


def gather_logs(logs_dir, tag):
    """
    CORRECTED log gatherer for the FLAT directory structure created by run_experiment.py.
    It looks for the 'global' folder directly inside the provided logs_dir.
    """
    logs_dict = dict()

    # The global log directory is expected to be directly inside the logs_dir
    global_events_dir = os.path.join(logs_dir, "global")

    if not os.path.exists(global_events_dir):
        # This is the only check we need now.
        return logs_dict # Return empty dict if the global folder is missing

    steps, values = parse_tf_events_file(global_events_dir, tag=tag)

    if values:
        # We use a dummy method ('markov') and seed (42) to fit the data structure
        # that the plot_logs_dict function expects to receive.
        logs_dict["markov"] = {42: (steps, values)}
    
    return logs_dict


def smooth_results(results_array, steps, discount_coeff=1.0):
    """

    Parameters
    ----------
    results_array: numpy.array
        shape=(n_trials, time_steps)

    steps: 1-D numpy.array

    discount_coeff: float


    Returns
    -------
        numpy.array: shape=(n_trials, time_steps)
    """
    if discount_coeff <= 1e-2:
        return results_array

    # Ensure steps is a numpy array for vectorized operations
    steps = np.array(steps)
    
    # Handle empty or inconsistent steps/results
    if steps.size == 0 or results_array.shape[1] != steps.size:
        return np.array([]) # Return empty array if there's a mismatch

    with np.errstate(divide='ignore', invalid='ignore'):
        discount_factors = np.power(1 / discount_coeff, steps)
        discount_factors_cumsum = discount_factors.cumsum()
        # Avoid division by zero if the cumsum is zero
        discount_factors = np.divide(discount_factors, discount_factors_cumsum, out=np.zeros_like(discount_factors), where=discount_factors_cumsum!=0)

    smooth_results_array = results_array.cumsum(axis=1) * discount_factors

    return smooth_results_array


def plot_logs_dict(logs_dict, discount_coeff, tag, save_path):
    """plots logs

    Parameters
    ----------
    logs_dict: Dict[str: Dict[int: Tuple[List[int], List[float]]]

    discount_coeff: float
        parameter used to smooth the curve, `0` corresponds to non smoothed curve and `1` to cumulative average

    tag: str

    save_path: str
        path given as '.pdf' file. Parent folder should already exist

    Returns
    -------
        None

    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for method in METHODS:
        if method not in logs_dict:
            continue
            
        results_array = []
        steps_list = []

        for seed in logs_dict[method]:
            steps, values = logs_dict[method][seed]
            # Ensure consistent lengths for np.array conversion
            if steps:
                steps_list.append(steps)
                results_array.append(values)
        
        # Skip method if no valid data was found
        if not results_array:
            continue

        # Pad shorter runs to the length of the longest run to create a rectangular array
        max_len = max(len(r) for r in results_array)
        results_array_padded = np.array([r + [r[-1]] * (max_len - len(r)) for r in results_array])
        steps = steps_list[np.argmax([len(s) for s in steps_list])] # Use steps from the longest run
        
        smooth_results_array = \
            smooth_results(results_array=results_array_padded, steps=steps, discount_coeff=discount_coeff)

        if smooth_results_array.size > 0:
            ax.plot(
                steps,
                smooth_results_array.mean(axis=0),
                linewidth=LINE_WIDTH,
                marker=MARKERS[method],
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_WIDTH,
                label=f"{LEGEND[method]}",
                color=COLORS[method]
            )

    ax.grid(True, linewidth=2)

    if discount_coeff == 1.0:
        ax.set_ylabel(AXE_LABELS_AVG[tag], fontsize=FONT_SIZE)
    else:
        ax.set_ylabel(AXE_LABELS[tag], fontsize=FONT_SIZE)
    ax.set_xlabel("Communication round", fontsize=FONT_SIZE)

    ax.tick_params(axis='both', labelsize=TICKS_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)

    plt.savefig(save_path, bbox_inches='tight')

    plt.close()


def plot_all_logs(logs_dir, save_dir):
    """

    Parameters
    ----------
    logs_dir: str
        directory containing all the logs; see `gather_logs`

    save_dir: str
        directory to save the plots; will be created if does not exist

    Returns
    -------
        None

    """
    os.makedirs(save_dir, exist_ok=True)

    for tag in FILES_NAMES:
        logs_dict = gather_logs(logs_dir, tag)
        
        if not any(logs_dict.values()): # Check if any data was gathered
            print(f"No log data found for tag '{tag}' in directory '{logs_dir}'. Skipping plot.")
            continue

        plot_logs_dict(
            logs_dict=logs_dict,
            discount_coeff=0.0,
            tag=tag,
            save_path=os.path.join(save_dir, FILES_NAMES[tag])
        )

        plot_logs_dict(
            logs_dict=logs_dict,
            discount_coeff=1.0,
            tag=tag,
            save_path=os.path.join(save_dir, f"smooth_{FILES_NAMES[tag]}")
        )