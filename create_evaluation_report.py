import re
import re
import time
import typing
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subject_invariant_rep.models.utils import to_labels, to_logits
from subject_invariant_rep.utils.metrics import single_value_metrics, conf_matrix
from subject_invariant_rep.utils.path import Path
from subject_invariant_rep.utils.plot import create_pdf_from_figures
from subject_invariant_rep.visualization import plot_embeddings

metric_names = ["accuracy", "f1_score", "precision",
                "recall", "balanced_accuracy", "roc",
                "average_precision"]


def convert_to_correct_format(y_true, y_pred, embeddings, model_name):
    if model_name == "MIN2Net" or model_name == "spectral_spatial_cnn" or model_name == "EEGNet":
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        if model_name == "MIN2Net" or model_name == "EEGNet":
            embeddings = embeddings.squeeze()
    return y_true, y_pred, embeddings


def extract_values_from_dict(y_results: dict, model_name, num_classes, val=False,train=False):
    y_true_key = "y_true"
    y_pred_key = "y_pred"
    embedding_key = "embeddings"
    if train:
        y_true_key = "y_true_train"
        y_pred_key = "y_pred_train"
        embedding_key = "embeddings_train"
    if val:
        y_true_key = "y_true_val"
        y_pred_key = "y_pred_val"
        embedding_key = "embeddings_val"
    y_true = y_results[y_true_key]
    y_pred = y_results[y_pred_key]
    embeddings_name = embedding_key
    if embeddings_name in y_results.keys():
        embeddings = y_results[embeddings_name]
    else:
        embeddings = None
    y_true, y_pred, embeddings = convert_to_correct_format(y_true, y_pred, embeddings, model_name)
    if model_name == "ssl" and val:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    y_pred_logits = to_logits(y_pred, num_classes)
    y_true_logits = to_logits(y_true, num_classes)
    metrics = single_value_metrics(
        predictions=y_pred, predictions_logits=y_pred_logits,
        targets=y_true, targets_logits=y_true_logits,
        num_classes=num_classes)
    if val:
        metrics = {f"{k}_val": v for k, v in metrics.items()}
    return y_true, y_pred, embeddings, metrics


def block_creator(file, root, subdirs, input_dir: Path):
    p = Path(root, file)
    extension = str(p).split(".")[-1]
    if extension != "npz":
        return None
    sub_path = p.copy().relative_path(input_dir.path)
    sub_path = Path(sub_path).split()

    dataset_name = sub_path[0]
    model_name = sub_path[1]
    exp_name = sub_path[2]
    class_combo = sub_path[3]
    train_type = sub_path[4]
    subject = sub_path[5].split("_")[0]

    all_info = [model_name, exp_name, class_combo, train_type, subject]
    num_classes = len(class_combo.split("_"))
    # MIN2Net/spectral_spatial_cnn
    # (1, 20)
    # (1, 20)
    # (1, 20, 8)

    # SSL
    # (20, 2)
    # (20, 2)
    # (20, 112)

    results = np.load(p.path)
    accuracy = results["accuracy"].item()

    block = {
        "model_name": model_name,
        "exp_name": exp_name,
        "class_combo": class_combo,
        "train_type": train_type,
        "subject": subject,
        "dataset_name": dataset_name,
        "accuracy": accuracy,
        "path": p.path
    }

    return block


def walk_callback(root, subdirs, files, input_dir: Path, blocks):
    print("Processing files in {} ...".format(root))
    callback = partial(block_creator, root=root,
                       subdirs=subdirs,
                       input_dir=input_dir)
    results = [callback(file) for file in files]
    # pool = mp.Pool(mp.cpu_count() // 2)
    # results = pool.map(callback, files)
    # pool.close()
    # pool.join()

    for block in results:
        if block is not None:
            blocks.append(block)


def create_name(chosen_dataset, chosen_model, chosen_exp, chosen_class_combo, chosen_train_type):
    return f"{chosen_dataset}|{chosen_model}|{chosen_exp}|{chosen_class_combo}|{chosen_train_type}"


def get_components_from_name(name):
    res = name.split("|")
    return {
        "chosen_dataset": res[0],
        "chosen_model": res[1],
        "chosen_exp": res[2],
        "chosen_class_combo": res[3],
        "chosen_train_type": res[4]
    }


def valid_embeddings(embeddings):
    if embeddings is None:
        return False
    if all(v is None for v in embeddings):
        return False
    return not np.isnan(embeddings).all()


def get_results(df_grouped, chosen_dataset, chosen_model, chosen_exp,
                chosen_class_combo, chosen_train_type, metric_names_in,
                val=False,train=False):
    all_subjects_y_true = []
    all_subjects_y_pred = []
    all_subjects_embeddings = []
    metrics = {metric_name: [] for metric_name in metric_names_in}

    name = create_name(
        chosen_dataset=chosen_dataset,
        chosen_model=chosen_model,
        chosen_exp=chosen_exp,
        chosen_class_combo=chosen_class_combo,
        chosen_train_type=chosen_train_type
    )
    for info, row in df_grouped:
        dataset_name = info[0]
        model_name = info[1]
        exp_name = info[2]
        class_combo = info[3]
        train_type = info[4]

        if dataset_name != chosen_dataset:
            continue
        if model_name != chosen_model:
            continue
        if train_type != chosen_train_type:
            continue
        if class_combo != chosen_class_combo:
            continue
        if exp_name != chosen_exp:
            continue
        paths = row["path"].to_numpy()
        for data_path in paths:
            print(f"Loading {data_path}")
            data = np.load(data_path, allow_pickle=True)
            num_classes = len(class_combo.split("_"))
            print(model_name)
            y_true, y_pred, embeddings, extracted_metrics = extract_values_from_dict(y_results=data,
                                                                                     model_name=model_name,
                                                                                     num_classes=num_classes, val=val,train=train)
            if embeddings is not None:
                if embeddings.size == 1:
                    embeddings = None

            for metric_name in metric_names_in:
                original_metric_name = metric_name
                metric_name = f"{metric_name}_val" if val else metric_name
                metrics[original_metric_name].append(extracted_metrics[metric_name])

            all_subjects_y_true.append(y_true)
            all_subjects_y_pred.append(y_pred)
            all_subjects_embeddings.append(embeddings)

    y_true = all_subjects_y_true  # np.stack(all_subjects_y_true, axis=0)
    y_pred = all_subjects_y_pred  # np.stack(all_subjects_y_pred, axis=0)
    embeddings = all_subjects_embeddings  # np.stack(all_subjects_embeddings, axis=0)
    assert len(y_true) == len(y_pred) == len(embeddings), "y_true, y_pred and embeddings must have the same length"

    return y_true, y_pred, embeddings, metrics, name


def get_results_from_configs(df_grouped, configs: typing.List[dict], val=False,train=False):
    results = {}
    for config in configs:
        y_true, y_pred, embeddings, metrics, name = get_results(df_grouped=df_grouped,
                                                                chosen_dataset=config["chosen_dataset"],
                                                                chosen_model=config["chosen_model"],
                                                                chosen_exp=config["chosen_exp"],
                                                                chosen_class_combo=config["chosen_class_combo"],
                                                                chosen_train_type=config["chosen_train_type"],
                                                                metric_names_in=metric_names,
                                                                train=train,
                                                                val=val
                                                                )
        if val:
            metrics = {k.replace("_val", ""): v for k, v in metrics.items()}

        results[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "embeddings": embeddings,
            "metrics": metrics,
            "config": config,
            "subject_count": len(y_true)

        }

    return results


def annotate_best_score(ax, y_values, x_values, color=None):
    best_score_idx = np.argmax(y_values)
    best_score = y_values[best_score_idx]
    ax.annotate(f"{int(best_score)}%", (x_values[best_score_idx], best_score + 1))
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            x_values[best_score_idx]
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        color=color,
        marker="x",
        markeredgewidth=3,
        ms=8,
    )


def plot_accuracy_and_recall(n_subjects, accuracies, recalls, f1_scores, model_names, window_title=""):
    assert isinstance(accuracies, np.ndarray), "accuracies must be a numpy array"
    assert isinstance(recalls, np.ndarray), "recalls must be a numpy array"
    assert isinstance(f1_scores, np.ndarray), "f1_scores must be a numpy array"
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False)
    axes = axes.flatten()
    subject_ids = np.arange(1, n_subjects + 1)

    for idx in range(len(axes)):
        # axes[idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        # axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
        xmin = np.min(subject_ids) - 1
        xmax = np.max(subject_ids) + 1
        axes[idx].set_xlim([xmin, xmax])
        axes[idx].set_yticks(np.arange(0, 100, 10), minor=True)
    for accuracy, recall, f1_score, model_name in zip(accuracies, recalls, f1_scores, model_names):
        accuracy *= 100
        recall *= 100
        f1_score *= 100
        p = axes[0].plot(subject_ids, accuracy, '--o', label=model_name.capitalize())
        color = "g"  # p[0].get_color()
        annotate_best_score(axes[0], accuracy, subject_ids, color=color)
        axes[1].plot(subject_ids, recall, '--o', label=model_name.capitalize())
        annotate_best_score(axes[1], recall, subject_ids, color=color)
        axes[2].plot(subject_ids, f1_score, '--o', label=model_name.capitalize())
        annotate_best_score(axes[2], f1_score, subject_ids, color=color)

    axes[0].set_ylabel("Accuracy [%]")
    axes[0].set_xlabel("Subject")
    axes[1].set_ylabel("Recall [%]")
    axes[1].set_xlabel("Subject")
    axes[2].set_ylabel("F1 Score [%]")
    axes[2].set_xlabel("Subject")
    # fig.legend()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    # Set title
    fig.canvas.manager.set_window_title(window_title)
    return fig, axes


def get_best_subject_values(result: dict, subject_metric="accuracy", best=True):
    y_pred = result["y_pred"]
    y_true = result["y_true"]
    embeddings = result["embeddings"]
    chosen_class_combo = result["config"]["chosen_class_combo"]
    class_combo_mapper = chosen_class_combo.split("_")
    if best:
        best_subject_idx = np.argmax(result["metrics"][subject_metric])
    else:
        best_subject_idx = np.argmin(result["metrics"][subject_metric])
    y_pred = y_pred[best_subject_idx]
    embeddings = embeddings[best_subject_idx]
    y_true = y_true[best_subject_idx]
    y_true = y_true.astype(np.int32)
    y_true = [class_combo_mapper[i] for i in y_true]
    y_true = np.array(y_true)
    y_pred = [class_combo_mapper[i] for i in y_pred]
    y_pred = np.array(y_pred)
    return y_true, y_pred, embeddings, best_subject_idx


def compare_conf_matrices(y_true, y_pred, titles):
    number_of_matrices = len(y_true)
    nrows = 1
    ncols = number_of_matrices
    if number_of_matrices > 2:
        nrows = number_of_matrices // 2
        ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    if number_of_matrices > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for idx in range(number_of_matrices):
        conf_matrix(target=y_true[idx], prediction=y_pred[idx], normalize=True,
                    title=titles[idx],
                    ax=axes[idx])
    return fig, axes


def compare_embeddings(embeddings, labels, titles):
    number_of_embeddings = len(embeddings)
    fig, axes = plt.subplots(nrows=1, ncols=number_of_embeddings, sharex=True, sharey=True)
    axes = axes.flatten()
    for idx in range(number_of_embeddings):
        plot_embeddings(method="tsne", data=embeddings[idx],
                        labels=labels[idx],
                        title=None,
                        perplexity=min(labels[idx].shape[0] - 1, 50),
                        save_path=None,
                        ax=axes[idx],
                        marker_size=400
                        )
        axes[idx].set_title(titles[idx])


def plot_embeddings_and_conf(y_true, y_pred, embeddings, titles, figure_title=None, W=12.5):
    assert len(y_true) == len(y_pred) == len(embeddings)

    nrows = len(y_true)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=False, sharey=False, figsize=(W, W / 2))
    axes = axes.flatten()
    j = 0
    for i in range(0, nrows * 2, 2):
        conf_matrix(target=y_true[j], prediction=y_pred[j], normalize=True,
                    title=titles[j],
                    ax=axes[i])
        if valid_embeddings(embeddings[j]):
            plot_embeddings(method="tsne", data=embeddings[j],
                            labels=y_true[j],
                            title=titles[j],
                            perplexity=min(y_true[j].shape[0] - 1, 50),
                            save_path=None,
                            ax=axes[i + 1],
                            marker_size=400
                            )
        else:
            print(f"Embeddings are None {titles[j]}")
        j += 1
    if figure_title is not None:
        fig.canvas.manager.set_window_title(figure_title)
    return fig, axes


def summarize_metrics(results: dict):
    metrics = {}
    for name, result in results.items():
        metrics[name] = {}
        for metric_name in metric_names:
            metrics[name][metric_name] = round(np.mean(result["metrics"][metric_name]), 2)
            metrics[name][f"{metric_name}_min"] = round(np.min(result["metrics"][metric_name]), 2)
            metrics[name][f"{metric_name}_max"] = round(np.max(result["metrics"][metric_name]), 2)
            metrics[name][f"{metric_name}_diff"] = (metrics[name][f"{metric_name}_max"] - metrics[name][
                f"{metric_name}"]) + \
                                                   (metrics[name][f"{metric_name}"] - metrics[name][
                                                       f"{metric_name}_min"])
            metrics[name][f"{metric_name}_diff"] /= 2
            metrics[name][f"{metric_name}_diff"] = round(metrics[name][f"{metric_name}_diff"], 2)
    return metrics


def check_type(val):
    out = []
    if type(val) == str:
        out.append(val)
    elif type(val) == list:
        out += val
    return out


def stack_values(best_values, key="y_true", model_name=None, class_combo=None, dataset_name=None, train_type=None):
    stacked = []
    model_name_constraint = check_type(model_name)
    class_combo_constraint = check_type(class_combo)
    dataset_name_constraint = check_type(dataset_name)
    train_type_constraint = check_type(train_type)
    for value in best_values:

        if len(model_name_constraint) > 0:
            if value["model_name"] not in model_name_constraint:
                continue
        if len(class_combo_constraint) > 0:
            if value["class_combo"] not in class_combo_constraint:
                continue
        if len(dataset_name_constraint) > 0:
            if value["dataset_name"] not in dataset_name_constraint:
                continue

        if len(train_type_constraint) > 0:
            if value["train_type"] not in train_type_constraint:
                continue
        stacked.append(value[key])
    return stacked


def box_plot_compare_tasks(results: dict, title, chosen_train_type, chosen_dataset):
    organized_metrics = organize_metrics(results=results,
                                         chosen_train_type=chosen_train_type,
                                         chosen_dataset=chosen_dataset)
    model_names = list(organized_metrics.keys())
    if len(model_names) == 0:
        return None, None
    fig, axes = plt.subplots(nrows=len(model_names), ncols=2, sharex=True)
    axes = axes.flatten()
    idx = 0
    colors = ['pink', 'lightblue', 'lightgreen', 'lightgray', 'lightyellow', 'lightcyan', 'lightcoral', 'lightpink']
    bplots = []

    for model_name in model_names:
        for metric_name in ["accuracy", "f1_score"]:
            all_data, labels = get_tasks_metric(organized_results=organized_metrics,
                                                chosen_model_name=model_name,
                                                metric_name=metric_name)

            dataset_name, real_model_name = model_name.split("|")

            if dataset_name == "nback":
                labels_out = []
                for label in labels:
                    label_parts = label.split("_")
                    reconstructed_label = ""
                    for part in label_parts:
                        if part == "baseline":
                            reconstructed_label += "baseline_"
                        else:
                            reconstructed_label += "".join(re.findall(r'\d+', part))
                    reconstructed_label = reconstructed_label.split("_")
                    reconstructed_label = [f"n{task}" if task != "baseline" else task for task in reconstructed_label]
                    reconstructed_label = "_".join(reconstructed_label)
                    labels_out.append(reconstructed_label)
                labels = labels_out

            bplot = axes[idx].boxplot(all_data,
                                      vert=True,  # vertical box alignment
                                      patch_artist=True,  # fill with color
                                      notch=True,  # notch shape
                                      labels=labels)  # will be used to label x-ticks
            t = "I" if chosen_train_type == "subject_independent" else "D"
            if real_model_name == "ssl":
                real_model_name = "Subject-Aware"
            axes[idx].set_title(real_model_name  # model_name.upper() + f"|{t}"
                                )
            axes[idx].set_ylabel(metric_name.upper())
            axes[idx].yaxis.grid(True)
            bplots.append(bplot)
            idx += 1

    for bplot in bplots:
        for patch, color in zip(bplot['boxes'], colors[:len(bplots)]):
            patch.set_facecolor(color)

    if title is not None:
        # axes[0].set_title(title.upper())
        fig.canvas.manager.set_window_title(title.upper())
    return fig, axes


def get_tasks_metric(organized_results: dict, chosen_model_name: str, metric_name: str):
    labels, values = [], []

    for model_name, result in organized_results.items():
        if model_name != chosen_model_name:
            continue

        for class_combo in result.keys():
            labels.append(class_combo)
            values.append(result[class_combo][metric_name])
    return values, labels


def organize_metrics(results: dict, chosen_train_type: str, chosen_dataset: str, chosen_exp: str = None):
    original_results = {}
    for name, result in results.items():
        components = get_components_from_name(name)
        dataset_name = components["chosen_dataset"]
        train_type = components["chosen_train_type"]
        exp_name = components["chosen_exp"]
        if train_type != chosen_train_type:
            continue
        if dataset_name != chosen_dataset:
            continue
        if chosen_exp is not None:
            if exp_name != chosen_exp:
                continue
        print(f"Adding {name} because train type is {train_type}")
        model_name = dataset_name + "|" + components["chosen_model"]
        class_combo = components["chosen_class_combo"]
        if model_name not in original_results.keys():
            original_results[model_name] = {}
        if class_combo not in original_results[model_name].keys():
            original_results[model_name][class_combo] = {}
        original_results[model_name][class_combo] = result["metrics"]
    return original_results


def get_values(best_subjects, model_name=None, class_combo=None, dataset_name=None, train_type=None):
    y_true = stack_values(best_values=best_subjects, key="y_true", model_name=model_name, class_combo=class_combo,
                          dataset_name=dataset_name, train_type=train_type)
    y_pred = stack_values(best_values=best_subjects, key="y_pred", model_name=model_name, class_combo=class_combo,
                          dataset_name=dataset_name, train_type=train_type)
    titles = stack_values(best_values=best_subjects, key="title", model_name=model_name, class_combo=class_combo,
                          dataset_name=dataset_name, train_type=train_type)
    embeddings = stack_values(best_values=best_subjects, key="embeddings", model_name=model_name,
                              class_combo=class_combo, dataset_name=dataset_name, train_type=train_type)
    return y_true, y_pred, titles, embeddings


def plot_best_subject_values(best_subjects):
    figs = []
    axes_count = 3
    for idx in range(0, len(best_subjects), axes_count):
        best = []
        best.extend(best_subjects[idx:idx + axes_count])

        all_y_true, all_y_pred, all_titles, all_embeddings = [], [], [], []
        all_model_names, all_class_combos, all_dataset_names, all_train_types = [], [], [], []
        for best_subject in best:
            model_name = best_subject["model_name"]
            class_combos = best_subject["class_combo"]
            dataset_name = best_subject["dataset_name"]
            train_type = best_subject["train_type"]
            title = best_subject["title"]
            all_model_names.append(model_name)
            all_class_combos.append(class_combos)
            all_dataset_names.append(dataset_name)
            all_train_types.append(train_type)
            all_titles.append(title)

        for model_name, class_combo, dataset_name, train_type in zip(all_model_names,
                                                                     all_class_combos,
                                                                     all_dataset_names,
                                                                     all_train_types):
            y_true, y_pred, titles, embeddings = get_values(best_subjects=best_subjects,
                                                            model_name=model_name,
                                                            class_combo=class_combo,
                                                            dataset_name=dataset_name,
                                                            train_type=train_type)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_titles.extend(titles)
            all_embeddings.extend(embeddings)

        y_true, y_pred, titles, embeddings = all_y_true, all_y_pred, all_titles, all_embeddings

        fig, axes = plot_embeddings_and_conf(y_true=y_true, y_pred=y_pred,
                                             embeddings=embeddings,
                                             titles=titles, )
        handles, labels = axes[1].get_legend_handles_labels()
        legend_data = {label: handle for label, handle in zip(labels, handles)}
        fig.legend(loc='upper right',
                   bbox_to_anchor=(1.0, 1.0),
                   handles=legend_data.values(),
                   labels=legend_data.keys(),
                   )
        figs.append(fig)
        plt.tight_layout()
    return figs


def extract_best_subjects(results: dict, accuracy_metric="accuracy", best=True):
    best_subjects = []
    accuracies, recalls, f1_scores = [], [], []
    for name, result in results.items():
        components = get_components_from_name(name)

        model_name = components["chosen_model"]
        dataset_name = components["chosen_dataset"]
        class_combo = components["chosen_class_combo"]
        train_type = components["chosen_train_type"]
        exp_name = components["chosen_exp"]

        accuracies.append(result["metrics"]["accuracy"])
        recalls.append(result["metrics"]["recall"])
        f1_scores.append(result["metrics"]["f1_score"])
        y_true, y_pred, embeddings, best_subject_idx = get_best_subject_values(result=result,
                                                                               subject_metric=accuracy_metric,
                                                                               best=best)
        train_type_name = "D" if train_type == "subject_dependent" else "I"
        if best_subject_idx in [3, 5, 9]:
            best_subject_idx += 1
        best_subjects.append({
            "y_true": y_true,
            "y_pred": y_pred,
            "embeddings": embeddings,
            "best_subject_idx": best_subject_idx,
            "title": f"{model_name}-{train_type_name}-" + f"{'Best' if best else 'Worst'}" + " P{:02d}".format(
                best_subject_idx + 1),
            "model_name": model_name,
            "class_combo": class_combo,
            "dataset_name": dataset_name,
            "train_type": train_type,
            "exp_name": exp_name
        })
    return best_subjects, accuracies, recalls, f1_scores


def compare_conf_mats(y_true, y_pred, y_true_val, y_pred_val, title):
    assert len(y_true) == len(y_pred) == len(y_true_val) == len(y_pred_val)

    nrows = len(y_true)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=False, sharey=False)
    axes = axes.flatten()
    j = 0
    for i in range(0, nrows * 2, 2):
        conf_matrix(target=y_true[j], prediction=y_pred[j], normalize=True,
                    title=None,
                    ax=axes[i])
        axes[i].set_title("Test")
        conf_matrix(target=y_true_val[j], prediction=y_pred_val[j], normalize=True,
                    title=None,
                    ax=axes[i + 1])
        axes[i + 1].set_title("Validation")
        j += 1

    if title is not None:
        fig.canvas.manager.set_window_title(title)
    return fig, axes


def compare_subjects(results, results_val, query_config: dict, output_path):
    figs = []
    # "chosen_model": model,
    # "chosen_exp": exp,
    # "chosen_class_combo": class_combo,
    # "chosen_train_type": train_type,

    for name in results.keys():
        components = get_components_from_name(name)
        if components["chosen_dataset"] != query_config["chosen_dataset"]:
            continue
        if components["chosen_model"] != query_config["chosen_model"]:
            continue
        if components["chosen_exp"] != query_config["chosen_exp"]:
            continue
        if components["chosen_class_combo"] != query_config["chosen_class_combo"]:
            continue
        if components["chosen_train_type"] != query_config["chosen_train_type"]:
            continue
        y_true, y_pred = results[name]["y_true"], results[name]["y_pred"]
        y_true_val, y_pred_val = results_val[name]["y_true"], results_val[name]["y_pred"]
        n_subjects = results[name]["subject_count"]
        print(f"Plotting {n_subjects} subjects")
        for idx in range(0, n_subjects, 2):
            fig, axes = compare_conf_mats(y_true=y_true[idx:idx + 2], y_pred=y_pred[idx:idx + 2],
                                          y_true_val=y_true_val[idx:idx + 2], y_pred_val=y_pred_val[idx:idx + 2],
                                          title=None)
            plt.tight_layout()
            figs.append(fig)
    dataset_name = query_config["chosen_dataset"]
    model_name = query_config["chosen_model"]
    training_type = query_config["chosen_train_type"]
    class_combo = query_config["chosen_class_combo"]
    create_pdf_from_figures(
        output_path=Path(output_path,
                         f"{dataset_name}_{model_name}_{training_type}_{class_combo}_all_subjects.pdf").path,
        figures=figs)
    return figs


def create_boxplot_pdf(results: dict, title: str, chosen_dataset,output_path):
    figs = []
    for train_type in ["subject_dependent", "subject_independent"]:
        fig, axes = box_plot_compare_tasks(results=results, title=title,
                                           chosen_train_type=train_type,
                                           chosen_dataset=chosen_dataset)
        if fig is None:
            continue
        figs.append(fig)
        plt.tight_layout()

    if len(figs) == 0:
        print(f"No figures to plot for :{chosen_dataset} {title} ")
        return
    create_pdf_from_figures(
        output_path=Path(output_path, f"{chosen_dataset}_{title}_boxplots.pdf").path,
        figures=figs)


def compare_test_and_val(configs, df_grouped, output_path, best=True, detailed=False):
    accuracy_metric = "accuracy"

    results = get_results_from_configs(df_grouped=df_grouped, configs=configs, val=False)
    results_val = get_results_from_configs(df_grouped=df_grouped, configs=configs, val=True)

    best_subjects, accuracies, recalls, f1_scores = extract_best_subjects(results=results,
                                                                          accuracy_metric=accuracy_metric,
                                                                          best=best)

    best_subjects_val, accuracies_val, recalls_val, f1_scores_val = extract_best_subjects(results=results_val,
                                                                                          accuracy_metric=accuracy_metric,
                                                                                          best=best)

    best_name = "Best" if best else "Worst"
    figs = plot_best_subject_values(best_subjects=best_subjects)
    create_pdf_from_figures(
        output_path=Path(output_path, f"all_{best_name}_results_test.pdf").path,
        figures=figs)
    figs = plot_best_subject_values(best_subjects=best_subjects_val)
    create_pdf_from_figures(
        output_path=Path(output_path, f"all_{best_name}_results_validation.pdf").path,
        figures=figs)

    #create_boxplot_pdf(results=results, title="Test", chosen_dataset="mmi",output_path=output_path)
    #create_boxplot_pdf(results=results_val, title="Validation", chosen_dataset="mmi",output_path=output_path)
    create_boxplot_pdf(results=results, title="Test", chosen_dataset="nback",output_path=output_path)
    create_boxplot_pdf(results=results_val, title="Validation", chosen_dataset="nback",output_path=output_path)

    if detailed:
        for query_config in configs:
            figs = compare_subjects(results=results, results_val=results_val,
                                    query_config=query_config,
                                    output_path=output_path)


def extract_df(path: str = "training_results"):
    start = time.time()

    input_dir = Path(path)
    blocks = []
    callback = partial(walk_callback, input_dir=input_dir, blocks=blocks)
    input_dir.walk(callback=callback, recursive=True)
    columns_names = list(blocks[0].keys())

    # df = pd.read_pickle("training_results/cropped_window.pkl")
    df = pd.DataFrame(blocks, columns=columns_names)
    df = df.round(2)

    columns_sort = ["dataset_name", "model_name", "exp_name", "class_combo", "train_type", "subject"]
    df = df.sort_values(columns_sort,
                        )
    accuracies = df.groupby(["dataset_name", "model_name", "exp_name", "class_combo", "train_type"])["accuracy"]

    accuracies = accuracies.agg(Subjects="count",
                                Accuracy="mean",
                                Accuracy_Max="max",
                                Accuracy_Min="min",
                                Best_Subject_ID=np.argmax,
                                Worst_Subject_ID=np.argmin

                                )
    accuracies = accuracies.round(2)
    accuracies.sort_values(by=["dataset_name", "model_name", "exp_name", "class_combo"], inplace=True)
    # fig, ax = plt.subplots()
    # ax.axis('tight')
    # ax.axis('off')
    #
    # the_table = ax.table(cellText=accuracies.values, colLabels=accuracies.columns, loc='center')
    # plt.tight_layout()
    # create_pdf_from_figures(output_path=Path("training_results", f"summary.pdf").path,
    #                         figures=[fig])
    # print(df)
    accuracies.to_html(f"{path}/summary.html")

    # df.to_pickle("training_results/cropped_window.pkl")
    end = time.time()
    models_by_dataset = df.groupby(["dataset_name", "model_name"])
    print(models_by_dataset["model_name"].unique().values)
    print("Time elapsed : %.2f seconds" % (end - start))
    print("Done loading raw data ...")
    out = df.groupby(["dataset_name", "model_name", "exp_name", "class_combo", "train_type"])
    configs = []
    for info, row in out:
        config = {
            "chosen_dataset": info[0],
            "chosen_model": info[1],
            "chosen_exp": info[2],
            "chosen_class_combo": info[3],
            "chosen_train_type": info[4],
        }
        configs.append(config)
    grouped = out
    return df, grouped, configs


def run():
    path = "training_results"
    df, grouped, configs = extract_df(path)
    compare_test_and_val(df_grouped=grouped, configs=configs,
                         best=True, detailed=False,
                         output_path=path)

    plt.show()


if __name__ == "__main__":
    run()
