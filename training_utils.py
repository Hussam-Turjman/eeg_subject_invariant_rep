import json
import typing

import numpy as np

from subject_invariant_rep.utils.resources import machine
import argparse
import os
from easydict import EasyDict

from subject_invariant_rep.utils.path import Path
from abc import ABC, abstractmethod
import multiprocessing as mp
import torch.multiprocessing as torch_mp


def create_args(subject: typing.List[int] = None,
                exp: typing.List[int] = None,
                train_type: str = None,
                dataset_name: str = None,
                GPU: str = None,
                class_combo: str = None,
                continue_training: bool = None,
                ):
    maybe_args = {
        'subject': subject,
        'exp': exp,
        'train_type': train_type,
        'GPU': GPU,
        'class_combo': class_combo,
        'continue_training': continue_training,
        'dataset_name': dataset_name
    }
    if all([v is None for v in maybe_args.values()]):
        pass
    else:
        maybe_args = EasyDict(maybe_args)
        return maybe_args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str,
                        default=None,
                        help="Dataset name ex. nback, mmi",
                        required=True)
    parser.add_argument('--subject', required=True, type=int,
                        help='list of test subject id, None=all subject')
    parser.add_argument('--exp', required=True, type=int,
                        help='list of experiments id ')
    parser.add_argument('--train_type', type=str, required=True,
                        help='Train type: ex. subject_dependent, subject_independent')
    parser.add_argument('--GPU', type=str, default='0', help='GPU ID')
    parser.add_argument('--class_combo', type=str, required=True, help='Class combination .i.e n0_n1')
    parser.add_argument("--continue", dest="continue_training", action="store_true")
    args = parser.parse_args()

    return args


class TrainerInterface(ABC):
    def __init__(self, subject: int,
                 exp: int,
                 dataset_name: str,
                 train_type: str,
                 GPU: str,
                 class_combo: str,
                 continue_training: bool,
                 model_base_name: str,
                 ):

        self.model_base_name = model_base_name
        self.subject = subject
        self.exp = exp
        self.dataset_name = dataset_name
        self.train_type = train_type
        self.GPU = GPU
        self.class_combo = class_combo
        self.continue_training = continue_training
        self._init()
        print(self)

    def _init(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.device = machine.select_gpu_device(self.GPU)
        # print(machine)
        input_dir = Path( "datasets")
        self.input_dir = input_dir
        self.output_dir = Path("training_results", self.dataset_name, self.model_base_name)
        self.output_dir.make(directory=True, override=False, ignore_errors=True)
        self.exp_path = Path("configs", self.model_base_name, f"exp{self.exp}.json")
        self.raw_config = json.load(open(input_dir.copy().join(self.dataset_name, "raw.json").path, "r"))
        self.time_domain_config = json.load(
            open(input_dir.copy().join(self.dataset_name, "time_domain.json").path, "r"))
        self.sensor_positions = json.load(open(Path("configs", "sensor_positions.json").path, "r"))

    def __repr__(self):
        return f"dataset: {self.dataset_name}, subject: {self.subject}, exp: {self.exp}, train_type: {self.train_type}, GPU: {self.GPU}, class_combo: {self.class_combo}, continue_training: {self.continue_training}"

    def start(self):
        exp = self.exp_path
        exp: Path
        exp_name = exp.file_name().split(".")[0]
        model_output_dir = self.output_dir.copy().join(exp_name, self.class_combo, self.train_type)
        model_output_dir.make(directory=True, override=False, ignore_errors=True)
        model_config = json.load(open(exp.path, "r"))
        self.on_exp(model_output_dir, model_config)
        n_subjects = self.raw_config["n_subjects"]
        valid_subjects = list(range(1, n_subjects + 1))
        assert self.subject in valid_subjects, f"subject {self.subject} not in {valid_subjects}"
        subject = self.subject
        if self.continue_training:
            model_name = 'S{:03d}_Y_results.npz'.format(subject)
            results_path = model_output_dir.copy().join(model_name)
            info = f"{subject}, dataset_name {self.dataset_name} train_type {self.train_type} class_combo {self.class_combo} exp {exp_name}"
            if results_path.exists():
                print(
                    f"Skipping subject : {info}."
                    f" Results already exists")
                return
            else:
                print(
                    f"Training subject {info}.")
        self.on_training_started(subject=subject, exp=exp_name)

    @abstractmethod
    def on_exp(self, model_output_dir: Path, model_config: dict):
        pass

    @abstractmethod
    def on_training_started(self, subject: int, exp: str):
        pass


def get_results(callback, n_folds, cpu_count=2, use_torch_mp=False):
    if cpu_count is None:
        results = [callback(fold) for fold in range(1, n_folds + 1)]
    else:
        assert cpu_count > 0, "cpu_count must be greater than 0"
        if use_torch_mp:
            # torch_mp.set_start_method('spawn')
            pool = torch_mp.Pool(cpu_count)
        else:
            pool = mp.Pool(cpu_count)
        results = pool.map(callback, range(1, n_folds + 1))
        pool.close()
        pool.join()
    return results


def save_results(callback, n_folds, model_output_dir, subject, cpu_count=None, use_torch_mp=False):
    results = get_results(callback=callback, n_folds=n_folds, cpu_count=cpu_count, use_torch_mp=use_torch_mp)
    best = max(results, key=lambda x: x["accuracy"])
    worst = min(results, key=lambda x: x["accuracy"])
    print("Worst fold: ", worst["fold"], "Worst accuracy: ", worst["accuracy"])
    print("Best fold: ", best["fold"], "Best accuracy: ", best["accuracy"])
    for r in results:
        print("Fold: ", r["fold"], "Accuracy: ", r["accuracy"])

    model_name = 'S{:03d}_Y_results.npz'.format(subject)
    if best["accuracy"] > 0.0:
        if best["embeddings"] is not None:
            print("Embedding shape ", best["embeddings"].shape)
        np.savez(model_output_dir.copy().join(model_name).path,
                 **best
                 )
    else:
        print("Skipping saving results for subject {}. All folds have 0.0 accuracy".format(subject))


__all__ = ["create_args", "TrainerInterface", "get_results", "save_results"]
