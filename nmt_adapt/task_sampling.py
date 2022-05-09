import random
import typing
from itertools import combinations
from collections import defaultdict
import pickle

def build_confusion_matrix_from_eval_data(fp):
    """
    It takes a pickle file containing a evaluation data and constructs a confusion matrix

    Args:
      fp: filepath to the pickle file containing the evaluation data

    Returns:
      A dictionary of dictionaries. The outer dictionary is keyed by the ground truth mtag set. The
    inner dictionary is keyed by the predicted mtag set. The values are the confusion probabilities.
    """

    with open(fp, "rb") as f:
        eval_data = pickle.load(f)

    confusion_matrix = defaultdict(lambda: defaultdict(list))

    for gt_mtag_set in eval_data.keys():
        for lemma in eval_data[gt_mtag_set]:
            for example in eval_data[gt_mtag_set][lemma]:
                for pred_mtag_set, v in example["confusion"].items():
                    confusion_matrix[gt_mtag_set][pred_mtag_set] += [v]
                    confusion_matrix[gt_mtag_set]["sum"] += [v]

    # Normalize the confusion matrix to get confusion probabilities
    # Also includes correct class ('diagonal')
    for k, v in confusion_matrix.items():
        Z = sum(v["sum"])

        for kk, vv in v.items():
            v[kk] = sum(vv) / Z

        del v["sum"]

        confusion_matrix[k] = dict(v)
    confusion_matrix = dict(confusion_matrix)

    return confusion_matrix

class TaskSampler(object):
    """_summary_

    Args:
        index (_type_): _description_
    """

    def __init__(self, index):

        marginal_tasks = list(index.keys())

        # For all possible task pairings, find the lemmas present in both
        # Only records those with at least one difference in lemma script
        # Only records those with at least 2 lemmas
        lemma_intersection = defaultdict(set)
        for t1, t2 in combinations(marginal_tasks, 2):
            if t1.match(t2) >= 1:
                continue

            overlap = set.intersection(set(index[t1].keys()), set(index[t2].keys()))

            if len(overlap) >= 2:
                lemma_intersection[(t1, t2)].update(overlap)

        self.lemma_intersection = dict(lemma_intersection)
        self.task_weights = dict()

    def set_weights(self, weights_lookup: typing.Optional[typing.Dict[typing.Tuple[set, set], float]] = None):
        """
        The function takes a dictionary of dictionaries of weights, and assigns weights to each task in
        the task list.

        Args:
          weights_lookup (typing.Optional[typing.Dict[typing.Tuple[set, set], float]]): a dictionary of dictionaries, where the first key is a set of morphological tags, and the second key is another set of morphological tags. The value is a float.
        """

        if weights_lookup is not None:
            for t1, t2 in list(self.lemma_intersection.keys()):

                # Look for weight in confusion matrix
                w = weights_lookup.get(frozenset(t1.morph_tag_set), dict()).get(frozenset(t2.morph_tag_set), 0.0)
                w += weights_lookup.get(frozenset(t2.morph_tag_set), dict()).get(frozenset(t1.morph_tag_set), 0.0)

                # If weight is still not found, assign 0 weight
                if w > 0:
                    self.task_weights[(t1, t2)] = w

        else:
            for t1, t2 in list(self.lemma_intersection.keys()):
                self.task_weights[(t1, t2)] = 1

        Z = sum(self.task_weights.values())
        for k, w in self.task_weights.items():
            self.task_weights[k]  = w / Z

    def sample_tasks(self, informed: bool = True):
        """
        It samples two tasks from the task_weights dictionary.

        Args:
          informed (bool): whether to sample tasks from the informed distribution or the uniform
        distribution. Defaults to True

        Returns:
          A tuple of two tasks.
        """

        if informed:
            t1, t2 = random.choices(list(self.task_weights.keys()), weights=self.task_weights.values(), k=1)[0]

        else:
            t1, t2 = random.choices(list(self.lemma_intersection.keys()), k=1)[0]

        return t1, t2

    @property
    def length_str(self):
        return f"{len(self.lemma_intersection)} tasks and {len(self.task_weights)} tasks with weights."

#! DEPRECATED
from typing import Dict, Optional, Tuple
from collections import defaultdict
from itertools import product
import random

import torch

from nmt_adapt.marginal_task import MarginalTask


class TaskSamplerDEPRECATED(object):
    """A class that helps sampling tasks for meta-learning.

    Args:
        parallel_dataset (_type_): _description_
        index (_type_): _description_
        filter_level (str, optional): Choice of {"script", "lemma", None}. Defaults to "script".

    """

    def __init__(self, parallel_dataset, index, filter_level: str = "script"):

        self.filter_level = filter_level

        self.task_lemmas, self.lemma_task_locs = self._get_lemma_task_locs(
            parallel_dataset, index, self.filter_level
        )

        self.lemma_intersection = self._get_task_lemma_intersections(self.task_lemmas)

        self._filter_out_empty_lemma_sets()
        self.n_tasks, self.n_lemmas = self._get_lens()

    def __repr__(self):
        return f"TaskSampler(n_tasks={self.n_tasks}, n_lemmas={self.n_lemmas})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _get_lemma_task_locs(parallel_dataset, index, filter_level):

        # Given a marginal task, what lemmas can I sample?
        task_lemmas = defaultdict(set)

        # Given a lemma, given a task, record the locations of
        # instance in the data where these might be found
        lemma_task_locs = defaultdict(lambda: defaultdict(list))

        for tag_set, ids_list in index.index.items():
            for (sent_id, token_id) in ids_list:

                lemma = parallel_dataset[sent_id]["lemmas"][token_id]
                lemma_script = parallel_dataset[sent_id]["lemma_scripts"][token_id]

                if filter_level == "script":
                    marg_task = MarginalTask(tag_set, lemma_script)
                elif filter_level == "lemma":
                    marg_task = MarginalTask(tag_set, lemma)
                elif filter_level in {None, "ignore", "None", "ign"}:
                    marg_task = MarginalTask(tag_set, None)

                lemma_task_locs[lemma][marg_task].append((sent_id, token_id))

                task_lemmas[marg_task].add(lemma)

        lemma_task_locs = dict(lemma_task_locs)
        for k, v in lemma_task_locs.items():
            lemma_task_locs[k] = dict(v)

        task_lemmas = dict(task_lemmas)

        return task_lemmas, lemma_task_locs

    @staticmethod
    def _get_task_lemma_intersections(task_lemmas):

        # Given two marginal tasks, what lemmas can I sample?
        task_task_lemmas = defaultdict(lambda: defaultdict(set))
        for t1, t2 in product(task_lemmas.keys(), task_lemmas.keys()):
            # Skip if matching on any part of marginal task
            if t1.match(t2) > 1:
                continue

            lemma_intersection = set.intersection(task_lemmas[t1], task_lemmas[t2])
            if len(lemma_intersection) >= 1:
                task_task_lemmas[t1][t2] |= lemma_intersection
                task_task_lemmas[t2][t1] |= lemma_intersection

        task_task_lemmas = dict(task_task_lemmas)
        for k, v in task_task_lemmas.items():
            task_task_lemmas[k] = dict(v)

        return task_task_lemmas

    @property
    def task_task_lemmas(self):
        """Alias for lemma intersection.
        """
        return self.lemma_intersection

    def _get_lens(self):

        unique_tasks = dict()
        for t1 in self.lemma_intersection.keys():
            for t2 in self.lemma_intersection[t1].keys():
                tasks_sorted = tuple(sorted([t1, t2], key=lambda x: str(x)))

                unique_tasks[tasks_sorted] = self.lemma_intersection[t1][t2]

        n_tasks = len(unique_tasks.keys())
        n_lemmas = len(set.union(*map(set, unique_tasks.values())))

        return n_tasks, n_lemmas

    def filter(self, lambda_filter):
        """Filters the joint task distribution based on conditions specified by the lambda function.
        Only keeps instances where the lambda function evaluates to `true`.

        Args:
            lambda_filter Callable: function that takes as arguments, in order,
                the tasks (t1, t2) and the lemma intersection of those tasks (lemma_intersection).
                Keeps data where the function returns True.
        """

        task_task_lemmas_filtered = dict()

        for t1, subdict in self.lemma_intersection.items():
            filtered_subdict = {
                t2: lemma_set
                for t2, lemma_set in subdict.items()
                if lambda_filter(t1, t2, lemma_set)
            }
            if len(filtered_subdict) > 0:
                task_task_lemmas_filtered[t1] = filtered_subdict

        self.lemma_intersection = task_task_lemmas_filtered
        self.n_tasks, self.n_lemmas = self._get_lens()

    def _filter_out_empty_lemma_sets(self):
        """Require the tasks' lemma intersection to have at least 2 lemmas
        """
        lambda_filter = lambda t1, t2, lemma_set: len(lemma_set) > 1

        self.filter(lambda_filter)

    def filter_out_trivial_edit_scripts(self):
        """Requires the tasks' lemma intersection to have at least 1 non-trivial ("L0|d|d") lemma edit script
        """
        lambda_filter = lambda t1, t2, lemma_set: (
            not (("d|d" in t1.lemma_edit_script) and ("d|d" in t2.lemma_edit_script))
        )

        self.filter(lambda_filter)

    def sample_batches_cross_transfer(
        self, t1, t2, n_lemmas: int = 4, n_samples_per_lemma: int = 2
    ):
        """
        Given two tasks, generates cross-transfer batches that:
            - have `n_lemmas` lemmas over the tasks
            - have `n_samples_per_lemma` examples per lemma
        The batch size is thus, at most, `n_lemmas` * `n_samples_per_lemma` and
        at least 2 (2 lemmas, each with only 1 example).

        This function does not check the examples, and can return identical
        data if present within the dataset.

        Returns two lists of (sent_id, token_id). If converted to text,
        you will notice the morphological tag set and lemma edit script (the task)
        are flipped for the same lemma across the lists.

        Args:
            t1 (_type_): _description_
            t2 (_type_): _description_
            n_lemmas (int, optional): _description_. Defaults to 4.
            n_samples_per_lemma (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """

        # Get the lemmas at the intersection of the tasks
        joint_lemma_set = self.lemma_intersection[t1][t2]

        # Limit the number of lemmas sampled per task to
        # the nearest multiple of 2 <= the set size
        max_allowed_lemmas_per_task = 2 * (len(self.lemma_intersection[t1][t2]) // 2)

        n_lemmas_per_task_ = min(n_lemmas, max_allowed_lemmas_per_task)

        # Initialize the splits
        support_batch, query_batch = [], []

        # Sample uniformly from the possible lemmas
        for i, lemma in enumerate(
            random.sample(list(joint_lemma_set), n_lemmas_per_task_)
        ):

            instances_t1 = self.lemma_task_locs[lemma][t1]
            instances_t2 = self.lemma_task_locs[lemma][t2]

            # Limit the number of examples per lemma to the minimum size
            # of instances possible for a task
            n_samples_per_lemma_ = min(
                n_samples_per_lemma, min(len(instances_t1), len(instances_t2))
            )

            sampled_t1 = random.sample(instances_t1, n_samples_per_lemma_)
            sampled_t2 = random.sample(instances_t2, n_samples_per_lemma_)

            # If in the first half, task 1 goes to support, task to query
            if i < (n_lemmas_per_task_ // 2):
                support_batch.extend(sampled_t1)
                query_batch.extend(sampled_t2)

            # If in the second half, vice versa
            else:
                query_batch.extend(sampled_t1)
                support_batch.extend(sampled_t2)

        return support_batch, query_batch

    def set_task_pair_weights(
        self,
        task_pair_weights: Optional[Dict[Tuple[MarginalTask], float]] = None,
        min_val: float = 0.0,
    ):
        """Set the task pair weights using a dict of unnormalized weights.
        If `None`, assumes a uniform distribution.

        Args:
            task_pair_weights (Optional[Dict[Tuple[MarginalTask], float]], optional): dict of task pairs as keys, unnormalized weight as value. Defaults to None.
            min_val (float, optional): minimum weight to be considered. Defaults to 0.0.
        """

        id_to_task_pair = []
        task_weights = []
        for t1 in list(self.lemma_intersection.keys()):

            for t2 in list(self.lemma_intersection[t1].keys()):
                if task_pair_weights is None:
                    task_weights.append(1)
                else:
                    task_weights.append(task_pair_weights.get((t1, t2), min_val))

                id_to_task_pair.append((t1, t2))

        self.task_pair_to_id = {k: i for i, k in enumerate(id_to_task_pair)}
        self.id_to_task_pair = id_to_task_pair

        self.task_weights = torch.tensor(task_weights).float()

    def _score_task_pair(self, task_pair: Tuple[MarginalTask]):
        return self.task_weights[self.task_pair_to_id[task_pair]]

    @property
    def unif_p_val(self):
        return 1 / len(self.task_weights)

    def sample_task_pair(self, uninformed: bool = False):

        # If uninformed, sample from the uniform distribution
        if uninformed:
            id = (
                torch.full((len(self.task_weights),), fill_value=self.unif_p_val)
                .multinomial(1)
                .item()
            )

        # Otherwise sample proportionally from the task weights
        else:
            id = torch.multinomial(torch.softmax(self.task_weights, dim=-1), 1).item()

        t1, t2 = self.id_to_task_pair[id]

        # Finally, shuffle the task pairs
        if random.random() < 0.5:
            return t1, t2
        else:
            return t2, t1
