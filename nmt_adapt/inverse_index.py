import typing
import random
import pickle
from collections import Counter, defaultdict
from itertools import filterfalse

from nmt_adapt.marginal_task import MarginalTask
from utils.errors import ConfigurationError


class InverseIndexv2(object):
    """_summary_

    Args:
        par_data (_type_): _description_
        index_level (str, optional): _description_. Defaults to "tag_set".
        filter_level (str, optional): _description_. Defaults to "lemma_script".
    """

    def __init__(
        self,
        par_data: typing.Optional[object] = None,
        index: typing.Optional[typing.Dict[str, typing.Dict[str, list]]] = None,
        index_level: str = "tag_set",
        filter_level: str = "lemma_script",
    ):

        self.index_level = index_level
        self.filter_level = filter_level

        if index is None and par_data is not None:
            # ======================================================================
            # Generate the index
            # ======================================================================
            index = defaultdict(lambda: defaultdict(list))

            for i, doc in enumerate(par_data):
                local_index = defaultdict(lambda: defaultdict(list))

                # Set the filter level
                # Relevant for later
                if self.filter_level == "lemma":
                    filter_val = doc["lemmas"]
                elif (
                    self.filter_level == "lemma_script" or self.filter_level == "script"
                ):
                    filter_val = doc["lemma_scripts"]
                elif self.filter_level == "tag_set" or self.filter_level is None:
                    filter_val = [None for _ in range(len(doc["morph_tags"]))]

                # Get the index keys
                for tok_id, key_parts in enumerate(
                    zip(map(frozenset, doc["morph_tags"]), doc["lemma_scripts"])
                ):

                    if self.index_level == "tag":
                        for m_tag in key_parts[0]:
                            key = MarginalTask(m_tag, key_parts[1])

                    elif self.index_level == "tag_set":
                        key = MarginalTask(key_parts[0], key_parts[1])

                    local_index[key][filter_val[tok_id]].append((i, tok_id))

                # Update the inverted index
                # Only keeps 1 index_level, filter_level combination
                # Now samples to avoid always selecting earliest
                for index_key, filter_items in local_index.items():
                    for filter_key, filter_vals in filter_items.items():
                        if filter_level is not None and len(filter_vals) > 1:
                            sampled_index_vals = random.sample(filter_vals, k=1)

                        else:
                            sampled_index_vals = filter_vals

                    for (i, tok_id) in sampled_index_vals:
                        index[index_key][doc["lemmas"][tok_id]] += [(i, tok_id)]

            for k in index.keys():
                index[k] = dict(index[k])
            self.index = dict(index)

        elif index is not None and par_data is None:
            self.index = index

        else:
            raise ConfigurationError(
                "Must provide either a pre-built index or a dataset to build one from."
            )

    def filter(self, fn):
        for k_del in list(filterfalse(fn, self.keys())):
            del self.index[k_del]

    def keys(self):
        return self.index.keys()

    def __len__(self):
        return sum(len(v) for _, v in self.index.items())

    def __getitem__(self, i):
        return self.index[i]

    def reduce(
        self,
        max_samples: typing.Optional[int] = None,
        min_samples: int = 0,
        min_lemmas: int = 0,
        min_samples_per_lemma: int = 0,
    ):

        # ======================================================================
        # Reduce on samples
        # ======================================================================
        remove_keys = []
        reduce_keys = []

        for k, v in self.index.items():
            n_samples = sum(len(vv) for vv in v.values())
            if n_samples <= min_samples:
                remove_keys.append(k)

            elif max_samples is not None and n_samples >= max_samples:
                reduce_keys.append(k)

        for k in remove_keys:
            del self.index[k]

        # Less than or fewer? less has 4 chars
        print(
            f"Removed {len(remove_keys)} keys containing less than {min_samples} values."
        )

        if max_samples is not None:
            for k in reduce_keys:
                lemma_counts = Counter(
                    random.sample(
                        list(self.index[k].keys()),
                        k=max_samples,
                        counts=[len(v) for _, v in self.index[k].items()],
                    )
                )
                self.index[k] = {
                    kk: random.sample(self.index[k][kk], k=count)
                    for kk, count in lemma_counts.items()
                }

            print(
                f"Reduced {len(reduce_keys)} keys containing more than {max_samples} values."
            )

        # ======================================================================
        # Reduce on lemmas
        # ======================================================================
        remove_secondary_keys = []
        for k, v in self.index.items():
            for kk, vv in v.items():
                if len(vv) <= min_samples_per_lemma:
                    remove_secondary_keys.append((k, kk))

        for k, kk in remove_secondary_keys:
            del self.index[k][kk]

        print(
            f"Removed {len(remove_secondary_keys)} secondary keys containing less than {min_samples_per_lemma} examples."
        )

        remove_keys = []
        for k, v in self.index.items():
            if len(v) <= min_lemmas:
                remove_keys.append(k)

        for k in remove_keys:
            del self.index[k]

        print(
            f"Removed {len(remove_keys)} keys containing less than {min_lemmas} secondary keys."
        )

    @property
    def length_str(self):
        return f"Index has {len(self.index.keys())} keys, {sum(len(v) for v in self.index.values())} secondary keys and {sum(len(vv) for v in self.index.values() for vv in v.values())} values."

    @property
    def coverage(self):
        return len({vvv[0] for _, v in self.index.items() for _, vv in v.items() for vvv in vv})

    def save(self, fp):

        state_dict = {
            "index": self.index,
            "filter_level": self.filter_level,
            "index_level": self.index_level,
        }

        with open(fp, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, fp):

        with open(fp, "rb") as f:
            state_dict = pickle.load(f)

        return InverseIndexv2(**state_dict)

    def __iter__(self, shuffle: bool = True):

        # Now forces a stratified sampling scheme
        # Will sample in cycles from each task, until tasks are exhausted
        # e.g. tasks might be a, b, c, a, b, a, b, a, a, ...
        stratified_sequence = defaultdict(list)
        for k, v in self.index.items():
            i = 0
            for kk in random.sample(list(v.keys()), len(v)):
                for vv in random.sample(v[kk], len(v[kk])) if shuffle else v[kk]:
                    stratified_sequence[i].append((k, vv))
                    i += 1

        stratified_sequence = [
            (k, v)
            for _, l in sorted(stratified_sequence.items(), key=lambda x: x[0])
            for (k, v) in l
        ]

        for item in stratified_sequence:
            yield item

class InverseIndex(object):
    """A class that generates an inverted index and provides filtering, reducing and sampling, utilities.

    Args:
        parallel_dataset (_type_, optional): the parallel dataset for which the index is valid
        index_level (str, optional): which index keys to use. Choice of {'tag_set', 'tag'}. Defaults to 'tag_set'.
        filter_level (str, optional): only the first occurence of the index_level, filter_level will be taken into the index. Choice of {'lemma', 'script', 'tag_set', None}. Defaults to 'lemma'.
        index (_type, optional): pre-built index
    """

    def __init__(
        self,
        parallel_dataset=None,
        index_level: str = "tag_set",
        filter_level: str = "script",
        index=None,
    ):

        self.filter_level = filter_level
        self.index_level = index_level

        if index is None and parallel_dataset is not None:
            self.index = self._generate_inverted_index(parallel_dataset)

        elif index is not None:
            self.index = index

        else:
            raise ConfigurationError(
                "Must specify either a parallel dataset or a pre-built index"
            )

    def _generate_inverted_index(self, parallel_dataset):

        inverted_index = defaultdict(list)

        for i, doc in enumerate(parallel_dataset):
            seen_filter_vals = set()

            # Set the second filter level
            # Will only keep the first occurence of the index_level, filter_level combination
            if self.filter_level == "lemma":
                second_filter_val = doc["lemmas"]
            elif self.filter_level == "lemma_script" or self.filter_level == "script":
                second_filter_val = doc["lemma_scripts"]
            elif self.filter_level == "tag_set" or self.filter_level is None:
                second_filter_val = [None for _ in doc["tokens"]]

            gen = zip(map(frozenset, doc["morph_tags"]), second_filter_val)
            for tok_id, filter_val in enumerate(gen):
                if filter_val in seen_filter_vals and self.filter_level is not None:
                    continue

                if self.index_level == "tag":
                    for m_tag in filter_val[0]:
                        inverted_index[m_tag].append((i, tok_id))

                elif "set" in self.index_level:
                    inverted_index[filter_val[0]].append((i, tok_id))

                seen_filter_vals.add(filter_val)

        inverted_index = dict(inverted_index)

        return inverted_index

    def filter(self, filter_vals: set):
        """Retains only keys where some part of the keys are in the `filter_vals` set.

        """

        # Generate an inverted index to sample sentence ids over
        filtered_keys = {
            k for k in self.index.keys() if len(set.intersection(filter_vals, k)) >= 1
        }

        self.index = {k: self.index[k] for k in filtered_keys}

    def reduce(
        self,
        parallel_dataset,
        max_samples: int,
        min_samples: int,
        stratified: bool = True,
    ):
        """Uses stratified sampling (to the filter_level) to reduce the index to a maximum of `max_samples` per key.
        The samples remaining should be equally spread over the filter_level.

        Args:
            parallel_dataset (_type_): _description_
            max_samples (_type_): _description_
            min_samples (int): the minimum number of samples a key needs to be retained
        """

        def reduce_tag_stratified(index, tag: frozenset, max_samples: int):
            """
            Applies stratified sampling over the lemma scripts to reduce the number of total samples to max_samples only.
            Thus, each lemma_script is roughly equally likely

            Args:
                index (_type_): _description_
                tag (frozenset): _description_
                max_samples (int): _description_

            Returns:
                _type_: _description_
            """

            tags_by_scripts = defaultdict(list)
            for sent_id, tok_id in index[tag]:
                tags_by_scripts[
                    parallel_dataset[sent_id]["lemma_scripts"][tok_id]
                ].append((sent_id, tok_id))
            tags_by_scripts = dict(tags_by_scripts)
            tags_by_scripts = {
                k: random.sample(v, len(v)) for k, v in tags_by_scripts.items()
            }

            collected_tag_docs = []
            if len(tags_by_scripts.keys()) <= max_samples:
                while len(collected_tag_docs) <= max_samples:
                    for _, ids_l in tags_by_scripts.items():
                        if len(ids_l) >= 1:
                            collected_tag_docs.append(ids_l.pop(0))
            else:
                for k in random.sample(tags_by_scripts.keys(), max_samples):
                    collected_tag_docs.append(tags_by_scripts[k][0])

            return random.sample(collected_tag_docs, k=max_samples)

        tags_to_remove = []
        for tag, vals in self.index.items():

            tag_len = len(vals)

            if min_samples > 0 and tag_len < min_samples:
                tags_to_remove.append(tag)

            if tag_len > max_samples:
                if stratified:
                    collected_tag_docs = reduce_tag_stratified(
                        self.index, tag, max_samples
                    )
                    self.index[tag] = collected_tag_docs
                else:
                    self.index[tag] = random.sample(self.index[tag], k=max_samples)

        for tag in tags_to_remove:
            del self.index[tag]

    def keys(self):
        return self.index.keys()

    def __len__(self):
        return sum(len(v) for _, v in self.index.items())

    def __iter__(self, shuffle: bool = True):

        # Now forces a stratified sampling scheme
        # Will sample in cycles from each task, until tasks are exhausted
        # e.g. tasks might be a, b, c, a, b, a, b, a, a, ...
        stratified_sequence = defaultdict(list)
        for k, v in self.index.items():
            for i, vv in enumerate(random.sample(v, len(v)) if shuffle else v):
                stratified_sequence[i].append((k, vv))

        stratified_sequence = [
            (k, v)
            for _, l in sorted(stratified_sequence.items(), key=lambda x: x[0])
            for (k, v) in l
        ]

        for item in stratified_sequence:
            yield item

    def save(self, fp):

        state_dict = {
            "filter_level": self.filter_level,
            "index_level": self.index_level,
            "index": self.index,
        }

        with open(fp, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, fp):

        with open(fp, "rb") as f:
            state_dict = pickle.load(f)

        return InverseIndex(**state_dict)
