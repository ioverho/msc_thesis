from nltk import ngrams
from nltk.metrics.distance import edit_distance
import numpy as np


def exact_match(ref_token, gen_token):

    max_len = max(len(gen_token), len(ref_token))

    try:

        match_score = sum(cg == cr for cg, cr in zip(gen_token, ref_token)) / max_len

    except ZeroDivisionError:
        # If both strings are empty, will raise `ZeroDivisionError`
        # Perfect match though
        match_score = 1.0

    return match_score


def chrf(ref_token, gen_token, beta: int = 1, n: int = 1):
    ref_char_set = set(ngrams(ref_token, n=n))
    gen_char_set = set(ngrams(gen_token, n=n))

    intersection_length = len(set.intersection(ref_char_set, gen_char_set))

    try:
        precision = intersection_length / len(ref_char_set)
        recall = intersection_length / len(gen_char_set)

        chrf = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

    except ZeroDivisionError:
        chrf = 0.0

    return chrf


def lev_dist(ref_token, gen_token, norm: bool = False, **edit_distance_kwargs):
    dist = edit_distance(gen_token, ref_token, **edit_distance_kwargs)

    if norm:
        try:
            return dist / (len(ref_token))
        except ZeroDivisionError:
            return dist
    else:
        return dist


def exact_spe_match(ref_spe, gen_spe, tokenizer):

    with tokenizer.as_target_tokenizer():
        spe = tokenizer.encode(gen_spe, return_tensors="pt")[0, :-1]

    return exact_match(ref_spe, spe.detach().cpu().numpy())


def spe_lev_dist(ref_spe, gen_spe, tokenizer, **lev_dist_kwargs):

    with tokenizer.as_target_tokenizer():
        spe = tokenizer.encode(gen_spe, return_tensors="pt")[0, :-1]

    return lev_dist(ref_spe, spe.detach().cpu().numpy(), **lev_dist_kwargs)


def weighted_metric(ref_token, gen_tokens, metric, **metric_kwargs):

    metric_sum = 0.0
    count_sum = 0
    for tok, count in gen_tokens:
        metric_val = metric(ref_token, tok, **metric_kwargs)

        metric_sum += count * metric_val
        count_sum += count

    return metric_sum / count_sum


def entropy(sample_counts):

    counts = [count for _, count in sample_counts]

    probs = counts / np.sum(counts)

    entropy = -np.sum(probs * np.log2(probs))

    return entropy


def token_metrics(ref_token, terminated_tokens, tokenizer):

    result = {}

    result |= {"char_match": weighted_metric(ref_token, terminated_tokens, exact_match)}
    result |= {
        "lev_dist": weighted_metric(ref_token, terminated_tokens, lev_dist),
        "norm_lev_dist": weighted_metric(
            ref_token, terminated_tokens, lev_dist, norm=True
        ),
    }
    result |= {
        f"chrf_{n}": weighted_metric(ref_token, terminated_tokens, chrf, n=n, beta=3)
        for n in range(1, 5 + 1)
    }

    with tokenizer.as_target_tokenizer():
        tgt = tokenizer(ref_token)["input_ids"][:-1]

    result |= {
        "spe_match": weighted_metric(
            tgt, terminated_tokens, exact_spe_match, tokenizer=tokenizer
        )
    }
    result |= {
        "spe_lev_dist": weighted_metric(
            tgt, terminated_tokens, spe_lev_dist, tokenizer=tokenizer, norm=False
        ),
        "spe_norm_lev_dist": weighted_metric(
            tgt, terminated_tokens, spe_lev_dist, tokenizer=tokenizer, norm=True
        ),
    }

    return result


def morphological_metrics(annotated_sent, sample_counts, t: int, tagger):
    def weight_agg(values, weights):
        try:
            return sum(w * v for v, w in zip(values, weights)) / sum(w for w in weights)
        except ZeroDivisionError:
            return 0

    sents_to_tag = [
        annotated_sent["tgt_tokens"][:t]
        # + [sample if len(sample) >= 1 else "\u2800"]
        + [sample] + annotated_sent["tgt_tokens"][t + 1 :]
        for sample, _ in sample_counts
        if len(sample) >= 1
    ]
    tag_sents_counts = [count for sample, count in sample_counts if len(sample) >= 1]

    assert len(sents_to_tag) == len(
        tag_sents_counts
    ), f"{len(sents_to_tag)} != {len(tag_sents_counts)}"

    lemmas, _, morph_tags, morph_cats = tagger.forward(
        sents_to_tag, is_pre_tokenized=True
    )

    lemma_match = [int(lem_seq[t] == annotated_sent["lemmas"][t]) for lem_seq in lemmas]
    lemma_norm_lev_dist = [
        lev_dist(annotated_sent["lemmas"][t], lem_seq[t], norm=True)
        for lem_seq in lemmas
    ]
    morph_tag_match = [
        len(set.intersection(seq[t], annotated_sent["morph_tags"][t]))
        == len(annotated_sent["morph_tags"][t])
        for seq in morph_tags
    ]
    morph_tag_iou = [
        (
            (
                len(set.intersection(seq[t], annotated_sent["morph_tags"][t]))
                / len(set.union(seq[t], annotated_sent["morph_tags"][t]))
            )
            if len(set.union(seq[t], annotated_sent["morph_tags"][t])) > 0
            else 0
        )
        for seq in morph_tags
    ]
    morph_cat_iou = [
        (
            (
                len(set.intersection(seq[t], annotated_sent["morph_cats"][t]))
                / len(set.union(seq[t], annotated_sent["morph_cats"][t]))
            )
            if len(set.union(seq[t], annotated_sent["morph_cats"][t])) > 0
            else 0
        )
        for seq in morph_cats
    ]

    confusion = {frozenset(seq[t]): 0 for seq in morph_tags}
    for morph_tag_seq, v in zip(morph_tags, tag_sents_counts):
        confusion[frozenset(morph_tag_seq[t])] += v

    result = {
        "lemma_match": weight_agg(lemma_match, tag_sents_counts),
        "lemma_norm_lev_dist": weight_agg(lemma_norm_lev_dist, tag_sents_counts),
        "morph_tag_match": weight_agg(morph_tag_match, tag_sents_counts),
        "morph_tag_iou": weight_agg(morph_tag_iou, tag_sents_counts),
        "morph_cat_iou": weight_agg(morph_cat_iou, tag_sents_counts),
        "confusion": confusion,
    }

    try:
        result |= {
            "morph_tag_iou_at_match": sum(
                count * tag_iou
                for lem_match, tag_iou, count in zip(
                    lemma_match, morph_tag_iou, tag_sents_counts
                )
                if lem_match
            )
            / sum(
                count
                for lem_match, count in zip(lemma_match, tag_sents_counts)
                if lem_match
            )
        }
    except ZeroDivisionError:
        result |= {"morph_tag_iou_at_match": 0.0}

    return result
