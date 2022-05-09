from collections import defaultdict
import math
from typing import Optional, List, Tuple
import warnings

from transformers.tokenization_utils import _is_start_of_word
import torch

HELSINKI_NMT_SPACE_CHAR = "\u2581"


@torch.no_grad()
def get_logits(
    model, batch_size: int, encoder_input_ids, encoder_attention_mask, decoder_input_ids
):
    """Function that processes a seq2seq batch.
    Will return logits corresponding to p(y_{t+1}|y_{t}, x).

    Args:
        model (_type_): _description_
        batch_size (int): _description_
        encoder_input_ids (_type_): _description_
        encoder_attention_mask (_type_): _description_
        decoder_input_ids (_type_): _description_

    Returns:
        _type_: _description_
    """

    b = decoder_input_ids.size(0)

    # In case the input exceeds the maximum batch-size
    # Run multiple forward passes and return concatenation of the results
    batch_sizes = [
        batch_size if i * batch_size <= b else b % batch_size
        for i in range(1, math.ceil(b / batch_size) + 1)
    ]

    tgt_logits = []
    for i, bs in enumerate(batch_sizes):
        tgt_logits.append(
            model.forward(
                input_ids=encoder_input_ids.expand(bs, -1).to(model.device),
                attention_mask=encoder_attention_mask.expand(bs, -1).to(model.device),
                decoder_input_ids=decoder_input_ids[
                    i * batch_size : (i + 1) * batch_size
                ].to(model.device),
            )
            .logits[:, -1, :]
            .to(torch.device("cpu"))
        )

    tgt_logits = torch.cat(tgt_logits, dim=0)

    return tgt_logits


def multinomial_sampler(
    probs: torch.Tensor, n_samples: Optional[int] = None, override: bool = False
):

    if (probs.size(0) == 1 or len(probs.size()) == 0) and (n_samples is None):
        raise ValueError("If the batchsize is 1, n_samples must be given.")

    elif (probs.size(0) == 1 or len(probs.size()) != 0) and not (n_samples is None):
        probs = probs.squeeze()

    elif (
        (probs.size(0) != 1 and len(probs.size()) != 0)
        and (not n_samples is None)
        and (not override)
    ):
        warnings.warn(
            f"Batch size is non-unit, {probs.size(0)}, and n_samples is {n_samples}. Will only sample once per sample in batch unles `override` is `True`."
        )
        n_samples = 1

    else:
        n_samples = 1

    return torch.multinomial(probs, num_samples=n_samples, replacement=True)


def nucleus_mask(
    probs: torch.Tensor,
    tau: float,
    ratings_tensor: Optional[torch.Tensor] = None,
    descending: bool = True,
):
    """Generate the mask for top-p/nucleus sampling, based on ranks of some other (or same) Tensor.

    Args:
        probs (torch.Tensor): _description_
        tau (float): _description_
        ratings_tensor (Optional[torch.Tensor], optional): _description_. Defaults to None.
        descending (bool, optional): _description_. Defaults to True.

    Returns:
        torch.Tensor: the mask.
    """
    # Sort and argsort
    if ratings_tensor is None:
        probs_sorted, idx_to_rank = torch.sort(probs, dim=-1, descending=descending)
    else:
        idx_to_rank = torch.argsort(ratings_tensor, dim=-1, descending=descending)
        probs_sorted = probs.scatter(dim=-1, index=idx_to_rank, src=probs)

    # Compute the cumulative sum of probability mass in descending order
    cum_mass = torch.cumsum(probs_sorted, dim=-1)

    # Find the minimum value such that the cumulative sum is above `tau`
    min_vals, _ = torch.min(
        torch.where(
            cum_mass - tau > 0, cum_mass, torch.tensor([torch.inf], dtype=torch.float)
        ),
        dim=-1,
    )

    # Generate a mask using the per row min value
    mask = cum_mass <= min_vals[:, None]

    # Unsort the mask back to the original indices (use argsorted indices)
    mask = mask.scatter(dim=-1, index=idx_to_rank, src=mask)

    return mask


def generate_tuncated_categorical(
    logits: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    sampling_method: str = "greedy",
    **sampling_kwargs,
):
    """A function to convert a categorical/multinomial distribution, $$p(y_{t}|.)$$, into some form of truncated variant.
    Truncation can occur in the following manners:
        - {'greedy', 'vanilla', 'basic', 'ancestral'}:
            Simply uses the model's softmax normalized logits. Performs very poorly when model is trained with label smoothing.

        - {'top-k'}:
            Truncates distribution to only the $$k$$ highest probability values.

        - {'top-p', 'nucleus'}:
            Truncates distribution to probability values that satisfy $$\\min_{y_{t}\\in\\mathcal{Y}}\\sum_{y_{t}}p(y_{t}|.)\\geq \\tau$$. Much like $$k$$, $$tau$$ is now a hyper-parameter controlling how much of the sampling space is retained.

        - {'typical'}:
            Truncates distribution to probability values that are most typical, while still satisfying the nucleus-sampling sum constraint. Typicality is defined as the L1-distance to the conditional entropy of the distribution.
            See:
                Meister, C., Pimentel, T., Wiher, G., & Cotterell, R. (2022). Typical Decoding for Natural Language Generation. arXiv preprint arXiv:2202.00666.

    Args:
        logits (Optional[torch.Tensor], optional):  model outputs, unnormalized. Must be given if probs is not. Defaults to None.
        probs (Optional[torch.Tensor], optional): softmax normalized model outputs. Must be given if logits is not. Defaults to None.
        sampling_method (str, optional): sampling method name. See description. Defaults to "greedy".
        sampling_kwargs: the free parameter for each sampling method must be given. `k` for top-k, `tau` for nucleus or typical sampling.

    Returns:
        torch.Tensor: truncated (masked) and renormalized probabilities
    """

    if logits is None and probs is None:
        raise ValueError("Either logits or probs must be provided.")

    if logits is not None:
        probs = torch.softmax(logits, dim=-1)

    if sampling_method.lower() in {"ancestral", "basic", "greedy", "vanilla"}:
        # Simply samples with the generated (softmax normalized) logits
        mask = torch.ones_like(probs)

    elif sampling_method.lower() in {"top-k"}:
        # Truncates the distribution and selects the top `k`-probabilities
        k = sampling_kwargs.get("k", None)

        if k is None:
            raise ValueError("If sampling with top-k method, must provide `k`.")

        # Sort the probabilities
        probs_sorted, _ = torch.sort(probs, dim=-1, descending=True)

        # Generate the top-p mask
        mask = probs > probs_sorted[:, k][:, None]

    elif sampling_method.lower() in {"top-p", "nucleus"}:
        # Truncates the distribution by selecting the top probabilities such that
        # their sum is equal to `tau`
        # Actually selects those probabilities such that their sum is as close to,
        # but greater than `tau`
        tau = sampling_kwargs.get("tau", None)

        if tau is None:
            raise ValueError(
                "If sampling with top-p/nucleus method, must provide desired sum `tau`."
            )

        # Get mask using special function
        mask = nucleus_mask(probs=probs, tau=tau, descending=True)

    elif sampling_method.lower() in {"typical"}:
        # Truncates the distribution by selecting the most typical probabilities
        # subject to the constaint that their sum is equal to `tau`
        # Again, actually selects those probabilities such that their sum is as close to,
        # but greater than `tau`
        tau = sampling_kwargs.get("tau", None)

        if tau is None:
            raise ValueError(
                "If sampling with top-p/nucleus method, must provide desired sum `tau`."
            )

        # Compute typicality
        # Defined as the L1-distance of the NLP to the distribution's conditional Entropy
        # Uses base 2, bits, by default (discrete distribution)
        cond_entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
        typicality = torch.abs(cond_entropy[:, None] + torch.log2(probs))

        # Get mask using same special function as top-p, but sort by typicality instead
        mask = nucleus_mask(
            probs=probs, tau=0.2, ratings_tensor=typicality, descending=False
        )

    probs_ = (mask * probs) / torch.sum(mask * probs)

    return probs_


def generate_samples(
    src_text: str,
    ref_context: str,
    ref_label: str,
    tokenizer,
    model,
    n_samples: int,
    sampling_method: str,
    max_batch_size: int = 64,
    max_over_T: int = 10,
    **sampling_method_kwargs,
) -> List[Tuple[str, int]]:
    """
    Generates, as efficiently as it can, samples using a model to estimate
        p(y_{t}|y_{<t}, x)
    and a corresponding tokenizer.

    Only feeds unique contexts (y_{<t}) through model until it produces a terminating token.
    Terminating tokens include any punctuation, spaces (tokenized), or <EOS>, amongst others.

    Args:
        src_text (str): the source side text as a string
        ref_context (str): the target side context as a string
        ref_label (str): the target word to be produced, as a string
        tokenizer (_type_): the HF tokenizer
        model (_type_): the HF model
        max_batch_size (int): the maximum allowed batchsize. If a larger batch_sizemore is fed
            to the model, it runs multiple batches in a row and concatenates the result
        n_samples (int): the desired number of samples (non-unique)
        sampling_method (str): the sampling method. See `generate_tuncated_categorical`
        max_over_T (int): the maximum allowed number of additional BPEs before cutting the model off
        **sampling_method_kwargs (Dict): any keyword arguments that need to be fed through
            to `generate_tuncated_categorical`

    Returns:
        (List[Tuple[str, int]]): a list of tuples, representing the text generated and the number of occurences
    """

    def merge_generations_with_context(context, generations):
        """Merges a tokenized tensor with a new set of samples
        """
        if generations.size(0) == 1 or len(generations.size()) == 1:
            # Feign seq. length of 1
            generations = generations.unsqueeze(-1)

        context_ = torch.cat(
            [
                # Expand the original context to number of unique generations
                context.expand(generations.size(0), -1),
                generations,
            ],
            dim=-1,
        )

        return context_

    # Convert texts to model input
    src = tokenizer(src_text, padding=True, return_tensors="pt")

    generated_texts_stack = [
        " ".join([tokenizer._pad_token] + ref_context)
        if len(ref_context) == 0
        else " ".join(ref_context)
    ]
    with tokenizer.as_target_tokenizer():
        tgt_context = tokenizer(
            generated_texts_stack,
            padding=True,
            return_tensors="pt",
            is_split_into_words=False,
        )
        tgt = tokenizer(
            ref_label, padding=True, return_tensors="pt", is_split_into_words=False
        )

    tgt_context = tgt_context["input_ids"][:, :-1]

    tgt = tgt["input_ids"][:, :-1]

    # Generate logits corresponding to p(y_{t}|y_{t-1}, x)
    tgt_logits = get_logits(
        model, max_batch_size, src["input_ids"], src["attention_mask"], tgt_context
    )

    # Get first samples
    probs = generate_tuncated_categorical(
        logits=tgt_logits, sampling_method=sampling_method, **sampling_method_kwargs
    )

    samples = multinomial_sampler(probs=probs, n_samples=n_samples, override=True)

    unique_samples, sample_counts = torch.unique(
        samples, sorted=False, return_counts=True,
    )

    # Expand the initial context with the new generations
    contextualized_samples = merge_generations_with_context(tgt_context, unique_samples)

    i = 1
    terminated_tokens = defaultdict(int)
    while len(contextualized_samples) != 0 and i < max_over_T + tgt.size(1):

        tgt_logits = get_logits(
            model,
            max_batch_size,
            src["input_ids"],
            src["attention_mask"],
            contextualized_samples,
        )

        # Expand each row according to the number of occurence in the previous sample batch
        tgt_logits = torch.repeat_interleave(tgt_logits, sample_counts, dim=0)
        contextualized_samples = torch.repeat_interleave(
            contextualized_samples, sample_counts, dim=0
        )

        # Convert to probs and generate samples
        probs = generate_tuncated_categorical(
            logits=tgt_logits,
            sampling_method=sampling_method,
            **sampling_method_kwargs,
        )

        samples = multinomial_sampler(probs=probs, n_samples=1, override=False)

        # Merge back context
        contextualized_samples = merge_generations_with_context(
            contextualized_samples, samples
        )

        unique_samples, sample_counts = torch.unique(
            contextualized_samples, sorted=False, return_counts=True, dim=0
        )

        terminated_mask = torch.tensor(
            [
                (
                    gen[0] == HELSINKI_NMT_SPACE_CHAR
                    or gen == tokenizer.eos_token
                    or _is_start_of_word(gen)
                )
                for gen in tokenizer.convert_ids_to_tokens(unique_samples[:, -1])
            ]
        )

        # Add terminated tokens to the stack
        terminated_toks = unique_samples[terminated_mask][:, -(i + 1) : -1]
        terminated_toks = [
            (tokenizer.decode(gen), count.item())
            for gen, count in zip(terminated_toks, sample_counts[terminated_mask])
        ]
        if len(terminated_toks) != 0:
            for tok, count in terminated_toks:
                terminated_tokens[tok] += count

        # Then remove the terminated tokens from contention
        contextualized_samples = unique_samples[~terminated_mask]
        sample_counts = sample_counts[~terminated_mask]

        i += 1

    terminated_tokens = list(
        sorted(terminated_tokens.items(), key=lambda x: x[1], reverse=True)
    )

    return terminated_tokens
