import os
from datetime import datetime
import yaml
import jsonlines as jsonl
import argparse

import torch
from sacremoses import MosesDetokenizer
import pycountry
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, Trainer
from datasets import Dataset
import mbr_nmt
import mbr_nmt.translate as mbr_translate
import mbr_nmt.convert as mbr_convert
import hydra
from omegaconf import DictConfig, OmegaConf

from nmt_adapt.data.corpus_functional import load_custom_dataset
from utils.experiment import set_seed, set_deterministic, Timer
from utils.errors import ConfigurationError

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


TRANSLATIONS_DIR = "./nmt_adapt/translations"

def load_nmt_checkpoint(checkpoint_dir = None, model_name = None):

    if checkpoint_dir is not None:
        with open(f"{checkpoint_dir}/config.yaml", "r") as f:
            train_config = yaml.safe_load(f)

        model_name = train_config["trainer"]["model_name"]

        model_config = AutoConfig.from_pretrained(model_name)
        if "nmt_kwargs" in train_config["trainer"].keys():
            model_config.dropout = train_config["trainer"]["nmt_kwargs"].get("dropout", model_config.dropout)

        name = "_".join(checkpoint_dir.split("/")[-2:])

    elif model_name is not None:
        model_config = AutoConfig.from_pretrained(model_name)

        name = "_".join(model_name.split("/")[-2:])

    else:
        raise ConfigurationError("Must specify either a checkpoint_dir or a model_name.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, enable_sampling=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=model_config)

    if checkpoint_dir is not None:
        model.load_state_dict(torch.load(f"{checkpoint_dir}/checkpoints/best.ckpt"))

    return tokenizer, model,name

def generate_translations(config):

    print("=" * 100 + f"\nGENERATING TRANSLATIONS" + "\n" + "=" * 100)

    timer = Timer()

    # *==========================================================================
    # *Experiment
    # *==========================================================================
    print(f"\n{timer.time()} | EXPERIMENTAL SETUP" + "\n" + "+" * 50)

    src_lang = pycountry.languages.get(name=config["src_lang"])
    tgt_lang = pycountry.languages.get(name=config["tgt_lang"])

    print(f"\n{timer.time()} | SETUP" + "\n" + "+" * 50)

    # == Device
    use_cuda = config["gpu"] or config["gpu"] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Training on {device}" + f"- {torch.cuda.get_device_name(0)}"
        if use_cuda
        else ""
    )

    # == Reproducibility
    set_seed(config["seed"])
    if config["deterministic"]:
        set_deterministic()

    # *==========================================================================
    # *Dataset
    # *==========================================================================
    print(f"\n{timer.time()} | LOADING DATASET" + "\n" + "+" * 50)

    test_dataset = load_custom_dataset(
        src_lang=src_lang.name,
        tgt_lang=tgt_lang.name,
        dataset_name=config["dataset_name_lower"],
        split=config["split"]
        )

    if config.get("cut_off", None) is not None:
        test_dataset = test_dataset.select(list(range(config["cut_off"] - 1)))

    print(f"Test dataset: {len(test_dataset)}")

    # *==========================================================================
    # *Model
    # *==========================================================================
    print(f"\n{timer.time()} | LOADING MODELS" + "\n" + "+" * 50)
    detokenizer = MosesDetokenizer(lang=tgt_lang.alpha_2)

    tokenizer, model, name = load_nmt_checkpoint(
        config.get("checkpoint_dir", None),
        config.get("model_name", None),
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)

    dataset_dir = name
    dataset_dir += f"_{config.get('affix', '')}" if config.get('affix', None) is not None else ""
    dataset_dir += f"_{config['dataset_name_lower']}"
    dataset_dir += f"_{config['split']}" if config.get("split", None) is not None else ""
    dataset_dir += f"_{config['src_lang'].lower()}_{config['tgt_lang'].lower()}"

    os.makedirs(f"{TRANSLATIONS_DIR}/{dataset_dir}", exist_ok=True)
    print(f"Output can be found in: {dataset_dir}")

    # *==========================================================================
    # *Loop
    # *==========================================================================
    print(f"\n{timer.time()} | GENERATING TRANSLATIONS" + "\n" + "+" * 50)
    #for i in range(test_dataset.num_rows):
    try:
        for i in range(len(test_dataset)):
            # Load the data and prepare reference translation ==========================
            sample = test_dataset[i]

            src_text = sample["src_text"]

            detokenized_tgt_text = detokenizer.detokenize(sample["tgt_tokens"])

            # Find beam search solution ================================================
            src_input = tokenizer(src_text, return_tensors="pt", padding=True, truncate=True)
            with tokenizer.as_target_tokenizer():
                tgt_input = tokenizer(sample["tgt_tokens"], is_split_into_words=True, padding=True, truncate=True)

            outputs = model.generate(
                src_input.input_ids.to(device),
                do_sample=False,
                early_stopping=True,
                num_beams=config["num_beams"],
                num_return_sequences=1,
                max_length=min(len(tgt_input.input_ids)-1+config["max_extra_tokens"], 512),
                **config["generate_kwargs"],
                )
            beam_search_translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Generate samples ========================================================
            outputs = model.generate(
                src_input.input_ids.to(device),
                do_sample=True,
                early_stopping=True,
                num_beams=1,
                num_return_sequences=config["num_beams"],
                max_length=min(len(tgt_input.input_ids)-1+config["max_extra_tokens"], 512),
                **config["generate_kwargs"],
                )
            sampled_translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Write to file ========================================================
            with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/source.txt", "a", encoding="utf-8") as f:
                f.write(f"{sample['src_text']}\n")

            with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/references.txt", "a", encoding="utf-8") as f:
                f.write(f"{detokenized_tgt_text}\n")

            with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/beam_search.txt", "a", encoding="utf-8") as f:
                f.write(f"{beam_search_translation[0]}\n")

            with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/mbr_samples.txt", "a", encoding="utf-8") as f:
                for s in sampled_translation:
                    f.write(f"{s}\n")

    except:
        # Write to file ========================================================
        with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/source.txt", "a", encoding="utf-8") as f:
            f.write(f"{sample['src_text']}\n")

        with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/references.txt", "a", encoding="utf-8") as f:
            f.write(f"{detokenized_tgt_text}\n")

        with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/beam_search.txt", "a", encoding="utf-8") as f:
            f.write(f"{beam_search_translation[0]}\n")

        with open(f"{TRANSLATIONS_DIR}/{dataset_dir}/mbr_samples.txt", "a", encoding="utf-8") as f:
            for s in sampled_translation:
                f.write(f"{s}\n")

    timer.end()

    return f"{TRANSLATIONS_DIR}/{dataset_dir}", tgt_lang

def mbr_rerank(config, dataset_dir, tgt_lang):

    # Rerank the MBR samples to get the MBR solution ===============================
    parser = argparse.ArgumentParser(description="mbr-nmt: minimum Bayes-risk decoding for neural machine translation")
    subparsers = parser.add_subparsers(dest="command")
    mbr_translate.create_parser(subparsers)
    mbr_convert.create_parser(subparsers)

    input_file = f'{dataset_dir}/mbr_samples.txt'

    for utility in config["utility_functions"]:
        print(f"MBR reranking using {utility} as the utility function.")
        output_file = f"{dataset_dir}/mbr_{utility}.txt"

        mbr_nmt_translate_args = [
            'translate',
            '--samples',
            input_file,
            '--num-samples',
            str(config["num_beams"]),
            '--utility',
            utility,
            '--lang',
            str(tgt_lang.alpha_2),
            '--output-file',
            output_file,
            '--encoding',
            'utf-8',
            '--threads',
            str(config.get("threads", -1)),
            ]
        mbr_nmt_translate_args = parser.parse_args(mbr_nmt_translate_args)

        mbr_translate.translate(mbr_nmt_translate_args)

        mbr_nmt_convert_args = [
            'convert',
            '--input-files',
            output_file,
            '--output-file',
            output_file,
            '--input-format',
            'mbr-nmt',
            '--lang',
            str(tgt_lang.alpha_2),
            '--encoding',
            'utf-8',
            ]
        mbr_nmt_convert_args = parser.parse_args(mbr_nmt_convert_args)

        mbr_convert.convert(mbr_nmt_convert_args)

@hydra.main(config_path="./nmt_adapt/config", config_name="translate")
def translate(config: DictConfig):

    dataset_dir, tgt_lang = generate_translations(config)

    mbr_rerank(config, dataset_dir, tgt_lang)

if __name__ == "__main__":

    print("Starting to generate translations.")
    translate()
