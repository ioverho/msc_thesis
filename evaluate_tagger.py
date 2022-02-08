import os
import pickle
from pathlib import Path
import re

import torch
import hydra

from utils.experiment import progressbar, Timer
from morphological_tagging.data.corpus import TreebankDataModule
from morphological_tagging.data.lemma_script import apply_lemma_script
from morphological_tagging.models2 import UDIFY, UDPipe2

EVAL_PATH = Path("./morphological_tagging/eval")
CORPORA_PATH = Path("./morphological_tagging/data/corpora")
CHECKPOINT_PATH = Path("./morphological_tagging/checkpoints")

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="./morphological_tagging/config", config_name="eval")
def eval(config):

    timer = Timer()

    print(f"\n{timer.time()} | INITIALIZATION")
    use_cuda = config["gpu"] or config["gpu"] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Running on {device}" + f"- {torch.cuda.get_device_name(0)}"
        if use_cuda
        else ""
    )

    # ==========================================================================
    # Model Import
    # ==========================================================================
    print(f"\n{timer.time()} | MODEL")
    chkpt_dir_path = f"**/{config['model_name']}_{config['dataset_name']}/*"

    matches = list(CHECKPOINT_PATH.glob(chkpt_dir_path))

    if len(matches) == 0:
        raise ValueError(
            f"No model checkpoint found at {CHECKPOINT_PATH}/{config['model_name']}_{config['dataset_name']}"
        )

    elif len(matches) >= 1:
        dirs = []
        for m in matches:
            if len(list(m.glob("**/*.ckpt"))) > 0:
                re_search = re.search("(?<=version_)([0-9]+)", m.parts[-1])
                dirs.append((m, int(re_search.groups()[0])))

        latest_dir = sorted(dirs, key=lambda x: x[1], reverse=True)[0][0]

        if config["mode"] == "best":
            match = list((latest_dir / "checkpoints").glob("**/epoch*.ckpt"))[0]

        elif config["mode"] == "last":
            match = list((latest_dir / "checkpoints").glob("**/last.ckpt"))[0]

        else:
            raise ValueError(f"Mode {config['mode']} not recognized.")

        print(f"Found model at {match}")

    if config["model_name"].lower() == "udpipe2":
        model = UDPipe2.load_from_checkpoint(str(match), map_location=device)
        model.eval()
        model.freeze()

    elif config["model_name"].lower() == "udify":
        raise NotImplementedError()

    # ==========================================================================
    # Dataset Import
    # ==========================================================================
    print(f"\n{timer.time()} | DATASET")

    expected_dataset_path = (
        CORPORA_PATH
        / f"{config['dataset_name']}_{config['quality_limit']}_{config['batch_first']}.pickle"
    )

    if expected_dataset_path.exists():
        print(f"Found predefined dataset at {expected_dataset_path}")

    data_module = TreebankDataModule.load(expected_dataset_path)

    # ==========================================================================
    # Eval File
    # ==========================================================================
    print(f"\n{timer.time()} | EVALUATION")
    eval_fp = (
        EVAL_PATH
        / f"{config['model_name']}_{config['dataset_name']}_{config['quality_limit']}_{config['batch_first']}.pickle"
    )
    print(f"Saving output to {eval_fp}")
    if EVAL_PATH.exists() and config["overwrite_ok"]:
        print(">>>OVERWRITING<<<")

    with open(eval_fp, "wb") as f:

        # ======================================================================
        # Evaluating and streaming to file
        # ======================================================================
        for batch in progressbar(data_module.test_dataloader(), "Evaluating", size=60):
            (_, _, _, _, tokens, _, lemma_tags, morph_tags, _,) = batch

            lemma_preds, morph_preds = model.pred_step(batch)

            for i, (tok, l_pred, l_tag, m_pred, m_tag) in enumerate(
                zip(
                    torch.flatten(tokens).cpu().numpy(),
                    torch.flatten(lemma_preds).cpu().numpy(),
                    torch.flatten(lemma_tags).cpu().numpy(),
                    torch.flatten(morph_preds, end_dim=1).cpu().long().numpy(),
                    torch.flatten(morph_tags, end_dim=1).cpu().numpy(),
                )
            ):

                if l_tag == -1:
                    continue

                word_form = data_module.corpus.token_vocab.lookup_token(tok)

                pred_lemma_script = data_module.corpus.id_to_script[l_pred]
                gt_lemma_script = data_module.corpus.id_to_script[l_tag]

                gt_lemma = apply_lemma_script(word_form, gt_lemma_script)
                # Since not all predicted scripts will align with the word form
                # The application might raise some errors
                pred_lemma = apply_lemma_script(
                    word_form, pred_lemma_script, verbose=False
                )

                pickle.dump(
                    (word_form, gt_lemma, pred_lemma, l_tag, l_pred, m_tag, m_pred,), f
                )

    print(f"\n{timer.time()} | FINISHED")

    return eval_fp


if __name__ == "__main__":

    eval_fp = eval()
