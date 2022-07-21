import argparse
import logging
import subprocess
from pathlib import Path
from typing import List

import pyconll
import lambre
from lambre.metric import check_lang, parse_doc, compute_metric
from lambre.parse_utils import get_depd_tree

def parse_doc(
    doc: str,
    lg: str,
    stanza_path: Path,
    output: Path,
    ssplit: bool,
    verbose: bool,
    file_name: str = None,
):

    depd_tree = get_depd_tree(
        doc=doc,
        lg=lg,
        stanza_model_path=stanza_path,
        ssplit=ssplit,
        verbose=verbose,
    )
    sentences = pyconll.load_from_string(depd_tree)
    if file_name:
        parser_out_path = output / f"{file_name}.conllu"
        logging.info(f"storing .conllu file at {parser_out_path}")
        with open(parser_out_path, "w", encoding="utf-8") as wf:
            wf.write(depd_tree)

    return sentences

def parse_args():
    parser = argparse.ArgumentParser(
        description="compute morphological well-formedness"
    )
    parser.add_argument("lg", type=str, help="input language ISO 639-1 code")
    parser.add_argument("input", type=Path, help="input file (.txt or .conllu)")
    parser.add_argument(
        "--rule-set",
        type=str,
        choices=["chaudhary-etal-2021", "pratapa-etal-2021"],
        default="chaudhary-etal-2021",
        help="rule set name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="out",
        help="specify path to output directory. Stores parser output and error visualizations.",
    )
    parser.add_argument(
        "--score-sent", action="store_true", help="return sentence level scores"
    )
    parser.add_argument(
        "--ssplit",
        action="store_true",
        help="perform sentence segmentation in addition to tokenization",
    )
    parser.add_argument("--report", action="store_true", help="report scores per rule")
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=Path.home() / "lambre_files" / "rules",
        help="path to rule sets",
    )
    parser.add_argument(
        "--stanza-path",
        type=Path,
        default=Path.home() / "lambre_files" / "lambre_stanza_resources",
        help="path to stanza resources",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    return parser.parse_args()

def main():

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    args = vars(parse_args())

    if not check_lang(lg=args["lg"], stanza_path=args["stanza_path"]):
        return

    inp = Path(args["input"])
    args["output"].mkdir(exist_ok=True)
    if inp.suffix == ".conllu":
        # input CoNLL-U file, directly load the file
        sentences = pyconll.load_from_file(inp)
    else:
        # input txt file, parse
        doc = ""
        with open(inp, "r", encoding="utf-8") as rf:
            for line in rf:
                doc += line
                if not args["ssplit"]:
                    doc += "\n"
        sentences = parse_doc(
            doc=doc,
            lg=args["lg"],
            stanza_path=args["stanza_path"],
            output=args["output"],
            ssplit=args["ssplit"],
            verbose=args["verbose"],
            file_name=inp.stem,
        )

    compute_metric(
        sentences=sentences,
        lg=args["lg"],
        score_sent=args["score_sent"],
        rule_set=args["rule_set"],
        rules_path=args["rules_path"],
        report=args["report"],
        verbose=args["verbose"],
        output=args["output"],
    )


if __name__ == "__main__":
    main()
