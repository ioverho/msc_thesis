import yaml
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import re

from bs4 import BeautifulSoup, Comment, Tag

URL = "https://universaldependencies.org/"


def scrape():
    """Scrape the UD project page for metadata of the treebanks.
    """
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.get(URL)

    webpage_content = session.get(URL).text

    soup = BeautifulSoup(webpage_content, "html.parser")

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    treebank_metadata = {}
    for c in comments:
        matches = re.search(r"(?<=start of )(.*) \/ (.*)(?= entry)", c.string)
        if matches is not None:
            language, treebank = matches.group(1), matches.group(2)
            language = language.replace(" ", "_")
            parent = c.parent

            meta_data = []
            for i, subtag in enumerate(parent.contents):
                if isinstance(subtag, Tag):
                    if len(subtag.contents):
                        if isinstance(subtag.contents[0], Tag):
                            subsubstag = subtag.contents[0]
                        else:
                            continue

                        meta_data.append(subsubstag.attrs.get("data-hint"))

            size_matches = re.search(
                r"(.+)(?: tokens )(.+)(?: words )(.+)(?: sentences)", meta_data[0]
            )

            treebank_metadata[f"{language}_{treebank}"] = {
                "language": language,
                "size": {
                    "tokens": int(size_matches.group(1).replace(",", "")),
                    "words": int(size_matches.group(2).replace(",", "")),
                    "sentences": int(size_matches.group(3).replace(",", "")),
                },
                "source_genres": meta_data[2].split(" "),
                "quality": float(meta_data[4]),
                "license": meta_data[3],
            }

    treebank_metadata["French_Spoken"] = treebank_metadata["French_Rhapsodie"]
    del treebank_metadata["French_Rhapsodie"]

    treebank_metadata["Polish_SZ"] = treebank_metadata["Polish_PDB"]
    del treebank_metadata["Polish_PDB"]


if "__name__" == "__main__":

    treebank_metadata = scrape()

    with open("./morphological_tagging/data/treebank_metadata.yaml", "w") as outfile:
        yaml.dump(treebank_metadata, outfile, default_flow_style=False)
