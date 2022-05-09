NAVA_POS = {"N", "ADJ", "AUX", "V", "ADV"}

class MarginalTask(object):
    """A marginal task: (morph tag set, lemma script) combination.

    Args:
        object (_type_): _description_
    """

    def __init__(self, morph_tag_set, lemma_edit_script):

        self.morph_tag_set = sorted(list(morph_tag_set))
        self.lemma_edit_script = lemma_edit_script

    def contains(self, check_set):
        return len(set.intersection(set(self.morph_tag_set), check_set)) >= 1

    def is_nava(self):
        return self.contains(NAVA_POS)

    def __repr__(self):
        return f"MarginalTask(f{set(self.morph_tag_set)}, '{self.lemma_edit_script}')"

    def __str__(self):
        return self.__repr__()

    def match(self, other):

        matches = 0
        if self.morph_tag_set == other.morph_tag_set:
            matches += 1

        if self.lemma_edit_script == other.lemma_edit_script:
            matches += 1

        return matches

    def __eq__(self, other):
        return self.match(other) == 2

    def __hash__(self):
        return hash((frozenset(self.morph_tag_set), self.lemma_edit_script))