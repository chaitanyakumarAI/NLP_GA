"""
Part 2: Candidate Generation Methods
--------------------------------------
Method A: Standard edit-distance-1 generation (brute force).
Method B: Symmetric Delete Spelling Correction (SymSpell-style).
"""

import string
from collections import defaultdict

ALPHABET = string.ascii_lowercase


# ---------------------------------------------------------------------------
# Method A: Standard edit distance 1
# ---------------------------------------------------------------------------
def edits1(word):
    """
    Generate every string at edit distance 1 from `word`:
    deletions, transpositions, replacements, and insertions.
    (This is the classic Norvig-style generator.)
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET]
    inserts = [L + c + R for L, R in splits for c in ALPHABET]

    return set(deletes + transposes + replaces + inserts)


class MethodA:
    """Standard edit-distance-1 candidate generator."""

    def __init__(self, vocab):
        self.vocab = vocab

    def candidates(self, word):
        """Return the subset of edit-distance-1 strings that are real vocab words."""
        return {w for w in edits1(word) if w in self.vocab}


# ---------------------------------------------------------------------------
# Method B: Symmetric Delete Spelling Correction (SymSpell)
# ---------------------------------------------------------------------------
def deletes1(word):
    """All strings formed by deleting exactly one character from `word`."""
    if len(word) == 0:
        return set()
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def is_edit_distance_1(w1, w2):
    """
    Check if w1 and w2 are at Damerau-Levenshtein distance <= 1:
    - 0 edits: w1 == w2
    - 1 substitution: same length, 1 mismatch
    - 1 transposition: same length, 2 adjacent swapped chars
    - 1 deletion / insertion: length diff 1, all other chars match
    Runs in O(len(w)) time.
    """
    if w1 == w2:
        return True
    l1, l2 = len(w1), len(w2)
    if abs(l1 - l2) > 1:
        return False
    if l1 == l2:
        diffs = [i for i in range(l1) if w1[i] != w2[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return j == i + 1 and w1[i] == w2[j] and w1[j] == w2[i]
        return False
    if l1 > l2:
        w1, w2 = w2, w1
        l1, l2 = l2, l1
    i = 0
    while i < l1 and w1[i] == w2[i]:
        i += 1
    return w1[i:] == w2[i + 1:]


class MethodB:
    """
    Symmetric Delete candidate generator.

    Preprocessing (done ONCE, upfront):
        For every vocabulary word, generate all its one-character deletions
        and store them in a dict: deleted_variant -> [original_word, ...]

    Query time:
        Generate the one-character deletions of the *misspelled* word
        (and include the word itself, in case it's a 1-letter insertion away
        from a vocab word), then look those up in the precomputed dict.
        Candidates are verified to be at edit distance <= 1 in O(len(w)) time,
        preventing spurious distance-2 words (where both words share a delete)
        from leaking into the candidate set.
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.delete_dict = defaultdict(set)
        self._preprocess()

    def _preprocess(self):
        for w in self.vocab:
            self.delete_dict[w].add(w)  # word maps to itself (0 deletions)
            for d in deletes1(w):
                self.delete_dict[d].add(w)

    def candidates(self, word):
        found = set()

        # 0-deletion: word itself might already be valid or a 1-deletion of vocab word
        if word in self.delete_dict:
            found |= self.delete_dict[word]

        # 1-deletion from the misspelled word
        for d in deletes1(word):
            if d in self.delete_dict:
                found |= self.delete_dict[d]

        # Filter out candidates with edit distance > 1 (SymSpell verification step)
        return {c for c in found if is_edit_distance_1(word, c)}


if __name__ == "__main__":
    vocab = {"hello", "help", "hell", "yellow", "apple", "an", "sentence", "test"}
    a = MethodA(vocab)
    b = MethodB(vocab)

    for typo in ["helo", "sentnce", "aple"]:
        print(f"{typo!r} -> MethodA: {a.candidates(typo)} | MethodB: {b.candidates(typo)}")
