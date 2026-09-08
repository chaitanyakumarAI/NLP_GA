"""
Part 1: Corpus and Model Preparation
--------------------------------------
Builds:
  - A vocabulary of unique words from the Brown corpus
  - A unigram frequency distribution
  - A bigram probability model (with add-one/Laplace smoothing)

These are used downstream for:
  - Non-word error correction (unigram frequency ranks candidates)
  - Real-word error correction (bigram probability judges context fit)
"""

import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import brown

nltk.download("brown", quiet=True)


def load_brown_sentences():
    """Return Brown corpus as a list of sentences (each a list of lowercase, alpha-only tokens)."""
    sentences = []
    for sent in brown.sents():
        cleaned = [w.lower() for w in sent if re.match(r"^[a-zA-Z]+$", w)]
        if cleaned:
            sentences.append(cleaned)
    return sentences


class LanguageModel:
    def __init__(self, sentences):
        self.sentences = sentences

        # ---- Unigram model ----
        self.unigram_counts = Counter()
        for sent in sentences:
            self.unigram_counts.update(sent)
        self.vocab = set(self.unigram_counts.keys())
        self.total_unigrams = sum(self.unigram_counts.values())

        # ---- Bigram model ----
        # bigram_counts[w1][w2] = count of w2 following w1
        self.bigram_counts = defaultdict(Counter)
        for sent in sentences:
            padded = ["<s>"] + sent + ["</s>"]
            for w1, w2 in zip(padded, padded[1:]):
                self.bigram_counts[w1][w2] += 1

        self.context_counts = {w1: sum(counts.values()) for w1, counts in self.bigram_counts.items()}
        self.vocab_size = len(self.vocab) + 2  # + <s> and </s>

    def unigram_freq(self, word):
        """Raw frequency count of a word (0 if unseen)."""
        return self.unigram_counts.get(word, 0)

    def unigram_prob(self, word):
        """Simple MLE unigram probability."""
        return self.unigram_counts.get(word, 0) / self.total_unigrams

    def bigram_prob(self, w1, w2, lam=0.7):
        """
        Linearly-interpolated bigram probability:
            P(w2|w1) = lam * MLE(w2|w1) + (1-lam) * P_unigram(w2)

        Plain add-one (Laplace) smoothing was tried first, but with a ~40k
        word vocabulary it drowns out real bigram counts: almost every
        bigram (seen or unseen) gets pushed to nearly the same tiny
        probability, so context barely matters. Interpolating with the
        unigram distribution instead means: when we've actually seen
        (w1, w2) enough times, that dominates; when we haven't, we fall
        back gracefully to how common w2 is on its own, rather than to a
        near-uniform floor.
        """
        count_w1 = self.context_counts.get(w1, 0)
        mle = (self.bigram_counts[w1][w2] / count_w1) if count_w1 > 0 else 0.0
        unigram_p = (self.unigram_counts.get(w2, 0) + 1) / (self.total_unigrams + self.vocab_size)
        return lam * mle + (1 - lam) * unigram_p

    def phrase_log_prob(self, words):
        """
        Log probability of a short phrase (list of words) using the bigram
        chain rule: P(w1)*P(w2|w1)*P(w3|w2)*...
        Using log-probabilities avoids underflow and makes comparison easy.
        """
        import math
        tokens = list(words)
        if not tokens:
            return 0.0
        if tokens[0] != "<s>":
            tokens = ["<s>"] + tokens
        if tokens[-1] != "</s>":
            tokens = tokens + ["</s>"]
        log_p = 0.0
        for w1, w2 in zip(tokens, tokens[1:]):
            p = self.bigram_prob(w1, w2)
            log_p += math.log(p)
        return log_p


if __name__ == "__main__":
    sents = load_brown_sentences()
    lm = LanguageModel(sents)
    print(f"Sentences: {len(sents)}")
    print(f"Vocabulary size: {len(lm.vocab)}")
    print(f"Total tokens: {lm.total_unigrams}")
    print(f"Freq('the') = {lm.unigram_freq('the')}")
    print(f"Freq('apple') = {lm.unigram_freq('apple')}")
    print(f"P(apple | an) = {lm.bigram_prob('an', 'apple'):.6f}")
    print(f"P(apply | an) = {lm.bigram_prob('an', 'apply'):.6f}")
