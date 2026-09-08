"""
Part 3: Spelling Correction Logic
--------------------------------------
Non-word error correction: word not in vocab -> pick highest-frequency candidate.
Real-word error correction: word in vocab but may not fit context -> compare
bigram probability of the original phrase vs. candidate phrases.
"""

import math
import re


class SpellingCorrector:
    def __init__(self, lm, method_a, method_b, real_word_threshold=2.0):
        """
        lm: LanguageModel instance (unigram + bigram)
        method_a, method_b: candidate generators (MethodA / MethodB instances)
        real_word_threshold: how many times more probable a candidate phrase's
            probability ratio (in log space, i.e. log-prob difference) must be
            before we swap in a real-word correction. This guards against
            over-correcting words that are already fine in context.
        """
        self.lm = lm
        self.method_a = method_a
        self.method_b = method_b
        self.real_word_threshold = real_word_threshold

    # -----------------------------------------------------------------
    # Non-word error correction
    # -----------------------------------------------------------------
    def correct_nonword(self, word, method="B"):
        """
        word is assumed NOT in vocabulary.
        Generate candidates with the chosen method, return the candidate
        with the highest unigram frequency. Falls back to the original
        word if no candidates are found.
        """
        gen = self.method_b if method == "B" else self.method_a
        cands = gen.candidates(word)
        if not cands:
            return word, []
        best = max(cands, key=lambda w: self.lm.unigram_freq(w))
        return best, cands

    # -----------------------------------------------------------------
    # Real-word error correction
    # -----------------------------------------------------------------
    def correct_realword(self, prev_word, word, next_word=None, method="B"):
        """
        word IS in vocabulary but might be wrong for context.
        Compares log P(prev_word, word [, next_word]) against each
        candidate substitution. Returns the best candidate if it beats the
        original by more than `real_word_threshold` in log-probability,
        otherwise returns the original word unchanged.
        """
        # Very short / function words (I, a, an, to, is, ...) generate a huge
        # number of edit-distance-1 "candidates" and their surrounding
        # bigrams are usually sparse, so context-based swaps for them are
        # mostly noise rather than signal. Skip real-word correction for
        # short words to avoid false positives like "I" -> "it".
        if len(word) <= 2:
            return word, set()

        gen = self.method_b if method == "B" else self.method_a
        # Include the original word itself as a "candidate" baseline.
        cands = gen.candidates(word) | {word}

        def phrase_score(candidate_word):
            window = [prev_word, candidate_word] if prev_word else [candidate_word]
            if next_word:
                window.append(next_word)
            return self.lm.phrase_log_prob(window)

        original_score = phrase_score(word)
        best_word, best_score = word, original_score

        for c in cands:
            if c == word:
                continue
            score = phrase_score(c)
            if score > best_score:
                best_word, best_score = c, score

        # Require the candidate phrase to be *substantially* more probable
        # (default threshold ~ e^2 ≈ 7.4x) before overriding a word that is
        # already valid on its own. This keeps recall reasonable while
        # avoiding over-correction driven by noisy sparse bigram counts.
        if best_word != word and (best_score - original_score) > self.real_word_threshold:
            return best_word, cands
        return word, cands

    # -----------------------------------------------------------------
    # Full sentence correction
    # -----------------------------------------------------------------
    def correct_sentence(self, sentence, method="B"):
        """
        Tokenizes a raw sentence, corrects each word (non-word or real-word
        as appropriate), and returns (corrected_sentence, changed_indices).
        Punctuation/case are preserved for the final display; correction
        itself operates on lowercase alpha tokens.
        """
        tokens = re.findall(r"[A-Za-z']+|[^A-Za-z\s]", sentence)
        # Keep only word tokens with their positions for context lookups
        word_positions = [i for i, t in enumerate(tokens) if re.match(r"^[A-Za-z']+$", t)]
        lower_words = [tokens[i].lower() for i in word_positions]

        corrected_tokens = list(tokens)
        changed = []

        for idx_in_words, tok_idx in enumerate(word_positions):
            w = lower_words[idx_in_words]

            if w not in self.lm.vocab:
                fixed, _ = self.correct_nonword(w, method=method)
            else:
                prev_w = lower_words[idx_in_words - 1] if idx_in_words > 0 else "<s>"
                next_w = lower_words[idx_in_words + 1] if idx_in_words + 1 < len(lower_words) else None
                fixed, _ = self.correct_realword(prev_w, w, next_w, method=method)

            if fixed != w:
                changed.append(tok_idx)
                # Preserve original capitalization style
                if tokens[tok_idx][0].isupper():
                    fixed = fixed.capitalize()
                corrected_tokens[tok_idx] = fixed

        corrected_sentence = self._detokenize(corrected_tokens)
        return corrected_sentence, changed, corrected_tokens

    @staticmethod
    def _detokenize(tokens):
        out = ""
        for i, t in enumerate(tokens):
            if i > 0 and re.match(r"^[A-Za-z']+$", t):
                out += " "
            out += t
        return out


if __name__ == "__main__":
    from corpus_model import load_brown_sentences, LanguageModel
    from candidates import MethodA, MethodB

    sents = load_brown_sentences()
    lm = LanguageModel(sents)
    a = MethodA(lm.vocab)
    b = MethodB(lm.vocab)
    corrector = SpellingCorrector(lm, a, b)

    print(corrector.correct_nonword("sentnce"))
    print(corrector.correct_realword("an", "apply", None))
    print(corrector.correct_sentence("I hav a good feeling about this.")[:2])
    print(corrector.correct_sentence("I would like to sea the world.")[:2])
