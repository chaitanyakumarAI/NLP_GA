"""
Part 5: Live Interactive Application
--------------------------------------
A continuous terminal CLI:
  - prompts the user for a sentence
  - prints the corrected sentence, with changed words wrapped in **asterisks**
    and shown in color (ANSI) when the terminal supports it
  - prints the latency of the correction
  - exits when the user types "exit"
"""

import sys
import time

from corpus_model import load_brown_sentences, LanguageModel
from candidates import MethodA, MethodB
from correction import SpellingCorrector

GREEN = "\033[92m"
RESET = "\033[0m"


import re


def highlight(tokens, changed_word_indices):
    """
    Wrap changed word tokens in **asterisks** + ANSI green, then rebuild the
    sentence, using the ORIGINAL (undecorated) tokens to decide spacing —
    decorated tokens contain ANSI escape codes and asterisks, so they no
    longer match the "is this a word" regex used by plain detokenization.
    """
    is_word = [bool(re.match(r"^[A-Za-z']+$", t)) for t in tokens]
    decorated = list(tokens)
    for i in changed_word_indices:
        decorated[i] = f"{GREEN}**{decorated[i]}**{RESET}"

    out = ""
    for i, t in enumerate(decorated):
        if i > 0 and is_word[i]:
            out += " "
        out += t
    return out


def main():
    print("=" * 60)
    print("  Spelling Corrector — Interactive CLI")
    print("  (built on Brown corpus unigram + bigram models)")
    print("  Type a sentence and press Enter. Type 'exit' to quit.")
    print("=" * 60)

    print("\nLoading corpus and building models, please wait...")
    t0 = time.perf_counter()
    sentences = load_brown_sentences()
    lm = LanguageModel(sentences)
    method_a = MethodA(lm.vocab)
    method_b = MethodB(lm.vocab)  # preprocessing done once, here, at startup
    corrector = SpellingCorrector(lm, method_a, method_b, real_word_threshold=1.0)
    print(f"Ready. (Setup took {time.perf_counter() - t0:.2f}s)\n")

    while True:
        try:
            sentence = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if sentence.lower() == "exit":
            print("Goodbye!")
            break
        if not sentence:
            continue

        t0 = time.perf_counter()
        corrected, changed, corrected_tokens = corrector.correct_sentence(sentence, method="B")
        latency = time.perf_counter() - t0

        display = highlight(corrected_tokens, changed)

        if changed:
            print(f"Corrected: {display}")
        else:
            print(f"No changes needed: {corrected}")
        print(f"(latency: {latency * 1000:.2f} ms)\n")


if __name__ == "__main__":
    main()
