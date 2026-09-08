"""
Part 4: Evaluation and "Speed Demon" Benchmark
--------------------------------------------------
1. Build a test set: sample 10% of Brown sentences, inject one random
   single-edit typo per sentence -> a non-word version and a real-word version.
2. Report accuracy for both error types.
3. Speed Demon: time Method A vs Method B on 1,000 misspelled words.
"""

import random
import string
import time

from corpus_model import load_brown_sentences, LanguageModel
from candidates import MethodA, MethodB
from correction import SpellingCorrector

random.seed(42)
ALPHABET = string.ascii_lowercase


# ---------------------------------------------------------------------------
# Typo injection
# ---------------------------------------------------------------------------
def make_single_edit_typo(word, rng):
    """Apply one random edit (delete/insert/substitute/transpose) to `word`."""
    if len(word) < 2:
        return None
    ops = ["delete", "insert", "substitute"]
    if len(word) >= 2:
        ops.append("transpose")
    op = rng.choice(ops)

    if op == "delete":
        i = rng.randrange(len(word))
        return word[:i] + word[i + 1:]
    if op == "insert":
        i = rng.randrange(len(word) + 1)
        c = rng.choice(ALPHABET)
        return word[:i] + c + word[i:]
    if op == "substitute":
        i = rng.randrange(len(word))
        c = rng.choice([ch for ch in ALPHABET if ch != word[i]])
        return word[:i] + c + word[i + 1:]
    if op == "transpose" and len(word) >= 2:
        i = rng.randrange(len(word) - 1)
        lst = list(word)
        lst[i], lst[i + 1] = lst[i + 1], lst[i]
        return "".join(lst)
    return word


def build_test_set(sentences, lm, sample_fraction=0.1, rng=None):
    """
    For 10% of sentences, pick one word at random and create:
      - a non-word-error version: typo that (ideally) is NOT a real vocab word
      - a real-word-error version: typo that (ideally) IS a real vocab word

    Each test item is a dict:
      {sentence_words, idx, original_word, nonword_typo, realword_typo}
    Some sentences may not yield a valid real-word typo (i.e. no single-edit
    variant happens to also be a vocab word); those are skipped for the
    real-word set but can still contribute to the non-word set.
    """
    rng = rng or random.Random(42)
    n = int(len(sentences) * sample_fraction)
    sampled = rng.sample(sentences, n)

    nonword_tests = []
    realword_tests = []

    for sent in sampled:
        eligible_idx = [i for i, w in enumerate(sent) if len(w) >= 3]
        if not eligible_idx:
            continue
        idx = rng.choice(eligible_idx)
        original = sent[idx]

        # --- non-word typo: keep retrying until it's NOT a vocab word ---
        nonword_typo = None
        for _ in range(10):
            cand = make_single_edit_typo(original, rng)
            if cand and cand not in lm.vocab and cand != original:
                nonword_typo = cand
                break
        if nonword_typo:
            nonword_tests.append({
                "sentence": sent, "idx": idx,
                "original": original, "typo": nonword_typo,
            })

        # --- real-word typo: keep retrying until it IS a vocab word ---
        realword_typo = None
        for _ in range(30):
            cand = make_single_edit_typo(original, rng)
            if cand and cand in lm.vocab and cand != original:
                realword_typo = cand
                break
        if realword_typo:
            realword_tests.append({
                "sentence": sent, "idx": idx,
                "original": original, "typo": realword_typo,
            })

    return nonword_tests, realword_tests


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------
def evaluate_nonword(corrector, tests, method="B"):
    correct = 0
    for t in tests:
        fixed, _ = corrector.correct_nonword(t["typo"], method=method)
        if fixed == t["original"]:
            correct += 1
    return correct / len(tests) if tests else 0.0


def evaluate_realword(corrector, tests, method="B"):
    correct = 0
    for t in tests:
        sent, idx = t["sentence"], t["idx"]
        prev_w = sent[idx - 1] if idx > 0 else "<s>"
        next_w = sent[idx + 1] if idx + 1 < len(sent) else None
        fixed, _ = corrector.correct_realword(prev_w, t["typo"], next_w, method=method)
        if fixed == t["original"]:
            correct += 1
    return correct / len(tests) if tests else 0.0


# ---------------------------------------------------------------------------
# Speed Demon Benchmark
# ---------------------------------------------------------------------------
def speed_demon_benchmark(method_a, method_b, vocab, n=1000, rng=None):
    """
    Build a batch of exactly `n` misspelled words (drawn from real vocab
    words with one random edit applied), then time Method A and Method B
    processing the SAME batch.
    """
    rng = rng or random.Random(7)
    vocab_list = [w for w in vocab if len(w) >= 3]
    batch = []
    while len(batch) < n:
        w = rng.choice(vocab_list)
        typo = make_single_edit_typo(w, rng)
        if typo:
            batch.append(typo)

    start = time.perf_counter()
    for w in batch:
        method_a.candidates(w)
    time_a = time.perf_counter() - start

    start = time.perf_counter()
    for w in batch:
        method_b.candidates(w)
    time_b = time.perf_counter() - start

    return time_a, time_b, batch


if __name__ == "__main__":
    print("Loading corpus and building models...")
    sentences = load_brown_sentences()
    lm = LanguageModel(sentences)
    method_a = MethodA(lm.vocab)
    method_b = MethodB(lm.vocab)  # preprocessing happens here (timed separately below)
    corrector = SpellingCorrector(lm, method_a, method_b, real_word_threshold=1.0)

    print("\nBuilding test set (10% of sentences)...")
    nonword_tests, realword_tests = build_test_set(sentences, lm, sample_fraction=0.1)
    print(f"Non-word test items: {len(nonword_tests)}")
    print(f"Real-word test items: {len(realword_tests)}")

    print("\n--- Accuracy (Method B candidates used for correction) ---")
    acc_nonword = evaluate_nonword(corrector, nonword_tests, method="B")
    acc_realword = evaluate_realword(corrector, realword_tests, method="B")
    print(f"Non-word error correction accuracy: {acc_nonword:.2%}")
    print(f"Real-word error correction accuracy: {acc_realword:.2%}")

    print("\n--- Speed Demon Benchmark (1,000 misspelled words) ---")
    # Time Method B's one-time preprocessing separately, since it's a
    # fixed upfront cost that's already paid by the time we reach this point.
    t0 = time.perf_counter()
    _ = MethodB(lm.vocab)
    preprocess_time = time.perf_counter() - t0
    print(f"Method B preprocessing (build deletion dictionary once): {preprocess_time:.4f}s")

    time_a, time_b, batch = speed_demon_benchmark(method_a, method_b, lm.vocab, n=1000)
    print(f"Method A total time for 1,000 words: {time_a:.4f}s  ({time_a/1000*1000:.3f} ms/word)")
    print(f"Method B total time for 1,000 words: {time_b:.4f}s  ({time_b/1000*1000:.3f} ms/word)")
    print(f"Speedup (A/B): {time_a / time_b:.1f}x")

    print("\n" + "=" * 60)
    print("CONCLUSION: SPEED DEMON BENCHMARK ANALYSIS")
    print("=" * 60)
    print(
        f"1. Query-Time Complexity Difference:\n"
        f"   - Method A generates O(len(w) * |alphabet|) = ~54 * len(w) candidate strings\n"
        f"     per query (all deletions, transpositions, substitutions, and insertions)\n"
        f"     and checks each against the vocabulary set. For average English words (length 5-8),\n"
        f"     this constructs 250-450 strings per word in memory.\n"
        f"   - Method B (Symmetric Delete) generates only O(len(w)) strings at query time\n"
        f"     (only its own 1-character deletions, e.g. 5-8 strings, with no alphabet expansion),\n"
        f"     followed by instant O(1) hash table lookups into the precomputed deletion dictionary.\n"
        f"\n"
        f"2. Upfront vs. Per-Query Trade-off:\n"
        f"   - Method B pays a one-time preprocessing cost ({preprocess_time:.4f}s) to build the\n"
        f"     deletion index across {len(lm.vocab):,} vocabulary words.\n"
        f"   - In return, per-word latency drops from {time_a/1000*1000:.3f} ms/word to {time_b/1000*1000:.3f} ms/word,\n"
        f"     achieving a ~{time_a / time_b:.1f}x speedup at query time.\n"
        f"   - This makes Method B asymptotically superior for live typing, search engines, and\n"
        f"     large-scale document proofreading where query throughput is critical.\n"
        f"=" * 60
    )
