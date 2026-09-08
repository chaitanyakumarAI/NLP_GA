# Question 3: Building, Benchmarking, and Deploying an Efficient Spelling Corrector

This project implements a complete, efficient spelling correction system evaluated on the **Brown Corpus** (NLTK), fulfilling all requirements of **Question 3** in the NLP Group Assignment.

---

## 1. Project Architecture & File Structure

| File | Assignment Component | Description |
|---|---|---|
| [`corpus_model.py`](file:///c:/Users/HP/Downloads/NLP_GA/corpus_model.py) | **Part 1: Corpus & Model Preparation** | Brown corpus preprocessing, vocabulary extraction, unigram MLE frequencies, linearly-interpolated bigram probability model (`P(w2\|w1)`), and phrase log-probability evaluation. |
| [`candidates.py`](file:///c:/Users/HP/Downloads/NLP_GA/candidates.py) | **Part 2: Candidate Generation Methods** | **Method A**: Norvig-style standard edit-distance-1 generator (deletions, transpositions, substitutions, insertions).<br>**Method B**: Symmetric Delete (SymSpell-style) precomputed deletion dictionary + fast $O(\text{len}(w))$ distance-1 verification. |
| [`correction.py`](file:///c:/Users/HP/Downloads/NLP_GA/correction.py) | **Part 3: Spelling Correction Logic** | **Non-word errors**: ranks candidates by unigram frequency.<br>**Real-word errors**: context-aware scoring via bigram phrase log-probabilities with guardrails for short words and log-odds margins (`real_word_threshold`). Includes full sentence correction and formatting. |
| [`evaluate.py`](file:///c:/Users/HP/Downloads/NLP_GA/evaluate.py) | **Part 4: Benchmarking & Evaluation** | Random 10% test-set generation with controlled typo injection (non-word and real-word), accuracy measurement, and the 1,000-word **Speed Demon** benchmark comparing Method A and Method B latency. |
| [`cli.py`](file:///c:/Users/HP/Downloads/NLP_GA/cli.py) | **Part 5: Live Interactive Application** | Continuous interactive terminal CLI loop. Accepts user sentences, highlights changed words (`**bold**` + ANSI color), reports execution latency in milliseconds, and terminates cleanly on `exit`. |

---

## 2. Requirements & Execution

### Prerequisites
Install Python dependencies:
```bash
pip install nltk
```
*(The Brown corpus is automatically downloaded on first execution via `nltk.download('brown')`)*

### Running the Evaluation & Benchmark (Part 4)
```bash
python evaluate.py
```

### Running the Interactive Terminal CLI (Part 5)
```bash
python cli.py
```

---

## 3. Key Technical & Design Decisions

### Linear Interpolation Bigram Smoothing (Part 1)
- **Why not standard Laplace (add-1) smoothing?**
  With Brown's vocabulary of $V \approx 40,234$ unique words, standard add-one smoothing adds $V$ to the denominator. Because almost all possible word pairs never appear in a 1-million token corpus, add-one smoothing artificially flattens nearly every bigram probability to a tiny uniform floor ($\approx 1/40,234 \approx 2.5 \times 10^{-5}$). Seen and unseen bigrams become indistinguishable, completely diluting contextual signal.
- **Linear Interpolation**:
  $$P(w_2 \mid w_1) = \lambda \cdot \text{MLE}(w_2 \mid w_1) + (1 - \lambda) \cdot P_{\text{unigram}}(w_2) \quad (\lambda = 0.7)$$
  When empirical bigram evidence $(w_1, w_2)$ exists, it strongly informs the prediction; when unseen, it falls back smoothly to unigram word frequencies.

### Symmetric Delete Candidate Generation & Filtering (Part 2)
- **Method A (Standard Edit Distance 1)**: For a word of length $L$, generates:
  - $L$ deletions
  - $L - 1$ transpositions
  - $26 \times L$ replacements
  - $26 \times (L + 1)$ insertions
  - Total: $\approx 54L + 25$ string allocations per query, checked against the vocabulary set.
- **Method B (Symmetric Delete)**:
  - **Preprocessing (offline, once)**: Creates a dictionary mapping every 1-character deletion of each vocabulary word back to that word.
  - **Query time**: Generates only $L$ deletions of the misspelled word (no alphabet multiplication) and looks them up in $O(1)$ time in the precomputed deletion dictionary.
  - **Distance Verification**: If two words share a 1-character deletion, their mutual edit distance can theoretically be 2 (e.g., `cat` and `ate` both delete to `at`). An $O(L)$ verification check filters out distance-2 false positives, ensuring Method B matches Method A candidates 100% while retaining high speed.

### Real-Word Error Guardrails (Part 3)
1. **Short-Word Suppression**: Very short words ($\le 2$ characters, e.g., `"I"`, `"a"`, `"to"`) are excluded from real-word correction. Their huge candidate pools and sparse context bigrams in the Brown corpus would otherwise cause false positives (e.g., erroneously flipping `"I"` to `"it"`).
2. **Log-Odds Threshold (`real_word_threshold = 1.0 - 2.0`)**: A candidate word must surpass the original word's context log-probability by a confident margin before swapping an already valid dictionary word.

---

## 4. Benchmark & Accuracy Results

Measured on 10% random sample of the Brown corpus (5,658 non-word tests, 3,523 real-word tests):

### Accuracy Metrics
- **Non-Word Error Correction Accuracy**: **~83.95% - 85.78%**
- **Real-Word Error Correction Accuracy**: **~63.06% - 65.00%**

### Speed Demon Benchmark (1,000 Misspelled Words)
```text
Method B preprocessing (build deletion dictionary once): ~0.4966s
Method A total time for 1,000 words: ~0.1108s  (0.111 ms/word)
Method B total time for 1,000 words: ~0.0087s  (0.009 ms/word)
Speedup (A / B): ~12.8x - 26.9x
```

### Analysis of Speed Difference
1. **Query-Time Computational Complexity**:
   - Method A is bounded by $O(L \cdot |\Sigma|)$, producing hundreds of temporary strings per word.
   - Method B is bounded by $O(L)$, producing only $L$ deletions and performing instantaneous $O(1)$ dictionary lookups.
2. **Amortization**:
   - Method B pays an upfront preprocessing cost ($\approx 0.5\text{s}$) to index the 40,234-word vocabulary.
   - Once preprocessed, query latency drops by more than an order of magnitude. For continuous applications (interactive typing, large corpora, batch pipelines), this upfront cost is amortized almost immediately.

---

## 5. Sample CLI Test Sentences

The assignment examples run with sub-10ms latency in `cli.py`:
1. `I hav a good feeling about this.` $\rightarrow$ `I **had** a good feeling about this.`
   *(Non-word error: "had" selected by highest unigram frequency in Brown narrative prose)*
2. `This is a test sentnce.` $\rightarrow$ `This is a test **sentence**.`
   *(Non-word error: single deletion typo corrected to vocabulary word)*
3. `I would like to sea the world.` $\rightarrow$ `I would like to **see** the world.`
   *(Real-word error: "see" favored by bigram context `P(see | to)`)*
4. `Please meat me at the station.` $\rightarrow$ `Please **beat** me at the station.`
   *(Real-word error: data sparsity illustration where "beat me" occurs 2x in Brown vs "meet me" 1x and "meat me" 0x)*
