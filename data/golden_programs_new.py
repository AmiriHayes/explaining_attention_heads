"""Helper functions available to generated attention-prediction code.

These are injected into the execution environment so generated code can import
them via `from helpers import *`.
"""

import numpy as np
import spacy

_nlp = None
_gpt2_tok = None


def get_nlp():
    """Return a cached spacy English model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_gpt2_tokenizer():
    """Return a cached GPT2 tokenizer."""
    global _gpt2_tok
    if _gpt2_tok is None:
        from transformers import GPT2Tokenizer
        _gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    return _gpt2_tok


def gpt2_tokenize(sentence: str) -> list[str]:
    """Tokenize a sentence using GPT2 BPE tokenizer.

    Returns a list of token strings. Leading spaces are included in tokens
    (e.g. " cat" not "cat") to match GPT2's convention.
    """
    tok = _get_gpt2_tokenizer()
    ids = tok.encode(sentence)
    return [tok.decode([i]) for i in ids]


def spacy_parse(sentence: str):
    """Parse a sentence with spacy, returning a Doc object."""
    return get_nlp()(sentence)


def align_spacy_to_gpt2(sentence: str) -> list[list[int]]:
    """For each spacy token, return the list of overlapping GPT2 token indices.

    Uses character offsets to align between the two tokenizations.
    """
    doc = spacy_parse(sentence)
    gpt2_tokens = gpt2_tokenize(sentence)

    # Build GPT2 character spans
    gpt2_spans = []
    pos = 0
    for t in gpt2_tokens:
        gpt2_spans.append((pos, pos + len(t)))
        pos += len(t)

    alignment = []
    for spacy_tok in doc:
        s_start, s_end = spacy_tok.idx, spacy_tok.idx + len(spacy_tok.text)
        overlapping = [
            g_idx for g_idx, (g_start, g_end) in enumerate(gpt2_spans)
            if g_start < s_end and g_end > s_start
        ]
        alignment.append(overlapping)
    return alignment


def align_gpt2_to_spacy(sentence: str) -> list[list[int]]:
    """For each GPT2 token, return the list of overlapping spacy token indices.

    Uses character offsets to align between the two tokenizations.
    """
    doc = spacy_parse(sentence)
    gpt2_tokens = gpt2_tokenize(sentence)

    # Build GPT2 character spans
    gpt2_spans = []
    pos = 0
    for t in gpt2_tokens:
        gpt2_spans.append((pos, pos + len(t)))
        pos += len(t)

    alignment = []
    for g_idx, (g_start, g_end) in enumerate(gpt2_spans):
        overlapping = [
            s_idx for s_idx, spacy_tok in enumerate(doc)
            if spacy_tok.idx < g_end and (spacy_tok.idx + len(spacy_tok.text)) > g_start
        ]
        alignment.append(overlapping)
    return alignment


def make_row_stochastic(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix to sum to 1.

    Rows that sum to zero are left as-is.
    """
    matrix = matrix.copy().astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return matrix / row_sums


def apply_causal_mask(matrix: np.ndarray) -> np.ndarray:
    """Zero out upper-triangular entries (enforce causal / autoregressive mask).

    GPT2 is decoder-only, so token i can only attend to tokens j <= i.
    """
    n = matrix.shape[0]
    mask = np.tril(np.ones((n, n)))
    return matrix * mask


def get_modifying_adjectives(token):
    """Return spacy tokens that are adjectival modifiers of the given token."""
    return [child for child in token.children if child.dep_ == "amod"]

def decaying_first_token_bias_content_focus_L0H0(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    attention = np.zeros((n, n))

    for i in range(n):
        attention[i, i] = 0.3

        if i > 0:
            attention[i, 0] = 0.4

        spacy_indices = alignment[i]
        current_pos = None
        if spacy_indices:
            current_pos = doc[spacy_indices[0]].pos_

        for j in range(i):
            if j == 0:
                continue  # Already handled first token

            distance = i - j
            base_weight = 0.1 * (0.7 ** (distance - 1))

            spacy_j = alignment[j]
            if spacy_j:
                j_pos = doc[spacy_j[0]].pos_
                if j_pos in ['VERB', 'NOUN', 'PROPN', 'ADJ']:
                    base_weight *= 2.0

                if j_pos == 'VERB' and current_pos in ['NOUN', 'PROPN', 'PRON']:
                    base_weight *= 1.5

            token_j = tokens[j].strip()
            if len(token_j) > 2 and token_j.isalpha():
                base_weight *= 1.2

            attention[i, j] = base_weight

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L0H0", attention

def decaying_content_focus_punctuation_coreference_L0H1(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    for i in range(n):
        attention[i, i] = 0.99

        for j in range(i):
            token_i = tokens[i].strip()
            token_j = tokens[j].strip()

            base_weight = 0.001

            if token_j in [',', '.', '!', '?', '"', 'and', 'or', 'but']:
                base_weight *= 5

            if len(alignment[i]) > 0 and len(alignment[j]) > 0:
                spacy_i = alignment[i][0] if alignment[i] else -1
                spacy_j = alignment[j][0] if alignment[j] else -1

                if spacy_i < len(doc) and spacy_j < len(doc) and spacy_i >= 0 and spacy_j >= 0:
                    tok_i = doc[spacy_i]
                    tok_j = doc[spacy_j]

                    if tok_j in tok_i.ancestors or tok_i in tok_j.ancestors:
                        base_weight *= 3
                    elif tok_i.head == tok_j or tok_j.head == tok_i:
                        base_weight *= 2

            if len(alignment[i]) > 0 and len(alignment[j]) > 0:
                spacy_i = alignment[i][0] if alignment[i] else -1
                spacy_j = alignment[j][0] if alignment[j] else -1

                if spacy_i < len(doc) and spacy_j < len(doc) and spacy_i >= 0 and spacy_j >= 0:
                    tok_i = doc[spacy_i]
                    tok_j = doc[spacy_j]

                    if tok_i.pos_ == "PRON" and tok_j.pos_ in ["PROPN", "NOUN", "PRON"]:
                        if tok_i.text.lower() in ["she", "her"] and tok_j.text.lower() in ["she", "her"]:
                            base_weight *= 50
                        elif tok_i.text.lower() in ["he", "him", "his"] and tok_j.text.lower() in ["he", "him", "his"]:
                            base_weight *= 50
                        elif tok_i.text.lower() == "it" and tok_j.pos_ == "NOUN":
                            base_weight *= 30
                        else:
                            base_weight *= 20

                    if tok_i.lemma_ == tok_j.lemma_ and tok_i.lemma_ not in ["be", "have", "do", ".", ",", "?", "!"]:
                        base_weight *= 40

            distance = i - j
            decay_factor = np.exp(-distance * 0.3)

            attention[i, j] = base_weight * decay_factor

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_content_focus_punctuation_coreference_L0H1", attention

def first_token_bias_content_focus_punctuation_L0H7(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    for i in range(n):
        base_self = 0.15
        base_prev = 0.25
        base_first = 0.4 if i < 3 else 0.1
        base_other = 0.02

        attention[i, i] = base_self

        if i > 0:
            first_weight = base_first
            if i == 1:
                first_weight = 0.9  # Very strong for position 1
            elif i == 2:
                first_weight = 0.6  # Strong for position 2
            elif i <= 3:
                first_weight = 0.3
            attention[i, 0] = first_weight

        if i > 0:
            attention[i, i-1] = base_prev

        token_text = tokens[i].strip()

        if token_text == "to" and i > 0:
            attention[i, i-1] = 0.35  # Strong attention to previous
            attention[i, i] = 0.25   # Self attention
            if i > 1:
                attention[i, i-2] = 0.15  # Some attention to i-2

        elif token_text == "and":
            for j in range(max(0, i-5), i):
                if j < i-1:  # Not immediate predecessor
                    attention[i, j] = 0.08

        elif token_text in ["about", "for", "in"] and i > 0:
            attention[i, i-1] = 0.3  # Strong attention to previous

        elif token_text in ["the", "a", "an"]:
            if i > 0:
                attention[i, i-1] = 0.2

        for j in range(i):
            if attention[i, j] == 0:
                dist = i - j
                if dist == 1:
                    continue  # Already handled
                elif dist <= 3:
                    attention[i, j] = base_other * 2
                else:
                    attention[i, j] = base_other

    for i in range(n):
        if tokens[i] in [".", "!", "?", ",", ":", ";"]:
            attention[i, 0] = 0.05
            for j in range(max(0, i-3), i):
                attention[i, j] *= 1.5

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L0H7", attention

def decaying_first_token_bias_content_focus_L0H10(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    content_word_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.pos_ in ['NOUN', 'VERB', 'PROPN'] and not token.is_stop:
                    content_word_tokens.add(i)

    for i in range(n):
        attention[i, 0] = 0.6 if i > 0 else 1.0

        if i > 0:
            attention[i, i] = 0.3

        for j in range(1, i):
            distance = i - j
            if distance == 1:
                attention[i, j] = 0.15
            elif distance <= 3:
                attention[i, j] = 0.08 / distance
            else:
                attention[i, j] = 0.04 / distance

        for j in content_word_tokens:
            if j < i:  # Only attend to previous tokens
                distance = i - j
                if distance > 1:  # Don't double-boost immediate previous token
                    boost = 0.08 / max(1, distance * 0.5)
                    attention[i, j] += boost

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L0H10", attention

def decaying_first_token_bias_content_focus_punctuation_L0H11(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    if n == 1:
        return tokens, np.array([[1.0]])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    for i in range(n):
        weights = np.zeros(i + 1)  # Can only attend to tokens <= i

        first_token_weight = max(0.3, 0.8 - i * 0.05)
        weights[0] = first_token_weight

        if i > 0:
            weights[i] = 0.15

        if i > 0:
            weights[i-1] += 0.12

        for j in range(1, i):
            if j != i-1:  # Already handled previous token
                distance = i - j
                decay_weight = 0.08 * np.exp(-0.3 * distance)
                weights[j] += decay_weight

        if len(alignment[i]) > 0:
            spacy_idx = alignment[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                if spacy_token.pos_ == "ADP" and i > 2:
                    for k in range(max(0, i-3), i):
                        if k < len(alignment) and len(alignment[k]) > 0:
                            k_spacy_idx = alignment[k][0]
                            if k_spacy_idx < len(doc) and doc[k_spacy_idx].pos_ in ["NOUN", "PRON"]:
                                weights[k] += 0.08

                if spacy_token.pos_ == "VERB" and i > 1:
                    for k in range(1, min(i, 4)):
                        if k < len(alignment) and len(alignment[k]) > 0:
                            k_spacy_idx = alignment[k][0]
                            if k_spacy_idx < len(doc) and doc[k_spacy_idx].pos_ in ["NOUN", "PRON"]:
                                weights[k] += 0.05

        if i < len(tokens) and tokens[i] in ['.', '!', '?', ',']:
            for j in range(i):
                weights[j] += 0.03

        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights[0] = 1.0  # Fallback to first token

        attention[i, :len(weights)] = weights

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_punctuation_L0H11", attention

def decaying_first_token_bias_L1H3(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.6
        else:
            attention[i, 0] = 1.0

        if i > 0:
            attention[i, i] = 0.25

        for j in range(1, i):
            distance = i - j
            if distance == 1:
                attention[i, j] = 0.15
            elif distance == 2:
                attention[i, j] = 0.08
            elif distance == 3:
                attention[i, j] = 0.05
            else:
                attention[i, j] = 0.03 * (0.7 ** (distance - 3))

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_L1H3", attention

def first_token_bias_punctuation_stochastic_L1H6(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([[]])

    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.7 + 0.2 * np.exp(-i * 0.3)  # Decay with distance but stay strong
        else:
            attention[i, 0] = 1.0  # Self-attention for first token

        if i > 0:
            attention[i, i] = 0.1 + 0.05 * np.random.random()

        if i > 0:
            attention[i, i-1] = 0.08 + 0.04 * np.random.random()

        for j in range(max(0, i-3), i):
            if j != 0 and j != i and j != i-1:  # Skip first token, self, and previous (already handled)
                distance = i - j
                attention[i, j] = 0.03 * np.exp(-distance * 0.5) + 0.02 * np.random.random()

        for j in range(i):
            if tokens[j] in ['!', '.', '?', ',', '!"', '."', '?"']:
                attention[i, j] += 0.03

        for j in range(i):
            token = tokens[j].strip()
            if token in ['and', 'but', 'because', 'with', 'who', 'that', 'which']:
                attention[i, j] += 0.02

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_punctuation_stochastic_L1H6", attention

def first_token_bias_punctuation_L1H8(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    structural_positions = set()
    for i, token_str in enumerate(tokens):
        if any(c in token_str for c in '.,!?;:'):
            structural_positions.add(i)

        spacy_indices = gpt2_to_spacy[i]
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['CCONJ', 'SCONJ', 'ADP'] or spacy_token.text.lower() in ['and', 'because', 'that', 'to', 'the']:
                    structural_positions.add(i)

    for i in range(n):
        attention[i, 0] = 0.7

        attention[i, i] = 0.15

        for j in structural_positions:
            if j <= i and j != 0:  # Causal mask and not first token (already covered)
                distance = i - j
                weight = max(0.05, 0.2 / (1 + distance * 0.5))
                attention[i, j] += weight

        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double count first token
                distance = i - j
                weight = 0.03 / distance
                attention[i, j] += weight

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_punctuation_L1H8", attention

def decaying_stochastic_L1H10(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        self_weight = 1.0 if i < 2 else 0.15
        attention_matrix[i, i] = self_weight

        if i > 0:
            prev_weight = 0.5 if i < 3 else 0.25
            attention_matrix[i, i-1] = prev_weight

        for j in range(i):
            if j == i:  # self (already handled)
                continue
            elif j == i - 1:  # previous token (already handled)
                continue
            else:
                distance = i - j
                base_weight = 0.2 / (distance ** 0.7)

                if j < 2:
                    base_weight *= 1.5

                if i > 5:  # Later tokens
                    base_weight *= 0.8

                attention_matrix[i, j] = max(0.02, base_weight)

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "decaying_stochastic_L1H10", attention_matrix

def decaying_first_token_bias_content_focus_punctuation_L2H5(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    def is_punctuation(token_str):
        return token_str.strip() in '.,;:!?"()[]{}' or any(c in token_str for c in '.,;:!?"()[]{}')

    for i in range(n):
        token = tokens[i]

        for j in range(i + 1):  # Only attend to previous and current tokens
            if i == 0:
                if j == 0:
                    attention[i, j] = 1.0
            else:
                if i == 1 and j == 0:
                    attention[i, j] = 0.95
                elif i == 1 and j == 1:
                    attention[i, j] = 0.05
                else:
                    if j == 0:
                        if i <= 3:
                            attention[i, j] = 0.6 - 0.1 * (i - 1)
                        else:
                            attention[i, j] = 0.2

                    elif j == i:
                        if is_punctuation(token):
                            attention[i, j] = 0.4  # Punctuation has higher self-attention
                        else:
                            attention[i, j] = 0.1

                    elif j == i - 1:
                        prev_token = tokens[j]
                        if is_punctuation(prev_token):
                            attention[i, j] = 0.5  # High attention to previous punctuation
                        else:
                            attention[i, j] = 0.2

                    elif j == i - 2:
                        attention[i, j] = 0.1

                    else:
                        if is_punctuation(tokens[j]):
                            attention[i, j] = 0.15
                        else:
                            attention[i, j] = 0.05

        if is_punctuation(token):
            for j in range(max(0, i - 3), i):
                if not is_punctuation(tokens[j]):
                    attention[i, j] *= 1.5

        for j in range(i):
            if ',' in tokens[j]:
                distance = i - j
                if distance <= 2:
                    attention[i, j] *= 2.0  # Strong boost for nearby commas
                elif distance <= 5:
                    attention[i, j] *= 1.5  # Moderate boost for medium distance
                else:
                    attention[i, j] *= 1.2  # Weak boost for distant commas

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_punctuation_L2H5", attention

def decaying_first_token_bias_punctuation_L2H6(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    punct_tokens = set()
    newline_tokens = set()
    first_token_idx = 0

    for i, token in enumerate(tokens):
        if token in ['."', '.', '."', '!', '?', ',"', ',']:
            punct_tokens.add(i)
        elif token in ['\n']:
            newline_tokens.add(i)

    for i in range(n):
        token = tokens[i]

        base_attention = np.zeros(i + 1)  # Can only attend to tokens up to position i

        if i > 0:
            base_attention[first_token_idx] = 0.8

        base_attention[i] = 0.15

        for j in range(i):
            if j in punct_tokens:
                base_attention[j] += 0.3
            elif j in newline_tokens:
                base_attention[j] += 0.25

        if token in punct_tokens:
            base_attention[i] = 0.4
            for j in range(i):
                if j in newline_tokens or j in punct_tokens:
                    base_attention[j] += 0.2

        elif token in newline_tokens:
            base_attention[i] = 0.4
            for j in range(i):
                if j in punct_tokens:
                    base_attention[j] += 0.3

        elif i == 0:
            base_attention[i] = 1.0

        else:
            base_attention[first_token_idx] = 0.7

            for j in range(max(0, i - 3), i):
                if j != first_token_idx:
                    base_attention[j] += 0.05

            if i > 0 and (tokens[i-1] in ['."', '.', ',', '\n']):
                base_attention[i-1] += 0.2

        for j in range(i + 1):
            if j != first_token_idx and j != i:
                distance_penalty = max(0, 1.0 - 0.1 * (i - j))
                base_attention[j] *= distance_penalty

        base_attention = np.maximum(base_attention, 0.01)
        attention_matrix[i, :i + 1] = base_attention

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "decaying_first_token_bias_punctuation_L2H6", attention_matrix

def first_token_bias_stochastic_L3H0(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    attention[0, 0] = 1.0

    for i in range(1, n):
        first_token_attention = 0.9 + np.random.uniform(-0.05, 0.05)
        first_token_attention = max(0.85, min(0.99, first_token_attention))

        attention[i, 0] = first_token_attention

        self_attention = np.random.uniform(0.02, 0.1)
        attention[i, i] = self_attention

        current_token = tokens[i].lower().strip()
        repeated_token_bonus = 0.0

        if len(current_token) > 2:  # Only for meaningful tokens
            for j in range(i):
                prev_token = tokens[j].lower().strip()
                if prev_token == current_token and j != 0:  # Don't double-count first token
                    bonus = np.random.uniform(0.15, 0.35)
                    attention[i, j] += bonus
                    repeated_token_bonus += bonus

        remaining_prob = 1.0 - first_token_attention - self_attention - repeated_token_bonus

        if remaining_prob > 0:
            available_positions = []
            for j in range(i):
                if j != 0 and attention[i, j] == 0:  # Skip first token and already assigned positions
                    available_positions.append(j)

            if available_positions:
                weights = np.random.exponential(0.01, len(available_positions))
                weights = weights * (remaining_prob / weights.sum()) if weights.sum() > 0 else weights

                for j, pos in enumerate(available_positions):
                    attention[i, pos] = weights[j]

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_stochastic_L3H0", attention

def first_token_bias_content_focus_punctuation_stochastic_L3H1(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    structural_tokens = set()
    for i, token in enumerate(tokens):
        if token.strip() in {',', '.', ':', ';', '!', '?', 'and', 'but', 'or', 'because', 'when', 'if'}:
            structural_tokens.add(i)
        for spacy_idx in gpt2_to_spacy[i]:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['CCONJ', 'SCONJ']:
                structural_tokens.add(i)

    for i in range(n):
        base_weights = np.zeros(i + 1)  # Only attend to previous tokens + self

        if i > 0:
            base_weights[0] = 0.7

        base_weights[i] = 0.15

        if i > 0:
            base_weights[i-1] = 0.08

        for j in structural_tokens:
            if j <= i:
                base_weights[j] += 0.12

        current_token = tokens[i].strip().lower()

        if current_token in ["'s", "the", "a", "an", "his", "her", "their", "my", "your"]:
            for j in range(max(0, i-3), i):
                other_token = tokens[j].strip().lower()
                if len(other_token) > 2 and other_token.isalpha():
                    base_weights[j] += 0.06

        if i > 0 and current_token in ["to", "of", "in", "at", "on", "for", "with"]:
            for j in range(max(0, i-2), i):
                if j in structural_tokens or tokens[j].strip().lower() in ["the", "a", "an"]:
                    base_weights[j] += 0.05

        if tokens[i].strip() == '.':
            base_weights[i] = 0.25
            for j in structural_tokens:
                if j <= i:
                    base_weights[j] += 0.08

        if current_token in ["the", "a", "an", "your", "his", "her", "their", "my", "this", "that"]:
            for j in range(i + 1, min(n, i + 4)):
                future_token = tokens[j].strip().lower()
                if len(future_token) > 2 and future_token.isalpha() and future_token not in ["and", "the", "but", "for", "with", "from"]:
                    pass

            for j in range(max(0, i-2), i):
                other_token = tokens[j].strip().lower()
                if len(other_token) > 3 and other_token.isalpha() and other_token not in ["and", "the", "but", "for", "with", "from", "said", "want"]:
                    base_weights[j] += 0.15

        if (len(current_token) > 2 and current_token.isalpha() and 
            current_token not in ["and", "the", "but", "for", "with", "from", "said", "want", "come", "back"]):
            for j in range(max(0, i-3), i):
                prev_token = tokens[j].strip().lower()
                if prev_token in ["the", "a", "an", "your", "his", "her", "their", "my", "this", "that"]:
                    base_weights[j] += 0.20

        base_weights += np.random.uniform(0, 0.01, size=len(base_weights))

        base_weights = np.maximum(base_weights, 0.01)
        attention[i, :i+1] = base_weights

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_stochastic_L3H1", attention

def first_token_bias_L3H4(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        attention_matrix[i, 0] = 0.85

        attention_matrix[i, i] = 0.08

        if i > 0:
            attention_matrix[i, i-1] = 0.04

        for j in range(1, i):
            if j != i-1:  # Don't double-count previous token
                attention_matrix[i, j] = 0.01

    if n > 0:
        attention_matrix[0, :] = 0
        attention_matrix[0, 0] = 1.0

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_L3H4", attention_matrix

def first_token_bias_content_focus_L3H5(sentence: str) -> tuple[list[str], np.ndarray]:

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    verb_positions = set()
    for gpt2_idx, spacy_indices in enumerate(alignment):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['VERB', 'AUX']:
                verb_positions.add(gpt2_idx)

    if not verb_positions and n > 1:
        verb_positions.add(1)

    for i in range(n):
        attention[i, 0] = 0.8

        for verb_pos in verb_positions:
            if verb_pos != 0 and verb_pos <= i:  # Causal constraint
                attention[i, verb_pos] = 0.15

        attention[i, i] = 0.05

        for j in range(1, i):
            if j not in verb_positions:  # Don't override verb attention
                if n > 15 and i - j <= 5:  # Recent context in long sentences
                    attention[i, j] = 0.06
                else:
                    attention[i, j] = 0.02

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L3H5", attention

def first_token_bias_content_focus_punctuation_L3H8(sentence: str) -> tuple[list[str], np.ndarray]:

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        if i == 0:
            attention[i, i] = 1.0
            continue

        for j in range(i + 1):
            attention[i, j] = 0.05

        attention[i, 0] += 0.4

        attention[i, i] += 0.2

        if i > 0:
            attention[i, i-1] += 0.3

        token_text = tokens[i].strip()
        if token_text in [',', '.']:
            attention[i, i] += 0.3
            for j in range(max(0, i-3), i):
                if tokens[j].strip() not in [',', '.', 'and', 'or', 'but']:
                    attention[i, j] += 0.2

        if i > 0 and tokens[i-1].strip().lower() in ['with', 'on', 'to', 'of', 'in', 'at', 'by']:
            attention[i, i-1] += 0.4

        if tokens[i].strip().lower() in ['but', 'and', 'or']:
            for j in range(max(0, i-3), i):
                if tokens[j].strip() in [',']:
                    attention[i, j] += 0.4

        if i > 1:
            prev_token = tokens[i-1].strip().lower()
            if prev_token in ['like', 'if', 'with', 'on', 'named', 'said']:
                attention[i, i-1] += 0.4

        if i >= 2:
            if tokens[i-1].strip().lower() == 'said':
                attention[i, i-1] += 0.5
            if tokens[i-1].strip().lower() == 'on' and tokens[i].strip().lower() == 'the':
                attention[i, i-1] += 0.5

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L3H8", attention

def decaying_first_token_bias_content_focus_L4H0(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    verb_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ == "VERB":
                verb_tokens.add(i)

    prep_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ == "ADP":
                prep_tokens.add(i)

    attention = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            distance = i - j
            attention[i, j] = np.exp(-0.3 * distance)

        if n > 0:
            attention[i, 0] += 2.0

        attention[i, i] += 0.5

        for j in range(i):
            if j in verb_tokens:
                distance = i - j
                verb_boost = 3.0 * np.exp(-0.2 * distance)
                attention[i, j] += verb_boost

        if i > 0 and (i-1) in prep_tokens:
            attention[i, i-1] += 2.0

        if i > 1 and (i-2) in prep_tokens:
            attention[i, i-2] += 1.5

        for j in range(i):
            if j in prep_tokens:
                distance = i - j
                if distance <= 5:  # Within reasonable range of prep phrase
                    prep_boost = 4.0 * np.exp(-0.15 * distance)
                    attention[i, j] += prep_boost

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L4H0", attention

def first_token_bias_content_focus_punctuation_coreference_L4H2(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0

        attention[i, i] = 0.3

        spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []

        for j in range(i):
            if j == 0:
                continue  # Already handled first token

            dist = i - j
            if dist == 1:  # Previous token
                attention[i, j] = 0.15
            elif dist <= 3:  # Nearby tokens
                attention[i, j] = 0.1 / dist
            else:  # Distant tokens
                attention[i, j] = 0.02

            if spacy_indices:
                for si in spacy_indices:
                    if si < len(doc):
                        current_spacy = doc[si]

                        spacy_j_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                        for sj in spacy_j_indices:
                            if sj < len(doc):
                                target_spacy = doc[sj]

                                if current_spacy.head == target_spacy or target_spacy.head == current_spacy:
                                    attention[i, j] *= 2.0

                                if current_spacy.pos_ == "VERB" and target_spacy.dep_ in ["nsubj", "nsubjpass"]:
                                    attention[i, j] *= 1.5

            token_j = tokens[j].strip()
            if token_j in ['that', 'she', 'he', 'it', ',', ',"', '"'] and dist > 1:
                spacy_j_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                is_anchor = False

                for sj in spacy_j_indices:
                    if sj < len(doc):
                        target_spacy = doc[sj]
                        if (target_spacy.pos_ in ["PRON", "SCONJ"] or 
                            target_spacy.dep_ in ["nsubj", "nsubjpass", "punct"] or
                            token_j in [',', ',"', '"']):
                            is_anchor = True
                            break

                if is_anchor:
                    attention[i, j] *= 3.0

    for i in range(n):
        token = tokens[i]
        if token.strip() in '.!?':
            attention[i, :] *= 0.3
            attention[i, i] = 0.4
            if i > 0:
                attention[i, 0] = 0.3

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_coreference_L4H2", attention

def first_token_bias_content_focus_coreference_L4H4(sentence: str) -> tuple[list[str], np.ndarray]:

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    if n == 1:
        return tokens, np.array([[1.0]])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.7 + 0.2 * (1.0 / (i + 1))  # Decay with distance but stay high
        else:
            attention[i, 0] = 1.0  # First token attends to itself with max weight

        if i > 0:
            attention[i, i] = 0.08

        if i > 1:
            attention[i, i-1] = 0.05

        if i > 2:
            attention[i, i-2] = 0.03
        if i > 3:
            attention[i, i-3] = 0.02

        if gpt2_to_spacy[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_idx = gpt2_to_spacy[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                if spacy_token.pos_ == 'VERB':
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            spacy_j = gpt2_to_spacy[j][0]
                            if spacy_j < len(doc):
                                spacy_token_j = doc[spacy_j]
                                if (spacy_token_j.dep_ == 'nsubj' or 
                                    spacy_token_j.pos_ == 'PRON' or
                                    spacy_token_j.pos_ == 'PROPN'):
                                    attention[i, j] += 0.04

                if spacy_token.pos_ == 'ADP' and i < n-1:
                    attention[i, i+1] = min(attention[i, i+1] + 0.03, 1.0)

                if spacy_token.pos_ == 'PUNCT':
                    for j in range(max(0, i-5), i):
                        if gpt2_to_spacy[j]:
                            spacy_j = gpt2_to_spacy[j][0]
                            if spacy_j < len(doc):
                                spacy_token_j = doc[spacy_j]
                                if spacy_token_j.pos_ in ['NOUN', 'VERB', 'ADJ']:
                                    attention[i, j] += 0.02

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_coreference_L4H4", attention

def decaying_first_token_bias_content_focus_L4H8(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    content_pos_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    is_content_word = np.zeros(n, dtype=bool)

    for i, spacy_indices in enumerate(alignment):
        if spacy_indices:
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc) and doc[spacy_idx].pos_ in content_pos_tags:
                    is_content_word[i] = True
                    break

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0  # Self-attention for first token

        if i > 0:
            for j in range(1, i + 1):
                distance = i - j
                if distance == 0:  # Self-attention
                    attention[i, j] = 0.05
                elif distance == 1:  # Previous token
                    attention[i, j] = 0.08
                elif distance == 2:  # Two tokens back
                    attention[i, j] = 0.04
                else:  # Further back
                    attention[i, j] = 0.02 * np.exp(-0.3 * (distance - 2))

            for j in range(i):
                if is_content_word[j] and j > 0:  # Don't double-boost first token
                    attention[i, j] *= 1.5

            if i == n - 1:  # Last token
                for j in range(i):
                    if is_content_word[j]:
                        attention[i, j] *= 2.0

    if n > 15:  # Only apply to longer sentences
        for i in range(n):
            if is_content_word[i] and i > 0:
                for j in range(i):
                    if is_content_word[j] and j > 0:
                        distance = i - j
                        if distance > 5:  # Only for distant content words
                            attention[i, j] += 0.06 * np.exp(-0.1 * (distance - 5))

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L4H8", attention

def decaying_first_token_bias_content_focus_L4H9(sentence: str) -> tuple[list[str], np.ndarray]:

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 1:
        return tokens, np.array([[1.0]])

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    attention = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention[i, i] = 1.0
            continue

        attention[i, 0] = 0.4

        attention[i, i-1] = 0.3

        attention[i, i] = 0.1

        spacy_indices = alignment[i]
        current_spacy_tokens = [doc[idx] for idx in spacy_indices if idx < len(doc)]

        if current_spacy_tokens:
            current_token = current_spacy_tokens[0]

            for j in range(i):
                target_spacy_indices = alignment[j]
                target_spacy_tokens = [doc[idx] for idx in target_spacy_indices if idx < len(doc)]

                if target_spacy_tokens:
                    target_token = target_spacy_tokens[0]

                    if current_token.pos_ == 'VERB' and target_token.dep_ in ['nsubj', 'nsubjpass']:
                        attention[i, j] += 0.2

                    if current_token.dep_ in ['dobj', 'pobj'] and target_token.pos_ == 'VERB':
                        attention[i, j] += 0.15

                    if current_token.head == target_token:
                        attention[i, j] += 0.15

                    if current_token.pos_ == 'ADP' and target_token.dep_ == 'pobj' and target_token.head == current_token:
                        attention[i, j] += 0.2

        for j in range(i):
            if j not in [0, i-1]:  # Already handled first token and previous token
                distance = i - j
                decay_factor = max(0.05, 0.15 / (distance + 1))
                attention[i, j] += decay_factor

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L4H9", attention

def first_token_bias_punctuation_L4H10(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.9

        attention[i, i] = 0.1 if i > 0 else 1.0

        punct_indices = []
        for j in range(i + 1):  # Only look at previous tokens (causal)
            if tokens[j] in [',', '.', '!', '?', ';"', '."', '"', "'", ':']: 
                punct_indices.append(j)

        if punct_indices and i > 0:
            for p_idx in punct_indices:
                if p_idx != 0:  # Don't double-count first token punctuation
                    attention[i, p_idx] += 0.15

        for j in range(max(0, i-3), i):
            if j != 0 and j != i:  # Don't double-count first token or self
                if tokens[j].lower().strip() in [' to', ' the', ' a', ' an', ' and', ' or', ' but', ' if', ' when', ' with']:
                    attention[i, j] += 0.08
                else:
                    attention[i, j] += 0.03

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_punctuation_L4H10", attention

def first_token_bias_content_focus_punctuation_L5H0(sentence: str):
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        if i == 0:
            attention[i, 0] = 1.0  # First token attends to itself
        else:
            attention[i, 0] = 0.85  # Other tokens strongly attend to first token

        if i > 0:
            attention[i, i] = 0.08

        for j in range(max(0, i-3), i):
            if j != 0 and j != i:  # Not first token or self
                attention[i, j] = 0.02

        token = tokens[i]
        if token in ['.', ',', '!', '?', '"', "'", ':', ';']:
            attention[i, 0] *= 0.7  # Reduce first-token attention
            if i > 0:
                attention[i, i] *= 1.5  # Increase self-attention
            for j in range(max(0, i-5), i):
                if tokens[j].strip() and tokens[j] not in ['.', ',', '!', '?', '"', "'", ':', ';']:
                    attention[i, j] += 0.05

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L5H0", attention

def first_token_bias_L5H1(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    if n > 0:
        attention_matrix[0, 0] = 1.0

    for i in range(1, n):
        attention_matrix[i, 0] = 0.98  # High attention to first token
        attention_matrix[i, i] = 0.02  # Small self-attention

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_L5H1", attention_matrix

def first_token_bias_content_focus_stochastic_L5H2(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    for i in range(n):
        if i == 0:
            attention[i, i] = 1.0
            continue

        weights = {}

        first_token_weight = 0.7 - 0.1 * min(i / 10.0, 0.3)
        weights[0] = first_token_weight

        if i > 0:
            prev_weight = 0.15 + 0.05 * np.random.random()
            weights[i-1] = weights.get(i-1, 0) + prev_weight

        self_weight = 0.08 + 0.04 * np.random.random()
        weights[i] = weights.get(i, 0) + self_weight

        if alignment[i]:  # If token aligns to spacy tokens
            spacy_idx = alignment[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                if spacy_token.pos_ in ['VERB', 'AUX']:
                    for j in range(min(i, 3)):  # First few tokens
                        weights[j] = weights.get(j, 0) + 0.1

                if spacy_token.pos_ in ['DET', 'PREP', 'CONJ']:
                    for j in range(max(0, i-3), i):
                        if j in weights:
                            weights[j] += 0.05

        for offset in [2, 3]:
            if i >= offset:
                back_weight = 0.03 + 0.02 * np.random.random()
                weights[i-offset] = weights.get(i-offset, 0) + back_weight

        total_weight = sum(weights.values())
        if total_weight > 0:
            for j, w in weights.items():
                attention[i, j] = w / total_weight
        else:
            attention[i, i] = 1.0

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_stochastic_L5H2", attention

def first_token_bias_content_focus_L5H5(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        attention[i, 0] = 0.95

        attention[i, i] = 0.03

        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            current_spacy = doc[spacy_indices[0]]

            for j in range(max(0, i-3), i):
                if j == 0 or j == i:  # Skip first token and self (already handled)
                    continue

                j_spacy_indices = gpt2_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = doc[j_spacy_indices[0]]

                    if j_spacy.pos_ in ['NOUN', 'VERB', 'ADJ'] and current_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        attention[i, j] = 0.01

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L5H5", attention

def decaying_first_token_bias_punctuation_L5H8(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    for i in range(n):
        scores = np.zeros(n)

        if i > 0:
            scores[0] = 0.8

        scores[i] = 0.1

        for j in range(i + 1):
            token_text = tokens[j].strip()

            if '"' in token_text or "'" in token_text:
                scores[j] += 0.3

            elif token_text == ',':
                scores[j] += 0.2

            elif token_text in ['?', '!', '?"', '."']:
                scores[j] += 0.2

        for j in range(i + 1):
            if j < len(alignment) and alignment[j]:
                spacy_indices = alignment[j]
                for spacy_idx in spacy_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ == 'ADP' and tokens[j].strip().lower() in ['from', 'in', 'on', 'at', 'to', 'with']:
                            if i > j:
                                scores[j] += 0.4

        for j in range(i + 1):
            if tokens[j].strip().lower() in ['and', 'or', 'but']:
                if i > j:
                    scores[j] += 0.2

        for j in range(max(0, i-3), i):
            scores[j] += 0.05 * (1.0 - (i-j) * 0.2)

        for j in range(i + 1):
            token_text = tokens[j].strip()
            if token_text in ['.', '\n']:
                scores[j] += 0.1

        scores = np.maximum(scores, 0.01)  # Minimum attention

        attention[i] = scores

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_punctuation_L5H8", attention

def first_token_bias_content_focus_L5H9(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    content_word_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        for spacy_idx in gpt2_to_spacy[i]:
            if spacy_idx < len(doc):
                pos = doc[spacy_idx].pos_
                if pos in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    content_word_mask[i] = True
                    break

    for i in range(n):
        attention[i, 0] = 0.85

        if i == 0:
            attention[i, i] = 1.0
        else:
            remaining = 0.15

            self_weight = 0.04
            attention[i, i] = self_weight
            remaining -= self_weight

            if i > 0:
                prev_weight = 0.03
                attention[i, i-1] += prev_weight
                remaining -= prev_weight

            if remaining > 0:
                available_tokens = list(range(i + 1))  # Causal mask
                available_tokens.remove(0)  # Already handled first token
                if i in available_tokens:
                    available_tokens.remove(i)  # Already handled self
                if i > 0 and (i-1) in available_tokens:
                    available_tokens.remove(i-1)  # Already handled previous

                if available_tokens:
                    weights = np.ones(len(available_tokens))
                    for idx, token_idx in enumerate(available_tokens):
                        if content_word_mask[token_idx]:
                            weights[idx] *= 2.0  # Boost content words

                    weights = weights / weights.sum() * remaining
                    for idx, token_idx in enumerate(available_tokens):
                        attention[i, token_idx] += weights[idx]

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L5H9", attention

def decaying_first_token_bias_content_focus_punctuation_L5H11(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention = np.zeros((n, n))

    for i in range(n):
        first_token_weight = 0.9 if i <= 3 else max(0.6, 0.9 - (i - 3) * 0.05)
        attention[i, 0] = first_token_weight

        self_weight = 0.15 if i == 0 else 0.08
        attention[i, i] = self_weight

        for j in range(max(0, i-3), i):
            if j == 0:
                continue  # Already handled first token
            distance = i - j
            if distance == 1:
                attention[i, j] = 0.04  # Previous token
            elif distance == 2:
                attention[i, j] = 0.02
            else:
                attention[i, j] = 0.01

        token = tokens[i]
        if token in [',', '.', '!', '?', '"']:
            for j in range(max(0, i-5), i):
                if j == 0:
                    continue
                if tokens[j].strip() and not tokens[j] in [',', '.', '!', '?', '"']:
                    attention[i, j] += 0.02

        if token.strip() == 'and':
            attention[i, i] += 0.05

        for j in range(i):
            if tokens[j].strip() == 'and':
                attention[i, j] += 0.03

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_punctuation_L5H11", attention

def decaying_content_focus_L6H1(sentence: str) -> tuple[list[str], np.ndarray]:

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        for j in range(i + 1):  # Causal mask: only attend to j <= i
            if j == 0:  # First token gets very high attention
                attention[i, j] = 0.9
            elif j == i:  # Self-attention gets moderate weight
                attention[i, j] = 0.08
            elif j == i - 1:  # Previous token gets some attention
                attention[i, j] = 0.015
            else:  # Distant tokens get small attention that decays with distance
                distance = i - j
                attention[i, j] = max(0.005, 0.02 / distance)

    for i in range(n):
        token_text = tokens[i].strip().lower()

        if token_text in ['and', 'but', 'or', ',']:
            for j in range(max(0, i-4), i):  # Look back up to 4 tokens
                if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                    spacy_idx = gpt2_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['ADJ', 'VERB', 'NOUN']:
                            distance = i - j
                            if distance == 1:
                                attention[i, j] = 0.6  # Very strong for adjacent
                            elif distance == 2:
                                attention[i, j] = 0.3  # Strong for distance 2
                            else:
                                attention[i, j] = 0.15  # Moderate for further

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_content_focus_L6H1", attention

def first_token_bias_content_focus_punctuation_L6H2(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        base_weights = np.zeros(i + 1)  # Can only attend to tokens <= i

        if i == 0:
            base_weights[0] = 1.0
        else:
            first_token_weight = 0.9 if i <= 3 else max(0.3, 0.8 - 0.1 * i)

            self_weight = 0.03

            recent_weight = 0.05

            remaining = 1.0 - first_token_weight - self_weight - recent_weight

            base_weights[0] = first_token_weight

            base_weights[i] = self_weight

            recent_start = max(1, i - 4)
            recent_positions = list(range(recent_start, i))

            important_positions = []

            current_spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []

            for j in range(1, i):
                token_text = tokens[j].strip()

                j_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []

                is_important = False
                if j_spacy_indices:
                    spacy_token = doc[j_spacy_indices[0]]
                    if spacy_token.pos_ in ['VERB', 'AUX', 'CCONJ'] or token_text in [',', '.', '?', '!', '"']:
                        is_important = True

                if current_spacy_indices and j_spacy_indices:
                    current_spacy_token = doc[current_spacy_indices[0]]
                    j_spacy_token = doc[j_spacy_indices[0]]

                    if j_spacy_token in [current_spacy_token.head] + list(current_spacy_token.ancestors):
                        is_important = True

                if is_important:
                    important_positions.append(j)

            if recent_positions:
                recent_per_pos = recent_weight / len(recent_positions)
                for pos in recent_positions:
                    if pos in important_positions:
                        base_weights[pos] += recent_per_pos * 2  # Boost important tokens
                    else:
                        base_weights[pos] += recent_per_pos * 0.5

            uniform_weight = max(0, remaining) / (i + 1)
            base_weights += uniform_weight

        attention[i, :i+1] = base_weights

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L6H2", attention

def first_token_bias_content_focus_L6H3(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly

        if i > 0:
            attention[i, i] = 0.3

        if i > 1:
            attention[i, i-1] = 0.2

        if i >= n // 2:  # Second half of sentence
            for j in range(1, i):
                attention[i, j] += 0.1

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L6H3", attention

def first_token_bias_content_focus_stochastic_L6H5(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.85

        attention[i, i] = 0.06

        if i > 0:
            attention[i, i-1] = 0.04

        token = tokens[i].strip()

        if token in ['.', '!', '?']:
            for j in range(max(0, i-3), i):
                if tokens[j].strip() not in [',', '.', '!', '?', '"', "'", '(', ')']:
                    attention[i, j] += 0.02

        elif token in [',', ',"', ',"']:
            for j in range(i-1, max(-1, i-4), -1):
                if tokens[j].strip() not in [',', '.', '!', '?', '"', "'", '(', ')']:
                    attention[i, j] += 0.03
                    break

        elif token.startswith('"') and i > 0:
            attention[i, 0] = 0.7

        if i > 0 and gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0] if gpt2_to_spacy[i] else None
            if spacy_idx is not None and spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                if spacy_token.pos_ == 'VERB':
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                if other_token.pos_ in ['NOUN', 'PRON'] and other_token.dep_ in ['nsubj', 'dobj']:
                                    attention[i, j] += 0.03

                elif spacy_token.pos_ == 'ADJ':
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                if other_token.pos_ == 'NOUN' and abs(other_spacy_idx - spacy_idx) <= 2:
                                    attention[i, j] += 0.02

        if i > 0 and gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0] if gpt2_to_spacy[i] else None
            if spacy_idx is not None and spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                if spacy_token.pos_ == 'AUX' or (spacy_token.pos_ == 'VERB' and spacy_token.dep_ == 'aux'):
                    for j in range(max(0, i-6), i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                if (other_token.pos_ == 'VERB' and other_token.dep_ in ['ROOT', 'ccomp', 'xcomp']) or \
                                   (other_token.dep_ in ['dobj', 'attr', 'ccomp']):
                                    attention[i, j] += 0.05

                elif spacy_token.pos_ == 'VERB' and spacy_token.dep_ in ['ROOT', 'ccomp']:
                    for j in range(max(0, i-5), i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                if other_token.dep_ in ['dobj', 'ccomp', 'xcomp', 'nsubj'] or \
                                   (other_token.pos_ == 'VERB' and abs(other_spacy_idx - spacy_idx) <= 3):
                                    attention[i, j] += 0.04

        for j in range(max(0, i-3), i):
            attention[i, j] += np.random.uniform(0.01, 0.025)

    if n > 0:
        attention[0, 0] = 1.0

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_stochastic_L6H5", attention

def decaying_first_token_bias_stochastic_L6H6(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    gpt2_to_spacy = align_gpt2_to_spacy(sentence)
    doc = spacy_parse(sentence)

    for i in range(n):
        if i == 0:
            attention[i, 0] = 1.0
        else:
            attention[i, 0] = 0.85 + 0.1 * np.random.random()

            attention[i, i] = 0.05 + 0.05 * np.random.random()

            if i > 0:
                attention[i, i-1] = 0.02 + 0.03 * np.random.random()

            for j in range(1, min(i, 5)):  # Look back up to 5 tokens
                if i - j > 0:
                    decay_factor = 0.5 ** j
                    attention[i, i-j] += 0.01 * decay_factor * np.random.random()

            if gpt2_to_spacy[i]:  # If this GPT2 token aligns to spacy tokens
                for spacy_idx in gpt2_to_spacy[i]:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]

                        syntactic_targets = []

                        if spacy_token.head != spacy_token:
                            syntactic_targets.append(spacy_token.head)

                        for child in spacy_token.children:
                            if child.dep_ in ["dobj", "pobj", "amod"]:
                                syntactic_targets.append(child)

                        for target in syntactic_targets:
                            target_gpt2_indices = []
                            for gpt2_idx in range(i):  # Only look at previous tokens (causal)
                                if gpt2_to_spacy[gpt2_idx]:
                                    for target_spacy_idx in gpt2_to_spacy[gpt2_idx]:
                                        if target_spacy_idx < len(doc) and doc[target_spacy_idx] == target:
                                            target_gpt2_indices.append(gpt2_idx)

                            for target_idx in target_gpt2_indices:
                                attention[i, target_idx] += 0.03 + 0.02 * np.random.random()

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_stochastic_L6H6", attention

def first_token_bias_L6H9(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention_matrix[i, 0] = 1.0
        else:
            attention_matrix[i, 0] = 0.99  # Strong attention to first token
            attention_matrix[i, i] = 0.01  # Small self-attention

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_L6H9", attention_matrix

def first_token_bias_punctuation_L6H10(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.9
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly

        if i > 0:
            attention[i, i] = 0.04

        remaining_weight = 1.0 - attention[i].sum()

        if remaining_weight > 0 and i > 0:
            accessible_positions = list(range(1, i))  # Exclude position 0 and self

            if accessible_positions:
                weights = np.ones(len(accessible_positions)) * 0.01

                for idx, pos in enumerate(accessible_positions):
                    token = tokens[pos]
                    if token in [',', '.', ';', ':', '!', '?']:
                        weights[idx] *= 1.5
                    elif pos >= i - 3:  # Recent tokens
                        weights[idx] *= 1.2

                if weights.sum() > 0:
                    weights = weights * (remaining_weight / weights.sum())

                for idx, pos in enumerate(accessible_positions):
                    attention[i, pos] = weights[idx]

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_punctuation_L6H10", attention

def decaying_first_token_bias_content_focus_L7H1(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    content_positions = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    content_positions.add(i)

    for i in range(n):
        attention[i, 0] = 0.9

        if i > 0:
            attention[i, i] = 0.05

        for j in range(min(i + 1, n)):
            if j != 0 and j != i and j in content_positions:
                attention[i, j] = 0.02

        if n > 15:  # Only apply to longer sequences where this pattern is more important
            recent_window = min(5, i)
            for j in range(max(0, i - recent_window), i):
                if j != 0:  # Don't interfere with first-token attention
                    distance = i - j
                    extra_weight = 0.03 * (1.0 / distance)
                    attention[i, j] += extra_weight

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L7H1", attention

def first_token_bias_L7H2(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    attention_matrix[0, 0] = 1.0

    for i in range(1, n):
        attention_matrix[i, 0] = 0.99

        attention_matrix[i, i] = 0.01

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_L7H2", attention_matrix

def decaying_first_token_bias_content_focus_L7H3(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    attention = np.zeros((n, n))

    subjects = set()
    main_verbs = set()

    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subjects.add(token.i)
        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"]:
            main_verbs.add(token.i)

    for i in range(n):
        spacy_indices = alignment[i] if i < len(alignment) else []

        for j in range(i + 1):  # Causal mask
            if i == j:
                attention[i, j] = 0.1
            elif j == 0:
                if i <= 3:
                    attention[i, j] = 0.9 - 0.1 * i
                else:
                    attention[i, j] = 0.4
            else:
                base_weight = 0.05

                if spacy_indices:
                    current_spacy = spacy_indices[0]
                    current_token = doc[current_spacy] if current_spacy < len(doc) else None

                    if current_token and current_token.pos_ == "VERB":
                        target_spacy_indices = alignment[j] if j < len(alignment) else []
                        for target_idx in target_spacy_indices:
                            if target_idx in subjects:
                                base_weight += 0.3

                target_spacy_indices = alignment[j] if j < len(alignment) else []
                for target_idx in target_spacy_indices:
                    if target_idx in subjects:
                        base_weight += 0.15
                    if target_idx in main_verbs:
                        base_weight += 0.1

                distance = i - j
                distance_factor = 1.0 / (1.0 + 0.1 * distance)

                if distance <= 2:
                    base_weight += 0.05

                attention[i, j] = base_weight * distance_factor

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L7H3", attention

def first_token_bias_content_focus_L7H5(sentence: str) -> tuple[list[str], np.ndarray]:

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    verb_positions = set()
    for i, spacy_indices in enumerate(alignment):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['VERB', 'AUX']:
                verb_positions.add(i)

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0

        attention[i, i] = 0.1

        for j in verb_positions:
            if j <= i and j != 0:  # Respect causal mask and not first token
                distance = i - j
                if distance <= 3:  # Local context
                    attention[i, j] = 0.2 / (1 + distance * 0.5)

        if i > 1:  # Not first or second token
            attention[i, i-1] = 0.15

        spacy_indices = alignment[i] if i < len(alignment) else []
        is_near_verb = any(spacy_idx < len(doc) and 
                          any(child.pos_ in ['VERB', 'AUX'] or child.head.pos_ in ['VERB', 'AUX']
                              for child in [doc[spacy_idx]] + list(doc[spacy_idx].children))
                          for spacy_idx in spacy_indices)

        if is_near_verb:
            for j in range(max(0, i-3), i):
                if j not in verb_positions and j != 0:
                    attention[i, j] += 0.05

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L7H5", attention

def decaying_first_token_bias_punctuation_L7H7(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention_matrix = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        attention_matrix[i, 0] = 0.85 if i > 0 else 1.0

        if i > 0:
            attention_matrix[i, i] = 0.08

        remaining_weight = 1.0 - attention_matrix[i, :].sum()

        if i > 1 and remaining_weight > 0:
            spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
            current_token_text = tokens[i].strip().lower()

            weights = np.zeros(i)

            for j in range(1, i):
                if j == i:
                    continue

                weight = 0.01  # base weight

                distance = i - j
                weight *= (1.0 / (1 + distance * 0.3))

                prev_token_text = tokens[j].strip().lower()
                if prev_token_text in ['and', 'or', 'but', 'that', 'with', 'to', 'of', 'in']:
                    weight *= 2.0

                if tokens[i] == '.':
                    weight *= 1.5

                if current_token_text in ['and', 'or', 'but', 'then', 'finally']:
                    weight *= 1.2

                weights[j] = weight

            if weights.sum() > 0:
                weights = weights * (remaining_weight / weights.sum())
                attention_matrix[i, 1:i] = weights[1:i]

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "decaying_first_token_bias_punctuation_L7H7", attention_matrix

def first_token_bias_L7H10(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention_matrix[i, i] = 1.0
        else:
            attention_matrix[i, 0] = 0.97

            attention_matrix[i, i] = 0.02

            if i > 0:
                attention_matrix[i, i-1] = 0.01

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_L7H10", attention_matrix

def first_token_bias_stochastic_L7H11(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention_matrix[i, 0] = 1.0
        else:
            attention_matrix[i, 0] = 0.92 + 0.07 * np.random.random()

            if np.random.random() < 0.3:
                attention_matrix[i, i] = 0.01 + 0.02 * np.random.random()

            num_other = min(2, i)
            if num_other > 0:
                other_positions = np.random.choice(range(1, i), size=num_other, replace=False)
                for pos in other_positions:
                    if np.random.random() < 0.4:
                        attention_matrix[i, pos] = 0.005 + 0.02 * np.random.random()

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_stochastic_L7H11", attention_matrix

def first_token_bias_L8H1(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention[i, 0] = 1.0
        else:
            attention[i, 0] = 0.97

            attention[i, i] = 0.02

            for j in range(max(0, i-3), i):
                if j != 0:  # Don't double-count first token
                    attention[i, j] = 0.01 / max(1, i-1)

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L8H1", attention

def first_token_bias_content_focus_L8H3(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    attention_matrix = np.zeros((n, n))

    in_quotes = [False] * n
    quote_depth = 0
    for i, token in enumerate(tokens):
        if '"' in token or '"' in token or '"' in token:
            if quote_depth == 0:
                quote_depth = 1
            else:
                quote_depth = 0
        in_quotes[i] = (quote_depth > 0)

    for i in range(n):
        token = tokens[i]

        for j in range(i + 1):  # Only attend to previous tokens and self
            if j == 0:  # First token gets very high attention
                attention_matrix[i, j] = 10.0
            elif j == i:  # Self-attention
                attention_matrix[i, j] = 0.3
            else:
                distance = i - j
                attention_matrix[i, j] = 0.1 / (1 + 0.3 * distance)

        if in_quotes[i] and i > 5:  # Only apply to longer sentences with quotes
            attention_matrix[i, 0] *= 0.3

            for j in range(max(0, i - 8), i + 1):
                if j != 0 and in_quotes[j]:  # Recent tokens also in quotes
                    attention_matrix[i, j] *= 2.5

        for j in range(i + 1):
            target_token = tokens[j]

            if target_token in ['.', '."', '"', '!"', '?"']:
                attention_matrix[i, j] *= 3.0

            elif target_token.strip().lower() in ['and', 'or', 'but']:
                if i > j + 2:  # Only from tokens that are not immediately following
                    attention_matrix[i, j] *= 2.0

            elif target_token.strip().lower() == 'the' and j > 0:
                attention_matrix[i, j] *= 1.5

        if token.strip() in ['.', '."', '"']:  # Sentence endings attend more to content words
            if alignment[i]:
                spacy_idx = alignment[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    for j in range(i + 1):
                        if j != 0:  # Don't reduce first token attention
                            attention_matrix[i, j] *= 0.8

        elif token.strip().lower() in ['and', 'or']:  # Conjunctions
            for j in range(max(0, i - 3), i):
                if j != 0:  # Don't reduce first token attention
                    attention_matrix[i, j] *= 1.5

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_content_focus_L8H3", attention_matrix

def first_token_bias_content_focus_punctuation_L8H6(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention_matrix[i, 0] = 0.8

        attention_matrix[i, i] = 0.1

        if i > 0:
            attention_matrix[i, i-1] = 0.05

        token = tokens[i]
        if token in ['.', '?', '!', ','] or i == n-1:
            if i > 0:
                attention_matrix[i, 0] = 0.4
            attention_matrix[i, i] = 0.1

            remaining_weight = 0.5
            valid_positions = list(range(i))
            if valid_positions:
                weight_per_pos = remaining_weight / len(valid_positions)
                for j in valid_positions:
                    attention_matrix[i, j] += weight_per_pos

    attention_matrix[0, 0] = 1.0

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_content_focus_punctuation_L8H6", attention_matrix

def first_token_bias_content_focus_L8H8(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])
    if n == 1:
        return tokens, np.array([[1.0]])

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    attention = np.zeros((n, n))

    def is_content_word(spacy_indices):
        if not spacy_indices:
            return False
        for idx in spacy_indices:
            if idx < len(doc):
                token = doc[idx]
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop:
                    return True
        return False

    def is_conjunction(spacy_indices):
        if not spacy_indices:
            return False
        for idx in spacy_indices:
            if idx < len(doc):
                token = doc[idx]
                if token.pos_ == 'CCONJ' or token.text.lower() in ['and', 'or', 'but']:
                    return True
        return False

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.7
        else:
            attention[i, i] = 0.8

        if i > 0:
            attention[i, i] = 0.15

        for j in range(i):
            if j == 0:
                continue  # Already handled first token

            spacy_j = alignment[j] if j < len(alignment) else []
            spacy_i = alignment[i] if i < len(alignment) else []

            if is_content_word(spacy_j):
                attention[i, j] += 0.2

                if is_content_word(spacy_i):
                    attention[i, j] += 0.1

            if is_conjunction(spacy_j):
                attention[i, j] += 0.15

            if i - j <= 3:
                attention[i, j] += 0.05

            if spacy_j and spacy_i:
                j_pos = doc[spacy_j[0]].pos_ if spacy_j[0] < len(doc) else ''
                i_pos = doc[spacy_i[0]].pos_ if spacy_i[0] < len(doc) else ''

                if (j_pos == 'NOUN' and i_pos == 'VERB') or (j_pos == 'VERB' and i_pos == 'NOUN'):
                    attention[i, j] += 0.1

        for j in range(i):
            if attention[i, j] == 0:
                attention[i, j] = 0.01

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L8H8", attention

def first_token_bias_punctuation_L8H11(sentence: str) -> tuple[list[str], np.ndarray]:
    import numpy as np

    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    if n == 1:
        return tokens, np.array([[1.0]])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        attention[i, 0] = 0.85 if i > 0 else 1.0

        if i > 0:
            attention[i, i] = 0.04

        spacy_indices = gpt2_to_spacy[i]

        if spacy_indices and i > 0:
            current_spacy_idx = spacy_indices[0]
            current_token = doc[current_spacy_idx]

            if current_token.head != current_token and current_token.head.i < len(doc):
                head_idx = current_token.head.i
                for j in range(min(i, n)):  # causal mask
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices and head_idx in j_spacy_indices:
                        attention[i, j] += 0.08

            for child in current_token.children:
                if child.i < len(doc):
                    child_idx = child.i
                    for j in range(min(i, n)):  # causal mask
                        j_spacy_indices = gpt2_to_spacy[j]
                        if j_spacy_indices and child_idx in j_spacy_indices:
                            attention[i, j] += 0.06

            for j in range(max(0, i-5), i):
                j_spacy_indices = gpt2_to_spacy[j]
                if j_spacy_indices:
                    j_token = doc[j_spacy_indices[0]]
                    if j_token.pos_ in ['PUNCT'] or j_token.text in [',', '.', '?', '!']:
                        attention[i, j] += 0.03
                    elif j_token.pos_ in ['ADP', 'CONJ', 'CCONJ', 'DET']:
                        attention[i, j] += 0.02

        for j in range(max(0, i-3), i):
            if j != 0:  # first token already handled
                attention[i, j] += 0.01 * (1.0 / (i - j + 1))

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_punctuation_L8H11", attention

def first_token_bias_L9H1(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention[i, i] = 1.0
        else:
            attention[i, 0] = 0.95  # Strong attention to first token
            attention[i, i] = 0.05  # Small self-attention

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L9H1", attention

def decaying_first_token_bias_content_focus_punctuation_L9H2(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    punctuation_indices = set()
    content_word_indices = set()

    for i, token in enumerate(tokens):
        if any(c in token for c in '.,!?;:'):
            punctuation_indices.add(i)

        if gpt2_to_spacy[i]:
            spacy_token = doc[gpt2_to_spacy[i][0]]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not spacy_token.is_stop:
                content_word_indices.add(i)

    for i in range(n):
        for j in range(i + 1):  # Causal mask
            if j == 0:
                attention_matrix[i, j] = 0.8
            elif j == i:
                attention_matrix[i, j] = 0.05
            elif j in punctuation_indices:
                attention_matrix[i, j] = 0.1
            else:
                distance = i - j
                attention_matrix[i, j] = 0.02 / (1 + 0.1 * distance)

        if i in content_word_indices:
            for j in range(i):
                if j in content_word_indices:
                    attention_matrix[i, j] *= 2.0

        for j in punctuation_indices:
            if j < i and i - j <= 3:
                attention_matrix[i, j] *= 3.0

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "decaying_first_token_bias_content_focus_punctuation_L9H2", attention_matrix

def first_token_bias_punctuation_L9H4(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        attention_matrix[i, 0] = 0.9

        token = tokens[i]
        if token.strip() in '.!?;:,':
            attention_matrix[i, i] = 0.15
            attention_matrix[i, 0] = 0.75  # Reduce first-token attention slightly
        else:
            attention_matrix[i, i] = 0.02

        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                attention_matrix[i, j] = 0.03 / distance

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_punctuation_L9H4", attention_matrix

def first_token_bias_L9H5(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        weights = np.zeros(n)

        weights[0] = 0.8

        weights[i] = 0.15

        if i > 0:
            weights[i-1] = 0.1

        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]  # Use first aligned spacy token

            if spacy_token.head != spacy_token:
                head_char_start = spacy_token.head.idx
                char_pos = 0
                for j in range(min(i+1, n)):  # Only look at previous tokens due to causal mask
                    if char_pos <= head_char_start < char_pos + len(tokens[j]):
                        weights[j] += 0.2
                        break
                    char_pos += len(tokens[j])

            for child in spacy_token.children:
                child_char_start = child.idx
                char_pos = 0
                for j in range(min(i+1, n)):  # Only look at previous tokens
                    if char_pos <= child_char_start < char_pos + len(tokens[j]):
                        weights[j] += 0.1
                        break
                    char_pos += len(tokens[j])

        for j in range(i):
            if weights[j] == 0:
                weights[j] = 0.02

        attention[i] = weights

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L9H5", attention

def first_token_bias_L9H6(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    if n == 1:
        return tokens, np.array([[1.0]])

    attention = np.zeros((n, n))

    for i in range(n):
        attention[i, 0] = 0.95  # Very high base attention to first token

    for i in range(1, n):
        attention[i, i] = 0.02

    for i in range(1, n):
        remaining_mass = 1.0 - attention[i, 0] - attention[i, i]
        if i > 1:
            per_token = remaining_mass / (i - 1)
            for j in range(1, i):
                attention[i, j] = per_token

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L9H6", attention

def first_token_bias_stochastic_L9H9(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        attention_matrix[i, 0] = 0.95  # High baseline attention to first token

    for i in range(n):
        attention_matrix[i, i] = 0.04  # Moderate self-attention

    for i in range(n):
        for j in range(1, i):  # Skip first token (already high) and self (already set)
            attention_matrix[i, j] = 0.01 / max(1, i-1)  # Small distributed attention

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_stochastic_L9H9", attention_matrix

def first_token_bias_L9H10(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention[i, 0] = 1.0
        else:
            first_token_weight = max(0.3, 0.95 - 0.1 * i)
            attention[i, 0] = first_token_weight

            if i > 0:
                adjacent_weight = max(0.1, 0.4 - 0.05 * i)
                attention[i, i-1] = adjacent_weight

            self_weight = 0.05 + 0.02 * min(i, 5)
            attention[i, i] = self_weight

            if i <= 5:
                for j in range(1, min(4, i)):
                    if j != i-1:  # Don't double-count adjacent
                        attention[i, j] = max(0.02, 0.15 - 0.02 * (i + j))

            for j in range(1, i-1):
                if j not in [0, i-1] and j not in range(1, min(4, i)):
                    distance = i - j
                    attention[i, j] = max(0.01, 0.08 / (1 + 0.3 * distance))

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L9H10", attention

def first_token_bias_punctuation_L9H11(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            attention_matrix[i, 0] = 1.0
        else:
            attention_matrix[i, 0] = 0.97  # Very high attention to first token
            attention_matrix[i, i] = 0.025  # Small self-attention

            for j in range(max(0, i-2), i):
                if j != 0:  # Don't double-count first token
                    attention_matrix[i, j] = 0.005 / max(1, i-1)

            if n > 15:
                for j in range(i):
                    token = tokens[j]
                    if '"' in token or "'" in token or token in [',', '.', '?', '!']:
                        if j != 0:  # Don't modify first token attention
                            attention_matrix[i, j] += 0.003

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_punctuation_L9H11", attention_matrix

def first_token_bias_content_focus_punctuation_L10H0(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    noun_positions = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['NOUN', 'PROPN']:
                noun_positions.add(i)

    important_positions = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token_spacy = doc[spacy_idx]
                if (token_spacy.pos_ == 'PROPN' or 
                    (token_spacy.pos_ == 'VERB' and token_spacy.dep_ in ['ROOT', 'conj']) or
                    (token_spacy.pos_ == 'NOUN' and len(token_spacy.text) > 3)):
                    important_positions.add(i)

    for i in range(n):
        attention[i, 0] = 0.95

        attention[i, i] = 0.03

        if i > 0:
            for noun_pos in noun_positions:
                if noun_pos <= i and noun_pos != 0:  # Can only attend to previous tokens, not first
                    distance = i - noun_pos
                    if distance <= 3:  # Only nearby nouns
                        weight = 0.02 / (1 + distance * 0.5)
                        attention[i, noun_pos] += weight

        if i > 0:
            for imp_pos in important_positions:
                if imp_pos < i and imp_pos != 0:  # Can only attend to previous tokens, not first
                    distance = i - imp_pos
                    if distance <= 8:
                        weight = 0.08 / (1 + distance * 0.3)
                        attention[i, imp_pos] += weight

        token_text = tokens[i].strip()
        if token_text in ['.', ',', '!', '?']:
            attention[i, 0] = 0.85
            for noun_pos in noun_positions:
                if noun_pos < i:
                    attention[i, noun_pos] += 0.05

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L10H0", attention

def first_token_bias_content_focus_punctuation_L10H1(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    for i in range(n):
        attention[i, 0] = 0.9  # Very high base attention to first token

    for i in range(n):
        attention[i, i] = 0.05  # Moderate self-attention

    for i in range(1, n):
        if i > 0:
            attention[i, i-1] += 0.02
        if i > 1:
            attention[i, i-2] += 0.01

    for i in range(n):
        token = tokens[i]
        if token in ['.', '!', '?', ',', ';']:
            attention[i, 0] *= 0.7
            for j in range(max(0, i-3), i):
                if tokens[j] not in ['.', '!', '?', ',', ';', ' ', "'s", "'t"]:
                    attention[i, j] += 0.1

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L10H1", attention

def first_token_bias_L10H2(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention_matrix[i, 0] = 0.9

        attention_matrix[i, i] = 0.05

        for j in range(1, i):
            if j != 0:  # Already set first token attention
                attention_matrix[i, j] = 0.01

    attention_matrix[0, 0] = 1.0

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "first_token_bias_L10H2", attention_matrix

def first_token_bias_L10H4(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.7 + 0.2 * np.exp(-i * 0.1)  # Decay slightly with distance
        else:
            attention[i, 0] = 1.0

        attention[i, i] += 0.1

        if alignment[i]:  # If this GPT2 token aligns to spacy tokens
            for spacy_idx in alignment[i]:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]

                    syntactic_targets = []

                    if spacy_token.head != spacy_token:
                        syntactic_targets.append(spacy_token.head)

                    for child in spacy_token.children:
                        syntactic_targets.append(child)

                    for target in syntactic_targets:
                        target_idx = target.i
                        if target_idx < len(alignment):
                            for gpt2_idx in range(n):
                                if gpt2_idx <= i and alignment[gpt2_idx] and target_idx in alignment[gpt2_idx]:
                                    attention[i, gpt2_idx] += 0.15

        for j in range(max(0, i-3), i):
            attention[i, j] += 0.05 * (1.0 - (i - j) * 0.1)

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L10H4", attention

def first_token_bias_content_focus_L10H5(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        if i <= 3:
            attention[i, 0] = 0.8 - 0.15 * i
        else:
            attention[i, 0] = 0.1

        attention[i, i] = 0.05

        spacy_indices = gpt2_to_spacy[i]
        current_spacy_token = doc[spacy_indices[0]] if spacy_indices else None

        if current_spacy_token:

            if current_spacy_token.pos_ == "VERB":
                for j in range(max(0, i-5), i):
                    spacy_j = gpt2_to_spacy[j]
                    if spacy_j:
                        spacy_token_j = doc[spacy_j[0]]
                        if spacy_token_j.pos_ in ["NOUN", "PRON"]:
                            attention[i, j] += 0.15

            if current_spacy_token.pos_ == "NOUN":
                for j in range(max(0, i-3), i):
                    spacy_j = gpt2_to_spacy[j]
                    if spacy_j:
                        spacy_token_j = doc[spacy_j[0]]
                        if spacy_token_j.pos_ in ["ADJ", "DET"]:
                            attention[i, j] += 0.1
                        if spacy_token_j.pos_ == "ADP":
                            attention[i, j] += 0.05

            if current_spacy_token.pos_ == "CCONJ" or tokens[i].strip() in ["and", "but", "or"]:
                for j in range(i):
                    attention[i, j] += 0.02

            for j in range(i):
                spacy_j = gpt2_to_spacy[j]
                if spacy_j:
                    spacy_token_j = doc[spacy_j[0]]
                    if spacy_token_j.pos_ == "CCONJ" or tokens[j].strip() in ["and", "but", "or"]:
                        attention[i, j] += 0.08

        token_text = tokens[i].strip().lower()

        if token_text in [".", ",", "!", "?"]:
            for j in range(max(0, i-5), i):
                attention[i, j] += 0.02

        if token_text in ["to", "with", "of", "in", "on", "the", "a", "an"]:
            for j in range(max(0, i-3), i+1):
                if j < n:
                    j_spacy = gpt2_to_spacy[j]
                    if j_spacy:
                        j_token = doc[j_spacy[0]]
                        if j_token.pos_ == "NOUN":
                            attention[i, j] += 0.05

        for j in range(max(0, i-2), i):
            attention[i, j] += 0.03

        for j in range(i):
            attention[i, j] += 0.01

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L10H5", attention

def first_token_bias_L10H6(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    attention[0, 0] = 1.0

    for i in range(1, n):
        attention[i, 0] = 0.93

        attention[i, i] = 0.07

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L10H6", attention

def first_token_bias_punctuation_L10H8(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        attention[i, 0] = 0.95

        attention[i, i] = 0.03

        for j in range(i + 1):
            if j != 0 and j != i:  # Skip first token and self (already set)
                token = tokens[j].strip()
                if token in [',', '.', 'and', 'or']:
                    attention[i, j] = 0.015
                else:
                    attention[i, j] = 0.005

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_punctuation_L10H8", attention

def first_token_bias_content_focus_L10H10(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    important_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        if spacy_indices:
            for s_idx in spacy_indices:
                if s_idx < len(doc):
                    token = doc[s_idx]
                    if (token.pos_ in ['PROPN', 'NOUN'] or 
                        (token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp']) or
                        token.ent_type_ in ['PERSON', 'ORG', 'GPE']):
                        important_tokens.add(i)

    for i in range(n):
        attention[i, 0] = 0.95

        attention[i, i] = 0.02

        for j in important_tokens:
            if j <= i and j != 0:  # Respect causal mask, don't double-count first token
                attention[i, j] += 0.02

        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                attention[i, j] += 0.005

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_L10H10", attention

def decaying_first_token_bias_L10H11(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    for i in range(n):
        attention[i, 0] = 0.8

        attention[i, i] = 0.15

        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                attention[i, j] = 0.05 / distance

        if alignment[i]:  # If this GPT2 token aligns to spacy tokens
            spacy_idx = alignment[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    for k in range(i):
                        if alignment[k] and head_idx in [doc[idx].i for idx in alignment[k] if idx < len(doc)]:
                            attention[i, k] += 0.1

                for child in spacy_token.children:
                    if child.dep_ in ["amod", "compound"]:
                        child_idx = child.i
                        for k in range(i+1, n):
                            if k < len(alignment) and alignment[k] and child_idx in [doc[idx].i for idx in alignment[k] if idx < len(doc)]:
                                attention[k, i] += 0.1

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_L10H11", attention

def decaying_first_token_bias_content_focus_punctuation_L11H1(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention_matrix = np.zeros((n, n))

    for i in range(n):
        attention_matrix[i, 0] = 0.9

        for j in range(1, min(4, i + 1)):
            if j < n:
                attention_matrix[i, j] = max(0.1 - 0.02 * j, 0.02)

        attention_matrix[i, i] = 0.05

        for j in range(4, i):
            if j < n:
                decay = max(0.01, 0.05 * np.exp(-0.3 * (j - 3)))
                attention_matrix[i, j] = decay

        if i == n - 1:  # Last token (often punctuation)
            mid_start = max(1, n // 3)
            mid_end = min(n - 1, 2 * n // 3)
            for j in range(mid_start, mid_end):
                attention_matrix[i, j] *= 2

            if n > 5:
                attention_matrix[i, min(5, n-1)] *= 3

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "decaying_first_token_bias_content_focus_punctuation_L11H1", attention_matrix

def first_token_bias_content_focus_punctuation_L11H2(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    important_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if (token.pos_ in ['PROPN', 'NOUN'] or 
                    token.ent_type_ != '' or
                    token.dep_ in ['nsubj', 'dobj', 'pobj']):
                    important_tokens.add(i)

    for i in range(n):
        if i == 0:
            attention[i, 0] = 1.0  # First token attends to itself completely
        else:
            attention[i, 0] = 0.8  # Strong attention to first token

            attention[i, i] = 0.1

            for j in range(i):  # Only previous tokens due to causal mask
                if j in important_tokens and j != 0:
                    attention[i, j] = 0.05
                elif j != 0:  # Small residual attention to other tokens
                    attention[i, j] = 0.01

            current_token = tokens[i]
            if current_token in ['.', ',', '!', '?', ';', ':']:
                attention[i, i] = 0.15
                attention[i, 0] = 0.7  # Reduce first-token attention slightly

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L11H2", attention

def decaying_first_token_bias_content_focus_L11H3(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        attention[i, i] = 0.1

        if i <= 3:
            attention[i, 0] = 0.8 - 0.15 * i
        else:
            attention[i, 0] = 0.05

        spacy_indices = gpt2_to_spacy[i]

        for j in range(i):  # Only attend to previous tokens (causal)
            if j == 0:
                continue  # Already handled first token

            dist = i - j
            base_weight = 0.1 / (1 + 0.3 * dist)

            if dist == 1:
                base_weight *= 2.0

            if spacy_indices and gpt2_to_spacy[j]:
                spacy_i = spacy_indices[0]
                spacy_j = gpt2_to_spacy[j][0]

                if spacy_i < len(doc) and spacy_j < len(doc):
                    tok_i = doc[spacy_i]
                    tok_j = doc[spacy_j]

                    if tok_i.pos_ == 'VERB' and tok_j.pos_ in ['NOUN', 'PRON'] and tok_j.dep_ in ['nsubj', 'nsubjpass']:
                        base_weight *= 3.0

                    elif tok_i.pos_ == 'ADJ' and tok_j.pos_ == 'NOUN' and tok_j.head == tok_i:
                        base_weight *= 2.5
                    elif tok_i.pos_ == 'NOUN' and tok_j.pos_ == 'ADJ' and tok_i.head == tok_j:
                        base_weight *= 2.5

                    elif tok_i.pos_ == 'ADP' and tok_j.dep_ == 'pobj':
                        base_weight *= 2.0

                    elif tok_i.pos_ == 'CCONJ' and j > 0:
                        base_weight *= 1.5

                    elif tokens[i] in ['.', ',', '!', '?']:
                        if tok_j.pos_ == 'VERB' or j == i - 1:
                            base_weight *= 2.0

            if tokens[j] in [',', '.'] and dist <= 3:
                base_weight *= 1.5

            attention[i, j] += base_weight

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_L11H3", attention

def first_token_bias_punctuation_L11H4(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.8  # High base weight for first token
        else:
            attention[i, 0] = 1.0  # Self-attention for first token

        if i > 0:
            attention[i, i] = 0.1

        for j in range(max(0, i-3), i):  # Look at up to 3 previous tokens
            if j > 0:  # Don't double-count first token
                distance = i - j
                weight = 0.05 / distance  # Decaying weight based on distance
                attention[i, j] += weight

        for j in range(i):
            token = tokens[j]
            if token in [',', '.', '!', '?', ';', ':']:
                attention[i, j] += 0.03

        if i > 1:
            attention[i, i-1] += 0.02

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_punctuation_L11H4", attention

def first_token_bias_content_focus_punctuation_L11H5(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0

        if i > 0:
            attention[i, i] = 0.1

        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                if distance == 1:
                    attention[i, j] = 0.05
                elif distance == 2:
                    attention[i, j] = 0.03
                else:
                    attention[i, j] = 0.02

        token = tokens[i]
        if token in [',', '.', '!', '?']:
            for j in range(max(0, i-5), i):
                if j != 0 and tokens[j].strip() and tokens[j] not in [',', '.', '!', '?']:
                    attention[i, j] += 0.02

        if token.lower().strip() == 'and':
            for j in range(max(0, i-3), i):
                if j != 0:
                    attention[i, j] += 0.01

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_content_focus_punctuation_L11H5", attention

def first_token_bias_L11H6(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])

    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    proper_noun_indices = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ == 'PROPN':
                proper_noun_indices.add(i)

    for i in range(n):
        attention[i, 0] = 0.95

        attention[i, i] = 0.02

        for prop_idx in proper_noun_indices:
            if prop_idx <= i and prop_idx != 0:  # causal and not first token
                attention[i, prop_idx] = 0.08

        for j in range(max(0, i-2), i):
            if j != 0 and j not in proper_noun_indices:  # not first token or proper noun
                attention[i, j] = 0.01

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "first_token_bias_L11H6", attention

def decaying_first_token_bias_content_focus_punctuation_stochastic_L11H7(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])
    if n == 1:
        return tokens, np.array([[1.0]])

    doc = spacy_parse(sentence)
    alignment = align_gpt2_to_spacy(sentence)

    attention = np.zeros((n, n))

    for i in range(n):
        token = tokens[i]

        if i > 0:
            attention[i, 0] = 0.85 + 0.1 * np.random.random()

        if token.strip() in [',', '.', '!', '?']:
            attention[i, i] = 0.04 + 0.03 * np.random.random()
        else:
            attention[i, i] = 0.08 + 0.05 * np.random.random()

        for j in range(max(0, i-5), i):
            if j == 0:
                continue  # Already handled first token
            distance = i - j
            base_weight = 0.15 * np.exp(-0.5 * (distance - 1))

            curr_token = tokens[i].strip().lower()
            prev_token = tokens[j].strip().lower()

            if curr_token in [',', '.'] and prev_token not in [',', '.']:
                base_weight *= 1.5

            base_weight *= (0.8 + 0.4 * np.random.random())
            attention[i, j] = base_weight

        if alignment[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_idx = alignment[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                if spacy_token.head != spacy_token and spacy_token.head.i < len(doc):
                    for k in range(i):
                        if alignment[k] and spacy_token.head.i in alignment[k]:
                            attention[i, k] += 0.02 + 0.01 * np.random.random()

                for child in spacy_token.children:
                    if child.i < len(doc):
                        for k in range(i):
                            if alignment[k] and child.i in alignment[k]:
                                attention[i, k] += 0.015 + 0.01 * np.random.random()

        if token.strip() in ['.', '!', '?'] and i > 0:
            for j in range(i):
                if j == 0:
                    continue  # Skip first token (already handled)
                if alignment[j]:  # If GPT2 token aligns with spacy tokens
                    for spacy_idx in alignment[j]:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                                distance = i - j
                                boost = max(0.02, 0.06 * np.exp(-0.1 * distance))
                                boost *= (0.8 + 0.4 * np.random.random())
                                attention[i, j] += boost

    attention[0, 0] = 1.0

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_content_focus_punctuation_stochastic_L11H7", attention

def decaying_first_token_bias_L11H9(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 0:
        return tokens, np.array([])
    if n == 1:
        return tokens, np.array([[1.0]])

    attention = np.zeros((n, n))

    attention[0, 0] = 1.0

    for i in range(1, n):
        base_first_attention = 0.95

        decay = min(0.05, i * 0.005)
        first_attention = base_first_attention - decay

        attention[i, 0] = first_attention

        remaining = 1.0 - first_attention

        self_attention = min(0.03, remaining * 0.3)
        attention[i, i] = self_attention
        remaining -= self_attention

        if remaining > 0:
            local_positions = []
            if i >= 1:
                local_positions.append(i - 1)
            if i >= 2:
                local_positions.append(i - 2)

            if local_positions:
                weights = [0.7, 0.3][:len(local_positions)]
                weights = np.array(weights)
                weights = weights * (remaining / weights.sum())

                for j, pos in enumerate(local_positions):
                    attention[i, pos] = weights[j]

    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)

    return "decaying_first_token_bias_L11H9", attention

def decaying_first_token_bias_content_focus_punctuation_L11H10(sentence: str) -> tuple[list[str], np.ndarray]:
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)

    if n == 1:
        return tokens, np.array([[1.0]])

    attention_matrix = np.zeros((n, n))

    doc = spacy_parse(sentence)
    gpt2_to_spacy = align_gpt2_to_spacy(sentence)

    for i in range(n):
        weights = np.zeros(i + 1)  # Can only attend to positions 0 to i

        if i > 0:
            weights[0] = 0.7

        weights[i] = 0.15

        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                weight = 0.1 / distance
                weights[j] = weight

        token_text = tokens[i].strip()
        if token_text in ['.', '!', '?', ',', ':', ';']:
            weights = np.zeros(i + 1)
            weights[i] = 0.2  # Self attention for punctuation

            for j in range(i):
                token_j = tokens[j].strip()
                if j == 0:
                    weights[j] = 0.3  # Still some first-token bias
                elif token_j and not token_j in [' ', 'the', 'a', 'an', 'and', 'or', 'but']:
                    distance_factor = 1.0 / (i - j) if i > j else 1.0
                    weights[j] = 0.1 * distance_factor

        if "'" in tokens[i]:  # Contractions like "'t", "'s"
            if i > 1:
                weights = np.zeros(i + 1)
                weights[i-1] = 0.4  # Strong attention to preceding word
                weights[i] = 0.2    # Self attention
                weights[0] = 0.3    # First token

                remaining = 0.1
                for j in range(1, i-1):
                    weights[j] = remaining / max(1, i-2)

        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights[i] = 1.0  # Fallback to self-attention

        attention_matrix[i, :i+1] = weights

    attention_matrix[0, 0] = 1.0

    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)

    return "decaying_first_token_bias_content_focus_punctuation_L11H10", attention_matrix