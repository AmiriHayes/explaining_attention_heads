"""
Auto-converted bert_programs.py.
Signature: (sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]
"""
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def spacy_parse(sentence: str):
    return _get_nlp()(sentence)

def _align_to_spacy(sentence: str, tokens: list) -> list:
    doc = _get_nlp()(sentence)
    spans, pos = [], 0
    for t in tokens:
        clean = t.lstrip("##").lstrip()
        span_len = max(len(clean), 1)
        spans.append((pos, pos + span_len))
        pos += span_len
    alignment = []
    for gs, ge in spans:
        overlapping = [si for si, st in enumerate(doc)
                       if st.idx < ge and st.idx + len(st.text) > gs]
        alignment.append(overlapping)
    return alignment

def make_row_stochastic(matrix: np.ndarray) -> np.ndarray:
    matrix = np.clip(matrix, 0, None)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums

def apply_causal_mask(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    return matrix * np.tril(np.ones((n, n)))


def program_L0H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Entity/subject tracking head that attends to important entities and maintains cross-references, with improved subword handling."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    for i in range(n):
        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i]

        # Default self-attention - increase for better baseline
        attention[i, i] = 0.4

        for j in range(n):
            if i == j:
                continue

            target_spacy_indices = token_to_spacy[j]

            # Handle subword tokens that don't align to spacy
            if not spacy_indices and tokens[i].startswith('##'):
                # Subword token - give it moderate attention to nearby content words
                if target_spacy_indices:
                    for spacy_idx in target_spacy_indices:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ['NOUN', 'VERB', 'PROPN']:
                                attention[i, j] += 0.3
                # Also attend to the main word it's part of
                if j > 0 and not tokens[j].startswith('##'):
                    attention[i, j] += 0.4

            # Special handling for subword tokens: boost attention FROM other tokens TO subwords
            if not target_spacy_indices and tokens[j].startswith('##'):
                # Find the root word that this subword belongs to
                root_idx = j - 1
                while root_idx >= 0 and tokens[root_idx].startswith('##'):
                    root_idx -= 1

                if root_idx >= 0:
                    # If current token has spacy alignment and relates to important categories
                    if spacy_indices:
                        for spacy_idx in spacy_indices:
                            if spacy_idx < len(doc):
                                spacy_token = doc[spacy_idx]
                                # Content words should attend to subword continuations
                                if spacy_token.pos_ in ['NOUN', 'VERB', 'PROPN', 'ADJ', 'ADV']:
                                    attention[i, j] += 0.4
                                # Functional words get moderate attention to subwords
                                elif spacy_token.pos_ in ['ADP', 'DET', 'CONJ']:
                                    attention[i, j] += 0.2

                    # Adjacent tokens should attend more to subwords
                    if abs(i - root_idx) <= 2:
                        attention[i, j] += 0.3

            # High attention to entities and subjects
            for spacy_idx in target_spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]

                    # Attention to named entities
                    if spacy_token.ent_type_ in ['PERSON', 'ORG', 'GPE']:
                        attention[i, j] += 0.8

                    # Attention to subjects and important nouns
                    if spacy_token.dep_ in ['nsubj', 'nsubjpass'] or spacy_token.pos_ == 'PROPN':
                        attention[i, j] += 0.6

                    # Attention to pronouns and their potential antecedents
                    if spacy_token.pos_ == 'PRON':
                        attention[i, j] += 0.4

            # Cross-reference attention between semantically similar tokens
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.7:
                attention[i, j] += 0.5

            # Attention to early structural tokens
            if j < 3 and j > 0:  # Skip [CLS] but include early tokens like "upon", "suddenly"
                attention[i, j] += 0.3

            # Special tokens attention patterns
            if tokens[i] == '[SEP]':
                # SEP attends to sentence-final punctuation
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] += 0.8

            if tokens[j] == '[CLS]':
                # Reduce attention to [CLS] from important tokens to fix over-prediction
                for spacy_idx in spacy_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if (spacy_token.pos_ in ['NOUN', 'PROPN', 'PRON'] or 
                            spacy_token.dep_ in ['nsubj', 'dobj']):
                            attention[i, j] += 0.2  # Reduced from 0.4

            # Attention to verbs from their arguments
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ in ['NOUN', 'PRON']:
                        # Look for governing verbs
                        for target_idx in target_spacy_indices:
                            if target_idx < len(doc):
                                target_token = doc[target_idx]
                                if (target_token.pos_ == 'VERB' and 
                                    (spacy_token.head == target_token or target_token.head == spacy_token)):
                                    attention[i, j] += 0.4

            # Boost attention for function words to content words
            if (tokens[i] in [',', 'and', 'but', 'so'] and 
                j > 0 and tokens[j] not in ['[CLS]', '[SEP]']):
                for target_idx in target_spacy_indices:
                    if target_idx < len(doc):
                        target_token = doc[target_idx]
                        if target_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                            attention[i, j] += 0.3

    return "program_L0H0", make_row_stochastic(attention)



def program_L0H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic backward attention with enhanced self-attention for structural tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L0H1", np.array([])

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Identify semantically important positions (verbs, nouns, adjectives)
    important_positions = set()
    for i, spacy_indices in enumerate(token_to_spacy):
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['VERB', 'NOUN', 'PROPN', 'ADJ']:
                important_positions.add(i)

    # Identify function word positions (conjunctions, prepositions, pronouns, etc.)
    function_positions = set()
    for i, spacy_indices in enumerate(token_to_spacy):
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['CCONJ', 'SCONJ', 'ADP', 'PRON', 'DET', 'AUX', 'PART']:
                function_positions.add(i)

    for i in range(n):
        token = tokens[i]

        # Enhanced self-attention baseline - boost for structural tokens
        base_self_attention = 0.1
        if token in ['.', '!', '?']:
            base_self_attention = 0.4  # Strong self-attention for sentence-ending punctuation
        elif token in [',']:
            base_self_attention = 0.3  # Strong self-attention for commas
        elif i in function_positions:
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ in ['CCONJ']:  # Coordinating conjunctions like "and"
                    base_self_attention = 0.3

        attention[i, i] = base_self_attention

        # Special token attention patterns
        if token == '[CLS]':
            attention[i, i] = 0.8
            continue
        elif token == '[SEP]':
            # SEP attends strongly to important earlier tokens
            for j in important_positions:
                if j < i:
                    attention[i, j] = 0.3
            # Enhanced: SEP also attends to function words
            for j in function_positions:
                if j < i:
                    attention[i, j] = 0.25
            continue
        elif token in ['.', '!', '?']:
            # Punctuation attends to important earlier tokens
            for j in important_positions:
                if j < i:
                    attention[i, j] = 0.25
            # Enhanced: Punctuation also attends to function words
            for j in function_positions:
                if j < i:
                    attention[i, j] = 0.2
            continue
        elif token in [',']:
            # Commas attend moderately to recent important tokens
            for j in range(max(0, i-3), i):
                if j in important_positions:
                    attention[i, j] = 0.2
            continue

        # Content tokens attend to earlier important tokens
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Boost attention to semantically important earlier tokens
            for j in important_positions:
                if j < i:
                    # Distance decay
                    distance = i - j
                    decay = max(0.1, 1.0 / (1 + 0.3 * distance))

                    # Similarity boost
                    sim = embedding_similarity(tokens, i, j)
                    sim_boost = max(0, sim) * 0.5

                    attention[i, j] = 0.2 * decay + sim_boost

            # Special case for verbs - they attend strongly to earlier verbs and nouns
            if spacy_token.pos_ == 'VERB':
                for j in range(i):
                    j_spacy = token_to_spacy[j]
                    if j_spacy:
                        j_spacy_token = doc[j_spacy[0]]
                        if j_spacy_token.pos_ in ['VERB', 'NOUN', 'PROPN']:
                            attention[i, j] += 0.3

            # Nouns attend to earlier related nouns and verbs
            elif spacy_token.pos_ in ['NOUN', 'PROPN']:
                for j in range(i):
                    j_spacy = token_to_spacy[j]
                    if j_spacy:
                        j_spacy_token = doc[j_spacy[0]]
                        if j_spacy_token.pos_ in ['VERB', 'NOUN', 'PROPN']:
                            attention[i, j] += 0.2

        # General backward attention with recency bias
        for j in range(i):
            if attention[i, j] == 0:  # Only add if not already set
                distance = i - j
                attention[i, j] = 0.05 / (1 + 0.5 * distance)

    return "program_L0H1", make_row_stochastic(attention)



def program_L0H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sequential-syntactic attention with [SEP] content aggregation for salient tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special handling for [CLS] token
        if tokens[i] == '[CLS]':
            attention[i, i] = 0.8  # Strong self-attention
            continue

        # Special handling for [SEP] token  
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.4  # Moderate self-attention
            # Strong attention to final punctuation
            if i > 0 and tokens[i-1] in '.!?':
                attention[i, i-1] = 0.35
            # Attention to [CLS]
            attention[i, 0] = 0.15

            # NEW: Enhanced [SEP] attention to salient content words
            for j in range(max(0, i-15), i):
                if tokens[j] not in ['[CLS]', '[SEP]'] and not tokens[j].strip() in '.,!?':
                    j_spacy = token_to_spacy[j]
                    if j_spacy:
                        spacy_token = doc[j_spacy[0]]
                        # Strong attention to nouns, proper nouns, and verbs
                        if spacy_token.pos_ in ['NOUN', 'PROPN']:
                            attention[i, j] += 0.12
                        elif spacy_token.pos_ == 'VERB':
                            attention[i, j] += 0.08
                        else:
                            attention[i, j] += 0.03
                    else:
                        attention[i, j] += 0.03
            continue

        # Base self-attention for all other tokens
        attention[i, i] = 0.05

        # Enhanced attention to [CLS] for content words - REDUCED
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            # Much stronger [CLS] attention for nouns and proper nouns - REDUCED
            if spacy_token.pos_ in ['NOUN', 'PROPN']:
                attention[i, 0] = 0.15 + 0.08 * (1 - i / n)  # Reduced from 0.25 + 0.15
            elif spacy_token.pos_ == 'VERB' and i <= n // 2:
                attention[i, 0] = 0.08 + 0.05 * (1 - i / n)  # Reduced from 0.15 + 0.1

        # Regular attention to [CLS] for other tokens (reduced from original)
        if attention[i, 0] == 0:
            attention[i, 0] = 0.01 + 0.02 * (1 - i / n)

        # Sequential/adjacency attention (slightly reduced)
        if i > 0:
            # Strong attention to immediately preceding token
            attention[i, i-1] = 0.06 + 0.04 * np.exp(-0.3 * (i-1))

            # Moderate attention to 2 tokens back
            if i > 1:
                attention[i, i-2] = 0.025

        # Syntactic attention using spacy
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Attention to syntactic head
            if spacy_token.head != spacy_token:
                for j in range(n):
                    j_spacy = token_to_spacy[j]
                    if j_spacy and j_spacy[0] == spacy_token.head.i:
                        attention[i, j] += 0.06

            # Attention to modifiers (for heads)
            for child in spacy_token.children:
                for j in range(n):
                    j_spacy = token_to_spacy[j]
                    if j_spacy and j_spacy[0] == child.i:
                        if child.dep_ in ['amod', 'det', 'compound']:
                            attention[i, j] += 0.04

            # Special patterns for specific POS/dependencies
            if spacy_token.pos_ == 'ADP':  # Prepositions attend to their objects
                for child in spacy_token.children:
                    if child.dep_ == 'pobj':
                        for j in range(n):
                            j_spacy = token_to_spacy[j]
                            if j_spacy and j_spacy[0] == child.i:
                                attention[i, j] += 0.05

        # Semantic similarity boost for nearby tokens
        for j in range(max(0, i-3), min(n, i+4)):
            if j != i and tokens[j] not in ['[CLS]', '[SEP]']:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:
                    attention[i, j] += 0.03 * sim

        # Boost attention to tokens with high embedding similarity
        for j in range(n):
            if j != i and tokens[j] not in ['[CLS]', '[SEP]']:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.8:
                    attention[i, j] += 0.02 * (sim - 0.8)

    # Apply final normalization
    attention = make_row_stochastic(attention)

    return "program_L0H10", attention



def program_L0H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that combines early-token bias, syntactic relationships, special token self-attention, and compound word detection."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Special tokens get strong self-attention
        if tokens[i].strip() in ['[CLS]', '[SEP]']:
            if tokens[i].strip() == '[CLS]':
                attention[i, i] = 0.8
            else:  # [SEP]
                attention[i, i] = 0.3  # Reduce [SEP] self-attention
            # Also attend to other positions with lower weight
            for j in range(n):
                if j != i:
                    attention[i, j] = 0.02
            continue

        # Get spacy token info for current token
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        for j in range(n):
            # Base attention weight
            weight = 0.01

            # Strong self-attention
            if i == j:
                weight += 0.05

            # Special tokens get extra attention
            if tokens[j].strip() in ['[CLS]', '[SEP]']:
                weight += 0.15

            # Reduced early position bias - only for first 2 positions
            if j < min(2, n):
                weight += 0.2 * (2 - j) / 2

            # Attend to content words early in sentence
            target_spacy_indices = token_to_spacy[j]
            if target_spacy_indices:
                target_spacy = doc[target_spacy_indices[0]]
                if target_spacy.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and j < n // 2:
                    weight += 0.2

            # Compound word detection - strong attention between adjacent content words
            if current_spacy and target_spacy_indices and abs(i - j) == 1:
                target_spacy = doc[target_spacy_indices[0]]
                if (current_spacy.pos_ in ['NOUN', 'ADJ'] and target_spacy.pos_ in ['NOUN', 'ADJ']):
                    # Check if they form a compound (high embedding similarity)
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:  # Lower threshold for compound detection
                        weight += 0.8  # Very strong boost for compounds

            # Syntactic relationships
            if current_spacy and target_spacy_indices:
                target_spacy = doc[target_spacy_indices[0]]

                # Auxiliary verbs attend strongly to main verbs
                if current_spacy.pos_ == 'AUX' and target_spacy.pos_ == 'VERB':
                    if target_spacy.dep_ in ['ROOT', 'ccomp', 'xcomp'] or current_spacy.head == target_spacy:
                        weight += 0.5

                # Main verbs attend to auxiliary verbs
                if current_spacy.pos_ == 'VERB' and target_spacy.pos_ == 'AUX':
                    if current_spacy.dep_ in ['ROOT', 'ccomp', 'xcomp'] or target_spacy.head == current_spacy:
                        weight += 0.4

                # Verbs attend to their subjects
                if current_spacy.pos_ == 'VERB':
                    for child in current_spacy.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            child_tokens = spacy_to_token[child.i]
                            if j in child_tokens:
                                weight += 0.15

                # Tokens attend to their syntactic heads
                if target_spacy == current_spacy.head:
                    weight += 0.1

                # Related tokens via dependency
                if current_spacy.head == target_spacy or target_spacy.head == current_spacy:
                    weight += 0.08

            # Recency bias - recent tokens get more attention
            if j < i and i - j <= 3:
                weight += 0.1 * (4 - (i - j)) / 4

            # Semantic similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.5:
                weight += 0.1 * sim

            attention[i, j] = weight

    return "program_L0H11", make_row_stochastic(attention)



def program_L0H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Previous-token attention with [CLS] bias and punctuation adjacency patterns."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    for i in range(n):
        token = tokens[i]

        # Special tokens get uniform attention to [CLS]
        if token in ['[CLS]', '[SEP]']:
            if i == 0:  # [CLS] attends to itself
                attention[i, 0] = 1.0
            else:  # [SEP] attends strongly to final punctuation if present
                if i > 0 and tokens[i-1] in ['.', '!', '?']:
                    attention[i, i-1] = 0.6
                    attention[i, 0] = 0.4
                else:
                    attention[i, 0] = 1.0
            continue

        # Punctuation attends strongly to adjacent content
        if token in ['.', '!', '?', ',', ';', ':']:
            if i > 0:
                attention[i, i-1] = 0.7  # Previous token
                attention[i, 0] = 0.3    # [CLS]
            else:
                attention[i, 0] = 1.0
            continue

        # Quotes attend to adjacent content
        if token in ['"', "'", '"', '"', "'", "'"]:
            if i > 0 and tokens[i-1] not in ['"', "'", '"', '"', "'", "'"]:
                attention[i, i-1] = 0.8  # Previous content
                attention[i, 0] = 0.2    # [CLS]
            elif i < n-1:
                attention[i, i+1] = 0.6  # Next token
                attention[i, 0] = 0.4    # [CLS]
            else:
                attention[i, 0] = 1.0
            continue

        # Regular tokens: strong previous-token bias with [CLS] attention
        base_prev_weight = 0.6
        base_cls_weight = 0.3

        # Adjust based on position
        if i == 1:  # First real token after [CLS]
            attention[i, 0] = 0.9  # Very strong [CLS] attention
            attention[i, i] = 0.1  # Small self-attention
        elif i > 0:
            # Strong previous token attention
            attention[i, i-1] = base_prev_weight

            # [CLS] attention
            attention[i, 0] = base_cls_weight

            # Small self-attention
            attention[i, i] = 0.05

            # Tiny attention to other nearby tokens
            remaining = 1.0 - attention[i].sum()
            if remaining > 0:
                # Distribute remaining attention to nearby tokens
                for j in range(max(0, i-3), min(n, i+2)):
                    if j != i and j != i-1 and j != 0:
                        attention[i, j] = remaining * 0.1
        else:
            attention[i, 0] = 1.0

    return "program_L0H2", make_row_stochastic(attention)



def program_L0H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic head-finding attention with function word focus and positional bias."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L0H3", np.array([])

    attention_matrix = np.zeros((n, n))

    # Parse sentence for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Handle special tokens
        if token_i in ['[CLS]', '[SEP]']:
            if token_i == '[CLS]':
                # [CLS] attends strongly to itself
                attention_matrix[i, i] = 0.7
                # And moderately to sentence end
                attention_matrix[i, n-2:] = 0.15
            else:  # [SEP]
                # [SEP] attends to sentence-final content and punctuation
                for j in range(max(0, n-4), n):
                    if tokens[j] not in ['[CLS]', '[SEP]']:
                        attention_matrix[i, j] = 0.3
                attention_matrix[i, i] = 0.1
            continue

        # Handle punctuation
        if token_i in '.,!?"':
            # Punctuation attends strongly to immediate predecessor
            if i > 0:
                attention_matrix[i, i-1] = 0.6
            # And to nearby content words
            for j in range(max(0, i-3), i):
                if tokens[j] not in '.,!?"' and tokens[j] not in ['[CLS]', '[SEP]']:
                    attention_matrix[i, j] += 0.2
            attention_matrix[i, i] = 0.1
            continue

        # Get spacy information for current token
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            # Fallback: attend to previous token
            if i > 0:
                attention_matrix[i, i-1] = 0.8
            attention_matrix[i, i] = 0.2
            continue

        spacy_token = doc[spacy_indices[0]]

        # Function words (prepositions, auxiliaries, articles) attend to their syntactic heads
        if spacy_token.pos_ in ['ADP', 'AUX', 'DET', 'PART']:
            # Find syntactic head
            head_found = False
            for j in range(n):
                if j == i:
                    continue
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy_token = doc[j_spacy_indices[0]]
                    # Check if j is the syntactic head of i
                    if (spacy_token.head == j_spacy_token or 
                        (spacy_token.head.idx <= j_spacy_token.idx < spacy_token.head.idx + len(spacy_token.head.text))):
                        attention_matrix[i, j] = 0.8
                        head_found = True
                        break

            if not head_found and i > 0:
                # Fallback to immediate predecessor
                attention_matrix[i, i-1] = 0.6

            # Some attention to self
            attention_matrix[i, i] = 0.1

        else:
            # Content words attend to immediate predecessor with high weight
            if i > 0:
                attention_matrix[i, i-1] = 0.5

            # And to syntactically related tokens
            for j in range(n):
                if j == i:
                    continue
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy_token = doc[j_spacy_indices[0]]

                    # Attend to syntactic children/dependents
                    if (j_spacy_token.head == spacy_token or 
                        spacy_token.head == j_spacy_token):
                        attention_matrix[i, j] += 0.2

                    # Attend to tokens with high semantic similarity
                    if embedding_similarity(tokens, i, j) > 0.7:
                        attention_matrix[i, j] += 0.1

            # Some self-attention
            attention_matrix[i, i] = 0.1

            # Distance-based decay for remaining attention
            for j in range(max(0, i-5), min(n, i+3)):
                if j != i and attention_matrix[i, j] == 0:
                    distance = abs(i - j)
                    attention_matrix[i, j] = 0.05 / (1 + distance)

    return "program_L0H3", make_row_stochastic(attention_matrix)



def program_L0H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attends to key narrative markers, [CLS], and early contextual verbs, with strong [SEP] self-attention and content word self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong self-attention for [SEP] with minimal outgoing attention
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.85
            # Very minimal attention to other tokens
            for j in range(n):
                if j != i:
                    attention[i, j] = 0.01
            continue

        # Strong self-attention for [CLS]
        if tokens[i] == '[CLS]':
            attention[i, i] = 0.6
            # Distribute remaining attention to verbs and content words
            for j in range(n):
                if j != i:
                    spacy_indices = alignment[j] if j < len(alignment) else []
                    if spacy_indices:
                        spacy_token = doc[spacy_indices[0]]
                        if spacy_token.pos_ in ['VERB', 'NOUN']:
                            attention[i, j] = 0.1
            continue

        # Add self-attention for content words
        spacy_indices_i = alignment[i] if i < len(alignment) else []
        if spacy_indices_i:
            spacy_token_i = doc[spacy_indices_i[0]]
            if spacy_token_i.pos_ in ['VERB', 'NOUN', 'ADJ', 'PRON']:
                attention[i, i] = 0.08
            elif spacy_token_i.pos_ == 'DET':  # determiners like "the"
                attention[i, i] = 0.06

        # High attention to [CLS] from content words
        if '[CLS]' in tokens:
            cls_idx = tokens.index('[CLS]')
            spacy_indices = alignment[i] if i < len(alignment) else []
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ in ['VERB', 'NOUN', 'ADJ']:
                    attention[i, cls_idx] = 0.15

        # Punctuation self-attention
        if tokens[i] in [',', '.', '!', '?']:
            attention[i, i] = 0.1

        # Find early verbs and narrative markers - but reduce their influence
        early_verb_indices = []
        for j in range(min(5, n)):  # Look at first 5 tokens
            if j < len(alignment) and alignment[j]:
                spacy_token = doc[alignment[j][0]]
                if spacy_token.pos_ == 'VERB' or spacy_token.lemma_ in ['upon', 'once']:
                    early_verb_indices.append(j)

        # Reduced attention to early verbs/markers (was 0.15, now 0.08)
        for verb_idx in early_verb_indices:
            # Skip if this would create too much attention from non-content words
            if spacy_indices_i and doc[spacy_indices_i[0]].pos_ not in ['VERB', 'NOUN', 'ADJ', 'PRON']:
                attention[i, verb_idx] = 0.04
            else:
                attention[i, verb_idx] = 0.08

        # Attention to previous tokens (recency bias)
        for j in range(max(0, i-3), i):
            attention[i, j] = 0.05

        # Additional attention based on similarity to key words
        spacy_indices_i = alignment[i] if i < len(alignment) else []
        if spacy_indices_i:
            spacy_token_i = doc[spacy_indices_i[0]]

            for j in range(n):
                if i != j:
                    # Similarity-based attention
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.5:
                        attention[i, j] += 0.08

                    # Special patterns for certain POS combinations
                    spacy_indices_j = alignment[j] if j < len(alignment) else []
                    if spacy_indices_j:
                        spacy_token_j = doc[spacy_indices_j[0]]

                        # Pronouns attend to verbs
                        if spacy_token_i.pos_ == 'PRON' and spacy_token_j.pos_ == 'VERB':
                            attention[i, j] += 0.1

                        # Verbs attend to subjects and objects
                        if spacy_token_i.pos_ == 'VERB' and spacy_token_j.pos_ in ['PRON', 'NOUN']:
                            attention[i, j] += 0.08

    return "program_L0H4", make_row_stochastic(attention)



def program_L0H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic content word association head with positional bias toward earlier tokens and enhanced [SEP] attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L0H5", np.array([])

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Get spacy alignment for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Helper to check if token is content word
    def is_content_word(token_idx):
        if not token_to_spacy[token_idx]:
            return False
        spacy_indices = token_to_spacy[token_idx]
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                    return True
        return False

    # Helper to check if token is punctuation
    def is_punctuation(token):
        return token.strip() in [',', '.', '!', '?', ';', ':', "'", '"']

    # Helper to check if token is special
    def is_special_token(token):
        return token in ['[CLS]', '[SEP]']

    for i in range(n):
        current_token = tokens[i]

        # Base uniform attention
        for j in range(n):
            attention[i, j] = 0.01

        # Strong self-attention for all tokens
        attention[i, i] = 0.15

        # Special token [CLS] gets attention from many positions
        if '[CLS]' in tokens:
            cls_idx = tokens.index('[CLS]')
            if not is_special_token(current_token):
                attention[i, cls_idx] += 0.08

        # [SEP] attends strongly to final punctuation if present
        if current_token == '[SEP]' and i > 0:
            prev_token = tokens[i-1]
            if is_punctuation(prev_token):
                attention[i, i-1] += 0.25

        # Enhanced [SEP] attention to [CLS] and punctuation throughout sentence
        if current_token == '[SEP]':
            if '[CLS]' in tokens:
                cls_idx = tokens.index('[CLS]')
                attention[i, cls_idx] += 0.15
            # Attend to all punctuation in the sentence
            for j in range(i):
                if is_punctuation(tokens[j]):
                    attention[i, j] += 0.08

        # Punctuation attends to [CLS] and nearby tokens
        if is_punctuation(current_token):
            if '[CLS]' in tokens:
                cls_idx = tokens.index('[CLS]')
                attention[i, cls_idx] += 0.12
            # Attend to previous few tokens
            for offset in range(1, min(4, i+1)):
                attention[i, i-offset] += 0.05

        # Content words attend to semantically related earlier content words
        if is_content_word(i):
            for j in range(i):  # Only look at earlier positions
                if is_content_word(j):
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:  # High semantic similarity
                        attention[i, j] += 0.15 + (sim - 0.3) * 0.3
                    elif sim > 0.1:  # Moderate semantic similarity
                        attention[i, j] += 0.05 + (sim - 0.1) * 0.2

            # Content words also attend to verbs
            for j in range(i):
                if token_to_spacy[j]:
                    for spacy_idx in token_to_spacy[j]:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ == 'VERB':
                                attention[i, j] += 0.08

        # Function words attend to nearby content words
        if not is_content_word(i) and not is_special_token(current_token) and not is_punctuation(current_token):
            # Look for nearby content words
            for offset in range(1, min(4, i+1)):
                j = i - offset
                if is_content_word(j):
                    attention[i, j] += 0.06 / offset  # Decay with distance

    return "program_L0H5", make_row_stochastic(attention)



def program_L0H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic similarity head with verb-centric and self-attention patterns."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L0H6", np.array([])

    attention = np.zeros((n, n))

    # Parse sentence for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens for current position
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        for j in range(n):
            # Base attention weight
            weight = 0.0

            # Strong self-attention for content words
            if i == j:
                if current_spacy and current_spacy.pos_ in ['VERB', 'NOUN', 'ADJ', 'PROPN']:
                    weight += 0.4
                else:
                    weight += 0.15

            # Attention to [CLS] token
            if j == 0 and tokens[j] in ['[CLS]', '<s>']:
                weight += 0.1

            # Semantic similarity component
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.3:  # Threshold for semantic relatedness
                weight += 0.3 * sim

            # Verb-centric attention
            target_spacy_indices = token_to_spacy[j]
            target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

            if target_spacy and target_spacy.pos_ == 'VERB':
                weight += 0.2

            # Distance decay (nearby tokens get slight boost)
            distance = abs(i - j)
            if distance <= 3 and distance > 0:
                weight += 0.05 * (4 - distance) / 4

            # Content word to content word attention
            if (current_spacy and current_spacy.pos_ in ['NOUN', 'PROPN', 'ADJ'] and
                target_spacy and target_spacy.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']):
                weight += 0.1

            # Special token attention patterns
            if tokens[i] in ['[SEP]', '</s>']:
                # SEP tokens attend to various important positions
                if target_spacy and target_spacy.pos_ in ['VERB', 'NOUN', 'PROPN']:
                    weight += 0.08
                elif j == 0:  # [CLS]
                    weight += 0.05

            attention[i, j] = max(weight, 0.01)  # Minimum attention

    return "program_L0H6", make_row_stochastic(attention)



def program_L0H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that focuses on punctuation, syntactic relationships, semantic connections, and pronoun resolution with enhanced self-attention and CLS focus."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        for j in range(n):
            score = 0.0

            # Strong self-attention for [CLS]
            if i == 0 and j == 0:
                score += 0.5

            # Strong attention to [CLS] from other tokens
            if j == 0 and i != 0:
                score += 0.2

            # High attention to comma tokens
            if tokens[j] == ',':
                score += 0.15
                # Extra boost for sentence-final tokens attending to commas
                if i >= n - 3:  # Last few positions
                    score += 0.1

            # Attention to other punctuation
            if tokens[j] in ['.', ';', ':', '!', '?']:
                score += 0.05

            # NEW: Strong pronoun-antecedent relationships
            if token_to_spacy[i] and token_to_spacy[j]:
                spacy_i_idx = token_to_spacy[i][0] if token_to_spacy[i] else None
                spacy_j_idx = token_to_spacy[j][0] if token_to_spacy[j] else None

                if (spacy_i_idx is not None and spacy_j_idx is not None and 
                    spacy_i_idx < len(doc) and spacy_j_idx < len(doc)):
                    spacy_i_tok = doc[spacy_i_idx]
                    spacy_j_tok = doc[spacy_j_idx]

                    # Pronouns attending to their likely antecedents
                    if spacy_i_tok.pos_ == 'PRON' and spacy_j_tok.pos_ in ['NOUN', 'PROPN']:
                        # Strong boost for pronouns attending to nouns, especially if earlier in sentence
                        if spacy_j_idx < spacy_i_idx:
                            score += 0.25
                        else:
                            score += 0.1

                    # Possessive pronouns to nouns
                    if spacy_i_tok.tag_ in ['PRP$', 'POS'] and spacy_j_tok.pos_ in ['NOUN', 'PROPN']:
                        if spacy_j_idx < spacy_i_idx:
                            score += 0.2

            # NEW: Enhanced semantic relationships
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                # Stronger boost for highly related words
                if sim > 0.8:
                    score += 0.15 * sim
                elif sim > 0.6:
                    score += 0.1 * sim
                elif sim > 0.4:
                    score += 0.05 * sim

            # Syntactic relationships via spacy
            if token_to_spacy[i] and token_to_spacy[j]:
                spacy_i = token_to_spacy[i][0]
                spacy_j = token_to_spacy[j][0]

                if spacy_i < len(doc) and spacy_j < len(doc):
                    spacy_tok_i = doc[spacy_i]
                    spacy_tok_j = doc[spacy_j]

                    # Adjective to noun it modifies
                    if spacy_tok_i.pos_ == 'ADJ' and spacy_tok_j.pos_ == 'NOUN':
                        score += 0.08

                    # Dependencies
                    if spacy_tok_j == spacy_tok_i.head:
                        score += 0.06
                    if spacy_tok_i == spacy_tok_j.head:
                        score += 0.04

            # Positional patterns
            # Early tokens get some boost
            if j <= 3:
                score += 0.02

            # Previous token attention
            if j == i - 1:
                score += 0.03

            # Self-attention boost
            if i == j:
                score += 0.04

            # Special handling for certain token types
            # Articles and determiners attend to following nouns
            if token_to_spacy[i] and token_to_spacy[j]:
                spacy_i_idx = token_to_spacy[i][0] if token_to_spacy[i] else None
                spacy_j_idx = token_to_spacy[j][0] if token_to_spacy[j] else None

                if (spacy_i_idx is not None and spacy_j_idx is not None and 
                    spacy_i_idx < len(doc) and spacy_j_idx < len(doc)):
                    spacy_i_tok = doc[spacy_i_idx]
                    spacy_j_tok = doc[spacy_j_idx]

                    if spacy_i_tok.pos_ == 'DET' and spacy_j_tok.pos_ == 'NOUN':
                        score += 0.05

            # TARGETED FIX 1: Enhanced self-attention for content words
            if i == j and token_to_spacy[i]:
                spacy_i_idx = token_to_spacy[i][0] if token_to_spacy[i] else None
                if spacy_i_idx is not None and spacy_i_idx < len(doc):
                    spacy_tok = doc[spacy_i_idx]
                    # Strong self-attention boost for pronouns, nouns, and proper nouns
                    if spacy_tok.pos_ in ['PRON', 'NOUN', 'PROPN']:
                        score += 0.15
                    # Medium boost for other content words
                    elif spacy_tok.pos_ in ['VERB', 'ADJ', 'ADV']:
                        score += 0.08

            # TARGETED FIX 2: Enhanced attention to [CLS] from all tokens
            if j == 0 and i != 0:
                # Additional boost based on token type
                if token_to_spacy[i]:
                    spacy_i_idx = token_to_spacy[i][0] if token_to_spacy[i] else None
                    if spacy_i_idx is not None and spacy_i_idx < len(doc):
                        spacy_tok = doc[spacy_i_idx]
                        # Content words attend more strongly to [CLS]
                        if spacy_tok.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                            score += 0.15
                        elif spacy_tok.pos_ in ['PRON', 'DET', 'ADP']:
                            score += 0.1
                        else:
                            score += 0.05

            attention_matrix[i, j] = score

    # Add small random component to avoid all zeros
    attention_matrix += np.random.uniform(0, 0.01, (n, n))

    return "program_L0H7", make_row_stochastic(attention_matrix)



def program_L0H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation-focused attention with [SEP] attending strongly to semantic content and balanced punctuation attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Get spacy alignment for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Handle special tokens with strong self-attention
        if token_i in ['[CLS]', '[SEP]']:
            attention_matrix[i, i] = 1.0
            continue

        # Handle punctuation with reduced self-attention and stronger content attention
        if token_i.strip() in ['!', '.', ',', '?', ';', ':']:
            attention_matrix[i, i] = 0.6  # Reduced from 0.8
            # Punctuation attends more strongly to nearby content words
            for j in range(max(0, i-3), min(n, i+3)):
                if j != i:
                    token_j = tokens[j]
                    if token_j.strip() not in ['!', '.', ',', '?', ';', ':', '[CLS]', '[SEP]']:
                        # Check if target token is semantically important
                        spacy_indices_j = token_to_spacy[j]
                        content_boost = 1.0
                        if spacy_indices_j:
                            spacy_token_j = doc[spacy_indices_j[0]]
                            if spacy_token_j.pos_ in ['NOUN', 'VERB', 'PROPN']:
                                content_boost = 2.5  # Strong boost for important content
                            elif spacy_token_j.pos_ in ['ADJ', 'ADV']:
                                content_boost = 1.5

                        distance_weight = 1.0 / (abs(i - j) + 1)
                        attention_matrix[i, j] = distance_weight * 0.15 * content_boost
        else:
            # Get POS and dependency info for current token
            spacy_indices_i = token_to_spacy[i]
            is_content_i = False
            if spacy_indices_i:
                spacy_token_i = doc[spacy_indices_i[0]]
                is_content_i = spacy_token_i.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']

            for j in range(n):
                if i == j:
                    # Moderate self-attention for content words
                    if is_content_i:
                        attention_matrix[i, j] = 0.2
                    else:
                        attention_matrix[i, j] = 0.1
                    continue

                token_j = tokens[j]

                # Skip special tokens for content word attention
                if token_j in ['[CLS]', '[SEP]']:
                    if token_j == '[CLS]':
                        attention_matrix[i, j] = 0.02
                    continue

                # Skip punctuation
                if token_j.strip() in ['!', '.', ',', '?', ';', ':']:
                    continue

                # Get linguistic features for target token
                spacy_indices_j = token_to_spacy[j]
                is_content_j = False
                if spacy_indices_j:
                    spacy_token_j = doc[spacy_indices_j[0]]
                    is_content_j = spacy_token_j.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']

                # Base attention with recency bias
                distance = abs(i - j)
                position_weight = 1.0 / (distance + 1)

                # Boost for backwards attention (attending to earlier tokens)
                if j < i:
                    position_weight *= 1.5

                # Semantic similarity component
                similarity = embedding_similarity(tokens, i, j)
                similarity_weight = max(0, similarity) ** 2

                # Content word boost
                content_boost = 1.0
                if is_content_i and is_content_j:
                    content_boost = 2.0
                elif is_content_i or is_content_j:
                    content_boost = 1.3

                # Combine factors
                base_score = position_weight * content_boost
                semantic_score = similarity_weight * content_boost * 0.5

                attention_matrix[i, j] = base_score + semantic_score

    # Enhanced [SEP] handling - stronger attention to semantic content
    for i in range(n):
        if tokens[i] == '[SEP]':
            attention_matrix[i, :] = 0
            attention_matrix[i, 0] = 0.2  # Reduced attention to [CLS]

            # [SEP] strongly attends to sentence-ending punctuation
            for j in range(1, n):
                token_j = tokens[j]
                if token_j.strip() in ['!', '.', '?']:  # Sentence-ending punctuation
                    attention_matrix[i, j] = 0.25
                elif token_j.strip() in [',', ';', ':']:  # Other punctuation
                    attention_matrix[i, j] = 0.1
                elif token_j not in ['[CLS]', '[SEP]']:
                    # Enhanced attention to important content words
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_token = doc[spacy_indices[0]]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'PROPN']:
                            attention_matrix[i, j] = 0.25  # Increased from 0.15
                        elif spacy_token.pos_ in ['ADJ', 'ADV']:
                            attention_matrix[i, j] = 0.12  # Increased from 0.08
                        else:
                            attention_matrix[i, j] = 0.02

    return "program_L0H8", make_row_stochastic(attention_matrix)



def program_L0H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head combining positional structure patterns with enhanced content-to-CLS attention and semantic similarity."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Strong self-attention baseline
        attention[i, i] = 0.3

        # Special token patterns
        if token == '[CLS]':
            attention[i, i] = 0.8
            # Distribute remaining to nearby tokens
            for j in range(min(5, n)):
                if j != i:
                    attention[i, j] = 0.04

        elif token == '[SEP]':
            # SEP attends to punctuation and structure
            attention[i, i] = 0.4
            for j in range(n):
                if tokens[j] in [',', '.', '"', '!', '?']:
                    attention[i, j] += 0.15
                elif tokens[j] == '[CLS]':
                    attention[i, j] += 0.1

            # Enhanced: SEP attends to content words with linguistic features
            if token_to_spacy[i]:
                for j in range(n):
                    if token_to_spacy[j]:
                        spacy_idx = token_to_spacy[j][0]
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                                attention[i, j] += 0.05

        # Punctuation patterns
        elif token in ['.', '!', '?']:
            attention[i, i] = 0.4
            # Enhanced: End punctuation strongly attends to [CLS]
            if n > 0:
                attention[i, 0] += 0.4
            # End punctuation attends to quotes and commas
            for j in range(n):
                if tokens[j] in ['"', ',']:
                    attention[i, j] += 0.2

        elif token in ['"', "'", ',']: 
            attention[i, i] = 0.3
            # Quotes and commas attend to CLS and each other
            for j in range(n):
                if tokens[j] == '[CLS]':
                    attention[i, j] += 0.25
                elif tokens[j] in ['"', "'", ','] and j != i:
                    attention[i, j] += 0.15

        # Content word patterns
        else:
            # Early tokens attend strongly to [CLS]
            if i <= 3 and n > 0:
                attention[i, 0] += 0.3

            # Enhanced content word to [CLS] attention
            if token_to_spacy[i] and n > 0:
                spacy_idx = token_to_spacy[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    # Content words (nouns, verbs, adjectives, adverbs) get strong CLS attention
                    if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                        attention[i, 0] += 0.35
                    # Function words get moderate CLS attention
                    elif spacy_token.pos_ in ['ADP', 'DET', 'PRON', 'AUX']:
                        attention[i, 0] += 0.2

            # Enhanced: Strong semantic similarity attention
            for j in range(n):
                if j != i:
                    sim = embedding_similarity(tokens, i, j)
                    # Boost high-similarity pairs significantly
                    if sim > 0.7:
                        attention[i, j] += sim * 0.4
                    else:
                        attention[i, j] += max(0, sim) * 0.1

            # Positional biases
            # Attend to nearby tokens
            for offset in [-2, -1, 1, 2]:
                j = i + offset
                if 0 <= j < n:
                    attention[i, j] += 0.05

            # Spacy-based linguistic attention
            if token_to_spacy[i]:
                spacy_idx = token_to_spacy[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]

                    # Attend to syntactic head
                    if spacy_token.head != spacy_token:
                        head_text = spacy_token.head.text.lower()
                        for j in range(n):
                            if tokens[j].lower().strip() == head_text:
                                attention[i, j] += 0.08

                    # Content words attend to function words
                    if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        for j in range(n):
                            if token_to_spacy[j]:
                                other_spacy = token_to_spacy[j][0]
                                if other_spacy < len(doc):
                                    if doc[other_spacy].pos_ in ['ADP', 'DET', 'AUX']:
                                        attention[i, j] += 0.03

    return "program_L0H9", make_row_stochastic(attention)



def program_L10H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that focuses on syntactically prominent tokens (verbs, [CLS]) and handles punctuation specially."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)

    # Identify important token types
    verb_indices = set()
    punct_indices = set()
    function_word_indices = set()

    for i, token in enumerate(tokens):
        # Handle special tokens
        if token in ['[CLS]', '[SEP]']:
            continue

        # Check if punctuation
        if token.strip() in '.,;:!?"\'`':
            punct_indices.add(i)
            continue

        # Get spacy alignment
        spacy_indices = alignment[i]
        if not spacy_indices:
            continue

        spacy_token = doc[spacy_indices[0]]

        # Identify verbs
        if spacy_token.pos_ in ['VERB', 'AUX']:
            verb_indices.add(i)

        # Identify function words
        elif spacy_token.pos_ in ['DET', 'ADP', 'CONJ', 'CCONJ', 'SCONJ']:
            function_word_indices.add(i)

    for i in range(n):
        token = tokens[i]

        # Handle [CLS] token - attends to itself strongly
        if token == '[CLS]':
            attention_matrix[i, i] = 1.0
            continue

        # Handle [SEP] token - attends to punctuation and important tokens
        if token == '[SEP]':
            weights = np.zeros(n)

            # Strong attention to final punctuation
            for j in range(n-2, -1, -1):
                if tokens[j].strip() in '.,;:!?':
                    weights[j] = 0.3
                    break

            # Moderate attention to verbs and [CLS]
            for j in verb_indices:
                weights[j] = 0.1
            weights[0] = 0.1  # [CLS]

            # Self attention
            weights[i] = 0.1

            attention_matrix[i] = weights
            continue

        # Handle punctuation - strong self-attention and attention to nearby punctuation
        if i in punct_indices:
            weights = np.zeros(n)
            weights[i] = 0.4  # Strong self-attention

            # Attend to other punctuation
            for j in punct_indices:
                if j != i:
                    weights[j] = 0.1

            # Some attention to nearby tokens
            for j in range(max(0, i-2), min(n, i+3)):
                if j not in punct_indices and j != i:
                    weights[j] = 0.02

            attention_matrix[i] = weights
            continue

        # Handle function words - attend strongly to verbs and [CLS]
        if i in function_word_indices:
            weights = np.zeros(n)

            # Strong attention to verbs
            for j in verb_indices:
                weights[j] = 0.3 / max(1, len(verb_indices))

            # Attention to [CLS]
            weights[0] = 0.2

            # Some self-attention
            weights[i] = 0.1

            # Small attention to other content words
            for j in range(n):
                if j not in verb_indices and j != 0 and j != i and j not in punct_indices:
                    weights[j] = 0.02

            attention_matrix[i] = weights
            continue

        # Handle verbs - moderate self-attention and attention to related tokens
        if i in verb_indices:
            weights = np.zeros(n)
            weights[i] = 0.2  # Self-attention
            weights[0] = 0.1  # [CLS]

            # Attend to other verbs
            for j in verb_indices:
                if j != i:
                    weights[j] = 0.1

            # Attend to nearby content words
            for j in range(max(0, i-3), min(n, i+4)):
                if j != i and j not in verb_indices and j != 0 and j not in punct_indices:
                    weights[j] = 0.03

            attention_matrix[i] = weights
            continue

        # Handle other content words - attend to verbs and important tokens
        weights = np.zeros(n)

        # Strong attention to verbs
        for j in verb_indices:
            weights[j] = 0.25 / max(1, len(verb_indices))

        # Attention to [CLS]
        weights[0] = 0.15

        # Self-attention
        weights[i] = 0.08

        # Attention to similar tokens (semantic similarity)
        for j in range(n):
            if j != i and j not in verb_indices and j != 0:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:
                    weights[j] += 0.05 * sim

        # Small attention to nearby tokens
        for j in range(max(0, i-2), min(n, i+3)):
            if j != i:
                weights[j] += 0.02

        attention_matrix[i] = weights

    return "program_L10H0", make_row_stochastic(attention_matrix)



def program_L10H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation-focused attention head with positional bias and semantic relationships."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy parse for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        for j in range(n):
            weight = 0.01  # base attention

            # Strong self-attention
            if i == j:
                weight += 0.15

            # Check if target token is punctuation
            target_token = tokens[j].strip()
            is_punct = target_token in ",.!?;:"

            if is_punct:
                # Very strong attention to punctuation
                if target_token == ",":
                    weight += 0.4
                elif target_token in ".!":
                    weight += 0.5
                else:
                    weight += 0.3

                # Special tokens attend even more strongly to sentence-final punctuation
                if tokens[i] == "[SEP]" and j == n - 2:  # Usually period is second to last
                    weight += 0.4

            # Positional bias - prefer earlier tokens, especially commas
            if j < i:
                distance = i - j
                if is_punct and target_token == ",":
                    weight += 0.2 / (1 + 0.1 * distance)
                else:
                    weight += 0.05 / (1 + 0.2 * distance)

            # Semantic similarity for content words
            if (i < len(token_to_spacy) and token_to_spacy[i] and 
                j < len(token_to_spacy) and token_to_spacy[j]):

                spacy_i_tokens = [doc[idx] for idx in token_to_spacy[i] if idx < len(doc)]
                spacy_j_tokens = [doc[idx] for idx in token_to_spacy[j] if idx < len(doc)]

                if spacy_i_tokens and spacy_j_tokens:
                    spacy_i = spacy_i_tokens[0]
                    spacy_j = spacy_j_tokens[0]

                    # Pronouns attending to their antecedents
                    if spacy_i.pos_ == "PRON" and spacy_j.pos_ in ["NOUN", "PROPN"]:
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:
                            weight += 0.15 * sim

                    # General semantic similarity
                    if (spacy_i.pos_ in ["NOUN", "VERB", "ADJ"] and 
                        spacy_j.pos_ in ["NOUN", "VERB", "ADJ"]):
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.4:
                            weight += 0.1 * sim

            attention[i, j] = weight

    return "program_L10H1", make_row_stochastic(attention)



def program_L10H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Self-attention head with enhanced positional and syntactic biases, stronger self-attention for content words."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L10H10", np.array([])

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Get spacy alignment for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i].strip()

        # Get spacy features for this token
        spacy_indices = token_to_spacy[i]
        spacy_token = doc[spacy_indices[0]] if spacy_indices else None

        # Special handling for [CLS] token
        if token == '[CLS]':
            attention[i, i] = 1.0
            continue

        # Special handling for [SEP] token - attend to sentence-final and important tokens
        if token == '[SEP]':
            for j in range(n):
                other_token = tokens[j].strip()
                other_spacy_indices = token_to_spacy[j]
                other_spacy_token = doc[other_spacy_indices[0]] if other_spacy_indices else None

                if j == i:  # Self-attention
                    attention[i, j] = 0.3
                elif other_token in ['.', '!', '?']:  # Sentence endings
                    attention[i, j] = 0.15
                elif other_spacy_token and other_spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                    attention[i, j] = 0.08
                else:
                    attention[i, j] = 0.02
            continue

        # Sentence-final punctuation - attend broadly across sentence
        if token in ['.', '!', '?']:
            for j in range(n):
                other_token = tokens[j].strip()
                other_spacy_indices = token_to_spacy[j]
                other_spacy_token = doc[other_spacy_indices[0]] if other_spacy_indices else None

                if other_token == '[CLS]':
                    attention[i, j] = 0.08
                elif other_token in [',', '"', "'"]:
                    attention[i, j] = 0.12
                elif other_spacy_token and other_spacy_token.pos_ in ['NOUN', 'PROPN']:
                    attention[i, j] = 0.08
                elif other_spacy_token and other_spacy_token.pos_ == 'VERB':
                    attention[i, j] = 0.06
                else:
                    attention[i, j] = 0.04
            continue

        # Early position tokens (position 1-3) - high attention to [CLS]
        if i <= 3 and i > 0:
            attention[i, 0] = 0.4  # Strong attention to [CLS]

        # Handle content words with strong self-attention
        if spacy_token and spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
            # Enhanced self-attention and [CLS] attention for content words
            attention[i, i] = 0.85  # Much stronger self-attention
            attention[i, 0] = 0.08  # Reduced [CLS] attention to compensate

            # Semantic similarity attention (reduced to compensate)
            for j in range(n):
                if j != i and j != 0:
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.7:  # High similarity
                        attention[i, j] = 0.03
                    elif sim > 0.4:  # Medium similarity  
                        attention[i, j] = 0.015

        # Handle function words and punctuation
        elif spacy_token:
            if spacy_token.pos_ in ['ADP', 'DET', 'AUX', 'PART']:
                attention[i, i] = 0.3
                attention[i, 0] = 0.2
            elif token in [',', '"', "'"]:
                attention[i, i] = 0.4
                attention[i, 0] = 0.2
            else:
                attention[i, i] = 0.4
                attention[i, 0] = 0.1
        else:
            # Default case
            attention[i, i] = 0.3
            attention[i, 0] = 0.1

        # Add small uniform attention to all other positions (reduced)
        for j in range(n):
            if attention[i, j] == 0:
                attention[i, j] = 0.01

    # Special case: boost [CLS] attention for tokens that strongly attend to it
    for i in range(1, n):  # Skip [CLS] itself
        token = tokens[i].strip()
        spacy_indices = token_to_spacy[i]
        spacy_token = doc[spacy_indices[0]] if spacy_indices else None

        # Boost [CLS] attention for certain patterns observed in failures
        if (spacy_token and spacy_token.pos_ in ['AUX', 'VERB'] and i < n//2) or \
           (token in ['there', 'to'] and i < 10):
            attention[i, 0] *= 3.0  # Significant boost to [CLS] attention

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L10H10", attention



def program_L10H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Pronoun and entity reference head - connects pronouns to entities and punctuation to key referents."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L10H11", np.array([]).reshape(0, 0)

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Identify pronouns and key entities
    is_pronoun = [False] * n
    is_entity = [False] * n
    is_punctuation = [False] * n
    is_special = [False] * n
    is_verb = [False] * n

    for i, token in enumerate(tokens):
        # Check for special tokens
        if token in ['[CLS]', '[SEP]']:
            is_special[i] = True
        # Check for punctuation
        elif token.strip() in '.,!?;:':
            is_punctuation[i] = True
        else:
            # Get spacy features
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ == 'PRON':
                    is_pronoun[i] = True
                if spacy_token.ent_type_ or spacy_token.pos_ in ['NOUN', 'PROPN']:
                    is_entity[i] = True
                if spacy_token.pos_ == 'VERB':
                    is_verb[i] = True

    for i in range(n):
        # Self-attention baseline
        attention[i, i] = 0.1

        if is_special[i]:
            # Special tokens attend strongly to pronouns, entities, and punctuation
            for j in range(n):
                if is_pronoun[j] or is_entity[j]:
                    attention[i, j] += 0.3
                elif is_punctuation[j]:
                    attention[i, j] += 0.2
            # Special tokens also attend to themselves and each other
            for j in range(n):
                if is_special[j]:
                    attention[i, j] += 0.3

        elif is_punctuation[i]:
            # Punctuation attends very strongly to pronouns and entities
            for j in range(n):
                if is_pronoun[j]:
                    attention[i, j] += 0.4
                elif is_entity[j]:
                    attention[i, j] += 0.2
            # Strong self-attention for punctuation
            attention[i, i] += 0.3
            # Moderate attention to special tokens
            for j in range(n):
                if is_special[j]:
                    attention[i, j] += 0.2

        elif is_pronoun[i]:
            # Pronouns attend to earlier entities and other pronouns
            for j in range(i):
                if is_entity[j] or is_pronoun[j]:
                    # Distance decay
                    distance_factor = 1.0 / (1 + 0.1 * (i - j))
                    attention[i, j] += 0.3 * distance_factor

        elif is_entity[i]:
            # Entities attend moderately to pronouns and other entities
            for j in range(n):
                if is_pronoun[j]:
                    attention[i, j] += 0.2
                elif is_entity[j] and j != i:
                    # Semantic similarity for entities
                    sim = embedding_similarity(tokens, i, j)
                    attention[i, j] += 0.1 * (1 + sim)

        # NEW: Special case for verbs - they should attend to nearby arguments more than distant pronouns
        elif is_verb[i]:
            # Verbs attend strongly to nearby entities and moderately to pronouns
            for j in range(n):
                if is_entity[j]:
                    # Stronger attention to nearby entities
                    distance_factor = 1.0 / (1 + 0.05 * abs(i - j))
                    attention[i, j] += 0.25 * distance_factor
                elif is_pronoun[j]:
                    # Moderate attention to pronouns with distance decay
                    distance_factor = 1.0 / (1 + 0.1 * abs(i - j))
                    attention[i, j] += 0.15 * distance_factor

        else:
            # Other words attend moderately to pronouns and entities, but reduce pronoun attention
            for j in range(n):
                if is_pronoun[j]:
                    attention[i, j] += 0.08  # Reduced from 0.15
                elif is_entity[j]:
                    attention[i, j] += 0.1

    return "program_L10H11", make_row_stochastic(attention)



def program_L10H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Head with reduced self-attention, minimal CLS bias, strong SEP self-attention, and punctuation focus."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Much reduced self-attention for most tokens
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.4  # Strong SEP self-attention
        elif tokens[i] in ['.', '!', '?']:
            attention[i, i] = 0.5  # Strong punctuation self-attention
        else:
            attention[i, i] = 0.1  # Much lower base self-attention

        # Minimal attention to [CLS] token
        if tokens[0] in ['[CLS]', '<|endoftext|>']:
            attention[i, 0] = 0.05  # Reduced from 0.15

        # Handle special tokens
        if tokens[i] == '[SEP]':
            # [SEP] attends strongly to sentence-final punctuation
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 0.2  # Reduced from 0.4
                elif tokens[j] in [',', ';', ':']:
                    attention[i, j] = 0.1  # Reduced from 0.2

            # [SEP] also attends to key content words
            for j in range(n):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'PROPN']:
                            attention[i, j] += 0.05  # Reduced from 0.1

        # Strong attention to contractions and special punctuation
        for j in range(n):
            if tokens[j] in ["'", "'", "`"]:
                attention[i, j] += 0.15

        # Punctuation receives attention from nearby tokens
        if tokens[i] in ['.', '!', '?', ',', ';', ':']:
            for j in range(max(0, i-3), min(n, i+3)):
                if j != i:
                    attention[j, i] += 0.05  # Reduced from 0.1

        # Content word relationships
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Pronouns attend to potential antecedents
                if spacy_token.pos_ == 'PRON':
                    for j in range(i):
                        if j < len(token_to_spacy) and token_to_spacy[j]:
                            j_spacy_idx = token_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_spacy_token = doc[j_spacy_idx]
                                if j_spacy_token.pos_ in ['NOUN', 'PROPN']:
                                    attention[i, j] += 0.04  # Reduced from 0.08

                # Verbs attend to their subjects/objects
                if spacy_token.pos_ == 'VERB':
                    for child in spacy_token.children:
                        if child.dep_ in ['nsubj', 'dobj', 'iobj']:
                            for j in range(n):
                                if j < len(token_to_spacy) and token_to_spacy[j]:
                                    if child.i in token_to_spacy[j]:
                                        attention[i, j] += 0.03  # Reduced from 0.06

                # Semantic similarity between content words
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    for j in range(n):
                        if i != j:
                            sim = embedding_similarity(tokens, i, j)
                            if sim > 0.3:
                                attention[i, j] += sim * 0.08  # Reduced from 0.15

        # Reduced distance decay for nearby tokens
        for j in range(n):
            if i != j:
                distance = abs(i - j)
                if distance <= 3:
                    attention[i, j] += 0.02 / (distance + 1)  # Reduced from 0.05

    return "program_L10H2", make_row_stochastic(attention)



def program_L10H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation and sentence structure attention head that focuses on commas, periods, quotes and [CLS] with context-sensitive weighting."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # High attention to specific punctuation marks
        for j in range(n):
            token_j = tokens[j]

            # Context-sensitive attention to commas - reduce when there are quotes
            if token_j == ',':
                # Check if there are quotes in the sentence - if so, reduce comma attention
                has_quotes = any(t in ['"', "'"] for t in tokens)
                if has_quotes:
                    attention[i, j] = 0.25  # Reduced from 0.4
                else:
                    attention[i, j] = 0.4

            # Strong attention to quotes
            elif token_j in ['"', "'"]:
                attention[i, j] = 0.35

            # Strong attention to periods
            elif token_j == '.':
                attention[i, j] = 0.15

            # Context-sensitive attention to [CLS] token - much lower base rate
            elif token_j == '[CLS]':
                # Reduce [CLS] attention significantly, except for specific cases
                if token_i in ['"', 'johnny', 'lily', 'timmy']:  # Proper nouns and quotes get higher [CLS] attention
                    attention[i, j] = 0.08
                else:
                    attention[i, j] = 0.03  # Much reduced from 0.1

            # Self-attention for special tokens
            elif i == j and token_i in ['[SEP]', '.', ',', '"', "'"]:
                attention[i, j] = 0.12

            # Moderate attention from [SEP] to period
            elif token_i == '[SEP]' and token_j == '.':
                attention[i, j] = 0.25

            # [SEP] self-attention
            elif token_i == '[SEP]' and token_j == '[SEP]':
                attention[i, j] = 0.2

            # Baseline attention to content words
            else:
                # Check if it's a content word using spacy
                if token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            attention[i, j] = 0.02
                        else:
                            attention[i, j] = 0.01
                else:
                    attention[i, j] = 0.01

    # Special handling for some tokens
    for i in range(n):
        token_i = tokens[i]

        # Period tokens have special attention patterns
        if token_i == '.':
            # Strong attention to [CLS]
            if 0 < n:
                attention[i, 0] = 0.18
            # Moderate attention to first content word
            for j in range(1, min(5, n)):
                if token_to_spacy[j]:
                    attention[i, j] = 0.05

    return "program_L10H3", make_row_stochastic(attention)



def program_L10H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation and sentence boundary detection head with syntactic awareness."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Very high attention to periods and sentence-ending punctuation
        for j in range(n):
            if tokens[j] in ['.', '!', '?']:
                attention[i, j] = 0.4

        # High attention to [CLS] token
        if 0 < n:
            attention[i, 0] = 0.15

        # High attention to commas
        for j in range(n):
            if tokens[j] == ',':
                attention[i, j] = 0.2

        # Self-attention boost
        attention[i, i] = 0.1

        # Special handling for punctuation tokens themselves
        if token in ['.', '!', '?']:
            attention[i, i] = 0.8
            # Periods attend strongly to commas
            for j in range(n):
                if tokens[j] == ',':
                    attention[i, j] = 0.3

        if token == ',':
            attention[i, i] = 0.2

        # [SEP] token attends very strongly to periods
        if token == '[SEP]':
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 0.6
                if tokens[j] == ',':
                    attention[i, j] = 0.2

        # Some syntactic patterns using spacy alignment
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Verbs attend to their subjects
                if spacy_token.pos_ == 'VERB':
                    for child in spacy_token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            # Find corresponding token indices
                            for k in range(n):
                                if k < len(token_to_spacy) and token_to_spacy[k]:
                                    if child.i in token_to_spacy[k]:
                                        attention[i, k] = 0.15

                # Nouns attend to their modifiers
                if spacy_token.pos_ == 'NOUN':
                    for child in spacy_token.children:
                        if child.dep_ in ['amod', 'det']:
                            for k in range(n):
                                if k < len(token_to_spacy) and token_to_spacy[k]:
                                    if child.i in token_to_spacy[k]:
                                        attention[i, k] = 0.08

        # Add some positional bias - slight attention to nearby tokens
        for j in range(max(0, i-3), min(n, i+4)):
            if i != j:
                attention[i, j] += 0.02

    return "program_L10H4", make_row_stochastic(attention)



def program_L10H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence boundary detection head - attends from [SEP] to periods and maintains strong self-attention for boundary tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Strong self-attention baseline
        attention[i, i] = 0.8

        # Special behavior for [SEP] token - much stronger self-attention
        if token_i.strip() == '[SEP]':
            attention[i, i] = 0.7  # Increase from 0.3 to match real patterns

            # Very strong attention to periods
            for j in range(n):
                if tokens[j].strip() == '.':
                    attention[i, j] = 3.0

            # Some attention to other sentence-ending tokens
            for j in range(n):
                token_j = tokens[j].strip()
                if token_j in ['!', '?', ';']:
                    attention[i, j] = 1.0

            # Add broader attention to content words from [SEP]
            for j in range(n):
                if i == j or tokens[j].strip() in ['.', '!', '?', ';']:
                    continue
                spacy_j_indices = token_to_spacy[j]
                for spacy_idx in spacy_j_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        # Attend to content words (nouns, verbs, adjectives)
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                            attention[i, j] += 0.15

        # Stronger self-attention for periods
        elif token_i.strip() == '.':
            attention[i, i] = 1.2  # Increase period self-attention

        # For non-SEP tokens, add attention to main verbs and important content words
        else:
            spacy_indices = token_to_spacy[i]

            for j in range(n):
                if i == j:
                    continue

                token_j = tokens[j]
                spacy_j_indices = token_to_spacy[j]

                # Attend to main verbs
                for spacy_idx in spacy_j_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ == 'VERB' and spacy_token.dep_ in ['ROOT', 'ccomp']:
                            attention[i, j] += 0.4

                # Attend to coordinating conjunctions
                for spacy_idx in spacy_j_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ == 'CCONJ':
                            attention[i, j] += 0.3

                # Some semantic similarity attention
                if len(spacy_indices) > 0 and len(spacy_j_indices) > 0:
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.7:
                        attention[i, j] += 0.2 * sim

    return "program_L10H5", make_row_stochastic(attention)



def program_L10H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference and punctuation attention head with structural focus."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        # Strong self-attention for punctuation and special tokens
        if tokens[i] in [".", ",", "?", "!", "[SEP]", "[CLS]"]:
            attention[i, i] = 0.8

        # [SEP] token strongly attends to sentence-final punctuation
        if tokens[i] == "[SEP]":
            for j in range(n):
                if tokens[j] == ".":
                    attention[i, j] = 0.6
                elif tokens[j] in [",", "?", "!"]:
                    attention[i, j] = 0.3

        # Punctuation gets attention from many other tokens
        for j in range(n):
            if tokens[j] in [".", ","]:
                if tokens[i] not in ["[SEP]", "[CLS]"] and i != j:
                    attention[i, j] = 0.15

        # Get spacy token info for current position
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Pronouns attend strongly to their antecedents
            if spacy_token.pos_ == "PRON":
                for j in range(n):
                    target_spacy = token_to_spacy[j]
                    if target_spacy:
                        target_token = doc[target_spacy[0]]
                        if target_token.pos_ == "NOUN":
                            # Use embedding similarity to find likely antecedents
                            sim = embedding_similarity(tokens, i, j)
                            if sim > 0.1:
                                attention[i, j] = 0.4 + sim * 0.3

            # Verbs attend to their subjects/objects
            if spacy_token.pos_ == "VERB":
                for j in range(n):
                    target_spacy = token_to_spacy[j]
                    if target_spacy:
                        target_token = doc[target_spacy[0]]
                        if target_token.pos_ in ["NOUN", "PROPN"]:
                            # Check if it's a likely subject/object
                            if target_token.dep_ in ["nsubj", "dobj", "pobj"]:
                                attention[i, j] = 0.3

            # Content words get moderate self-attention
            if spacy_token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]:
                attention[i, i] = max(attention[i, i], 0.2)

        # General attention to [CLS] token
        if tokens[0] == "[CLS]":
            attention[i, 0] = 0.08

        # Some attention to nearby tokens
        if i > 0:
            attention[i, i-1] = 0.05
        if i < n-1:
            attention[i, i+1] = 0.05

    return "program_L10H6", make_row_stochastic(attention)



def program_L10H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation and sentence boundary detection head - attends strongly to commas, apostrophes, and special tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse sentence for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_text = tokens[i].strip()

        # Strong attention to punctuation marks
        for j in range(n):
            target_text = tokens[j].strip()

            # Very high attention to commas
            if target_text == ',':
                attention[i, j] = 0.4
            # High attention to apostrophes  
            elif target_text == "'":
                attention[i, j] = 0.35
            # Moderate attention to periods
            elif target_text == '.':
                attention[i, j] = 0.2
            # Attention to special tokens
            elif target_text == '[CLS]':
                attention[i, j] = 0.15
            elif target_text == '[SEP]':
                attention[i, j] = 0.12

        # Self-attention for punctuation tokens
        if token_text in [',', "'", '.', '[CLS]', '[SEP]']:
            attention[i, i] = max(attention[i, i], 0.2)

        # When no strong punctuation targets, attend to early tokens and syntactic heads
        if attention[i].sum() < 0.1:
            # Attend to first few content tokens
            for j in range(min(3, n)):
                if tokens[j].strip() not in ['[CLS]', '[SEP]']:
                    attention[i, j] = 0.08

            # Add some syntactic attention using spacy
            if alignment[i]:
                spacy_idx = alignment[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]

                    # Attend to syntactic head
                    if spacy_token.head != spacy_token:
                        head_alignment = align_spacy_to_tokens(sentence)
                        if spacy_token.head.i < len(head_alignment):
                            for head_tok_idx in head_alignment[spacy_token.head.i]:
                                if head_tok_idx < n:
                                    attention[i, head_tok_idx] = 0.06

        # Add baseline attention to avoid zero rows
        if attention[i].sum() == 0:
            attention[i, 0] = 0.1  # Fallback to [CLS]

    return "program_L10H7", make_row_stochastic(attention)



def program_L10H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence structure and self-attention head with focus on punctuation and boundaries."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Find sentence-final punctuation positions
    punct_positions = []
    for i, token in enumerate(tokens):
        if token.strip() in ['.', '!', '?']:
            punct_positions.append(i)

    # Find [CLS] and [SEP] positions
    cls_pos = None
    sep_pos = None
    for i, token in enumerate(tokens):
        if token.strip() == '[CLS]':
            cls_pos = i
        elif token.strip() == '[SEP]':
            sep_pos = i

    for i, query_token in enumerate(tokens):
        query_stripped = query_token.strip()

        # Strong self-attention for most tokens
        attention[i, i] = 0.15

        # Special handling for [SEP] token
        if query_stripped == '[SEP]':
            # Very strong attention to sentence-final punctuation
            for punct_pos in punct_positions:
                attention[i, punct_pos] = 0.4
            # Strong attention to [CLS]
            if cls_pos is not None:
                attention[i, cls_pos] = 0.1
            continue

        # Special handling for sentence-final punctuation
        if query_stripped in ['.', '!', '?']:
            # Attention to [CLS]
            if cls_pos is not None:
                attention[i, cls_pos] = 0.05
            continue

        # Special handling for [CLS]
        if query_stripped == '[CLS]':
            attention[i, i] = 0.1  # Lower self-attention for [CLS]
            continue

        # For regular tokens, attend to sentence-final punctuation
        for punct_pos in punct_positions:
            attention[i, punct_pos] = 0.12

        # Attend to [CLS] token
        if cls_pos is not None:
            attention[i, cls_pos] = 0.03

        # Add syntactic relationships using spacy alignment
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Attend to syntactic head
                if spacy_token.head != spacy_token:
                    for j, target_spacy_indices in enumerate(token_to_spacy):
                        if target_spacy_indices and spacy_token.head.i in target_spacy_indices:
                            attention[i, j] += 0.08
                            break

                # Attend to syntactic children
                for child in spacy_token.children:
                    for j, target_spacy_indices in enumerate(token_to_spacy):
                        if target_spacy_indices and child.i in target_spacy_indices:
                            attention[i, j] += 0.04
                            break

        # Add some semantic similarity-based attention
        for j, target_token in enumerate(tokens):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity threshold
                    attention[i, j] += 0.06 * sim

    return "program_L10H8", make_row_stochastic(attention)



def program_L10H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation-focused attention head with strong self-attention and comma/period emphasis."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Base self-attention for all tokens
        attention[i, i] = 1.0

        # Special case: [SEP] token heavily attends to final punctuation
        if tokens[i] == '[SEP]':
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 8.0
                elif tokens[j] == ',':
                    attention[i, j] = 2.0

        # Tokens attend to commas with moderate strength
        elif tokens[i] != ',' and tokens[i] != '[CLS]':
            for j in range(n):
                if tokens[j] == ',':
                    attention[i, j] = 0.8

        # Content words get stronger self-attention
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'PROPN', 'ADJ']:
                attention[i, i] = 2.0

        # Pronouns attend to potential antecedents
        if spacy_indices and doc[spacy_indices[0]].pos_ == 'PRON':
            for j in range(i):
                j_spacy = token_to_spacy[j]
                if j_spacy and doc[j_spacy[0]].pos_ in ['NOUN', 'PROPN']:
                    # Use embedding similarity to find related tokens
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        attention[i, j] = 0.5

        # Boost attention between semantically similar tokens
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:
                    attention[i, j] += 0.3

        # Final punctuation gets attention from many tokens
        if tokens[i] in ['.', '!', '?']:
            for j in range(n):
                if tokens[j] != '[CLS]' and j != i:
                    attention[j, i] += 0.2

    return "program_L10H9", make_row_stochastic(attention)



def program_L11H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Comma and sentence boundary attention head - focuses on punctuation that marks clause/sentence boundaries."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Find comma and period positions
    comma_positions = []
    period_positions = []

    for i, token in enumerate(tokens):
        if token.strip() == ',':
            comma_positions.append(i)
        elif token.strip() == '.':
            period_positions.append(i)

    for i in range(n):
        token = tokens[i].strip()

        # Special tokens have their own patterns
        if token == '[CLS]':
            attention[i, i] = 1.0
            continue
        elif token == '[SEP]':
            # SEP attends strongly to periods, then commas, then itself
            if period_positions:
                for pos in period_positions:
                    attention[i, pos] = 0.4
            if comma_positions:
                for pos in comma_positions:
                    attention[i, pos] = 0.3
            attention[i, i] = 0.2
            attention[i, 0] = 0.1  # Some attention to [CLS]
            continue
        elif token == '.':
            # Periods attend to themselves and commas
            attention[i, i] = 0.6
            if comma_positions:
                for pos in comma_positions:
                    attention[i, pos] = 0.4 / len(comma_positions)
            continue
        elif token == ',':
            # Commas attend strongly to themselves
            attention[i, i] = 0.8
            attention[i, 0] = 0.2  # Some attention to [CLS]
            continue

        # Regular tokens attend primarily to commas
        if comma_positions:
            total_comma_weight = 0.8
            for pos in comma_positions:
                attention[i, pos] = total_comma_weight / len(comma_positions)

        # Some self-attention for content tokens
        attention[i, i] = 0.1

        # Small amount of attention to [CLS]
        attention[i, 0] = 0.1

    return "program_L11H0", make_row_stochastic(attention)



def program_L11H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Comma and punctuation boundary detection head that identifies clause boundaries and cross-sentence structure."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L11H1", np.array([])

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Find all sentence-final punctuation positions
    sentence_final_positions = []
    for j in range(n):
        if tokens[j] in ['.', '!', '?']:
            sentence_final_positions.append(j)

    for i in range(n):
        token = tokens[i]

        # Strong self-attention for punctuation
        if token in ['.', ',', '!', '?', ';', ':']:
            attention[i, i] = 0.6

        # [SEP] token attends strongly to sentence-final punctuation
        if token == '[SEP]':
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 0.5
                    break

        # NEW: Multi-sentence cross-boundary attention
        # Tokens in later parts of multi-sentence input attend to earlier sentence boundaries
        if len(sentence_final_positions) >= 2:
            # Find which sentence this token belongs to
            token_sentence_idx = 0
            for sent_idx, sent_final_pos in enumerate(sentence_final_positions):
                if i > sent_final_pos:
                    token_sentence_idx = sent_idx + 1
                else:
                    break

            # If token is in second+ sentence, attend strongly to earlier sentence finals
            if token_sentence_idx > 0:
                for sent_idx in range(token_sentence_idx):
                    sent_final_pos = sentence_final_positions[sent_idx]
                    # Strong attention to earlier sentence boundaries
                    attention[i, sent_final_pos] = 0.4

        # Tokens after comma attend strongly to the comma
        if i > 0 and tokens[i-1] == ',':
            attention[i, i-1] = 0.4

        # All tokens attend to comma after clause boundaries
        for j in range(n):
            if tokens[j] == ',':
                # Tokens after comma attend more strongly
                if i > j:
                    attention[i, j] = 0.3
                else:
                    attention[i, j] = 0.1

        # [CLS] self-attention
        if token == '[CLS]':
            attention[i, i] = 0.2

        # Content words attend to [CLS] with moderate strength
        if i > 0 and token not in ['.', ',', '!', '?', ';', ':', '[SEP]']:
            attention[i, 0] = 0.05

        # Some attention to nearby tokens for content words
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Attend to syntactic head
            if spacy_token.head != spacy_token:
                head_char_start = spacy_token.head.idx
                for j in range(n):
                    j_spacy = token_to_spacy[j]
                    if j_spacy:
                        j_char_start = doc[j_spacy[0]].idx
                        if abs(head_char_start - j_char_start) < 2:
                            attention[i, j] = 0.08
                            break

        # Fill remaining attention uniformly
        current_sum = np.sum(attention[i, :])
        if current_sum < 1.0:
            remaining = 1.0 - current_sum
            # Distribute remaining attention
            for j in range(n):
                if attention[i, j] == 0:
                    attention[i, j] = remaining / (n - np.count_nonzero(attention[i, :]))

    # Ensure row stochastic
    attention = make_row_stochastic(attention)

    return "program_L11H1", attention



def program_L11H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Comma-focused attention with enhanced punctuation self-attention and broader SEP distribution."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Find punctuation positions
    comma_positions = []
    period_positions = []
    for i, token in enumerate(tokens):
        if token.strip() == ',':
            comma_positions.append(i)
        elif token.strip() in ['.', '!', '?']:
            period_positions.append(i)

    for i in range(n):
        token = tokens[i].strip()

        # Special handling for [SEP] token - distribute more broadly
        if token == '[SEP]':
            # [SEP] attends strongly to final punctuation
            if period_positions:
                attention[i, period_positions[-1]] = 0.4
            # Moderate attention to commas
            if comma_positions:
                for comma_pos in comma_positions:
                    attention[i, comma_pos] = 0.15 / len(comma_positions)
            # Distribute remaining attention across all content tokens
            remaining_mass = 1.0 - attention[i].sum()
            content_tokens = [j for j in range(n) if tokens[j].strip() not in ['[CLS]', '[SEP]', ',', '.', '!', '?']]
            if content_tokens and remaining_mass > 0:
                content_weight = remaining_mass / len(content_tokens)
                for j in content_tokens:
                    attention[i, j] = content_weight
            continue

        # Enhanced self-attention for commas
        if token == ',':
            attention[i, i] = 0.5  # Much stronger self-attention for commas
            attention[i, 0] = 0.08  # Reduced [CLS] attention
            # Fill remaining with small values
            remaining_mass = 1.0 - attention[i].sum()
            if remaining_mass > 0:
                for j in range(n):
                    if attention[i, j] == 0:
                        attention[i, j] = remaining_mass / (n - np.count_nonzero(attention[i]))
            continue

        # Enhanced self-attention for final punctuation
        if token in ['.', '!', '?']:
            attention[i, i] = 0.6  # Stronger self-attention for periods
            if comma_positions:
                attention[i, comma_positions[0]] = 0.25  # attend to first comma
            attention[i, 0] = 0.15  # attend to [CLS]
            continue

        # For all other tokens
        base_attention = 0.05

        # Strong attention to commas (especially first comma)
        if comma_positions:
            first_comma = comma_positions[0]
            attention[i, first_comma] = 0.4
            # Distribute remaining comma attention
            for comma_pos in comma_positions[1:]:
                attention[i, comma_pos] = 0.1 / max(1, len(comma_positions) - 1)

        # Reduced attention to [CLS] token
        attention[i, 0] = 0.08  # Reduced from 0.15

        # Self-attention
        attention[i, i] = 0.08

        # Fill remaining attention with small values to nearby tokens and similar tokens
        remaining_mass = 1.0 - attention[i].sum()
        if remaining_mass > 0:
            for j in range(n):
                if attention[i, j] == 0:  # only fill empty slots
                    # Small positional bias for nearby tokens
                    distance = abs(i - j)
                    pos_weight = max(0, 1.0 - distance * 0.1)

                    # Small semantic similarity bonus
                    sem_weight = 0
                    if i != j:
                        try:
                            sim = embedding_similarity(tokens, i, j)
                            sem_weight = max(0, sim * 0.5)
                        except:
                            sem_weight = 0

                    attention[i, j] = base_attention * (0.5 + 0.3 * pos_weight + 0.2 * sem_weight)

    return "program_L11H10", make_row_stochastic(attention)



def program_L11H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation and clause boundary attention head with content word secondary focus and strong punctuation self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Very high self-attention for punctuation tokens (especially [SEP] and commas)
        if token_i == '[SEP]':
            attention[i, i] = 0.9
        elif token_i == ',':
            attention[i, i] = 0.6

        # Very high attention from period and [SEP] to period
        if token_i in ['.', '[SEP]']:
            for j in range(n):
                if tokens[j] == '.':
                    attention[i, j] = 0.8

        # Reduced attention from various tokens to commas (original was too strong)
        if any(tokens[j] == ',' for j in range(n)):
            comma_indices = [j for j in range(n) if tokens[j] == ',']
            for comma_j in comma_indices:
                # Tokens after comma attend to comma, but less strongly
                if i > comma_j:
                    attention[i, comma_j] = 0.25
                # Some tokens before comma also attend to it, but less strongly
                elif i < comma_j and i > 0:
                    attention[i, comma_j] = 0.15

        # Self-attention for content words
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                attention[i, i] = 0.05

        # Attention to semantically important tokens
        for j in range(n):
            if i != j:
                spacy_j_indices = token_to_spacy[j]
                if spacy_j_indices:
                    spacy_j = doc[spacy_j_indices[0]]

                    # Attention to subjects and main verbs
                    if spacy_j.pos_ in ['NOUN', 'PROPN', 'PRON'] and spacy_j.dep_ in ['nsubj', 'nsubjpass']:
                        attention[i, j] += 0.02

                    # Attention to main verbs
                    if spacy_j.pos_ == 'VERB' and spacy_j.dep_ == 'ROOT':
                        attention[i, j] += 0.02

        # Some attention to [CLS] token
        if tokens[0] in ['[CLS]', '<|endoftext|>']:
            attention[i, 0] = 0.015

        # Special handling for [SEP] token attention distribution
        if token_i == '[SEP]':
            # Already handled period attention above
            # Add some attention to content words
            for j in range(n):
                if j != i and tokens[j] not in ['.', ',', '[CLS]', '[SEP]']:
                    spacy_j_indices = token_to_spacy[j]
                    if spacy_j_indices:
                        spacy_j = doc[spacy_j_indices[0]]
                        if spacy_j.pos_ in ['NOUN', 'PROPN', 'VERB']:
                            attention[i, j] = 0.02

    # Add small baseline attention everywhere
    attention += 0.01

    return "program_L11H11", make_row_stochastic(attention)



def program_L11H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation and syntactic structure attention with strong self-attention for delimiters and enhanced [SEP] behavior."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Strong self-attention for punctuation and special tokens
        if token_i in [',', '.', '[CLS]', '[SEP]'] or token_i.startswith('##'):
            attention[i, i] = 0.4
        else:
            attention[i, i] = 0.05

        # High attention to commas from most tokens
        for j in range(n):
            if tokens[j] == ',' and i != j:
                attention[i, j] = 0.25

        # Attention to [CLS] token
        if tokens[0] == '[CLS]':
            attention[i, 0] = 0.08

        # [SEP] token attends broadly to content words
        if token_i == '[SEP]':
            for j in range(n):
                if tokens[j] not in ['[CLS]', '[SEP]', ',', '.']:
                    # Get spacy features for content detection
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_token = doc[spacy_indices[0]]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                            attention[i, j] = 0.06

        # Syntactic attention patterns
        spacy_indices_i = token_to_spacy[i]
        if spacy_indices_i:
            spacy_token_i = doc[spacy_indices_i[0]]

            # Modifiers attend to their heads
            if spacy_token_i.dep_ in ['amod', 'nmod', 'advmod']:
                head = spacy_token_i.head
                for j in range(n):
                    spacy_indices_j = token_to_spacy[j]
                    if spacy_indices_j and doc[spacy_indices_j[0]] == head:
                        attention[i, j] = 0.08

            # Determiners attend to their nouns
            if spacy_token_i.pos_ == 'DET':
                for j in range(i+1, min(i+3, n)):
                    spacy_indices_j = token_to_spacy[j]
                    if spacy_indices_j:
                        spacy_token_j = doc[spacy_indices_j[0]]
                        if spacy_token_j.pos_ in ['NOUN', 'PROPN']:
                            attention[i, j] = 0.06

        # Recency bias - attend more to recent tokens
        for j in range(max(0, i-3), i):
            if tokens[j] not in [',', '.', '[CLS]']:
                attention[i, j] += 0.03

        # Semantic similarity boost
        for j in range(n):
            if i != j and tokens[j] not in ['[CLS]', '[SEP]', ',', '.']:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:
                    attention[i, j] += sim * 0.05

    # Enhanced [SEP] token behavior - handle the specific failure pattern
    for i in range(n):
        if tokens[i] == '[SEP]':
            # Much stronger self-attention for [SEP]
            attention[i, i] = 0.65

            # Strong attention to sentence-final punctuation
            for j in range(n):
                if tokens[j] == '.' and j == n-2:  # Period before [SEP]
                    attention[i, j] = 0.15

    # Enhanced sentence-final punctuation self-attention
    for i in range(n):
        if tokens[i] == '.' and i == n-2:  # Period before [SEP]
            attention[i, i] = 0.65

    return "program_L11H2", make_row_stochastic(attention)



def program_L11H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence boundary and punctuation attention head with strong punctuation self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong self-attention baseline
        attention[i, i] = 0.1

        # Special handling for [SEP] token (usually last)
        if tokens[i] == '[SEP]':
            # Very strong attention to final punctuation
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 0.6
                elif tokens[j] == ',':
                    attention[i, j] = 0.3
                elif tokens[j] == '[CLS]':
                    attention[i, j] = 0.02
                else:
                    attention[i, j] = 0.01
            continue

        # Reduced attention to commas from all tokens
        for j in range(n):
            if tokens[j] == ',':
                attention[i, j] = 0.2

        # Moderate attention to important punctuation
        for j in range(n):
            if tokens[j] in ['.', '!', '?']:
                attention[i, j] = 0.05

        # Special attention for [CLS] token
        if tokens[i] == '[CLS]':
            attention[i, i] = 0.07

        # Content word patterns - attend to pronouns and important words
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # If this is a pronoun or proper noun, boost self-attention
                if spacy_token.pos_ in ['PRON', 'PROPN']:
                    attention[i, i] = 0.12

                # If this is a content word, attend to related words
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    for j in range(n):
                        if j != i and j < len(token_to_spacy) and token_to_spacy[j]:
                            sim = embedding_similarity(tokens, i, j)
                            if sim > 0.5:
                                attention[i, j] += 0.03

        # Attend to [CLS] from various positions
        for j in range(n):
            if tokens[j] == '[CLS]':
                attention[i, j] += 0.02

        # Slight attention to previous tokens (recency bias)
        for j in range(max(0, i-3), i):
            attention[i, j] += 0.01

    # NEW: Special case for strong punctuation self-attention
    for i in range(n):
        if tokens[i] in ['.', '!', '?']:
            # Very strong self-attention for sentence-ending punctuation
            attention[i, i] = 0.4
        elif tokens[i] == "'":
            # Strong self-attention for apostrophes  
            attention[i, i] = 0.5
        elif tokens[i] == '[SEP]':
            # Boost [SEP] self-attention significantly
            attention[i, i] = 0.8

    return "program_L11H3", make_row_stochastic(attention)



def program_L11H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Structural attention head focusing on special tokens, punctuation, and [CLS] with some content relationships."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Strong self-attention for punctuation
        if token_i in ['.', ',', '"', '!', '?', ';', ':']:
            attention[i, i] = 0.5

        # [SEP] token special behavior
        if token_i == '[SEP]':
            # Strong self-attention for [SEP]
            attention[i, i] = 0.4
            # Very high attention to sentence-final punctuation
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 0.4
            # High attention to commas
            for j in range(n):
                if tokens[j] == ',':
                    attention[i, j] = 0.3
            # Moderate attention to [CLS]
            if '[CLS]' in tokens:
                cls_idx = tokens.index('[CLS]')
                attention[i, cls_idx] = 0.1

        # Attention to [CLS] from many tokens
        if '[CLS]' in tokens:
            cls_idx = tokens.index('[CLS]')
            if i != cls_idx:
                # Content words attend more to [CLS]
                if i < len(token_to_spacy) and token_to_spacy[i]:
                    spacy_idx = token_to_spacy[i][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                            attention[i, cls_idx] = 0.06
                        else:
                            attention[i, cls_idx] = 0.03
                else:
                    attention[i, cls_idx] = 0.02

        # Self-attention for [CLS]
        if token_i == '[CLS]':
            attention[i, i] = 0.05

        # Attention from sentence-final punctuation to commas
        if token_i in ['.', '!', '?']:
            for j in range(n):
                if tokens[j] == ',':
                    attention[i, j] = 0.3

        # Attention from tokens after commas to the comma
        if i > 0 and tokens[i-1] == ',':
            attention[i, i-1] = 0.25

        # Handle subword token relationships
        if token_i.startswith('##'):
            # Find the root token (first token of this word)
            for j in range(i-1, -1, -1):
                if not tokens[j].startswith('##'):
                    # This is the root token
                    attention[i, j] = 0.15
                    break

        # Some content-based attention patterns
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Verbs attend to their subjects/objects
                if spacy_token.pos_ == 'VERB':
                    for child in spacy_token.children:
                        if child.dep_ in ['nsubj', 'dobj']:
                            # Find corresponding token indices
                            for j in range(n):
                                if j < len(token_to_spacy) and token_to_spacy[j]:
                                    j_spacy_idx = token_to_spacy[j][0]
                                    if j_spacy_idx < len(doc) and doc[j_spacy_idx] == child:
                                        attention[i, j] = 0.04

                # Nouns attend to their modifiers
                if spacy_token.pos_ in ['NOUN', 'PROPN']:
                    for child in spacy_token.children:
                        if child.dep_ in ['amod', 'det']:
                            for j in range(n):
                                if j < len(token_to_spacy) and token_to_spacy[j]:
                                    j_spacy_idx = token_to_spacy[j][0]
                                    if j_spacy_idx < len(doc) and doc[j_spacy_idx] == child:
                                        attention[i, j] = 0.03

        # Moderate self-attention for various tokens
        if token_i not in ['.', ',', '"', '[SEP]', '[CLS]']:
            attention[i, i] = 0.02

        # Some similarity-based attention
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity
                    attention[i, j] += 0.02

    # Normalize to make row-stochastic
    return "program_L11H4", make_row_stochastic(attention)



def program_L11H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation-focused attention head with strong bias toward commas as discourse anchors."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Find punctuation tokens, especially commas
    comma_indices = []
    punct_indices = []
    for i, token in enumerate(tokens):
        if token.strip() == ',':
            comma_indices.append(i)
            punct_indices.append(i)
        elif token.strip() in ['.', '!', '?', ';', ':', '"']:
            punct_indices.append(i)

    for i in range(n):
        # Strong self-attention baseline
        attention[i, i] = 0.1

        # Very strong attention to commas from all positions
        for comma_idx in comma_indices:
            if comma_idx != i:
                attention[i, comma_idx] = 0.4

        # Moderate attention to other punctuation
        for punct_idx in punct_indices:
            if punct_idx not in comma_indices and punct_idx != i:
                attention[i, punct_idx] = 0.15

        # Attention to [CLS] token
        if tokens[0] in ['[CLS]', '<s>']:
            attention[i, 0] += 0.08

        # Special case: [SEP] token attends strongly to final punctuation
        if tokens[i] in ['[SEP]', '</s>']:
            for j in range(n-2, -1, -1):  # Look backwards for punctuation
                if tokens[j].strip() in ['.', '!', '?']:
                    attention[i, j] += 0.3
                    break

        # Add some local attention to adjacent tokens
        if i > 0:
            attention[i, i-1] += 0.05
        if i < n-1:
            attention[i, i+1] += 0.05

    return "program_L11H5", make_row_stochastic(attention)



def program_L11H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Structural attention head focusing on special tokens, punctuation, and [CLS] with positional biases."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Special handling for [SEP] - much higher self-attention
        if token_i == '[SEP]':
            attention[i, i] = 0.85
            # [SEP] attends more broadly to all tokens
            for j in range(n):
                if i != j:
                    if tokens[j] in ['.', ',', '!', '?', ';', ':']:
                        attention[i, j] = 0.05
                    else:
                        attention[i, j] = 0.01
        # [CLS] gets reduced self-attention
        elif token_i == '[CLS]':
            attention[i, i] = 0.2
        # Punctuation gets slightly reduced self-attention
        elif token_i in ['.', ',', '!', '?', ';', ':']:
            attention[i, i] = 0.4
        else:
            attention[i, i] = 0.03

        # Strong attention to [CLS] from content tokens
        if token_i not in ['[CLS]', '[SEP]'] and not token_i in ['.', ',', '!', '?', ';', ':']:
            attention[i, 0] = 0.06

        # [SEP] attends strongly to final punctuation - but this is handled above
        if token_i == '[SEP]' and n > 1:
            for j in range(n-1, -1, -1):
                if tokens[j] in ['.', ',', '!', '?', ';', ':']:
                    attention[i, j] = 0.4
                    break

        # Strong attention to punctuation from nearby tokens (but not from [SEP])
        if token_i != '[SEP]':
            for j in range(n):
                if tokens[j] in [',', '.', '!', '?', ';', ':'] and i != j:
                    distance = abs(i - j)
                    if distance <= 8:  # Within reasonable range
                        punct_weight = 0.35 * np.exp(-0.1 * distance)
                        attention[i, j] += punct_weight

        # Attention to first content token (position 1 typically)
        if i > 1 and n > 1 and token_i != '[SEP]':
            attention[i, 1] = 0.025

        # Some recency bias for recent tokens (but not from [SEP])
        if token_i != '[SEP]':
            for j in range(max(0, i-3), i):
                if tokens[j] not in ['[CLS]', '[SEP]'] and not tokens[j] in ['.', ',', '!', '?', ';', ':']:
                    attention[i, j] += 0.015

        # Add some random baseline attention
        for j in range(n):
            if token_i != '[SEP]' or i == j:  # [SEP] baseline already handled above
                attention[i, j] += 0.01

    return "program_L11H6", make_row_stochastic(attention)



def program_L11H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation-focused attention head that strongly attends to commas, periods, and apostrophes."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Find punctuation tokens (commas, periods, apostrophes, etc.)
    punct_indices = []
    for i, token in enumerate(tokens):
        token_clean = token.strip()
        if token_clean in [',', '.', '!', '?', ';', ':', "'"]:
            punct_indices.append(i)

    for i in range(n):
        token_clean = tokens[i].strip()

        # Special tokens get different patterns
        if token_clean in ['[CLS]', '[SEP]']:
            if token_clean == '[SEP]':
                # [SEP] attends strongly to punctuation, moderately to itself and [CLS]
                for punct_idx in punct_indices:
                    attention_matrix[i, punct_idx] = 0.4
                attention_matrix[i, i] = 0.2  # self-attention
                attention_matrix[i, 0] = 0.1  # to [CLS]
            else:  # [CLS]
                # [CLS] has strong self-attention
                attention_matrix[i, i] = 0.8

        elif token_clean in [',', '.', '!', '?', ';', ':', "'"]:
            # Punctuation tokens attend strongly to themselves
            attention_matrix[i, i] = 0.6
            # And moderately to other punctuation
            for punct_idx in punct_indices:
                if punct_idx != i:
                    attention_matrix[i, punct_idx] = 0.2

        else:
            # Regular tokens attend very strongly to punctuation
            total_punct_weight = 0.7
            if punct_indices:
                punct_weight = total_punct_weight / len(punct_indices)
                for punct_idx in punct_indices:
                    attention_matrix[i, punct_idx] = punct_weight

            # Small self-attention
            attention_matrix[i, i] = 0.1

            # Small attention to [CLS]
            attention_matrix[i, 0] = 0.05

    # Ensure all rows sum to something positive before normalization
    for i in range(n):
        if attention_matrix[i].sum() == 0:
            attention_matrix[i, i] = 1.0

    attention_matrix = make_row_stochastic(attention_matrix)
    return "program_L11H7", attention_matrix



def program_L11H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Punctuation and negation attention head with adaptive self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L11H8", np.array([])

    # Initialize with uniform attention
    attention = np.ones((n, n)) * 0.01

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        for j in range(n):
            token_i = tokens[i].strip()
            token_j = tokens[j].strip()

            # Adaptive self-attention based on token type
            if i == j:
                # Strong self-attention for punctuation
                if token_i in [".", ",", "!", "?", ":", ";"]:
                    attention[i, j] += 0.7
                # Moderate self-attention for content words
                elif any(token_to_spacy[i]):
                    spacy_indices = token_to_spacy[i]
                    is_content_word = False
                    for spacy_idx in spacy_indices:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and len(spacy_token.text) > 2:
                                is_content_word = True
                                break
                    if is_content_word:
                        attention[i, j] += 0.4
                    else:
                        attention[i, j] += 0.2
                # Weak self-attention for special tokens and short function words
                elif token_i in ["[CLS]", "[SEP]"] or len(token_i) <= 2:
                    attention[i, j] += 0.1
                else:
                    attention[i, j] += 0.3

            # High attention from punctuation to apostrophes/quotes
            if token_i in [".", ",", "!", "?"]:
                if "'" in token_j or '"' in token_j:
                    attention[i, j] += 0.4
                elif token_j in [".", ","]:
                    attention[i, j] += 0.2

            # Strong attention to apostrophes from various tokens
            if "'" in token_j:
                attention[i, j] += 0.2
                # Extra boost for negation-related tokens
                if any(token_to_spacy[i]) and any(doc[idx].text.lower() in ["didn", "don", "won", "can", "t"] 
                       for idx in token_to_spacy[i]):
                    attention[i, j] += 0.1

            # Attention to negation markers
            if any(token_to_spacy[j]):
                spacy_indices = token_to_spacy[j]
                for spacy_idx in spacy_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        # Negation-related words get extra attention
                        if spacy_token.text.lower() in ["not", "didn", "don", "won", "can"] or spacy_token.dep_ == "neg":
                            attention[i, j] += 0.15

            # Special token behavior
            if token_i == "[CLS]" and token_j == "[CLS]":
                attention[i, j] += 0.2
            elif token_i == "[SEP]":
                if token_j in [".", ",", "!", "?"]:
                    attention[i, j] += 0.3
                elif token_j == "[SEP]":
                    attention[i, j] += 0.1
                elif token_j == "[CLS]":
                    attention[i, j] += 0.08

            # Auxiliary verbs and important function words get some attention
            if any(token_to_spacy[j]):
                spacy_indices = token_to_spacy[j]
                for spacy_idx in spacy_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ["AUX", "VERB"] and spacy_token.text.lower() in ["was", "were", "is", "are", "did", "does"]:
                            attention[i, j] += 0.05
                        elif spacy_token.pos_ == "DET" and spacy_token.text.lower() in ["a", "an", "the"]:
                            attention[i, j] += 0.03

            # Slight positional bias - attention to earlier tokens
            if j < i:
                distance = i - j
                if distance <= 3:
                    attention[i, j] += 0.02 / distance

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L11H8", attention



def program_L11H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Structural landmark attention with enhanced [SEP] sentence summarization behavior."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_alignment = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Strong self-attention for most tokens
        attention_matrix[i, i] = 0.15

        # Special handling for punctuation tokens
        if token.strip() in [',', '.', '!', '?', ';', ':']:
            # Punctuation gets very high self-attention
            attention_matrix[i, i] = 0.4

            # Punctuation also attends to [CLS] and other structural elements
            if tokens[0] in ['[CLS]', '<s>']:
                attention_matrix[i, 0] = 0.2

            # Find other punctuation to attend to
            for j in range(n):
                if j != i and tokens[j].strip() in [',', '.', '!', '?', ';', ':']:
                    attention_matrix[i, j] = 0.15

        # Enhanced [SEP] token behavior
        elif token == '[SEP]':
            # [SEP] gets very high self-attention (sentence summarization behavior)
            attention_matrix[i, i] = 0.6

            # [SEP] attends strongly to final punctuation
            for j in range(n):
                if tokens[j].strip() in ['.', '!', '?']:
                    attention_matrix[i, j] = 0.4

            # Also attends to [CLS] and key content words
            if tokens[0] in ['[CLS]', '<s>']:
                attention_matrix[i, 0] = 0.15

            # Attend broadly to content words (verbs, nouns, adjectives, proper nouns)
            for j in range(n):
                if j < len(token_alignment) and token_alignment[j]:
                    spacy_indices = token_alignment[j]
                    for spacy_idx in spacy_indices:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ['VERB', 'NOUN', 'ADJ', 'PROPN']:
                                attention_matrix[i, j] = 0.12
                            # Also attend to determiners, prepositions, and pronouns
                            elif spacy_token.pos_ in ['DET', 'ADP', 'PRON']:
                                attention_matrix[i, j] = 0.08

        # [CLS] token behavior
        elif token == '[CLS]':
            attention_matrix[i, i] = 0.2

        # Regular tokens
        else:
            # Attend to [CLS] if present
            if tokens[0] in ['[CLS]', '<s>']:
                attention_matrix[i, 0] = 0.08

            # Attend to punctuation, especially commas
            for j in range(n):
                if tokens[j].strip() == ',':
                    attention_matrix[i, j] = 0.25
                elif tokens[j].strip() in ['.', '!', '?']:
                    attention_matrix[i, j] = 0.1

            # Syntactic relationships using spacy
            if i < len(token_alignment) and token_alignment[i]:
                spacy_indices = token_alignment[i]
                for spacy_idx in spacy_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]

                        # Determiners attend to their head nouns
                        if spacy_token.pos_ == 'DET':
                            head = spacy_token.head
                            for j in range(n):
                                if j < len(token_alignment) and token_alignment[j]:
                                    for target_spacy_idx in token_alignment[j]:
                                        if target_spacy_idx < len(doc) and doc[target_spacy_idx] == head:
                                            attention_matrix[i, j] = 0.12

                        # Prepositions attend to their objects
                        if spacy_token.pos_ == 'ADP':
                            for child in spacy_token.children:
                                if child.dep_ == 'pobj':
                                    for j in range(n):
                                        if j < len(token_alignment) and token_alignment[j]:
                                            for target_spacy_idx in token_alignment[j]:
                                                if target_spacy_idx < len(doc) and doc[target_spacy_idx] == child:
                                                    attention_matrix[i, j] = 0.1

                        # Conjunctions attend to [CLS]
                        if spacy_token.pos_ == 'CCONJ':
                            if tokens[0] in ['[CLS]', '<s>']:
                                attention_matrix[i, 0] = 0.2

            # Add some attention to nearby tokens
            for j in range(max(0, i-2), min(n, i+3)):
                if j != i:
                    attention_matrix[i, j] += 0.05

    # Normalize to make row stochastic
    attention_matrix = make_row_stochastic(attention_matrix)

    return "program_L11H9", attention_matrix



def program_L1H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic relationship head with strong [CLS] attention and positional biases."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Base self-attention
        attention[i, i] = 0.03

        # Strong attention to [CLS] for content words
        if tokens[0] == '[CLS]':
            if i > 0:  # Non-CLS tokens
                spacy_indices = token_to_spacy[i]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    # Strong CLS attention for determiners, auxiliaries, adverbs, early content words
                    if spacy_token.pos_ in ['DET', 'AUX', 'ADV'] or spacy_token.dep_ in ['det', 'aux', 'advmod']:
                        attention[i, 0] = 0.4 + 0.3 / (i + 1)  # Position decay
                    elif spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ'] and i < 5:
                        attention[i, 0] = 0.2 + 0.2 / (i + 1)
                    else:
                        attention[i, 0] = 0.05 + 0.1 / (i + 1)
                else:
                    attention[i, 0] = 0.05 + 0.1 / (i + 1)

                # TARGETED FIX: Boost [CLS] attention for function words and punctuation
                spacy_indices = token_to_spacy[i]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    # Strong boost for prepositions, conjunctions, and other function words
                    if spacy_token.pos_ in ['ADP', 'CCONJ', 'SCONJ', 'PRON', 'INTJ'] or spacy_token.dep_ in ['prep', 'cc', 'mark']:
                        attention[i, 0] = max(attention[i, 0], 0.6)

                # Strong boost for punctuation (including commas, periods, quotes, exclamations)
                if tokens[i] in [',', '.', '!', '?', '"', "'", ';', ':']:
                    attention[i, 0] = max(attention[i, 0], 0.7)
            else:  # CLS to itself
                attention[i, 0] = 0.1

        # Special handling for punctuation and [SEP]
        if tokens[i] in ['.', ',', ';', '!', '?'] or tokens[i] == '[SEP]':
            # Attend to various content words with moderate strength
            for j in range(n):
                if j != i:
                    spacy_indices_j = token_to_spacy[j]
                    if spacy_indices_j:
                        spacy_token_j = doc[spacy_indices_j[0]]
                        if spacy_token_j.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                            # Distance decay
                            distance = abs(i - j)
                            attention[i, j] += 0.08 / (1 + distance * 0.2)
                        elif tokens[j] in ['.', '[SEP]']:
                            attention[i, j] += 0.12

        # Syntactic relationships
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Verb to subject relationships
            if spacy_token.pos_ == 'VERB':
                for child in spacy_token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        child_tokens = spacy_to_token[child.i]
                        for ct in child_tokens:
                            if ct < n:
                                attention[i, ct] += 0.15

                # Also attend to auxiliaries and modals
                for j in range(max(0, i-3), i):
                    spacy_indices_j = token_to_spacy[j]
                    if spacy_indices_j:
                        spacy_token_j = doc[spacy_indices_j[0]]
                        if spacy_token_j.pos_ in ['AUX', 'MODAL']:
                            attention[i, j] += 0.1

            # Noun to modifiers
            if spacy_token.pos_ == 'NOUN':
                for child in spacy_token.children:
                    if child.dep_ in ['amod', 'det', 'compound']:
                        child_tokens = spacy_to_token[child.i]
                        for ct in child_tokens:
                            if ct < n:
                                attention[i, ct] += 0.08

        # Positional attention patterns
        for j in range(n):
            if i != j:
                distance = abs(i - j)

                # Moderate local attention
                if distance <= 3:
                    attention[i, j] += 0.02 * (4 - distance) / 4

                # Slight preference for attending backwards
                if j < i:
                    attention[i, j] += 0.01

        # Embedding similarity boost for related tokens
        for j in range(n):
            if i != j and tokens[i] != '[CLS]' and tokens[j] != '[CLS]' and tokens[i] != '[SEP]' and tokens[j] != '[SEP]':
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity
                    attention[i, j] += 0.03 * sim

    return "program_L1H0", make_row_stochastic(attention)



def program_L1H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Strong attention to [CLS] token with moderate self-attention and weak local dependencies."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong attention to [CLS] token (index 0)
        if tokens[0] in ['[CLS]', '<s>', '<|endoftext|>']:
            attention[i, 0] = 0.4

        # Self-attention with moderate strength
        attention[i, i] = 0.15

        # Special handling for [SEP] token - much stronger [CLS] attention
        if i == n - 1 and tokens[i] in ['[SEP]', '</s>']:
            attention[i, 0] = 0.8  # Boosted from 0.4 to 0.8
            attention[i, i] = 0.2

        # Add weak attention to nearby tokens
        for j in range(max(0, i-2), min(n, i+3)):
            if j != i and j != 0:
                attention[i, j] += 0.02

        # Boost attention for content words (nouns, verbs, adjectives)
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    # Content words attend more strongly to [CLS]
                    attention[i, 0] *= 1.8  # Increased from 1.5 to 1.8

                    # Content words have stronger self-attention
                    attention[i, i] *= 1.2

        # Handle punctuation differently
        if tokens[i] in [',', '.', '!', '?', ';', ':']:
            # Punctuation attends strongly to [CLS]
            attention[i, 0] *= 1.3
            # Reduce self-attention for punctuation
            attention[i, i] *= 0.5

        # Special case for named entities and pronouns
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['PRON', 'PROPN'] or spacy_token.ent_type_:
                    # Named entities and pronouns get very strong [CLS] attention
                    attention[i, 0] *= 2.0

        # Reduce [CLS] attention for function words that are over-predicted
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['DET', 'PREP', 'ADP', 'AUX', 'CCONJ', 'SCONJ']:
                    # Function words get reduced [CLS] attention
                    attention[i, 0] *= 0.7

        # Add small random baseline for all positions
        for j in range(n):
            attention[i, j] += 0.01

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L1H1", attention



def program_L1H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that combines [CLS] bias, syntactic relationships, and semantic similarity with strong [SEP]->[CLS] attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special case: [SEP] tokens have very strong attention to [CLS]
        if tokens[i] == '[SEP]' and tokens[0] == '[CLS]':
            attention[i, 0] = 0.85  # Strong [SEP] -> [CLS] attention
            # Distribute remaining weight more evenly across other positions
            remaining_weight = 0.15
            for j in range(1, n):
                if j != i:
                    attention[i, j] = remaining_weight / (n - 2)
            continue

        # Strong bias toward [CLS] token, especially for early positions
        if tokens[0] == '[CLS]':
            cls_weight = 0.6 if i < 4 else 0.3
            attention[i, 0] = cls_weight

        # Get spacy tokens aligned with current LM token
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        for j in range(n):
            if i == j:
                # Self-attention with moderate weight
                attention[i, j] += 0.1
                continue

            # Positional bias - prefer earlier tokens
            pos_bias = max(0.05, 0.2 - 0.02 * abs(i - j))
            attention[i, j] += pos_bias

            # Syntactic relationships
            if current_spacy and token_to_spacy[j]:
                target_spacy = doc[token_to_spacy[j][0]]

                # Head-dependent relationships
                if target_spacy.head == current_spacy or current_spacy.head == target_spacy:
                    attention[i, j] += 0.15

                # Subject-verb relationships
                if (current_spacy.dep_ in ['nsubj', 'nsubjpass'] and target_spacy.pos_ == 'VERB') or \
                   (current_spacy.pos_ == 'VERB' and target_spacy.dep_ in ['nsubj', 'nsubjpass']):
                    attention[i, j] += 0.2

                # Modifier relationships
                if current_spacy.dep_ in ['amod', 'det'] or target_spacy.dep_ in ['amod', 'det']:
                    attention[i, j] += 0.1

            # Semantic similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.3:
                attention[i, j] += 0.15 * sim

            # Special handling for punctuation attending to content words
            if tokens[i] in ['.', '?', ','] and j < i:
                spacy_j_indices = token_to_spacy[j]
                if spacy_j_indices:
                    target_spacy = doc[spacy_j_indices[0]]
                    if target_spacy.pos_ in ['VERB', 'NOUN', 'ADJ']:
                        attention[i, j] += 0.1

    return "program_L1H10", make_row_stochastic(attention)



def program_L1H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Pronoun resolution and coreference head with self-attention for content words."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens for current position
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        # Base attention distribution
        base_weight = 1.0 / n

        for j in range(n):
            weight = base_weight

            # Strong self-attention for content words
            if i == j:
                if current_spacy and current_spacy.pos_ in ['VERB', 'NOUN', 'PRON', 'PROPN']:
                    weight *= 15.0
                else:
                    weight *= 3.0

            # Attention to [CLS] token (position 0)
            elif j == 0:
                weight *= 4.0

            # Pronoun resolution patterns
            elif current_spacy and current_spacy.pos_ == 'PRON':
                target_spacy_indices = token_to_spacy[j]
                if target_spacy_indices:
                    target_spacy = doc[target_spacy_indices[0]]

                    # Pronouns attend to matching referents
                    if target_spacy.pos_ in ['PRON', 'PROPN', 'NOUN']:
                        # Use embedding similarity for coreference
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:  # Threshold for related tokens
                            weight *= 8.0
                        elif target_spacy.pos_ == 'PRON':
                            weight *= 3.0

                    # Distance decay for pronouns
                    distance = abs(i - j)
                    weight *= np.exp(-distance * 0.1)

            # Attention from other tokens to pronouns and proper nouns
            else:
                target_spacy_indices = token_to_spacy[j]
                if target_spacy_indices:
                    target_spacy = doc[target_spacy_indices[0]]
                    if target_spacy.pos_ in ['PRON', 'PROPN']:
                        weight *= 2.0

            # Boost attention to previous token
            if j == i - 1:
                weight *= 2.0

            # Boost attention between similar tokens
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:
                    weight *= 3.0
                elif sim > 0.3:
                    weight *= 1.5

            attention[i, j] = weight

    return "program_L1H11", make_row_stochastic(attention)



def program_L1H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """[CLS]-focused aggregation head with strong attention to the sentence start token."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Identify important token types
    def is_special_token(i):
        return tokens[i] in ['[CLS]', '[SEP]']

    def is_subject_or_main_verb(i):
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            return False
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.dep_ in ['nsubj', 'nsubjpass'] or (token.pos_ == 'VERB' and token.dep_ == 'ROOT'):
                    return True
        return False

    def is_early_content_word(i):
        if i >= min(6, n):  # Only consider first few tokens
            return False
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            return False
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    return True
        return False

    for i in range(n):
        # Detect if this looks like a [CLS]-heavy pattern (long sentences or multiple clauses)
        is_cls_heavy = n > 10 or any(tokens[j] in ['"', ',', ';', ':'] for j in range(n))

        if is_cls_heavy:
            # Strong [CLS] aggregation pattern
            if tokens[0] == '[CLS]':
                if i == 0:
                    attention_matrix[i, 0] = 0.7  # [CLS] self-attention
                else:
                    attention_matrix[i, 0] = 0.6  # Everything else to [CLS]

            # Much lower self-attention for non-[CLS] tokens
            if not is_special_token(i):
                attention_matrix[i, i] = 0.05
            elif tokens[i] == '[SEP]':
                attention_matrix[i, i] = 0.15

            # Minimal attention to other patterns
            for j in range(n):
                if j != i and j != 0 and is_subject_or_main_verb(j):
                    attention_matrix[i, j] += 0.05
        else:
            # Original behavior for simpler sentences
            # Very strong attention to [CLS]
            if tokens[0] == '[CLS]':
                attention_matrix[i, 0] = 0.8 if i == 0 else 0.3

            # Self-attention
            attention_matrix[i, i] = 0.1

            # Attention to subjects and main verbs
            for j in range(n):
                if j != i and is_subject_or_main_verb(j):
                    attention_matrix[i, j] += 0.15

            # Attention to early content words with position decay
            for j in range(min(8, n)):
                if j != i and is_early_content_word(j):
                    position_weight = 1.0 / (j + 1)
                    attention_matrix[i, j] += 0.1 * position_weight

            # Local attention (previous few tokens)
            for offset in [-2, -1, 1, 2]:
                j = i + offset
                if 0 <= j < n and j != i:
                    attention_matrix[i, j] += 0.05

            # Special token self-attention
            if is_special_token(i):
                attention_matrix[i, i] = 0.2

    return "program_L1H2", make_row_stochastic(attention_matrix)



def program_L1H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic relationship head with strong [CLS] bias and coordination detection."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L1H3", np.array([])

    attention = np.zeros((n, n))
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Very strong attention from punctuation to [CLS]
        if token_i in [',', ';', ':', '!', '?'] and 0 < n:
            attention[i, 0] = 0.8

        # Strong [CLS] self-attention
        if i == 0:
            attention[i, i] = 0.4

        # Strong attention from early content tokens to [CLS]
        if i < 4 and i > 0 and token_i not in ['.', '!', '?', '[SEP]']:
            attention[i, 0] = 0.3

        # NEW: Very strong attention from semantically important content words to [CLS]
        spacy_indices = token_to_spacy[i]
        if spacy_indices and i > 0:
            spacy_token = doc[spacy_indices[0]]
            # Key content words get much stronger attention to [CLS]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and len(spacy_token.text) > 2:
                # Check if this is a semantically rich content word
                if spacy_token.pos_ == 'VERB' or spacy_token.pos_ == 'ADJ' or (spacy_token.pos_ == 'NOUN' and not spacy_token.dep_ in ['det', 'prep']):
                    attention[i, 0] = 0.75

        # [SEP] attends to final punctuation (reduced weight)
        if token_i == '[SEP]':
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 0.15

        # Semantic similarity attention (main content) - reduced weight
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:  # High semantic similarity
                    attention[i, j] += 0.1 * sim  # Reduced from 0.15

        # Coordination attention - conjunctions receive attention
        if token_i in ['and', 'or', 'but']:
            attention[i, i] = 0.08

        # Content words attend to conjunctions - reduced weight
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                for j in range(n):
                    if tokens[j] in ['and', 'or']:
                        attention[i, j] += 0.04  # Reduced from 0.06

        # Modest self-attention for content words
        if token_i not in ['[CLS]', '[SEP]', '.', ',', '!', '?']:
            attention[i, i] += 0.05

        # Local context attention (adjacent tokens) - reduced weight
        if i > 0:
            attention[i, i-1] += 0.02  # Reduced from 0.03
        if i < n-1:
            attention[i, i+1] += 0.02  # Reduced from 0.03

        # Recency bias - attend more to recent tokens - reduced weight
        for j in range(n):
            if i != j:
                distance = abs(i - j)
                if distance <= 5:
                    attention[i, j] += 0.015 / (1 + distance * 0.5)  # Reduced from 0.02

    return "program_L1H3", make_row_stochastic(attention)



def program_L1H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Previous-token attention head with special token and grammatical relationship handling."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []
        spacy_token = doc[spacy_indices[0]] if spacy_indices else None

        # High attention to [CLS] for many tokens
        if token != '[CLS]' and tokens[0] == '[CLS]':
            attention[i, 0] = 0.5

        # Strong previous token attention (main pattern)
        if i > 0:
            attention[i, i-1] = 0.8

            # Even stronger for certain relationships
            if spacy_token:
                # Possessive relationships
                if "'" in token or token == 's':
                    attention[i, i-1] = 0.9

                # Prepositions and articles to their heads
                if spacy_token.pos_ in ['ADP', 'DET'] and i > 0:
                    attention[i, i-1] = 0.85

        # Self-attention (moderate)
        attention[i, i] = 0.1

        # Special patterns for punctuation
        if token in [',', '.', '!', '?']:
            # Punctuation attends more to nearby content
            if i > 0:
                attention[i, i-1] = 0.6
            if i > 1:
                attention[i, i-2] = 0.2

        # [CLS] token has high self-attention
        if token == '[CLS]':
            attention[i, i] = 0.7

        # [SEP] token attends strongly to [CLS] - boost this significantly
        if token == '[SEP]' and tokens[0] == '[CLS]':
            attention[i, 0] = 0.9

        # Handle compound words and subwords (like ##s)
        if token.startswith('##') and i > 0:
            attention[i, i-1] = 0.95

        # ADDED: Handle contractions more aggressively
        if i > 0:
            # Common contraction suffixes that should strongly attend to their root
            if token in ["'re", "'s", "'t", "'ll", "'ve", "'d", "'m"]:
                attention[i, i-1] = 0.95
            # Also handle cases where the token starts with apostrophe
            elif token.startswith("'") and len(token) > 1:
                attention[i, i-1] = 0.95
            # Handle single character contractions that are part of contractions
            elif len(token) == 1 and token in ["s", "t", "d", "m"] and i > 1:
                # Check if previous token ends with apostrophe or is an apostrophe
                prev_token = tokens[i-1]
                if prev_token == "'" or prev_token.endswith("'"):
                    attention[i, i-1] = 0.95

        # Add some linguistic relationship bonuses
        if spacy_token and i > 0:
            prev_spacy_indices = token_to_spacy[i-1] if i-1 < len(token_to_spacy) else []
            if prev_spacy_indices:
                prev_spacy = doc[prev_spacy_indices[0]]

                # Boost attention for certain dependency relationships
                if (spacy_token.head == prev_spacy or 
                    prev_spacy.head == spacy_token or
                    spacy_token.dep_ in ['pobj', 'dobj', 'nsubj']):
                    attention[i, i-1] = min(0.95, attention[i, i-1] * 1.2)

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L1H4", attention



def program_L1H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Mixed syntactic-positional head: connects content words to [CLS] and captures coordination patterns."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L1H5", np.array([])

    attention = np.zeros((n, n))
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Special handling for [SEP] token - should attend strongly to [CLS]
        if token == '[SEP]':
            attention[i, 0] = 0.7  # Strong [CLS] attention
            # Distribute remaining attention
            remaining = 0.3
            for j in range(1, n):
                if j != i:
                    attention[i, j] = remaining / (n - 1)
            continue

        # Special token self-attention
        if token in ['[CLS]', '"', '.', ',']:
            # Punctuation should attend more to [CLS], less self-attention
            if token in ['.', ',']:
                attention[i, 0] = 0.3  # More [CLS] attention for punctuation
                attention[i, i] = 0.4  # Reduced self-attention
                # Distribute remaining
                remaining = 0.3
                for j in range(1, n):
                    if j != i:
                        attention[i, j] = remaining / max(1, n - 1)
            else:
                attention[i, i] = 0.8
                # [CLS] gets some distributed attention
                if token == '[CLS]':
                    for j in range(n):
                        if j != i:
                            attention[i, j] = 0.2 / (n - 1)
            continue

        # Get spacy properties if available
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []
        is_content_word = False
        is_conjunction = False

        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            is_content_word = spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']
            is_conjunction = spacy_token.pos_ == 'CCONJ' or spacy_token.text.lower() in ['and', 'or', 'but']

        # Content words attend strongly to [CLS]
        if is_content_word and i <= 4:  # Early content words
            attention[i, 0] = 0.6
            # Distribute remaining attention
            remaining = 0.4
            for j in range(1, n):
                if j != i:
                    attention[i, j] = remaining / (n - 1)

        # Coordination patterns - tokens after "and" attend to "and"
        elif i > 0 and i < n - 1:
            prev_spacy = token_to_spacy[i-1] if i-1 < len(token_to_spacy) else []
            if prev_spacy and doc[prev_spacy[0]].text.lower() == 'and':
                attention[i, i-1] = 0.4
                attention[i, 0] = 0.3  # Also some [CLS] attention
                # Distribute rest
                remaining = 0.3
                for j in range(n):
                    if j != i and j != i-1 and j != 0:
                        attention[i, j] = remaining / max(1, n - 3)

            # Tokens near end of sentence
            elif i >= n - 3:
                # Attend to content words and punctuation
                total_weight = 0
                for j in range(n):
                    if j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_j = doc[token_to_spacy[j][0]]
                        if spacy_j.pos_ in ['NOUN', 'VERB'] or tokens[j] in ['.', ',']:
                            attention[i, j] = 0.15
                            total_weight += 0.15

                # Normalize and add some distributed attention
                if total_weight > 0:
                    scale = min(1.0, 0.8 / total_weight)
                    for j in range(n):
                        attention[i, j] *= scale

                # Fill remaining with uniform
                current_sum = attention[i].sum()
                remaining = 1.0 - current_sum
                if remaining > 0:
                    for j in range(n):
                        attention[i, j] += remaining / n

            else:
                # Default case: some self-attention and distributed
                attention[i, i] = 0.2
                attention[i, 0] = 0.2  # Some [CLS] attention
                remaining = 0.6
                for j in range(n):
                    if j != i and j != 0:
                        attention[i, j] = remaining / max(1, n - 2)

        else:
            # Fallback: uniform distribution
            for j in range(n):
                attention[i, j] = 1.0 / n

    return "program_L1H5", make_row_stochastic(attention)



def program_L1H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """CLS-focused attention head - most tokens attend strongly to the [CLS] token."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Find [CLS] token position (should be 0 for BERT)
    cls_pos = None
    for i, token in enumerate(tokens):
        if token.strip() == "[CLS]":
            cls_pos = i
            break

    if cls_pos is not None:
        # Most tokens attend very strongly to [CLS]
        for i in range(n):
            if i == cls_pos:
                # [CLS] attends to itself with high weight
                attention[i, cls_pos] = 0.98
            else:
                # Special case: token[1] (first real token after [CLS]) has different pattern
                if i == 1:
                    # Token[1] attends very strongly to [CLS], minimal self-attention
                    attention[i, cls_pos] = 0.97
                    attention[i, i] = 0.01
                else:
                    # Other tokens attend to [CLS] with very high weight
                    attention[i, cls_pos] = 0.95

                    # Add tiny amounts of self-attention for punctuation and special tokens
                    token = tokens[i].strip()
                    if token in [".", ",", "!", "?", "[SEP]"]:
                        attention[i, i] = 0.02
                    else:
                        # Very small residual attention to nearby tokens
                        if i > 0:
                            attention[i, i-1] = 0.01
                        if i < n-1:
                            attention[i, i+1] = 0.01
                        attention[i, i] = 0.01
    else:
        # Fallback if no [CLS] found - attend to first token
        for i in range(n):
            attention[i, 0] = 0.95
            attention[i, i] = 0.03

    # Normalize rows to sum to 1
    attention = make_row_stochastic(attention)

    return "program_L1H6", attention



def program_L1H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Structural attention head focusing on sentence boundaries, conjunctions, and global [CLS] aggregation."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong baseline attention to [CLS] token (position 0)
        if i > 0:  # Non-[CLS] tokens attend to [CLS]
            attention[i, 0] = 0.6
        else:  # [CLS] attends strongly to itself - INCREASED
            attention[i, 0] = 0.95  # Increased from 0.85

        # Self-attention component - REDUCED for non-[CLS] tokens
        if i == 0:  # [CLS] token keeps normal self-attention
            attention[i, i] = 0.1
        else:  # Non-[CLS] tokens get minimal self-attention
            attention[i, i] = 0.02  # Reduced from 0.1

        # Check if current token is punctuation
        is_punct = tokens[i].strip() in '.,!?;:'

        if is_punct:
            # Punctuation gets strong self-attention
            attention[i, i] = 0.2

        # Find conjunctions and important structural words
        spacy_indices = token_to_spacy[i]
        is_conjunction = False
        is_prep = False

        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ == 'CCONJ' or spacy_token.text.lower() in ['and', 'but', 'or']:
                    is_conjunction = True
                elif spacy_token.pos_ == 'ADP':  # Prepositions
                    is_prep = True

        # Distribute remaining attention
        remaining_mass = 1.0 - attention[i].sum()

        if is_conjunction:
            # Conjunctions get extra self-attention
            attention[i, i] += 0.05
            remaining_mass -= 0.05

        # Look for nearby conjunctions to attend to
        for j in range(n):
            if i != j and j != 0:  # Skip self and [CLS]
                spacy_j_indices = token_to_spacy[j]
                j_is_conjunction = False
                j_is_prep = False

                for spacy_idx in spacy_j_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ == 'CCONJ' or spacy_token.text.lower() in ['and', 'but', 'or']:
                            j_is_conjunction = True
                        elif spacy_token.pos_ == 'ADP':
                            j_is_prep = True

                # Attend to conjunctions
                if j_is_conjunction:
                    attention[i, j] += 0.08

                # Attend to prepositions with moderate strength
                if j_is_prep:
                    attention[i, j] += 0.04

                # Positional bias - attend to nearby tokens
                distance = abs(i - j)
                if distance <= 3:
                    attention[i, j] += 0.02 * (4 - distance) / 4

        # Add small amounts of distributed attention to all positions
        uniform_attention = remaining_mass * 0.1 / n
        attention[i] += uniform_attention

        # Boost attention from later tokens to earlier important tokens
        if i > n // 2:  # Later tokens
            for j in range(min(i, 5)):  # First few tokens
                if j != 0:  # Not [CLS], already handled
                    attention[i, j] += 0.02

    return "program_L1H7", make_row_stochastic(attention)



def program_L1H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Mixed head: strong first-token attention with local syntactic patterns and punctuation anchoring."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong self-attention for special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.5
            if tokens[i] == '[CLS]':
                attention[i, i] = 0.8  # Very strong self-attention for [CLS]

        # Strong attention to [CLS] from most tokens (but reduce strength)
        if i > 0:  # Not from [CLS] itself
            attention[i, 0] = 0.25  # Reduced from 0.4
            # Even stronger for certain word types
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_tok = doc[spacy_indices[0]]
                if spacy_tok.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    attention[i, 0] = 0.35  # Reduced from 0.6

        # Enhanced punctuation patterns with strong self-attention
        if tokens[i] in [',', '.', '"', "'", '!', '?']:
            # Strong self-attention for punctuation
            attention[i, i] = 0.15
            # Attend to nearby content words
            for j in range(max(0, i-5), min(n, i+3)):
                if j != i:
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_tok = doc[spacy_indices[0]]
                        if spacy_tok.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                            distance = abs(i - j)
                            attention[i, j] = 0.3 / (1 + distance * 0.5)

        # Pronoun and determiner reference patterns
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            if spacy_tok.pos_ in ['PRON', 'DET']:
                # Look for potential referents (nouns/proper nouns)
                for j in range(n):
                    if j != i:
                        ref_spacy = token_to_spacy[j]
                        if ref_spacy:
                            ref_tok = doc[ref_spacy[0]]
                            if ref_tok.pos_ in ['NOUN', 'PROPN']:
                                # Distance decay
                                distance = abs(i - j)
                                attention[i, j] = 0.2 / (1 + distance * 0.3)

        # Coordination patterns
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            if spacy_tok.text.lower() == 'and' or spacy_tok.pos_ == 'CCONJ':
                # Attend to coordinated elements
                for j in range(max(0, i-3), min(n, i+4)):
                    if j != i:
                        coord_spacy = token_to_spacy[j]
                        if coord_spacy:
                            coord_tok = doc[coord_spacy[0]]
                            if coord_tok.pos_ in ['NOUN', 'VERB', 'ADJ']:
                                attention[i, j] = 0.15

        # Enhanced semantic similarity attention for content words
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            if spacy_tok.pos_ in ['VERB', 'NOUN', 'ADJ']:
                for j in range(n):
                    if j != i and j != 0:  # Skip self and [CLS]
                        ref_spacy = token_to_spacy[j]
                        if ref_spacy:
                            ref_tok = doc[ref_spacy[0]]
                            if ref_tok.pos_ in ['VERB', 'NOUN', 'ADJ', 'PRON']:
                                # Use embedding similarity
                                sim = embedding_similarity(tokens, i, j)
                                if sim > 0.3:  # Threshold for semantic relatedness
                                    distance = abs(i - j)
                                    attention[i, j] += 0.15 * sim / (1 + distance * 0.2)

        # Local positional attention (attend to previous token)
        if i > 0:
            attention[i, i-1] += 0.1

        # Attention from sentence-final tokens to key content
        if i == n-1 or (i < n-1 and tokens[i] in ['.', '!', '?']):
            for j in range(n):
                if j != i:
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_tok = doc[spacy_indices[0]]
                        if spacy_tok.pos_ in ['VERB', 'NOUN']:
                            attention[i, j] += 0.1

    # Add small uniform attention everywhere to prevent zero rows
    attention += 0.01

    return "program_L1H8", make_row_stochastic(attention)



def program_L1H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Head that emphasizes sentence boundaries and strong content word attention to [CLS]."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L1H9", np.array([])

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        for j in range(n):
            score = 0.0

            # Base uniform attention
            score += 0.01

            # Strong attention to [CLS] from content words
            if j == 0 and tokens[j] == '[CLS]':
                # Get spacy info for token i
                spacy_indices = token_to_spacy[i]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                        score += 0.4
                    elif spacy_token.pos_ in ['DET', 'ADP', 'ADV']:
                        score += 0.2

            # Additional targeted boost: most content words should attend strongly to [CLS]
            if j == 0 and tokens[j] == '[CLS]':
                spacy_indices = token_to_spacy[i]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    # Boost for all content-bearing tokens that aren't already covered
                    if spacy_token.pos_ in ['PRON', 'AUX', 'CCONJ'] or spacy_token.is_alpha:
                        score += 0.3
                    # Extra boost for function words that often attend strongly to CLS
                    if spacy_token.pos_ in ['ADP', 'DET', 'PART']:
                        score += 0.2

            # [SEP] token attends to various content throughout sentence
            if i == n-1 and tokens[i] == '[SEP]':
                spacy_indices = token_to_spacy[j]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                        score += 0.08
                    elif tokens[j] in [',', '.', 'and']:
                        score += 0.06

            # Self-attention for special tokens and punctuation
            if i == j:
                if tokens[i] in ['[CLS]', '[SEP]', '.', ',']:
                    score += 0.08
                else:
                    score += 0.02

            # Punctuation attends to nearby content
            if tokens[i] in ['.', ',']:
                spacy_indices_j = token_to_spacy[j]
                if spacy_indices_j:
                    spacy_token_j = doc[spacy_indices_j[0]]
                    if spacy_token_j.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        # Distance decay
                        distance = abs(i - j)
                        if distance <= 3:
                            score += 0.06 * (4 - distance) / 4

            # Content words attend to other content words with similarity boost
            spacy_indices_i = token_to_spacy[i]
            spacy_indices_j = token_to_spacy[j]
            if spacy_indices_i and spacy_indices_j:
                spacy_i = doc[spacy_indices_i[0]]
                spacy_j = doc[spacy_indices_j[0]]

                if (spacy_i.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and 
                    spacy_j.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']):

                    # Base content-to-content attention
                    score += 0.02

                    # Similarity boost
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        score += 0.03 * sim

            # Verbs attend to auxiliary/modal verbs
            if spacy_indices_i and spacy_indices_j:
                spacy_i = doc[spacy_indices_i[0]]
                spacy_j = doc[spacy_indices_j[0]]

                if (spacy_i.pos_ in ['VERB'] and 
                    spacy_j.pos_ in ['VERB', 'AUX']):
                    score += 0.04

            # Articles and determiners attend to nearby nouns
            if spacy_indices_i and spacy_indices_j:
                spacy_i = doc[spacy_indices_i[0]]
                spacy_j = doc[spacy_indices_j[0]]

                if (spacy_i.pos_ == 'DET' and 
                    spacy_j.pos_ in ['NOUN', 'PROPN'] and
                    abs(i - j) <= 3):
                    score += 0.05

            attention[i, j] = score

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L1H9", attention



def program_L2H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Special token attention head: [CLS] attends to self, [SEP] attends to [CLS]."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.ones((n, n)) / n  # Start with uniform distribution

    # Find [CLS] and [SEP] positions
    cls_pos = None
    sep_pos = None

    for i, token in enumerate(tokens):
        if token.strip() == '[CLS]':
            cls_pos = i
        elif token.strip() == '[SEP]':
            sep_pos = i

    # Apply the observed patterns
    if cls_pos is not None:
        # [CLS] attends strongly to itself
        attention_matrix[cls_pos] = 0.001  # Small uniform base
        attention_matrix[cls_pos, cls_pos] = 0.98

    if sep_pos is not None and cls_pos is not None:
        # [SEP] attends strongly to [CLS] and weakly to itself
        attention_matrix[sep_pos] = 0.001  # Small uniform base
        attention_matrix[sep_pos, cls_pos] = 0.98
        attention_matrix[sep_pos, sep_pos] = 0.02

    return "program_L2H0", make_row_stochastic(attention_matrix)



def program_L2H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head combining [CLS] focus with local syntactic dependencies, including strong function word to [CLS] attention."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L2H1", np.array([])

    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    # Base uniform attention
    attention += 0.01

    for i in range(n):
        # Strong self-attention for special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] += 0.5

        # Very strong [SEP] to [CLS] attention
        if tokens[i] == '[SEP]' and tokens[0] == '[CLS]':
            attention[i, 0] += 0.6

        # Strong [CLS] attention for content words, especially early ones
        if i > 0 and tokens[0] == '[CLS]':
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'PRON', 'ADJ', 'ADV']:
                    # Higher weight for earlier positions
                    cls_weight = max(0.3, 0.8 - 0.05 * i)
                    attention[i, 0] += cls_weight
                # Strong [CLS] attention for function words too
                elif spacy_token.pos_ in ['ADP', 'CCONJ', 'DET', 'AUX']:
                    cls_weight = max(0.4, 0.7 - 0.03 * i)
                    attention[i, 0] += cls_weight

        # Syntactic dependencies
        if token_to_spacy[i]:
            spacy_token = doc[token_to_spacy[i][0]]

            # Head-dependent relationships
            if spacy_token.head != spacy_token:
                head_char_start = spacy_token.head.idx
                # Find corresponding token
                for j in range(n):
                    if token_to_spacy[j]:
                        spacy_j = doc[token_to_spacy[j][0]]
                        if spacy_j.idx == head_char_start:
                            # Strong attention for key dependencies
                            if spacy_token.dep_ in ['nsubj', 'dobj', 'pobj', 'advmod', 'amod', 'det']:
                                attention[i, j] += 0.4
                            else:
                                attention[i, j] += 0.2
                            break

            # Special patterns
            # Negation to auxiliary
            if spacy_token.dep_ == 'neg' and spacy_token.head.pos_ == 'AUX':
                head_char_start = spacy_token.head.idx
                for j in range(n):
                    if token_to_spacy[j] and doc[token_to_spacy[j][0]].idx == head_char_start:
                        attention[i, j] += 0.6
                        break

            # Determiners and possessives to their nouns
            if spacy_token.pos_ in ['DET', 'PRON'] and spacy_token.dep_ in ['det', 'poss']:
                head_char_start = spacy_token.head.idx
                for j in range(n):
                    if token_to_spacy[j] and doc[token_to_spacy[j][0]].idx == head_char_start:
                        attention[i, j] += 0.3
                        break

        # Adjacent token attention for certain patterns
        if i > 0:
            # Previous token attention with some weight
            attention[i, i-1] += 0.05

            # Preposition phrases
            if token_to_spacy[i]:
                spacy_token = doc[token_to_spacy[i][0]]
                if spacy_token.pos_ == 'ADP':  # Preposition
                    if i > 1:
                        attention[i, i-2] += 0.1  # Skip over determiner

        # Punctuation patterns
        if tokens[i] in [',', '.']:
            # Punctuation attends to nearby content
            for j in range(max(0, i-3), i):
                if token_to_spacy[j]:
                    spacy_j = doc[token_to_spacy[j][0]]
                    if spacy_j.pos_ in ['NOUN', 'VERB']:
                        attention[i, j] += 0.1

    return "program_L2H1", make_row_stochastic(attention)



def program_L2H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """First-token attention combined with semantic relationship detection, especially entity-pronoun coreference."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L2H10", np.array([])

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong baseline attention to first token ([CLS])
        attention[i, 0] = 0.4

        # Self-attention
        attention[i, i] = 0.1

        # Get spacy tokens aligned to current position
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        for j in range(n):
            if i == j or j == 0:  # Skip self and first token (already handled)
                continue

            target_spacy_indices = token_to_spacy[j] 
            target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

            # Semantic similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.3:
                attention[i, j] += sim * 0.2

            if current_spacy and target_spacy:
                # Pronoun to entity/noun relationships
                if current_spacy.pos_ == "PRON" and target_spacy.pos_ in ["NOUN", "PROPN"]:
                    attention[i, j] += 0.3

                # Noun to modifier relationships  
                if current_spacy.pos_ in ["NOUN", "PROPN"]:
                    if target_spacy.dep_ == "amod" and target_spacy.head == current_spacy:
                        attention[i, j] += 0.15
                    if current_spacy.dep_ == "amod" and current_spacy.head == target_spacy:
                        attention[i, j] += 0.15

                # Verb relationships
                if current_spacy.pos_ == "VERB" and target_spacy.pos_ in ["NOUN", "PROPN"]:
                    if target_spacy.dep_ in ["nsubj", "dobj"]:
                        attention[i, j] += 0.1

                # Dependency-based relationships
                if current_spacy.head == target_spacy or target_spacy.head == current_spacy:
                    attention[i, j] += 0.08

            # Punctuation gets some attention
            if tokens[j] in [",", ".", "?", "!", ";"]:
                attention[i, j] += 0.05

            # Recency bias - slight preference for recent tokens
            if j < i and i - j <= 3:
                attention[i, j] += 0.02 * (4 - (i - j)) / 4

    return "program_L2H10", make_row_stochastic(attention)



def program_L2H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic-semantic attention head with [CLS] focus, dependency connections, and strong sentence boundary attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Strong self-attention for [CLS] token
        if tokens[i].strip() == '[CLS]':
            attention[i, i] = 0.95
            continue

        # Special handling for sentence boundary tokens - much stronger attention to [CLS]
        if tokens[i].strip() == '[SEP]':
            attention[i, 0] = 0.85 if tokens[0].strip() == '[CLS]' else 0.0
            attention[i, i] = 0.05
            continue
        elif tokens[i].strip() in ['.', '!', '?']:
            # Check if this is sentence-final punctuation by looking at position
            if i >= n - 3:  # Near end of sequence (accounting for [SEP])
                attention[i, 0] = 0.6 if tokens[0].strip() == '[CLS]' else 0.0
            else:
                attention[i, 0] = 0.3 if tokens[0].strip() == '[CLS]' else 0.0
            attention[i, i] = 0.05
        else:
            # High attention to [CLS] from all other tokens
            attention[i, 0] = 0.3 if tokens[0].strip() == '[CLS]' else 0.0
            # Self-attention baseline
            attention[i, i] = 0.05

        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            continue

        spacy_token = doc[spacy_indices[0]]

        # Syntactic dependencies - attend to head and children
        if spacy_token.head != spacy_token:  # Has a head
            head_token_indices = spacy_to_token[spacy_token.head.i]
            for j in head_token_indices:
                if j < n:
                    attention[i, j] += 0.15

        # Attend to syntactic children
        for child in spacy_token.children:
            child_token_indices = spacy_to_token[child.i]
            for j in child_token_indices:
                if j < n:
                    attention[i, j] += 0.1

        # Positional attention - slight preference for recent tokens
        for j in range(max(0, i-3), i):
            attention[i, j] += 0.03

        # Semantic similarity attention
        for j in range(n):
            if i != j and j != 0:  # Skip self and [CLS]
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity threshold
                    attention[i, j] += 0.08
                elif sim > 0.5:  # Medium similarity
                    attention[i, j] += 0.03

        # Special attention patterns for verbs
        if spacy_indices and doc[spacy_indices[0]].pos_ == 'VERB':
            # Verbs attend to their subjects and objects
            for child in doc[spacy_indices[0]].children:
                if child.dep_ in ['nsubj', 'dobj', 'iobj']:
                    child_token_indices = spacy_to_token[child.i]
                    for j in child_token_indices:
                        if j < n:
                            attention[i, j] += 0.12

        # Nouns attend to their modifiers
        if spacy_indices and doc[spacy_indices[0]].pos_ == 'NOUN':
            for child in doc[spacy_indices[0]].children:
                if child.dep_ in ['amod', 'det']:
                    child_token_indices = spacy_to_token[child.i]
                    for j in child_token_indices:
                        if j < n:
                            attention[i, j] += 0.08

    return "program_L2H11", make_row_stochastic(attention)



def program_L2H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that primarily attends to [CLS] with strong content word focus."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L2H2", np.array([])

    attention = np.zeros((n, n))
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Strong attention to [CLS] token (position 0)
        if i > 0:  # Don't attend from [CLS] to itself via this mechanism
            # Check if this is a content word that should have very high [CLS] attention
            is_content_word = False
            if i < len(token_to_spacy) and token_to_spacy[i]:
                spacy_idx = token_to_spacy[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    # Content words: verbs, nouns, adjectives, adverbs
                    if spacy_token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV', 'PROPN']:
                        is_content_word = True

            if is_content_word:
                attention[i, 0] = 0.85  # Much higher for content words
            else:
                attention[i, 0] = 0.6

        # Self-attention - reduce for content words
        if i > 0 and i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV', 'PROPN']:
                    attention[i, i] = 0.05  # Much lower self-attention for content words
                else:
                    attention[i, i] = 0.2
            else:
                attention[i, i] = 0.2
        else:
            attention[i, i] = 0.2

        # Special handling for [CLS] token
        if i == 0:
            attention[i, i] = 0.7

        # Special handling for sentence-final tokens ([SEP], punctuation)
        if token in ['[SEP]', '.', '!', '?'] or i == n - 1:
            # Distribute attention more evenly across the sequence
            for j in range(n):
                if j != i:
                    # Distance-based decay
                    distance = abs(i - j)
                    weight = 1.0 / (1.0 + distance * 0.1)
                    attention[i, j] += 0.3 * weight

        # Add some positional bias for nearby tokens
        for j in range(max(0, i-3), min(n, i+4)):
            if j != i and j != 0:
                distance = abs(i - j)
                weight = 0.1 / (1.0 + distance)
                attention[i, j] += weight

        # Add attention based on syntactic relationships if available
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Attend to syntactic head
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    if head_idx < len(token_to_spacy):
                        for target_token_idx in range(n):
                            if target_token_idx < len(token_to_spacy) and token_to_spacy[target_token_idx]:
                                if head_idx in [doc[si].i for si in token_to_spacy[target_token_idx] if si < len(doc)]:
                                    attention[i, target_token_idx] += 0.1

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L2H2", attention



def program_L2H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that combines [CLS] aggregation with local content word binding."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L2H3", np.array([])

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens aligned to this LM token
        spacy_indices = token_to_spacy[i]
        spacy_tokens = [doc[idx] for idx in spacy_indices] if spacy_indices else []

        # Check if this token is punctuation
        is_punct = any(tok.is_punct for tok in spacy_tokens) or tokens[i].strip() in '.,!?;:"'

        # Check if this is [CLS] or [SEP]
        is_cls = i == 0 and tokens[i] in ['[CLS]', '<s>']
        is_sep = tokens[i] in ['[SEP]', '</s>']

        # Check if this is a content word
        is_content = any(tok.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] for tok in spacy_tokens)

        if is_cls:
            # [CLS] attends strongly to itself
            attention[i, 0] = 1.0

        elif is_sep:
            # [SEP] should behave like content words - strong [CLS] attention
            attention[i, 0] = 0.8
            # Some self attention
            attention[i, i] = 0.1
            # Minor attention to content words
            for j in range(n-1):
                j_spacy = token_to_spacy[j]
                if j_spacy and any(doc[idx].pos_ in ['NOUN', 'VERB'] for idx in j_spacy):
                    attention[i, j] = 0.05

        elif is_punct:
            # Punctuation has mixed behavior
            # Some self-attention
            attention[i, i] = 0.3

            # Attend to [CLS] 
            attention[i, 0] = 0.2

            # Enhanced attention to previous content words (not just nearby)
            for j in range(i):
                if j == 0:
                    continue
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_tokens = [doc[idx] for idx in j_spacy]
                    if any(tok.pos_ in ['NOUN', 'VERB', 'ADJ'] for tok in j_tokens):
                        # Distance decay but allow longer range
                        dist = abs(i - j)
                        weight = 0.4 / (1 + 0.3 * dist)
                        attention[i, j] = weight

            # Keep nearby attention for other tokens
            for j in range(max(0, i-3), min(n, i+2)):
                if j == i or j == 0:
                    continue
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_tokens = [doc[idx] for idx in j_spacy]
                    if any(tok.pos_ in ['NOUN', 'VERB', 'ADJ'] for tok in j_tokens):
                        # Only add if not already covered by previous content logic
                        if j >= i:
                            dist = abs(i - j)
                            weight = 0.4 / (1 + dist)
                            attention[i, j] = weight

        elif is_content:
            # Content words show strong [CLS] attention
            attention[i, 0] = 0.6

            # Self attention
            attention[i, i] = 0.1

            # Attend to other content words based on similarity and position
            for j in range(n):
                if j == i or j == 0:
                    continue

                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_tokens = [doc[idx] for idx in j_spacy]
                    if any(tok.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] for tok in j_tokens):
                        # Use embedding similarity
                        sim = embedding_similarity(tokens, i, j)
                        # Position bias - prefer earlier tokens
                        pos_bias = 1.0 if j < i else 0.7
                        # Distance decay
                        dist = abs(i - j)
                        dist_decay = 1.0 / (1 + 0.1 * dist)

                        weight = max(0, sim * 0.3 + 0.1) * pos_bias * dist_decay
                        attention[i, j] = weight

        else:
            # Function words and other tokens
            # Strong [CLS] attention
            attention[i, 0] = 0.8

            # Some local attention
            for j in range(max(0, i-2), min(n, i+2)):
                if j != i and j != 0:
                    attention[i, j] = 0.1 / (1 + abs(i - j))

    return "program_L2H3", make_row_stochastic(attention)



def program_L2H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic structure head with first-token bias and clause boundary awareness."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L2H4", np.array([])

    attention = np.zeros((n, n))
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong baseline attention to [CLS] token
        if tokens[0] in ['[CLS]', '<s>', '<bos>']:
            attention[i, 0] = 0.4

        # Self-attention for punctuation and special tokens
        token_text = tokens[i].strip()
        if token_text in ['.', ',', '!', '?', '"', "'", '[SEP]', '</s>', '<eos>'] or tokens[i] == '[CLS]':
            attention[i, i] = 0.3

        # Get spacy tokens aligned to current LM token
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            continue

        current_spacy = doc[spacy_indices[0]]

        # Attend to syntactic head
        if current_spacy.head != current_spacy and current_spacy.head.i < len(doc):
            head_spacy_idx = current_spacy.head.i
            # Find LM tokens that align to this spacy head
            for j in range(n):
                j_spacy_indices = token_to_spacy[j]
                if head_spacy_idx in j_spacy_indices:
                    attention[i, j] += 0.2

        # Attend to syntactic children
        for child in current_spacy.children:
            if child.i < len(doc):
                child_spacy_idx = child.i
                for j in range(n):
                    j_spacy_indices = token_to_spacy[j]
                    if child_spacy_idx in j_spacy_indices:
                        attention[i, j] += 0.15

        # Special attention patterns for punctuation
        if token_text in ['.', ',', '!', '?']:
            # Period/comma attends to clause content
            for j in range(max(0, i-8), i):
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = doc[j_spacy_indices[0]]
                    if j_spacy.pos_ in ['VERB', 'NOUN', 'ADJ']:
                        attention[i, j] += 0.1

        # Verb-argument relationships
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ == 'VERB':
                # Verb attends to its arguments
                for j in range(n):
                    j_spacy_indices = token_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy = doc[j_spacy_indices[0]]
                        if j_spacy.dep_ in ['dobj', 'iobj', 'nsubj', 'pobj']:
                            attention[i, j] += 0.1

            # Objects attend to their verbs
            if spacy_token.dep_ in ['dobj', 'iobj', 'pobj']:
                for j in range(n):
                    j_spacy_indices = token_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy = doc[j_spacy_indices[0]]
                        if j_spacy.pos_ == 'VERB':
                            attention[i, j] += 0.15

        # Prepositions attend to their objects
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ == 'ADP':  # preposition
                for j in range(i+1, min(n, i+4)):
                    j_spacy_indices = token_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy = doc[j_spacy_indices[0]]
                        if j_spacy.pos_ in ['NOUN', 'PRON']:
                            attention[i, j] += 0.2

        # Add some local attention for content words
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                for j in range(max(0, i-3), min(n, i+4)):
                    if i != j:
                        attention[i, j] += 0.05

        # Small uniform baseline
        for j in range(n):
            attention[i, j] += 0.01

    return "program_L2H4", make_row_stochastic(attention)



def program_L2H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content word attention head with strong [CLS] focus, syntactic awareness, and [SEP] → [CLS] pattern."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Define content word POS tags
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

    for i in range(n):
        # Special handling for [CLS] token
        if tokens[i] == '[CLS]':
            attention[i, i] = 1.0
            continue

        # Special handling for [SEP] token - strong attention to [CLS]
        if tokens[i] == '[SEP]':
            for j in range(n):
                if tokens[j] == '[CLS]':
                    attention[i, j] = 0.6  # Much stronger than original 0.131-0.241
                elif j == i:
                    attention[i, j] = 0.1
                else:
                    attention[i, j] = 0.02
            continue

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i]
        current_pos = None
        current_dep = None
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            current_pos = spacy_tok.pos_
            current_dep = spacy_tok.dep_

        for j in range(n):
            if i == j:
                # Self-attention baseline
                attention[i, j] = 0.1
            elif tokens[j] == '[CLS]':
                # Strong attention to [CLS] from content words
                if current_pos in content_pos:
                    attention[i, j] = 0.4
                elif current_pos in {'PUNCT', 'SPACE'}:
                    attention[i, j] = 0.2
                else:
                    attention[i, j] = 0.1
            else:
                # Get spacy info for target token
                target_spacy_indices = token_to_spacy[j]
                target_pos = None
                target_dep = None
                if target_spacy_indices:
                    target_spacy_tok = doc[target_spacy_indices[0]]
                    target_pos = target_spacy_tok.pos_
                    target_dep = target_spacy_tok.dep_

                weight = 0.02  # Base weight

                # Higher attention to content words
                if target_pos in content_pos:
                    weight += 0.06

                # Boost for semantic similarity
                if j < len(tokens) and i < len(tokens):
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        weight += 0.04

                # Syntactic relationships
                if spacy_indices and target_spacy_indices:
                    current_spacy = doc[spacy_indices[0]]
                    target_spacy = doc[target_spacy_indices[0]]

                    # Head-dependent relationships
                    if target_spacy == current_spacy.head or current_spacy == target_spacy.head:
                        weight += 0.03

                # Special patterns for punctuation and end tokens
                if tokens[i] in {'.', ',', '[SEP]'}:
                    if target_pos in content_pos:
                        weight += 0.08

                # Distance decay for long sentences
                dist = abs(i - j)
                if dist > 5:
                    weight *= 0.7

                attention[i, j] = weight

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)
    return "program_L2H5", attention



def program_L2H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Self-attention head with enhanced self-attention and broader semantic similarity matching."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    for i in range(n):
        token_i = tokens[i].strip()

        # Enhanced self-attention baseline - much stronger for content tokens
        if token_i in ['[CLS]', '[SEP]', '.', ',', '!', '?', ';', ':']:
            attention[i, i] = 0.6
        else:
            attention[i, i] = 1.2  # Boost self-attention for content tokens

        # Special handling for special tokens and punctuation
        if token_i in ['[CLS]', '[SEP]', '.', ',', '!', '?', ';', ':']:
            # These tokens attend strongly to [CLS]
            if i > 0:  # Don't override [CLS] self-attention
                attention[i, 0] = 0.3

            # [SEP] and punctuation also attend to some content words
            if token_i in ['[SEP]', '.']:
                for j in range(n):
                    if j != i and j != 0:
                        token_j = tokens[j].strip()
                        if token_j not in ['[CLS]', '[SEP]', '.', ',', '!', '?', ';', ':']:
                            attention[i, j] = 0.05

        else:
            # Content tokens: attend based on semantic similarity and position
            for j in range(n):
                if j != i:
                    token_j = tokens[j].strip()

                    # Strong attention to [CLS] for many tokens
                    if j == 0:
                        attention[i, j] = 0.1

                    # High attention to semantically similar tokens - lowered thresholds
                    elif token_j not in ['[CLS]', '[SEP]']:
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.6:  # Lowered from 0.8 - more similar tokens
                            attention[i, j] = 0.4
                        elif sim > 0.3:  # Lowered from 0.5 - catch more relationships
                            attention[i, j] = 0.1
                        else:
                            # Small baseline attention with recency bias
                            attention[i, j] = 0.02 * (1.0 / (abs(i - j) + 1))

    # Normalize to make row stochastic
    attention = make_row_stochastic(attention)

    return "program_L2H6", attention



def program_L2H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic association head with strong [CLS] focus: connects tokens to [CLS] and related semantic content."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong self-attention for [CLS]
        if tokens[i] == '[CLS]':
            attention[i, i] = 1.0
            continue

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i]
        current_pos = None
        current_dep = None
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            current_pos = spacy_token.pos_
            current_dep = spacy_token.dep_

        # Special case: [SEP] tokens have extremely high attention to [CLS]
        if tokens[i] == '[SEP]':
            if '[CLS]' in tokens:
                cls_idx = tokens.index('[CLS]')
                attention[i, cls_idx] = 0.9
            continue

        # Enhanced [CLS] attention based on token type
        if '[CLS]' in tokens:
            cls_idx = tokens.index('[CLS]')
            # Punctuation gets very high [CLS] attention
            if tokens[i] in ['.', ',', '!', '?', ';', ':']:
                attention[i, cls_idx] = 0.6
            # Content words get higher [CLS] attention
            elif current_pos in ['NOUN', 'VERB', 'ADJ', 'PRON']:
                attention[i, cls_idx] = 0.5
            # Function words get moderate [CLS] attention  
            else:
                attention[i, cls_idx] = 0.4

        # For nouns and pronouns, attend strongly to semantically related tokens
        if current_pos in ['NOUN', 'PRON']:
            for j in range(n):
                if i == j:
                    continue

                # Get target token info
                target_spacy = token_to_spacy[j]
                if target_spacy:
                    target_token = doc[target_spacy[0]]
                    target_pos = target_token.pos_

                    # High attention to verbs
                    if target_pos == 'VERB':
                        attention[i, j] = 0.4
                    # Moderate attention to other nouns if semantically similar
                    elif target_pos == 'NOUN':
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:
                            attention[i, j] = 0.3 * sim

        # For verbs, moderate attention to [CLS] and related nouns
        elif current_pos == 'VERB':
            for j in range(n):
                if i == j:
                    continue
                target_spacy = token_to_spacy[j]
                if target_spacy:
                    target_token = doc[target_spacy[0]]
                    if target_token.pos_ == 'NOUN':
                        attention[i, j] = 0.2

        # Punctuation attends to nearby content words
        elif tokens[i] in ['.', ',', '!', '?']:
            for j in range(max(0, i-5), i):
                target_spacy = token_to_spacy[j]
                if target_spacy:
                    target_token = doc[target_spacy[0]]
                    if target_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        attention[i, j] = 0.3 / (i - j)

        # Add some uniform attention to all tokens as baseline
        attention[i, :] += 0.05

        # Self-attention for some token types
        if current_pos in ['ADP', 'DET', 'PUNCT']:
            attention[i, i] += 0.1

    return "program_L2H7", make_row_stochastic(attention)



def program_L2H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Strong [CLS] aggregation head where content words attend heavily to [CLS] with reduced self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong baseline attention to [CLS] token
        if tokens[0] in ['[CLS]', '<s>']:  # Handle different model types
            attention[i, 0] = 0.3

        # Self-attention
        attention[i, i] = 0.2

        # Special handling for special tokens
        if tokens[i] in ['[CLS]', '<s>']:
            attention[i, i] = 0.8  # Strong self-attention for [CLS]
            continue
        elif tokens[i] in ['[SEP]', '</s>']:
            # [SEP] has very strong attention to [CLS] - boost significantly
            attention[i, 0] = 0.7  # Much higher than default 0.3
            attention[i, i] = 0.6  # Strong self-attention for [SEP]
            continue
        elif tokens[i] in ['.', ',', '"', '!', '?']:
            # Punctuation attends to content words and [CLS]
            for j in range(n):
                if j != i and tokens[j] not in ['[CLS]', '[SEP]', '<s>', '</s>']:
                    # Get spacy info for target token
                    spacy_indices_j = token_to_spacy[j]
                    if spacy_indices_j:
                        spacy_token_j = doc[spacy_indices_j[0]]
                        if spacy_token_j.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            attention[i, j] += 0.1
            continue

        # Get spacy information for current token
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            continue

        current_spacy = doc[spacy_indices[0]]

        # MAJOR CHANGE: Content words get much stronger [CLS] attention and weaker self-attention
        if current_spacy.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET']:
            # Boost [CLS] attention significantly for content words
            attention[i, 0] = 0.5  # Much higher than default 0.3
            # Reduce self-attention for content words
            attention[i, i] = 0.1  # Much lower than default 0.2

        # Syntactic attention patterns
        for j in range(n):
            if i == j:
                continue

            spacy_indices_j = token_to_spacy[j]
            if not spacy_indices_j:
                continue

            target_spacy = doc[spacy_indices_j[0]]

            # Verbs attend to their subjects and objects
            if current_spacy.pos_ == 'VERB':
                if target_spacy.dep_ in ['nsubj', 'dobj', 'pobj']:
                    attention[i, j] += 0.15
                elif target_spacy.pos_ == 'NOUN':
                    attention[i, j] += 0.08

            # Nouns attend to their modifiers
            elif current_spacy.pos_ == 'NOUN':
                if target_spacy.dep_ in ['amod', 'det'] and target_spacy.head == current_spacy:
                    attention[i, j] += 0.12
                elif target_spacy.pos_ in ['ADJ', 'DET']:
                    attention[i, j] += 0.06

            # Articles and determiners attend to their head nouns
            elif current_spacy.pos_ in ['DET']:
                if target_spacy.pos_ == 'NOUN':
                    attention[i, j] += 0.1

            # Semantic similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.5:
                attention[i, j] += 0.1 * sim

            # Distance-based attention (prefer nearby tokens)
            distance = abs(i - j)
            if distance <= 3:
                attention[i, j] += 0.05 / (distance + 1)

        # Quoted speech patterns
        if '"' in tokens:
            quote_indices = [k for k, token in enumerate(tokens) if token == '"']
            if len(quote_indices) >= 2:
                # Inside quotes, attend to quote boundaries
                if i > quote_indices[0] and i < quote_indices[-1]:
                    for quote_idx in quote_indices:
                        attention[i, quote_idx] += 0.08
                # Speech attribution verbs attend to quotes
                if spacy_indices and doc[spacy_indices[0]].lemma_ in ['say', 'tell', 'speak', 'ask']:
                    for quote_idx in quote_indices:
                        attention[i, quote_idx] += 0.12

    return "program_L2H8", make_row_stochastic(attention)



def program_L2H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """SEP-to-CLS attention head with minimal self-attention for regular tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    # Initialize attention matrix with small uniform values
    attention_matrix = np.full((n, n), 0.01)

    # Find [CLS] and [SEP] positions
    cls_pos = None
    sep_pos = None

    for i, token in enumerate(tokens):
        if token.strip() == '[CLS]':
            cls_pos = i
        elif token.strip() == '[SEP]':
            sep_pos = i

    # If we found both special tokens, create the main pattern
    if cls_pos is not None and sep_pos is not None:
        # [SEP] attends very strongly to [CLS]
        attention_matrix[sep_pos, cls_pos] = 0.98

        # [CLS] attends to itself with moderate strength
        attention_matrix[cls_pos, cls_pos] = 0.80

        # Reduce other weights for these rows to maintain reasonable distributions
        for j in range(n):
            if j != cls_pos:
                attention_matrix[sep_pos, j] = 0.002
            if j != cls_pos:
                attention_matrix[cls_pos, j] = 0.01

    # All other tokens have relatively uniform low attention
    for i in range(n):
        if i != cls_pos and i != sep_pos:
            attention_matrix[i, :] = 0.05
            # Regular tokens have minimal self-attention (matching real data)
            attention_matrix[i, i] = 0.001

    return "program_L2H9", make_row_stochastic(attention_matrix)



def program_L3H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head combining first-token bias with cross-referential/identity matching."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    for i in range(n):
        query_token = tokens[i].strip()

        # Base attention distribution
        for j in range(n):
            key_token = tokens[j].strip()

            # Self-attention baseline
            if i == j:
                attention_matrix[i, j] = 0.05

            # Stronger first-token ([CLS]) attention for most tokens
            elif j == 0:
                attention_matrix[i, j] = 0.45  # Increased from 0.3

            # Special token self-attention (very high for [SEP])
            elif query_token in ['[SEP]', '[CLS]'] and i == j:
                attention_matrix[i, j] = 0.6

            # Much more restrictive cross-referential attention
            else:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.95:  # Very high similarity threshold
                    attention_matrix[i, j] = 0.4 * sim
                else:
                    attention_matrix[i, j] = 0.01  # Reduced baseline

        # Boost attention for exact token matches (pronouns, repeated words)
        for j in range(n):
            if i != j and tokens[i].strip() == tokens[j].strip():
                # Strong boost for identical tokens
                attention_matrix[i, j] = max(attention_matrix[i, j], 0.6)

    # Special handling for [SEP] tokens - they have very high self-attention
    for i in range(n):
        if tokens[i].strip() == '[SEP]':
            attention_matrix[i, :] = 0.01  # Reset row
            attention_matrix[i, i] = 0.65  # High self-attention
            attention_matrix[i, 0] = 0.1   # Some [CLS] attention

    return "program_L3H0", make_row_stochastic(attention_matrix)



def program_L3H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention to [CLS] token plus semantic similarity with positional bias and improved self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Handle special tokens with proper [SEP] -> [CLS] attention
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.4
            if tokens[i] == '[CLS]':
                attention[i, i] = 0.39
            else:  # [SEP]
                attention[i, i] = 0.49
                # Add strong attention from [SEP] to [CLS]
                attention[i, 0] = 0.35
            continue

        # Strong attention to [CLS] token from most tokens
        if i > 0:
            attention[i, 0] = 0.15

        # Get spacy info for current token
        current_spacy_tokens = token_to_spacy[i] if i < len(token_to_spacy) else []
        current_pos = None
        current_is_punct = tokens[i] in '.,;:!?"'

        if current_spacy_tokens:
            current_pos = doc[current_spacy_tokens[0]].pos_

        # Add self-attention for regular tokens based on POS and other features
        self_attention_bonus = 0.0
        if current_pos in ['VERB', 'NOUN', 'ADJ', 'DET', 'ADP']:
            self_attention_bonus = 0.06
        elif current_is_punct:
            self_attention_bonus = 0.08
        elif current_pos in ['PRON', 'ADV']:
            self_attention_bonus = 0.04

        attention[i, i] = self_attention_bonus

        for j in range(n):
            if i == j:
                continue

            # Skip if already set (CLS attention or SEP->CLS)
            if j == 0 and attention[i, j] > 0:
                continue

            # Base similarity score
            sim_score = embedding_similarity(tokens, i, j)

            # Position-based factors
            distance = abs(i - j)
            recency_bonus = 0.0
            if j < i:  # Only attend to previous tokens primarily
                recency_bonus = 0.05 / (1 + distance * 0.5)

            # Semantic similarity bonus
            semantic_bonus = 0.0
            if sim_score > 0.3:
                semantic_bonus = sim_score * 0.1

            # Special patterns for content words
            content_bonus = 0.0
            if (current_pos in ['NOUN', 'VERB', 'ADJ', 'ADV'] and 
                j < i and distance <= 6):
                target_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                if target_spacy:
                    target_pos = doc[target_spacy[0]].pos_
                    if target_pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                        content_bonus = 0.08

            # Punctuation patterns
            punct_bonus = 0.0
            if current_is_punct and j < i and distance <= 3:
                target_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                if target_spacy:
                    target_pos = doc[target_spacy[0]].pos_
                    if target_pos in ['NOUN', 'VERB']:
                        punct_bonus = 0.06

            # Combine all factors
            total_score = (0.02 + recency_bonus + semantic_bonus + 
                          content_bonus + punct_bonus)

            attention[i, j] = max(total_score, 0.01)

    # Normalize to make row stochastic
    attention = make_row_stochastic(attention)

    return "program_L3H1", attention



def program_L3H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """[CLS] aggregation head that strongly attends to [CLS] for sentence-level information gathering."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong self-attention for special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 1.0
            continue

        # Base attention distribution
        base_weight = 0.02
        attention[i, :] = base_weight

        # Much stronger attention to [CLS] token - this head is primarily about [CLS] aggregation
        cls_idx = 0 if tokens[0] == '[CLS]' else None
        if cls_idx is not None:
            # Boost [CLS] attention significantly for content words
            if token_to_spacy[i]:
                spacy_token = doc[token_to_spacy[i][0]]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    attention[i, cls_idx] = 0.45  # Much higher for important content words
                elif spacy_token.pos_ in ['PRON', 'DET', 'ADV']:
                    attention[i, cls_idx] = 0.35  # High for other content
                else:
                    attention[i, cls_idx] = 0.25  # Still significant for function words
            else:
                attention[i, cls_idx] = 0.3  # Default high attention to [CLS]

        # Self-attention
        attention[i, i] = 0.08

        # Semantic similarity based attention
        for j in range(n):
            if i != j and tokens[j] not in ['[CLS]', '[SEP]']:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High semantic similarity
                    attention[i, j] += 0.4
                elif sim > 0.5:  # Moderate semantic similarity
                    attention[i, j] += 0.2

        # Get spacy tokens for current position
        if token_to_spacy[i]:
            spacy_token = doc[token_to_spacy[i][0]]

            # Attention based on dependency relations
            for j in range(n):
                if i != j and token_to_spacy[j]:
                    target_spacy = doc[token_to_spacy[j][0]]

                    # Head-dependent relationships
                    if target_spacy.head == spacy_token or spacy_token.head == target_spacy:
                        attention[i, j] += 0.15

                    # Adjacent modifier relationships
                    if (spacy_token.pos_ in ['ADJ', 'ADV'] and 
                        target_spacy.pos_ in ['NOUN', 'VERB'] and 
                        abs(spacy_token.i - target_spacy.i) <= 2):
                        attention[i, j] += 0.1

                    # Verb-object relationships
                    if (spacy_token.pos_ == 'VERB' and 
                        target_spacy.dep_ in ['dobj', 'pobj']):
                        attention[i, j] += 0.12

        # Boost attention to previous token for certain patterns
        if i > 0:
            prev_token = tokens[i-1]
            if not prev_token.startswith('['):
                attention[i, i-1] += 0.05

        # Special handling for punctuation - attend to semantically important preceding words
        if tokens[i] in ['.', '!', '?', ',']:
            for j in range(max(0, i-5), i):
                if token_to_spacy[j]:
                    spacy_j = doc[token_to_spacy[j][0]]
                    if spacy_j.pos_ in ['VERB', 'NOUN', 'ADJ']:
                        attention[i, j] += 0.08

    # Special case: [SEP] tokens should attend to [CLS]
    sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']
    cls_idx = 0 if tokens[0] == '[CLS]' else None
    if cls_idx is not None:
        for sep_idx in sep_indices:
            attention[sep_idx, cls_idx] = 0.3
            attention[sep_idx, sep_idx] = 0.7  # Reduce self-attention proportionally

    return "program_L3H10", make_row_stochastic(attention)



def program_L3H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head with strong [CLS] self-attention and [SEP]->[CLS] bias, plus local syntax."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L3H11", np.array([])

    attention = np.zeros((n, n))
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Strong bias toward [CLS] token (position 0)
        if tokens[0] in ['[CLS]', '<s>', '<|endoftext|>']:
            attention[i, 0] = 3.0

        # Self-attention
        attention[i, i] = 0.5

        # Find spacy tokens aligned with current token
        spacy_indices = token_to_spacy[i]

        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Syntactic dependencies - attend to head and children
            if spacy_token.head != spacy_token:
                head_indices = spacy_to_token[spacy_token.head.i]
                for head_idx in head_indices:
                    if head_idx < n:
                        attention[i, head_idx] += 2.0

            for child in spacy_token.children:
                child_indices = spacy_to_token[child.i]
                for child_idx in child_indices:
                    if child_idx < n:
                        attention[i, child_idx] += 1.5

        # Local context attention (previous few tokens)
        for j in range(max(0, i-3), i):
            attention[i, j] += 1.0 / (i - j + 1)

        # Semantic similarity for nearby tokens
        for j in range(n):
            if abs(i - j) <= 5 and i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.3:
                    attention[i, j] += sim * 0.8

        # Special patterns for punctuation and function words
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Punctuation attends to nearby content words
            if spacy_token.pos_ == 'PUNCT':
                for j in range(max(0, i-3), min(n, i+3)):
                    if j != i and token_to_spacy[j]:
                        other_spacy = doc[token_to_spacy[j][0]]
                        if other_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            attention[i, j] += 1.2

            # Function words attend to content words they modify
            if spacy_token.pos_ in ['ADP', 'DET', 'AUX']:
                for j in range(n):
                    if j != i and token_to_spacy[j]:
                        other_spacy = doc[token_to_spacy[j][0]]
                        if other_spacy.pos_ in ['NOUN', 'VERB'] and abs(i - j) <= 3:
                            attention[i, j] += 1.0

    # Special case fixes for [CLS] and [SEP] patterns
    if n > 0 and tokens[0] in ['[CLS]', '<s>', '<|endoftext|>']:
        # Much stronger [CLS] self-attention
        attention[0, 0] = 8.0

        # Reduce [CLS] bias for content words (but keep for some function words)
        for i in range(1, n):
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    attention[i, 0] *= 0.5  # Reduce [CLS] bias for content words

    # Special [SEP] token behavior
    if n > 1 and tokens[-1] in ['[SEP]', '</s>', '<|endoftext|>']:
        sep_idx = n - 1
        # Much stronger [SEP] -> [CLS] attention
        if tokens[0] in ['[CLS]', '<s>', '<|endoftext|>']:
            attention[sep_idx, 0] = 12.0

        # Reduce [SEP] attention to content words
        for j in range(1, n-1):
            spacy_indices = token_to_spacy[j]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    attention[sep_idx, j] *= 0.2  # Dramatically reduce content word attention

    return "program_L3H11", make_row_stochastic(attention)



def program_L3H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Structural token attention head focusing on articles, punctuation, syntactic relationships, and cross-sentence connections."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        token = tokens[i].strip()

        # Get spacy info for this token if available
        spacy_tokens = token_to_spacy[i] if i < len(token_to_spacy) else []
        spacy_token = doc[spacy_tokens[0]] if spacy_tokens else None

        # Strong self-attention for articles, commas, and some content words
        if token.lower() in ['a', 'an', 'the'] or token == ',':
            attention[i, i] = 0.5
        elif spacy_token and spacy_token.pos_ in ['NOUN', 'PROPN']:
            attention[i, i] = 0.3
        elif token in ['.', 'and']:
            attention[i, i] = 0.3
        else:
            attention[i, i] = 0.1

        # HIGH PRIORITY FIX: [SEP] token should have very high self-attention
        if token == '[SEP]':
            attention[i, i] = 0.7

        # High attention to [CLS] token
        if i > 0:  # Don't attend from [CLS] to itself again
            attention[i, 0] = 0.2

        # High attention to commas from most tokens
        for j in range(n):
            if tokens[j].strip() == ',' and j != i:
                attention[i, j] = 0.25

        # Articles attend to other articles
        if token.lower() in ['a', 'an', 'the']:
            for j in range(n):
                other_token = tokens[j].strip()
                if other_token.lower() in ['a', 'an', 'the'] and j != i:
                    attention[i, j] = 0.15

        # Adjectives attend to articles they modify
        if spacy_token and spacy_token.pos_ == 'ADJ':
            for j in range(n):
                other_token = tokens[j].strip()
                if other_token.lower() in ['a', 'an', 'the'] and abs(i - j) <= 2:
                    attention[i, j] = 0.2

        # Nouns attend to preceding articles
        if spacy_token and spacy_token.pos_ in ['NOUN', 'PROPN']:
            for j in range(max(0, i-3), i):
                other_token = tokens[j].strip()
                if other_token.lower() in ['a', 'an', 'the']:
                    attention[i, j] = 0.1

        # Period attends to commas and conjunctions
        if token == '.':
            for j in range(n):
                other_token = tokens[j].strip()
                if other_token == ',' or other_token == 'and':
                    attention[i, j] = 0.2

        # Conjunctions attend to commas
        if token == 'and':
            for j in range(n):
                if tokens[j].strip() == ',':
                    attention[i, j] = 0.25

        # NEW: Cross-sentence attention - tokens in second sentence attend to periods from first sentence
        if spacy_token:
            # Find period positions
            period_positions = [j for j in range(n) if tokens[j].strip() == '.']
            if len(period_positions) > 0:
                # If we're past the first period, attend to it
                first_period = period_positions[0]
                if i > first_period:
                    attention[i, first_period] = 0.15

        # NEW: Pronouns attend to similar tokens (potential antecedents)
        if spacy_token and spacy_token.pos_ == 'PRON':
            for j in range(i):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    other_spacy = doc[token_to_spacy[j][0]]
                    if other_spacy.pos_ in ['NOUN', 'PROPN']:
                        attention[i, j] = 0.08

        # NEW: Enhanced conjunction attention - conjunctions attend to related tokens across clauses
        if token == 'and' and spacy_token:
            for j in range(n):
                if j != i and j < len(token_to_spacy) and token_to_spacy[j]:
                    other_spacy = doc[token_to_spacy[j][0]]
                    # Attend to verbs and nouns that might be coordinated
                    if other_spacy.pos_ in ['VERB', 'NOUN', 'PROPN'] and abs(i - j) > 2:
                        attention[i, j] = 0.1

    # Normalize to make row stochastic
    attention = make_row_stochastic(attention)

    return "program_L3H2", attention



def program_L3H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic-syntactic attention head: routes important content to [CLS] while capturing local dependencies."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy parse and alignment
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        token = tokens[i]

        # Special case: sentence-final punctuation and [SEP] should attend strongly to [CLS]
        if (token == '.' and i == n-2 and tokens[-1] == '[SEP]') or token == '[SEP]':
            # Much lower self-attention for sentence-final tokens
            if token == '.':
                attention[i, i] = 0.1
            else:  # [SEP]
                attention[i, i] = 0.6

            # Strong attention to [CLS] for sentence-level information
            if tokens[0] == '[CLS]':
                if token == '.':
                    attention[i, 0] = 0.3
                else:  # [SEP]
                    attention[i, 0] = 0.4

            # Small attention to nearby tokens
            for j in range(max(0, i-2), min(n, i+3)):
                if j != i and j != 0:
                    attention[i, j] = 0.02

        # Special tokens get very strong self-attention
        elif token in ['[CLS]', '.', '?', '!', ',', '"']:
            attention[i, i] = 0.6
            # Small attention to nearby tokens
            for j in range(max(0, i-2), min(n, i+3)):
                if j != i:
                    attention[i, j] = 0.05

        else:
            # Get spacy info for this token
            spacy_indices = token_to_spacy[i]
            spacy_token = None
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]

            # Base attention distribution
            base_weight = 0.02
            for j in range(n):
                attention[i, j] = base_weight

            # Strong attention to [CLS] for content words, especially nouns
            if tokens[0] == '[CLS]':
                cls_weight = 0.1
                if spacy_token:
                    # Nouns get very high attention to [CLS]
                    if spacy_token.pos_ in ['NOUN', 'PROPN']:
                        cls_weight = 0.3
                    # Verbs and adjectives get moderate attention
                    elif spacy_token.pos_ in ['VERB', 'ADJ']:
                        cls_weight = 0.15
                    # Important function words get some attention
                    elif spacy_token.pos_ in ['ADP']:  # prepositions
                        cls_weight = 0.12

                attention[i, 0] = cls_weight

            # Self-attention
            attention[i, i] = 0.05

            # Syntactic relationships if we have spacy info
            if spacy_token:
                # Prepositions attend strongly to their objects
                if spacy_token.pos_ == 'ADP':
                    for child in spacy_token.children:
                        if child.dep_ == 'pobj':
                            child_tokens = spacy_to_token[child.i]
                            for ct in child_tokens:
                                if ct < n:
                                    attention[i, ct] = 0.08

                # Conjunctions and words around them
                if spacy_token.pos_ == 'CCONJ' or token == 'and':
                    # Attend to adjacent words
                    for j in [i-1, i+1]:
                        if 0 <= j < n:
                            attention[i, j] = 0.06

                # Adjectives attend to nearby conjunctions
                if spacy_token.pos_ == 'ADJ':
                    for j in range(max(0, i-3), min(n, i+3)):
                        if tokens[j] == 'and':
                            attention[i, j] = 0.07

                # Determiners and articles attend to their heads
                if spacy_token.pos_ == 'DET':
                    head_tokens = spacy_to_token[spacy_token.head.i]
                    for ht in head_tokens:
                        if ht < n:
                            attention[i, ht] = 0.06

            # Proximity bias - attend more to nearby tokens
            for j in range(n):
                if i != j:
                    distance = abs(i - j)
                    if distance <= 2:
                        attention[i, j] += 0.02
                    elif distance <= 4:
                        attention[i, j] += 0.01

            # Semantic similarity boost
            for j in range(n):
                if i != j and j > 0:  # Skip [CLS] for similarity
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.7:
                        attention[i, j] += 0.03

    return "program_L3H3", make_row_stochastic(attention)



def program_L3H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content word to [CLS] aggregation head with reduced self-attention for content words."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L3H4", np.array([])

    attention = np.zeros((n, n))

    # Parse sentence for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Special token handling
        if token == '[CLS]':
            # [CLS] attends strongly to itself
            attention[i, 0] = 0.8
            # Distribute remaining attention uniformly
            remaining = 0.2 / (n - 1) if n > 1 else 0.0
            for j in range(1, n):
                attention[i, j] = remaining

        elif token == '[SEP]':
            # [SEP] attends strongly to [CLS] and itself
            attention[i, 0] = 0.6  # to [CLS]
            attention[i, i] = 0.3  # to self
            # Small attention to sentence-final punctuation
            for j in range(n):
                if tokens[j] in '.!?':
                    attention[i, j] = 0.05
            # Normalize remaining
            remaining = max(0, 1.0 - attention[i].sum())
            if remaining > 0:
                for j in range(1, n-1):
                    if tokens[j] not in '.!?' and j != i:
                        attention[i, j] = remaining / max(1, n-3)

        # Sentence-final punctuation
        elif token in '.!?':
            # Strong attention to main verbs and important content words
            attention[i, 0] = 0.1  # Some attention to [CLS]

            verb_weight = 0.0
            content_weight = 0.0

            # Find verbs and important content words
            for j in range(n):
                if j == i:
                    attention[i, j] = 0.05  # Small self-attention
                    continue

                spacy_indices = token_to_spacy[j]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]

                    # Strong attention to main verbs
                    if spacy_token.pos_ in ['VERB', 'AUX'] and spacy_token.dep_ in ['ROOT', 'aux', 'cop']:
                        attention[i, j] = 0.3
                        verb_weight += 0.3

                    # Attention to important nouns and adjectives
                    elif spacy_token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and spacy_token.dep_ in ['nsubj', 'dobj', 'pobj', 'compound']:
                        attention[i, j] = 0.15
                        content_weight += 0.15

                    # Attention to other content words
                    elif spacy_token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV']:
                        attention[i, j] = 0.08
                        content_weight += 0.08

            # Normalize if we've assigned too much weight
            total_assigned = attention[i].sum()
            if total_assigned > 1.0:
                attention[i] /= total_assigned

        # Commas and other punctuation
        elif token in ',:;':
            # Attend to [CLS] and nearby important words
            attention[i, 0] = 0.15
            attention[i, i] = 0.1  # Self-attention

            # Look for nearby important words
            window = 3
            for offset in [-2, -1, 1, 2]:
                j = i + offset
                if 0 <= j < n:
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_token = doc[spacy_indices[0]]
                        if spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                            attention[i, j] = 0.2

            # Distribute remaining attention
            remaining = max(0, 1.0 - attention[i].sum())
            if remaining > 0:
                for j in range(n):
                    if attention[i, j] == 0:
                        attention[i, j] = remaining / max(1, n - np.count_nonzero(attention[i]))

        # Content words
        else:
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]

                # NEW: Check if this is a content word that should strongly attend to [CLS]
                if spacy_token.pos_ in ['VERB', 'AUX', 'NOUN', 'PROPN', 'ADJ', 'ADV']:
                    # Content words attend much more strongly to [CLS]
                    attention[i, 0] = 0.4

                    # Reduced self-attention for content words
                    if spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB'] and spacy_token.dep_ in ['ROOT', 'nsubj', 'dobj']:
                        attention[i, i] = 0.1
                    else:
                        attention[i, i] = 0.05

                    # Much reduced attention to other content words
                    for j in range(n):
                        if j == i or j == 0:
                            continue

                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]

                            # Attend to syntactic head
                            if spacy_token.head == other_token:
                                attention[i, j] = 0.1

                            # Attend to dependents
                            elif other_token.head == spacy_token:
                                attention[i, j] = 0.08

                            # Reduced attention to similar content words
                            elif (spacy_token.pos_ in ['NOUN', 'PROPN'] and 
                                  other_token.pos_ in ['NOUN', 'PROPN']):
                                sim = embedding_similarity(tokens, i, j)
                                if sim > 0.5:
                                    attention[i, j] = 0.05 * sim

                    # Distribute remaining attention
                    remaining = max(0, 1.0 - attention[i].sum())
                    if remaining > 0:
                        for j in range(n):
                            if attention[i, j] == 0:
                                attention[i, j] = remaining / max(1, n - np.count_nonzero(attention[i]))

                else:
                    # Non-content words use original logic
                    # Strong self-attention for important words
                    if spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB'] and spacy_token.dep_ in ['ROOT', 'nsubj', 'dobj']:
                        attention[i, i] = 0.25
                    else:
                        attention[i, i] = 0.1

                    # Attention to [CLS]
                    attention[i, 0] = 0.15

                    # Attention based on syntactic relationships
                    for j in range(n):
                        if j == i or j == 0:
                            continue

                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]

                            # Attend to syntactic head
                            if spacy_token.head == other_token:
                                attention[i, j] = 0.2

                            # Attend to dependents
                            elif other_token.head == spacy_token:
                                attention[i, j] = 0.15

                            # Attend to similar content words
                            elif (spacy_token.pos_ in ['NOUN', 'PROPN'] and 
                                  other_token.pos_ in ['NOUN', 'PROPN']):
                                sim = embedding_similarity(tokens, i, j)
                                if sim > 0.5:
                                    attention[i, j] = 0.1 * sim

                    # Distribute remaining attention
                    remaining = max(0, 1.0 - attention[i].sum())
                    if remaining > 0:
                        for j in range(n):
                            if attention[i, j] == 0:
                                attention[i, j] = remaining / max(1, n - np.count_nonzero(attention[i]))

            else:
                # Fallback for tokens without spacy alignment
                attention[i, 0] = 0.3  # Attend to [CLS]
                attention[i, i] = 0.2  # Self-attention
                remaining = 0.5 / max(1, n - 2)
                for j in range(n):
                    if j != i and j != 0:
                        attention[i, j] = remaining

    return "program_L3H4", make_row_stochastic(attention)



def program_L3H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Adjacent semantic relationship head with strong positional bias toward previous tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    for i in range(n):
        token_i = tokens[i]

        # Special tokens have strong self-attention and attend to [CLS]
        if token_i in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.8
            if i > 0:
                attention[i, 0] = 0.2  # Attend to [CLS]
            continue

        # Parse for linguistic features
        doc = spacy_parse(sentence)
        token_to_spacy = _align_to_spacy(sentence, tokens)
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []

        # Base attention distribution
        base_weights = np.zeros(n)

        # Reduced attention to [CLS] token
        base_weights[0] = 0.1

        # Very high attention to immediately previous token
        if i > 0:
            base_weights[i-1] = 0.7

        # Reduced self-attention
        base_weights[i] = 0.05

        # Look for semantic relationships with nearby tokens
        for j in range(max(0, i-3), min(n, i+3)):
            if j == i or j == i-1:
                continue

            # Check semantic similarity
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.7:  # High semantic similarity
                base_weights[j] += 0.3
            elif sim > 0.5:  # Moderate semantic similarity
                base_weights[j] += 0.1

        # Linguistic pattern bonuses
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Prepositions attend strongly to their objects
            if spacy_token.pos_ == 'ADP' and i < n-1:
                base_weights[i+1] += 0.4

            # Determiners and adjectives attend to following nouns
            if spacy_token.pos_ in ['DET', 'ADJ'] and i < n-1:
                base_weights[i+1] += 0.2

            # Conjunctions attend to preceding coordinated elements
            if spacy_token.pos_ == 'CCONJ' and i > 0:
                base_weights[i-1] += 0.3

            # Verbs attend to their subjects/objects
            if spacy_token.pos_ == 'VERB':
                for child in spacy_token.children:
                    if child.dep_ in ['nsubj', 'dobj']:
                        # Find corresponding token indices
                        for k in range(n):
                            if k < len(token_to_spacy):
                                k_spacy_indices = token_to_spacy[k]
                                if k_spacy_indices and child.i in k_spacy_indices:
                                    base_weights[k] += 0.2

        # Handle punctuation
        if token_i in [',', '.']:
            if i > 0:
                base_weights[i-1] += 0.4
            base_weights[0] += 0.2
            base_weights[i] = 0.1

        # Apply distance decay for non-adjacent tokens
        for j in range(n):
            if j != i and j != i-1 and j != 0:
                distance = abs(i - j)
                decay = max(0.1, 1.0 / (distance + 1))
                base_weights[j] *= decay

        attention[i] = base_weights

    return "program_L3H5", make_row_stochastic(attention)



def program_L3H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """First-token and conjunction attention with enhanced special token patterns."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L3H6", np.array([])

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        # Get spacy tokens for current token
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        # Base attention distribution
        base_weight = 0.01

        for j in range(n):
            target_spacy_indices = token_to_spacy[j]
            target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

            weight = base_weight

            # Self-attention boost for punctuation and special tokens
            if i == j:
                if tokens[i] in ['[CLS]', '[SEP]', '.', ',', '!', '?', ';', ':']:
                    weight += 0.4
                else:
                    weight += 0.05

            # Strong attention to [CLS] from content words
            if j == 0 and tokens[j] == '[CLS]':
                if current_spacy and current_spacy.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    # Stronger for early position words
                    position_factor = max(0.1, 1.0 - (i * 0.1))
                    weight += 0.3 * position_factor
                else:
                    weight += 0.1

            # Strong attention from/to coordinating conjunctions
            target_token = tokens[j].lower().strip()
            current_token = tokens[i].lower().strip()

            if target_token == 'and':
                # Nearby tokens attend to "and"
                distance = abs(i - j)
                if distance <= 3:
                    weight += 0.15 / (distance + 1)
                elif distance <= 8:
                    weight += 0.08

            if current_token == 'and':
                # "and" attends to other conjunctions and nearby content
                if target_token == 'and':
                    weight += 0.1
                elif target_spacy and target_spacy.pos_ in ['NOUN', 'VERB']:
                    distance = abs(i - j)
                    if distance <= 5:
                        weight += 0.05

            # Attention to commas from nearby tokens
            if tokens[j] == ',':
                distance = abs(i - j)
                if distance <= 4:
                    weight += 0.08 / (distance + 1)

            # Syntactic relationships
            if current_spacy and target_spacy:
                # Modifiers attend to their heads
                if current_spacy.head == target_spacy:
                    weight += 0.08

                # Nouns attend to their determiners/adjectives
                if (current_spacy.pos_ == 'NOUN' and 
                    target_spacy.pos_ in ['DET', 'ADJ'] and
                    target_spacy.head == current_spacy):
                    weight += 0.06

            # Sentence-final punctuation gets broad attention
            if i == n - 2 and tokens[i] in ['.', '!', '?']:
                if j < i - 1:  # Attend to earlier content
                    distance = i - j
                    weight += 0.04 / np.sqrt(distance)

            # Recency bias for nearby tokens
            if i != j:
                distance = abs(i - j)
                if distance == 1:
                    weight += 0.02
                elif distance <= 3:
                    weight += 0.01

            # ENHANCED: Special token attention patterns
            # [SEP] and sentence-final punctuation strongly attend to [CLS]
            if j == 0 and tokens[j] == '[CLS]':
                if tokens[i] == '[SEP]':
                    weight += 0.2  # Much stronger [SEP] -> [CLS] attention
                elif i == n - 2 and tokens[i] in ['.', '!', '?']:
                    weight += 0.15  # Stronger sentence-final punctuation -> [CLS] attention

            # Enhanced [SEP] self-attention
            if i == j and tokens[i] == '[SEP]':
                weight += 0.1  # Additional boost for [SEP] self-attention

            attention[i, j] = weight

    # Apply row normalization
    attention = make_row_stochastic(attention)

    return "program_L3H6", attention



def program_L3H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that combines strong first-token bias with early positional preference and syntactic relationships."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L3H7", np.array([])

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Strong attention to [CLS] token (position 0)
        if tokens[0] in ['[CLS]', '<s>', '<bos>'] or i == 0:
            attention[i, 0] = 0.4
        else:
            attention[i, 0] = 0.3

        # Self-attention
        attention[i, i] = 0.15

        # Early positional bias - attend more to earlier tokens
        for j in range(n):
            if j != i and j != 0:
                # Distance-based decay
                distance = abs(i - j)
                position_weight = 1.0 / (1.0 + distance * 0.3)

                # Prefer earlier tokens
                if j < i:
                    position_weight *= 1.5

                attention[i, j] += position_weight * 0.1

        # Add syntactic and semantic relationships
        if i < len(token_to_spacy) and token_to_spacy[i]:
            current_spacy_idx = token_to_spacy[i][0]
            if current_spacy_idx < len(doc):
                current_token = doc[current_spacy_idx]

                # Attend to syntactic head
                if current_token.head != current_token:
                    head_spacy_idx = current_token.head.i
                    if head_spacy_idx < len(spacy_to_token) and spacy_to_token[head_spacy_idx]:
                        head_token_idx = spacy_to_token[head_spacy_idx][0]
                        if head_token_idx < n:
                            attention[i, head_token_idx] += 0.1

                # Attend to syntactic children
                for child in current_token.children:
                    child_spacy_idx = child.i
                    if child_spacy_idx < len(spacy_to_token) and spacy_to_token[child_spacy_idx]:
                        child_token_idx = spacy_to_token[child_spacy_idx][0]
                        if child_token_idx < n:
                            attention[i, child_token_idx] += 0.05

        # Add semantic similarity component
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:  # Only for reasonably similar tokens
                    attention[i, j] += sim * 0.08

        # Special handling for punctuation - attend to nearby content words
        if tokens[i] in ['.', ',', '?', '!', ';', ':']:
            for j in range(max(0, i-3), min(n, i+2)):
                if j != i and tokens[j] not in ['.', ',', '?', '!', ';', ':', '[CLS]', '[SEP]']:
                    attention[i, j] += 0.1

    # Special case: boost self-attention for special tokens
    for i in range(n):
        if tokens[i] in ['[CLS]', '[SEP]', '<s>', '<bos>', '</s>', '<eos>']:
            # Significantly increase self-attention for special tokens
            attention[i, i] += 0.35

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L3H7", attention



def program_L3H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attends to early content words (especially verbs) and [CLS] token, with [SEP] strongly attending to [CLS]."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse sentence with spacy
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special handling for [SEP] token - should attend strongly to [CLS]
        if tokens[i] == '[SEP]':
            for j in range(n):
                if tokens[j] == '[CLS]':
                    attention_matrix[i, j] = 0.7  # Strong attention to [CLS]
                else:
                    attention_matrix[i, j] = 0.0  # No attention to other tokens
            continue

        # Strong self-attention for [CLS] token
        if tokens[i] == '[CLS]':
            attention_matrix[i, i] = 1.0
            continue

        # Get spacy tokens for current position
        spacy_indices = alignment[i] if i < len(alignment) else []
        current_spacy_tokens = [doc[idx] for idx in spacy_indices if idx < len(doc)]

        for j in range(n):
            weight = 0.0

            # Very strong attention to [CLS]
            if tokens[j] == '[CLS]':
                weight = 0.4

            # Strong attention to early content words, especially verbs
            elif j < min(6, n):  # Focus on first few tokens
                target_spacy_indices = alignment[j] if j < len(alignment) else []
                target_spacy_tokens = [doc[idx] for idx in target_spacy_indices if idx < len(doc)]

                for target_token in target_spacy_tokens:
                    # High attention to verbs
                    if target_token.pos_ in ['VERB', 'AUX']:
                        weight = max(weight, 0.3)
                    # Moderate attention to other content words
                    elif target_token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV']:
                        weight = max(weight, 0.15)

            # Self-attention
            if i == j:
                weight = max(weight, 0.1)

            # Local attention (previous token)
            elif j == i - 1:
                weight = max(weight, 0.08)

            # Similarity-based attention
            if j != i:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.3:
                    weight = max(weight, 0.1)

            # Distance decay for non-special cases
            if j > 0 and tokens[j] != '[CLS]':
                distance = abs(i - j) + 1
                weight *= (1.0 / (1.0 + 0.1 * distance))

            attention_matrix[i, j] = weight

    return "program_L3H8", make_row_stochastic(attention_matrix)



def program_L3H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content-word to CLS attention with enhanced SEP and dynamic CLS attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Base attention distribution
        row_attention = np.zeros(n)

        # Find corresponding spacy token(s) for linguistic analysis
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []

        # Determine if this is a content word
        is_content_word = False
        is_punctuation = False

        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            is_content_word = spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
            is_punctuation = spacy_token.pos_ == 'PUNCT'

        # Check if token is special tokens or punctuation by string
        token_text = tokens[i].strip()
        if token_text in ['[CLS]', '[SEP]'] or token_text in '.,!?;:"\'()[]{}':
            is_punctuation = True

        # Strong attention to [CLS] for content words
        if is_content_word:
            row_attention[0] = 0.6  # High attention to CLS
        elif is_punctuation:
            row_attention[0] = 0.2  # Moderate attention to CLS for punctuation
        else:
            row_attention[0] = 0.3  # Baseline attention to CLS

        # Self-attention
        row_attention[i] = 0.1

        # Local context attention (nearby tokens)
        for j in range(n):
            if i != j and j != 0:  # Not self, not CLS
                distance = abs(i - j)
                if distance == 1:  # Adjacent tokens
                    row_attention[j] = 0.08
                elif distance == 2:  # Two tokens away
                    row_attention[j] = 0.04
                elif distance <= 4:  # Nearby tokens
                    row_attention[j] = 0.02
                else:
                    row_attention[j] = 0.01

        # Special handling for [CLS] token itself
        if i == 0:
            row_attention[0] = 0.7  # Strong self-attention for CLS
            # Distribute remaining attention more evenly
            remaining = 0.3
            for j in range(1, n):
                row_attention[j] = remaining / (n - 1)

        # Special handling for [SEP] token - ENHANCED
        if i == n - 1 and token_text == '[SEP]':
            row_attention[0] = 0.85  # Much higher attention to CLS (was 0.3)
            row_attention[i] = 0.07   # Reduced self-attention (was 0.4)
            if i > 0:
                row_attention[i-1] = 0.03 # Reduced attention to previous token (was 0.2)
            # Small remaining attention distributed
            if n > 2:
                remaining = 0.05
                for j in range(1, n-2):
                    row_attention[j] = remaining / max(1, n-3)

        # Handle punctuation attending to nearby content
        if is_punctuation and i > 0 and i < n-1:
            # Look for nearby content words
            for offset in [-1, 1, -2, 2]:
                j = i + offset
                if 0 <= j < n:
                    j_spacy_indices = token_to_spacy[j] if j < len(token_to_spacy) else []
                    if j_spacy_indices:
                        j_spacy_token = doc[j_spacy_indices[0]]
                        if j_spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            row_attention[j] += 0.05

        attention[i] = row_attention

    return "program_L3H9", make_row_stochastic(attention)



def program_L4H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic relationship and prepositional dependency attention head with strong [SEP] self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Get spacy tokens for current position
        spacy_indices = token_to_spacy[i]

        for j in range(n):
            weight = 0.0

            # Base self-attention
            if i == j:
                weight += 0.05

            # Strong [SEP] self-attention (targeted fix)
            if tokens[i] == '[SEP]' and i == j:
                weight += 0.65  # Boost [SEP] self-attention to match observed pattern

            # Special token handling
            if tokens[i] == '[SEP]' and tokens[j] == '.':
                weight += 0.15
            elif tokens[i] == '[SEP]' and tokens[j] == '[CLS]':
                weight += 0.06
            elif tokens[i] == '[CLS]' and tokens[j] == '[CLS]':
                weight += 0.06

            # Punctuation patterns
            if tokens[i] == '.' and tokens[j] in [',', '!']:
                weight += 0.08
            elif tokens[i] == '.' and tokens[j] == '.':
                weight += 0.10

            # Process spacy-based features if available
            if spacy_indices and token_to_spacy[j]:
                spacy_i = spacy_indices[0]
                spacy_j_indices = token_to_spacy[j]

                if spacy_i < len(doc) and spacy_j_indices:
                    spacy_j = spacy_j_indices[0]
                    if spacy_j < len(doc):
                        spacy_token_i = doc[spacy_i]
                        spacy_token_j = doc[spacy_j]

                        # Strong prepositional relationships
                        if spacy_token_i.pos_ == 'ADP':  # Preposition
                            # Check if j is the object of the preposition
                            for child in spacy_token_i.children:
                                if child.dep_ in ['pobj', 'pcomp'] and any(idx in spacy_to_token[child.i] for idx in [j]):
                                    weight += 0.6

                        # Semantic similarity boost
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:
                            weight += 0.15 * sim

                        # Verb to object relationships
                        if spacy_token_i.pos_ == 'VERB' and spacy_token_j.dep_ in ['dobj', 'pobj']:
                            weight += 0.08

                        # Adjective to noun relationships
                        if spacy_token_i.pos_ == 'ADJ' and spacy_token_j.pos_ == 'NOUN':
                            weight += 0.06

                        # Name/entity relationships
                        if spacy_token_i.ent_type_ == spacy_token_j.ent_type_ and spacy_token_i.ent_type_ != '':
                            weight += 0.1

                        # Compound relationships
                        if spacy_token_j.dep_ == 'compound' and spacy_token_j.head.i == spacy_i:
                            weight += 0.4
                        elif spacy_token_i.dep_ == 'compound' and spacy_token_i.head.i == spacy_j:
                            weight += 0.2

            # Distance decay for nearby tokens
            if abs(i - j) == 1:
                weight += 0.03
            elif abs(i - j) <= 3:
                weight += 0.01

            attention_matrix[i, j] = weight

    return "program_L4H0", make_row_stochastic(attention_matrix)



def program_L4H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head focusing on structural anchors and content word relationships."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        for j in range(n):
            weight = 0.0

            # Self-attention baseline
            if i == j:
                weight += 0.03

            # High attention to punctuation, especially commas
            if tokens[j] in [',', '.', '!', '?', ';', ':']:
                if tokens[j] == ',':
                    weight += 0.12  # Commas get strongest attention
                else:
                    weight += 0.08  # Other punctuation gets moderate attention

            # Attention to determiners and articles
            if len(token_to_spacy[j]) > 0:
                spacy_idx = token_to_spacy[j][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ == 'DET' or spacy_token.tag_ in ['DT', 'WDT']:
                        weight += 0.08

            # Special token behavior
            if tokens[i] == '[CLS]':
                if i == j:
                    weight = 0.15  # Strong self-attention for [CLS]
                else:
                    weight = 0.01
            elif tokens[j] == '[CLS]':
                weight += 0.02

            # CRITICAL FIX: Much stronger [SEP] self-attention
            if tokens[i] == '[SEP]' and i == j:
                weight = 0.65  # Dramatically increased from 0.15

            if tokens[i] == '[SEP]':
                # [SEP] attends moderately to various content words
                if len(token_to_spacy[j]) > 0:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            weight += 0.03

            # End punctuation attending to earlier structural elements
            if tokens[i] in ['.', '!', '?']:
                if tokens[j] in [',', 'but', 'and', 'or']:
                    weight += 0.06
                # Also attend to determiners and key content words
                if len(token_to_spacy[j]) > 0:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['DET', 'NOUN']:
                            weight += 0.04

            # Conjunctions attending to commas and other structural elements
            if len(token_to_spacy[i]) > 0:
                spacy_idx = token_to_spacy[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ == 'CCONJ':  # Coordinating conjunctions
                        if tokens[j] == ',':
                            weight += 0.08
                        elif j < i:  # Backward attention
                            weight += 0.02

            # Pronouns and function words attending to structural elements
            if len(token_to_spacy[i]) > 0:
                spacy_idx = token_to_spacy[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ in ['PRON', 'ADP', 'AUX']:
                        if tokens[j] in [',', '.']:
                            weight += 0.05

            # NEW: Content word relationships - key missing pattern
            if len(token_to_spacy[i]) > 0 and len(token_to_spacy[j]) > 0:
                spacy_i = token_to_spacy[i][0]
                spacy_j = token_to_spacy[j][0]
                if spacy_i < len(doc) and spacy_j < len(doc):
                    token_i = doc[spacy_i]
                    token_j = doc[spacy_j]

                    # Content words attend to each other based on semantic similarity
                    if (token_i.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and 
                        token_j.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']):
                        # Use embedding similarity to determine attention strength
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:  # Moderate similarity threshold
                            weight += 0.05
                        elif sim > 0.5:  # High similarity threshold
                            weight += 0.08

                        # Additional boost for certain POS combinations
                        if token_i.pos_ == 'NOUN' and token_j.pos_ == 'ADJ':
                            weight += 0.03
                        elif token_i.pos_ == 'VERB' and token_j.pos_ == 'NOUN':
                            weight += 0.03

            attention[i, j] = weight

    return "program_L4H1", make_row_stochastic(attention)



def program_L4H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference and semantic association head with enhanced syntactic dependencies and possessive resolution."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Strong self-attention for SEP tokens
        if tokens[i] == '[SEP]':
            attention[i, i] = 1.0
            continue

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            # Fallback for tokens without spacy alignment
            attention[i, :] = 1.0 / n
            continue

        spacy_token = doc[spacy_indices[0]]

        # Enhanced possessive pronoun coreference resolution
        if spacy_token.pos_ == 'PRON':
            # Look for noun antecedents, preferring closer ones
            for j in range(i):
                j_spacy = token_to_spacy[j]
                if j_spacy and doc[j_spacy[0]].pos_ in ['NOUN', 'PROPN']:
                    # Higher weight for closer nouns
                    distance_weight = 1.0 / (i - j + 1)
                    # Boost for semantic similarity
                    sim_boost = 1.0 + max(0, embedding_similarity(tokens, i, j))
                    attention[i, j] = distance_weight * sim_boost * 3.0

            # Special handling for possessive pronouns - attend to related content words
            if spacy_token.dep_ in ['poss', 'nmod:poss'] or tokens[i].lower() in ['his', 'her', 'its', 'their']:
                for j in range(n):
                    j_spacy = token_to_spacy[j]
                    if j_spacy and j != i:
                        j_token = doc[j_spacy[0]]
                        if j_token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                            sim = embedding_similarity(tokens, i, j)
                            if sim > 0.2:
                                attention[i, j] = max(attention[i, j], sim * 2.0)

        # Enhanced determiners - attend to syntactic head and related content
        elif spacy_token.pos_ == 'DET':
            # Attend to syntactic head
            if spacy_token.head != spacy_token:
                head_tokens = spacy_to_token[spacy_token.head.i]
                for head_tok_idx in head_tokens:
                    if 0 <= head_tok_idx < n:
                        attention[i, head_tok_idx] = 3.0

            for j in range(n):
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                        # Stronger for nouns in same phrase
                        if j > i and j < i + 3:
                            attention[i, j] = 2.0
                        else:
                            sim = embedding_similarity(tokens, i, j)
                            attention[i, j] = max(0.5, sim + 0.5)

        # Enhanced coordinating conjunctions - attend to syntactic dependents and heads
        elif spacy_token.pos_ == 'CCONJ':
            # Attend to syntactic head and children
            if spacy_token.head != spacy_token:
                head_tokens = spacy_to_token[spacy_token.head.i]
                for head_tok_idx in head_tokens:
                    if 0 <= head_tok_idx < n:
                        attention[i, head_tok_idx] = 3.0

            for child in spacy_token.children:
                child_tokens = spacy_to_token[child.i]
                for child_tok_idx in child_tokens:
                    if 0 <= child_tok_idx < n:
                        attention[i, child_tok_idx] = 2.5

            for j in range(i):
                j_spacy = token_to_spacy[j]
                if j_spacy and doc[j_spacy[0]].pos_ in ['VERB', 'ADJ', 'NOUN']:
                    distance_weight = 1.0 / (i - j + 1)
                    attention[i, j] = max(attention[i, j], distance_weight * 2.0)

        # Enhanced prepositions - attend to syntactic head and object
        elif spacy_token.pos_ == 'ADP':
            # Attend to syntactic head
            if spacy_token.head != spacy_token:
                head_tokens = spacy_to_token[spacy_token.head.i]
                for head_tok_idx in head_tokens:
                    if 0 <= head_tok_idx < n:
                        attention[i, head_tok_idx] = 2.5

            # Attend to prepositional object
            for child in spacy_token.children:
                if child.dep_ == 'pobj':
                    child_tokens = spacy_to_token[child.i]
                    for child_tok_idx in child_tokens:
                        if 0 <= child_tok_idx < n:
                            attention[i, child_tok_idx] = 2.0

            for j in range(n):
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                        sim = embedding_similarity(tokens, i, j)
                        if j != i:
                            attention[i, j] = max(attention[i, j], max(0.3, sim + 0.3))

        # Enhanced auxiliary verbs - attend to main verb
        elif spacy_token.pos_ == 'AUX':
            # Attend to syntactic head (usually the main verb)
            if spacy_token.head != spacy_token:
                head_tokens = spacy_to_token[spacy_token.head.i]
                for head_tok_idx in head_tokens:
                    if 0 <= head_tok_idx < n:
                        attention[i, head_tok_idx] = 3.0

            # Also attend to subject
            for j in range(min(i, 5)):
                j_spacy = token_to_spacy[j]
                if j_spacy and doc[j_spacy[0]].pos_ in ['NOUN', 'PROPN']:
                    attention[i, j] = 2.0

        # Content words attend to semantically similar words and subjects
        elif spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
            # Attend to sentence subject
            for j in range(min(i, 5)):  # Look in early positions
                j_spacy = token_to_spacy[j]
                if j_spacy and doc[j_spacy[0]].pos_ in ['NOUN', 'PROPN']:
                    attention[i, j] = 1.5

            # Attend to semantically similar words
            for j in range(n):
                if j != i:
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        attention[i, j] = sim * 1.5

        # Punctuation attends to preceding content
        elif tokens[i] in ['.', ',', '!', '?']:
            for j in range(i):
                j_spacy = token_to_spacy[j]
                if j_spacy and doc[j_spacy[0]].pos_ in ['VERB', 'NOUN', 'ADJ']:
                    distance_weight = 1.0 / (i - j + 1)
                    attention[i, j] = distance_weight

        # Base attention to CLS and self
        attention[i, 0] = max(attention[i, 0], 0.1)  # CLS
        attention[i, i] = max(attention[i, i], 0.1)  # Self

    return "program_L4H10", make_row_stochastic(attention)



def program_L4H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that combines special token processing with sentence structure awareness."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Very high self-attention for [SEP] token
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.9
            # Small residual attention to other tokens
            for j in range(n):
                if j != i:
                    if tokens[j] == '[CLS]':
                        attention[i, j] = 0.02
                    elif tokens[j] == '.':
                        attention[i, j] = 0.015
                    else:
                        attention[i, j] = 0.01
            continue

        # High self-attention for [CLS]
        if tokens[i] == '[CLS]':
            attention[i, i] = 0.07

        # Strong baseline attention to [CLS] from most tokens
        cls_idx = 0 if tokens[0] == '[CLS]' else -1
        if cls_idx != -1 and i != cls_idx:
            # Get spacy info for this token
            spacy_indices = token_to_spacy[i]
            base_cls_attention = 0.03

            # Boost for content words (nouns, verbs, adjectives)
            if spacy_indices:
                spacy_tok = doc[spacy_indices[0]]
                if spacy_tok.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    base_cls_attention = 0.06
                elif spacy_tok.pos_ in ['DET', 'ADP', 'CONJ', 'SCONJ']:
                    base_cls_attention = 0.025

            attention[i, cls_idx] = base_cls_attention

        # Punctuation tokens attend strongly to sentence-initial content words
        if tokens[i] in ['.', '!', '?']:
            attention[i, i] = 0.1  # Some self-attention for punctuation

            # Find sentence-initial content word (usually position 1 after [CLS])
            for j in range(1, min(3, n)):  # Check positions 1-2
                if j < n:
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_tok = doc[spacy_indices[0]]
                        if spacy_tok.pos_ in ['NOUN', 'PROPN', 'PRON']:
                            attention[i, j] = 0.15
                            break

        # Self-attention for various token types
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_tok = doc[token_to_spacy[i][0]]
            if spacy_tok.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                attention[i, i] += 0.02
            elif tokens[i] in [',', 'and', 'but']:
                attention[i, i] += 0.02

        # Local syntactic attention patterns
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_tok = doc[token_to_spacy[i][0]]

            # Verbs attend to nearby function words and subjects
            if spacy_tok.pos_ == 'VERB':
                for j in range(max(0, i-3), min(n, i+2)):
                    if j != i and j < len(token_to_spacy) and token_to_spacy[j]:
                        other_spacy = doc[token_to_spacy[j][0]]
                        if other_spacy.pos_ in ['PRON', 'DET'] or other_spacy.dep_ == 'nsubj':
                            attention[i, j] += 0.015

            # Adjectives and determiners get small local attention
            if spacy_tok.pos_ in ['ADJ', 'DET']:
                for j in range(max(0, i-2), min(n, i+3)):
                    if j != i:
                        attention[i, j] += 0.01

        # Fill remaining attention uniformly
        current_sum = attention[i].sum()
        if current_sum < 0.95:
            remaining = 1.0 - current_sum
            # Distribute remaining attention
            for j in range(n):
                if attention[i, j] == 0:
                    attention[i, j] = remaining / (n - np.count_nonzero(attention[i]))

    return "program_L4H11", make_row_stochastic(attention)



def program_L4H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that combines positional biases ([CLS], self-attention) with punctuation attending to nearby content words and semantic similarity patterns."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong self-attention baseline
        attention[i, i] = 0.1

        # Strong attention to [CLS] token
        if tokens[0] in ['[CLS]', '<s>']:
            attention[i, 0] = 0.15

        # Special handling for [SEP] token - very high self-attention
        if tokens[i] in ['[SEP]', '</s>']:
            attention[i, i] = 0.8
            continue

        # Punctuation tokens attend to nearby content words
        if tokens[i] in [',', '.', '!', '?', '"', "'"]:
            # Look for content words in a window around this punctuation
            window_start = max(0, i - 3)
            window_end = min(n, i + 2)

            for j in range(window_start, window_end):
                if j == i:
                    continue

                # Check if target is a content word using spacy
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                            # Distance-based weighting
                            dist = abs(i - j)
                            weight = 0.2 / (1 + dist * 0.5)
                            attention[i, j] += weight

        # Content words attend to [CLS] and have moderate self-attention
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    attention[i, 0] += 0.1  # Additional [CLS] attention

        # Sentence-final tokens (like periods) attend to early content words
        if tokens[i] in ['.', '!', '?'] and i > n // 2:
            for j in range(1, min(i, 6)):  # Look at first few non-[CLS] tokens
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'PROPN']:
                            attention[i, j] += 0.15

        # Add some recency bias - attend to recent tokens
        for j in range(max(0, i - 3), i):
            attention[i, j] += 0.05 / (1 + (i - j))

        # NEW: Add semantic similarity-based attention
        # Look for semantically similar tokens across the entire sequence
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                # Apply to pronouns, determiners, and content words
                if spacy_token.pos_ in ['PRON', 'DET', 'NOUN', 'VERB', 'ADJ', 'PROPN']:
                    for j in range(n):
                        if j == i or j == 0:  # Skip self and [CLS]
                            continue

                        # Check semantic similarity
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:  # Threshold for meaningful similarity
                            # Weight by similarity strength, with distance decay
                            dist = abs(i - j)
                            weight = sim * 0.08 / (1 + dist * 0.1)
                            attention[i, j] += weight

    return "program_L4H2", make_row_stochastic(attention)



def program_L4H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference and semantic similarity head with adaptive self-attention and selective coordination."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy alignment for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Adaptive self-attention baseline
        base_self_attention = 0.3

        # Boost self-attention for special tokens and important words
        if tokens[i] == '[SEP]':
            base_self_attention = 0.95  # Very strong self-attention for [SEP]
        elif tokens[i] in [',', '.', '?', '!']:
            base_self_attention = 0.4
        elif tokens[i] == 'and':
            base_self_attention = 0.5

        # Check spacy features for additional self-attention boosts
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None
        if current_spacy:
            if current_spacy.pos_ in ['PRON', 'PROPN']:
                base_self_attention = max(base_self_attention, 0.4)
            elif current_spacy.pos_ in ['NOUN', 'VERB']:
                base_self_attention = max(base_self_attention, 0.35)

        attention[i, i] = base_self_attention

        # Moderate attention to [CLS] token
        attention[i, 0] = 0.1

        for j in range(n):
            if i == j:
                continue

            # Embedding similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.7:
                attention[i, j] += 0.4
            elif sim > 0.5:
                attention[i, j] += 0.2

            # Pronoun coreference patterns
            target_spacy_indices = token_to_spacy[j]
            target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

            if current_spacy and target_spacy:
                # Pronoun to antecedent
                if current_spacy.pos_ == "PRON" and target_spacy.pos_ in ["NOUN", "PROPN", "PRON"]:
                    if current_spacy.text.lower() in ["he", "him", "his"] and target_spacy.pos_ in ["PROPN", "NOUN"]:
                        attention[i, j] += 0.3
                    elif current_spacy.text.lower() in ["it", "its"] and target_spacy.pos_ == "NOUN":
                        attention[i, j] += 0.3
                    elif current_spacy.text.lower() in ["you"] and target_spacy.text.lower() == "you":
                        attention[i, j] += 0.4

                # More selective coordination attention - only to adjacent coordinated elements
                if tokens[i] == "and" and j < i and (i - j) <= 3:
                    # Check if target is likely a coordinated element
                    if target_spacy.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]:
                        attention[i, j] += 0.1

                # Punctuation patterns
                if tokens[i] == "," and tokens[j] == ",":
                    attention[i, j] += 0.2

                # Semantic adjective relationships
                if current_spacy.pos_ == "ADJ" and target_spacy.pos_ == "ADJ":
                    attention[i, j] += 0.15

    return "program_L4H3", make_row_stochastic(attention)



def program_L4H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attends primarily to past tense verbs and main predicates, with backward positional bias and enhanced punctuation attention to discourse markers."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # High self-attention for special tokens
        if tokens[i] in ['[CLS]', '[SEP]', '.']:
            attention[i, i] = 1.0
            continue

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        for j in range(n):
            if i == j:
                # Moderate self-attention for regular tokens
                attention[i, j] = 0.1
            else:
                # Get spacy info for target token
                target_spacy_indices = token_to_spacy[j]
                target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

                # Base attention with backward bias
                base_weight = 0.02
                if j < i:  # Backward attention gets boost
                    base_weight *= 1.5

                # Strong attention to past tense verbs
                if target_spacy and target_spacy.tag_ in ['VBD', 'VBN']:
                    if target_spacy.pos_ == 'VERB':
                        base_weight *= 8.0

                # Moderate attention to main verbs and auxiliary verbs
                if target_spacy and target_spacy.pos_ == 'VERB':
                    base_weight *= 3.0

                # Boost attention to determiners and articles from content words
                if (current_spacy and current_spacy.pos_ in ['NOUN', 'ADJ', 'VERB'] and
                    target_spacy and target_spacy.pos_ in ['DET', 'ADP']):
                    base_weight *= 2.0

                # Boost for punctuation attending to main content
                if (tokens[i] in ['.', ','] and target_spacy and 
                    target_spacy.pos_ in ['VERB', 'NOUN']):
                    base_weight *= 4.0

                # NEW: Enhanced punctuation attention to conjunctions and discourse markers
                if tokens[i] in ['.', ',']:
                    # Strong attention to conjunctions
                    if target_spacy and target_spacy.pos_ == 'CCONJ':
                        base_weight *= 8.0
                    # Strong attention to sentence-initial content words
                    if j == 1 and target_spacy and target_spacy.pos_ in ['NOUN', 'PRON', 'NUM']:
                        base_weight *= 6.0
                    # Moderate attention to auxiliary words and adverbs
                    if target_spacy and target_spacy.pos_ in ['ADV', 'PART']:
                        base_weight *= 3.0

                # Distance decay for very far tokens
                distance = abs(i - j)
                if distance > 3:
                    base_weight *= 0.7

                attention[i, j] = base_weight

    return "program_L4H4", make_row_stochastic(attention)



def program_L4H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content word to [CLS] attention with self-attention for important tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy parse for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special handling for [SEP] token - very high self-attention
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.98
            # Small residual attention to [CLS]
            attention[i, 0] = 0.02
            continue

        # Special handling for period - high self-attention
        if tokens[i] == '.':
            attention[i, i] = 0.4
            attention[i, 0] = 0.3
            # Distribute rest to nearby content words
            for j in range(max(0, i-3), i):
                if j != 0:  # Don't double-count [CLS]
                    attention[i, j] = 0.1
            continue

        # Get spacy properties for current token
        spacy_indices = token_to_spacy[i]
        is_content_word = False
        is_function_word = False

        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            pos = spacy_token.pos_
            # Content words: NOUN, VERB, ADJ, ADV
            is_content_word = pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
            # Function words: DET, ADP, CCONJ, etc.
            is_function_word = pos in ['DET', 'ADP', 'CCONJ', 'PRON', 'AUX']

        # Strong attention to [CLS] for content words
        if is_content_word:
            attention[i, 0] = 0.6
            # Self-attention for content words
            attention[i, i] = 0.2
            # Distribute remaining attention to other content words
            remaining = 0.2
        elif tokens[i] == '[CLS]':
            # [CLS] attends to itself
            attention[i, i] = 0.8
            remaining = 0.2
        else:
            # Function words and others - moderate attention to [CLS]
            attention[i, 0] = 0.3
            attention[i, i] = 0.1
            remaining = 0.6

        # Distribute remaining attention
        content_positions = []
        for j in range(n):
            if j == i or j == 0:  # Skip self and [CLS] (already assigned)
                continue
            j_spacy = token_to_spacy[j]
            if j_spacy:
                j_pos = doc[j_spacy[0]].pos_
                if j_pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                    content_positions.append(j)

        if content_positions and remaining > 0:
            per_content = remaining / len(content_positions)
            for j in content_positions:
                attention[i, j] = per_content
        elif remaining > 0:
            # If no content words found, distribute to nearby tokens
            nearby = [j for j in range(max(0, i-2), min(n, i+3)) if j != i and j != 0]
            if nearby:
                per_nearby = remaining / len(nearby)
                for j in nearby:
                    attention[i, j] = per_nearby
            else:
                # Fall back to [CLS]
                attention[i, 0] += remaining

    return "program_L4H5", make_row_stochastic(attention)



def program_L4H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that directs tokens to attend to main verbs and early content words in their clauses."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Handle special tokens first
    for i in range(n):
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention_matrix[i, i] = 1.0
            continue

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Find main verbs and their positions
    main_verbs = []
    for i, spacy_indices in enumerate(token_to_spacy):
        if not spacy_indices:
            continue
        spacy_token = doc[spacy_indices[0]]

        # Identify main verbs (not auxiliaries, not infinitives)
        if (spacy_token.pos_ == 'VERB' and 
            spacy_token.dep_ in ['ROOT', 'conj', 'ccomp', 'xcomp', 'advcl'] and
            spacy_token.tag_ not in ['TO']):
            main_verbs.append(i)

    # If no main verbs found, fall back to any verb
    if not main_verbs:
        for i, spacy_indices in enumerate(token_to_spacy):
            if not spacy_indices:
                continue
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ == 'VERB':
                main_verbs.append(i)

    # Find early content words (positions 1-3, excluding punctuation and function words)
    early_content_words = []
    for i in range(1, min(4, n)):  # positions 1, 2, 3
        if tokens[i] in ['[CLS]', '[SEP]'] or tokens[i] in [',', '.', '?', '!', '"', "'"]:
            continue
        if token_to_spacy[i]:
            spacy_token = doc[token_to_spacy[i][0]]
            # Include nouns, verbs, adjectives, adverbs, pronouns, determiners
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'PROPN']:
                early_content_words.append(i)
        else:
            # If no spacy alignment, assume it's content
            early_content_words.append(i)

    for i in range(n):
        if tokens[i] in ['[CLS]', '[SEP]']:
            continue

        if not token_to_spacy[i]:
            # No spacy alignment, use simple heuristics
            attention_matrix[i, i] = 0.3
            if i > 0:
                attention_matrix[i, i-1] = 0.4
            if main_verbs:
                closest_verb = min(main_verbs, key=lambda v: abs(v - i))
                attention_matrix[i, closest_verb] = 0.3
            continue

        spacy_token = doc[token_to_spacy[i][0]]

        # Find the most relevant main verb for this token
        target_verb = None

        if main_verbs:
            # Look for syntactic head that is a main verb
            head_token = spacy_token.head
            for verb_idx in main_verbs:
                if (token_to_spacy[verb_idx] and 
                    head_token in [doc[j] for j in token_to_spacy[verb_idx]]):
                    target_verb = verb_idx
                    break

            # If no syntactic connection, use closest main verb
            if target_verb is None:
                target_verb = min(main_verbs, key=lambda v: abs(v - i))

        # Set attention weights based on token type and relationship
        if spacy_token.pos_ in ['VERB']:
            if target_verb is not None and target_verb != i:
                attention_matrix[i, target_verb] = 0.3
            attention_matrix[i, i] = 0.2
            # Verbs also attend to [CLS] 
            if tokens[0] == '[CLS]':
                attention_matrix[i, 0] = 0.2
            # NEW: Verbs attend to early content words
            for early_idx in early_content_words:
                if early_idx != i:
                    attention_matrix[i, early_idx] = 0.3

        elif spacy_token.dep_ in ['dobj', 'iobj', 'nsubj', 'nsubjpass', 'ccomp', 'xcomp']:
            # Direct dependents of verbs
            if target_verb is not None:
                attention_matrix[i, target_verb] = 0.3
            attention_matrix[i, i] = 0.1
            if tokens[0] == '[CLS]':
                attention_matrix[i, 0] = 0.2
            # NEW: Direct dependents attend to early content words
            for early_idx in early_content_words:
                if early_idx != i:
                    attention_matrix[i, early_idx] = 0.4

        elif spacy_token.pos_ in ['ADP', 'SCONJ', 'CCONJ'] or spacy_token.dep_ == 'cc':
            # Prepositions and conjunctions
            if target_verb is not None:
                attention_matrix[i, target_verb] = 0.2
            # Also attend to previous token
            if i > 0:
                attention_matrix[i, i-1] = 0.3
            attention_matrix[i, i] = 0.1
            if tokens[0] == '[CLS]':
                attention_matrix[i, 0] = 0.1
            # NEW: Conjunctions and prepositions attend to early content words
            for early_idx in early_content_words:
                if early_idx != i:
                    attention_matrix[i, early_idx] = 0.3

        elif spacy_token.dep_ in ['amod', 'advmod', 'prep', 'pobj']:
            # Modifiers and prepositional objects
            if target_verb is not None:
                attention_matrix[i, target_verb] = 0.2
            # Strong attention to syntactic head
            if i > 0:
                attention_matrix[i, i-1] = 0.4
            attention_matrix[i, i] = 0.1
            if tokens[0] == '[CLS]':
                attention_matrix[i, 0] = 0.1
            # NEW: Modifiers attend to early content words
            for early_idx in early_content_words:
                if early_idx != i:
                    attention_matrix[i, early_idx] = 0.2

        else:
            # Default case - attend to closest main verb and previous token
            if target_verb is not None:
                attention_matrix[i, target_verb] = 0.2
            if i > 0:
                attention_matrix[i, i-1] = 0.3
            attention_matrix[i, i] = 0.1
            if tokens[0] == '[CLS]':
                attention_matrix[i, 0] = 0.1
            # NEW: Default tokens attend to early content words
            for early_idx in early_content_words:
                if early_idx != i:
                    attention_matrix[i, early_idx] = 0.3

    return "program_L4H6", make_row_stochastic(attention_matrix)



def program_L4H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic dependency attention with enhanced conjunction and preposition patterns."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Handle [SEP] token - strong self-attention
        if tokens[i].strip() in ['[SEP]', '</s>']:
            attention[i, i] = 1.0
            continue

        # Base attention distribution
        base_attention = np.zeros(n)

        # Strong attention to [CLS]/first token
        base_attention[0] = 0.05

        # Self-attention baseline
        base_attention[i] = 0.02

        # Local positional bias - attend to nearby tokens
        for j in range(n):
            distance = abs(i - j)
            if distance <= 3:
                decay = np.exp(-0.3 * distance)
                base_attention[j] += 0.03 * decay

        # Enhanced conjunction and preposition patterns
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Strong conjunction patterns - attend to immediate predecessor
                if spacy_token.pos_ in ['CCONJ', 'CONJ'] or spacy_token.text.lower() == 'and':
                    if i > 0:
                        # Find the nearest content word to the left
                        for prev_idx in range(i-1, max(0, i-4), -1):
                            if token_to_spacy[prev_idx]:
                                prev_spacy_idx = token_to_spacy[prev_idx][0]
                                if prev_spacy_idx < len(doc):
                                    prev_token = doc[prev_spacy_idx]
                                    if prev_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                                        base_attention[prev_idx] += 0.35
                                        break

                # Enhanced preposition patterns - strong attention to objects
                if spacy_token.pos_ == 'ADP':
                    # Look for prepositional objects
                    for child in spacy_token.children:
                        if child.dep_ == 'pobj' and child.i < len(doc):
                            if spacy_to_token[child.i]:
                                obj_token_idx = spacy_to_token[child.i][0]
                                if obj_token_idx < n:
                                    base_attention[obj_token_idx] += 0.45

                    # Also attend to syntactic head (the word this preposition modifies)
                    if spacy_token.head.i != spacy_idx:
                        head_spacy_idx = spacy_token.head.i
                        if head_spacy_idx < len(doc) and spacy_to_token[head_spacy_idx]:
                            head_token_idx = spacy_to_token[head_spacy_idx][0]
                            if head_token_idx < n:
                                base_attention[head_token_idx] += 0.25

        # Syntactic dependency patterns
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Function words attend to their syntactic heads
                if spacy_token.pos_ in ['ADP', 'CONJ', 'CCONJ', 'DET', 'AUX']:
                    # Find syntactic head
                    head_spacy_idx = spacy_token.head.i if spacy_token.head.i != spacy_idx else spacy_idx
                    if head_spacy_idx < len(doc) and spacy_to_token[head_spacy_idx]:
                        head_token_idx = spacy_to_token[head_spacy_idx][0]
                        if head_token_idx < n:
                            base_attention[head_token_idx] += 0.15

                # Content words attend to their immediate dependents and heads
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    # Attend to head
                    if spacy_token.head.i != spacy_idx:
                        head_spacy_idx = spacy_token.head.i
                        if head_spacy_idx < len(doc) and spacy_to_token[head_spacy_idx]:
                            head_token_idx = spacy_to_token[head_spacy_idx][0]
                            if head_token_idx < n:
                                base_attention[head_token_idx] += 0.08

                    # Attend to direct children
                    for child in spacy_token.children:
                        if child.i < len(doc) and spacy_to_token[child.i]:
                            child_token_idx = spacy_to_token[child.i][0]
                            if child_token_idx < n:
                                base_attention[child_token_idx] += 0.06

        # Special patterns for specific constructions
        if i > 0:
            # Punctuation attends to previous meaningful token
            if tokens[i].strip() in [',', '.', ';', ':', '!', '?']:
                prev_idx = i - 1
                while prev_idx >= 0 and tokens[prev_idx].strip() in [' ', '\t']:
                    prev_idx -= 1
                if prev_idx >= 0:
                    base_attention[prev_idx] += 0.12

            # Articles and determiners attend strongly to following nouns
            if tokens[i].strip().lower() in ['a', 'an', 'the'] and i + 1 < n:
                # Look for noun in next few positions
                for j in range(i + 1, min(i + 4, n)):
                    if token_to_spacy[j]:
                        spacy_idx = token_to_spacy[j][0]
                        if spacy_idx < len(doc) and doc[spacy_idx].pos_ == 'NOUN':
                            base_attention[j] += 0.08
                            break

        # Enhance attention using embedding similarity for related tokens
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity threshold
                    base_attention[j] += 0.04 * sim

        attention[i] = base_attention

    return "program_L4H7", make_row_stochastic(attention)



def program_L4H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Mixed positional and syntactic attention with strong [CLS] and punctuation focus, plus dominant [SEP] self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        # Base attention weights
        weights = np.ones(n) * 0.01

        # Strong attention to [CLS] token (first token)
        if n > 0:
            weights[0] += 0.08

        # Self-attention boost
        weights[i] += 0.05

        # Get spacy info for current token if available
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []
        current_spacy_tok = doc[spacy_indices[0]] if spacy_indices else None

        # Punctuation tokens get special treatment
        if tokens[i] in [',', '.', '?', '!', '"', ';', ':']:
            # Punctuation attends strongly to [CLS]
            weights[0] += 0.1
            # Self-attention for punctuation
            weights[i] += 0.1

            # Commas attend to nearby content words
            if tokens[i] == ',' and current_spacy_tok:
                for j in range(max(0, i-3), min(n, i+2)):
                    j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                    if j_spacy:
                        j_tok = doc[j_spacy[0]]
                        if j_tok.pos_ in ['NOUN', 'PROPN', 'ADJ', 'NUM']:
                            weights[j] += 0.03

        # Special handling for [SEP] token (usually last)
        elif tokens[i] == '[SEP]':
            # Dramatically increase self-attention for [SEP] to match real pattern
            weights[i] += 1.5  # Much higher self-attention for [SEP]
            weights[0] += 0.05  # Some attention to [CLS]

        # Content tokens
        elif current_spacy_tok:
            # Verbs attend to their subjects and objects
            if current_spacy_tok.pos_ == 'VERB':
                for child in current_spacy_tok.children:
                    if child.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'pobj']:
                        # Find corresponding token indices
                        for j in range(n):
                            j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                            if j_spacy and any(doc[k] == child for k in j_spacy):
                                weights[j] += 0.04

                # Verbs also attend to [CLS]
                weights[0] += 0.04

            # Nouns attend to their modifiers and heads
            elif current_spacy_tok.pos_ in ['NOUN', 'PROPN']:
                # Attend to head
                if current_spacy_tok.head != current_spacy_tok:
                    for j in range(n):
                        j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                        if j_spacy and any(doc[k] == current_spacy_tok.head for k in j_spacy):
                            weights[j] += 0.03

                # Attend to modifiers
                for child in current_spacy_tok.children:
                    if child.dep_ in ['amod', 'compound', 'det']:
                        for j in range(n):
                            j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                            if j_spacy and any(doc[k] == child for k in j_spacy):
                                weights[j] += 0.02

            # Function words (determiners, prepositions) attend to nearby content
            elif current_spacy_tok.pos_ in ['DET', 'ADP', 'AUX']:
                for j in range(max(0, i-2), min(n, i+3)):
                    if j != i:
                        j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                        if j_spacy:
                            j_tok = doc[j_spacy[0]]
                            if j_tok.pos_ in ['NOUN', 'PROPN', 'VERB']:
                                weights[j] += 0.025

        # Sentence-final tokens attend more to important content words
        if i >= n - 2:  # Last couple of tokens
            for j in range(n):
                j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                if j_spacy:
                    j_tok = doc[j_spacy[0]]
                    if j_tok.pos_ == 'VERB':
                        weights[j] += 0.03
                    elif j_tok.pos_ in ['NOUN', 'PROPN']:
                        weights[j] += 0.02

        # Boost attention to semantically similar tokens
        for j in range(n):
            if j != i and j < len(tokens) and i < len(tokens):
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:
                    weights[j] += 0.02

        attention[i] = weights

    return "program_L4H8", make_row_stochastic(attention)



def program_L4H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that focuses on punctuation, special tokens, and verbs with positional bias, with [SEP] as major attention sink."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L4H9", np.array([])

    attention = np.zeros((n, n))

    # Parse sentence to get linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Base uniform attention
        attention[i, :] = 0.01

        # Special case for [SEP] tokens - they get massive self-attention
        if tokens[i] in ['[SEP]', '</s>']:
            attention[i, i] = 0.85
            # Reduce attention to other tokens proportionally
            for j in range(n):
                if j != i:
                    attention[i, j] = 0.005
        else:
            # Regular self-attention (much lower than original)
            attention[i, i] = 0.05

            # High attention to [CLS] token (position 0)
            if tokens[0] in ['[CLS]', '<|endoftext|>']:
                attention[i, 0] = 0.12

            # High attention to punctuation marks
            for j in range(n):
                if tokens[j] in [',', '.', '!', '?', ';', ':']:
                    attention[i, j] += 0.15

            # High attention to [SEP] tokens from all other tokens
            for j in range(n):
                if tokens[j] in ['[SEP]', '</s>']:
                    attention[i, j] += 0.08

            # Get spacy tokens for current position
            spacy_indices = token_to_spacy[i]

            # Attention to verbs
            for j in range(n):
                target_spacy = token_to_spacy[j]
                if target_spacy:
                    target_token = doc[target_spacy[0]]
                    if target_token.pos_ == 'VERB':
                        attention[i, j] += 0.12

            # Attention to conjunctions and connectors
            for j in range(n):
                target_spacy = token_to_spacy[j]
                if target_spacy:
                    target_token = doc[target_spacy[0]]
                    if target_token.pos_ == 'CCONJ' or target_token.text.lower() in ['and', 'that']:
                        attention[i, j] += 0.08

            # Positional bias - attention to earlier tokens
            for j in range(min(i, 5)):
                attention[i, j] += 0.03 * (1 - j / 5.0)

            # Articles and determiners get some attention
            for j in range(n):
                target_spacy = token_to_spacy[j]
                if target_spacy:
                    target_token = doc[target_spacy[0]]
                    if target_token.pos_ == 'DET':
                        attention[i, j] += 0.05

    return "program_L4H9", make_row_stochastic(attention)



def program_L5H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Entity and structure-focused attention head that emphasizes proper nouns and key structural positions."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Helper to check if a token is a proper noun
    def is_proper_noun(token_idx):
        spacy_indices = token_to_spacy[token_idx]
        if not spacy_indices:
            return False
        return any(doc[si].pos_ == "PROPN" for si in spacy_indices)

    # Helper to check if token is punctuation
    def is_punctuation(token_idx):
        token = tokens[token_idx].strip()
        return token in ".,!?;:"

    # Helper to check if token is special token
    def is_special_token(token_idx):
        token = tokens[token_idx].strip()
        return token in ["[CLS]", "[SEP]"]

    for i in range(n):
        # Base self-attention
        attention[i, i] = 0.2

        # Strong attention to [CLS] from most positions
        if i > 0:
            attention[i, 0] = 0.3

        # Special handling for final punctuation
        if i == n - 1 or (i == n - 2 and tokens[n-1].strip() == "[SEP]"):
            if is_punctuation(i):
                # Final punctuation attends strongly to proper nouns and [CLS]
                attention[i, 0] = 0.4  # Strong to [CLS]
                for j in range(n):
                    if is_proper_noun(j):
                        attention[i, j] = 0.6
                    elif j == i:
                        attention[i, i] = 0.3

        # Proper nouns get and give special attention
        if is_proper_noun(i):
            attention[i, 0] = 0.4  # Proper nouns attend to [CLS]
            attention[i, i] = 0.5  # Strong self-attention

            # Other tokens attend to proper nouns
            for j in range(n):
                if j != i:
                    attention[j, i] += 0.2

        # Handle adjacent token relationships
        if i > 0:
            # Tokens often attend to previous token
            attention[i, i-1] = 0.3

            # Check for syntactic relationships using spacy
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                current_spacy = doc[spacy_indices[0]]

                # Prepositions attend to their objects
                if current_spacy.pos_ == "ADP" and i + 1 < n:
                    attention[i, i + 1] = 0.4

                # Determiners and adjectives attend to following nouns
                if current_spacy.pos_ in ["DET", "ADJ"] and i + 1 < n:
                    next_spacy_indices = token_to_spacy[i + 1]
                    if next_spacy_indices:
                        next_spacy = doc[next_spacy_indices[0]]
                        if next_spacy.pos_ in ["NOUN", "PROPN"]:
                            attention[i, i + 1] = 0.4

        # Special tokens get attention
        if is_special_token(i):
            attention[i, i] = 0.5
            for j in range(n):
                if j != i:
                    attention[j, i] += 0.1

    return "program_L5H0", make_row_stochastic(attention)



def program_L5H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic dependency attention head with enhanced function word and dependency handling."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy parse and alignment
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        token = tokens[i].strip()

        # [SEP] token attends almost entirely to itself
        if token == '[SEP]':
            attention[i, i] = 1.0
            continue

        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i]

        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]

            # Punctuation patterns
            if token in ['.', '!', '?']:
                # Period attends to the last content word with high weight
                for j in range(i-1, -1, -1):
                    if j < len(tokens):
                        j_spacy = token_to_spacy[j]
                        if j_spacy and doc[j_spacy[0]].pos_ in ['NOUN', 'PROPN', 'VERB']:
                            attention[i, j] = 0.6
                            break
                attention[i, i] = 0.2
                attention[i, 0] = 0.2  # Some attention to [CLS]

            elif token == ',':
                # Comma attends to preceding word with moderate weight
                if i > 0:
                    attention[i, i-1] = 0.4
                attention[i, i] = 0.3
                attention[i, 0] = 0.3

            # Enhanced function word patterns
            elif spacy_tok.pos_ in ['ADP', 'PART', 'AUX', 'DET'] or spacy_tok.dep_ in ['aux', 'auxpass', 'prep', 'prt', 'det']:
                # Function words attend strongly to their syntactic head
                head_found = False

                if spacy_tok.head != spacy_tok and spacy_tok.head.i < len(doc):
                    head_spacy_idx = spacy_tok.head.i
                    if head_spacy_idx in spacy_to_token:
                        head_token_indices = spacy_to_token[head_spacy_idx]
                        if head_token_indices:
                            head_idx = head_token_indices[0]
                            if head_idx < n:
                                attention[i, head_idx] = 0.7
                                head_found = True

                # Look for nearby content words if no syntactic head found
                if not head_found:
                    for j in range(max(0, i-2), min(n, i+4)):
                        if j != i and j < n:
                            j_spacy = token_to_spacy[j]
                            if j_spacy:
                                j_tok = doc[j_spacy[0]]
                                if j_tok.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                                    attention[i, j] = 0.5
                                    head_found = True
                                    break

                if head_found:
                    attention[i, i] = 0.2
                    attention[i, 0] = 0.1
                else:
                    attention[i, i] = 0.5
                    attention[i, 0] = 0.5

            # Content word patterns
            elif spacy_tok.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
                # Look for syntactic dependencies
                head_found = False

                # Find syntactic head or dependent
                if spacy_tok.head != spacy_tok and spacy_tok.head.i < len(doc):
                    head_spacy_idx = spacy_tok.head.i
                    if head_spacy_idx in spacy_to_token:
                        head_token_indices = spacy_to_token[head_spacy_idx]
                        if head_token_indices:
                            head_idx = head_token_indices[0]
                            if head_idx < n:
                                attention[i, head_idx] = 0.4
                                head_found = True

                # Also look for children/dependents
                if not head_found:
                    for child in spacy_tok.children:
                        if child.i in spacy_to_token:
                            child_token_indices = spacy_to_token[child.i]
                            if child_token_indices:
                                child_idx = child_token_indices[0]
                                if child_idx < n:
                                    attention[i, child_idx] = 0.4
                                    head_found = True
                                    break

                # Look for strong local dependencies (verb-object, adj-noun)
                if not head_found:
                    for j in range(max(0, i-3), min(n, i+3)):
                        if j != i and j < n:
                            j_spacy = token_to_spacy[j]
                            if j_spacy:
                                j_tok = doc[j_spacy[0]]

                                # Verb attending to following noun/object
                                if spacy_tok.pos_ == 'VERB' and j > i and j_tok.pos_ in ['NOUN', 'PROPN']:
                                    attention[i, j] = 0.3
                                    head_found = True
                                    break
                                # Adjective attending to following noun
                                elif spacy_tok.pos_ == 'ADJ' and j > i and j_tok.pos_ in ['NOUN', 'PROPN']:
                                    attention[i, j] = 0.4
                                    head_found = True
                                    break
                                # Preposition attending to following noun
                                elif spacy_tok.pos_ == 'ADP' and j > i and j_tok.pos_ in ['NOUN', 'PROPN']:
                                    attention[i, j] = 0.3
                                    head_found = True
                                    break

                # Fallback attention
                if head_found:
                    attention[i, i] = 0.3
                    attention[i, 0] = 0.3  # [CLS]
                else:
                    attention[i, i] = 0.4
                    attention[i, 0] = 0.6

            else:
                # Other tokens (pronouns, etc.)
                attention[i, i] = 0.5
                attention[i, 0] = 0.5
        else:
            # No spacy alignment - basic fallback
            attention[i, i] = 0.7
            attention[i, 0] = 0.3

    return "program_L5H1", make_row_stochastic(attention)



def program_L5H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Backward sequential attention with enhanced coordination and conjunction handling."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for syntactic information
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Special handling for [SEP] token
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.9
            # Distribute remaining weight to nearby tokens
            if i > 0:
                attention[i, i-1] = 0.05
            if i > 1:
                attention[i, i-2] = 0.03
            continue

        # Base backward sequential pattern - strong attention to previous token
        if i > 0:
            attention[i, i-1] = 0.4

        # Attention to [CLS] token (first token) - reduced for content words
        if i > 0:
            # Check if this is a content word via spacy
            is_content_word = False
            if i < len(token_to_spacy) and token_to_spacy[i]:
                spacy_idx = token_to_spacy[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                        is_content_word = True

            # Reduce [CLS] attention for content words
            if is_content_word:
                attention[i, 0] = 0.05
            else:
                attention[i, 0] = 0.15

        # Self-attention (moderate)
        attention[i, i] = 0.1

        # Enhanced handling for conjunctions and coordinating elements
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Strong backward attention for conjunctions
                if spacy_token.pos_ == 'CCONJ' or (spacy_token.pos_ == 'CONJ') or tokens[i].lower() in ['and', ',']:
                    # Look for coordinated elements further back
                    for j in range(max(0, i-5), i):
                        if j < len(token_to_spacy) and token_to_spacy[j]:
                            back_spacy_idx = token_to_spacy[j][0]
                            if back_spacy_idx < len(doc):
                                back_token = doc[back_spacy_idx]
                                # Attend to content words that could be coordinated
                                if back_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                                    distance_decay = 0.8 ** (i - j - 1)
                                    attention[i, j] += 0.2 * distance_decay

                # Attend to syntactic head
                if spacy_token.head != spacy_token and spacy_token.head.i in spacy_to_token:
                    head_token_indices = spacy_to_token[spacy_token.head.i]
                    for head_idx in head_token_indices:
                        if head_idx < n:
                            attention[i, head_idx] += 0.2

                # Attend to direct children
                for child in spacy_token.children:
                    if child.i in spacy_to_token:
                        child_token_indices = spacy_to_token[child.i]
                        for child_idx in child_token_indices:
                            if child_idx < n:
                                attention[i, child_idx] += 0.1

        # Additional backward attention (2-3 positions back) with decay
        if i > 1:
            attention[i, i-2] = 0.08
        if i > 2:
            attention[i, i-3] = 0.04

        # Slight forward attention for function words
        if i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['DET', 'PREP', 'CONJ', 'AUX']:
                    if i + 1 < n:
                        attention[i, i+1] = 0.05

    return "program_L5H10", make_row_stochastic(attention)



def program_L5H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Positional anchoring head with enhanced self/[CLS] attention for content words."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Align tokens to spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Special handling for [SEP] token - extremely high self-attention
        if token == '[SEP]':
            attention_matrix[i, i] = 0.9
            # Small residual attention to other tokens
            for j in range(n):
                if j != i:
                    if tokens[j] == '[CLS]':
                        attention_matrix[i, j] = 0.012
                    elif j == n - 2:  # Period before [SEP]
                        attention_matrix[i, j] = 0.01
                    else:
                        attention_matrix[i, j] = 0.01
            continue

        # Special handling for [CLS] token - moderate self-attention
        if token == '[CLS]':
            attention_matrix[i, i] = 0.025
            continue

        # Get spacy features for current token
        spacy_indices = token_to_spacy[i]
        is_punct = len(spacy_indices) > 0 and any(doc[si].pos_ == 'PUNCT' for si in spacy_indices)
        is_function = len(spacy_indices) > 0 and any(doc[si].pos_ in ['DET', 'ADP', 'PRON', 'AUX', 'CCONJ'] for si in spacy_indices)
        is_content = len(spacy_indices) > 0 and any(doc[si].pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] for si in spacy_indices)

        # Base attention weights
        cls_weight = 0.02
        self_weight = 0.02

        # Enhanced attention for content words - key insight from failure cases
        if is_content and not is_punct:
            cls_weight = 0.06  # Increased from 0.035
            self_weight = 0.08  # Significantly increased from 0.03
        # Modulate based on token characteristics
        elif is_punct:
            if token in ['.', ',']:
                cls_weight = 0.08
                self_weight = 0.12
            else:
                cls_weight = 0.05
                self_weight = 0.05
        elif is_function:
            cls_weight = 0.04
            self_weight = 0.025
        else:
            # Other tokens
            cls_weight = 0.035
            self_weight = 0.03

        # Set [CLS] attention
        attention_matrix[i, 0] = cls_weight

        # Set self-attention
        attention_matrix[i, i] = self_weight

        # Special cases for specific patterns observed
        if len(spacy_indices) > 0:
            spacy_token = doc[spacy_indices[0]]

            # Conjunctions attend more to themselves
            if spacy_token.pos_ == 'CCONJ':
                attention_matrix[i, i] = 0.09

            # Verbs in certain contexts
            if spacy_token.pos_ == 'VERB' and spacy_token.dep_ == 'ROOT':
                attention_matrix[i, 0] = 0.05

        # Distribute remaining attention with reduced uniform spreading
        remaining = 1.0 - attention_matrix[i].sum()
        if remaining > 0:
            # Reduce uniform distribution to combat over-prediction
            base_remaining = remaining / (n * 1.5)  # Reduced spreading factor
            for j in range(n):
                if j != i and j != 0:  # Not self or [CLS]
                    attention_matrix[i, j] = base_remaining

                    # Small boost for nearby tokens
                    if abs(j - i) == 1:
                        attention_matrix[i, j] *= 1.2

                    # Small boost for similar tokens
                    if embedding_similarity(tokens, i, j) > 0.7:
                        attention_matrix[i, j] *= 1.1

    return "program_L5H11", make_row_stochastic(attention_matrix)



def program_L5H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Discourse entity and syntactic relationship attention head with enhanced function word and cross-clausal attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i]
        spacy_tokens = [doc[idx] for idx in spacy_indices if idx < len(doc)]

        # Handle special tokens
        if token in ['[CLS]', '[SEP]']:
            if token == '[CLS]':
                attention[i, i] = 1.0
            elif token == '[SEP]':
                attention[i, i] = 1.0
            continue

        # Handle punctuation with enhanced [CLS] attention
        if token.strip() in [',', '.', '!', '?', '"', "'"]:
            # Punctuation attends to subjects, main verbs, and content words
            for j in range(n):
                if i == j:
                    attention[i, j] = 0.2  # Reduced self-attention
                elif j == 0:  # Enhanced [CLS] attention for punctuation
                    attention[i, j] = 0.3
                else:
                    other_spacy = token_to_spacy[j]
                    if other_spacy:
                        other_token = doc[other_spacy[0]]
                        # Attend to subjects, main content
                        if other_token.pos_ in ['NOUN', 'PRON', 'PROPN'] and 'subj' in other_token.dep_:
                            attention[i, j] = 0.3
                        elif other_token.pos_ in ['VERB'] and other_token.dep_ == 'ROOT':
                            attention[i, j] = 0.15
                        elif other_token.pos_ in ['NOUN', 'PROPN']:
                            attention[i, j] = 0.1
            continue

        if spacy_tokens:
            spacy_token = spacy_tokens[0]

            # Pronouns attend strongly to their antecedents and subjects
            if spacy_token.pos_ == 'PRON':
                for j in range(n):
                    if i == j:
                        attention[i, j] = 0.1  # Reduced self-attention
                    elif j == 0:  # [CLS]
                        attention[i, j] = 0.1
                    else:
                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]
                            # Strong attention to subjects and main nouns
                            if other_token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                                if 'subj' in other_token.dep_:
                                    attention[i, j] = 0.5
                                else:
                                    attention[i, j] = 0.2
                            elif other_token.pos_ == 'VERB' and other_token.dep_ == 'ROOT':
                                attention[i, j] = 0.1

            # Determiners attend to their head nouns and discourse entities
            elif spacy_token.pos_ == 'DET':
                for j in range(n):
                    if i == j:
                        attention[i, j] = 0.1
                    else:
                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]
                            # Attend to head noun
                            if other_token == spacy_token.head:
                                attention[i, j] = 0.4
                            # Attend to other content words and subjects
                            elif other_token.pos_ in ['NOUN', 'PROPN'] and 'subj' in other_token.dep_:
                                attention[i, j] = 0.3
                            elif other_token.pos_ in ['NOUN', 'PROPN']:
                                attention[i, j] = 0.2

            # Verbs attend to their subjects and objects
            elif spacy_token.pos_ == 'VERB':
                for j in range(n):
                    if i == j:
                        attention[i, j] = 0.15  # Reduced self-attention
                    elif j == 0:  # [CLS]
                        attention[i, j] = 0.1
                    else:
                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]
                            # Strong attention to subjects
                            if 'subj' in other_token.dep_:
                                attention[i, j] = 0.4
                            # Attention to objects
                            elif 'obj' in other_token.dep_:
                                attention[i, j] = 0.2
                            elif other_token.pos_ in ['PRON', 'NOUN', 'PROPN']:
                                attention[i, j] = 0.1

            # Nouns attend to modifiers and related entities
            elif spacy_token.pos_ in ['NOUN', 'PROPN']:
                for j in range(n):
                    if i == j:
                        attention[i, j] = 0.15  # Reduced self-attention
                    elif j == 0:  # [CLS]
                        attention[i, j] = 0.1
                    else:
                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]
                            # Attend to determiners, adjectives, and other nouns
                            if other_token.dep_ == 'det' and other_token.head == spacy_token:
                                attention[i, j] = 0.3
                            elif other_token.dep_ == 'amod' and other_token.head == spacy_token:
                                attention[i, j] = 0.2
                            elif other_token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                                sim = embedding_similarity(tokens, i, j)
                                attention[i, j] = max(0.1, 0.1 + 0.1 * sim)

            # Enhanced function words - attend strongly to verbs and content words
            elif spacy_token.pos_ in ['ADP', 'CONJ', 'CCONJ', 'SCONJ'] or spacy_token.text.lower() == 'to':
                for j in range(n):
                    if i == j:
                        attention[i, j] = 0.05  # Minimal self-attention for function words
                    else:
                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]
                            # Strong attention to verbs, especially for infinitive "to"
                            if other_token.pos_ == 'VERB':
                                if spacy_token.text.lower() == 'to' and other_token.dep_ in ['ROOT', 'ccomp', 'xcomp']:
                                    attention[i, j] = 0.5
                                elif other_token.dep_ == 'ROOT':
                                    attention[i, j] = 0.4
                                else:
                                    attention[i, j] = 0.3
                            # Attention to subjects and main content
                            elif other_token.pos_ in ['NOUN', 'PROPN', 'PRON'] and 'subj' in other_token.dep_:
                                attention[i, j] = 0.3
                            elif other_token.pos_ in ['NOUN', 'PROPN']:
                                attention[i, j] = 0.2

            # Default pattern for other tokens
            else:
                for j in range(n):
                    if i == j:
                        attention[i, j] = 0.2  # Reduced self-attention
                    elif j == 0:  # [CLS]
                        attention[i, j] = 0.2
                    else:
                        # General attention to content words
                        other_spacy = token_to_spacy[j]
                        if other_spacy:
                            other_token = doc[other_spacy[0]]
                            if other_token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                                attention[i, j] = 0.1

        # If no spacy alignment, use simple fallback
        else:
            attention[i, i] = 0.4  # Reduced self-attention
            attention[i, 0] = 0.2  # Attend to [CLS]
            # Distribute remaining attention
            remaining = 0.4 / max(1, n - 2)
            for j in range(1, n):
                if j != i:
                    attention[i, j] = remaining

    return "program_L5H2", make_row_stochastic(attention)



def program_L5H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic structure head: coordinates clause boundaries and connects conjunctions to verbs."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        # Special case: [SEP] tokens have very strong self-attention
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.95
            # Minimal attention to other positions
            attention[i, 0] = 0.03
            attention[i, max(0, i-1)] = 0.02
            continue

        # Get spacy tokens aligned to this position
        spacy_indices = alignment[i]
        spacy_tokens = [doc[idx] for idx in spacy_indices if idx < len(doc)]

        if not spacy_tokens:
            # Fallback for unaligned tokens
            attention[i, max(0, i-1)] = 0.5
            attention[i, 0] = 0.3
            attention[i, i] = 0.2
            continue

        primary_spacy = spacy_tokens[0]

        # Coordination conjunctions attend strongly to previous verbs
        if primary_spacy.pos_ == "CCONJ":
            for j in range(i):
                j_spacy = alignment[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ == "VERB":
                        attention[i, j] = 0.8
                        break
            else:
                # No verb found, attend to previous content word
                for j in range(i-1, -1, -1):
                    j_spacy = alignment[j]
                    if j_spacy:
                        j_token = doc[j_spacy[0]]
                        if j_token.pos_ in ["NOUN", "VERB", "ADJ"]:
                            attention[i, j] = 0.6
                            break

        # Punctuation patterns
        elif tokens[i] in [".", "!", "?"]:
            # Sentence-final punctuation attends to clause elements
            for j in range(i):
                j_spacy = alignment[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ == "VERB":
                        attention[i, j] = 0.4
                    elif j_token.pos_ in ["NOUN", "PRON"]:
                        attention[i, j] = 0.2
            # Also attend to speech markers
            for j in range(i):
                if tokens[j] in ['"', "'"]:
                    attention[i, j] = 0.3

        elif tokens[i] == ",":
            # Commas attend to verbs and coordination
            for j in range(i):
                j_spacy = alignment[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ == "VERB":
                        attention[i, j] = 0.5
                    elif j_token.pos_ == "CCONJ":
                        attention[i, j] = 0.3

        elif tokens[i] in ['"', "'"]:
            # Quote marks attend to speech verbs and previous quotes
            for j in range(i):
                j_spacy = alignment[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.lemma_ in ["say", "said", "tell", "ask", "reply"]:
                        attention[i, j] = 0.4
                if tokens[j] in ['"', "'"]:
                    attention[i, j] = 0.3

        # Prepositions attend to their objects and heads
        elif primary_spacy.pos_ == "ADP":
            # Attend to syntactic head
            if primary_spacy.head != primary_spacy:
                for j in range(n):
                    j_spacy = alignment[j]
                    if j_spacy and j_spacy[0] == primary_spacy.head.i:
                        attention[i, j] = 0.6
                        break
            # Also attend to previous content words
            for j in range(i-1, max(0, i-5), -1):
                j_spacy = alignment[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ in ["NOUN", "VERB"]:
                        attention[i, j] = 0.2
                        break

        # Auxiliary verbs and particles attend to main verbs
        elif primary_spacy.pos_ in ["AUX", "PART"] or primary_spacy.dep_ == "aux":
            for j in range(i+1, min(n, i+4)):
                j_spacy = alignment[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ == "VERB" and j_token.dep_ != "aux":
                        attention[i, j] = 0.6
                        break
            else:
                # Look backwards for main verb
                for j in range(i-1, max(0, i-4), -1):
                    j_spacy = alignment[j]
                    if j_spacy:
                        j_token = doc[j_spacy[0]]
                        if j_token.pos_ == "VERB":
                            attention[i, j] = 0.4
                            break

        # Determiners attend to their noun heads
        elif primary_spacy.pos_ == "DET":
            for j in range(i+1, min(n, i+3)):
                j_spacy = alignment[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ == "NOUN":
                        attention[i, j] = 0.5
                        break

        # General positional bias and self-attention
        if attention[i].sum() < 0.5:
            # Add positional bias to earlier tokens
            for j in range(max(0, i-3), i):
                attention[i, j] += 0.1
            # First token attention
            attention[i, 0] += 0.1
            # Self-attention
            attention[i, i] += 0.1

    return "program_L5H3", make_row_stochastic(attention)



def program_L5H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention to sentence-initial content words and special positions with positional decay, reduced special token self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Get spacy alignment for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Reduced self-attention for special tokens and punctuation
        if tokens[i] in ['[CLS]', '[SEP]'] or tokens[i].strip() in ['.', ',', '!', '?']:
            # Significantly reduced self-attention for special tokens
            if tokens[i] in ['[CLS]', '[SEP]']:
                attention_matrix[i, i] = 0.4
            else:
                attention_matrix[i, i] = 0.4
            # Distribute remaining attention
            for j in range(n):
                if j != i:
                    distance_penalty = 1.0 / (abs(i - j) + 1)
                    if tokens[i] in ['[CLS]', '[SEP]']:
                        attention_matrix[i, j] = 0.6 * distance_penalty
                    else:
                        attention_matrix[i, j] = 0.6 * distance_penalty
        else:
            # Base attention pattern for content tokens
            for j in range(n):
                weight = 0.0

                if i == j:
                    # Self-attention baseline
                    weight += 0.05

                # Strong attention to [CLS] token
                if j == 0:
                    weight += 0.15

                # Strong attention to [SEP] token (usually last)
                if tokens[j] == '[SEP]':
                    weight += 0.05

                # Reduced attention to sentence-initial content words
                if 1 <= j <= 3 and j < n-1:  # Skip [CLS] and [SEP]
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_token = doc[spacy_indices[0]]
                        if spacy_token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                            weight += 0.06  # Reduced from 0.12

                # Positional decay - prefer earlier tokens
                if j < i:
                    distance = i - j
                    decay = 0.08 / (distance + 1)
                    weight += decay

                # Special attention patterns for conjunctions and punctuation
                if tokens[j].strip() in [',', 'and', 'but']:
                    weight += 0.06

                # Content word similarity boost
                if j != i and j > 0 and i > 0:  # Skip special tokens
                    similarity = embedding_similarity(tokens, i, j)
                    if similarity > 0.5:
                        weight += 0.04 * similarity

                attention_matrix[i, j] = weight

    return "program_L5H4", make_row_stochastic(attention_matrix)



def program_L5H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Self-attention head with first-token bias, semantic cross-references, and strong [SEP] self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special handling for [SEP] token - extremely strong self-attention
        if tokens[i].strip() == '[SEP]':
            attention[i, i] = 0.95
            # [SEP] has minimal attention to other tokens
            if tokens[0] in ['[CLS]', '<s>']:
                attention[i, 0] = 0.03
            # Small residual attention to nearby tokens
            for j in range(max(0, i-2), min(n, i+3)):
                if j != i and attention[i, j] == 0:
                    attention[i, j] = 0.005
            continue

        # Strong self-attention baseline (reduced from original)
        attention[i, i] = 0.06

        # Strong attention to [CLS] token from most positions
        if tokens[0] in ['[CLS]', '<s>']:
            attention[i, 0] = 0.05

        # Special handling for different token types
        token_text = tokens[i].strip()

        # Get spacy info if available
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []
        spacy_token = doc[spacy_indices[0]] if spacy_indices else None

        # Punctuation patterns
        if token_text in ['.', '!', '?']:
            # Period attends strongly to [CLS] and some content words
            if tokens[0] in ['[CLS]', '<s>']:
                attention[i, 0] = 0.08
            # Find important content words to attend to
            for j in range(n):
                j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ in ['NOUN', 'VERB', 'ADJ'] and j < i:
                        attention[i, j] = 0.03

        elif token_text == ',':
            # Commas attend to [CLS] and have moderate self-attention
            if tokens[0] in ['[CLS]', '<s>']:
                attention[i, 0] = 0.04

        # Pronoun resolution patterns
        elif spacy_token and spacy_token.pos_ == 'PRON':
            # Pronouns attend to potential antecedents
            for j in range(i):
                j_spacy = token_to_spacy[j] if j < len(token_to_spacy) else []
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ in ['NOUN', 'PROPN']:
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:
                            attention[i, j] = 0.04

        # Articles and determiners (reduced self-attention)
        elif spacy_token and spacy_token.pos_ in ['DET', 'ADP']:
            # These often have high self-attention but less than original
            attention[i, i] = 0.10

        # Content words (reduced self-attention)
        elif spacy_token and spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
            # Content words have strong self-attention
            attention[i, i] = 0.08

            # Look for semantic relationships
            for j in range(n):
                if i != j:
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.7:  # High similarity
                        attention[i, j] = 0.06
                    elif sim > 0.5:  # Moderate similarity
                        attention[i, j] = 0.03

        # Add some general attention to [CLS] for all tokens
        if tokens[0] in ['[CLS]', '<s>'] and attention[i, 0] < 0.02:
            attention[i, 0] = 0.02

        # Add small baseline attention to nearby tokens
        for j in range(max(0, i-2), min(n, i+3)):
            if attention[i, j] < 0.01:
                attention[i, j] = 0.01

    return "program_L5H5", make_row_stochastic(attention)



def program_L5H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic dependency attention head focusing on verb-argument and modifier relationships."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for syntactic information
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Special token self-attention
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.9
            # [CLS] gets some distributed attention
            if tokens[i] == '[CLS]':
                for j in range(n):
                    if j != i:
                        attention[i, j] = 0.1 / (n - 1)
            continue

        # Get spacy token info for current token
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            attention[i, i] = 0.5
            attention[i, 0] = 0.5  # attend to [CLS]
            continue

        current_spacy = doc[spacy_indices[0]]

        # Syntactic dependency patterns
        high_attention_targets = []

        # For verbs: attend to subjects and objects
        if current_spacy.pos_ in ['VERB', 'AUX']:
            for child in current_spacy.children:
                if child.dep_ in ['nsubj', 'dobj', 'iobj', 'pobj']:
                    if child.i < len(spacy_to_token) and spacy_to_token[child.i]:
                        high_attention_targets.extend(spacy_to_token[child.i])

        # For nouns: attend to their modifiers and heads
        elif current_spacy.pos_ in ['NOUN', 'PRON', 'PROPN']:
            # Attend to head if it's a verb
            if current_spacy.head.pos_ in ['VERB', 'AUX'] and current_spacy.head.i < len(spacy_to_token):
                if spacy_to_token[current_spacy.head.i]:
                    high_attention_targets.extend(spacy_to_token[current_spacy.head.i])

            # Attend to modifiers
            for child in current_spacy.children:
                if child.dep_ in ['amod', 'det']:
                    if child.i < len(spacy_to_token) and spacy_to_token[child.i]:
                        high_attention_targets.extend(spacy_to_token[child.i])

        # For prepositions: attend to their objects
        elif current_spacy.pos_ == 'ADP':
            for child in current_spacy.children:
                if child.dep_ == 'pobj':
                    if child.i < len(spacy_to_token) and spacy_to_token[child.i]:
                        high_attention_targets.extend(spacy_to_token[child.i])

        # For infinitive 'to': attend to the verb
        elif current_spacy.text.lower() == 'to' and current_spacy.pos_ == 'PART':
            for child in current_spacy.children:
                if child.pos_ == 'VERB':
                    if child.i < len(spacy_to_token) and spacy_to_token[child.i]:
                        high_attention_targets.extend(spacy_to_token[child.i])

        # Possessive patterns
        if tokens[i] == "'s" or tokens[i] == "s" and i > 0 and tokens[i-1] == "'":
            if i > 0:
                high_attention_targets.append(i-1)
        elif tokens[i] == "'" and i < n-1 and (tokens[i+1] == "s" or tokens[i+1] == "re"):
            if i > 0:
                high_attention_targets.append(i-1)

        # Punctuation patterns
        if tokens[i] in ['.', '!', '?']:
            attention[i, i] = 0.3
            attention[i, 0] = 0.2  # attend to [CLS]
            # Find the main verb of the sentence
            for j in range(n):
                if j != i and token_to_spacy[j]:
                    spacy_j = doc[token_to_spacy[j][0]]
                    if spacy_j.pos_ in ['VERB', 'AUX'] and spacy_j.dep_ == 'ROOT':
                        attention[i, j] = 0.3
                        break
            continue

        # Apply high attention to syntactic targets
        if high_attention_targets:
            base_weight = 0.6 / len(high_attention_targets)
            for target in high_attention_targets:
                if 0 <= target < n:
                    attention[i, target] += base_weight

        # Self attention
        attention[i, i] = 0.1

        # Baseline attention to [CLS]
        attention[i, 0] = 0.1

        # Local context (previous token)
        if i > 0:
            attention[i, i-1] = 0.05

    return "program_L5H6", make_row_stochastic(attention)



def program_L5H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Structural attention head focusing on special tokens, punctuation, and sentence boundaries with content word linking."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse sentence for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Reduced self-attention for special tokens (was massively over-predicted)
        if token in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.03  # Reduced from 0.99
            continue

        # Reduced self-attention for sentence-ending punctuation
        if token in ['.', '!', '?']:
            attention[i, i] = 0.08  # Reduced from 0.15

            # From punctuation, attend to various content words
            for j in range(n):
                if i != j:
                    target_token = tokens[j]

                    # Reduced attention to [CLS]
                    if target_token == '[CLS]':
                        attention[i, j] = 0.02  # Reduced from 0.06

                    # Attend to verbs and important content words
                    elif j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_indices = token_to_spacy[j]
                        for spacy_idx in spacy_indices:
                            if spacy_idx < len(doc):
                                spacy_token = doc[spacy_idx]
                                if spacy_token.pos_ in ['VERB', 'ADJ', 'NOUN']:
                                    attention[i, j] += 0.03
                                elif spacy_token.pos_ in ['ADV']:
                                    attention[i, j] += 0.02

                    # Some attention to conjunctions and other structural words
                    if target_token in [',', 'and', 'but']:
                        attention[i, j] = 0.02

        # Moderate self-attention for content words
        elif i < len(token_to_spacy) and token_to_spacy[i]:
            spacy_indices = token_to_spacy[i]
            is_content = False
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']:
                        is_content = True
                        break

            if is_content:
                attention[i, i] = 0.05
            else:
                attention[i, i] = 0.03

        else:
            attention[i, i] = 0.02

        # Reduced general attention to [CLS] from all positions
        for j in range(n):
            if tokens[j] == '[CLS]' and i != j:
                attention[i, j] = 0.015  # Reduced from 0.04

        # Attention between similar tokens
        for j in range(n):
            if i != j and i < n and j < n:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:
                    attention[i, j] += 0.02
                elif sim > 0.5:
                    attention[i, j] += 0.01

        # Local positional attention (previous token)
        if i > 0:
            attention[i, i-1] += 0.01

        # Attention from conjunctions to nearby content
        if token in ['and', 'but']:
            attention[i, i] = 0.05
            # Reduced attention to [CLS]
            for j in range(n):
                if tokens[j] == '[CLS]':
                    attention[i, j] = 0.02  # Reduced from 0.06
            # Attend to nearby verbs and nouns
            for j in range(max(0, i-3), min(n, i+4)):
                if i != j and j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_indices = token_to_spacy[j]
                    for spacy_idx in spacy_indices:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ['VERB', 'NOUN']:
                                attention[i, j] += 0.02

    return "program_L5H7", make_row_stochastic(attention)



def program_L5H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content word attention head with special sentence boundary handling - attends to verbs, important nouns, and self with end-of-sentence aggregation."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Identify important content words and their positions
    verb_positions = set()
    noun_positions = set()
    important_positions = set()
    punctuation_positions = set()

    for i, spacy_indices in enumerate(token_to_spacy):
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            if spacy_tok.pos_ in ['VERB', 'AUX']:
                verb_positions.add(i)
                important_positions.add(i)
            elif spacy_tok.pos_ in ['NOUN', 'PROPN']:
                noun_positions.add(i)
                important_positions.add(i)
            elif spacy_tok.pos_ in ['ADJ', 'ADV']:
                important_positions.add(i)

        # Check for punctuation
        if tokens[i].strip() in ['.', '!', '?', ',']:
            punctuation_positions.add(i)

    # Add [CLS] as important if present
    if tokens[0] == '[CLS]':
        important_positions.add(0)

    for i in range(n):
        # Special handling for [SEP] tokens - extremely strong self-attention
        if tokens[i] == '[SEP]':
            for j in range(n):
                if i == j:
                    attention_matrix[i, j] = 0.97  # Extremely strong self-attention
                else:
                    attention_matrix[i, j] = 0.03 / (n - 1)  # Minimal attention to others
            continue

        # Strong self-attention for most tokens
        attention_matrix[i, i] = 0.05

        # Special behavior for end punctuation
        if i in punctuation_positions and tokens[i].strip() in ['.', '!']:
            # End punctuation attends strongly to verbs and important content
            for j in range(n):
                if j in verb_positions:
                    attention_matrix[i, j] = 0.08
                elif j in important_positions:
                    attention_matrix[i, j] = 0.04
                elif tokens[j] == '[CLS]':
                    attention_matrix[i, j] = 0.03
                else:
                    attention_matrix[i, j] = 0.015

        # Regular content word attention patterns
        elif i in important_positions:
            for j in range(n):
                if i == j:
                    attention_matrix[i, j] = 0.06  # Stronger self-attention for content words
                elif j in verb_positions and i != j:
                    # Content words attend to verbs
                    attention_matrix[i, j] = 0.04
                elif j in important_positions and i != j:
                    # Moderate attention between content words
                    attention_matrix[i, j] = 0.025
                elif tokens[j] == '[CLS]':
                    attention_matrix[i, j] = 0.02
                else:
                    attention_matrix[i, j] = 0.015

        # Function words and other tokens
        else:
            for j in range(n):
                if j in verb_positions:
                    attention_matrix[i, j] = 0.03
                elif j in important_positions:
                    attention_matrix[i, j] = 0.02
                elif tokens[j] == '[CLS]':
                    attention_matrix[i, j] = 0.015
                else:
                    attention_matrix[i, j] = 0.01

        # Boost attention to semantically similar tokens
        for j in range(n):
            if i != j and tokens[i] != '[SEP]':  # Don't modify SEP attention
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity
                    attention_matrix[i, j] *= 1.5
                elif sim > 0.5:  # Moderate similarity
                    attention_matrix[i, j] *= 1.2

    return "program_L5H8", make_row_stochastic(attention_matrix)



def program_L5H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head detecting sentence boundaries, possessive relationships, and syntactic dependencies."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong self-attention baseline for all tokens
        attention[i, i] = 0.3

        # Special handling for [CLS] and [SEP] tokens
        if tokens[i] == '[CLS]':
            attention[i, i] = 0.8  # Very strong self-attention
            continue
        elif tokens[i] == '[SEP]':
            attention[i, i] = 0.9  # Extremely strong self-attention
            # [SEP] also attends to [CLS] and period tokens
            if '[CLS]' in tokens:
                cls_idx = tokens.index('[CLS]')
                attention[i, cls_idx] = 0.4
            for j in range(n):
                if tokens[j] == '.':
                    attention[i, j] = 0.1
            continue

        # Period tokens attend strongly to proper nouns
        if tokens[i] == '.':
            attention[i, i] = 0.2  # Moderate self-attention
            for j in range(n):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_tokens = [doc[idx] for idx in token_to_spacy[j]]
                    if any(t.pos_ == 'PROPN' for t in spacy_tokens):
                        attention[i, j] = 0.6  # Strong attention to proper nouns
            continue

        # Possessive tokens attend to their head noun
        if tokens[i] in ["'s", "'"]:
            if i > 0:
                attention[i, i-1] = 0.7  # Strong attention to preceding token
            continue

        # Regular tokens: attend to syntactically related words
        if i < len(token_to_spacy) and token_to_spacy[i]:
            current_spacy = [doc[idx] for idx in token_to_spacy[i]]

            for j in range(n):
                if i == j:
                    continue

                if j < len(token_to_spacy) and token_to_spacy[j]:
                    target_spacy = [doc[idx] for idx in token_to_spacy[j]]

                    # Check for syntactic relationships
                    for curr_tok in current_spacy:
                        for tgt_tok in target_spacy:
                            # Head-dependent relationships
                            if curr_tok.head == tgt_tok or tgt_tok.head == curr_tok:
                                attention[i, j] += 0.15

                            # Subject-verb relationships
                            if (curr_tok.dep_ in ['nsubj', 'nsubjpass'] and tgt_tok.pos_ == 'VERB') or \
                               (tgt_tok.dep_ in ['nsubj', 'nsubjpass'] and curr_tok.pos_ == 'VERB'):
                                attention[i, j] += 0.2

        # Attention to [CLS] token as a fallback
        if '[CLS]' in tokens:
            cls_idx = tokens.index('[CLS]')
            if i != cls_idx:
                attention[i, cls_idx] = 0.1

    return "program_L5H9", make_row_stochastic(attention)



def program_L6H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic content word attention head focusing on verbs and important tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Helper function to get spacy features for a token
    def get_spacy_features(token_idx):
        spacy_indices = token_to_spacy[token_idx]
        if not spacy_indices:
            return None, None
        spacy_token = doc[spacy_indices[0]]
        return spacy_token.pos_, spacy_token.dep_

    # Helper function to check if token is punctuation
    def is_punct(token):
        return token.strip() in '.,!?;:'

    # Helper function to check if token is special
    def is_special(token):
        return token in ['[CLS]', '[SEP]']

    for i in range(n):
        query_token = tokens[i]
        query_pos, query_dep = get_spacy_features(i)

        # Special handling for [SEP] - strong self-attention
        if is_special(query_token):
            attention[i, i] = 1.0
            continue

        # For each potential key
        for j in range(n):
            key_token = tokens[j]
            key_pos, key_dep = get_spacy_features(j)

            # Base attention weight
            weight = 0.02

            # Self-attention baseline
            if i == j:
                weight += 0.05

            # Strong verb-to-verb attention
            if (query_pos in ['VERB', 'AUX'] and key_pos in ['VERB', 'AUX'] and 
                i != j):
                weight += 0.6

            # Punctuation attends to verbs and important content
            if is_punct(query_token):
                if key_pos in ['VERB', 'AUX']:
                    weight += 0.3
                elif key_pos in ['NOUN', 'PROPN']:
                    weight += 0.15
                elif key_pos in ['ADJ', 'ADV']:
                    weight += 0.1

            # Pronouns and articles attend to nearby nouns
            if query_pos in ['PRON', 'DET']:
                if key_pos in ['NOUN', 'PROPN']:
                    # Stronger if nearby
                    distance = abs(i - j)
                    if distance <= 3:
                        weight += 0.2 / (1 + distance * 0.1)

            # Prepositions attend to their objects
            if query_pos == 'ADP' and key_pos in ['NOUN', 'PROPN']:
                if j > i:  # Look ahead for object
                    weight += 0.15

            # Content words attend to semantically similar content words
            if (query_pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ'] and 
                key_pos in ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ']):
                similarity = embedding_similarity(tokens, i, j)
                if similarity > 0.3:
                    weight += similarity * 0.2

            # Coordinate conjunctions attend to coordinated elements
            if query_token.lower() == 'and':
                if key_pos in ['VERB', 'AUX', 'NOUN', 'PROPN', 'ADJ']:
                    weight += 0.4

            # Positional bias - slight preference for earlier tokens
            if j < i:
                weight += 0.02

            # Boost for first content word
            if j == 1 and key_pos in ['NOUN', 'PROPN', 'VERB']:  # Skip [CLS]
                weight += 0.1

            attention[i, j] = weight

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L6H0", attention



def program_L6H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that combines early-position bias with verb-focused syntactic attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Helper to get POS tag for a token
    def get_pos(token_idx):
        spacy_indices = token_to_spacy[token_idx]
        if spacy_indices:
            return doc[spacy_indices[0]].pos_
        return "X"

    # Helper to check if token is verb
    def is_verb(token_idx):
        return get_pos(token_idx) in ["VERB", "AUX"]

    # Helper to check if token is punctuation
    def is_punct(token_idx):
        return tokens[token_idx].strip() in [".", ",", "!", "?", ";", ":"]

    for i in range(n):
        # Special case: [SEP] tokens have very high self-attention
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.9
            # Small residual attention to [CLS]
            attention[i, 0] = 0.1
            continue

        # Base attention distribution with strong early position bias
        for j in range(n):
            if j == 0:  # [CLS] token gets moderate attention
                attention[i, j] = 0.15
            elif j == 1 and n > 1:  # First content token gets high attention
                attention[i, j] = 0.25
            elif j < i:  # Earlier positions get decreasing attention
                distance = i - j
                attention[i, j] = max(0.1, 0.2 / distance)
            elif j == i:  # Self-attention
                attention[i, j] = 0.1
            else:  # Later positions get small attention
                attention[i, j] = 0.05

        # Boost attention to verbs, especially early verbs
        for j in range(n):
            if is_verb(j):
                if j <= 3:  # Early verbs get strong boost
                    attention[i, j] *= 2.0
                else:  # Later verbs get moderate boost
                    attention[i, j] *= 1.5

        # Special patterns for punctuation
        if is_punct(i):
            # Punctuation attends strongly to early verbs
            for j in range(min(n, 5)):  # Focus on first few tokens
                if is_verb(j):
                    attention[i, j] *= 3.0
            # Also boost attention to position 1
            if n > 1:
                attention[i, 1] *= 2.0

        # Conditional/modal patterns
        token_lower = tokens[i].lower().strip()
        if token_lower in ["could", "would", "should", "might", "can", "ask"]:
            # Look for "if" tokens and boost attention
            for j in range(n):
                if tokens[j].lower().strip() == "if":
                    attention[i, j] *= 5.0

        # Boost attention to commas for conjunctions
        if get_pos(i) in ["CCONJ", "SCONJ"]:  # and, or, but, etc.
            for j in range(n):
                if tokens[j].strip() == ",":
                    attention[i, j] *= 2.0

    return "program_L6H1", make_row_stochastic(attention)



def program_L6H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic dependency head with backward attention to content words and high SEP self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Special handling for [SEP] token (usually last)
        if tokens[i].strip() == '[SEP]':
            attention[i, i] = 1.0
            continue

        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i]

        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]

            # For verbs, attend strongly to subjects and objects
            if spacy_tok.pos_ in ['VERB', 'AUX']:
                for child in spacy_tok.children:
                    if child.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'iobj', 'pobj']:
                        child_tokens = spacy_to_token[child.i]
                        for j in child_tokens:
                            attention[i, j] += 0.4

            # For prepositions, attend to their objects
            elif spacy_tok.pos_ == 'ADP':
                for child in spacy_tok.children:
                    if child.dep_ == 'pobj':
                        child_tokens = spacy_to_token[child.i]
                        for j in child_tokens:
                            attention[i, j] += 0.6

            # For nouns, attend to their modifiers and determiners
            elif spacy_tok.pos_ in ['NOUN', 'PROPN']:
                for child in spacy_tok.children:
                    if child.dep_ in ['amod', 'det', 'compound']:
                        child_tokens = spacy_to_token[child.i]
                        for j in child_tokens:
                            attention[i, j] += 0.2

        # Backward attention to content words (verbs, nouns)
        for j in range(i):
            j_spacy = token_to_spacy[j]
            if j_spacy and doc[j_spacy[0]].pos_ in ['VERB', 'NOUN', 'PROPN']:
                distance_decay = 1.0 / (1.0 + 0.1 * (i - j))
                attention[i, j] += 0.15 * distance_decay

        # Punctuation patterns
        if tokens[i].strip() == ',':
            # Commas attend to sentence beginning tokens
            for j in range(min(3, i)):
                if tokens[j].strip() not in ['[CLS]', ',']:
                    attention[i, j] += 0.3

        elif tokens[i].strip() == '.':
            # Periods attend to main verbs and important nouns
            for j in range(i):
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    spacy_j = doc[j_spacy[0]]
                    if spacy_j.pos_ == 'VERB':
                        attention[i, j] += 0.25
                    elif spacy_j.pos_ in ['NOUN', 'PROPN']:
                        attention[i, j] += 0.1

        # Self-attention for content words
        i_spacy = token_to_spacy[i]
        if i_spacy and doc[i_spacy[0]].pos_ in ['NOUN', 'VERB', 'ADJ']:
            attention[i, i] += 0.05

        # Small uniform attention to all tokens
        attention[i, :] += 0.01

    return "program_L6H10", make_row_stochastic(attention)



def program_L6H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Previous-token attention head with dependency-boosted immediate predecessors."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy parse for dependency information
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Special tokens get high self-attention
        if token in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.8
            # [SEP] also attends moderately to last content token
            if token == '[SEP]' and i > 0:
                attention[i, i-1] = 0.15
            continue

        # Punctuation attends strongly to previous token
        if token in [',', '.', '?', '"', "'", '!']:
            if i > 0:
                attention[i, i-1] = 0.85
            attention[i, i] = 0.1
            continue

        # Main pattern: attend to previous token
        prev_attention = 0.7

        # Boost attention to previous token if there's a strong dependency
        if i > 0 and token_to_spacy[i] and token_to_spacy[i-1]:
            current_spacy_tokens = [doc[idx] for idx in token_to_spacy[i]]
            prev_spacy_tokens = [doc[idx] for idx in token_to_spacy[i-1]]

            # Check if current token has a dependency on the previous token
            for curr_tok in current_spacy_tokens:
                for prev_tok in prev_spacy_tokens:
                    if curr_tok.head == prev_tok:
                        prev_attention = 0.9
                        break
                if prev_attention > 0.7:
                    break

        if i > 0:
            attention[i, i-1] = prev_attention

        # Self-attention (reduce if we boosted previous token attention)
        attention[i, i] = 0.05 if prev_attention > 0.8 else 0.15

        # Moderate attention to token two positions back
        if i > 1:
            attention[i, i-2] = 0.1

        # Small attention to [CLS] token
        if i > 0:
            attention[i, 0] = 0.05

    return "program_L6H11", make_row_stochastic(attention)



def program_L6H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference and syntactic dependency head that connects pronouns to antecedents and related tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens aligned to this position
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            # Fallback: uniform attention if no alignment
            attention[i] = np.ones(n) / n
            continue

        spacy_token = doc[spacy_indices[0]]  # Use first aligned token

        # Base attention distribution
        base_weight = 0.01
        attention[i] = np.full(n, base_weight)

        # Strong self-attention for special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.8
            continue

        # Pronoun coreference - high attention to potential antecedents
        if spacy_token.pos_ == 'PRON':
            for j in range(n):
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = doc[j_spacy_indices[0]]
                    # Attend to proper nouns and nouns (potential antecedents)
                    if j_spacy.pos_ in ['PROPN', 'NOUN'] and j < i:
                        attention[i, j] = 0.5
                    # Also attend to other pronouns with similar gender/number
                    elif j_spacy.pos_ == 'PRON' and j != i:
                        sim = embedding_similarity(tokens, i, j)
                        attention[i, j] = 0.3 * max(0, sim)

        # Coordinating conjunctions - attend to coordinated elements
        elif spacy_token.pos_ == 'CCONJ':
            for j in range(n):
                if j != i:
                    j_spacy_indices = token_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy = doc[j_spacy_indices[0]]
                        # High attention to coordinated verbs/nouns
                        if j_spacy.pos_ in ['VERB', 'NOUN', 'PROPN']:
                            attention[i, j] = 0.2

        # Sentence-final punctuation - attend to main verbs and subjects
        elif tokens[i] in ['.', '!', '?']:
            for j in range(n):
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = doc[j_spacy_indices[0]]
                    # High attention to main verbs
                    if j_spacy.pos_ == 'VERB' and j_spacy.dep_ in ['ROOT', 'ccomp']:
                        attention[i, j] = 0.3
                    # Moderate attention to subjects
                    elif j_spacy.dep_ in ['nsubj', 'nsubjpass']:
                        attention[i, j] = 0.2

        # Determiner-noun relationships
        elif spacy_token.pos_ == 'DET':
            for j in range(i+1, min(i+3, n)):  # Look ahead 1-2 positions
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = doc[j_spacy_indices[0]]
                    if j_spacy.pos_ in ['NOUN', 'PROPN']:
                        attention[i, j] = 0.4
                        break

        # Prepositions - attend to their objects and governing verbs
        elif spacy_token.pos_ == 'ADP':
            for j in range(n):
                if j != i:
                    j_spacy_indices = token_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy = doc[j_spacy_indices[0]]
                        # Attend to prepositional objects
                        if j_spacy.dep_ == 'pobj' and j > i:
                            attention[i, j] = 0.3
                        # Attend to governing verbs
                        elif j_spacy.pos_ == 'VERB' and j < i:
                            attention[i, j] = 0.2

        # Semantic similarity boost for all tokens
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.3:  # Only boost if reasonably similar
                    attention[i, j] += 0.1 * sim

        # Positional bias - slight preference for recent tokens
        for j in range(max(0, i-3), i):
            attention[i, j] *= 1.2

    return "program_L6H2", make_row_stochastic(attention)



def program_L6H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Backward syntactic dependency attention with local recency bias and punctuation chaining."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for syntactic information
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        token = tokens[i]

        # Strong self-attention for special tokens and punctuation
        if token in ['[CLS]', '[SEP]'] or token.strip() in '.,!?;:"':
            attention[i, i] = 0.8
            # Special tokens also attend to [CLS] and sentence end
            if token == '[SEP]':
                attention[i, 0] = 0.1  # to [CLS]
                if i > 0:
                    attention[i, i-1] = 0.1  # to previous token
            elif token == '[CLS]':
                attention[i, i] = 0.9
            elif token.strip() in '.,!?;:':
                attention[i, 0] = 0.1  # punctuation to [CLS]
                if i > 0:
                    attention[i, i-1] = 0.1

                # NEW: Enhanced punctuation chaining - punctuation attends more to nearby punctuation
                if token.strip() in '.,!?;:"':
                    for j in range(max(0, i-3), i):
                        prev_token = tokens[j]
                        if prev_token.strip() in '.,!?;:"':
                            # Stronger attention to recent punctuation
                            extra_weight = 0.4 / (i - j)
                            attention[i, j] += extra_weight
                            # Reduce self-attention proportionally
                            attention[i, i] -= extra_weight * 0.5
            continue

        # Find spacy token(s) corresponding to this LM token
        spacy_indices = token_to_spacy[i]

        base_weight = 0.3
        found_head = False

        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Look for syntactic head
            head = spacy_token.head
            if head != spacy_token and head.i < len(doc):
                head_lm_indices = spacy_to_token[head.i]
                if head_lm_indices:
                    head_idx = head_lm_indices[0]
                    if head_idx < i:  # Only backward attention
                        attention[i, head_idx] = base_weight
                        found_head = True

            # Special patterns for specific dependencies
            if spacy_token.dep_ in ['prep', 'pobj'] and not found_head:
                # Prepositions and their objects look backward for verbs
                for j in range(i-1, max(0, i-5), -1):
                    if j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_j = doc[token_to_spacy[j][0]]
                        if spacy_j.pos_ == 'VERB':
                            attention[i, j] = base_weight
                            found_head = True
                            break

            elif spacy_token.dep_ == 'advmod':
                # Adverbs look for what they modify
                for j in range(i-1, max(0, i-3), -1):
                    if j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_j = doc[token_to_spacy[j][0]]
                        if spacy_j.pos_ in ['VERB', 'ADJ'] or spacy_j.text.lower() in ['not', "n't"]:
                            attention[i, j] = base_weight
                            found_head = True
                            break

        # Local backward attention (previous token)
        if i > 0:
            prev_weight = 0.15 if found_head else 0.25
            attention[i, i-1] += prev_weight

        # Self-attention
        self_weight = 0.1 if found_head else 0.2
        attention[i, i] = self_weight

        # Weak attention to [CLS]
        if i > 0:
            attention[i, 0] = 0.05

        # Small recency bias for recent tokens
        for j in range(max(0, i-3), i):
            if j != i-1:  # Don't double-count previous token
                attention[i, j] += 0.02 * (1.0 / (i - j))

    return "program_L6H3", make_row_stochastic(attention)



def program_L6H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Detects verbal dependencies and argument structure connections."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        # Very high self-attention for [SEP] tokens
        if tokens[i].strip() in ['[SEP]']:
            attention[i, i] = 10.0
            continue

        # Get spacy analysis for current token
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            attention[i, i] = 0.1
            continue

        spacy_token = doc[spacy_indices[0]]

        # Base self-attention
        attention[i, i] = 0.1

        for j in range(n):
            if i == j:
                continue

            target_spacy_indices = token_to_spacy[j]
            if not target_spacy_indices:
                continue

            target_spacy = doc[target_spacy_indices[0]]

            # Strong patterns for specific POS combinations
            if spacy_token.pos_ == 'PART' and target_spacy.pos_ == 'VERB':
                # Infinitive "to" to verbs
                attention[i, j] = 5.0
            elif spacy_token.pos_ == 'ADP' and target_spacy.pos_ == 'VERB':
                # Prepositions to verbs
                attention[i, j] = 3.0
            elif spacy_token.pos_ == 'CCONJ' and target_spacy.pos_ == 'VERB':
                # Conjunctions to verbs
                attention[i, j] = 3.0
            elif spacy_token.pos_ == 'AUX' and target_spacy.pos_ in ['NOUN', 'PRON']:
                # Auxiliaries to subjects
                attention[i, j] = 2.0
            elif spacy_token.pos_ == 'VERB' and target_spacy.pos_ in ['NOUN', 'PRON']:
                # Verbs to arguments
                if j < i:  # Slight preference for earlier positions
                    attention[i, j] = 1.5
                else:
                    attention[i, j] = 1.0
            elif spacy_token.pos_ == 'PUNCT' and target_spacy.pos_ in ['NOUN', 'VERB']:
                # Punctuation to content words
                if abs(i - j) <= 3:  # Nearby tokens
                    attention[i, j] = 2.0
                else:
                    attention[i, j] = 0.5
            else:
                # General semantic similarity for other cases
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:
                    attention[i, j] = sim * 1.5
                else:
                    attention[i, j] = 0.1

    return "program_L6H4", make_row_stochastic(attention)



def program_L6H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Action-object relationship head that attends from objects to their governing verbs."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Create spacy token lookup
    spacy_tokens = list(doc)

    for i in range(n):
        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i]

        # Handle special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention_matrix[i, i] = 1.0
            continue

        # Handle punctuation - attend to key verbs and [CLS]
        if tokens[i] in ['.', ',', '?', '!']:
            # Strong attention to [CLS] if present
            if '[CLS]' in tokens:
                cls_idx = tokens.index('[CLS]')
                attention_matrix[i, cls_idx] = 0.3

            # Find key action verbs in the sentence
            for j in range(n):
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = spacy_tokens[j_spacy_indices[0]]
                    if j_spacy.pos_ == 'VERB' and j_spacy.dep_ in ['ROOT', 'ccomp', 'xcomp']:
                        attention_matrix[i, j] = 0.4

            # Self attention
            attention_matrix[i, i] = 0.2
            continue

        # Main token processing
        base_attention = 0.05

        # Self attention baseline
        attention_matrix[i, i] = base_attention

        if spacy_indices:
            spacy_token = spacy_tokens[spacy_indices[0]]

            # Direct objects attend strongly to their governing verbs
            if spacy_token.dep_ == 'dobj':
                if spacy_token.head and spacy_token.head.pos_ == 'VERB':
                    # Find the verb token
                    for j in range(n):
                        j_spacy_indices = token_to_spacy[j]
                        if j_spacy_indices:
                            j_spacy = spacy_tokens[j_spacy_indices[0]]
                            if j_spacy == spacy_token.head:
                                attention_matrix[i, j] = 0.8
                                break

            # Pronouns and objects attend to nearby verbs
            if spacy_token.pos_ in ['PRON'] or spacy_token.dep_ in ['dobj', 'pobj']:
                for j in range(n):
                    j_spacy_indices = token_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy = spacy_tokens[j_spacy_indices[0]]
                        if j_spacy.pos_ == 'VERB':
                            # Distance-based attention
                            distance = abs(i - j)
                            if distance <= 5:
                                attention_matrix[i, j] = max(0.3 - 0.05 * distance, 0.1)

            # Prepositions attend to their objects
            if spacy_token.pos_ == 'ADP':
                for j in range(i+1, min(i+4, n)):
                    j_spacy_indices = token_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy = spacy_tokens[j_spacy_indices[0]]
                        if j_spacy.dep_ == 'pobj' and j_spacy.head == spacy_token:
                            attention_matrix[i, j] = 0.4

            # Verbs get self-attention boost
            if spacy_token.pos_ == 'VERB':
                attention_matrix[i, i] = 0.2

        # General attention to action verbs based on embedding similarity
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:
                    attention_matrix[i, j] += 0.1 * sim

        # Attention to [CLS] token
        if '[CLS]' in tokens:
            cls_idx = tokens.index('[CLS]')
            if i != cls_idx:
                attention_matrix[i, cls_idx] += 0.05

    return "program_L6H5", make_row_stochastic(attention_matrix)



def program_L6H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence boundary tracker with strong SEP self-attention and reduced CLS attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Base self-attention
        attention[i, i] = 0.3

        # Special handling for [SEP] tokens - much stronger self-attention
        if tokens[i] in ['[SEP]', '</s>', '<eos>']:
            attention[i, i] = 0.9
            # SEP tokens attend much less to other tokens
            continue

        # Reduced attention to [CLS] token (was 0.4, now 0.15)
        if tokens[0] in ['[CLS]', '<s>', '<bos>']:
            attention[i, 0] = 0.15

        # Special handling for sentence-final punctuation
        spacy_indices = token_to_spacy[i]
        is_sent_final = False
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            is_sent_final = spacy_token.is_punct and spacy_token.text in '.!?'

        if is_sent_final:
            # Sentence-final punctuation attends strongly to sentence-initial content words
            for j in range(min(5, n)):  # Look at first few tokens
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_j = doc[token_to_spacy[j][0]]
                    if spacy_j.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not spacy_j.is_punct:
                        attention[i, j] = 0.8  # Increased from 0.6 to 0.8
                        break

        # Local syntactic attention for content words
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['NOUN', 'ADJ', 'VERB']:
                # Attend to nearby related tokens
                for j in range(max(0, i-3), min(n, i+4)):
                    if i != j and j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_j = doc[token_to_spacy[j][0]]
                        # Adjectives to nouns, determiners to nouns, etc.
                        if ((spacy_token.pos_ == 'NOUN' and spacy_j.pos_ in ['ADJ', 'DET']) or
                            (spacy_token.pos_ == 'ADJ' and spacy_j.pos_ in ['NOUN', 'DET']) or
                            (spacy_token.pos_ == 'VERB' and spacy_j.pos_ in ['NOUN', 'ADV'])):
                            attention[i, j] = 0.2

        # Boost attention based on embedding similarity for nearby tokens
        for j in range(n):
            if i != j and abs(i - j) <= 3:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.3:
                    attention[i, j] += sim * 0.1

    return "program_L6H6", make_row_stochastic(attention)



def program_L6H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head that connects tokens to earlier important content words (nouns, verbs)."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L6H7", np.zeros((0, 0))

    # Parse with spacy to get linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    for i in range(n):
        # Get spacy tokens aligned with this position
        spacy_indices = token_to_spacy[i]

        # Strong self-attention for special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 1.0
            continue

        for j in range(n):
            # Base attention score
            score = 0.0

            # Self-attention baseline
            if i == j:
                score += 0.3

            # Attention to earlier positions (recency bias)
            if j < i:
                distance = i - j
                recency_weight = 1.0 / (1.0 + 0.1 * distance)
                score += 0.2 * recency_weight

            # Strong attention to important content words
            target_spacy_indices = token_to_spacy[j]
            for spacy_idx in target_spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]

                    # Boost for nouns (especially proper nouns and entities)
                    if spacy_token.pos_ in ['NOUN', 'PROPN']:
                        score += 0.8
                        if spacy_token.ent_type_:  # Named entities get extra boost
                            score += 0.3

                    # Boost for verbs
                    elif spacy_token.pos_ == 'VERB':
                        score += 0.6

                    # Moderate boost for adjectives
                    elif spacy_token.pos_ == 'ADJ':
                        score += 0.3

            # Boost for first few content positions
            if j <= 2 and j > 0:  # Skip [CLS] but boost early content
                score += 0.4

            # Small boost for punctuation attention from end tokens
            if tokens[j] in ['.', ',', '!', '?'] and i >= n - 3:
                score += 0.2

            # Embedding similarity boost for semantically related tokens
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.5:
                score += 0.3 * sim

            attention[i, j] = score

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L6H7", attention



def program_L6H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Positional-syntactic attention head with strong [CLS] and self-attention bias, enhanced punctuation-to-CLS attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Strong attention to [CLS] token for early content tokens
        if tokens[0] == '[CLS]' and i <= 5 and i > 0:
            attention_matrix[i, 0] = 0.15

        # High self-attention for most tokens
        attention_matrix[i, i] = 0.1

        # Special handling for [CLS] token
        if tokens[0] == '[CLS]' and i == 0:
            attention_matrix[i, i] = 0.05

        # Special handling for [SEP] token - attend broadly but weakly
        if i == n-1 and tokens[i] == '[SEP]':
            attention_matrix[i, i] = 0.8  # Strong self-attention for [SEP]
            for j in range(n-1):
                if j != i:
                    attention_matrix[i, j] = 0.02

        # NEW: Enhanced punctuation-to-CLS attention
        if tokens[i] in ['.', '!', '?'] and tokens[0] == '[CLS]':
            attention_matrix[i, 0] = 0.55  # Strong punctuation-to-CLS attention

        # Get spacy information for current token
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []
        current_spacy_tok = doc[spacy_indices[0]] if spacy_indices else None

        # Local attention patterns
        for j in range(n):
            if i == j:  # Self-attention already handled
                continue

            # Distance-based decay
            distance = abs(i - j)
            distance_weight = 1.0 / (1.0 + distance * 0.1)

            # Get spacy info for target token
            target_spacy_indices = token_to_spacy[j] if j < len(token_to_spacy) else []
            target_spacy_tok = doc[target_spacy_indices[0]] if target_spacy_indices else None

            # Base attention
            base_attention = 0.01 * distance_weight

            # Boost for punctuation attending to nearby content
            if tokens[i] in [',', '.', '!', '?'] and distance <= 3:
                base_attention *= 2.0

            # Boost for function words attending to content words
            if (current_spacy_tok and target_spacy_tok and 
                current_spacy_tok.pos_ in ['DET', 'ADP', 'CCONJ'] and
                target_spacy_tok.pos_ in ['NOUN', 'VERB', 'ADJ'] and
                distance <= 4):
                base_attention *= 3.0

            # Boost for conjunctions attending to coordinated elements
            if (current_spacy_tok and target_spacy_tok and
                current_spacy_tok.pos_ == 'CCONJ'):
                # Look for nearby content words
                if (target_spacy_tok.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                    distance <= 6):
                    base_attention *= 2.5

            # Boost for verbs attending to their subjects/objects
            if (current_spacy_tok and target_spacy_tok and
                current_spacy_tok.pos_ == 'VERB' and
                target_spacy_tok.dep_ in ['nsubj', 'dobj', 'pobj'] and
                distance <= 5):
                base_attention *= 2.0

            # Boost for semantic similarity
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.5:
                base_attention *= (1.0 + sim)

            attention_matrix[i, j] += base_attention

    return "program_L6H8", make_row_stochastic(attention_matrix)



def program_L6H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic connector head: attends from phrase tokens to prepositions, conjunctions, connectors, and auxiliary verbs."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy to get syntactic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Default self-attention and attention to special tokens
        attention_matrix[i, i] = 0.05
        if tokens[0] == '[CLS]':
            attention_matrix[i, 0] = 0.02
        if tokens[-1] == '[SEP]':
            attention_matrix[i, -1] = 0.02

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i]

        for j in range(n):
            if i == j:
                continue

            target_spacy_indices = token_to_spacy[j]

            # Very strong attention to key syntactic connectors
            if tokens[j].lower() in ['from', 'to', 'in', 'and', 'with']:
                # Special case: possessive markers strongly attend to "from"
                if tokens[i] in ["'", 's'] and tokens[j].lower() == 'from':
                    attention_matrix[i, j] = 0.9
                # Tokens in phrases attend to their governing preposition/conjunction
                elif len(spacy_indices) > 0 and len(target_spacy_indices) > 0:
                    spacy_i = doc[spacy_indices[0]]
                    spacy_j = doc[target_spacy_indices[0]]

                    # Check if j is a preposition/conjunction that i depends on
                    if spacy_j.pos_ in ['ADP', 'CCONJ', 'PART'] and spacy_j.dep_ in ['prep', 'cc', 'aux', 'mark']:
                        # Check if i is in the phrase headed by j
                        if spacy_i.head == spacy_j or any(ancestor == spacy_j for ancestor in spacy_i.ancestors):
                            attention_matrix[i, j] = 0.6
                        # Or if they're closely related syntactically
                        elif abs(i - j) <= 3:
                            attention_matrix[i, j] = 0.4

            # Strong attention from conjunctions to preceding punctuation
            elif tokens[i].lower() == 'and' and tokens[j] in [',', ';']:
                if j < i and i - j <= 3:
                    attention_matrix[i, j] = 0.7

            # Attention from tokens after connectors back to the connector
            elif j < i and tokens[j].lower() in ['to', 'and', 'from', 'in']:
                if i - j <= 4:
                    attention_matrix[i, j] = 0.3 * (1.0 / (i - j))

            # Punctuation patterns
            elif tokens[j] in [',', '.', '"', "'"]:
                # Tokens near punctuation attend to it
                if abs(i - j) == 1:
                    attention_matrix[i, j] = 0.2
                # Quote/punctuation patterns
                elif tokens[i] in ['"', "'"] and j < i:
                    attention_matrix[i, j] = 0.15

            # Verb-to-auxiliary/modal patterns
            elif len(spacy_indices) > 0 and len(target_spacy_indices) > 0:
                spacy_i = doc[spacy_indices[0]]
                spacy_j = doc[target_spacy_indices[0]]

                if spacy_i.pos_ == 'VERB' and spacy_j.dep_ in ['aux', 'auxpass', 'mark']:
                    attention_matrix[i, j] = 0.25

    # NEW: Special handling for [SEP] tokens - they should have very high self-attention
    if tokens[-1] == '[SEP]':
        attention_matrix[-1, -1] = 0.8
        # Redistribute remaining attention
        remaining = 0.2
        non_self_count = n - 1
        if non_self_count > 0:
            for j in range(n-1):
                attention_matrix[-1, j] = remaining / non_self_count

    # NEW: Enhanced auxiliary/modal verb attention patterns
    for i in range(n):
        spacy_indices = token_to_spacy[i]
        if len(spacy_indices) > 0:
            spacy_i = doc[spacy_indices[0]]

            for j in range(n):
                if i == j:
                    continue

                target_spacy_indices = token_to_spacy[j]
                if len(target_spacy_indices) > 0:
                    spacy_j = doc[target_spacy_indices[0]]

                    # Strong attention to auxiliary verbs and modals
                    if spacy_j.pos_ in ['AUX', 'MODAL'] or spacy_j.dep_ in ['aux', 'auxpass']:
                        # Content words attend strongly to auxiliaries
                        if spacy_i.pos_ in ['NOUN', 'VERB', 'ADJ'] or spacy_i.dep_ in ['nsubj', 'dobj', 'pobj']:
                            attention_matrix[i, j] = max(attention_matrix[i, j], 0.4)

                    # Enhanced dependency-based attention
                    if spacy_i.head == spacy_j or spacy_j.head == spacy_i:
                        # Direct syntactic dependencies get stronger attention
                        attention_matrix[i, j] = max(attention_matrix[i, j], 0.3)
                    elif any(ancestor == spacy_j for ancestor in spacy_i.ancestors):
                        # Ancestor relationships
                        attention_matrix[i, j] = max(attention_matrix[i, j], 0.2)

    return "program_L6H9", make_row_stochastic(attention_matrix)



def program_L7H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Conjunction and coordination structure tracking head."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special token self-attention
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention_matrix[i, i] = 0.9
            # Distribute remaining mass uniformly
            remaining = 0.1 / (n - 1) if n > 1 else 0
            for j in range(n):
                if j != i:
                    attention_matrix[i, j] = remaining
            continue

        # Find coordinating conjunctions in the sentence
        conjunctions = []
        for j in range(n):
            if token_to_spacy[j]:
                spacy_token = doc[token_to_spacy[j][0]]
                if spacy_token.pos_ == 'CCONJ' or spacy_token.dep_ == 'cc':
                    conjunctions.append(j)

        # Base attention distribution
        for j in range(n):
            # Self-attention baseline
            if i == j:
                attention_matrix[i, j] = 0.05
            else:
                attention_matrix[i, j] = 0.01

        # Get spacy info for current token
        current_spacy_tokens = token_to_spacy[i]
        if current_spacy_tokens:
            current_spacy = doc[current_spacy_tokens[0]]

            # Strong attention from tokens after conjunctions to the conjunction
            for conj_idx in conjunctions:
                if i > conj_idx:
                    # Tokens after conjunction attend strongly to it
                    distance = i - conj_idx
                    weight = 0.6 / (1 + 0.1 * distance)  # Decay with distance
                    attention_matrix[i, conj_idx] = weight

            # Attention to subjects and main verbs
            for j in range(n):
                target_spacy_tokens = token_to_spacy[j]
                if target_spacy_tokens:
                    target_spacy = doc[target_spacy_tokens[0]]

                    # Attend to subjects
                    if target_spacy.dep_ in ['nsubj', 'nsubjpass']:
                        attention_matrix[i, j] += 0.15

                    # Attend to main verbs
                    if target_spacy.pos_ == 'VERB' and target_spacy.dep_ in ['ROOT', 'conj']:
                        attention_matrix[i, j] += 0.1

                    # Attend to pronouns (especially subjects)
                    if target_spacy.pos_ == 'PRON':
                        attention_matrix[i, j] += 0.12

            # If current token is a conjunction, attend to nearby content words
            if current_spacy.pos_ == 'CCONJ' or current_spacy.dep_ == 'cc':
                for j in range(max(0, i-3), min(n, i+4)):
                    if j != i and token_to_spacy[j]:
                        target_spacy = doc[token_to_spacy[j][0]]
                        if target_spacy.pos_ in ['NOUN', 'VERB', 'ADJ', 'PRON']:
                            attention_matrix[i, j] += 0.2

        # Boost attention to semantically similar tokens
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:
                    attention_matrix[i, j] += 0.05 * sim

    return "program_L7H0", make_row_stochastic(attention_matrix)



def program_L7H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content word integration head: connects sentence boundaries to key verbs/nouns with syntactic awareness."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Helper to get spacy features for a token
    def get_spacy_features(i):
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            return None, None, None
        spacy_tok = doc[spacy_indices[0]]
        return spacy_tok.pos_, spacy_tok.dep_, spacy_tok.text.lower()

    # Identify content words (verbs, nouns, adjectives, adverbs)
    content_words = set()
    verbs = set()
    for i in range(n):
        pos, dep, text = get_spacy_features(i)
        if pos in ['VERB', 'NOUN', 'PROPN', 'ADJ', 'ADV']:
            content_words.add(i)
        if pos == 'VERB':
            verbs.add(i)

    # Find sentence boundaries and special tokens
    cls_token = 0 if tokens[0] in ['[CLS]'] else None
    sep_token = n - 1 if n > 0 and tokens[n-1] in ['[SEP]'] else None
    punct_tokens = []
    for i in range(n):
        if tokens[i].strip() in ['.', '!', '?', ',']:
            punct_tokens.append(i)

    for i in range(n):
        pos_i, dep_i, text_i = get_spacy_features(i)

        # Base self-attention for all tokens (reduced from original)
        attention[i, i] = 0.2

        # Special handling for [SEP] token - much stronger self-attention
        if i == sep_token:
            attention[i, i] = 0.95
        # CLS token attention patterns
        elif i == cls_token:
            attention[i, i] = 0.7  # Strong self-attention for CLS
        elif cls_token is not None:
            attention[i, cls_token] = 0.15  # Moderate attention to CLS

        # Sentence-final punctuation: strong attention to early content words and main verbs
        if tokens[i].strip() in ['.', '!', '?']:
            early_content_boost = 0.0
            # Enhanced attention to main verbs across the sentence
            for j in range(n):
                if j in verbs:
                    pos_j, dep_j, text_j = get_spacy_features(j)
                    # Strong attention to main verbs regardless of position
                    if dep_j in ['ROOT', 'cop']:  # Main verbs
                        attention[i, j] += 0.15
                    else:
                        attention[i, j] += 0.08

            for j in range(min(n, 8)):  # Look at first 8 tokens
                if j in content_words:
                    pos_j, dep_j, text_j = get_spacy_features(j)
                    base_weight = 0.1
                    # Extra boost for verbs
                    if j in verbs:
                        base_weight += 0.08
                    # Distance decay
                    base_weight *= (0.8 ** j)
                    attention[i, j] += base_weight
                    early_content_boost += base_weight

            # Also attend to other sentence elements
            attention[i, cls_token if cls_token is not None else 0] += 0.05

            # Attend to nearby tokens
            for j in range(max(0, i-3), i):
                attention[i, j] += 0.02

        # Enhanced function word attention to syntactically related content words
        if pos_i in ['PRON', 'DET', 'ADP'] or dep_i in ['det', 'poss', 'prep']:
            for j in range(n):
                if j in content_words:
                    pos_j, dep_j, text_j = get_spacy_features(j)
                    # Check for syntactic relationships
                    if abs(i - j) <= 3:  # Nearby content words
                        # Enhanced weight for syntactic dependencies
                        if dep_i in ['det', 'poss'] and pos_j in ['NOUN', 'PROPN']:
                            attention[i, j] += 0.08
                        elif pos_i == 'PRON' and pos_j in ['VERB', 'NOUN']:
                            attention[i, j] += 0.06
                        else:
                            attention[i, j] += 0.04

        # Content words attend to other content words and function words
        if i in content_words:
            # Attend to other content words based on similarity and position
            for j in range(n):
                if j != i:
                    if j in content_words:
                        # Semantic similarity boost
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:
                            attention[i, j] += 0.03 * sim

                    # Local syntactic relationships
                    if abs(i - j) == 1:
                        attention[i, j] += 0.02
                    elif abs(i - j) <= 3:
                        pos_j, dep_j, text_j = get_spacy_features(j)
                        if pos_j in ['DET', 'ADP', 'CONJ', 'CCONJ']:
                            attention[i, j] += 0.015

        # Function words attend to nearby content words (original logic)
        if i not in content_words and tokens[i].strip() not in ['.', '!', '?', ','] and i != sep_token:
            for j in range(max(0, i-2), min(n, i+3)):
                if j in content_words:
                    distance_decay = 0.8 ** abs(i - j)
                    attention[i, j] += 0.03 * distance_decay

        # Comma and conjunction patterns
        if tokens[i].strip() == ',':
            # Look back for main clause elements
            for j in range(max(0, i-5), i):
                if j in content_words:
                    attention[i, j] += 0.02

        # Special handling for conjunctions
        pos_i, dep_i, text_i = get_spacy_features(i)
        if pos_i in ['CONJ', 'CCONJ'] or tokens[i].lower().strip() in ['and', 'but', 'or']:
            attention[i, i] += 0.2  # Strong self-attention
            # Attend to elements being conjoined
            for j in range(max(0, i-4), min(n, i+4)):
                if j in content_words:
                    attention[i, j] += 0.025

    return "program_L7H1", make_row_stochastic(attention)



def program_L7H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content summarization head: end-of-sentence tokens attend to key content words, with semantic clustering and syntactic dependencies."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L7H10", np.array([])

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Special tokens have strong self-attention
        if token_i in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.8
            # [CLS] gets some attention from other tokens
            if token_i == '[CLS]':
                for j in range(n):
                    if j != i:
                        attention[i, j] = 0.2 / (n - 1)
            continue

        # Punctuation (especially sentence-final) attends broadly to content
        if token_i in ['.', '!', '?']:
            # Strong attention to [CLS]
            attention[i, 0] = 0.15

            # Find content words to attend to
            content_weight = 0.0
            for j in range(n):
                if j == i:
                    attention[i, j] = 0.1  # Some self-attention
                elif tokens[j] in ['[CLS]', '[SEP]']:
                    continue
                elif tokens[j] in [',', '.', '!', '?', '"', "'", '(', ')']:
                    continue
                else:
                    # Check if it's a content word via spacy
                    is_content = False
                    if token_to_spacy[j]:
                        spacy_idx = token_to_spacy[j][0]
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                                is_content = True

                    if is_content:
                        # Distance decay
                        dist = abs(i - j)
                        weight = 0.08 * np.exp(-dist * 0.1)
                        attention[i, j] = weight
                        content_weight += weight

            # Normalize remaining weight
            remaining = 0.75 - content_weight
            if remaining > 0:
                for j in range(n):
                    if j != i and tokens[j] not in ['[CLS]', '[SEP]'] and attention[i, j] == 0:
                        attention[i, j] = remaining / max(1, n - 3)

            continue

        # Regular tokens
        # Self-attention baseline
        attention[i, i] = 0.05

        # NEW: Add syntactic dependency attention
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Conjunctions attend strongly to nearby nouns/pronouns
                if spacy_token.pos_ == 'CCONJ':
                    for j in range(max(0, i-3), min(n, i+4)):
                        if j != i and token_to_spacy[j]:
                            j_spacy_idx = token_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_spacy_token = doc[j_spacy_idx]
                                if j_spacy_token.pos_ in ['NOUN', 'PRON', 'PROPN']:
                                    attention[i, j] += 0.25

                # Pronouns attend to recent nouns they might reference
                if spacy_token.pos_ == 'PRON':
                    for j in range(max(0, i-5), i):
                        if token_to_spacy[j]:
                            j_spacy_idx = token_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_spacy_token = doc[j_spacy_idx]
                                if j_spacy_token.pos_ in ['NOUN', 'PROPN']:
                                    dist = i - j
                                    weight = 0.15 * np.exp(-dist * 0.2)
                                    attention[i, j] += weight

                # Adverbs like "now" attend to nearby verbs
                if spacy_token.pos_ == 'ADV':
                    for j in range(max(0, i-3), min(n, i+3)):
                        if j != i and token_to_spacy[j]:
                            j_spacy_idx = token_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_spacy_token = doc[j_spacy_idx]
                                if j_spacy_token.pos_ == 'VERB':
                                    attention[i, j] += 0.12

                # Nouns attend to verbs that act on them
                if spacy_token.pos_ in ['NOUN', 'PROPN']:
                    for j in range(max(0, i-4), min(n, i+4)):
                        if j != i and token_to_spacy[j]:
                            j_spacy_idx = token_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_spacy_token = doc[j_spacy_idx]
                                if j_spacy_token.pos_ == 'VERB':
                                    # Check if there's a semantic relationship
                                    sim = embedding_similarity(tokens, i, j)
                                    if sim > 0.2:
                                        attention[i, j] += 0.08

        # Semantic similarity with other tokens
        for j in range(n):
            if i == j:
                continue

            # Skip special tokens for semantic similarity
            if tokens[j] in ['[CLS]', '[SEP]']:
                attention[i, 0] = 0.02  # Small attention to [CLS]
                continue

            # Skip punctuation
            if tokens[j] in [',', '.', '!', '?', '"', "'", '(', ')']:
                continue

            # Compute semantic similarity
            sim = embedding_similarity(tokens, i, j)

            # High similarity gets more attention
            if sim > 0.7:
                weight = 0.15
            elif sim > 0.5:
                weight = 0.08
            elif sim > 0.3:
                weight = 0.03
            else:
                weight = 0.01

            # Distance decay
            dist = abs(i - j)
            weight *= np.exp(-dist * 0.05)

            attention[i, j] = weight

        # Adjacent token attention (mild)
        if i > 0:
            attention[i, i-1] += 0.02
        if i < n - 1:
            attention[i, i+1] += 0.02

        # Content words attend more to other content words
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    # Boost attention to other content words
                    for j in range(n):
                        if j != i and token_to_spacy[j]:
                            j_spacy_idx = token_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_spacy_token = doc[j_spacy_idx]
                                if j_spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                                    attention[i, j] *= 1.5

    return "program_L7H10", make_row_stochastic(attention)



def program_L7H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attends to semantically prominent tokens (subjects, main verbs) with strong local grammatical dependencies."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Identify semantically important positions
    important_positions = set()

    # Find subjects and main verbs from spacy analysis
    for spacy_idx, spacy_token in enumerate(doc):
        if (spacy_token.dep_ in ['nsubj', 'nsubjpass'] or 
            (spacy_token.pos_ == 'PRON' and spacy_token.dep_ in ['nsubj', 'ROOT']) or
            (spacy_token.pos_ == 'VERB' and spacy_token.dep_ == 'ROOT') or
            spacy_token.dep_ == 'ROOT'):
            # Map to token positions
            spacy_to_tokens = align_spacy_to_tokens(sentence)
            if spacy_idx < len(spacy_to_tokens):
                important_positions.update(spacy_to_tokens[spacy_idx])

    # Add position 0 ([CLS]) as important
    important_positions.add(0)

    for i in range(n):
        token = tokens[i]

        # Special token handling
        if token.strip() in ['[CLS]', '[SEP]']:
            # [SEP] has very strong self-attention
            if '[SEP]' in token:
                attention_matrix[i, i] = 0.9
                # Small attention to final punctuation
                if i > 0 and tokens[i-1].strip() in ['.', '!', '?']:
                    attention_matrix[i, i-1] = 0.05
                # Small attention to [CLS]
                attention_matrix[i, 0] = 0.05
            else:  # [CLS]
                attention_matrix[i, i] = 0.5
                # Distribute remaining to nearby tokens
                for j in range(min(3, n)):
                    if j != i:
                        attention_matrix[i, j] = 0.1
            continue

        # Get spacy information for current token
        current_spacy_tokens = token_to_spacy[i] if i < len(token_to_spacy) else []

        # Base attention distribution
        base_weight = 1.0 / n

        for j in range(n):
            if i == j:
                # Self-attention baseline
                attention_matrix[i, j] = 0.05
            else:
                attention_matrix[i, j] = base_weight

        # Strong attention to semantically important positions
        for imp_pos in important_positions:
            if imp_pos < n and imp_pos != i:
                attention_matrix[i, imp_pos] += 0.15

        # Local positional bias - attend to nearby tokens
        for j in range(max(0, i-3), i):
            attention_matrix[i, j] += 0.05

        # NEW: Strong local grammatical dependencies
        if current_spacy_tokens:
            spacy_token = doc[current_spacy_tokens[0]]

            # Auxiliaries, modals, prepositions strongly attend to nearby subjects
            if (spacy_token.pos_ in ['AUX', 'ADP'] or 
                spacy_token.tag_ in ['MD', 'TO']):  # Modals and 'to'
                for j in range(max(0, i-4), i):
                    if j in important_positions:  # Attend to subjects/main verbs
                        attention_matrix[i, j] += 0.4

            # Verbs strongly attend to their subjects
            if spacy_token.pos_ == 'VERB':
                for j in range(max(0, i-4), i):
                    if j in important_positions:
                        attention_matrix[i, j] += 0.3

        # Function word specific patterns
        is_function_word = False
        if current_spacy_tokens:
            spacy_token = doc[current_spacy_tokens[0]]
            if (spacy_token.pos_ in ['DET', 'ADP', 'CCONJ', 'SCONJ', 'AUX'] or
                token.strip().lower() in ['to', 'a', 'an', 'the', 'and', 'but', 'on', 'in']):
                is_function_word = True

                # Function words attend strongly to nearby content words
                for j in range(max(0, i-2), min(n, i+3)):
                    if j != i and j in important_positions:
                        attention_matrix[i, j] += 0.2

        # Punctuation patterns
        if token.strip() in [',', '.', '!', '?']:
            # Punctuation attends to preceding content
            for j in range(max(0, i-3), i):
                attention_matrix[i, j] += 0.1
            # Strong self-attention for sentence-final punctuation
            if token.strip() in ['.', '!', '?']:
                attention_matrix[i, i] += 0.1

        # NEW: Quote marks attend strongly to immediately preceding tokens
        if token.strip() in ['"', "'"]:
            if i > 0:
                attention_matrix[i, i-1] += 0.5

        # Semantic similarity boost
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity threshold
                    attention_matrix[i, j] += 0.1

    return "program_L7H11", make_row_stochastic(attention_matrix)



def program_L7H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence boundary and syntactic structure attention with subject-predicate linking."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L7H2", np.array([]).reshape(0, 0)

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i].strip()

        # Special tokens get strong self-attention
        if token in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.8
            # [CLS] gets some attention from content words
            if token == '[CLS]':
                for j in range(n):
                    if j != i:
                        attention[i, j] = 0.2 / (n - 1)
            continue

        # Sentence-final punctuation attends strongly to early tokens
        if token in ['.', '!', '?']:
            # Strong attention to [CLS] and early content words (especially subjects)
            for j in range(min(3, n)):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['PRON', 'PROPN', 'NOUN'] or tokens[j] == '[CLS]':
                            attention[i, j] = 0.4 if j == 0 else 0.3
                        else:
                            attention[i, j] = 0.1
                    else:
                        attention[i, j] = 0.2 if j == 0 else 0.1
            continue

        # Punctuation and function words attend to nearby context
        if token in [',', ';', ':', '"', "'"] or (
            len(token_to_spacy[i]) > 0 and 
            token_to_spacy[i][0] < len(doc) and 
            doc[token_to_spacy[i][0]].pos_ in ['DET', 'ADP', 'CONJ', 'CCONJ']
        ):
            # Moderate self-attention
            attention[i, i] = 0.3

            # Attend to adjacent tokens and early content
            for j in range(n):
                if abs(i - j) <= 2 and j != i:
                    attention[i, j] = 0.2
                elif j < 3 and j != i:  # Early tokens
                    attention[i, j] = 0.1
            continue

        # Content words
        base_attention = 0.2
        attention[i, i] = base_attention  # Self-attention

        # Attend backwards to previous content words
        for j in range(i):
            if len(token_to_spacy[j]) > 0 and token_to_spacy[j][0] < len(doc):
                spacy_token = doc[token_to_spacy[j][0]]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PRON', 'PROPN']:
                    # Distance decay
                    distance_factor = 1.0 / (1.0 + (i - j))
                    attention[i, j] = 0.15 * distance_factor

        # Attend to [CLS] token
        attention[i, 0] = 0.1

        # Attend to syntactically related tokens
        if len(token_to_spacy[i]) > 0 and token_to_spacy[i][0] < len(doc):
            spacy_token = doc[token_to_spacy[i][0]]

            # Attend to head
            if spacy_token.head != spacy_token:
                head_idx = spacy_token.head.i
                for j, spacy_indices in enumerate(token_to_spacy):
                    if head_idx in spacy_indices:
                        attention[i, j] += 0.1
                        break

            # Attend to children
            for child in spacy_token.children:
                child_idx = child.i
                for j, spacy_indices in enumerate(token_to_spacy):
                    if child_idx in spacy_indices:
                        attention[i, j] += 0.05
                        break

    return "program_L7H2", make_row_stochastic(attention)



def program_L7H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic dependency head that connects modifiers to their heads and semantically related terms."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens aligned with this position
        spacy_indices = token_to_spacy[i]

        if not spacy_indices:
            # Fallback: self-attention
            attention[i, i] = 1.0
            continue

        spacy_token = doc[spacy_indices[0]]

        # High attention patterns
        if tokens[i] in ['[CLS]', '[SEP]']:
            # Special tokens attend to themselves and sentence boundaries
            attention[i, i] = 0.6
            if i > 0 and tokens[i-1] in ['.', '!', '?']:
                attention[i, i-1] = 0.3
            attention[i, 0] = 0.1  # Some attention to [CLS]

        elif tokens[i] in ['.', '!', '?']:
            # Sentence-final punctuation
            attention[i, i] = 0.4
            attention[i, 0] = 0.3  # Attend to [CLS]
            # Attend to main verb or important content word
            for j in range(i):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    j_spacy = doc[token_to_spacy[j][0]]
                    if j_spacy.pos_ == 'VERB' or j_spacy.dep_ == 'ROOT':
                        attention[i, j] = 0.3
                        break

        else:
            # Content tokens - look for semantic relationships
            max_weight = 0.0
            best_target = i  # Default to self

            # Special case: Strong attention from prepositions/particles to their syntactic heads
            if spacy_token.pos_ in ['ADP', 'PART'] or spacy_token.dep_ == 'mark':
                # Find the syntactic head
                for j in range(i):
                    if j < len(token_to_spacy) and token_to_spacy[j]:
                        j_spacy = doc[token_to_spacy[j][0]]
                        if spacy_token.head.i == j_spacy.i:
                            # Strong attention to syntactic head
                            attention[i, j] = 0.7
                            attention[i, i] = 0.2
                            # Distribute remaining attention
                            remaining = 1.0 - attention[i].sum()
                            if remaining > 0:
                                if i > 0:
                                    attention[i, i-1] += remaining * 0.5
                                attention[i, 0] += remaining * 0.3
                                attention[i, i] += remaining * 0.2
                            continue

            # Look for semantic head or related terms
            for j in range(i):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    j_spacy = doc[token_to_spacy[j][0]]

                    # Check for modifier-head relationships
                    weight = 0.0

                    # Semantic similarity boost
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        weight += sim * 0.4

                    # Syntactic relationships
                    if spacy_token.head.i in token_to_spacy[j]:
                        weight += 0.5  # Head of current token

                    if j_spacy.head.i in spacy_indices:
                        weight += 0.3  # Current token is head of j

                    # Special cases for strong patterns observed
                    if (spacy_token.pos_ in ['ADJ', 'ADV'] and 
                        j_spacy.pos_ in ['VERB', 'NOUN']):
                        weight += 0.3

                    if (spacy_token.text.lower() in ['about', 'of', 'with'] and
                        j_spacy.pos_ == 'VERB'):
                        weight += 0.4

                    # Recent token bias
                    if i - j <= 3:
                        weight += 0.1

                    if weight > max_weight:
                        max_weight = weight
                        best_target = j

            # Set attention weights
            if max_weight > 0.15:
                attention[i, best_target] = min(0.7, max_weight)
                attention[i, i] = 0.3 - min(0.2, max_weight * 0.3)
            else:
                attention[i, i] = 0.6

            # Distribute remaining attention
            remaining = 1.0 - attention[i].sum()
            if remaining > 0:
                # Add some attention to previous tokens and [CLS]
                if i > 0:
                    attention[i, i-1] += remaining * 0.5
                attention[i, 0] += remaining * 0.3
                attention[i, i] += remaining * 0.2

    return "program_L7H3", make_row_stochastic(attention)



def program_L7H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Local sequential attention with strong syntactic dependencies and boosted predecessor weights."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L7H4", np.array([])

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Handle special tokens
        if tokens[i] == '[SEP]':
            attention[i, i] = 1.0
            continue
        if tokens[i] == '[CLS]':
            attention[i, i] = 0.5
            # Slight attention to next token if exists
            if i + 1 < n:
                attention[i, i + 1] = 0.3
            continue

        # Check if this token has strong syntactic relationship with predecessor
        strong_predecessor = False
        if i > 0:
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                pred_spacy_indices = token_to_spacy[i-1]
                if pred_spacy_indices:
                    pred_spacy_token = doc[pred_spacy_indices[0]]
                    # Check for strong syntactic relationships
                    if (spacy_token.head == pred_spacy_token or 
                        pred_spacy_token in list(spacy_token.children) or
                        spacy_token.pos_ in ['ADP', 'DET', 'PART'] or  # Function words
                        pred_spacy_token.pos_ in ['ADP', 'VERB']):  # Following prepositions/verbs
                        strong_predecessor = True

        # Strong attention to immediate predecessor (boosted for syntactic relationships)
        if i > 0:
            if strong_predecessor:
                attention[i, i - 1] = 0.85  # Much higher for syntactic relationships
            else:
                attention[i, i - 1] = 0.6

        # Moderate self-attention
        attention[i, i] = 0.2

        # Syntactic dependencies via spacy alignment
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]

            # Attention to syntactic head (boosted)
            if spacy_token.head != spacy_token:
                head_char_start = spacy_token.head.idx
                head_char_end = head_char_start + len(spacy_token.head.text)

                # Find corresponding token indices for the head
                for j in range(n):
                    if j != i:
                        spacy_j = token_to_spacy[j]
                        if spacy_j:
                            spacy_tok_j = doc[spacy_j[0]]
                            if (spacy_tok_j.idx < head_char_end and 
                                spacy_tok_j.idx + len(spacy_tok_j.text) > head_char_start):
                                attention[i, j] += 0.5  # Boosted from 0.3

            # Attention to syntactic children (boosted)
            for child in spacy_token.children:
                child_char_start = child.idx
                child_char_end = child_char_start + len(child.text)

                for j in range(n):
                    if j != i:
                        spacy_j = token_to_spacy[j]
                        if spacy_j:
                            spacy_tok_j = doc[spacy_j[0]]
                            if (spacy_tok_j.idx < child_char_end and 
                                spacy_tok_j.idx + len(spacy_tok_j.text) > child_char_start):
                                attention[i, j] += 0.3  # Boosted from 0.2

        # Semantic similarity boost
        for j in range(n):
            if i != j and j < i:  # Focus on preceding tokens
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:
                    attention[i, j] += 0.2
                elif sim > 0.5:
                    attention[i, j] += 0.1

        # Positional decay for distant tokens
        for j in range(max(0, i - 3), i):
            if j != i - 1:  # Don't double-count immediate predecessor
                distance = i - j
                attention[i, j] += max(0, 0.1 - 0.03 * distance)

    return "program_L7H4", make_row_stochastic(attention)



def program_L7H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic dependency head that connects verbs to objects and related content words."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Get spacy tokens aligned with this LM token
        spacy_indices = token_to_spacy[i]

        # Self-attention baseline
        attention[i, i] = 0.05

        # Special token handling
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.8
            # Distribute remaining weight to first few content tokens
            for j in range(min(5, n)):
                if j != i:
                    attention[i, j] = 0.04
            continue

        # For each spacy token aligned with current LM token
        for spacy_idx in spacy_indices:
            if spacy_idx >= len(doc):
                continue

            spacy_token = doc[spacy_idx]

            # High attention patterns for verbs
            if spacy_token.pos_ in ['VERB', 'AUX']:
                # Attend to direct objects
                for child in spacy_token.children:
                    if child.dep_ in ['dobj', 'pobj', 'attr', 'prep']:
                        child_token_indices = spacy_to_token[child.i]
                        for j in child_token_indices:
                            if j < n:
                                attention[i, j] += 0.15

                # Attend to subjects
                for child in spacy_token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        child_token_indices = spacy_to_token[child.i]
                        for j in child_token_indices:
                            if j < n:
                                attention[i, j] += 0.1

            # Conjunctions and prepositions attend to content words
            if spacy_token.pos_ in ['CCONJ', 'SCONJ', 'ADP'] or spacy_token.dep_ in ['prep', 'cc']:
                # Find nearest content words (nouns, verbs, adjectives)
                for j in range(n):
                    j_spacy_indices = token_to_spacy[j]
                    for j_spacy_idx in j_spacy_indices:
                        if j_spacy_idx < len(doc):
                            j_spacy_token = doc[j_spacy_idx]
                            if j_spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                                # Weight by distance and semantic importance
                                distance = abs(i - j)
                                weight = 0.08 / (1 + distance * 0.1)
                                attention[i, j] += weight

            # Nouns attend to their modifiers and heads
            if spacy_token.pos_ in ['NOUN', 'PROPN']:
                # Attend to head if it's a verb
                if spacy_token.head.pos_ in ['VERB', 'AUX']:
                    head_indices = spacy_to_token[spacy_token.head.i]
                    for j in head_indices:
                        if j < n:
                            attention[i, j] += 0.12

                # Attend to adjective modifiers
                for child in spacy_token.children:
                    if child.pos_ == 'ADJ' or child.dep_ == 'amod':
                        child_token_indices = spacy_to_token[child.i]
                        for j in child_token_indices:
                            if j < n:
                                attention[i, j] += 0.08

        # Semantic similarity component
        for j in range(n):
            if i != j:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:  # High similarity threshold
                    attention[i, j] += 0.1 * sim
                elif sim > 0.3:  # Medium similarity
                    attention[i, j] += 0.05 * sim

        # Positional bias - slight preference for earlier tokens
        for j in range(i):
            if tokens[j] not in ['[CLS]', '[SEP]']:
                attention[i, j] += 0.02 / (1 + (i - j) * 0.2)

        # Punctuation attends to nearby content
        if tokens[i] in ['.', ',', '!', '?', ';', ':']:
            for j in range(max(0, i-3), i):
                j_spacy_indices = token_to_spacy[j]
                for j_spacy_idx in j_spacy_indices:
                    if j_spacy_idx < len(doc):
                        j_spacy_token = doc[j_spacy_idx]
                        if j_spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            attention[i, j] += 0.1

    return "program_L7H5", make_row_stochastic(attention)



def program_L7H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence boundary and syntactic relationship attention with CLS focus and strong SEP self-attention."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Special handling for [SEP] tokens - much stronger self-attention
        if tokens[i] == '[SEP]':
            attention[i, i] = 1.5  # Much higher than original 0.8
            attention[i, 0] = 0.05  # Reduced CLS attention
        # Strong self-attention for CLS and regular punctuation
        elif tokens[i] == '[CLS]' or tokens[i].strip() in '.,!?;:':
            attention[i, i] = 0.8
            if tokens[i] != '[CLS]':
                attention[i, 0] = 0.15  # Also attend to CLS
        else:
            attention[i, i] = 0.05  # Moderate self-attention for regular tokens

        # High attention to [CLS] from most tokens (but reduced for [SEP])
        if i != 0 and tokens[i] != '[SEP]':
            base_cls_attention = 0.3 if i >= n-2 else 0.15  # Higher for end tokens
            attention[i, 0] = base_cls_attention

        # Syntactic relationships
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]

                # Attend to syntactic head
                if spacy_token.head != spacy_token and spacy_token.head.i in range(len(doc)):
                    head_tokens = spacy_to_token[spacy_token.head.i]
                    for head_idx in head_tokens:
                        if head_idx < n:
                            attention[i, head_idx] += 0.08

                # Attend to children
                for child in spacy_token.children:
                    if child.i < len(doc):
                        child_tokens = spacy_to_token[child.i]
                        for child_idx in child_tokens:
                            if child_idx < n:
                                attention[i, child_idx] += 0.04

        # Semantic similarity attention
        for j in range(n):
            if i != j and j != 0:  # Don't override CLS attention
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:
                    attention[i, j] += 0.06 * sim

        # Small uniform attention to nearby tokens
        for j in range(max(0, i-2), min(n, i+3)):
            if j != i and j != 0:
                attention[i, j] += 0.02

    return "program_L7H6", make_row_stochastic(attention)



def program_L7H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic relationship head focusing on function word to content word attention."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token_i = tokens[i]

        # Strong self-attention for special tokens and punctuation
        if token_i in ['[CLS]', '[SEP]'] or token_i in '.,!?;:':
            attention[i, i] = 1.0
            continue

        # Get spacy info for current token
        spacy_indices_i = token_to_spacy[i]
        if not spacy_indices_i:
            attention[i, i] = 0.3
            continue

        spacy_tok_i = doc[spacy_indices_i[0]]

        for j in range(n):
            token_j = tokens[j]
            spacy_indices_j = token_to_spacy[j]

            if not spacy_indices_j:
                continue

            spacy_tok_j = doc[spacy_indices_j[0]]

            # Base attention
            base_weight = 0.01

            # Function words (pronouns, determiners, auxiliaries) attend strongly to verbs
            if spacy_tok_i.pos_ in ['PRON', 'DET', 'AUX'] and spacy_tok_j.pos_ == 'VERB':
                base_weight = 0.4

            # Function words attend to prepositions
            if spacy_tok_i.pos_ in ['PRON', 'DET'] and spacy_tok_j.pos_ == 'ADP':
                base_weight = 0.3

            # Prepositions attend to their objects
            if spacy_tok_i.pos_ == 'ADP' and spacy_tok_j.pos_ in ['NOUN', 'PRON']:
                base_weight = 0.25

            # Objects attend to their governing prepositions
            if spacy_tok_i.pos_ in ['NOUN', 'PRON'] and spacy_tok_j.pos_ == 'ADP':
                base_weight = 0.2

            # Moderate attention from nouns to verbs
            if spacy_tok_i.pos_ == 'NOUN' and spacy_tok_j.pos_ == 'VERB':
                base_weight = 0.15

            # Self-attention for content words
            if i == j and spacy_tok_i.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                base_weight = max(base_weight, 0.08)

            # Coordinating conjunctions get moderate self-attention
            if i == j and spacy_tok_i.pos_ == 'CCONJ':
                base_weight = 0.1

            # Distance decay for non-special relationships
            if i != j:
                distance = abs(i - j)
                decay = 1.0 / (1.0 + distance * 0.1)
                base_weight *= decay

            # Slight boost for attending to earlier tokens
            if j < i:
                base_weight *= 1.2

            attention[i, j] = base_weight

    # Normalize to make row-stochastic
    return "program_L7H7", make_row_stochastic(attention)



def program_L7H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head with selective first-token bias and context-dependent self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L7H8", np.array([])

    attention = np.zeros((n, n))
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Base weights
        weights = np.zeros(n)

        # Selective attention to [CLS] token - only for certain token types
        token_text = tokens[i].strip()
        if n > 0:
            # Reduce [CLS] attention for punctuation and special tokens
            if token_text in ['.', ',', ';', ':', '!', '?', '[SEP]', '"', "'", '(', ')']:
                weights[0] += 0.05  # Much less for punctuation
            else:
                # Get spacy info for current token
                spacy_indices = token_to_spacy[i]
                current_spacy = doc[spacy_indices[0]] if spacy_indices else None

                # Content words get more [CLS] attention
                if current_spacy and current_spacy.pos_ in ['NOUN', 'VERB', 'PROPN', 'ADJ']:
                    weights[0] += 0.12
                else:
                    weights[0] += 0.08  # Less for function words

        # Conditional self-attention based on token type
        if token_text == '[SEP]':
            weights[i] += 0.8  # Very strong self-attention for [SEP]
        elif token_text in ['.', ',', ';', ':', '!', '?']:
            weights[i] += 0.35  # Strong self-attention for punctuation
        else:
            # Get spacy information for current token
            spacy_indices = token_to_spacy[i]
            current_spacy = doc[spacy_indices[0]] if spacy_indices else None

            # Content words get less self-attention than function words
            if current_spacy and current_spacy.pos_ in ['NOUN', 'VERB', 'PROPN', 'ADJ']:
                weights[i] += 0.06
            else:
                weights[i] += 0.10  # More for function words

        # Get spacy information for current token
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        # Attention to sentence-initial content words
        for j in range(min(5, n)):  # First few tokens
            if j < len(token_to_spacy):
                j_spacy_indices = token_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = doc[j_spacy_indices[0]]
                    if j_spacy.pos_ in ['NOUN', 'PRON', 'PROPN', 'VERB']:
                        weights[j] += 0.08

        # Syntactic attention patterns
        if current_spacy:
            # Verbs attend to their subjects
            if current_spacy.pos_ == 'VERB':
                for child in current_spacy.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        # Find corresponding token indices
                        for k in range(n):
                            k_spacy_indices = token_to_spacy[k]
                            if k_spacy_indices and child.i in k_spacy_indices:
                                weights[k] += 0.06

            # Nouns attend to their determiners and adjectives
            if current_spacy.pos_ in ['NOUN', 'PROPN']:
                for child in current_spacy.children:
                    if child.dep_ in ['det', 'amod']:
                        for k in range(n):
                            k_spacy_indices = token_to_spacy[k]
                            if k_spacy_indices and child.i in k_spacy_indices:
                                weights[k] += 0.04

        # Attention to nearby commas and conjunctions
        for j in range(n):
            j_token = tokens[j].strip()
            if j_token in [',', 'and', 'but', 'or']:
                distance = abs(i - j)
                if distance <= 5:
                    weights[j] += 0.03 / (1 + distance * 0.5)

        # Slight recency bias (attend to recent tokens)
        for j in range(max(0, i-3), i):
            weights[j] += 0.02

        # Add small random baseline
        weights += 0.001

        attention[i] = weights

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L7H8", attention



def program_L7H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention head combining previous-token dependencies with conjunction tracking and enhanced [SEP] self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy parse for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Base attention values
        base_prev = 0.3  # Previous token attention
        base_conj = 0.25  # Conjunction attention
        base_self = 0.1   # Self attention
        base_cls = 0.05   # First token attention
        base_punct = 0.2  # Punctuation-related attention

        # Enhanced self-attention for [SEP] token
        current_token = tokens[i].strip()
        if current_token == '[SEP]':
            attention[i, i] = 0.85  # Much higher self-attention for [SEP]
        else:
            attention[i, i] = base_self

        # Previous token attention (strong pattern)
        if i > 0:
            attention[i, i-1] = base_prev

            # Extra boost if previous token is a verb or auxiliary
            if token_to_spacy[i-1]:
                spacy_idx = token_to_spacy[i-1][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ in ['VERB', 'AUX']:
                        attention[i, i-1] += 0.3

        # Object pronouns attending to governing prepositions
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ == 'PRON':
                    # Look for prepositions that govern this pronoun
                    for j in range(max(0, i-3), i):
                        if token_to_spacy[j]:
                            spacy_j = token_to_spacy[j][0]
                            if spacy_j < len(doc):
                                spacy_j_token = doc[spacy_j]
                                if spacy_j_token.pos_ == 'ADP':
                                    # Check if this preposition governs the pronoun
                                    if any(child.i == spacy_token.i for child in spacy_j_token.children):
                                        attention[i, j] += 0.5

        # Attention to conjunctions and coordinating words
        for j in range(n):
            if i != j and token_to_spacy[j]:
                spacy_idx = token_to_spacy[j][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    # Strong attention to coordinating conjunctions
                    if spacy_token.pos_ == 'CCONJ' or spacy_token.text.lower() in ['so']:
                        attention[i, j] = base_conj
                        # Boost if it's immediately before
                        if j == i - 1:
                            attention[i, j] += 0.15

        # Punctuation patterns
        if current_token in [',', '.', '?', '!', ';', ':']:
            # Punctuation attends to nearby content words
            for j in range(max(0, i-3), i):
                if token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            attention[i, j] = base_punct
                            if j == i - 1:
                                attention[i, j] += 0.1

        # Tokens after punctuation attend to the punctuation
        if i > 0 and tokens[i-1].strip() in [',', '.', '?', '!', ';', ':']:
            attention[i, i-1] += base_punct

        # Attention to first token ([CLS])
        if i > 0 and current_token != '[SEP]':  # Don't override [SEP] self-attention
            attention[i, 0] = base_cls

        # Special boost for prepositions attending to their objects
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ == 'ADP':  # Preposition
                    # Look for following noun
                    for j in range(i+1, min(n, i+3)):
                        if token_to_spacy[j]:
                            spacy_j = token_to_spacy[j][0]
                            if spacy_j < len(doc) and doc[spacy_j].pos_ == 'NOUN':
                                attention[i, j] = 0.2
                                break

        # Pronouns attend more strongly to recent verbs
        if token_to_spacy[i]:
            spacy_idx = token_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ == 'PRON':
                    for j in range(max(0, i-5), i):
                        if token_to_spacy[j]:
                            spacy_j = token_to_spacy[j][0]
                            if spacy_j < len(doc) and doc[spacy_j].pos_ in ['VERB', 'AUX']:
                                attention[i, j] += 0.15

    return "program_L7H9", make_row_stochastic(attention)



def program_L8H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference resolution head: pronouns attend to potential antecedents, with positional and syntactic biases."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special handling for [SEP] token - strong self-attention
        if tokens[i].strip() == '[SEP]':
            attention[i, i] = 0.9
            continue

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i]
        spacy_token = doc[spacy_indices[0]] if spacy_indices else None

        # Base weights: slight bias toward [CLS] and self
        attention[i, 0] = 0.05  # [CLS] bias
        attention[i, i] = 0.02  # self-attention

        # Check if current token is a pronoun
        is_pronoun = (spacy_token and spacy_token.pos_ == 'PRON')

        for j in range(n):
            if i == j:
                continue

            # Get spacy info for target token
            target_spacy_indices = token_to_spacy[j]
            target_spacy_token = doc[target_spacy_indices[0]] if target_spacy_indices else None

            # Strong pronoun -> noun attention
            if is_pronoun and target_spacy_token:
                if target_spacy_token.pos_ in ['NOUN', 'PROPN']:
                    # Stronger for proper nouns and entities
                    if target_spacy_token.pos_ == 'PROPN' or target_spacy_token.ent_type_:
                        attention[i, j] += 0.3
                    else:
                        attention[i, j] += 0.15

                    # Recency bias - prefer closer antecedents
                    distance = abs(i - j)
                    if distance <= 5:
                        attention[i, j] += 0.1 / (1 + distance * 0.2)

            # Coordinating conjunction patterns
            if (spacy_token and spacy_token.pos_ == 'CCONJ' and 
                tokens[i].strip().lower() == 'and'):
                attention[i, i] = 0.08  # self-attention for "and"

                # Attend to coordinated elements
                if target_spacy_token and target_spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                    attention[i, j] += 0.05

            # First token ([CLS]) attraction for content words
            if j == 0 and spacy_token:
                if spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                    attention[i, j] += 0.08

            # Punctuation patterns
            if tokens[i].strip() in [',', '.', '!', '?']:
                # Punctuation attends to nearby content
                distance = abs(i - j)
                if distance <= 3 and target_spacy_token:
                    if target_spacy_token.pos_ in ['NOUN', 'VERB']:
                        attention[i, j] += 0.03

                # Period attends to sentence-final elements
                if tokens[i].strip() == '.' and j == i - 1:
                    attention[i, j] += 0.05

            # Embedding similarity boost for related words
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.5:
                attention[i, j] += 0.02 * sim

            # Positional proximity bias
            distance = abs(i - j)
            if distance == 1:
                attention[i, j] += 0.02
            elif distance <= 3:
                attention[i, j] += 0.01

    return "program_L8H0", make_row_stochastic(attention)



def program_L8H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Grammatical dependency and referential attention with enhanced punctuation-to-CLS patterns."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Strong self-attention for special tokens and punctuation
        if tokens[i] in ['[CLS]', '[SEP]'] or tokens[i].strip() in '.,!?':
            attention[i, i] = 0.6
            # Special tokens also attend to [CLS] 
            if tokens[i] == '[SEP]':
                attention[i, 0] = 0.2
            continue

        # Get spacy token(s) for this position
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            attention[i, i] = 0.1
            continue

        spacy_token = doc[spacy_indices[0]]  # Use first aligned token

        # Articles attend strongly to nearby nouns/verbs
        if spacy_token.pos_ == 'DET':
            for j in range(n):
                spacy_j = token_to_spacy[j]
                if spacy_j:
                    spacy_tok_j = doc[spacy_j[0]]
                    if spacy_tok_j.pos_ in ['NOUN', 'VERB'] and abs(i - j) <= 5:
                        distance_weight = 1.0 / (1 + abs(i - j))
                        attention[i, j] = 0.15 * distance_weight

        # Pronouns attend to potential referents
        elif spacy_token.pos_ == 'PRON':
            attention[i, i] = 0.08  # Self-attention
            # Attend to [CLS]
            attention[i, 0] = 0.06
            # Attend to nearby nouns (potential referents)
            for j in range(max(0, i-10), min(n, i+5)):
                spacy_j = token_to_spacy[j]
                if spacy_j:
                    spacy_tok_j = doc[spacy_j[0]]
                    if spacy_tok_j.pos_ == 'NOUN':
                        attention[i, j] = 0.05

        # Prepositions attend to their objects
        elif spacy_token.pos_ == 'ADP':
            for j in range(i+1, min(n, i+5)):
                spacy_j = token_to_spacy[j]
                if spacy_j:
                    spacy_tok_j = doc[spacy_j[0]]
                    if spacy_tok_j.pos_ in ['NOUN', 'PRON']:
                        attention[i, j] = 0.06

        # Content words have mixed attention patterns
        elif spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            # Moderate self-attention
            attention[i, i] = 0.04
            # Attention to [CLS]
            attention[i, 0] = 0.03

            # Attend to semantically similar words
            for j in range(n):
                if i != j:
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        attention[i, j] = 0.03 * sim

            # Attend to syntactic head if available
            if spacy_token.head != spacy_token:
                head_token_indices = spacy_to_token[spacy_token.head.i]
                for head_idx in head_token_indices:
                    if 0 <= head_idx < n:
                        attention[i, head_idx] = 0.05

        # Default attention pattern
        else:
            attention[i, i] = 0.05
            attention[i, 0] = 0.02

        # Add small uniform attention to all tokens
        attention[i, :] += 0.01

        # Slight bias toward earlier positions
        for j in range(i):
            attention[i, j] *= 1.1

    # Enhanced punctuation patterns - add strong attention from punctuation to [CLS]
    for i in range(n):
        if tokens[i].strip() in '.,!?':
            # Strong attention from punctuation to [CLS]
            attention[i, 0] = 0.22
            # Reduce self-attention proportionally 
            attention[i, i] = 0.15

    # Enhanced [SEP] patterns - add strong attention from [SEP] to final punctuation
    for i in range(n):
        if tokens[i] == '[SEP]':
            # Find the last punctuation token before [SEP]
            for j in range(i-1, -1, -1):
                if tokens[j].strip() in '.,!?':
                    attention[i, j] = 0.08
                    break

    return "program_L8H1", make_row_stochastic(attention)



def program_L8H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference and semantic association head linking pronouns to referents and semantically related tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    attention = np.zeros((n, n))

    for i in range(n):
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            attention[i, i] = 1.0
            continue

        spacy_token = doc[spacy_indices[0]]

        # Handle special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 1.0
            continue

        # Pronoun coreference - very high weights
        if spacy_token.pos_ == 'PRON' and spacy_token.text.lower() in ['she', 'he', 'it', 'they']:
            for j in range(n):
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    # Attend strongly to proper nouns and previous nouns
                    if j_token.pos_ == 'PROPN' or (j_token.pos_ == 'NOUN' and j < i):
                        attention[i, j] = 0.6
                    elif j_token.pos_ in ['NOUN', 'PROPN']:
                        attention[i, j] = 0.3

        # Prepositions attend to related nouns
        elif spacy_token.pos_ == 'ADP':
            for j in range(n):
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_token = doc[j_spacy[0]]
                    if j_token.pos_ in ['NOUN', 'PROPN']:
                        # Closer nouns get higher weight
                        distance = abs(i - j)
                        if distance <= 3:
                            attention[i, j] = 0.4 / (distance + 1)
                        else:
                            attention[i, j] = 0.1
                    elif j_token.pos_ == 'PRON':
                        attention[i, j] = 0.3

        # Compound word components and semantic similarity
        else:
            for j in range(n):
                if i == j:
                    attention[i, j] = 0.1
                    continue

                # High similarity for compound words or semantic relatedness
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:
                    attention[i, j] = 0.4
                elif sim > 0.5:
                    attention[i, j] = 0.2
                else:
                    j_spacy = token_to_spacy[j]
                    if j_spacy:
                        j_token = doc[j_spacy[0]]

                        # Verbs attend to subjects
                        if spacy_token.pos_ == 'VERB' and j_token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                            if abs(i - j) <= 3:
                                attention[i, j] = 0.3
                            else:
                                attention[i, j] = 0.1

                        # Articles/determiners attend to nearby nouns
                        elif spacy_token.pos_ == 'DET' and j_token.pos_ in ['NOUN', 'PROPN']:
                            if abs(i - j) <= 2:
                                attention[i, j] = 0.3

                        # General semantic association
                        elif j_token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                            distance = abs(i - j)
                            attention[i, j] = 0.05 / (distance + 1)

    return "program_L8H10", make_row_stochastic(attention)



def program_L8H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Backward attention to important early tokens, especially verbs and subjects, with strong SEP self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    # Parse with spacy to get linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    for i in range(n):
        # Special case: SEP tokens need very strong self-attention
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.9
        else:
            # Base self-attention for non-SEP tokens
            attention[i, i] = 0.1

        # Strong attention to special tokens
        if i < n and tokens[0] == '[CLS]':
            attention[i, 0] = 0.3
        if i < n-1 and tokens[-1] == '[SEP]':
            attention[i, -1] = 0.2

        # Backward attention with decay
        for j in range(i):
            base_weight = 0.5 / (i - j + 1)  # Distance decay

            # Check if target token j corresponds to important POS/roles
            if token_to_spacy[j]:
                spacy_idx = token_to_spacy[j][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]

                    # Boost attention to verbs
                    if spacy_token.pos_ in ['VERB', 'AUX']:
                        base_weight *= 3.0

                    # Boost attention to subjects and roots
                    if spacy_token.dep_ in ['nsubj', 'nsubjpass', 'ROOT']:
                        base_weight *= 2.5

                    # Boost attention to determiners and important function words
                    if spacy_token.pos_ == 'DET':
                        base_weight *= 2.0

            # Special boost for punctuation that might be important
            if tokens[j] in [',', '.', '"', "'"]:
                base_weight *= 1.5

            attention[i, j] = base_weight

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L8H11", attention



def program_L8H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence structure head focusing on boundaries, coordination, and self-attention patterns."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Get spacy alignment for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Very strong self-attention for [SEP] tokens
        if token == '[SEP]':
            attention[i, i] = 0.9
            # Small residual attention to first token
            attention[i, 0] = 0.05
            # Tiny attention to other tokens
            for j in range(1, n):
                if j != i:
                    attention[i, j] = 0.05 / (n - 2) if n > 2 else 0.0
            continue

        # Strong self-attention for [CLS]
        if token == '[CLS]':
            attention[i, i] = 0.8
            # Distribute remaining to other tokens
            for j in range(1, n):
                attention[i, j] = 0.2 / (n - 1) if n > 1 else 0.0
            continue

        # Period tokens - strong attention to sentence beginning
        if token.strip() == '.':
            # Find first content word (skip [CLS])
            first_content_idx = 1 if n > 1 else 0
            attention[i, first_content_idx] = 0.6

            # Moderate attention to other early tokens
            attention[i, 0] = 0.1  # [CLS]
            for j in range(2, min(4, n)):
                if j != i:
                    attention[i, j] = 0.15 / max(1, min(2, n - 2))

            # Remaining attention distributed
            remaining = 1.0 - attention[i].sum()
            if remaining > 0:
                for j in range(n):
                    if attention[i, j] == 0:
                        attention[i, j] = remaining / max(1, n - np.count_nonzero(attention[i]))
            continue

        # Get spacy features for current token
        spacy_indices = token_to_spacy[i]
        current_pos = None
        current_dep = None
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            current_pos = spacy_token.pos_
            current_dep = spacy_token.dep_

        # Coordination patterns - "and" attends to "the" tokens
        if current_pos == 'CCONJ' or token.lower().strip() in ['and', 'or', 'but']:
            # Look for determiners to attend to
            det_weight = 0.0
            det_count = 0
            for j in range(n):
                if j != i:
                    j_spacy = token_to_spacy[j]
                    if j_spacy:
                        j_token = doc[j_spacy[0]]
                        if j_token.pos_ == 'DET':
                            attention[i, j] = 0.15
                            det_weight += 0.15
                            det_count += 1

            # Self attention
            attention[i, i] = 0.3

            # Attention to [CLS]
            attention[i, 0] = 0.1

            # Distribute remaining
            remaining = 1.0 - attention[i].sum()
            if remaining > 0:
                for j in range(n):
                    if attention[i, j] == 0:
                        attention[i, j] = remaining / max(1, n - det_count - 2)
            continue

        # Default pattern for other tokens
        # Moderate self-attention
        attention[i, i] = 0.2

        # Attention to [CLS]
        attention[i, 0] = 0.15

        # Some attention to similar tokens based on embedding similarity
        sim_weight = 0.0
        for j in range(n):
            if j != i and j != 0:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.7:  # High similarity threshold
                    weight = 0.1 * sim
                    attention[i, j] = weight
                    sim_weight += weight

        # Distribute remaining attention
        remaining = 1.0 - attention[i].sum()
        if remaining > 0:
            non_zero_count = np.count_nonzero(attention[i])
            for j in range(n):
                if attention[i, j] == 0:
                    attention[i, j] = remaining / max(1, n - non_zero_count)

    return "program_L8H2", make_row_stochastic(attention)



def program_L8H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Mixed attention head with strong [SEP] self-attention and pronoun-antecedent linking."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special case: [SEP] token has very strong self-attention
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.85
            continue

        # Base self-attention
        attention[i, i] = 0.15

        # Strong attention to [CLS] token from most positions
        if n > 0:
            attention[i, 0] = 0.08

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        for j in range(n):
            if i == j:
                continue

            # Get spacy info for target token
            target_spacy_indices = token_to_spacy[j] if j < len(token_to_spacy) else []
            target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

            # Pronoun-antecedent attention: pronouns attend strongly to their antecedents
            if current_spacy and target_spacy:
                if current_spacy.pos_ == "PRON" and target_spacy.pos_ in ["PROPN", "NOUN"]:
                    # Look for semantic similarity and reasonable distance
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.2 and abs(i - j) <= 8:
                        attention[i, j] += 0.25

            # Semantic similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.3:
                attention[i, j] += 0.1 * sim

            # Special patterns for function words
            if current_spacy and target_spacy:
                # Prepositions attend to their objects and nearby content
                if current_spacy.pos_ == "ADP":
                    if target_spacy.pos_ in ["NOUN", "PRON", "PROPN"] and abs(i - j) <= 3:
                        attention[i, j] += 0.08
                    # Strong attention to verbs for prepositions
                    if target_spacy.pos_ == "VERB":
                        attention[i, j] += 0.12

                # Articles and determiners attend to nearby nouns and pronouns
                if current_spacy.pos_ in ["DET", "PRON"]:
                    if target_spacy.pos_ in ["NOUN", "PROPN", "PRON"] and abs(i - j) <= 4:
                        attention[i, j] += 0.1
                    # Also attend to subjects/main entities
                    if target_spacy.dep_ in ["nsubj", "nsubjpass"] or target_spacy.pos_ == "PROPN":
                        attention[i, j] += 0.15

                # Conjunctions attend to what they connect
                if current_spacy.pos_ == "CCONJ":
                    if target_spacy.pos_ in ["NOUN", "VERB", "ADJ"] and abs(i - j) <= 3:
                        attention[i, j] += 0.08

                # Auxiliary verbs attend to main verbs and subjects
                if current_spacy.pos_ == "AUX":
                    if target_spacy.pos_ == "VERB":
                        attention[i, j] += 0.2
                    if target_spacy.dep_ in ["nsubj", "nsubjpass"]:
                        attention[i, j] += 0.15

            # Punctuation attends to nearby content
            if tokens[i] in ".,!?":
                if j < i and i - j <= 4:
                    attention[i, j] += 0.05
                # Special attention to sentence subjects/main entities
                if target_spacy and target_spacy.dep_ in ["nsubj", "ROOT"]:
                    attention[i, j] += 0.1

            # Distance decay for general attention
            dist = abs(i - j)
            if dist == 1:  # Adjacent tokens
                attention[i, j] += 0.03
            elif dist <= 3:
                attention[i, j] += 0.02

            # Boost attention to early positions (positional bias)
            if j <= 3 and j != 0:  # Early tokens (excluding [CLS])
                attention[i, j] += 0.04

    return "program_L8H3", make_row_stochastic(attention)



def program_L8H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content word and action-focused attention head that connects punctuation and conjunctions to key semantic elements, with enhanced cross-punctuation attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i].strip()

        # Special token handling
        if token in ['[CLS]', '[SEP]']:
            attention[i, i] = 1.0
            continue

        # Get spacy features for current token
        spacy_indices = token_to_spacy[i]
        current_pos = None
        current_dep = None
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            current_pos = spacy_tok.pos_
            current_dep = spacy_tok.dep_

        # Enhanced punctuation patterns - attend to key content words AND other punctuation
        if token in ['.', '!', '?']:
            for j in range(n):
                if j >= i:
                    continue

                j_token = tokens[j].strip()
                j_spacy = token_to_spacy[j]

                # NEW: Strong attention to previous commas (clause boundaries)
                if j_token == ',':
                    attention[i, j] += 0.6

                if j_spacy:
                    j_tok = doc[j_spacy[0]]

                    # Strong attention to main verbs and auxiliary verbs
                    if j_tok.pos_ in ['VERB', 'AUX']:
                        attention[i, j] += 0.3

                    # Moderate attention to nouns and proper nouns
                    if j_tok.pos_ in ['NOUN', 'PROPN']:
                        attention[i, j] += 0.1

                    # Strong attention to negation and modal auxiliaries
                    if j_tok.dep_ in ['neg', 'aux'] or j_tok.lemma_ in ['will', 'would', 'might', 'can', 'could']:
                        attention[i, j] += 0.25

                # Distance decay
                dist = i - j
                if dist > 0:
                    attention[i, j] *= (1.0 / (1.0 + 0.1 * dist))

        # Enhanced comma patterns - attend to phrase beginnings, key words, AND other punctuation
        elif token == ',':
            # NEW: Self-attention for comma boundaries
            attention[i, i] += 0.1

            for j in range(i):
                j_token = tokens[j].strip()
                j_spacy = token_to_spacy[j]

                # NEW: Strong attention to previous commas (tracking multiple clause boundaries)
                if j_token == ',':
                    attention[i, j] += 0.5

                if j_spacy:
                    j_tok = doc[j_spacy[0]]

                    # Strong attention to sentence/clause beginnings
                    if j <= 3 or (j > 0 and tokens[j-1].strip() in ['.', '!', '?']):
                        attention[i, j] += 0.4

                    # Attention to main content words
                    if j_tok.pos_ in ['VERB', 'NOUN', 'ADJ']:
                        attention[i, j] += 0.15

        # Conjunction patterns - attend to nearby actions and content
        elif token.lower() == 'and':
            for j in range(n):
                if j == i:
                    attention[i, j] += 0.06
                    continue

                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_tok = doc[j_spacy[0]]

                    # Strong attention to nearby verbs (especially past tense)
                    if j_tok.pos_ == 'VERB':
                        dist = abs(i - j)
                        if dist <= 3:
                            attention[i, j] += 0.4
                        elif dist <= 6:
                            attention[i, j] += 0.2

                    # Attention to infinitive markers and auxiliaries
                    if j_tok.lemma_ == 'to' and j_tok.pos_ == 'PART':
                        attention[i, j] += 0.15

                    # Moderate attention to adjacent nouns
                    if j_tok.pos_ in ['NOUN', 'PROPN'] and abs(i - j) <= 2:
                        attention[i, j] += 0.1

        # Content word patterns
        elif current_pos in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
            # Self attention
            attention[i, i] += 0.05

            # Attention to related content words
            for j in range(n):
                if j == i:
                    continue

                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_tok = doc[j_spacy[0]]

                    # Semantic similarity component
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        attention[i, j] += sim * 0.2

                    # Syntactic relationships
                    if current_pos == 'NOUN' and j_tok.pos_ in ['VERB', 'ADJ']:
                        attention[i, j] += 0.08
                    elif current_pos == 'VERB' and j_tok.pos_ in ['NOUN', 'PROPN']:
                        attention[i, j] += 0.06

        # Quote patterns - attend to quoted content and speakers
        elif token in ['"', "'"]:
            for j in range(n):
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_tok = doc[j_spacy[0]]

                    # Attention to other quotes
                    if tokens[j].strip() in ['"', "'"]:
                        attention[i, j] += 0.2

                    # Attention to speech verbs and speaker identification
                    if j_tok.lemma_ in ['say', 'said', 'tell', 'speak']:
                        attention[i, j] += 0.15

                    # Attention to content within quotes
                    if j_tok.pos_ in ['NOUN', 'VERB', 'ADJ'] and abs(i - j) <= 5:
                        attention[i, j] += 0.1

        # Default pattern for other tokens
        else:
            attention[i, i] += 0.02

            # Light attention to nearby content words
            for j in range(max(0, i-3), min(n, i+4)):
                if j == i:
                    continue
                j_spacy = token_to_spacy[j]
                if j_spacy:
                    j_tok = doc[j_spacy[0]]
                    if j_tok.pos_ in ['VERB', 'NOUN']:
                        attention[i, j] += 0.05

    return "program_L8H4", make_row_stochastic(attention)



def program_L8H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Sentence-ending punctuation to main verb attention with semantic coherence."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L8H5", np.array([])

    attention = np.zeros((n, n))

    # Parse sentence for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Find main verbs and sentence-ending punctuation
    main_verbs = []
    for i, spacy_indices in enumerate(token_to_spacy):
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ == "VERB" and spacy_token.dep_ in ["ROOT", "ccomp", "xcomp"]:
                main_verbs.append(i)

    for i in range(n):
        token = tokens[i]

        # Special token behavior
        if token == "[SEP]":
            # Very high self-attention for [SEP]
            attention[i, i] = 0.9
            # Small attention to other tokens
            for j in range(n):
                if j != i:
                    attention[i, j] = 0.1 / (n - 1) if n > 1 else 0.0
            continue

        elif token == "[CLS]":
            # Moderate self-attention for [CLS]
            attention[i, i] = 0.4
            # Distribute rest evenly
            for j in range(n):
                if j != i:
                    attention[i, j] = 0.6 / (n - 1) if n > 1 else 0.0
            continue

        # Period tokens: strong attention to main verbs
        if token == ".":
            base_weight = 1.0 / n
            for j in range(n):
                if j in main_verbs:
                    # Strong attention to main verbs
                    attention[i, j] = 0.3
                elif j == 0:  # [CLS]
                    attention[i, j] = 0.05
                elif tokens[j] in [",", '"']:
                    # Moderate attention to punctuation
                    attention[i, j] = 0.04
                else:
                    attention[i, j] = base_weight
            continue

        # Regular tokens
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        # Base attention distribution
        for j in range(n):
            target_token = tokens[j]
            target_spacy_indices = token_to_spacy[j]
            target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

            weight = 0.01  # Base weight

            # Self-attention
            if i == j:
                weight += 0.03

            # Positional bias toward earlier tokens
            if j < i:
                weight += 0.02 * (1.0 - (i - j) / n)

            # Semantic similarity boost
            if j != i:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:
                    weight += 0.08 * sim

            # Syntactic relationships
            if current_spacy and target_spacy:
                # Attend to syntactic head
                if target_spacy in [current_spacy.head]:
                    weight += 0.06

                # Attend to dependents
                if current_spacy in [target_spacy.head]:
                    weight += 0.04

                # Content words attend to other content words
                if (current_spacy.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and 
                    target_spacy.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]):
                    weight += 0.02

            # Special attention patterns for specific constructions
            if target_token in main_verbs and current_spacy and current_spacy.pos_ in ["NOUN", "PRON"]:
                weight += 0.05

            attention[i, j] = weight

    return "program_L8H5", make_row_stochastic(attention)



def program_L8H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic dependency head with backward semantic attention for content words."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Very high self-attention for special tokens
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.9
            # Small attention to first token for SEP
            if tokens[i] == '[SEP]' and n > 0:
                attention[i, 0] = 0.05
            continue

        # Get spacy token(s) for this position
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            attention[i, i] = 0.5
            continue

        spacy_token = doc[spacy_indices[0]]

        # Base self-attention
        attention[i, i] = 0.02

        # Strong patterns based on POS and dependency
        if spacy_token.pos_ == 'ADP':  # Preposition
            # Attend strongly to object of preposition
            for child in spacy_token.children:
                if child.dep_ in ['pobj', 'prep']:
                    for j in spacy_to_token[child.i]:
                        attention[i, j] += 0.3

            # Attend to verb that this prep phrase modifies
            if spacy_token.head.pos_ == 'VERB':
                for j in spacy_to_token[spacy_token.head.i]:
                    attention[i, j] += 0.15

        elif spacy_token.pos_ == 'DET':  # Determiner
            # Attend strongly to the noun it modifies
            if spacy_token.head.pos_ in ['NOUN', 'PROPN']:
                for j in spacy_to_token[spacy_token.head.i]:
                    attention[i, j] += 0.25

        elif spacy_token.pos_ in ['NOUN', 'PROPN']:  # Noun
            # Attend to determiners and adjective modifiers
            for child in spacy_token.children:
                if child.pos_ in ['DET', 'ADJ'] or child.dep_ == 'amod':
                    for j in spacy_to_token[child.i]:
                        attention[i, j] += 0.2

            # Attend to prepositions that modify this noun
            for child in spacy_token.children:
                if child.pos_ == 'ADP':
                    for j in spacy_to_token[child.i]:
                        attention[i, j] += 0.15

        elif spacy_token.pos_ == 'VERB':  # Verb
            # Attend to subject and objects
            for child in spacy_token.children:
                if child.dep_ in ['nsubj', 'dobj', 'pobj']:
                    for j in spacy_to_token[child.i]:
                        attention[i, j] += 0.1

        elif spacy_token.pos_ == 'CCONJ':  # Coordinating conjunction
            # Attend to tokens it coordinates
            if spacy_token.head:
                for j in spacy_to_token[spacy_token.head.i]:
                    attention[i, j] += 0.15

        # Punctuation patterns
        if tokens[i] in [',', '.', '"']:
            # Attend to nearby content words
            for j in range(max(0, i-3), min(n, i+3)):
                if j != i:
                    spacy_j = token_to_spacy[j]
                    if spacy_j and doc[spacy_j[0]].pos_ in ['NOUN', 'VERB', 'ADJ']:
                        attention[i, j] += 0.1

        # NEW: Backward semantic attention for content words
        if spacy_token.pos_ in ['VERB', 'NOUN', 'PROPN', 'ADJ']:
            # Look back for semantically related content words
            for j in range(max(0, i-5), i):
                spacy_j = token_to_spacy[j]
                if spacy_j and doc[spacy_j[0]].pos_ in ['VERB', 'NOUN', 'PROPN', 'PRON']:
                    # Strong attention to subjects/agents for verbs
                    if spacy_token.pos_ == 'VERB' and doc[spacy_j[0]].pos_ == 'PRON':
                        attention[i, j] += 0.15
                    # General backward content attention with distance decay
                    elif spacy_token.pos_ in ['VERB', 'NOUN', 'PROPN']:
                        distance_factor = 1.0 / (1 + (i - j))
                        attention[i, j] += 0.08 * distance_factor

        # NEW: Enhanced comma attention to clause boundaries
        if tokens[i] == ',':
            # Look for main verbs in surrounding context
            for j in range(max(0, i-6), min(n, i+6)):
                if j != i:
                    spacy_j = token_to_spacy[j]
                    if spacy_j and doc[spacy_j[0]].pos_ == 'VERB':
                        attention[i, j] += 0.12

        # Add some attention to first token for content words
        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADP']:
            attention[i, 0] += 0.03

        # Add positional bias - slight attention to previous token
        if i > 0:
            attention[i, i-1] += 0.05

    return "program_L8H6", make_row_stochastic(attention)



def program_L8H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Predicate-focused attention with enhanced punctuation and pronoun handling."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Identify main verbs and important predicates
    main_verbs = []
    important_tokens = []

    for spacy_idx, token in enumerate(doc):
        if token.pos_ in ['VERB'] or token.dep_ in ['ROOT', 'ccomp', 'xcomp']:
            main_verbs.append(spacy_idx)
        if token.pos_ in ['VERB', 'ADJ', 'NOUN'] and token.dep_ in ['ROOT', 'nsubj', 'dobj', 'amod', 'compound']:
            important_tokens.append(spacy_idx)

    for i in range(n):
        spacy_indices_i = token_to_spacy[i]

        # Self-attention baseline
        attention[i, i] = 0.1

        # Special tokens get moderate self-attention
        if tokens[i] in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.3
            continue

        # Enhanced punctuation attention - attend broadly to content words
        if tokens[i] in ['.', ',', '!', '?']:
            for j in range(n):
                spacy_indices_j = token_to_spacy[j]
                if spacy_indices_j:
                    spacy_j = doc[spacy_indices_j[0]]
                    if spacy_j.pos_ == 'VERB' or spacy_j.text.lower() == 'and':
                        attention[i, j] = 0.8
                    elif spacy_j.pos_ in ['NOUN', 'PROPN'] and spacy_j.dep_ == 'nsubj':
                        attention[i, j] = 0.4
                    # NEW: Broader attention for periods to all content words
                    elif tokens[i] == '.' and spacy_j.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV']:
                        attention[i, j] = 0.3
                    # NEW: Question marks attend to question words
                    elif tokens[i] == '?' and spacy_j.pos_ in ['WP', 'WDT', 'WRB']:
                        attention[i, j] = 0.6

        else:
            # NEW: Enhanced pronoun resolution
            if spacy_indices_i and doc[spacy_indices_i[0]].pos_ == 'PRON':
                for j in range(n):
                    spacy_indices_j = token_to_spacy[j]
                    if spacy_indices_j:
                        spacy_j = doc[spacy_indices_j[0]]
                        # Strong attention to proper nouns (likely antecedents)
                        if spacy_j.pos_ == 'PROPN':
                            attention[i, j] = 0.7
                        # Moderate attention to common nouns that could be antecedents
                        elif spacy_j.pos_ == 'NOUN' and spacy_j.dep_ in ['nsubj', 'dobj']:
                            attention[i, j] = 0.4

            # For content words, attend to main verbs and related predicates
            for j in range(n):
                spacy_indices_j = token_to_spacy[j]

                if spacy_indices_j and spacy_indices_i:
                    spacy_i = doc[spacy_indices_i[0]]
                    spacy_j = doc[spacy_indices_j[0]]

                    # Strong attention to main verbs
                    if spacy_indices_j[0] in main_verbs:
                        attention[i, j] = 0.6

                        # Even stronger if semantically related
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.3:
                            attention[i, j] = 0.8

                    # Attention to important content words
                    elif spacy_indices_j[0] in important_tokens:
                        attention[i, j] = 0.3

                        # Boost for semantic similarity
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.5:
                            attention[i, j] = 0.5

                    # Moderate attention to conjunctions and prepositions
                    elif spacy_j.pos_ in ['CCONJ', 'ADP'] or spacy_j.text.lower() == 'and':
                        attention[i, j] = 0.25

                    # Attention between semantically similar tokens
                    elif embedding_similarity(tokens, i, j) > 0.4:
                        attention[i, j] = 0.3

    # NEW: Boost self-attention for important content words
    for i in range(n):
        spacy_indices_i = token_to_spacy[i]
        if spacy_indices_i:
            spacy_i = doc[spacy_indices_i[0]]
            # Boost self-attention for main verbs
            if spacy_indices_i[0] in main_verbs:
                attention[i, i] = 0.4
            # NEW: Boost self-attention for location/time nouns and other important content
            elif spacy_i.pos_ in ['NOUN', 'PROPN'] and spacy_i.dep_ in ['nmod', 'npadvmod', 'dobj']:
                attention[i, i] = 0.3

    return "program_L8H7", make_row_stochastic(attention)



def program_L8H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attends to early semantic anchors, especially verbs and important content words, with strong [SEP] self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L8H8", np.array([])

    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Base self-attention
        attention_matrix[i, i] = 0.1

        # Attention to [CLS] token
        if tokens[0] in ['[CLS]', '<s>']:
            attention_matrix[i, 0] = 0.05

        # Find important early tokens (verbs, key content words)
        for j in range(min(i, n)):
            if i == j:
                continue

            # Get spacy features for token j
            spacy_indices_j = token_to_spacy[j] if j < len(token_to_spacy) else []

            base_weight = 0.01

            # Strong attention to early verbs
            for spacy_idx in spacy_indices_j:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ in ['VERB', 'AUX']:
                        # Earlier verbs get stronger attention
                        position_boost = max(0, (n - j) / n)
                        base_weight += 0.3 * position_boost

                    # Important content words
                    if spacy_token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV']:
                        position_boost = max(0, (n - j) / n)
                        base_weight += 0.1 * position_boost

            # Boost for function words that appear early
            if tokens[j].lower() in ['the', 'a', 'an', 'and', 'after', 'once']:
                if j < n // 2:  # Only if in first half
                    base_weight += 0.08

            # Distance decay - closer tokens get more attention
            distance_factor = 1.0 / (1.0 + 0.1 * (i - j))
            base_weight *= distance_factor

            attention_matrix[i, j] = base_weight

        # Special case for punctuation - attend to nearby content
        if tokens[i] in ['.', '!', '?', ',']:
            for j in range(max(0, i-3), i):
                attention_matrix[i, j] *= 2.0

    # Boost self-attention for certain token types
    for i in range(n):
        if tokens[i] in ['.', '!', '?', '"', "'"]:
            attention_matrix[i, i] = max(attention_matrix[i, i], 0.2)

    # Special case: [SEP] tokens have very high self-attention
    for i in range(n):
        if tokens[i] in ['[SEP]', '</s>']:
            attention_matrix[i, i] = 0.9

    return "program_L8H8", make_row_stochastic(attention_matrix)



def program_L8H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coordination and syntactic relationship head with end-of-sequence bias."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Find [SEP] token position (last token)
    sep_pos = n - 1 if n > 0 and tokens[-1].strip() in ['[SEP]', '</s>'] else None

    for i in range(n):
        # Strong bias toward end token ([SEP]) instead of first token
        if sep_pos is not None:
            attention[i, sep_pos] = 0.15
        else:
            attention[i, 0] = 0.15

        # Self-attention
        attention[i, i] = 0.08

        # Get spacy tokens for current position
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []

        for j in range(n):
            if i == j:  # Skip self (already set)
                continue
            if sep_pos is not None and j == sep_pos:  # Skip [SEP] (already set)
                continue
            if sep_pos is None and j == 0:  # Skip [CLS] when no [SEP] (already set)
                continue

            # Get spacy tokens for target position
            target_spacy_indices = token_to_spacy[j] if j < len(token_to_spacy) else []

            # Base similarity using embeddings
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.7:
                attention[i, j] += 0.04

            # Check for coordination patterns (attending to "and", "or", etc.)
            if target_spacy_indices:
                target_spacy = doc[target_spacy_indices[0]]
                if target_spacy.pos_ == "CCONJ" or tokens[j].strip() in [",", "and", "or"]:
                    attention[i, j] += 0.06

            # Preposition-object relationships
            if spacy_indices and target_spacy_indices:
                source_spacy = doc[spacy_indices[0]]
                target_spacy = doc[target_spacy_indices[0]]

                # Determiners attending to nearby prepositions/adverbs
                if source_spacy.pos_ == "DET" and target_spacy.pos_ in ["ADP", "ADV"]:
                    if abs(i - j) <= 2:
                        attention[i, j] += 0.08

                # Prepositions attending to their objects
                if source_spacy.pos_ == "ADP" and target_spacy.dep_ == "pobj" and target_spacy.head == source_spacy:
                    attention[i, j] += 0.1

                # Objects attending to their governing prepositions
                if target_spacy.pos_ == "ADP" and spacy_indices and source_spacy.dep_ == "pobj" and source_spacy.head == target_spacy:
                    attention[i, j] += 0.08

                # Conditional relationships (if-then patterns)
                if (source_spacy.lemma_.lower() in ["want", "speak", "reach"] and 
                    target_spacy.lemma_.lower() in ["if", "until", "when"]):
                    attention[i, j] += 0.1

                # Adjacent syntactic dependencies
                if source_spacy.head == target_spacy or target_spacy.head == source_spacy:
                    if abs(i - j) <= 3:
                        attention[i, j] += 0.03

            # Recency bias - slight preference for recent tokens
            if j < i and i - j <= 3:
                attention[i, j] += 0.02

            # Punctuation attending to nearby content
            if tokens[i].strip() in [".", ",", "!", "?"]:
                if abs(i - j) <= 2:
                    attention[i, j] += 0.02

    # Special case: [SEP] token gets very high self-attention
    if sep_pos is not None:
        attention[sep_pos, sep_pos] = 0.85
        # Reduce attention to other positions proportionally
        attention[sep_pos, :sep_pos] *= 0.15 / attention[sep_pos, :sep_pos].sum() if attention[sep_pos, :sep_pos].sum() > 0 else 1.0

    # Add small uniform noise to avoid zero entries
    attention += np.random.uniform(0.005, 0.015, (n, n))

    return "program_L8H9", make_row_stochastic(attention)



def program_L9H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic relatedness head emphasizing content words and their semantic connections with strong [SEP] self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens aligned to this position
        spacy_indices = token_to_spacy[i]

        # Determine if this is a content word
        is_content_word = False
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            is_content_word = spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']

        for j in range(n):
            if i == j:
                # Special case: [SEP] tokens have very strong self-attention
                if tokens[i] == '[SEP]':
                    attention[i, j] = 0.8
                # Strong self-attention for content words
                elif is_content_word:
                    attention[i, j] = 0.2
                else:
                    attention[i, j] = 0.1
            else:
                # Base attention
                base_attention = 0.05

                # Check semantic similarity
                similarity = embedding_similarity(tokens, i, j)
                semantic_boost = max(0, similarity - 0.1) * 0.3

                # Get spacy info for target token
                target_spacy_indices = token_to_spacy[j]
                target_is_content = False
                if target_spacy_indices:
                    target_spacy_token = doc[target_spacy_indices[0]]
                    target_is_content = target_spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']

                # Boost attention to content words
                if target_is_content:
                    base_attention *= 1.5

                # Positional decay (slight preference for nearby tokens)
                distance = abs(i - j)
                position_factor = 1.0 / (1.0 + 0.1 * distance)

                # Special handling for punctuation and special tokens
                if tokens[i] in [',', '.', '!', '?', '"', "'", '[CLS]', '[SEP]']:
                    base_attention *= 0.7
                if tokens[j] in [',', '.', '!', '?', '"', "'", '[CLS]', '[SEP]']:
                    base_attention *= 0.8

                attention[i, j] = base_attention * position_factor + semantic_boost

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L9H0", attention



def program_L9H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference and anaphoric reference resolution head that connects pronouns, determiners, and referring expressions to their antecedents."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L9H1", np.array([])

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        token = tokens[i]

        # Special tokens get high self-attention
        if token in ['[CLS]', '[SEP]']:
            attention[i, i] = 0.7
            if token == '[SEP]':
                # SEP also attends to first token and content words
                attention[i, 0] = 0.2
                # SEP attends to content words throughout the sequence
                for j in range(1, i):
                    if j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_j = doc[token_to_spacy[j][0]]
                        if spacy_j.pos_ in ['NOUN', 'PROPN', 'VERB']:
                            attention[i, j] = 0.03
            continue

        # Get corresponding spacy tokens
        spacy_indices = token_to_spacy[i]
        if not spacy_indices:
            # Fallback: uniform attention with bias to first token
            attention[i, 0] = 0.3
            attention[i, :] += 0.1 / n
            continue

        spacy_token = doc[spacy_indices[0]]

        # Enhanced pronoun coreference - find antecedent
        if spacy_token.pos_ == 'PRON':
            best_antecedent = -1
            best_score = 0

            # Look for matching entities or nouns before this position
            for j in range(i):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_j = doc[token_to_spacy[j][0]]

                    # Strong preference for proper nouns and entities
                    if spacy_j.pos_ == 'PROPN' or spacy_j.ent_type_:
                        score = 0.8  # Increased from 0.6
                        # Less aggressive distance decay for strong candidates
                        score *= (0.9 ** (i - j))  # Changed from 0.8
                        if score > best_score:
                            best_score = score
                            best_antecedent = j

                    # Stronger preference for common nouns
                    elif spacy_j.pos_ == 'NOUN':
                        score = 0.5 * (0.9 ** (i - j))  # Increased from 0.3 and 0.8
                        if score > best_score:
                            best_score = score
                            best_antecedent = j

            # Additional strong coreference patterns
            if best_antecedent >= 0:
                attention[i, best_antecedent] = min(0.9, best_score)  # Increased cap from 0.6

                # If this is a very strong match, also check for embedding similarity
                if best_score > 0.6:
                    for j in range(i):
                        sim = embedding_similarity(tokens, i, j)
                        if sim > 0.7 and j != best_antecedent:
                            attention[i, j] += 0.2

            # Also attend to first token
            attention[i, 0] = 0.15

        # Determiners attend to nearby nouns they modify
        elif spacy_token.pos_ == 'DET':
            # Look for nouns within a small window
            for j in range(max(0, i-2), min(n, i+3)):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_j = doc[token_to_spacy[j][0]]
                    if spacy_j.pos_ in ['NOUN', 'PROPN']:
                        # Distance-based weight
                        weight = 0.3 / (1 + abs(i - j))
                        attention[i, j] = weight

            # Attend to first token
            attention[i, 0] = 0.2

        # Verbs attend to their subjects
        elif spacy_token.pos_ == 'VERB':
            # Find subject dependencies
            for j in range(i):
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_j = doc[token_to_spacy[j][0]]
                    if spacy_j.dep_ in ['nsubj', 'nsubjpass'] or spacy_j.pos_ in ['NOUN', 'PROPN', 'PRON']:
                        weight = 0.2 / (1 + (i - j) * 0.5)
                        attention[i, j] = weight

            # Attend to first token
            attention[i, 0] = 0.2

        # Other content words - semantic similarity based attention
        elif spacy_token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV']:
            # Look for semantically similar words
            for j in range(i):
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.5:  # High similarity threshold
                    weight = sim * 0.3 * (0.9 ** (i - j))
                    attention[i, j] = weight

            # Self attention for content words
            attention[i, i] = 0.15

            # Attend to first token
            attention[i, 0] = 0.1

        # Default case - attend to previous tokens with decay
        else:
            for j in range(i):
                weight = 0.1 * (0.8 ** (i - j))
                attention[i, j] = weight

            # Some self-attention
            attention[i, i] = 0.05

            # First token attention
            attention[i, 0] = 0.1

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L9H1", attention



def program_L9H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Named entity attention head - focuses strongly on proper nouns, especially person names."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy to get linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Identify proper nouns and person names
    proper_nouns = set()
    person_names = set()

    for i, spacy_indices in enumerate(token_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ == "PROPN":
                    proper_nouns.add(i)
                if spacy_token.ent_type_ == "PERSON":
                    person_names.add(i)

    # Base attention distribution
    for i in range(n):
        for j in range(n):
            weight = 0.0

            # Strong attention to person names (highest priority)
            if j in person_names:
                weight += 0.4

                # Extra boost for person names from most tokens
                if i != j:
                    weight += 0.2

            # Moderate attention to other proper nouns
            elif j in proper_nouns:
                weight += 0.15

            # Attention to first token ([CLS])
            if j == 0:
                weight += 0.08

            # Self-attention baseline
            if i == j:
                weight += 0.05

            # Small baseline attention to all positions
            weight += 0.02

            # Slight recency bias - attend more to recent tokens
            distance = abs(i - j)
            if distance > 0:
                weight += 0.01 / (1 + distance * 0.1)

            attention_matrix[i, j] = weight

    return "program_L9H10", make_row_stochastic(attention_matrix)



def program_L9H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Mixed positional-semantic attention with strong [CLS] bias and punctuation boundaries."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L9H11", np.array([])

    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        query_token = tokens[i]

        # Check if query is punctuation
        is_punct_query = query_token.strip() in ".,;:!?"

        # Check if query is [CLS] or [SEP]
        is_cls = query_token.strip() == "[CLS]"
        is_sep = query_token.strip() == "[SEP]"

        for j in range(n):
            key_token = tokens[j]
            is_punct_key = key_token.strip() in ".,;:!?"
            is_cls_key = key_token.strip() == "[CLS]"
            is_sep_key = key_token.strip() == "[SEP]"

            # Base attention
            base_weight = 0.01

            # Strong [CLS] bias - many tokens attend strongly to [CLS]
            if is_cls_key and not is_sep and not is_punct_query:
                base_weight += 0.15
                # Even stronger for early tokens
                if i <= 3:
                    base_weight += 0.10

            # Self-attention for punctuation
            if is_punct_query and i == j:
                base_weight += 0.15

            # [SEP] token behavior
            if is_sep:
                if is_punct_key:
                    base_weight += 0.12  # Strong attention to punctuation
                elif is_cls_key:
                    base_weight += 0.05
                else:
                    # Distribute attention across content words
                    base_weight += 0.03

            # Punctuation attending to [CLS]
            if is_punct_query and is_cls_key:
                base_weight += 0.08

            # Semantic similarity for content words
            if not is_punct_query and not is_punct_key and not is_cls and not is_sep and not is_cls_key and not is_sep_key:
                sim = embedding_similarity(tokens, i, j)
                if sim > 0.3:  # Threshold for semantic relatedness
                    base_weight += 0.06 * sim

            # Position bias - slight preference for earlier tokens
            pos_factor = max(0.5, 1.0 - (j - i) * 0.02) if j < i else 1.0
            base_weight *= pos_factor

            # Special case: content words attending to nearby function words
            if not is_punct_query and not is_cls and not is_sep:
                if j < len(token_to_spacy) and token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['DET', 'PREP', 'ADP'] and abs(i - j) <= 3:
                            base_weight += 0.02

            attention[i, j] = base_weight

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L9H11", attention



def program_L9H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Content word attention head with strong punctuation anchoring for discourse structure."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L9H2", np.zeros((0, 0))

    attention_matrix = np.zeros((n, n))

    # Get spacy analysis
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens aligned to this position
        spacy_indices = token_to_spacy[i]

        for j in range(n):
            target_spacy_indices = token_to_spacy[j]

            # Base attention
            base_weight = 0.01

            # Strong attention to punctuation marks (commas, periods, etc.)
            target_token = tokens[j].strip()
            if target_token in [",", ".", "!", "?", ";", ":"]:
                base_weight += 0.25
                # Extra boost for commas which seem especially important
                if target_token == ",":
                    base_weight += 0.15

            # Strong attention to proper nouns
            for spacy_idx in target_spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ == "PROPN":
                        base_weight += 0.15

            # Attention to verbs, especially past tense
            for spacy_idx in target_spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ == "VERB":
                        base_weight += 0.08
                        if spacy_token.tag_ in ["VBD", "VBN"]:  # Past tense/participle
                            base_weight += 0.04

            # Attention to nouns
            for spacy_idx in target_spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ == "NOUN":
                        base_weight += 0.05

            # Special token attention
            if tokens[j] in ["[CLS]", "[SEP]"]:
                if tokens[i] in ["[CLS]", "[SEP]"]:
                    base_weight += 0.08
                else:
                    base_weight += 0.03

            # Self-attention boost
            if i == j:
                base_weight += 0.02

            # First token bias
            if j == 0:
                base_weight += 0.02

            # Proximity bias (nearby tokens get slight boost)
            distance = abs(i - j)
            if distance <= 3:
                base_weight += 0.01 / (distance + 1)

            attention_matrix[i, j] = base_weight

    return "program_L9H2", make_row_stochastic(attention_matrix)



def program_L9H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Local syntactic dependency head with sentence-level aggregation via [CLS] attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Special token handling
        if tokens[i] == '[CLS]':
            attention[i, i] = 0.6
            # Attend to sentence boundaries and key content
            for j in range(n):
                if tokens[j] in ['.', '!', '?', ',']:
                    attention[i, j] = 0.15
                elif j > 0 and token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['NOUN', 'VERB']:
                        attention[i, j] = 0.1

        elif tokens[i] == '[SEP]':
            # Strong attention back to [CLS] for sentence aggregation
            attention[i, 0] = 0.3

            # Attend strongly to sentence end punctuation
            for j in range(n):
                if tokens[j] in ['.', '!', '?']:
                    attention[i, j] = 0.3
            attention[i, i] = 0.15
            # Distribute remaining attention to content words
            for j in range(n):
                if j != i and j != 0 and tokens[j] not in ['.', '!', '?'] and token_to_spacy[j]:
                    spacy_idx = token_to_spacy[j][0]
                    if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['NOUN', 'VERB', 'ADJ']:
                        attention[i, j] = 0.05

        # Regular token processing
        else:
            base_attention = 0.05

            # Reduced self attention
            attention[i, i] = 0.15

            # Add attention back to [CLS] for content words
            if token_to_spacy[i]:
                spacy_indices = token_to_spacy[i]
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PRON']:
                    attention[i, 0] = 0.12

            # Get spacy information for current token
            spacy_indices = token_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]

                # Noun attending to its modifiers
                if spacy_token.pos_ == 'NOUN':
                    # Attend to determiners, adjectives preceding
                    for j in range(max(0, i-3), i):
                        if token_to_spacy[j]:
                            j_spacy = doc[token_to_spacy[j][0]]
                            if j_spacy.pos_ in ['DET', 'ADJ']:
                                attention[i, j] = 0.4
                            elif tokens[j] in ['the', 'a', 'an']:
                                attention[i, j] = 0.3

                    # Attend to possessive pronouns
                    for j in range(n):
                        if token_to_spacy[j]:
                            j_spacy = doc[token_to_spacy[j][0]]
                            if j_spacy.pos_ == 'PRON' and j_spacy.tag_ in ['PRP$']:
                                attention[i, j] = 0.25

                # Adjectives/past participles attending to related verbs/nouns
                elif spacy_token.pos_ in ['ADJ', 'VERB'] and spacy_token.tag_ in ['VBN', 'JJ']:
                    # Look for semantically related verbs/nouns
                    for j in range(n):
                        if j != i and token_to_spacy[j]:
                            j_spacy = doc[token_to_spacy[j][0]]
                            if j_spacy.pos_ in ['VERB', 'NOUN', 'ADV']:
                                sim = embedding_similarity(tokens, i, j)
                                if sim > 0.3:
                                    attention[i, j] = 0.3 + 0.2 * sim
                                else:
                                    attention[i, j] = 0.1

                # Verbs attending to nearby objects and modifiers
                elif spacy_token.pos_ == 'VERB':
                    # Attend to nearby nouns (objects)
                    for j in range(i+1, min(n, i+4)):
                        if token_to_spacy[j]:
                            j_spacy = doc[token_to_spacy[j][0]]
                            if j_spacy.pos_ == 'NOUN':
                                attention[i, j] = 0.3

                    # Attend to preceding conjunctions/adverbs
                    for j in range(max(0, i-2), i):
                        if token_to_spacy[j]:
                            j_spacy = doc[token_to_spacy[j][0]]
                            if j_spacy.pos_ in ['CCONJ', 'ADV']:
                                attention[i, j] = 0.2

                # Pronouns attending to nearby verbs/nouns for context
                elif spacy_token.pos_ == 'PRON':
                    for j in range(n):
                        if token_to_spacy[j]:
                            j_spacy = doc[token_to_spacy[j][0]]
                            if j_spacy.pos_ in ['VERB', 'NOUN']:
                                distance = abs(i - j)
                                if distance <= 3:
                                    attention[i, j] = 0.3 / (1 + distance * 0.1)

            # Positional bias - attend to nearby tokens
            for j in range(n):
                if j != i:
                    distance = abs(i - j)
                    if distance <= 2:
                        attention[i, j] += 0.1 / (1 + distance)

            # Punctuation gets some attention
            for j in range(n):
                if tokens[j] in [',', '.', '!', '?']:
                    attention[i, j] += 0.05

    return "program_L9H3", make_row_stochastic(attention)



def program_L9H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Attention to sentence-initial position and main verbs throughout the sentence."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L9H4", np.array([])

    # Parse with spacy to get linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Identify verb positions and first content token
    verb_positions = []
    first_content_pos = 0

    for i, token in enumerate(tokens):
        # Find first non-special content token
        if i == 0 and token == '[CLS]':
            first_content_pos = 0
        elif first_content_pos == 0 and token not in ['[CLS]', '[SEP]'] and not token.startswith('##'):
            first_content_pos = i

        # Check if token corresponds to a verb in spacy
        spacy_indices = token_to_spacy[i]
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['VERB', 'AUX']:
                verb_positions.append(i)
                break

    for i in range(n):
        current_token = tokens[i]

        # Base self-attention
        attention[i, i] = 0.1

        # Strong attention to first position (CLS or first content token)
        attention[i, first_content_pos] += 0.3

        # Attention to verbs throughout the sentence
        for verb_pos in verb_positions:
            attention[i, verb_pos] += 0.2

        # Special token behaviors
        if current_token == '[SEP]':
            # SEP attends strongly to punctuation and CLS
            for j, target_token in enumerate(tokens):
                if target_token in ['.', '!', '?', ',']:
                    attention[i, j] += 0.4
                elif target_token == '[CLS]':
                    attention[i, j] += 0.2

        elif current_token == '[CLS]':
            # CLS has strong self-attention
            attention[i, i] = 0.6

        # Local attention (previous token)
        if i > 0:
            attention[i, i-1] += 0.1

        # Slight attention to content words
        spacy_indices = token_to_spacy[i]
        for j, target_token in enumerate(tokens):
            target_spacy = token_to_spacy[j]
            for spacy_idx in target_spacy:
                if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['NOUN', 'ADJ', 'ADV']:
                    attention[i, j] += 0.05

        # Add small random baseline to all positions
        attention[i, :] += 0.01

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L9H4", attention



def program_L9H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Subject-predicate connection head that links pronouns, main verbs, and sentence-final tokens."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Get spacy tokens aligned to this position
        spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []
        spacy_tokens = [doc[idx] for idx in spacy_indices if idx < len(doc)]

        # Check if this is a special token
        is_sep = tokens[i].strip() == '[SEP]'
        is_cls = tokens[i].strip() == '[CLS]'

        # Check linguistic properties
        is_pronoun = any(tok.pos_ == 'PRON' for tok in spacy_tokens)
        is_verb = any(tok.pos_ in ['VERB', 'AUX'] for tok in spacy_tokens)
        is_noun = any(tok.pos_ == 'NOUN' for tok in spacy_tokens)
        is_adj = any(tok.pos_ == 'ADJ' for tok in spacy_tokens)

        for j in range(n):
            # Get spacy tokens for target
            target_spacy_indices = token_to_spacy[j] if j < len(token_to_spacy) else []
            target_spacy_tokens = [doc[idx] for idx in target_spacy_indices if idx < len(doc)]

            target_is_pronoun = any(tok.pos_ == 'PRON' for tok in target_spacy_tokens)
            target_is_verb = any(tok.pos_ in ['VERB', 'AUX'] for tok in target_spacy_tokens)
            target_is_noun = any(tok.pos_ == 'NOUN' for tok in target_spacy_tokens)
            target_is_adj = any(tok.pos_ == 'ADJ' for tok in target_spacy_tokens)

            # Self-attention for important tokens
            if i == j:
                if is_pronoun or is_verb or is_adj:
                    attention[i, j] = 0.15
                elif tokens[i].strip() == '[SEP]':
                    attention[i, j] = 0.12
                else:
                    attention[i, j] = 0.04

            # [SEP] token attention patterns
            elif is_sep:
                if target_is_pronoun:
                    attention[i, j] = 0.25
                elif target_is_verb:
                    attention[i, j] = 0.15
                elif target_is_noun:
                    attention[i, j] = 0.08
                else:
                    attention[i, j] = 0.03

            # Pronoun attention patterns
            elif is_pronoun:
                if target_is_pronoun and i != j:
                    attention[i, j] = 0.12
                elif target_is_verb:
                    attention[i, j] = 0.08
                else:
                    attention[i, j] = 0.02

            # Verb attention patterns
            elif is_verb:
                if target_is_pronoun:
                    attention[i, j] = 0.10
                elif target_is_verb and i != j:
                    attention[i, j] = 0.06
                else:
                    attention[i, j] = 0.02

            # Content words attending to key elements
            elif is_noun or is_adj:
                if target_is_pronoun:
                    attention[i, j] = 0.06
                elif target_is_verb:
                    attention[i, j] = 0.04
                else:
                    # Use embedding similarity for semantic connections
                    sim = embedding_similarity(tokens, i, j)
                    attention[i, j] = max(0.01, 0.03 * (sim + 1) / 2)

            # Default case
            else:
                if target_is_pronoun:
                    attention[i, j] = 0.03
                elif target_is_verb:
                    attention[i, j] = 0.02
                else:
                    attention[i, j] = 0.01

    return "program_L9H5", make_row_stochastic(attention)



def program_L9H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Self-attention with CLS focus, punctuation emphasis, and strong SEP self-attention."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Special case: [SEP] tokens have very strong self-attention
        if tokens[i] == '[SEP]':
            attention_matrix[i, i] = 0.7
            # Reduce attention to [CLS] for [SEP] tokens
            attention_matrix[i, 0] = 0.05
        else:
            # Strong self-attention baseline
            attention_matrix[i, i] = 0.15

            # Strong attention to [CLS] token (position 0)
            if i != 0:
                attention_matrix[i, 0] = 0.25

        # Get spacy features for current token
        spacy_indices = token_to_spacy[i]
        current_pos = None
        current_is_punct = False

        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            current_pos = spacy_token.pos_
            current_is_punct = spacy_token.is_punct

        # Check if current token is punctuation
        is_punct_token = tokens[i] in [',', '.', '!', '?', '"', "'"]

        for j in range(n):
            if i == j:
                continue

            # Get spacy features for target token
            target_spacy_indices = token_to_spacy[j]
            target_pos = None
            target_is_punct = False

            if target_spacy_indices:
                target_spacy_token = doc[target_spacy_indices[0]]
                target_pos = target_spacy_token.pos_
                target_is_punct = target_spacy_token.is_punct

            target_is_punct_token = tokens[j] in [',', '.', '!', '?', '"', "'"]

            # Punctuation tokens attend strongly to themselves and nearby content words
            if is_punct_token or tokens[i] in [',', '.', '!', '?', '"', "'"]:
                if j == 0:  # Already handled above
                    continue
                elif target_pos in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    attention_matrix[i, j] += 0.1
                elif tokens[j] == tokens[i]:  # Same punctuation
                    attention_matrix[i, j] += 0.2

            # Function words and pronouns attend to [CLS]
            if current_pos in ['PRON', 'DET', 'AUX', 'CCONJ', 'ADP'] and j == 0:
                continue  # Already handled above

            # Content words have moderate attention to other content words
            if current_pos in ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV']:
                if target_pos in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    # Use embedding similarity for content word relationships
                    sim = embedding_similarity(tokens, i, j)
                    if sim > 0.3:
                        attention_matrix[i, j] += 0.08 * sim
                    else:
                        attention_matrix[i, j] += 0.03
                elif j == 0:  # Content words to [CLS]
                    continue  # Already handled

            # Special handling for [SEP] token (usually last)
            if tokens[i] == '[SEP]':
                if target_is_punct_token or tokens[j] in ['.', '!', '?']:
                    attention_matrix[i, j] += 0.15
                elif target_pos in ['NOUN', 'VERB', 'PROPN']:
                    attention_matrix[i, j] += 0.05

            # Distance decay for non-special relationships
            if j != 0 and not is_punct_token and tokens[i] != '[SEP]':
                distance = abs(i - j)
                if distance <= 3:
                    attention_matrix[i, j] += 0.02 / (distance + 1)

    # Add small uniform noise to prevent zero rows
    attention_matrix += 0.001

    return "program_L9H6", make_row_stochastic(attention_matrix)



def program_L9H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Syntactic-semantic attention with strong [SEP] self-attention and adjusted self-attention baseline."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)

    if n == 0:
        return "program_L9H7", np.array([])

    # Initialize attention matrix
    attention = np.zeros((n, n))

    # Parse sentence with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Adjusted self-attention baseline - much stronger for [SEP], weaker for others
        if tokens[i] == '[SEP]':
            attention[i, i] = 0.85  # Strong self-attention for [SEP]
        elif tokens[i] in ['.', '!', '?', ',', ';', ':']:
            attention[i, i] = 0.2   # Lower self-attention for punctuation
        else:
            attention[i, i] = 0.3   # Lower default self-attention

        # Get spacy tokens aligned with current token
        spacy_indices = token_to_spacy[i]
        current_spacy = doc[spacy_indices[0]] if spacy_indices else None

        for j in range(n):
            if i == j:
                continue

            # Get spacy tokens for target
            target_spacy_indices = token_to_spacy[j]
            target_spacy = doc[target_spacy_indices[0]] if target_spacy_indices else None

            weight = 0.0

            # High attention for punctuation to nearby content words
            if tokens[i] in [',', '.', '!', '?', ';', ':']:
                if current_spacy and target_spacy:
                    if target_spacy.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                        distance = abs(i - j)
                        if distance <= 3:
                            weight += 0.3 / (1 + distance * 0.1)

            # Function words attend to nearby content words
            if current_spacy and target_spacy:
                if current_spacy.pos_ in ['ADP', 'DET', 'AUX'] and target_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    distance = abs(i - j)
                    if distance <= 5:
                        weight += 0.2 / (1 + distance * 0.2)

                # Verbs attend to their objects/subjects
                if current_spacy.pos_ == 'VERB':
                    if target_spacy.dep_ in ['dobj', 'nsubj', 'pobj']:
                        weight += 0.15

                # Prepositions attend to their objects
                if current_spacy.pos_ == 'ADP' and target_spacy.dep_ == 'pobj':
                    weight += 0.2

            # Semantic similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.3:
                weight += sim * 0.1

            # Adjacent token attention (especially for compound words)
            if abs(i - j) == 1:
                weight += 0.05
                # Extra boost for subword tokens
                if tokens[j].startswith('##') or tokens[i].startswith('##'):
                    weight += 0.1

            # Positional decay for distant tokens
            distance = abs(i - j)
            if distance > 0:
                decay = 1.0 / (1 + distance * 0.1)
                weight *= decay

            # Special handling for special tokens
            if tokens[j] in ['[CLS]', '[SEP]']:
                weight *= 0.5

            if tokens[i] == '[SEP]':
                # [SEP] often attends strongly to content words
                if current_spacy and target_spacy and target_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    weight += 0.3

            attention[i, j] = weight

    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)

    return "program_L9H7", attention



def program_L9H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Semantic similarity and punctuation attention head with enhanced self-attention and coreference patterns."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)

    for i in range(n):
        # Enhanced self-attention based on token type
        spacy_indices = token_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            # Higher self-attention for content words
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                attention[i, i] = 0.25
            elif spacy_token.pos_ == 'PRON':
                attention[i, i] = 0.20
            else:
                attention[i, i] = 0.15
        else:
            attention[i, i] = 0.15

        # Special handling for [SEP] token - high attention to punctuation
        if tokens[i] == '[SEP]':
            for j in range(n):
                if tokens[j] in ['.', ',', '!', '?', ';', ':']:
                    attention[i, j] = 0.25
                elif tokens[j] == '[CLS]':
                    attention[i, j] = 0.10
                else:
                    # Moderate attention to content words
                    spacy_indices = token_to_spacy[j]
                    if spacy_indices:
                        spacy_token = doc[spacy_indices[0]]
                        if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                            attention[i, j] = 0.06
                        else:
                            attention[i, j] = 0.03
            continue

        # Special handling for punctuation tokens
        if tokens[i] in ['.', ',', '!', '?']:
            attention[i, 0] = 0.12  # Attention to [CLS]
            # Attention to previous content words
            for j in range(max(0, i-5), i):
                spacy_indices = token_to_spacy[j]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        attention[i, j] = 0.08
            continue

        # Get spacy info for current token
        spacy_indices = token_to_spacy[i]
        current_spacy = None
        if spacy_indices:
            current_spacy = doc[spacy_indices[0]]

        # Attention to [CLS]
        attention[i, 0] = 0.05

        # Semantic similarity-based attention
        for j in range(n):
            if i == j:
                continue

            # High attention based on embedding similarity
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.5:
                attention[i, j] += 0.20
            elif sim > 0.3:
                attention[i, j] += 0.10
            elif sim > 0.1:
                attention[i, j] += 0.05

            # Enhanced pronoun coreference patterns
            spacy_j = token_to_spacy[j]
            if current_spacy and spacy_j:
                spacy_token_j = doc[spacy_j[0]]

                # Strong pronoun-to-pronoun attention (coreference)
                if current_spacy.pos_ == 'PRON' and spacy_token_j.pos_ == 'PRON':
                    attention[i, j] += 0.15

                # Pronoun to noun attention
                if current_spacy.pos_ == 'PRON' and spacy_token_j.pos_ in ['NOUN', 'PROPN']:
                    attention[i, j] += 0.08

                # Determiner/preposition to following noun
                if current_spacy.pos_ in ['DET', 'ADP'] and spacy_token_j.pos_ in ['NOUN', 'PROPN']:
                    if abs(i - j) <= 3:
                        attention[i, j] += 0.12

                # Verb to related objects/subjects
                if current_spacy.pos_ == 'VERB':
                    if spacy_token_j.dep_ in ['nsubj', 'dobj', 'pobj']:
                        attention[i, j] += 0.08

        # Boost attention to nearby tokens slightly
        for j in range(max(0, i-2), min(n, i+3)):
            if i != j:
                attention[i, j] += 0.02

    return "program_L9H8", make_row_stochastic(attention)



def program_L9H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """Coreference and possessive relationship head that connects pronouns to antecedents."""

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))

    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    token_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_token = align_spacy_to_tokens(sentence)

    for i in range(n):
        # Very high self-attention for [SEP] token
        if tokens[i].strip() == '[SEP]':
            attention[i, i] = 0.65
            # [SEP] also attends moderately to punctuation, [CLS], and pronouns
            for j in range(n):
                if j != i:
                    if tokens[j].strip() in ['.', ',', '[CLS]']:
                        attention[i, j] = 0.04
                    elif j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_idx = token_to_spacy[j][0]
                        if spacy_idx < len(doc) and doc[spacy_idx].pos_ == 'PRON':
                            attention[i, j] = 0.03

                    # NEW: [SEP] attends to content words throughout the sentence
                    if j < len(token_to_spacy) and token_to_spacy[j]:
                        spacy_idx = token_to_spacy[j][0]
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            # Attend to nouns, verbs, adjectives, and adverbs
                            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                                attention[i, j] += 0.02
                            # Slightly higher attention to proper nouns and named entities
                            elif spacy_token.pos_ == 'PROPN' or spacy_token.ent_type_ in ['PERSON', 'GPE', 'ORG']:
                                attention[i, j] += 0.025
            continue

        # Get spacy information for current token
        current_spacy_indices = token_to_spacy[i] if i < len(token_to_spacy) else []

        for j in range(n):
            target_spacy_indices = token_to_spacy[j] if j < len(token_to_spacy) else []

            # Self-attention baseline
            if i == j:
                attention[i, j] = 0.02

            # Attention to [CLS]
            if tokens[j].strip() == '[CLS]':
                attention[i, j] = 0.025

            # Sequential patterns - adjacent tokens
            if abs(i - j) == 1:
                attention[i, j] += 0.03

            # Embedding similarity boost
            sim = embedding_similarity(tokens, i, j)
            if sim > 0.3:
                attention[i, j] += 0.02 * sim

            # Linguistic relationship patterns
            if current_spacy_indices and target_spacy_indices:
                current_spacy = current_spacy_indices[0]
                target_spacy = target_spacy_indices[0]

                if current_spacy < len(doc) and target_spacy < len(doc):
                    current_token = doc[current_spacy]
                    target_token = doc[target_spacy]

                    # Possessive relationships: possessive determiners to pronouns
                    if current_token.pos_ == 'NOUN' and target_token.tag_ == 'PRP$':
                        attention[i, j] += 0.12

                    # Pronoun to pronoun coreference  
                    if (current_token.pos_ == 'PRON' and target_token.pos_ == 'PRON' and
                        current_token.text.lower() != target_token.text.lower()):
                        attention[i, j] += 0.08

                    # Nouns attending to preceding pronouns (antecedent relationships)
                    if (current_token.pos_ == 'NOUN' and target_token.pos_ == 'PRON' and 
                        j < i):
                        attention[i, j] += 0.06

            # Quote-related attention
            if tokens[i].strip() in ['"', "'"]:
                if tokens[j].strip() in ['"', "'", ',', '[CLS]']:
                    attention[i, j] += 0.08
                # Quotes attend to names/pronouns in the quote
                elif j > i and target_spacy_indices:
                    target_spacy = target_spacy_indices[0]
                    if (target_spacy < len(doc) and 
                        (doc[target_spacy].pos_ == 'PRON' or doc[target_spacy].ent_type_ == 'PERSON')):
                        attention[i, j] += 0.04

            # Punctuation patterns
            if tokens[i].strip() in [',', '.', '?']:
                if tokens[j].strip() in ['"', "'", '[CLS]']:
                    attention[i, j] += 0.03
                # Period attends to words in the sentence
                if tokens[i].strip() in ['.', '?'] and target_spacy_indices:
                    target_spacy = target_spacy_indices[0]
                    if target_spacy < len(doc) and doc[target_spacy].pos_ in ['VERB', 'ADJ']:
                        attention[i, j] += 0.02

    return "program_L9H9", make_row_stochastic(attention)



def is_content_word(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    if not token_to_spacy[token_idx]:
        return False
    spacy_indices = token_to_spacy[token_idx]
    for spacy_idx in spacy_indices:
        if spacy_idx < len(doc):
            spacy_token = doc[spacy_idx]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                return True
    return False



def is_punctuation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    return token.strip() in [',', '.', '!', '?', ';', ':', "'", '"']



def is_special_token(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    return token in ['[CLS]', '[SEP]']



def is_special_token(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    return tokens[i] in ['[CLS]', '[SEP]']



def is_subject_or_main_verb(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    spacy_indices = token_to_spacy[i]
    if not spacy_indices:
        return False
    for spacy_idx in spacy_indices:
        if spacy_idx < len(doc):
            token = doc[spacy_idx]
            if token.dep_ in ['nsubj', 'nsubjpass'] or (token.pos_ == 'VERB' and token.dep_ == 'ROOT'):
                return True
    return False



def is_early_content_word(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    if i >= min(6, n):  # Only consider first few tokens
        return False
    spacy_indices = token_to_spacy[i]
    if not spacy_indices:
        return False
    for spacy_idx in spacy_indices:
        if spacy_idx < len(doc):
            token = doc[spacy_idx]
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                return True
    return False



def is_proper_noun(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    spacy_indices = token_to_spacy[token_idx]
    if not spacy_indices:
        return False
    return any(doc[si].pos_ == "PROPN" for si in spacy_indices)



def is_punctuation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    token = tokens[token_idx].strip()
    return token in ".,!?;:"



def is_special_token(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    token = tokens[token_idx].strip()
    return token in ["[CLS]", "[SEP]"]



def get_spacy_features(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    spacy_indices = token_to_spacy[token_idx]
    if not spacy_indices:
        return None, None
    spacy_token = doc[spacy_indices[0]]
    return spacy_token.pos_, spacy_token.dep_



def is_punct(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    return token.strip() in '.,!?;:'



def is_special(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    return token in ['[CLS]', '[SEP]']



def get_pos(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    spacy_indices = token_to_spacy[token_idx]
    if spacy_indices:
        return doc[spacy_indices[0]].pos_
    return "X"



def is_verb(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    return get_pos(token_idx) in ["VERB", "AUX"]



def is_punct(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    return tokens[token_idx].strip() in [".", ",", "!", "?", ";", ":"]



def get_spacy_features(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    spacy_indices = token_to_spacy[i]
    if not spacy_indices:
        return None, None, None
    spacy_tok = doc[spacy_indices[0]]
    return spacy_tok.pos_, spacy_tok.dep_, spacy_tok.text.lower()


