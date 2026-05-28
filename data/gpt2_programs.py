"""
Auto-converted program file.
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
    """Map each token index to list of overlapping spacy token indices."""
    doc = _get_nlp()(sentence)
    # reconstruct char spans from token strings
    spans, pos = [], 0
    for t in tokens:
        # strip BPE/sentencepiece prefix chars
        clean = t.lstrip("\u0120").lstrip("##").lstrip()
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
    mask = np.tril(np.ones((n, n)))
    return matrix * mask

"""Helper functions available to generated attention-prediction code.

These are injected into the execution environment so generated code can import
them via `from helpers import *`.
"""

import numpy as np
import spacy

_nlp = None
_gpt2_tok = None










import numpy as np
from typing import Tuple, List

# Layer 0, Head 0
def decaying_first_token_bias_content_focus_L0H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong self-attention
        attention[i, i] = 0.3
        
        # Strong first-token attention (except for first token itself)
        if i > 0:
            attention[i, 0] = 0.4
        
        # Get spacy tokens aligned with current GPT2 token
        spacy_indices = alignment[i]
        current_pos = None
        if spacy_indices:
            current_pos = doc[spacy_indices[0]].pos_
        
        # Attend to previous tokens with decaying weight
        for j in range(i):
            if j == 0:
                continue  # Already handled first token
            
            # Base decay based on distance
            distance = i - j
            base_weight = 0.1 * (0.7 ** (distance - 1))
            
            # Boost for content words (verbs, nouns, adjectives)
            spacy_j = alignment[j]
            if spacy_j:
                j_pos = doc[spacy_j[0]].pos_
                if j_pos in ['VERB', 'NOUN', 'PROPN', 'ADJ']:
                    base_weight *= 2.0
                
                # Extra boost for verbs when current token is a noun/object
                if j_pos == 'VERB' and current_pos in ['NOUN', 'PROPN', 'PRON']:
                    base_weight *= 1.5
            
            # Boost for tokens that look like important words (longer, alphabetic)
            token_j = tokens[j].strip()
            if len(token_j) > 2 and token_j.isalpha():
                base_weight *= 1.2
            
            attention[i, j] = base_weight
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 0, Head 1
def decaying_content_focus_punctuation_coreference_L0H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize with zeros
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong self-attention (0.95-1.0)
        attention[i, i] = 0.99
        
        # Add weak cross-attention patterns
        for j in range(i):
            token_i = tokens[i].strip()
            token_j = tokens[j].strip()
            
            # Weak attention to previous tokens
            base_weight = 0.001
            
            # Slightly higher attention for punctuation and conjunctions
            if token_j in [',', '.', '!', '?', '"', 'and', 'or', 'but']:
                base_weight *= 5
            
            # Attention between related words (weak)
            if len(alignment[i]) > 0 and len(alignment[j]) > 0:
                spacy_i = alignment[i][0] if alignment[i] else -1
                spacy_j = alignment[j][0] if alignment[j] else -1
                
                if spacy_i < len(doc) and spacy_j < len(doc) and spacy_i >= 0 and spacy_j >= 0:
                    tok_i = doc[spacy_i]
                    tok_j = doc[spacy_j]
                    
                    # Weak syntactic relationships
                    if tok_j in tok_i.ancestors or tok_i in tok_j.ancestors:
                        base_weight *= 3
                    elif tok_i.head == tok_j or tok_j.head == tok_i:
                        base_weight *= 2
            
            # NEW: Enhanced attention for pronoun-antecedent and repeated word patterns
            if len(alignment[i]) > 0 and len(alignment[j]) > 0:
                spacy_i = alignment[i][0] if alignment[i] else -1
                spacy_j = alignment[j][0] if alignment[j] else -1
                
                if spacy_i < len(doc) and spacy_j < len(doc) and spacy_i >= 0 and spacy_j >= 0:
                    tok_i = doc[spacy_i]
                    tok_j = doc[spacy_j]
                    
                    # Strong attention for pronouns to potential antecedents
                    if tok_i.pos_ == "PRON" and tok_j.pos_ in ["PROPN", "NOUN", "PRON"]:
                        # Stronger for same gender/number if available, moderate otherwise
                        if tok_i.text.lower() in ["she", "her"] and tok_j.text.lower() in ["she", "her"]:
                            base_weight *= 50
                        elif tok_i.text.lower() in ["he", "him", "his"] and tok_j.text.lower() in ["he", "him", "his"]:
                            base_weight *= 50
                        elif tok_i.text.lower() == "it" and tok_j.pos_ == "NOUN":
                            base_weight *= 30
                        else:
                            base_weight *= 20
                    
                    # Strong attention between repeated words (same lemma)
                    if tok_i.lemma_ == tok_j.lemma_ and tok_i.lemma_ not in ["be", "have", "do", ".", ",", "?", "!"]:
                        base_weight *= 40
            
            # Distance decay
            distance = i - j
            decay_factor = np.exp(-distance * 0.3)
            
            attention[i, j] = base_weight * decay_factor
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_content_focus_pu", attention


# Layer 0, Head 2
def first_token_bias_content_focus_punctuation_L0H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Self-attention baseline
        attention[i, i] = 0.05
        
        # Strong attention to first token (except for first token itself)
        if i > 0:
            attention[i, 0] = 0.4
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly
        
        # Backward recency bias - attend to recent previous tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                if distance == 1:
                    attention[i, j] = 0.15  # Previous token
                elif distance == 2:
                    attention[i, j] = 0.08  # Two tokens back
                else:
                    attention[i, j] = 0.04  # Three tokens back
        
        # Add syntactic attention if we can align to spacy
        if i < len(gpt2_to_spacy) and gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to syntactic head
                if spacy_token.head != spacy_token:
                    head_text = spacy_token.head.text
                    # Find corresponding GPT2 token(s)
                    for j in range(i):
                        if tokens[j].strip() == head_text or tokens[j].strip().startswith(head_text):
                            attention[i, j] += 0.06
                
                # Attend to syntactic children
                for child in spacy_token.children:
                    child_text = child.text
                    for j in range(i):
                        if tokens[j].strip() == child_text or tokens[j].strip().startswith(child_text):
                            attention[i, j] += 0.04
        
        # Enhanced punctuation handling - attend strongly to key sentence elements
        if tokens[i] in ['.', '!', '?', ',', ',"', '."', '!"', '?"']:
            # Find main verbs and attend to them strongly
            for j in range(i):
                if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                    spacy_idx = gpt2_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        # Strong attention to main verbs (especially "said", auxiliary verbs)
                        if spacy_token.pos_ == 'VERB' and spacy_token.dep_ in ['ROOT', 'ccomp', 'xcomp']:
                            attention[i, j] += 0.12
                        elif tokens[j].strip().lower() in ['said', 'asked', 'replied', 'told', 'spoke']:
                            attention[i, j] += 0.15
                        # Moderate attention to other verbs and important nouns
                        elif spacy_token.pos_ == 'VERB':
                            attention[i, j] += 0.08
                        elif spacy_token.pos_ == 'NOUN' and spacy_token.dep_ in ['nsubj', 'dobj']:
                            attention[i, j] += 0.06
        
        # Special handling for punctuation - attend to key content words (original logic preserved)
        if tokens[i] in ['.', '!', '?', ',']:
            # Find verbs and nouns to attend to
            for j in range(i):
                if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                    spacy_idx = gpt2_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        if spacy_token.pos_ in ['VERB', 'NOUN']:
                            attention[i, j] += 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 0, Head 3
def first_token_bias_content_focus_punctuation_stochastic_L0H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong self-attention (base weight ~0.5-1.0)
        attention[i, i] = 1.0
        
        # Attend to previous token with high weight
        if i > 0:
            attention[i, i-1] = 0.3
        
        # Moderate attention to tokens 2-3 positions back
        if i > 1:
            attention[i, i-2] = 0.15
        if i > 2:
            attention[i, i-3] = 0.08
            
        # Light attention to first token (sentence start bias)
        if i > 0:
            attention[i, 0] = 0.05
            
        # Special handling for punctuation and conjunctions
        token_text = tokens[i].strip()
        
        if token_text in [',', '.', '."', '!', '?']:
            # Punctuation attends more to nearby content words
            for j in range(max(0, i-3), i):
                if tokens[j].strip().isalpha():
                    attention[i, j] += 0.1
                    
        elif token_text in ['and', 'or', 'but']:
            # Conjunctions attend to elements they connect
            for j in range(max(0, i-5), i):
                if j != i-1:  # Don't double-count previous token
                    attention[i, j] += 0.05
                    
        # Add small random variations to break ties
        for j in range(i):
            if j != i and j != i-1 and j != 0:
                attention[i, j] += np.random.uniform(0.001, 0.02)
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 0, Head 4
def first_token_bias_content_focus_L0H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Self-attention baseline
        attention[i, i] = 0.2
        
        # Strong first-token attention for most tokens
        if i > 0:
            attention[i, 0] = 0.4
        else:
            # First token has perfect self-attention
            attention[i, i] = 1.0
            continue
            
        # Get spacy tokens aligned to current GPT2 token
        spacy_indices = alignment[i]
        if not spacy_indices:
            continue
            
        current_spacy = doc[spacy_indices[0]]
        
        # Enhanced verb-to-verb attention for complex sentences
        if current_spacy.pos_ == "VERB":
            for j in range(max(0, i-8), i):  # Extended range for verb connections
                spacy_j = alignment[j]
                if not spacy_j:
                    continue
                target_spacy = doc[spacy_j[0]]
                
                # Strong verb-to-verb attention
                if target_spacy.pos_ == "VERB":
                    # Stronger attention for nearby verbs
                    distance = i - j
                    if distance <= 3:
                        attention[i, j] += 0.4
                    elif distance <= 6:
                        attention[i, j] += 0.3
                    else:
                        attention[i, j] += 0.2
                        
                # Auxiliary/modal to main verb connections
                if (target_spacy.pos_ in ["AUX", "VERB"] and 
                    target_spacy.dep_ in ["aux", "auxpass"] and
                    target_spacy.head == current_spacy):
                    attention[i, j] += 0.5
        
        # Local syntactic patterns
        for j in range(max(0, i-3), i):
            spacy_j = alignment[j]
            if not spacy_j:
                continue
            target_spacy = doc[spacy_j[0]]
            
            # Verb attending to subject
            if (current_spacy.pos_ == "VERB" and 
                target_spacy.dep_ in ["nsubj", "nsubjpass"]):
                attention[i, j] += 0.3
                
            # Adjective attending to noun it modifies
            if (current_spacy.pos_ == "ADJ" and 
                target_spacy.pos_ == "NOUN" and
                current_spacy.head == target_spacy):
                attention[i, j] += 0.4
                
            # Preposition attending to its object
            if (current_spacy.pos_ == "ADP" and
                target_spacy.dep_ == "pobj" and
                target_spacy.head == current_spacy):
                attention[i, j] += 0.3
                
            # Pronoun attending to nearby noun (potential antecedent)
            if (current_spacy.pos_ == "PRON" and 
                target_spacy.pos_ == "NOUN"):
                attention[i, j] += 0.2
                
            # Determiner attending to following noun
            if (current_spacy.pos_ == "DET" and 
                target_spacy.pos_ == "NOUN"):
                attention[i, j] += 0.2
        
        # Recency bias - attend to previous token
        if i > 0:
            attention[i, i-1] += 0.1
            
        # Special patterns for specific token types
        if current_spacy.pos_ == "PUNCT":
            # Punctuation attends to nearby content words
            for j in range(max(0, i-2), i):
                spacy_j = alignment[j]
                if spacy_j and doc[spacy_j[0]].pos_ in ["NOUN", "VERB", "ADJ"]:
                    attention[i, j] += 0.15
                    
        # Particles and auxiliary verbs attend to main verbs
        if current_spacy.pos_ in ["PART", "AUX"]:
            for j in range(max(0, i-3), i):
                spacy_j = alignment[j]
                if spacy_j and doc[spacy_j[0]].pos_ == "VERB":
                    attention[i, j] += 0.25
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 0, Head 5
def first_token_bias_content_focus_punctuation_L0H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Get spacy token info for each GPT2 token
    def get_spacy_info(gpt2_idx):
        spacy_indices = alignment[gpt2_idx]
        if spacy_indices:
            return doc[spacy_indices[0]]  # Use first aligned spacy token
        return None
    
    for i in range(n):
        token = tokens[i]
        spacy_token = get_spacy_info(i)
        
        # Strong self-attention for all tokens
        attention[i, i] = 0.8
        
        # First token attention (stronger for later tokens)
        if i > 0:
            first_token_weight = 0.1 + 0.05 * min(i / n, 1.0)
            attention[i, 0] = first_token_weight
        
        # Determiners ("the", "a") attend to early content words
        if token.strip().lower() in ['the', 'a', 'an']:
            for j in range(i):
                other_spacy = get_spacy_info(j)
                if other_spacy and other_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    # Stronger attention to earlier content words
                    weight = 0.15 * (1.0 - j / max(i, 1))
                    attention[i, j] += weight
        
        # End punctuation attends broadly to earlier tokens
        if token.strip() in ['.', '!', '?']:
            for j in range(i):
                other_spacy = get_spacy_info(j)
                # Attend more to content words and early tokens
                base_weight = 0.05
                if other_spacy:
                    if other_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        base_weight *= 2
                    if j < i // 3:  # Early tokens get more attention
                        base_weight *= 1.5
                attention[i, j] += base_weight
        
        # Content words attend to first token and some previous content
        if spacy_token and spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            if i > 0:
                attention[i, 0] += 0.05
            
            # Attend to some previous content words
            for j in range(max(0, i-3), i):
                other_spacy = get_spacy_info(j)
                if other_spacy and other_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    attention[i, j] += 0.03
        
        # General backward bias - attend slightly more to recent tokens
        for j in range(max(0, i-2), i):
            if j != i:  # Don't double-count self
                attention[i, j] += 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 0, Head 6
def first_token_bias_content_focus_L0H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong bias toward first token
        if i > 0:
            attention[i, 0] = 0.4
            if n > 1:
                attention[i, 1] = 0.2
        
        # Self-attention
        attention[i, i] = 0.2
        
        # Recency bias - attend to previous 1-2 tokens
        if i > 0:
            attention[i, i-1] += 0.15
        if i > 1:
            attention[i, i-2] += 0.1
            
        # Find important content words to attend to
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # If this is a content word, distribute some attention to other content words
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                for j in range(i):
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy_token = doc[j_spacy_indices[0]]
                        if j_spacy_token.pos_ in ['NOUN', 'VERB']:
                            attention[i, j] += 0.1
            
            # Punctuation attends strongly to recent content
            if spacy_token.pos_ == 'PUNCT':
                for j in range(max(0, i-5), i):
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy_token = doc[j_spacy_indices[0]]
                        if j_spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            attention[i, j] += 0.15
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 0, Head 7
def first_token_bias_content_focus_punctuation_L0H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention values
        base_self = 0.15
        base_prev = 0.25
        base_first = 0.4 if i < 3 else 0.1
        base_other = 0.02
        
        # Self attention
        attention[i, i] = base_self
        
        # First token attention (very strong for early positions)
        if i > 0:
            first_weight = base_first
            if i == 1:
                first_weight = 0.9  # Very strong for position 1
            elif i == 2:
                first_weight = 0.6  # Strong for position 2
            elif i <= 3:
                first_weight = 0.3
            attention[i, 0] = first_weight
        
        # Previous token attention
        if i > 0:
            attention[i, i-1] = base_prev
            
        # Special patterns for specific tokens
        token_text = tokens[i].strip()
        
        # Preposition "to" patterns
        if token_text == "to" and i > 0:
            attention[i, i-1] = 0.35  # Strong attention to previous
            attention[i, i] = 0.25   # Self attention
            if i > 1:
                attention[i, i-2] = 0.15  # Some attention to i-2
        
        # Conjunction "and" patterns  
        elif token_text == "and":
            # Find coordinated elements - look back for relevant content
            for j in range(max(0, i-5), i):
                if j < i-1:  # Not immediate predecessor
                    attention[i, j] = 0.08
        
        # Prepositions like "about", "for", "in"
        elif token_text in ["about", "for", "in"] and i > 0:
            attention[i, i-1] = 0.3  # Strong attention to previous
            
        # Articles and determiners
        elif token_text in ["the", "a", "an"]:
            if i > 0:
                attention[i, i-1] = 0.2
        
        # Fill remaining attention mass
        for j in range(i):
            if attention[i, j] == 0:
                # Distance-based falloff
                dist = i - j
                if dist == 1:
                    continue  # Already handled
                elif dist <= 3:
                    attention[i, j] = base_other * 2
                else:
                    attention[i, j] = base_other
    
    # Handle punctuation - tend to attend to recent content words
    for i in range(n):
        if tokens[i] in [".", "!", "?", ",", ":", ";"]:
            # Reduce first-token attention for punctuation
            attention[i, 0] = 0.05
            # Increase attention to recent tokens
            for j in range(max(0, i-3), i):
                attention[i, j] *= 1.5
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 0, Head 8
def decaying_first_token_bias_content_focus_punctuation_L0H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Helper function to check if token is punctuation
    def is_punctuation(token_str):
        return token_str.strip() in [',', '.', '!', '?', ';', ':', '"', "'"]
    
    # Helper function to check if token is article/determiner
    def is_article_or_det(token_str):
        return token_str.strip().lower() in ['a', 'an', 'the']
    
    # Helper function to check if token is conjunction
    def is_conjunction(token_str):
        return token_str.strip().lower() in ['and', 'or', 'but']
    
    # Helper function to check if token is a content word (noun, verb, adjective)
    def is_content_word(token_idx):
        if not alignment[token_idx]:
            return False
        spacy_idx = alignment[token_idx][0]
        if spacy_idx >= len(doc):
            return False
        pos = doc[spacy_idx].pos_
        return pos in ['NOUN', 'VERB', 'ADJ', 'PROPN']
    
    for i in range(n):
        token_str = tokens[i]
        
        # Base attention weights
        base_weights = np.zeros(i + 1)  # Can only attend to tokens 0 to i
        
        # Strong self-attention for first token
        if i == 0:
            base_weights[0] = 1.0
        else:
            # Self-attention boost
            base_weights[i] = 0.2
            
            # First token attention
            base_weights[0] = 0.15
            
            # Attention to punctuation
            for j in range(i):
                if is_punctuation(tokens[j]):
                    base_weights[j] += 0.25
            
            # Attention to articles/determiners
            for j in range(i):
                if is_article_or_det(tokens[j]):
                    base_weights[j] += 0.2
            
            # Attention to conjunctions
            for j in range(i):
                if is_conjunction(tokens[j]):
                    base_weights[j] += 0.15
            
            # Additional self-attention for punctuation
            if is_punctuation(token_str):
                base_weights[i] += 0.3
            
            # NEW: Boost attention to earlier content words
            if is_content_word(i):
                for j in range(i):
                    if is_content_word(j):
                        # Stronger boost for closer content words
                        distance_factor = 1.0 / (1.0 + 0.1 * (i - j))
                        base_weights[j] += 0.3 * distance_factor
            
            # Local attention bias (attend to recent tokens)
            for j in range(max(0, i-3), i):
                base_weights[j] += 0.05 * (1.0 - (i - j) / 4.0)
            
            # Positional decay for distant tokens
            for j in range(i):
                decay = np.exp(-0.1 * (i - j))
                base_weights[j] *= (0.5 + 0.5 * decay)
        
        # Normalize and assign
        if base_weights.sum() > 0:
            base_weights = base_weights / base_weights.sum()
        attention[i, :i+1] = base_weights
    
    # Apply causal mask and make row stochastic
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 0, Head 9
def first_token_bias_L0H9(sentence: str, tokenizer: PreTrainedTokenizerBase):
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Function words that tend to receive attention
    function_words = {'to', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'that', 'this'}
    
    # Dampening factor for longer sequences to prevent unrealistic spikes
    length_dampen = min(1.0, 8.0 / n) if n > 8 else 1.0
    
    for i in range(n):
        # Strong attention to first token (BOS-like behavior) - dampened for long sequences
        if i > 0:
            base_first_attention = 0.6 - 0.1 * i
            attention[i, 0] = base_first_attention * length_dampen
        
        # Self-attention (moderate)
        attention[i, i] = 0.2
        
        # Local context attention - especially to function words
        for j in range(max(0, i-3), i):
            if j == 0:
                continue  # Already handled first token
                
            token_text = tokens[j].strip().lower()
            distance = i - j
            
            # Base local attention that decreases with distance
            base_weight = 0.15 / distance
            
            # Boost for function words - dampened for long sequences
            if token_text in function_words:
                boost_factor = 1.0 + (1.0 * length_dampen)  # Reduces from 2.0x to 1.0x as sequence gets longer
                base_weight *= boost_factor
            
            # Boost for prepositions and conjunctions using spacy
            if gpt2_to_spacy[j]:
                spacy_idx = gpt2_to_spacy[j][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ in ['ADP', 'CCONJ', 'SCONJ']:  # prepositions, conjunctions
                        base_weight *= 1.5
            
            attention[i, j] = base_weight
        
        # Additional attention to recent tokens for longer sequences
        if i > 1:
            attention[i, i-1] = max(attention[i, i-1], 0.1)
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L0H9", attention


# Layer 0, Head 10
def decaying_first_token_bias_content_focus_L0H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Get spacy analysis for content word detection
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify content words (nouns, verbs, proper nouns)
    content_word_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.pos_ in ['NOUN', 'VERB', 'PROPN'] and not token.is_stop:
                    content_word_tokens.add(i)
    
    for i in range(n):
        # Strong attention to first token
        attention[i, 0] = 0.6 if i > 0 else 1.0
        
        # Self-attention
        if i > 0:
            attention[i, i] = 0.3
        
        # Attention to recent previous tokens with decay
        for j in range(1, i):
            distance = i - j
            if distance == 1:
                # Previous token gets higher attention
                attention[i, j] = 0.15
            elif distance <= 3:
                # Recent tokens get moderate attention
                attention[i, j] = 0.08 / distance
            else:
                # Distant tokens get lower attention
                attention[i, j] = 0.04 / distance
        
        # Boost attention to content words
        for j in content_word_tokens:
            if j < i:  # Only attend to previous tokens
                # Add extra attention to content words, scaling by distance
                distance = i - j
                if distance > 1:  # Don't double-boost immediate previous token
                    boost = 0.08 / max(1, distance * 0.5)
                    attention[i, j] += boost
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 0, Head 11
def decaying_first_token_bias_content_focus_punctuation_L0H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention distribution
        weights = np.zeros(i + 1)  # Can only attend to tokens <= i
        
        # 1. Strong first-token attention (decreases with position)
        first_token_weight = max(0.3, 0.8 - i * 0.05)
        weights[0] = first_token_weight
        
        # 2. Self-attention (moderate, constant)
        if i > 0:
            weights[i] = 0.15
        
        # 3. Previous token attention (if exists)
        if i > 0:
            weights[i-1] += 0.12
        
        # 4. Recency bias for other tokens
        for j in range(1, i):
            if j != i-1:  # Already handled previous token
                distance = i - j
                decay_weight = 0.08 * np.exp(-0.3 * distance)
                weights[j] += decay_weight
        
        # 5. Special linguistic patterns
        if len(alignment[i]) > 0:
            spacy_idx = alignment[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Prepositions attend more to their objects
                if spacy_token.pos_ == "ADP" and i > 2:
                    # Look for nearby nouns
                    for k in range(max(0, i-3), i):
                        if k < len(alignment) and len(alignment[k]) > 0:
                            k_spacy_idx = alignment[k][0]
                            if k_spacy_idx < len(doc) and doc[k_spacy_idx].pos_ in ["NOUN", "PRON"]:
                                weights[k] += 0.08
                
                # Verbs attend more to subjects (earlier in sentence)
                if spacy_token.pos_ == "VERB" and i > 1:
                    for k in range(1, min(i, 4)):
                        if k < len(alignment) and len(alignment[k]) > 0:
                            k_spacy_idx = alignment[k][0]
                            if k_spacy_idx < len(doc) and doc[k_spacy_idx].pos_ in ["NOUN", "PRON"]:
                                weights[k] += 0.05
        
        # 6. Handle punctuation specially
        if i < len(tokens) and tokens[i] in ['.', '!', '?', ',']:
            # Punctuation spreads attention more evenly
            for j in range(i):
                weights[j] += 0.03
        
        # Normalize and assign
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights[0] = 1.0  # Fallback to first token
            
        attention[i, :len(weights)] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 1, Head 0
def decaying_first_token_bias_L1H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base weights for different attention types
        prev_weight = 0.4
        first_weight = 0.3
        self_weight = 0.15
        syntactic_weight = 0.1
        fallback_weight = 0.05
        
        # Previous token attention (very strong pattern)
        if i > 0:
            attention_matrix[i, i-1] += prev_weight
        
        # First token attention (strong for many tokens)
        if i > 0:
            attention_matrix[i, 0] += first_weight
        
        # Self attention
        attention_matrix[i, i] += self_weight
        
        # NEW: Enhanced compound word/multi-part token attention
        if i > 0:
            current_token = tokens[i]
            prev_token = tokens[i-1]
            
            # Check for word continuation patterns (subword tokens)
            is_continuation = (
                # Token starts with lowercase letter after alphanumeric
                (len(prev_token) > 0 and prev_token[-1].isalnum() and 
                 len(current_token) > 0 and current_token[0].islower()) or
                # Token is a suffix-like pattern
                current_token.startswith(("'s", "'t", "'ll", "'m", "'re", "'ve")) or
                # Both tokens are short and could be parts of same word
                (len(current_token) <= 3 and len(prev_token) <= 3 and 
                 not current_token.startswith(' ') and not prev_token.endswith(' ')) or
                # Special characters that are likely continuations
                current_token in ["'", '"', '€', '™', '�'] or prev_token in ["'", '"', '€', '™', '�']
            )
            
            if is_continuation:
                # Boost attention to previous token significantly for compound parts
                attention_matrix[i, i-1] += 0.2
        
        # Syntactic attention - look for modifiers attending to heads
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # If this is a modifier, attend to its head
                if spacy_token.dep_ in ['amod', 'det', 'compound']:
                    head = spacy_token.head
                    head_idx = head.i
                    # Find corresponding GPT2 token
                    for j in range(i):
                        if gpt2_to_spacy[j] and head_idx in gpt2_to_spacy[j]:
                            attention_matrix[i, j] += syntactic_weight
                            break
                
                # If this token has modifiers, attend to them
                for child in spacy_token.children:
                    if child.dep_ in ['amod', 'det', 'compound']:
                        child_idx = child.i
                        for j in range(i):
                            if gpt2_to_spacy[j] and child_idx in gpt2_to_spacy[j]:
                                attention_matrix[i, j] += syntactic_weight
        
        # Add small fallback attention to nearby tokens
        for j in range(max(0, i-3), i):
            if attention_matrix[i, j] == 0:
                attention_matrix[i, j] += fallback_weight * (0.5 ** (i - j))
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 1, Head 1
def first_token_bias_content_focus_punctuation_L1H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention distribution
        row = np.zeros(n)
        
        # Strong first-token attention for most positions
        if i > 0:
            row[0] = 0.6
        else:
            row[0] = 1.0  # Self-attention for first token
        
        # Self-attention (moderate)
        if i > 0:
            row[i] = 0.15
        
        # Check if current token is punctuation
        is_punct = any(char in tokens[i] for char in ',.!?;:"')
        
        # Punctuation gets attention from nearby tokens
        for j in range(max(0, i-3), i):
            if any(char in tokens[j] for char in ',.!?;:"'):
                row[j] += 0.3
        
        # Local context attention (previous few tokens)
        for j in range(max(0, i-2), i):
            if j != 0:  # Don't double-count first token
                row[j] += 0.1
        
        # If we have spacy alignment, add syntactic dependencies
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to head of current token
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    for k in range(n):
                        if gpt2_to_spacy[k] and head_idx in gpt2_to_spacy[k]:
                            if k < i:
                                row[k] += 0.15
                            break
                
                # Attend to modifiers
                for child in spacy_token.children:
                    child_idx = child.i
                    for k in range(n):
                        if gpt2_to_spacy[k] and child_idx in gpt2_to_spacy[k]:
                            if k < i:
                                row[k] += 0.1
                            break
        
        # Special handling for specific token patterns
        token_text = tokens[i].strip().lower()
        
        # Articles and prepositions attend more to nearby content words
        if token_text in ['a', 'an', 'the', 'to', 'of', 'in', 'on', 'at']:
            for j in range(max(0, i-2), i):
                if not any(char in tokens[j] for char in ',.!?;:"') and tokens[j].strip().lower() not in ['a', 'an', 'the']:
                    row[j] += 0.1
        
        # Special case: In long sentences (>15 tokens), reduce first-token attention and boost punctuation attention
        if n > 15 and i > 0:
            # Reduce first token dominance in longer sentences
            row[0] *= 0.7
            
            # Look for important punctuation marks and boost attention to them
            for j in range(max(0, i-8), i):
                punct_token = tokens[j].strip()
                # Boost attention to clause-separating punctuation
                if punct_token in [',', ',"', ':"', '!"', '?"', '."']:
                    row[j] += 0.4
                # Also boost attention to quote marks that start dialogue
                elif '"' in punct_token and j > 0:
                    row[j] += 0.3
        
        attention[i] = row
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 1, Head 2
def first_token_bias_L1H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token
        attention_matrix[i, 0] = 0.8
        
        # Self-attention
        attention_matrix[i, i] = 0.15
        
        # Previous token attention (if not first token)
        if i > 0:
            attention_matrix[i, i-1] = 0.3
        
        # Look for commas and give them extra attention
        for j in range(i+1):
            if tokens[j] in [',', ';']:
                attention_matrix[i, j] += 0.4
        
        # Look for prepositions and conjunctions
        for j in range(i+1):
            if tokens[j].strip().lower() in ['to', 'and', 'in', 'of', 'for', 'with', 'at', 'on']:
                attention_matrix[i, j] += 0.2
        
        # Look for possessive markers
        for j in range(i+1):
            if "'s" in tokens[j]:
                attention_matrix[i, j] += 0.15
        
        # Slight attention to tokens 2-3 positions back
        if i >= 2:
            attention_matrix[i, i-2] = 0.08
        if i >= 3:
            attention_matrix[i, i-3] = 0.05
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L1H2", attention_matrix


# Layer 1, Head 3
def decaying_first_token_bias_L1H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (BOS-like behavior)
        if i > 0:
            attention[i, 0] = 0.6
        else:
            attention[i, 0] = 1.0
            
        # Self-attention
        if i > 0:
            attention[i, i] = 0.25
            
        # Attention to recent previous tokens with decay
        for j in range(1, i):
            distance = i - j
            if distance == 1:
                # Immediate previous token
                attention[i, j] = 0.15
            elif distance == 2:
                attention[i, j] = 0.08
            elif distance == 3:
                attention[i, j] = 0.05
            else:
                # Exponential decay for distant tokens
                attention[i, j] = 0.03 * (0.7 ** (distance - 3))
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 1, Head 4
def decaying_first_token_bias_content_focus_punctuation_L1H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Get spacy analysis for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify content words (nouns, verbs, adjectives)
    content_word_positions = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                content_word_positions.add(i)
    
    for i in range(n):
        # Strong first token attention for early positions (0-3)
        if i <= 3:
            first_token_weight = max(0.7 - i * 0.05, 0.1)
            attention[i, 0] = first_token_weight
        else:
            # Moderate first token attention for later positions
            attention[i, 0] = 0.05
        
        # Self-attention
        attention[i, i] = 0.1
        
        # Recent token attention with decay
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                base_weight = 0.08 / distance
                
                # Boost for content words
                if j in content_word_positions:
                    base_weight *= 1.5
                
                attention[i, j] += base_weight
        
        # Additional content word attraction for longer distances
        for j in range(i):
            if j in content_word_positions and i - j > 3:
                distance_penalty = 1.0 / (i - j)
                attention[i, j] += 0.03 * distance_penalty
        
        # Slight boost for punctuation attending to nearby content
        if i < len(tokens) and tokens[i] in [',', '.', '!', '?']:
            for j in range(max(0, i-2), i):
                if j in content_word_positions:
                    attention[i, j] *= 1.2
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 1, Head 5
def first_token_bias_punctuation_L1H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Get spacy parsing for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify punctuation tokens
    punct_tokens = set()
    for i, token in enumerate(tokens):
        if any(c in token for c in ',.!?"\''):
            punct_tokens.add(i)
    
    # Identify sentence boundary tokens (periods, newlines, etc.)
    sentence_boundary_tokens = set()
    for i, token in enumerate(tokens):
        if token in ['\n', '.\n', '!"', '."', '?"', ".'", '!', '.'] or '\n' in token:
            sentence_boundary_tokens.add(i)
    
    for i in range(n):
        # Base attention weights
        weights = np.zeros(n)
        
        # Strong first-token attention (except for first token itself)
        if i > 0:
            weights[0] = 0.6
        
        # Self-attention
        weights[i] = 0.3
        
        # Enhanced attention to sentence boundaries for tokens after them
        for j in sentence_boundary_tokens:
            if j < i:
                # Strong attention to recent sentence boundaries
                if i - j <= 3:
                    weights[j] += 0.4
                elif i - j <= 8:
                    weights[j] += 0.2
                else:
                    weights[j] += 0.05
        
        # Attention to punctuation within reasonable distance
        for j in punct_tokens:
            if j < i and i - j <= 5:  # Recent punctuation
                weights[j] += 0.2
            elif j < i:  # Distant punctuation
                weights[j] += 0.05
        
        # Local context (previous few tokens)
        for j in range(max(0, i-3), i):
            if j not in punct_tokens:
                weights[j] += 0.1 * (1 + j - max(0, i-3)) / 3
        
        # Special handling for quotes and sentence boundaries
        if i > 0 and tokens[i-1] in ['"', "'", '."', '!"', '?"']:
            weights[i-1] += 0.15
        
        # If token is part of a compound or phrase, attend to related tokens
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to head if this is a modifier
                if spacy_token.head != spacy_token and spacy_token.head.idx < sentence.find(tokens[i]):
                    for j in range(i):
                        if gpt2_to_spacy[j] and spacy_token.head.idx in [doc[k].idx for k in gpt2_to_spacy[j] if k < len(doc)]:
                            weights[j] += 0.1
                
                # Attend to modifiers if this is a head
                for child in spacy_token.children:
                    if child.idx < sentence.find(tokens[i]):
                        for j in range(i):
                            if gpt2_to_spacy[j] and child.idx in [doc[k].idx for k in gpt2_to_spacy[j] if k < len(doc)]:
                                weights[j] += 0.1
        
        # Special boost for contractions and articles
        if i > 0:
            if tokens[i-1] in ["'m", "'s", "'re", "'ve", "'ll", "'d"]:
                weights[i-1] += 0.2
            if tokens[i] in [" a", " an", " the"] and i > 0:
                weights[0] += 0.1  # Additional first-token boost for articles
        
        # Normalize and store
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights[i] = 1.0
            
        attention[i] = weights
    
    # Apply causal mask and make row stochastic
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 1, Head 6
def first_token_bias_punctuation_stochastic_L1H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([[]])
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (dominant pattern)
        if i > 0:
            attention[i, 0] = 0.7 + 0.2 * np.exp(-i * 0.3)  # Decay with distance but stay strong
        else:
            attention[i, 0] = 1.0  # Self-attention for first token
        
        # Self-attention (moderate)
        if i > 0:
            attention[i, i] = 0.1 + 0.05 * np.random.random()
        
        # Attention to previous token
        if i > 0:
            attention[i, i-1] = 0.08 + 0.04 * np.random.random()
        
        # Attention to nearby tokens (decreasing with distance)
        for j in range(max(0, i-3), i):
            if j != 0 and j != i and j != i-1:  # Skip first token, self, and previous (already handled)
                distance = i - j
                attention[i, j] = 0.03 * np.exp(-distance * 0.5) + 0.02 * np.random.random()
        
        # Special attention to punctuation tokens
        for j in range(i):
            if tokens[j] in ['!', '.', '?', ',', '!"', '."', '?"']:
                attention[i, j] += 0.03
        
        # Boost attention to tokens that might be syntactically important
        for j in range(i):
            token = tokens[j].strip()
            if token in ['and', 'but', 'because', 'with', 'who', 'that', 'which']:
                attention[i, j] += 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 1, Head 7
def first_token_bias_content_focus_punctuation_L1H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Detect complex contexts that need reduced first-token dominance
    has_quotes = any('"' in token for token in tokens)
    is_long = len(tokens) > 15
    complex_context = has_quotes or is_long
    
    for i in range(n):
        # Adjust first-token attention based on context complexity
        if complex_context:
            attention[i, 0] = 0.4  # Reduced from 0.7
        else:
            attention[i, 0] = 0.7  # Original weight
        
        # Self-attention
        attention[i, i] = 0.1
        
        # Get spacy tokens for current GPT2 token
        spacy_indices = alignment[i] if i < len(alignment) else []
        
        if spacy_indices:
            current_spacy = doc[spacy_indices[0]]
            
            # Attend to syntactic head
            if current_spacy.head != current_spacy:
                head_token = current_spacy.head
                for j in range(i):
                    j_spacy_indices = alignment[j] if j < len(alignment) else []
                    if j_spacy_indices and head_token.i in j_spacy_indices:
                        attention[i, j] += 0.15
            
            # Attend to syntactic children
            for child in current_spacy.children:
                for j in range(i):
                    j_spacy_indices = alignment[j] if j < len(alignment) else []
                    if j_spacy_indices and child.i in j_spacy_indices:
                        attention[i, j] += 0.1
        
        # Special attention to commas and punctuation
        for j in range(i):
            if tokens[j] in [',', '.', ';', ':', '!', '?']:
                attention[i, j] += 0.08
        
        # Enhanced local context bias for complex contexts
        for j in range(max(0, i-3), i):
            base_weight = 0.05 * (1.0 - (i - j) * 0.2)
            if complex_context:
                base_weight *= 1.5  # Boost local attention in complex contexts
            attention[i, j] += base_weight
        
        # Verb-subject relationships
        if spacy_indices:
            current_spacy = doc[spacy_indices[0]]
            if current_spacy.pos_ == 'VERB':
                # Look for subject
                for child in current_spacy.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        for j in range(i):
                            j_spacy_indices = alignment[j] if j < len(alignment) else []
                            if j_spacy_indices and child.i in j_spacy_indices:
                                attention[i, j] += 0.12
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 1, Head 8
def first_token_bias_punctuation_L1H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Get spacy parse for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify structural tokens (punctuation, conjunctions, prepositions)
    structural_positions = set()
    for i, token_str in enumerate(tokens):
        # Check if token is punctuation
        if any(c in token_str for c in '.,!?;:'):
            structural_positions.add(i)
        
        # Check spacy alignment for POS tags
        spacy_indices = gpt2_to_spacy[i]
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                # Add conjunctions, prepositions, and other function words
                if spacy_token.pos_ in ['CCONJ', 'SCONJ', 'ADP'] or spacy_token.text.lower() in ['and', 'because', 'that', 'to', 'the']:
                    structural_positions.add(i)
    
    for i in range(n):
        # Strong attention to first token
        attention[i, 0] = 0.7
        
        # Self attention
        attention[i, i] = 0.15
        
        # Attention to structural tokens within context
        for j in structural_positions:
            if j <= i and j != 0:  # Causal mask and not first token (already covered)
                distance = i - j
                # Stronger attention to closer structural tokens
                weight = max(0.05, 0.2 / (1 + distance * 0.5))
                attention[i, j] += weight
        
        # Local attention to nearby tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double count first token
                distance = i - j
                weight = 0.03 / distance
                attention[i, j] += weight
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 1, Head 9
def first_token_bias_L1H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Get token types
    articles = set()
    function_words = set()
    
    for i, token in enumerate(tokens):
        # Check if this token is an article
        if token.strip().lower() in ['a', 'an', 'the']:
            articles.add(i)
        
        # Check if this token is a function word via spacy alignment
        spacy_indices = gpt2_to_spacy[i]
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['ADP', 'CONJ', 'CCONJ', 'SCONJ', 'AUX', 'PART']:
                    function_words.add(i)
    
    for i in range(n):
        # Base attention distribution
        base_weights = np.zeros(n)
        
        # Strong first-token attention for early positions
        if i == 0:
            base_weights[0] = 1.0
        elif i <= 3:  # First few tokens attend strongly to first token
            base_weights[0] = 0.8 - 0.15 * (i - 1)
            
            # Add some attention to self and articles
            base_weights[i] = 0.1
            for art_idx in articles:
                if art_idx <= i:
                    base_weights[art_idx] += 0.1
        else:
            # Later tokens have more distributed attention
            
            # Moderate attention to first token
            base_weights[0] = 0.05
            
            # High attention to articles
            for art_idx in articles:
                if art_idx <= i:
                    base_weights[art_idx] += 0.15
            
            # Moderate attention to function words
            for func_idx in function_words:
                if func_idx <= i:
                    base_weights[func_idx] += 0.08
            
            # Self attention
            base_weights[i] += 0.1
            
            # Local context (previous few tokens)
            for j in range(max(0, i-3), i):
                base_weights[j] += 0.03
            
            # Small uniform attention to all previous tokens
            for j in range(i):
                base_weights[j] += 0.01
        
        # Normalize and assign to attention matrix
        if base_weights.sum() > 0:
            attention[i] = base_weights / base_weights.sum()
        else:
            attention[i, i] = 1.0
    
    # Apply causal mask and ensure row-stochastic
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L1H9", attention


# Layer 1, Head 10
def decaying_stochastic_L1H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Self-attention component - strongest for early tokens
        self_weight = 1.0 if i < 2 else 0.15
        attention_matrix[i, i] = self_weight
        
        # Previous token attention - very strong pattern
        if i > 0:
            prev_weight = 0.5 if i < 3 else 0.25
            attention_matrix[i, i-1] = prev_weight
        
        # Attention to earlier tokens with decay
        for j in range(i):
            if j == i:  # self (already handled)
                continue
            elif j == i - 1:  # previous token (already handled)
                continue
            else:
                # Distance-based decay with some randomness
                distance = i - j
                base_weight = 0.2 / (distance ** 0.7)
                
                # Boost for very early tokens
                if j < 2:
                    base_weight *= 1.5
                
                # Add some position-specific adjustments
                if i > 5:  # Later tokens
                    base_weight *= 0.8
                
                attention_matrix[i, j] = max(0.02, base_weight)
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_stochastic_L1H10", attention_matrix


# Layer 1, Head 11
def decaying_content_focus_punctuation_L1H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        token = tokens[i]
        
        # Base self-attention (very strong)
        attention[i, i] = 0.8
        
        # First token attention (strong for most positions)
        if i > 0:
            attention[i, 0] = 0.3
        
        # Positional decay for recent tokens
        for j in range(max(0, i-3), i):
            if j != i:
                decay = 0.15 * (0.7 ** (i - j - 1))
                attention[i, j] += decay
        
        # NEW: Long-distance semantic attention for repeated content words
        if i > 5:  # Only for tokens far enough from start
            token_clean = token.strip().lower()
            if len(token_clean) > 2 and token_clean.isalpha():  # Content words only
                for j in range(i-5):  # Look at tokens before recent window
                    prev_token_clean = tokens[j].strip().lower()
                    if token_clean == prev_token_clean:
                        # Strong attention for exact matches
                        attention[i, j] += 0.15
                    elif len(prev_token_clean) > 2 and prev_token_clean.isalpha():
                        # Weaker attention for semantic similarity (same stem)
                        if (token_clean.startswith(prev_token_clean[:3]) or 
                            prev_token_clean.startswith(token_clean[:3])):
                            attention[i, j] += 0.05
        
        # Special handling for punctuation
        if token in ['.', '!', '?']:
            # Period/end punctuation attends broadly
            attention[i, i] = 0.4  # Reduce self-attention
            attention[i, 0] = 0.2  # First token
            
            # Attend to recent content words and commas
            for j in range(i):
                if tokens[j] in [',', ';']:
                    attention[i, j] += 0.1
                elif j > 0 and not tokens[j].startswith(' '):  # Not space-prefixed suggests continuation
                    continue
                elif j > i - 6:  # Recent tokens
                    attention[i, j] += 0.05
                    
        elif token in [',', ';']:
            # Commas have complex patterns
            attention[i, i] = 0.5
            attention[i, 0] = 0.25
            
            # Attend to recent major tokens
            for j in range(max(0, i-5), i):
                if j > 0:
                    attention[i, j] += 0.08
                    
        elif token.startswith('"') or token.endswith('"') or token in ['!"', '."', '?"']:
            # Quote-related tokens
            attention[i, i] = 0.6
            attention[i, 0] = 0.2
            
            # Look for other quote/punctuation tokens
            for j in range(i):
                if tokens[j] in [',', '"'] or tokens[j].startswith('"'):
                    attention[i, j] += 0.08
                    
        else:
            # Regular tokens
            
            # Check if this is likely a function word or content word
            is_function_word = token.lower() in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            
            if is_function_word:
                # Function words have stronger positional bias
                attention[i, 0] = 0.4
                attention[i, i] = 0.6
            else:
                # Content words
                attention[i, i] = 0.7
                attention[i, 0] = 0.2
                
                # Look for related content in context
                for j in range(max(0, i-4), i):
                    if tokens[j] not in [',', '.', '!', '?'] and j > 0:
                        attention[i, j] += 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_content_focus_pu", attention


# Layer 2, Head 0
def decaying_first_token_bias_content_focus_punctuation_L2H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first token attention for most tokens
        if i > 0:
            attention[i, 0] = 0.7
        
        # Self attention
        attention[i, i] = 0.15
        
        # Previous token attention (especially for prepositions and function words)
        if i > 0:
            token_text = tokens[i].strip()
            prev_text = tokens[i-1].strip()
            
            # Strong previous token attention for certain patterns
            if any(prep in token_text.lower() for prep in ['to', 'on', 'with', 'and']):
                attention[i, i-1] = 0.3
            elif any(word in prev_text.lower() for word in ['to', 'on', 'with', 'and']):
                attention[i, i-1] = 0.35
            else:
                attention[i, i-1] = 0.1
        
        # Attention to "and" tokens - create coordination hubs
        for j in range(i):
            if 'and' in tokens[j].lower():
                attention[i, j] += 0.2
        
        # Linguistic patterns using spacy alignment
        if alignment[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_idx = alignment[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # If this is a preposition, attend more to its object
                if spacy_token.pos_ == 'ADP' and spacy_token.head.i != spacy_token.i:
                    # Find GPT2 tokens that align with the preposition's head
                    for k in range(min(i + 3, n)):  # Look ahead a bit
                        if k < len(alignment) and alignment[k]:
                            if spacy_token.head.i in alignment[k]:
                                attention[i, k] = 0.25
                
                # If this token's head is earlier, attend to it
                if spacy_token.head.i < spacy_token.i:
                    for j in range(i):
                        if j < len(alignment) and alignment[j]:
                            if spacy_token.head.i in alignment[j]:
                                attention[i, j] += 0.15
        
        # Positional decay for general context
        for j in range(max(0, i-3), i):
            if attention[i, j] < 0.1:  # Only if not already set high
                distance = i - j
                attention[i, j] += 0.08 / distance
        
        # Special handling for punctuation - attend to nearby content words
        if tokens[i] in ['.', '?', '!', ',']:
            for j in range(max(0, i-2), i):
                attention[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 2, Head 1
def first_token_bias_punctuation_L2H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Identify punctuation tokens
    punct_tokens = set()
    for i, token in enumerate(tokens):
        if any(c in token for c in '.,!?;:"()[]{}'):
            punct_tokens.add(i)
    
    for i in range(n):
        token = tokens[i]
        
        # Base attention weights
        weights = np.zeros(n)
        
        # 1. Strong first-token attention for most tokens
        if i == 0:
            weights[0] = 1.0  # First token attends to itself with weight 1.0
        else:
            weights[0] = 0.8  # Strong attention to first token
        
        # 2. Self-attention
        weights[i] = 0.3
        
        # 3. Strong punctuation attention
        for punct_idx in punct_tokens:
            if punct_idx <= i:
                weights[punct_idx] += 0.4
        
        # 4. Local positional bias - attend to recent tokens
        for j in range(max(0, i-3), i):
            weights[j] += 0.1 * (1.0 - (i-j) * 0.2)
        
        # 5. Special handling for specific token types
        if i > 0:
            # Tokens after punctuation get extra attention to that punctuation
            for punct_idx in punct_tokens:
                if punct_idx == i - 1:
                    weights[punct_idx] += 0.3
            
            # Second token gets very high attention to first token
            if i <= 3:
                weights[0] += 0.2
        
        # 6. Sentence structure attention using spacy
        if alignment[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_indices = alignment[i]
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    
                    # Attend to head of syntactic dependencies
                    if spacy_token.head != spacy_token:
                        head_token = spacy_token.head
                        # Find GPT2 tokens that align with this head
                        for j in range(i):
                            if alignment[j] and any(idx == head_token.i for idx in alignment[j]):
                                weights[j] += 0.15
                    
                    # Attend to clause/sentence beginnings
                    if spacy_token.dep_ in ['ROOT', 'ccomp', 'xcomp']:
                        for j in range(max(0, i-5), i):
                            weights[j] += 0.1
        
        # 7. Quote and dialogue attention
        if '"' in token or '"' in token or '"' in token:
            for j in range(i):
                if any(q in tokens[j] for q in ['"', '"', '"']):
                    weights[j] += 0.2
        
        # Apply causal mask (only attend to previous and current tokens)
        weights = weights[:i+1]
        weights = np.pad(weights, (0, n-i-1), mode='constant', constant_values=0)
        
        attention_matrix[i] = weights
    
    # Normalize and apply causal mask
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_punctuat", attention_matrix


# Layer 2, Head 2
def first_token_bias_content_focus_punctuation_stochastic_L2H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base weights
        base_first = 0.3  # Strong attention to first token
        base_prev = 0.4   # Strong attention to previous token
        base_self = 0.15  # Moderate self-attention
        
        # First token gets maximum self-attention
        if i == 0:
            attention[i, 0] = 1.0
            continue
            
        # Start with basic patterns
        attention[i, 0] = base_first  # Attend to first token
        attention[i, i-1] = base_prev  # Attend to previous token
        attention[i, i] = base_self    # Self-attention
        
        # Get spacy token info if available
        spacy_indices = gpt2_to_spacy[i]
        current_spacy_tok = None
        if spacy_indices:
            current_spacy_tok = doc[spacy_indices[0]]
        
        # Special patterns based on token content and position
        token = tokens[i]
        
        # Handle complex punctuation and encoding artifacts
        is_complex_punct = any(char in token for char in ['â', '€', '™', '�', '"', '"', ''', '''])
        if is_complex_punct or len(token) == 1 and not token.isalnum():
            # For complex punctuation, attend strongly to previous complex punctuation in sequence
            if i > 0:
                prev_token = tokens[i-1]
                prev_is_complex = any(char in prev_token for char in ['â', '€', '™', '�', '"', '"', ''', ''']) or (len(prev_token) == 1 and not prev_token.isalnum())
                if prev_is_complex:
                    attention[i, i-1] = 0.7  # Very strong attention to previous in sequence
                    attention[i, i] = 0.2
                    attention[i, 0] = 0.1
                    continue
        
        # Punctuation attends strongly to itself
        if token in ['.', '!', '?', ',']:
            attention[i, i] = 0.6
            if i > 0:
                attention[i, i-1] = 0.3
            attention[i, 0] = 0.1
        
        # Preposition-like patterns (attend to following tokens more)
        elif token.strip() in ['to', 'on', 'in', 'at', 'with', 'of', 'for']:
            # Look ahead for objects (within causal constraint)
            if i > 1:
                attention[i, i-2:i] = [0.2, 0.5]  # Attend to context
            attention[i, i] = 0.2
            attention[i, 0] = 0.1
            
        # Articles and determiners
        elif token.strip() in ['the', 'a', 'an']:
            if i > 0:
                attention[i, i-1] = 0.6  # Strong previous attention
            attention[i, i] = 0.1
            attention[i, 0] = 0.3
            
        # Conjunctions
        elif token.strip() in ['and', 'or', 'but']:
            if i > 0:
                attention[i, i-1] = 0.5
            attention[i, i] = 0.2
            attention[i, 0] = 0.3
            
        # Use spacy features if available
        if current_spacy_tok:
            # If this is a verb, attend to subject
            if current_spacy_tok.pos_ == 'VERB':
                # Look for subject in previous tokens
                for j in range(i):
                    spacy_j = gpt2_to_spacy[j]
                    if spacy_j:
                        spacy_tok_j = doc[spacy_j[0]]
                        if spacy_tok_j.dep_ in ['nsubj', 'nsubjpass']:
                            attention[i, j] += 0.2
                            
            # If this is an adjective, attend to the noun it modifies
            elif current_spacy_tok.pos_ == 'ADJ':
                for j in range(i):
                    spacy_j = gpt2_to_spacy[j]
                    if spacy_j:
                        spacy_tok_j = doc[spacy_j[0]]
                        if spacy_tok_j.pos_ in ['NOUN', 'PROPN'] and abs(j - i) <= 3:
                            attention[i, j] += 0.15
        
        # Add small random component to non-zero entries
        for j in range(i + 1):
            if attention[i, j] > 0:
                attention[i, j] += np.random.normal(0, 0.02)
                attention[i, j] = max(0.01, attention[i, j])  # Ensure positive
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 2, Head 3
def first_token_bias_content_focus_L2H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))
    
    # Get spacy parse and alignment
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_gpt2 = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base self-attention
        attention_matrix[i, i] = 0.1
        
        # Strong first token attention for most tokens
        if i > 0:
            attention_matrix[i, 0] = 0.3
        else:
            attention_matrix[i, 0] = 1.0  # First token attends to itself strongly
        
        # Get corresponding spacy tokens
        spacy_indices = gpt2_to_spacy[i]
        
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            token_text = tokens[i].strip()
            
            # Quote handling - tokens in quotes attend to quote boundaries
            if '"' in sentence:
                quote_positions = [j for j, t in enumerate(tokens) if '"' in t]
                for q_pos in quote_positions:
                    if abs(i - q_pos) <= 3:  # Near quote boundaries
                        attention_matrix[i, q_pos] = 0.2
            
            # Syntactic dependencies
            if spacy_tok.head != spacy_tok:  # Has a syntactic head
                head_gpt2_indices = []
                for head_idx in spacy_to_gpt2[spacy_tok.head.i]:
                    if head_idx <= i:  # Causal constraint
                        head_gpt2_indices.append(head_idx)
                
                for head_idx in head_gpt2_indices:
                    attention_matrix[i, head_idx] = 0.4
            
            # Children dependencies
            for child in spacy_tok.children:
                child_gpt2_indices = []
                for child_idx in spacy_to_gpt2[child.i]:
                    if child_idx <= i:  # Causal constraint
                        child_gpt2_indices.append(child_idx)
                
                for child_idx in child_gpt2_indices:
                    attention_matrix[i, child_idx] = 0.3
            
            # Special patterns for specific POS/dependency types
            if spacy_tok.pos_ in ['DET', 'ADP']:  # Determiners and prepositions
                # Look for nearby nouns
                for j in range(max(0, i-3), min(i+1, n)):
                    if j < len(tokens):
                        j_spacy = gpt2_to_spacy[j]
                        if j_spacy and doc[j_spacy[0]].pos_ == 'NOUN':
                            attention_matrix[i, j] = 0.5
            
            # Previous token attention for certain patterns
            if i > 0:
                prev_token = tokens[i-1].strip()
                if spacy_tok.pos_ in ['NOUN', 'ADJ', 'VERB']:
                    attention_matrix[i, i-1] = 0.2
        
        # Local attention bias - attend to nearby tokens
        for j in range(max(0, i-2), i):
            attention_matrix[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_content_", attention_matrix


# Layer 2, Head 4
def first_token_bias_content_focus_punctuation_L2H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Base attention weights
        for j in range(i + 1):  # Only attend to previous and current tokens (causal)
            if i == j == 0:
                # First token attends to itself with maximum weight
                attention[i, j] = 1.0
            elif j == 0:
                # Strong attention to first token for early positions
                if i <= 3:
                    attention[i, j] = 0.9 - (i * 0.15)
                else:
                    attention[i, j] = 0.3 - (i * 0.02)
                attention[i, j] = max(attention[i, j], 0.1)
            elif i == j:
                # Self-attention
                if tokens[i] in ['.', '!', '?', ',']:
                    attention[i, j] = 0.4  # High self-attention for punctuation
                else:
                    attention[i, j] = 0.15
            elif j == i - 1:
                # Previous token attention
                attention[i, j] = 0.4
            elif j == i - 2 and i >= 2:
                # Two tokens back gets some attention
                attention[i, j] = 0.2
            else:
                # Distant tokens get decreasing attention
                distance = i - j
                attention[i, j] = max(0.05 / distance, 0.01)
    
    # Special patterns for specific token types
    for i in range(n):
        token = tokens[i]
        
        # Contractions and compound words
        if "'" in token or "-" in token:
            # Attend more to the token that started the compound
            for j in range(i):
                if j == i - 1:
                    attention[i, j] *= 2.0
                elif j == i - 2:
                    attention[i, j] *= 1.5
        
        # Articles and determiners attend to nearby nouns
        if token.lower().strip() in ['the', 'a', 'an']:
            if i + 1 < n:  # Look ahead isn't allowed, so boost backward connections
                for j in range(max(0, i - 3), i):
                    attention[i, j] *= 1.2
        
        # Prepositions
        if token.lower().strip() in ['on', 'in', 'at', 'with', 'by', 'of']:
            # Attend to previous content words
            for j in range(max(0, i - 4), i):
                if tokens[j].strip().lower() not in ['the', 'a', 'an', 'and', 'or']:
                    attention[i, j] *= 1.3
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 2, Head 5
def decaying_first_token_bias_content_focus_punctuation_L2H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Helper function to check if a token is punctuation
    def is_punctuation(token_str):
        return token_str.strip() in '.,;:!?"()[]{}' or any(c in token_str for c in '.,;:!?"()[]{}')
    
    for i in range(n):
        token = tokens[i]
        
        # Base attention distribution
        for j in range(i + 1):  # Only attend to previous and current tokens
            if i == 0:
                # First token: full self-attention
                if j == 0:
                    attention[i, j] = 1.0
            else:
                # Strong first token bias for second token
                if i == 1 and j == 0:
                    attention[i, j] = 0.95
                elif i == 1 and j == 1:
                    attention[i, j] = 0.05
                else:
                    # General attention pattern
                    if j == 0:
                        # First token attention (moderate to strong)
                        if i <= 3:
                            attention[i, j] = 0.6 - 0.1 * (i - 1)
                        else:
                            attention[i, j] = 0.2
                    
                    elif j == i:
                        # Self attention
                        if is_punctuation(token):
                            attention[i, j] = 0.4  # Punctuation has higher self-attention
                        else:
                            attention[i, j] = 0.1
                    
                    elif j == i - 1:
                        # Previous token attention
                        prev_token = tokens[j]
                        if is_punctuation(prev_token):
                            attention[i, j] = 0.5  # High attention to previous punctuation
                        else:
                            attention[i, j] = 0.2
                    
                    elif j == i - 2:
                        # Two tokens back
                        attention[i, j] = 0.1
                    
                    else:
                        # Earlier tokens - look for punctuation
                        if is_punctuation(tokens[j]):
                            attention[i, j] = 0.15
                        else:
                            attention[i, j] = 0.05
        
        # Special handling for punctuation patterns
        if is_punctuation(token):
            # Punctuation tends to attend more to recent content words
            for j in range(max(0, i - 3), i):
                if not is_punctuation(tokens[j]):
                    attention[i, j] *= 1.5
        
        # Boost attention to commas specifically, but with distance decay
        for j in range(i):
            if ',' in tokens[j]:
                distance = i - j
                if distance <= 2:
                    attention[i, j] *= 2.0  # Strong boost for nearby commas
                elif distance <= 5:
                    attention[i, j] *= 1.5  # Moderate boost for medium distance
                else:
                    attention[i, j] *= 1.2  # Weak boost for distant commas
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 2, Head 6
def decaying_first_token_bias_punctuation_L2H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))
    
    # Identify important boundary tokens
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
        
        # Base attention distribution
        base_attention = np.zeros(i + 1)  # Can only attend to tokens up to position i
        
        # Strong first token attention for most tokens
        if i > 0:
            base_attention[first_token_idx] = 0.8
        
        # Self attention
        base_attention[i] = 0.15
        
        # Attention to punctuation and boundaries
        for j in range(i):
            if j in punct_tokens:
                base_attention[j] += 0.3
            elif j in newline_tokens:
                base_attention[j] += 0.25
        
        # Special cases based on token type
        if token in punct_tokens:
            # Punctuation attends strongly to itself and boundaries
            base_attention[i] = 0.4
            for j in range(i):
                if j in newline_tokens or j in punct_tokens:
                    base_attention[j] += 0.2
        
        elif token in newline_tokens:
            # Newlines attend to previous punctuation and themselves
            base_attention[i] = 0.4
            for j in range(i):
                if j in punct_tokens:
                    base_attention[j] += 0.3
        
        elif i == 0:
            # First token attends only to itself
            base_attention[i] = 1.0
        
        else:
            # Regular tokens: strong first token attention + some local context
            base_attention[first_token_idx] = 0.7
            
            # Add some attention to recent tokens
            for j in range(max(0, i - 3), i):
                if j != first_token_idx:
                    base_attention[j] += 0.05
            
            # Extra attention if following punctuation or newlines
            if i > 0 and (tokens[i-1] in ['."', '.', ',', '\n']):
                base_attention[i-1] += 0.2
        
        # Add small positional decay for distant tokens
        for j in range(i + 1):
            if j != first_token_idx and j != i:
                distance_penalty = max(0, 1.0 - 0.1 * (i - j))
                base_attention[j] *= distance_penalty
        
        # Ensure non-negative and fill the attention matrix row
        base_attention = np.maximum(base_attention, 0.01)
        attention_matrix[i, :i + 1] = base_attention
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 2, Head 7
def decaying_first_token_bias_content_focus_L2H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention distribution
        weights = np.zeros(n)
        
        # Strong self-attention
        weights[i] = 0.3
        
        # Strong first-token attention (except for first token itself)
        if i > 0:
            weights[0] = 0.4
        
        # Recency bias - attend to recent previous tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                decay = 0.8 ** (i - j)
                weights[j] += 0.2 * decay
        
        # Syntactic attention using spacy
        if alignment[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_indices = alignment[i]
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    
                    # Find syntactically related tokens
                    related_spacy = []
                    
                    # Add head if it exists
                    if spacy_token.head != spacy_token:
                        related_spacy.append(spacy_token.head.i)
                    
                    # Add children
                    for child in spacy_token.children:
                        related_spacy.append(child.i)
                    
                    # Add subject for verbs
                    if spacy_token.pos_ == "VERB":
                        for child in spacy_token.children:
                            if "subj" in child.dep_:
                                related_spacy.append(child.i)
                    
                    # Convert spacy indices back to GPT2 indices
                    for related_idx in related_spacy:
                        if related_idx < len(doc):
                            # Find GPT2 tokens that align with this spacy token
                            for k in range(i):  # Only look at previous tokens (causal)
                                if alignment[k] and related_idx in alignment[k]:
                                    weights[k] += 0.15
        
        # Special handling for final token
        if i == n - 1:
            # Final token attends more broadly to content words
            weights = np.zeros(n)
            weights[i] = 0.2  # Reduced self-attention
            
            # Distribute attention across all previous tokens with bias toward content
            for j in range(i):
                base_weight = 0.8 / i if i > 0 else 0
                
                # Boost content words
                if alignment[j]:
                    for spacy_idx in alignment[j]:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ["NOUN", "VERB", "ADJ"]:
                                base_weight *= 1.5
                
                weights[j] = base_weight
        
        # Normalize and apply to attention matrix
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights[i] = 1.0  # Fallback to self-attention
            
        attention[i] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 2, Head 8
def decaying_first_token_bias_content_focus_L2H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # First token always attends to itself
        if i == 0:
            attention[i, i] = 1.0
            continue
            
        # Base attention with distance decay
        for j in range(i + 1):
            distance = i - j
            if distance == 0:  # Self-attention
                attention[i, j] = 0.15
            elif distance == 1:  # Previous token
                attention[i, j] = 0.6
            elif distance <= 3:  # Recent tokens
                attention[i, j] = 0.3 / (distance)
            else:  # Distant tokens
                attention[i, j] = 0.1 / (distance + 1)
        
        # Strong attention to first token for most tokens
        if i > 2:
            attention[i, 0] += 0.2
            
        # Special case: Boost first-token attention for common sentence starters
        sentence_lower = sentence.lower()
        if (sentence_lower.startswith('once upon') or 
            sentence_lower.startswith('after ') or
            sentence_lower.startswith('before ') or
            sentence_lower.startswith('when ') or
            sentence_lower.startswith('while ')):
            if i <= 4:  # First few tokens get very strong first-token attention
                if i == 1:
                    attention[i, 0] = max(attention[i, 0], 0.8)
                elif i == 2:
                    attention[i, 0] = max(attention[i, 0], 0.6)
                elif i == 3:
                    attention[i, 0] = max(attention[i, 0], 0.5)
                elif i == 4:
                    attention[i, 0] = max(attention[i, 0], 0.4)
            
        # Boost attention based on linguistic relationships
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Find syntactic head
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    # Find corresponding GPT2 tokens for the head
                    for k in range(i):
                        if gpt2_to_spacy[k] and head_idx in gpt2_to_spacy[k]:
                            attention[i, k] += 0.4
                            break
                
                # Special patterns for function words
                if spacy_token.pos_ in ['ADP', 'DET']:  # prepositions, determiners
                    # Attend to next content word
                    for k in range(max(0, i-3), i):
                        if gpt2_to_spacy[k]:
                            spacy_k = gpt2_to_spacy[k][0]
                            if spacy_k < len(doc) and doc[spacy_k].pos_ in ['NOUN', 'VERB']:
                                attention[i, k] += 0.3
                                break
                                
                # Verbs attend to subjects/objects
                if spacy_token.pos_ == 'VERB':
                    for child in spacy_token.children:
                        if child.dep_ in ['nsubj', 'dobj', 'pobj']:
                            child_idx = child.i
                            for k in range(i):
                                if gpt2_to_spacy[k] and child_idx in gpt2_to_spacy[k]:
                                    attention[i, k] += 0.2
                                    break
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 2, Head 9
def first_token_bias_content_focus_stochastic_L2H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # First token always has maximum self-attention
        if i == 0:
            attention[i, 0] = 1.0
            continue
            
        # Base attention distribution
        weights = np.zeros(n)
        
        # Strong first-token bias for most tokens
        weights[0] = 0.3
        
        # Self-attention
        weights[i] = 0.15
        
        # Previous token attention (local dependency)
        if i > 0:
            weights[i-1] = 0.25
            
        # Two tokens back (for longer dependencies)
        if i > 1:
            weights[i-2] = 0.1
            
        # Add syntactic attention if we can align to spacy
        if alignment[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_idx = alignment[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to syntactic head
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    # Find GPT2 tokens that align with this spacy head
                    for j in range(min(i+1, n)):  # Only look at available positions
                        if alignment[j] and head_idx in alignment[j]:
                            weights[j] += 0.2
                            break
                
                # If this is a modifier, attend to what it modifies
                if spacy_token.dep_ in ['amod', 'advmod', 'det']:
                    head_idx = spacy_token.head.i
                    for j in range(min(i+1, n)):
                        if alignment[j] and head_idx in alignment[j]:
                            weights[j] += 0.15
                            break
                
                # If this token has modifiers, they should attend to it
                # (This is handled when processing the modifier tokens)
        
        # Special patterns based on token content
        current_token = tokens[i].strip().lower()
        
        # Punctuation tends to attend to nearby content words
        if current_token in ['.', ',', '!', '?', ';', ':']:
            # Find the most recent content-heavy token
            for j in range(i-1, max(-1, i-4), -1):
                if j >= 0:
                    weights[j] += 0.1
        
        # Function words often attend to content words
        if current_token in ['the', 'a', 'an', 'to', 'of', 'for', 'with', 'by']:
            # Look for nearby nouns or verbs
            for j in range(max(0, i-3), i):
                other_token = tokens[j].strip().lower()
                if len(other_token) > 2 and other_token not in ['the', 'a', 'an', 'to', 'of', 'for', 'with', 'by', 'and', 'or', 'but']:
                    weights[j] += 0.1
        
        # Apply causal mask (zero out future positions)
        weights[i+1:] = 0
        
        # Normalize and add small random noise to break ties
        if weights.sum() > 0:
            weights = weights / weights.sum()
            # Add tiny amount of noise to remaining positions
            noise = np.random.random(n) * 0.001
            noise[i+1:] = 0  # Respect causal mask
            weights = weights + noise
            weights = weights / weights.sum()
        else:
            # Fallback: uniform over available positions
            weights[:i+1] = 1.0 / (i+1)
            
        attention[i] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 2, Head 10
def first_token_bias_content_focus_L2H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # First token always attends to itself with weight 1.0
        if i == 0:
            attention[i, 0] = 1.0
            continue
            
        # Base weights
        weights = np.zeros(i + 1)  # Can only attend to positions 0 to i
        
        # Strong first-token attention for early positions
        if i <= 3:
            first_token_weight = max(0.6, 0.9 - 0.1 * i)
            weights[0] = first_token_weight
        else:
            # Moderate first-token attention for later positions
            weights[0] = 0.1
        
        # Self-attention
        self_weight = 0.15 if i <= 3 else 0.1
        weights[i] = self_weight
        
        # Get spacy tokens for current position
        current_spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
        
        # Syntactic attention patterns
        if current_spacy_indices:
            current_spacy_tok = doc[current_spacy_indices[0]]
            
            # Look for syntactic relationships
            for j in range(max(0, i-5), i):  # Look back up to 5 positions
                target_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                if target_spacy_indices:
                    target_spacy_tok = doc[target_spacy_indices[0]]
                    
                    # Head-dependent relationships
                    if target_spacy_tok.head == current_spacy_tok or current_spacy_tok.head == target_spacy_tok:
                        weights[j] += 0.2
                    
                    # Specific syntactic patterns
                    if current_spacy_tok.dep_ in ["dobj", "pobj"] and target_spacy_tok == current_spacy_tok.head:
                        weights[j] += 0.15
                    
                    if current_spacy_tok.pos_ == "NOUN" and target_spacy_tok.pos_ == "ADJ" and abs(i - j) <= 2:
                        weights[j] += 0.1
                        
                    if current_spacy_tok.pos_ == "VERB" and target_spacy_tok.pos_ in ["NOUN", "PRON"] and abs(i - j) <= 3:
                        weights[j] += 0.1
        
        # Local recency bias
        for j in range(max(0, i-3), i):
            distance_weight = 0.05 * (1.0 / (i - j + 1))
            weights[j] += distance_weight
        
        # Adjacent token attention
        if i > 0:
            weights[i-1] += 0.08
        
        # Ensure minimum weights
        for j in range(i + 1):
            weights[j] = max(weights[j], 0.01)
        
        attention[i, :i+1] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 2, Head 11
def first_token_bias_content_focus_L2H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (observed in all examples)
        attention[i, 0] = 0.8
        
        # Self-attention (moderate weight)
        attention[i, i] = 0.3
        
        # Recent token bias (attend to previous token)
        if i > 0:
            attention[i, i-1] = 0.2
        
        # Syntactic attention based on spacy parsing
        spacy_indices = gpt2_to_spacy[i]
        
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to syntactic head
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    # Find GPT2 tokens that align with this spacy head
                    for j in range(min(i+1, n)):  # causal constraint
                        if head_idx in gpt2_to_spacy[j]:
                            attention[i, j] += 0.4
                
                # Attend to syntactic children
                for child in spacy_token.children:
                    child_idx = child.i
                    for j in range(min(i+1, n)):  # causal constraint
                        if child_idx in gpt2_to_spacy[j]:
                            attention[i, j] += 0.3
                
                # Special patterns for specific POS tags and dependencies
                if spacy_token.pos_ == "VERB":
                    # Verbs attend to their subjects and objects
                    for child in spacy_token.children:
                        if child.dep_ in ["nsubj", "dobj", "iobj"]:
                            child_idx = child.i
                            for j in range(min(i+1, n)):
                                if child_idx in gpt2_to_spacy[j]:
                                    attention[i, j] += 0.3
                
                if spacy_token.pos_ in ["NOUN", "PROPN"]:
                    # Nouns attend to their modifiers
                    for child in spacy_token.children:
                        if child.dep_ in ["amod", "compound"]:
                            child_idx = child.i
                            for j in range(min(i+1, n)):
                                if child_idx in gpt2_to_spacy[j]:
                                    attention[i, j] += 0.2
        
        # Additional positional patterns
        # Attend to tokens that are 2-3 positions back with decreasing weight
        if i >= 2:
            attention[i, i-2] = 0.1
        if i >= 3:
            attention[i, i-3] = 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 0
def first_token_bias_stochastic_L3H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # First token: perfect self-attention
    attention[0, 0] = 1.0
    
    # For all other tokens
    for i in range(1, n):
        # Strong attention to first token (base: 0.9, with small random variation)
        first_token_attention = 0.9 + np.random.uniform(-0.05, 0.05)
        first_token_attention = max(0.85, min(0.99, first_token_attention))
        
        attention[i, 0] = first_token_attention
        
        # Self-attention (moderate, around 0.05-0.1)
        self_attention = np.random.uniform(0.02, 0.1)
        attention[i, i] = self_attention
        
        # Check for repeated tokens - add special attention pattern
        current_token = tokens[i].lower().strip()
        repeated_token_bonus = 0.0
        
        # Look for exact matches of the current token in earlier positions
        if len(current_token) > 2:  # Only for meaningful tokens
            for j in range(i):
                prev_token = tokens[j].lower().strip()
                if prev_token == current_token and j != 0:  # Don't double-count first token
                    # Add bonus attention to repeated tokens
                    bonus = np.random.uniform(0.15, 0.35)
                    attention[i, j] += bonus
                    repeated_token_bonus += bonus
        
        # Distribute remaining probability among other available positions
        remaining_prob = 1.0 - first_token_attention - self_attention - repeated_token_bonus
        
        if remaining_prob > 0:
            # Available positions (excluding first token, self, and already boosted repeated tokens)
            available_positions = []
            for j in range(i):
                if j != 0 and attention[i, j] == 0:  # Skip first token and already assigned positions
                    available_positions.append(j)
            
            if available_positions:
                # Small random weights for other positions
                weights = np.random.exponential(0.01, len(available_positions))
                weights = weights * (remaining_prob / weights.sum()) if weights.sum() > 0 else weights
                
                for j, pos in enumerate(available_positions):
                    attention[i, pos] = weights[j]
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_stochast", attention


# Layer 3, Head 1
def first_token_bias_content_focus_punctuation_stochastic_L3H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify structural tokens (punctuation, conjunctions)
    structural_tokens = set()
    for i, token in enumerate(tokens):
        if token.strip() in {',', '.', ':', ';', '!', '?', 'and', 'but', 'or', 'because', 'when', 'if'}:
            structural_tokens.add(i)
        # Also check if any aligned spacy token is a conjunction
        for spacy_idx in gpt2_to_spacy[i]:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['CCONJ', 'SCONJ']:
                structural_tokens.add(i)
    
    for i in range(n):
        # Base attention weights
        base_weights = np.zeros(i + 1)  # Only attend to previous tokens + self
        
        # 1. Strong first-token attention (very high weight)
        if i > 0:
            base_weights[0] = 0.7
        
        # 2. Self-attention (moderate weight)
        base_weights[i] = 0.15
        
        # 3. Attention to immediately preceding token
        if i > 0:
            base_weights[i-1] = 0.08
        
        # 4. Attention to structural tokens (commas, conjunctions, etc.)
        for j in structural_tokens:
            if j <= i:
                base_weights[j] += 0.12
        
        # 5. Local syntactic relationships
        # Articles attend to nearby nouns, modifiers to heads, etc.
        current_token = tokens[i].strip().lower()
        
        # If current token is a possessive or determiner, attend to nearby content words
        if current_token in ["'s", "the", "a", "an", "his", "her", "their", "my", "your"]:
            for j in range(max(0, i-3), i):
                other_token = tokens[j].strip().lower()
                if len(other_token) > 2 and other_token.isalpha():
                    base_weights[j] += 0.06
        
        # Prepositions attend to their objects
        if i > 0 and current_token in ["to", "of", "in", "at", "on", "for", "with"]:
            # Look for content words after prepositions in the attention pattern
            for j in range(max(0, i-2), i):
                if j in structural_tokens or tokens[j].strip().lower() in ["the", "a", "an"]:
                    base_weights[j] += 0.05
        
        # 6. Special patterns for sentence endings
        if tokens[i].strip() == '.':
            # Sentence endings attend strongly to themselves and structural elements
            base_weights[i] = 0.25
            for j in structural_tokens:
                if j <= i:
                    base_weights[j] += 0.08
        
        # 7. NEW: Strong determiner-to-noun attention
        # Check if current token is a determiner/article that should attend strongly to a following noun
        if current_token in ["the", "a", "an", "your", "his", "her", "their", "my", "this", "that"]:
            # Look ahead for content words (nouns) within a reasonable window
            for j in range(i + 1, min(n, i + 4)):
                future_token = tokens[j].strip().lower()
                # Check if it's a content word (noun-like)
                if len(future_token) > 2 and future_token.isalpha() and future_token not in ["and", "the", "but", "for", "with", "from"]:
                    # Add strong attention to this future noun (but we can only attend backwards)
                    # So instead, when we process that future token, it will attend back to this determiner
                    pass
            
            # Also boost attention to nearby content words that came before
            for j in range(max(0, i-2), i):
                other_token = tokens[j].strip().lower()
                if len(other_token) > 3 and other_token.isalpha() and other_token not in ["and", "the", "but", "for", "with", "from", "said", "want"]:
                    base_weights[j] += 0.15
        
        # NEW: If current token is a content word, attend strongly to recent determiners
        if (len(current_token) > 2 and current_token.isalpha() and 
            current_token not in ["and", "the", "but", "for", "with", "from", "said", "want", "come", "back"]):
            for j in range(max(0, i-3), i):
                prev_token = tokens[j].strip().lower()
                if prev_token in ["the", "a", "an", "your", "his", "her", "their", "my", "this", "that"]:
                    base_weights[j] += 0.20
        
        # 8. Add small random variation to break ties
        base_weights += np.random.uniform(0, 0.01, size=len(base_weights))
        
        # Ensure non-negative and normalize
        base_weights = np.maximum(base_weights, 0.01)
        attention[i, :i+1] = base_weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 2
def first_token_bias_L3H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Adaptive parameters based on sentence length
    is_long_sentence = n > 12
    local_boost = 0.3 if is_long_sentence else 0.0
    first_token_weight = 0.3 if is_long_sentence else 0.4
    
    for i in range(n):
        # Strong attention to first token (BOS-like behavior)
        if i > 0:
            attention[i, 0] = first_token_weight
        else:
            attention[i, 0] = 1.0
        
        # Self-attention
        attention[i, i] = 0.15
        
        # Local attention to previous tokens (decreasing with distance)
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                base_weight = 0.3 / distance
                # Boost local attention for longer sentences
                if is_long_sentence and distance <= 2:
                    base_weight += local_boost / distance
                attention[i, j] = base_weight
        
        # Syntactic dependencies using spacy
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to syntactic head
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    # Find corresponding GPT2 tokens
                    for gpt2_idx in range(i):
                        if gpt2_to_spacy[gpt2_idx] and head_idx in gpt2_to_spacy[gpt2_idx]:
                            attention[i, gpt2_idx] += 0.25
                
                # Attend to children (modifiers, objects, etc.)
                for child in spacy_token.children:
                    child_idx = child.i
                    for gpt2_idx in range(i):
                        if gpt2_to_spacy[gpt2_idx] and child_idx in gpt2_to_spacy[gpt2_idx]:
                            attention[i, gpt2_idx] += 0.15
                
                # Special patterns for specific dependencies
                if spacy_token.dep_ in ['nmod', 'compound']:
                    # Attend more to the head
                    if spacy_token.head != spacy_token:
                        head_idx = spacy_token.head.i
                        for gpt2_idx in range(i):
                            if gpt2_to_spacy[gpt2_idx] and head_idx in gpt2_to_spacy[gpt2_idx]:
                                attention[i, gpt2_idx] += 0.2
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L3H2", attention


# Layer 3, Head 3
def first_token_bias_content_focus_punctuation_L3H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (BOS-like behavior)
        if i > 0:
            attention[i, 0] = 0.6
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly
        
        # Self-attention (moderate)
        if i > 0:
            attention[i, i] = 0.15
        
        # Previous token attention (local context)
        if i > 0:
            attention[i, i-1] = 0.4
        
        # Special patterns for different token types
        token_text = tokens[i].strip()
        
        # Enhanced article handling - attend to verbs they're grammatically linked to
        if token_text.lower() in ['the', 'a', 'an'] and i > 0:
            # First try to find grammatically related verb through spacy dependencies
            found_verb = False
            spacy_indices = gpt2_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                # Check if this article's noun is an object or part of a prepositional phrase
                if spacy_token.head and spacy_token.head.pos_ == 'NOUN':
                    noun_head = spacy_token.head
                    # Check if the noun is an object of a verb or in a prep phrase
                    if noun_head.head and noun_head.head.pos_ == 'VERB':
                        # Find the verb token in GPT2 tokenization
                        verb_spacy_idx = list(doc).index(noun_head.head)
                        for j in range(max(0, i-5), i):
                            if verb_spacy_idx in gpt2_to_spacy[j]:
                                attention[i, j] = 0.7
                                found_verb = True
                                break
                    # Also check for prepositional relationships
                    elif noun_head.dep_ == 'pobj' and noun_head.head.head and noun_head.head.head.pos_ == 'VERB':
                        verb_spacy_idx = list(doc).index(noun_head.head.head)
                        for j in range(max(0, i-5), i):
                            if verb_spacy_idx in gpt2_to_spacy[j]:
                                attention[i, j] = 0.6
                                found_verb = True
                                break
            
            # Fallback to original behavior if no specific verb relationship found
            if not found_verb:
                for j in range(max(0, i-3), i):
                    spacy_indices = gpt2_to_spacy[j]
                    if spacy_indices:
                        spacy_token = doc[spacy_indices[0]]
                        if spacy_token.pos_ in ['VERB', 'NOUN']:
                            attention[i, j] = 0.5
                            break
        
        # Conjunctions attend to previous content
        elif token_text.lower() in ['and', 'but', 'or'] and i > 1:
            # Find previous content word
            for j in range(i-1, max(0, i-4), -1):
                spacy_indices = gpt2_to_spacy[j]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        attention[i, j] = 0.3
                        break
        
        # Prepositions attend to previous verb
        if i > 0:
            spacy_indices = gpt2_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                if spacy_token.pos_ == 'ADP':  # Preposition
                    for j in range(i-1, max(0, i-3), -1):
                        prev_spacy = gpt2_to_spacy[j]
                        if prev_spacy:
                            prev_token = doc[prev_spacy[0]]
                            if prev_token.pos_ == 'VERB':
                                attention[i, j] = 0.4
                                break
        
        # Final tokens (punctuation) attend to recent content
        if i == n - 1 and token_text in ['.', '!', '?']:
            attention[i, i] = 0.2
            # Attend to recent content words
            for j in range(max(0, i-3), i):
                spacy_indices = gpt2_to_spacy[j]
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    if spacy_token.pos_ in ['NOUN', 'VERB']:
                        attention[i, j] = 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 4
def first_token_bias_L3H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (dominant pattern)
        attention_matrix[i, 0] = 0.85
        
        # Self-attention
        attention_matrix[i, i] = 0.08
        
        # Small attention to previous token (if exists)
        if i > 0:
            attention_matrix[i, i-1] = 0.04
        
        # Very small attention to other previous tokens
        for j in range(1, i):
            if j != i-1:  # Don't double-count previous token
                attention_matrix[i, j] = 0.01
    
    # Handle first token separately (can only attend to itself)
    if n > 0:
        attention_matrix[0, :] = 0
        attention_matrix[0, 0] = 1.0
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L3H4", attention_matrix


# Layer 3, Head 5
def first_token_bias_content_focus_L3H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Find verb positions in GPT2 tokens
    verb_positions = set()
    for gpt2_idx, spacy_indices in enumerate(alignment):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['VERB', 'AUX']:
                verb_positions.add(gpt2_idx)
    
    # If no verbs found, use position 1 as fallback (often a verb)
    if not verb_positions and n > 1:
        verb_positions.add(1)
    
    for i in range(n):
        # Strong attention to first token
        attention[i, 0] = 0.8
        
        # Secondary attention to verbs (excluding first token to avoid double-counting)
        for verb_pos in verb_positions:
            if verb_pos != 0 and verb_pos <= i:  # Causal constraint
                attention[i, verb_pos] = 0.15
        
        # Self-attention
        attention[i, i] = 0.05
        
        # Small attention to other earlier positions
        for j in range(1, i):
            if j not in verb_positions:  # Don't override verb attention
                # Enhanced attention to recent context in longer sentences
                if n > 15 and i - j <= 5:  # Recent context in long sentences
                    attention[i, j] = 0.06
                else:
                    attention[i, j] = 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 6
def first_token_bias_content_focus_punctuation_coreference_L3H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for most tokens
        if i > 0:
            attention[i, 0] = 0.4
        
        # Self-attention
        attention[i, i] = 0.2
        
        # Strong attention to immediate predecessor for continuation tokens
        if i > 0:
            # Check if current token continues the previous (no leading space)
            if not tokens[i].startswith(' ') and tokens[i-1] != '"':
                attention[i, i-1] = 0.6
            else:
                attention[i, i-1] = 0.1
        
        # NEW: Enhanced subword token handling
        if i > 0 and not tokens[i].startswith(' '):
            # For subword continuations, look for all prefix parts of the same word
            for j in range(i-1, -1, -1):
                if not tokens[j].startswith(' ') or j == 0:
                    # This is part of the same word - give stronger attention
                    attention[i, j] += 0.2
                else:
                    # Found the word boundary, stop looking
                    break
        
        # Look for syntactic relationships using spacy alignment
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Attend to syntactic head
            if spacy_token.head != spacy_token:
                head_text = spacy_token.head.text
                # Find GPT2 tokens that match the head
                for j in range(i):
                    if head_text.lower() in tokens[j].lower().replace(' ', ''):
                        attention[i, j] += 0.3
            
            # Attend to modifiers (adjectives to nouns they modify)
            for child in spacy_token.children:
                if child.dep_ in ['amod', 'compound']:
                    child_text = child.text
                    for j in range(i):
                        if child_text.lower() in tokens[j].lower().replace(' ', ''):
                            attention[i, j] += 0.3
        
        # Special patterns for punctuation and quotes
        token_clean = tokens[i].strip()
        if token_clean in ['.', ',"', '!"', '?"']:
            # Punctuation attends to main content words nearby
            for j in range(max(0, i-5), i):
                if tokens[j].strip() and tokens[j].startswith(' '):
                    attention[i, j] += 0.2
        
        # Pattern for quoted speech - attend to speaker
        if '"' in tokens[i] or 'said' in tokens[i].lower():
            # Look for names or pronouns nearby
            for j in range(i):
                if any(c.isupper() for c in tokens[j]) or tokens[j].lower().strip() in ['he', 'she', 'i', 'lily', 'timmy']:
                    attention[i, j] += 0.3
        
        # Recency bias - small attention to recent tokens
        for j in range(max(0, i-3), i):
            attention[i, j] += 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 7
def first_token_bias_content_focus_punctuation_L3H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Get spacy parse for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base self-attention
        attention[i, i] = 0.1
        
        # Strong first token attention for most tokens
        if i > 0:
            attention[i, 0] = 0.3
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly
        
        # Previous token attention (especially strong for function words and continuations)
        if i > 0:
            # Check if current token looks like a continuation or function word
            token_text = tokens[i].strip()
            if token_text in ['and', 'but', 'the', 'a', 'an', 'in', 'on', 'with', 'to', 'of']:
                attention[i, i-1] = 0.5
            else:
                attention[i, i-1] = 0.2
        
        # Get spacy tokens aligned to current GPT2 token
        spacy_indices = gpt2_to_spacy[i]
        
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Adjective-noun relationships
                if spacy_token.pos_ == 'ADJ':
                    # Adjectives attend strongly to their head noun
                    if spacy_token.head.pos_ == 'NOUN':
                        # Find GPT2 tokens corresponding to the head noun
                        for j in range(i+1, min(i+3, n)):  # Look ahead a bit
                            head_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                            if any(idx == spacy_token.head.i for idx in head_spacy_indices):
                                attention[i, j] = 0.6
                
                # Noun attending to modifying adjectives
                if spacy_token.pos_ == 'NOUN':
                    for child in spacy_token.children:
                        if child.dep_ == 'amod':  # adjectival modifier
                            # Find GPT2 token for this adjective
                            for j in range(max(0, i-3), i):
                                adj_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                                if any(idx == child.i for idx in adj_spacy_indices):
                                    attention[i, j] = 0.5
                
                # Verb-subject/object relationships
                if spacy_token.pos_ == 'VERB':
                    # Attend to subject
                    for child in spacy_token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            for j in range(i):
                                subj_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                                if any(idx == child.i for idx in subj_spacy_indices):
                                    attention[i, j] = 0.4
                
                # Preposition attending to object
                if spacy_token.pos_ == 'ADP':  # preposition
                    # Look for object of preposition
                    for child in spacy_token.children:
                        if child.dep_ == 'pobj':
                            for j in range(i+1, min(i+4, n)):
                                obj_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                                if any(idx == child.i for idx in obj_spacy_indices):
                                    attention[i, j] = 0.4
        
        # Special handling for punctuation
        if tokens[i].strip() in [',', '.', '!', '?']:
            # Punctuation often attends to the main content word before it
            for j in range(max(0, i-3), i):
                if j < len(gpt2_to_spacy):
                    spacy_indices_j = gpt2_to_spacy[j]
                    for spacy_idx_j in spacy_indices_j:
                        if spacy_idx_j < len(doc) and doc[spacy_idx_j].pos_ in ['NOUN', 'VERB', 'ADJ']:
                            attention[i, j] = 0.3
                            break
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 8
def first_token_bias_content_focus_punctuation_L3H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # First token always has strong self-attention
        if i == 0:
            attention[i, i] = 1.0
            continue
            
        # Initialize with small uniform attention to all previous tokens
        for j in range(i + 1):
            attention[i, j] = 0.05
        
        # Strong attention to first token
        attention[i, 0] += 0.4
        
        # Self-attention
        attention[i, i] += 0.2
        
        # Strong attention to immediately previous token
        if i > 0:
            attention[i, i-1] += 0.3
        
        # Handle punctuation patterns
        token_text = tokens[i].strip()
        if token_text in [',', '.']:
            # Punctuation attends strongly to itself
            attention[i, i] += 0.3
            # And to nearby content words
            for j in range(max(0, i-3), i):
                if tokens[j].strip() not in [',', '.', 'and', 'or', 'but']:
                    attention[i, j] += 0.2
        
        # Handle prepositions and function words
        if i > 0 and tokens[i-1].strip().lower() in ['with', 'on', 'to', 'of', 'in', 'at', 'by']:
            attention[i, i-1] += 0.4
        
        # Handle conjunctions and connectives  
        if tokens[i].strip().lower() in ['but', 'and', 'or']:
            # Look for nearby punctuation to attend to
            for j in range(max(0, i-3), i):
                if tokens[j].strip() in [',']:
                    attention[i, j] += 0.4
        
        # Handle words after conjunctions/prepositions
        if i > 1:
            prev_token = tokens[i-1].strip().lower()
            if prev_token in ['like', 'if', 'with', 'on', 'named', 'said']:
                attention[i, i-1] += 0.4
        
        # Special patterns for specific constructions
        if i >= 2:
            # Pattern: "X said Y" -> Y attends to "said"  
            if tokens[i-1].strip().lower() == 'said':
                attention[i, i-1] += 0.5
            # Pattern: "on the" -> "the" attends strongly to "on"
            if tokens[i-1].strip().lower() == 'on' and tokens[i].strip().lower() == 'the':
                attention[i, i-1] += 0.5
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 9
def decaying_first_token_bias_content_focus_L3H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Get verb positions (in GPT2 token space)
    verb_positions = []
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['VERB', 'AUX']:
                verb_positions.append(i)
                break
    
    for i in range(n):
        if i == 0:
            # First token: perfect self-attention
            attention[i, 0] = 1.0
        else:
            # Base attention weights
            weights = np.zeros(i + 1)
            
            # Strong attention to first token (positional bias)
            weights[0] = 0.3
            
            # Find most recent verb before current position
            recent_verb = None
            for v_pos in reversed(verb_positions):
                if v_pos < i:
                    recent_verb = v_pos
                    break
            
            if recent_verb is not None:
                # High attention to most recent verb
                weights[recent_verb] = 0.5
                
                # Medium attention to token immediately after verb
                if recent_verb + 1 < i:
                    weights[recent_verb + 1] = 0.2
            
            # Self-attention (lower priority)
            weights[i] = 0.1
            
            # Attention to previous token
            if i > 0:
                weights[i-1] += 0.15
            
            # Small positional decay for other tokens
            for j in range(1, i):
                if weights[j] == 0:  # Don't override verb attention
                    decay_factor = 1.0 / (i - j + 1)
                    weights[j] = 0.05 * decay_factor
            
            # Special handling for specific token types
            current_token = tokens[i].strip()
            
            # If current token looks like it might be part of a verb phrase
            if current_token in ['to', 'for', 'and', 'had', 'made', 'take']:
                # Boost attention to recent verbs
                if recent_verb is not None:
                    weights[recent_verb] *= 1.5
                    
            # Punctuation tends to attend to main verbs
            if current_token in ['.', ',']:
                if recent_verb is not None:
                    weights[recent_verb] *= 1.2
                # Also some attention to sentence start
                weights[0] *= 1.3
            
            attention[i, :i+1] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 3, Head 10
def first_token_bias_content_focus_L3H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention (decreases with distance from start)
        if i > 0:
            first_token_weight = 0.8 - 0.1 * min(i, 6)  # Decreases but stays strong
            attention[i, 0] = first_token_weight
        
        # Self-attention (stronger for content words)
        self_weight = 0.15
        # Check if this token corresponds to a content word in spacy
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    self_weight = 0.25
        attention[i, i] = self_weight
        
        # Previous token attention
        if i > 0:
            prev_weight = 0.08
            attention[i, i-1] = prev_weight
        
        # Syntactic attention - look for dependencies
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to head
                if spacy_token.head != spacy_token and spacy_token.head.i < len(doc):
                    head_gpt2_indices = []
                    for j in range(n):
                        if gpt2_to_spacy[j] and spacy_token.head.i in gpt2_to_spacy[j]:
                            head_gpt2_indices.append(j)
                    for head_j in head_gpt2_indices:
                        if head_j <= i:  # Causal constraint
                            attention[i, head_j] += 0.06
                
                # If this is a head, attend to its children
                for child in spacy_token.children:
                    if child.i < len(doc):
                        child_gpt2_indices = []
                        for j in range(i):  # Only previous tokens
                            if gpt2_to_spacy[j] and child.i in gpt2_to_spacy[j]:
                                child_gpt2_indices.append(j)
                        for child_j in child_gpt2_indices:
                            attention[i, child_j] += 0.04
        
        # Add some uniform attention to nearby tokens
        for j in range(max(0, i-3), i):
            if j != i-1 and j != 0:  # Not previous token or first token
                attention[i, j] += 0.02
        
        # Special case for first token - only self-attention
        if i == 0:
            attention[i, 0] = 1.0
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 3, Head 11
def first_token_bias_content_focus_punctuation_coreference_L3H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for all tokens
        attention[i, 0] = 0.4
        
        # Self-attention
        attention[i, i] = 0.15
        
        # Previous token attention (especially strong for adjacent relationships)
        if i > 0:
            attention[i, i-1] = 0.25
        
        # Look for syntactic relationships
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Attend to syntactic head
            if spacy_token.head != spacy_token:  # Not root
                head_token = spacy_token.head
                # Find GPT2 tokens that align with the head
                for j in range(i):
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices and any(idx == head_token.i for idx in j_spacy_indices):
                        attention[i, j] += 0.3
                        break
            
            # Special patterns based on token relationships
            token_text = tokens[i].strip().lower()
            
            # Prepositions and articles attend to nearby content words
            if spacy_token.pos_ in ['ADP', 'DET'] and i > 1:
                attention[i, i-2] = 0.2
            
            # Verbs attend to their objects/complements
            if spacy_token.pos_ == 'VERB':
                for child in spacy_token.children:
                    if child.dep_ in ['dobj', 'pobj', 'attr']:
                        for j in range(i+1, n):
                            j_spacy_indices = gpt2_to_spacy[j]
                            if j_spacy_indices and any(idx == child.i for idx in j_spacy_indices):
                                attention[i, j] = 0.25
                                break
            
            # Adjectives attend to their noun heads more strongly
            if spacy_token.pos_ == 'ADJ' and spacy_token.head.pos_ == 'NOUN':
                for j in range(i):
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices and any(idx == spacy_token.head.i for idx in j_spacy_indices):
                        attention[i, j] += 0.2
                        break
            
            # NEW: Handle pronoun coreference patterns
            if spacy_token.pos_ == 'PRON':
                # Pronouns attend more strongly to earlier nouns/pronouns with similar properties
                for j in range(i):
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices:
                        j_spacy_token = doc[j_spacy_indices[0]]
                        # Same-person pronouns or nouns that could be antecedents
                        if ((j_spacy_token.pos_ == 'PRON' and 
                             spacy_token.tag_ == j_spacy_token.tag_) or
                            (j_spacy_token.pos_ in ['NOUN', 'PROPN'] and 
                             j > 0 and i - j < 10)):  # Within reasonable distance
                            attention[i, j] += 0.2
        
        # Special handling for punctuation
        if tokens[i] in [',', '.', '"', "'", '!', '?']:
            # Punctuation attends more to recent content
            if i > 0:
                attention[i, i-1] += 0.2
            if i > 1:
                attention[i, i-2] += 0.15
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 4, Head 0
def decaying_first_token_bias_content_focus_L4H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Find verbs in GPT2 tokens
    verb_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ == "VERB":
                verb_tokens.add(i)
    
    # Find prepositions
    prep_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ == "ADP":
                prep_tokens.add(i)
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Base attention with exponential decay
        for j in range(i + 1):
            distance = i - j
            attention[i, j] = np.exp(-0.3 * distance)
        
        # Strong first token bias
        if n > 0:
            attention[i, 0] += 2.0
        
        # Self-attention boost
        attention[i, i] += 0.5
        
        # Verb attraction - tokens attend strongly to recent verbs
        for j in range(i):
            if j in verb_tokens:
                # Stronger for closer verbs
                distance = i - j
                verb_boost = 3.0 * np.exp(-0.2 * distance)
                attention[i, j] += verb_boost
        
        # Preposition-object relationship
        # If previous token is preposition, attend to it strongly
        if i > 0 and (i-1) in prep_tokens:
            attention[i, i-1] += 2.0
        
        # If token 2 positions back is preposition, also attend to it
        if i > 1 and (i-2) in prep_tokens:
            attention[i, i-2] += 1.5
        
        # NEW: Enhanced prepositional phrase attention
        # Tokens in prepositional phrases should attend strongly to the preposition
        for j in range(i):
            if j in prep_tokens:
                distance = i - j
                # Strong attention to prepositions, especially for nearby tokens
                if distance <= 5:  # Within reasonable range of prep phrase
                    prep_boost = 4.0 * np.exp(-0.15 * distance)
                    attention[i, j] += prep_boost
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 4, Head 1
def first_token_bias_content_focus_L4H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base first-token attention (very strong pattern)
        attention[i, 0] = 0.8
        
        # Self-attention
        attention[i, i] = 0.15
        
        # Previous token attention (moderate)
        if i > 0:
            attention[i, i-1] = 0.2
        
        # Find syntactic relationships using spacy
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Attend to syntactic head
            if spacy_token.head != spacy_token:
                head_text = spacy_token.head.text
                # Find GPT2 tokens that contain this head
                for j in range(i):
                    if head_text.lower() in tokens[j].lower().replace(' ', ''):
                        attention[i, j] += 0.3
            
            # Special attention for conjunctions and auxiliary verbs
            if spacy_token.pos_ in ['CCONJ', 'AUX'] or spacy_token.text.lower() in ['and', 'was', 'were']:
                for j in range(i):
                    attention[i, j] += 0.1
        
        # Attention to function words and clause boundaries
        for j in range(i):
            token_text = tokens[j].strip().lower()
            if token_text in ['and', 'was', 'were', 'the', ',', 'to', 'in', 'on']:
                attention[i, j] += 0.15
        
        # Special patterns for specific constructions
        current_token = tokens[i].strip().lower()
        
        # Tokens after "and" attend strongly to "and"
        if i > 0 and tokens[i-1].strip().lower() == 'and':
            for j in range(i):
                if tokens[j].strip().lower() == 'and':
                    attention[i, j] += 0.4
        
        # Verbs attend to subjects (first few tokens often)
        if spacy_indices and doc[spacy_indices[0]].pos_ == 'VERB':
            for j in range(min(4, i)):  # Look at first few tokens
                attention[i, j] += 0.2
        
        # Adjectives and adverbs attend to nearby nouns/verbs
        if spacy_indices and doc[spacy_indices[0]].pos_ in ['ADJ', 'ADV']:
            for j in range(max(0, i-3), i):
                spacy_j = gpt2_to_spacy[j]
                if spacy_j and doc[spacy_j[0]].pos_ in ['NOUN', 'VERB']:
                    attention[i, j] += 0.25
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 4, Head 2
def first_token_bias_content_focus_punctuation_coreference_L4H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Base first-token attention (very strong for most tokens)
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0
        
        # Self-attention (moderate)
        attention[i, i] = 0.3
        
        # Get spacy info for current token
        spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
        
        # Add attention to nearby tokens and syntactic relations
        for j in range(i):
            if j == 0:
                continue  # Already handled first token
            
            # Distance-based local attention
            dist = i - j
            if dist == 1:  # Previous token
                attention[i, j] = 0.15
            elif dist <= 3:  # Nearby tokens
                attention[i, j] = 0.1 / dist
            else:  # Distant tokens
                attention[i, j] = 0.02
            
            # Boost attention for syntactic relationships
            if spacy_indices:
                for si in spacy_indices:
                    if si < len(doc):
                        current_spacy = doc[si]
                        
                        # Check if j aligns with syntactically related tokens
                        spacy_j_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                        for sj in spacy_j_indices:
                            if sj < len(doc):
                                target_spacy = doc[sj]
                                
                                # Boost for head relationships
                                if current_spacy.head == target_spacy or target_spacy.head == current_spacy:
                                    attention[i, j] *= 2.0
                                
                                # Boost for verb-subject relationships
                                if current_spacy.pos_ == "VERB" and target_spacy.dep_ in ["nsubj", "nsubjpass"]:
                                    attention[i, j] *= 1.5
            
            # NEW: Boost attention to syntactic anchor tokens
            token_j = tokens[j].strip()
            if token_j in ['that', 'she', 'he', 'it', ',', ',"', '"'] and dist > 1:
                # Get spacy info for token j to check if it's a syntactic anchor
                spacy_j_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                is_anchor = False
                
                for sj in spacy_j_indices:
                    if sj < len(doc):
                        target_spacy = doc[sj]
                        # Boost if it's a pronoun, subordinating conjunction, or punctuation
                        if (target_spacy.pos_ in ["PRON", "SCONJ"] or 
                            target_spacy.dep_ in ["nsubj", "nsubjpass", "punct"] or
                            token_j in [',', ',"', '"']):
                            is_anchor = True
                            break
                
                if is_anchor:
                    attention[i, j] *= 3.0
    
    # Handle special cases for punctuation
    for i in range(n):
        token = tokens[i]
        if token.strip() in '.!?':
            # Punctuation attends strongly to itself and moderately to sentence elements
            attention[i, :] *= 0.3
            attention[i, i] = 0.4
            if i > 0:
                attention[i, 0] = 0.3
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 4, Head 3
def first_token_bias_content_focus_L4H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_gpt2 = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention distribution
        row_attention = np.zeros(n)
        
        # Strong attention to first token (positional bias)
        if i > 0:
            row_attention[0] = 0.6
        else:
            row_attention[0] = 0.8  # Self-attention for first token
        
        # Self-attention (moderate)
        if i > 0:
            row_attention[i] = 0.2
        
        # Syntactic relationships
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Attend to syntactic head
            if spacy_token.head != spacy_token and spacy_token.head.i < len(doc):
                head_gpt2_indices = spacy_to_gpt2[spacy_token.head.i]
                for head_idx in head_gpt2_indices:
                    if head_idx <= i:  # Causal constraint
                        row_attention[head_idx] += 0.4
            
            # Attend to syntactic children (modifiers, objects, etc.)
            for child in spacy_token.children:
                if child.i < len(doc):
                    child_gpt2_indices = spacy_to_gpt2[child.i]
                    for child_idx in child_gpt2_indices:
                        if child_idx <= i:  # Causal constraint
                            row_attention[child_idx] += 0.3
            
            # Special patterns for specific POS/dependencies
            if spacy_token.pos_ == "VERB":
                # Verbs attend strongly to subjects
                for child in spacy_token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subj_gpt2_indices = spacy_to_gpt2[child.i]
                        for subj_idx in subj_gpt2_indices:
                            if subj_idx <= i:
                                row_attention[subj_idx] += 0.5
            
            elif spacy_token.pos_ in ["ADJ", "DET"]:
                # Modifiers attend to their heads
                if spacy_token.head != spacy_token:
                    head_gpt2_indices = spacy_to_gpt2[spacy_token.head.i]
                    for head_idx in head_gpt2_indices:
                        if head_idx <= i:
                            row_attention[head_idx] += 0.4
        
        # Local context (previous few tokens)
        for j in range(max(0, i-2), i):
            row_attention[j] += 0.1
        
        # Apply causal mask
        for j in range(i+1, n):
            row_attention[j] = 0
        
        attention[i] = row_attention
    
    # Normalize and apply causal mask
    attention = make_row_stochastic(attention)
    attention = apply_causal_mask(attention)
    
    return "first_token_bias_content_", attention


# Layer 4, Head 4
def first_token_bias_content_focus_coreference_L4H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (except for first token itself)
        if i > 0:
            attention[i, 0] = 0.7 + 0.2 * (1.0 / (i + 1))  # Decay with distance but stay high
        else:
            attention[i, 0] = 1.0  # First token attends to itself with max weight
        
        # Self-attention (moderate for non-first tokens)
        if i > 0:
            attention[i, i] = 0.08
        
        # Previous token attention (local context)
        if i > 1:
            attention[i, i-1] = 0.05
        
        # Add some attention to tokens 2-3 positions back
        if i > 2:
            attention[i, i-2] = 0.03
        if i > 3:
            attention[i, i-3] = 0.02
            
        # Try to identify syntactic relationships using spacy alignment
        if gpt2_to_spacy[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_idx = gpt2_to_spacy[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # If this is a verb, attend more to subject-like tokens
                if spacy_token.pos_ == 'VERB':
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            spacy_j = gpt2_to_spacy[j][0]
                            if spacy_j < len(doc):
                                spacy_token_j = doc[spacy_j]
                                # Look for subjects, pronouns, proper nouns
                                if (spacy_token_j.dep_ == 'nsubj' or 
                                    spacy_token_j.pos_ == 'PRON' or
                                    spacy_token_j.pos_ == 'PROPN'):
                                    attention[i, j] += 0.04
                
                # If this is a preposition, attend to following noun
                if spacy_token.pos_ == 'ADP' and i < n-1:
                    attention[i, i+1] = min(attention[i, i+1] + 0.03, 1.0)
                
                # Punctuation attends to recent content words
                if spacy_token.pos_ == 'PUNCT':
                    for j in range(max(0, i-5), i):
                        if gpt2_to_spacy[j]:
                            spacy_j = gpt2_to_spacy[j][0]
                            if spacy_j < len(doc):
                                spacy_token_j = doc[spacy_j]
                                if spacy_token_j.pos_ in ['NOUN', 'VERB', 'ADJ']:
                                    attention[i, j] += 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 4, Head 5
def first_token_bias_content_focus_L4H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (except for first token itself)
        if i > 0:
            attention[i, 0] = 0.7
        
        # Self-attention (moderate)
        attention[i, i] = 0.2
        
        # Attention to immediate predecessor
        if i > 0:
            attention[i, i-1] = 0.3
        
        # Attention to tokens 2 positions back (weaker)
        if i > 1:
            attention[i, i-2] = 0.1
        
        # NEW: Special handling for tokens following "named"
        if i > 0 and " named" in tokens[i-1]:
            # Tokens immediately after "named" should strongly attend to "named"
            attention[i, i-1] = 0.7
        
        # NEW: Look for "named" in earlier positions and boost attention
        for j in range(i):
            if " named" in tokens[j] or tokens[j] == "named":
                # Names following "named" should attend strongly to it
                if i > j:
                    attention[i, j] += 0.5
        
        # Try to identify syntactic relationships using spacy
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]  # Use first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # If this is a verb, attend to its subject
                if spacy_token.pos_ == "VERB":
                    for child in spacy_token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            # Find GPT2 tokens that align with this spacy token
                            for j in range(min(i, n)):
                                if gpt2_to_spacy[j] and child.i in gpt2_to_spacy[j]:
                                    attention[i, j] += 0.2
                
                # If this is a noun, attend to its modifiers
                if spacy_token.pos_ in ["NOUN", "PROPN"]:
                    for child in spacy_token.children:
                        if child.dep_ in ["amod", "det", "compound"]:
                            for j in range(min(i, n)):
                                if gpt2_to_spacy[j] and child.i in gpt2_to_spacy[j]:
                                    attention[i, j] += 0.15
                
                # If this token depends on something, attend to its head
                if spacy_token.head != spacy_token and spacy_token.dep_ in ["amod", "compound", "det"]:
                    head = spacy_token.head
                    for j in range(min(i, n)):
                        if gpt2_to_spacy[j] and head.i in gpt2_to_spacy[j]:
                            attention[i, j] += 0.25
    
    # Special handling for first token (attends only to itself)
    if n > 0:
        attention[0, :] = 0
        attention[0, 0] = 1.0
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 4, Head 6
def first_token_bias_content_focus_L4H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base weights
        weights = np.zeros(n)
        
        # Strong self-attention
        weights[i] = 0.1
        
        # Very strong first token attention, especially for early tokens
        if i > 0:
            first_token_weight = max(0.8, 1.0 - 0.1 * i)
            weights[0] = first_token_weight
        
        # Previous token attention (local dependencies)
        if i > 0:
            weights[i-1] += 0.3
            
        # Two tokens back (less strong)
        if i > 1:
            weights[i-2] += 0.1
            
        # Get spacy alignment for current token
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Syntactic dependencies
            for j in range(i):
                j_spacy_indices = gpt2_to_spacy[j]
                if j_spacy_indices:
                    j_spacy_token = doc[j_spacy_indices[0]]
                    
                    # Head-dependent relationships
                    if spacy_token.head == j_spacy_token:
                        weights[j] += 0.4
                    elif j_spacy_token.head == spacy_token:
                        weights[j] += 0.3
                        
                    # Adjective-noun relationships
                    if (spacy_token.pos_ == "NOUN" and j_spacy_token.pos_ == "ADJ" and 
                        abs(spacy_token.i - j_spacy_token.i) <= 3):
                        weights[j] += 0.2
                        
                    # Verb-subject relationships
                    if (spacy_token.pos_ == "VERB" and j_spacy_token.pos_ in ["NOUN", "PRON"] and
                        j_spacy_token.dep_ in ["nsubj", "nsubjpass"]):
                        weights[j] += 0.3
        
        # Punctuation patterns
        token_text = tokens[i].strip()
        if token_text in ['"', "'", ',"', '!"', '.']:
            # Punctuation attends to nearby content
            for j in range(max(0, i-5), i):
                if tokens[j].strip() and not tokens[j].strip() in ['"', "'", ',', '.', '!', '?']:
                    weights[j] += 0.15
                    
        # Quote handling - content after quotes attends to quote
        if i > 0 and tokens[i-1].strip() in ['"', "'"]:
            weights[i-1] += 0.2
            
        # Special handling for sentence-final tokens
        if i == n-1:
            # Final token has more distributed attention
            for j in range(max(0, i-3), i):
                weights[j] += 0.05
        
        # Normalize and apply causal mask
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights[i] = 1.0
            
        attention[i] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 4, Head 7
def decaying_first_token_bias_content_focus_L4H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention distribution
        base_weights = np.zeros(i + 1)  # Can only attend to tokens 0..i
        
        if i == 0:
            # First token: perfect self-attention
            base_weights[0] = 1.0
        else:
            # Strong attention to first token (BOS-like behavior)
            base_weights[0] = 0.8
            
            # Self-attention for content words
            token_text = tokens[i].strip().lower()
            if len(token_text) > 2 and token_text.isalpha():
                base_weights[i] = 0.4
            elif i > 0:
                base_weights[i] = 0.2
            
            # Get spacy information if available
            spacy_indices = alignment[i] if i < len(alignment) else []
            
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                
                # Syntactic dependencies
                if spacy_token.dep_ in ['prep', 'aux', 'det']:
                    # Prepositions, auxiliaries, determiners attend to their head
                    if spacy_token.head != spacy_token:
                        head_spacy_idx = list(doc).index(spacy_token.head)
                        # Find corresponding GPT2 tokens
                        for j in range(i):
                            if j < len(alignment) and head_spacy_idx in alignment[j]:
                                base_weights[j] += 0.3
                                break
                
                # Conjunctions attend to coordinated elements
                if spacy_token.text.lower() in ['and', 'but', 'or']:
                    # Look for nearby content words
                    for j in range(max(0, i-5), i):
                        if j < len(alignment) and alignment[j]:
                            spacy_j = doc[alignment[j][0]]
                            if spacy_j.pos_ in ['NOUN', 'VERB', 'ADJ']:
                                base_weights[j] += 0.2
                
                # Verbs attend to their subjects/objects
                if spacy_token.pos_ == 'VERB':
                    for child in spacy_token.children:
                        if child.dep_ in ['nsubj', 'dobj']:
                            child_spacy_idx = list(doc).index(child)
                            for j in range(i):
                                if j < len(alignment) and child_spacy_idx in alignment[j]:
                                    base_weights[j] += 0.2
                                    break
            
            # Sequential patterns
            if i > 0:
                prev_token = tokens[i-1].strip().lower()
                curr_token = tokens[i].strip().lower()
                
                # Specific sequential dependencies
                if curr_token in ['to'] and prev_token in ['went', 'had']:
                    base_weights[i-1] += 0.3
                elif curr_token in ['not'] and prev_token in ['learn']:
                    base_weights[i-1] += 0.4
                elif prev_token in ['named', 'like'] and len(curr_token) > 2:
                    base_weights[i-1] += 0.2
                
                # General adjacency for function words
                if len(curr_token) <= 3 and not curr_token.isalpha():
                    base_weights[i-1] += 0.1
            
            # Punctuation patterns
            if tokens[i] in ['.', ',', '!', '?']:
                # Attend to important content words
                for j in range(i):
                    if j < len(alignment) and alignment[j]:
                        spacy_j = doc[alignment[j][0]]
                        if spacy_j.pos_ in ['NOUN', 'VERB'] and spacy_j.dep_ in ['ROOT', 'nsubj', 'dobj']:
                            base_weights[j] += 0.15
            
            # Distance decay for first token attention
            if i > 2:
                base_weights[0] *= (0.9 ** (i - 2))
        
        # Normalize and set
        if base_weights.sum() > 0:
            base_weights = base_weights / base_weights.sum()
        attention[i, :i+1] = base_weights
    
    # Apply causal mask and ensure row-stochastic
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 4, Head 8
def decaying_first_token_bias_content_focus_L4H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Get content word indicators
    content_pos_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    is_content_word = np.zeros(n, dtype=bool)
    
    for i, spacy_indices in enumerate(alignment):
        if spacy_indices:
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc) and doc[spacy_idx].pos_ in content_pos_tags:
                    is_content_word[i] = True
                    break
    
    for i in range(n):
        # Very strong attention to first token
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0  # Self-attention for first token
            
        if i > 0:
            # Recency bias - attend to previous tokens with decay
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
            
            # Boost attention to content words
            for j in range(i):
                if is_content_word[j] and j > 0:  # Don't double-boost first token
                    attention[i, j] *= 1.5
            
            # Special boost for sentence-final tokens attending to content
            if i == n - 1:  # Last token
                for j in range(i):
                    if is_content_word[j]:
                        attention[i, j] *= 2.0
    
    # NEW: Long-range content word connections for longer sentences
    if n > 15:  # Only apply to longer sentences
        for i in range(n):
            if is_content_word[i] and i > 0:
                for j in range(i):
                    if is_content_word[j] and j > 0:
                        distance = i - j
                        if distance > 5:  # Only for distant content words
                            # Add significant long-range attention
                            attention[i, j] += 0.06 * np.exp(-0.1 * (distance - 5))
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 4, Head 9
def decaying_first_token_bias_content_focus_L4H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong self-attention for first token
        if i == 0:
            attention[i, i] = 1.0
            continue
            
        # Base attention to first token (strong bias)
        attention[i, 0] = 0.4
        
        # Strong attention to immediately preceding token
        attention[i, i-1] = 0.3
        
        # Moderate self-attention
        attention[i, i] = 0.1
        
        # Get spacy information for current token
        spacy_indices = alignment[i]
        current_spacy_tokens = [doc[idx] for idx in spacy_indices if idx < len(doc)]
        
        if current_spacy_tokens:
            current_token = current_spacy_tokens[0]
            
            # Syntactic dependencies
            for j in range(i):
                target_spacy_indices = alignment[j]
                target_spacy_tokens = [doc[idx] for idx in target_spacy_indices if idx < len(doc)]
                
                if target_spacy_tokens:
                    target_token = target_spacy_tokens[0]
                    
                    # Verb attending to subject
                    if current_token.pos_ == 'VERB' and target_token.dep_ in ['nsubj', 'nsubjpass']:
                        attention[i, j] += 0.2
                    
                    # Object attending to verb
                    if current_token.dep_ in ['dobj', 'pobj'] and target_token.pos_ == 'VERB':
                        attention[i, j] += 0.15
                    
                    # Modifier attending to head
                    if current_token.head == target_token:
                        attention[i, j] += 0.15
                    
                    # Preposition attending to object
                    if current_token.pos_ == 'ADP' and target_token.dep_ == 'pobj' and target_token.head == current_token:
                        attention[i, j] += 0.2
        
        # Distance-based decay for remaining positions
        for j in range(i):
            if j not in [0, i-1]:  # Already handled first token and previous token
                distance = i - j
                decay_factor = max(0.05, 0.15 / (distance + 1))
                attention[i, j] += decay_factor
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 4, Head 10
def first_token_bias_punctuation_L4H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (except first token to itself)
        if i > 0:
            attention[i, 0] = 0.9
        
        # Self-attention
        attention[i, i] = 0.1 if i > 0 else 1.0
        
        # Identify punctuation tokens
        punct_indices = []
        for j in range(i + 1):  # Only look at previous tokens (causal)
            if tokens[j] in [',', '.', '!', '?', ';"', '."', '"', "'", ':']: 
                punct_indices.append(j)
        
        # If there are punctuation marks, distribute some attention to them
        if punct_indices and i > 0:
            for p_idx in punct_indices:
                if p_idx != 0:  # Don't double-count first token punctuation
                    attention[i, p_idx] += 0.15
        
        # Add small attention to nearby tokens
        for j in range(max(0, i-3), i):
            if j != 0 and j != i:  # Don't double-count first token or self
                # Higher attention to function words
                if tokens[j].lower().strip() in [' to', ' the', ' a', ' an', ' and', ' or', ' but', ' if', ' when', ' with']:
                    attention[i, j] += 0.08
                else:
                    attention[i, j] += 0.03
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 4, Head 11
def general_pattern_L4H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Get spacy parse and alignment for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Initialize attention matrix
    attention_matrix = np.zeros((n, n))
    
    # Main pattern: attend to previous token
    for i in range(n):
        if i == 0:
            # First token attends to itself
            attention_matrix[i, i] = 1.0
        else:
            # Check if this token should have very strong attention to previous token
            strong_prev_attention = False
            
            # Detect patterns that indicate strong previous-token attention
            if i > 0:
                current_token = tokens[i].strip()
                prev_token = tokens[i-1].strip()
                
                # Strong attention for certain linguistic patterns:
                # 1. Content words following function words
                # 2. Tokens that complete common phrases
                # 3. Punctuation following words
                if (
                    # Adjectives/nouns following determiners or possessives
                    (prev_token.lower() in ['the', 'your', 'this', 'that', 'her', 'his', 'my'] and 
                     len(current_token) > 2 and current_token.isalpha()) or
                    # Verbs following auxiliaries or modals  
                    (prev_token.lower() in ["'m", 'can', 'want', 'will', 'have'] and
                     len(current_token) > 2) or
                    # Punctuation following words
                    (current_token in [',', '.', '!', '?', ';"', ',"', '."', '!"', '?"'] and
                     prev_token not in [' ', '\n']) or
                    # Nouns following prepositions
                    (prev_token.lower() in ['to', 'into', 'at', 'for', 'with'] and
                     len(current_token) > 2 and current_token.isalpha())
                ):
                    strong_prev_attention = True
            
            if strong_prev_attention:
                # Very strong attention to previous token
                attention_matrix[i, i-1] = 0.95
                attention_matrix[i, i] = 0.05
            else:
                # Original pattern: attend primarily to previous token
                attention_matrix[i, i-1] = 0.8
                
                # Add some self-attention and attention to earlier tokens
                attention_matrix[i, i] = 0.1
                
                # Distribute remaining attention to earlier positions
                if i >= 2:
                    for j in range(i-1):
                        attention_matrix[i, j] = 0.1 / (i-1)
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "general_pattern_L4H11", attention_matrix


# Layer 5, Head 0
def first_token_bias_content_focus_punctuation_L5H0(sentence: str, tokenizer: PreTrainedTokenizerBase):
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (dominant pattern)
        if i == 0:
            attention[i, 0] = 1.0  # First token attends to itself
        else:
            attention[i, 0] = 0.85  # Other tokens strongly attend to first token
        
        # Self-attention (moderate weight)
        if i > 0:
            attention[i, i] = 0.08
        
        # Local context attention (weaker)
        for j in range(max(0, i-3), i):
            if j != 0 and j != i:  # Not first token or self
                attention[i, j] = 0.02
        
        # Special handling for punctuation
        token = tokens[i]
        if token in ['.', ',', '!', '?', '"', "'", ':', ';']:
            # Punctuation has more diverse attention
            attention[i, 0] *= 0.7  # Reduce first-token attention
            if i > 0:
                attention[i, i] *= 1.5  # Increase self-attention
            # Add some attention to recent content words
            for j in range(max(0, i-5), i):
                if tokens[j].strip() and tokens[j] not in ['.', ',', '!', '?', '"', "'", ':', ';']:
                    attention[i, j] += 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 1
def first_token_bias_L5H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention_matrix = np.zeros((n, n))
    
    # First token attends only to itself
    if n > 0:
        attention_matrix[0, 0] = 1.0
    
    # All other tokens attend primarily to first token, with small self-attention
    for i in range(1, n):
        attention_matrix[i, 0] = 0.98  # High attention to first token
        attention_matrix[i, i] = 0.02  # Small self-attention
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L5H1", attention_matrix


# Layer 5, Head 2
def first_token_bias_content_focus_stochastic_L5H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong self-attention for first token
        if i == 0:
            attention[i, i] = 1.0
            continue
            
        # Base weights for different positions
        weights = {}
        
        # Very strong attention to first token (decreases slightly with distance)
        first_token_weight = 0.7 - 0.1 * min(i / 10.0, 0.3)
        weights[0] = first_token_weight
        
        # Moderate attention to previous token
        if i > 0:
            prev_weight = 0.15 + 0.05 * np.random.random()
            weights[i-1] = weights.get(i-1, 0) + prev_weight
        
        # Self attention
        self_weight = 0.08 + 0.04 * np.random.random()
        weights[i] = weights.get(i, 0) + self_weight
        
        # Additional patterns based on linguistic analysis
        if alignment[i]:  # If token aligns to spacy tokens
            spacy_idx = alignment[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Verbs attend more strongly to subjects/early tokens
                if spacy_token.pos_ in ['VERB', 'AUX']:
                    for j in range(min(i, 3)):  # First few tokens
                        weights[j] = weights.get(j, 0) + 0.1
                
                # Function words attend to nearby content
                if spacy_token.pos_ in ['DET', 'PREP', 'CONJ']:
                    for j in range(max(0, i-3), i):
                        if j in weights:
                            weights[j] += 0.05
        
        # Add some attention to tokens 2-3 positions back
        for offset in [2, 3]:
            if i >= offset:
                back_weight = 0.03 + 0.02 * np.random.random()
                weights[i-offset] = weights.get(i-offset, 0) + back_weight
        
        # Normalize and assign
        total_weight = sum(weights.values())
        if total_weight > 0:
            for j, w in weights.items():
                attention[i, j] = w / total_weight
        else:
            attention[i, i] = 1.0
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 3
def first_token_bias_content_focus_punctuation_L5H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Helper to check if a token is punctuation or conjunction
    def is_structural_token(token_text):
        return token_text.strip() in [',', '.', '!', '?', ';', ':', 'and', 'or', 'but']
    
    for i in range(n):
        # Strong attention to first token (very consistent pattern)
        if i > 0:
            attention[i, 0] = 0.6
        
        # Self attention (moderate for all tokens)
        attention[i, i] = 0.15
        
        # Recent token attention (especially immediate predecessor)
        if i > 0:
            attention[i, i-1] = 0.3
        if i > 1:
            attention[i, i-2] = 0.15
        
        # Attention to structural tokens (punctuation, conjunctions) within window
        for j in range(max(0, i-5), i):
            if is_structural_token(tokens[j]):
                attention[i, j] += 0.2
        
        # Additional attention patterns based on token content
        current_token = tokens[i].strip()
        
        # Function words tend to attend more to recent content words
        if current_token.lower() in ['the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'by']:
            for j in range(max(0, i-3), i):
                if tokens[j].strip() and not tokens[j].strip().lower() in ['the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'by']:
                    attention[i, j] += 0.1
        
        # Verbs tend to attend to earlier content
        spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
        is_verb = False
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ == 'VERB':
                is_verb = True
                break
        
        if is_verb:
            # Verbs attend more to subjects and earlier content
            for j in range(max(0, i-4), i):
                attention[i, j] += 0.08
    
    # Apply causal mask (no future attention)
    attention = apply_causal_mask(attention)
    
    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 4
def first_token_bias_content_focus_punctuation_L5H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention pattern
        if i == 0:
            attention[i, 0] = 1.0
        else:
            # Base attention to first token (very strong pattern observed)
            attention[i, 0] = 0.8 - 0.1 * min(i, 6)  # Decay with distance but stay strong
            
            # Previous token attention (local context)
            attention[i, i-1] = 0.4 - 0.05 * min(i, 5)
            
            # Self attention (moderate)
            attention[i, i] = 0.1
            
            # Attention to tokens 2-4 positions back (local window)
            for j in range(max(0, i-4), i-1):
                if j != 0:  # Don't double-count first token
                    distance = i - j
                    attention[i, j] = max(0.05, 0.2 / distance)
            
            # Special patterns for final tokens (punctuation, sentence endings)
            if i >= n - 2:  # Last two tokens
                # Find important content words (verbs, nouns)
                for j in range(i):
                    if gpt2_to_spacy[j]:
                        spacy_idx = gpt2_to_spacy[j][0]
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            if spacy_token.pos_ in ['VERB', 'NOUN'] and j > 0:
                                attention[i, j] += 0.3
            
            # Boost attention for certain positional patterns
            # Early tokens (1-3) get extra attention from later tokens
            if i > 3:
                for j in range(1, min(4, i)):
                    attention[i, j] += 0.1
                    
            # Comma and punctuation patterns
            token_text = tokens[i].strip()
            if token_text in [',', '.', '!', '?']:
                # Punctuation attends more to recent content
                for j in range(max(0, i-3), i):
                    if j > 0 and tokens[j].strip() not in [',', '.', '!', '?']:
                        attention[i, j] += 0.2
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 5
def first_token_bias_content_focus_L5H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Very strong attention to first token (dominant pattern)
        attention[i, 0] = 0.95
        
        # Self-attention
        attention[i, i] = 0.03
        
        # Small amount of attention to nearby content words
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            current_spacy = doc[spacy_indices[0]]
            
            # Look for nearby content words to attend to weakly
            for j in range(max(0, i-3), i):
                if j == 0 or j == i:  # Skip first token and self (already handled)
                    continue
                    
                j_spacy_indices = gpt2_to_spacy[j]
                if j_spacy_indices:
                    j_spacy = doc[j_spacy_indices[0]]
                    
                    # Weak attention to nearby nouns, verbs, adjectives
                    if j_spacy.pos_ in ['NOUN', 'VERB', 'ADJ'] and current_spacy.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        attention[i, j] = 0.01
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 6
def decaying_first_token_bias_content_focus_L5H6(sentence: str, tokenizer: PreTrainedTokenizerBase):
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for syntactic information
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token
        attention_matrix[i, 0] = 0.7
        
        # Attention to previous token
        if i > 0:
            attention_matrix[i, i-1] = 0.4
            
        # Self attention
        attention_matrix[i, i] = 0.1
        
        # Look for syntactic relationships
        if gpt2_to_spacy[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_idx = gpt2_to_spacy[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # If this is a preposition, attend to its object
                if spacy_token.pos_ == "ADP" and spacy_token.head != spacy_token:
                    for j in range(n):
                        if j < i and gpt2_to_spacy[j]:
                            target_spacy_idx = gpt2_to_spacy[j][0]
                            if target_spacy_idx < len(doc) and doc[target_spacy_idx] == spacy_token.head:
                                attention_matrix[i, j] = 0.3
                
                # If this token has a head, attend to it
                if spacy_token.head != spacy_token:
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            target_spacy_idx = gpt2_to_spacy[j][0]
                            if target_spacy_idx < len(doc) and doc[target_spacy_idx] == spacy_token.head:
                                attention_matrix[i, j] = 0.2
                
                # Special case for articles/determiners - attend to their noun
                if spacy_token.pos_ in ["DET", "ADP"]:
                    for child in spacy_token.children:
                        if child.pos_ in ["NOUN", "PROPN"]:
                            for j in range(i+1, n):
                                if j < n and gpt2_to_spacy[j]:
                                    target_spacy_idx = gpt2_to_spacy[j][0]
                                    if target_spacy_idx < len(doc) and doc[target_spacy_idx] == child:
                                        attention_matrix[i, j] = 0.3
        
        # Add some decay for distant tokens
        for j in range(i):
            if j > 0 and attention_matrix[i, j] == 0:
                distance_decay = max(0, 0.05 - 0.01 * (i - j))
                attention_matrix[i, j] = distance_decay
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 5, Head 7
def first_token_bias_content_focus_punctuation_L5H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (except from first token itself)
        if i > 0:
            attention[i, 0] = 0.7
        else:
            attention[i, 0] = 0.4  # First token attends to itself strongly
        
        # Self-attention
        attention[i, i] = 0.15
        
        # Recency bias - attend to previous few tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                if distance == 1:  # Previous token
                    attention[i, j] = 0.2
                elif distance == 2:
                    attention[i, j] = 0.1
                elif distance == 3:
                    attention[i, j] = 0.05
        
        # Syntactic relationships using spacy
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Attend to head of current token
            if spacy_token.head != spacy_token:
                head_idx = spacy_token.head.i
                # Find corresponding GPT2 tokens
                for j in range(min(i+1, n)):  # Only look at previous tokens (causal)
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices and head_idx in j_spacy_indices:
                        attention[i, j] += 0.15
                        break
            
            # If this is a verb, attend to its direct object
            if spacy_token.pos_ == "VERB":
                for child in spacy_token.children:
                    if child.dep_ == "dobj":
                        child_idx = child.i
                        for j in range(min(i+1, n)):
                            j_spacy_indices = gpt2_to_spacy[j]
                            if j_spacy_indices and child_idx in j_spacy_indices:
                                attention[i, j] += 0.1
                                break
            
            # If this is a modifier, attend to what it modifies
            if spacy_token.dep_ in ["amod", "advmod", "compound"]:
                head_idx = spacy_token.head.i
                for j in range(min(i+1, n)):
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices and head_idx in j_spacy_indices:
                        attention[i, j] += 0.2
                        break
        
        # Special handling for punctuation - attend to nearby content words
        if tokens[i] in ['.', '!', '?', ',', ';', ':']:
            # Attend less to first token for punctuation
            attention[i, 0] = 0.4
            # Attend more to recent content
            for j in range(max(0, i-5), i):
                if j != 0 and tokens[j].strip() and tokens[j] not in ['.', '!', '?', ',', ';', ':']:
                    attention[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 8
def decaying_first_token_bias_punctuation_L5H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention scores
        scores = np.zeros(n)
        
        # Strong first-token bias - most tokens attend heavily to position 0
        if i > 0:
            scores[0] = 0.8
        
        # Self-attention
        scores[i] = 0.1
        
        # Find punctuation tokens that act as attractors
        for j in range(i + 1):
            token_text = tokens[j].strip()
            
            # High attention to quotation marks
            if '"' in token_text or "'" in token_text:
                scores[j] += 0.3
            
            # Attention to commas
            elif token_text == ',':
                scores[j] += 0.2
            
            # Attention to question/exclamation marks
            elif token_text in ['?', '!', '?"', '."']:
                scores[j] += 0.2
        
        # Find prepositions that create attention hotspots
        for j in range(i + 1):
            if j < len(alignment) and alignment[j]:
                spacy_indices = alignment[j]
                for spacy_idx in spacy_indices:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        # If this is a preposition, following tokens attend to it
                        if spacy_token.pos_ == 'ADP' and tokens[j].strip().lower() in ['from', 'in', 'on', 'at', 'to', 'with']:
                            if i > j:
                                scores[j] += 0.4
        
        # Attention to coordinating conjunctions
        for j in range(i + 1):
            if tokens[j].strip().lower() in ['and', 'or', 'but']:
                if i > j:
                    scores[j] += 0.2
        
        # Positional decay for nearby tokens
        for j in range(max(0, i-3), i):
            scores[j] += 0.05 * (1.0 - (i-j) * 0.2)
        
        # Add some attention to sentence boundaries and special tokens
        for j in range(i + 1):
            token_text = tokens[j].strip()
            if token_text in ['.', '\n']:
                scores[j] += 0.1
        
        # Normalize to avoid extreme values
        scores = np.maximum(scores, 0.01)  # Minimum attention
        
        attention[i] = scores
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 5, Head 9
def first_token_bias_content_focus_L5H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify content words (nouns, verbs, adjectives)
    content_word_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        for spacy_idx in gpt2_to_spacy[i]:
            if spacy_idx < len(doc):
                pos = doc[spacy_idx].pos_
                if pos in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    content_word_mask[i] = True
                    break
    
    for i in range(n):
        # Very strong attention to first token
        attention[i, 0] = 0.85
        
        if i == 0:
            # First token attends to itself
            attention[i, i] = 1.0
        else:
            # Distribute remaining attention
            remaining = 0.15
            
            # Self attention (moderate)
            self_weight = 0.04
            attention[i, i] = self_weight
            remaining -= self_weight
            
            # Previous token attention (small)
            if i > 0:
                prev_weight = 0.03
                attention[i, i-1] += prev_weight
                remaining -= prev_weight
            
            # Distribute remaining attention among available tokens
            if remaining > 0:
                available_tokens = list(range(i + 1))  # Causal mask
                available_tokens.remove(0)  # Already handled first token
                if i in available_tokens:
                    available_tokens.remove(i)  # Already handled self
                if i > 0 and (i-1) in available_tokens:
                    available_tokens.remove(i-1)  # Already handled previous
                
                if available_tokens:
                    # Prefer content words
                    weights = np.ones(len(available_tokens))
                    for idx, token_idx in enumerate(available_tokens):
                        if content_word_mask[token_idx]:
                            weights[idx] *= 2.0  # Boost content words
                    
                    # Normalize and apply
                    weights = weights / weights.sum() * remaining
                    for idx, token_idx in enumerate(available_tokens):
                        attention[i, token_idx] += weights[idx]
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 10
def first_token_bias_content_focus_punctuation_L5H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify important nouns (content words that should be attention hubs)
    important_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token_obj = doc[spacy_idx]
                # Key nouns and content words become attention hubs
                if token_obj.pos_ in ['NOUN', 'PROPN'] and not token_obj.is_stop:
                    important_tokens.add(i)
    
    for i in range(n):
        # Very strong attention to first token (dominant pattern)
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0
            
        # Self-attention
        attention[i, i] = 0.15
        
        # Attention to important nouns/content words
        for j in important_tokens:
            if j <= i and j != 0:  # Respect causal mask, don't double-count first token
                attention[i, j] = 0.3
                
        # Local context attention (previous few tokens)
        for j in range(max(0, i-3), i):
            if j != 0 and j not in important_tokens:  # Don't double-count
                attention[i, j] = 0.05
                
        # Boost attention to punctuation and function words in local context
        for j in range(max(0, i-2), i):
            if j < len(tokens):
                token_text = tokens[j].strip()
                if token_text in [',', '.', '!', '?', '"', "'", 'the', 'a', 'an', 'and', 'or', 'but']:
                    attention[i, j] += 0.03
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 5, Head 11
def decaying_first_token_bias_content_focus_punctuation_L5H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (decays with distance from start)
        first_token_weight = 0.9 if i <= 3 else max(0.6, 0.9 - (i - 3) * 0.05)
        attention[i, 0] = first_token_weight
        
        # Self-attention
        self_weight = 0.15 if i == 0 else 0.08
        attention[i, i] = self_weight
        
        # Local attention to previous tokens
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
        
        # Special patterns for punctuation and conjunctions
        token = tokens[i]
        if token in [',', '.', '!', '?', '"']:
            # Punctuation attends more to nearby content words
            for j in range(max(0, i-5), i):
                if j == 0:
                    continue
                if tokens[j].strip() and not tokens[j] in [',', '.', '!', '?', '"']:
                    attention[i, j] += 0.02
        
        if token.strip() == 'and':
            # "and" gets extra self-attention
            attention[i, i] += 0.05
            
        # Boost attention from later tokens to "and"
        for j in range(i):
            if tokens[j].strip() == 'and':
                attention[i, j] += 0.03
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 6, Head 0
def decaying_first_token_bias_content_focus_L6H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_gpt2 = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base weights
        base_first_token = 0.3
        base_self = 0.1
        base_prev = 0.15
        base_decay = 0.05
        
        # First token gets strong attention
        attention[i, 0] = base_first_token
        
        # Self attention
        attention[i, i] = base_self
        
        # Previous token attention
        if i > 0:
            attention[i, i-1] += base_prev
        
        # Distance decay for nearby tokens
        for j in range(max(0, i-3), i):
            if j != i-1 and j != 0:  # Skip previous and first (already handled)
                distance = i - j
                attention[i, j] += base_decay / distance
        
        # Get spacy tokens for current position
        spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
        
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                
                # Verbs attend strongly to their subjects
                if token.pos_ == "VERB":
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            for gpt2_idx in spacy_to_gpt2[child.i]:
                                if gpt2_idx < i:
                                    attention[i, gpt2_idx] += 0.4
                    
                    # Verbs also attend to auxiliary verbs
                    for child in token.children:
                        if child.dep_ == "aux":
                            for gpt2_idx in spacy_to_gpt2[child.i]:
                                if gpt2_idx < i:
                                    attention[i, gpt2_idx] += 0.2
                
                # Nouns attend to their modifiers
                if token.pos_ == "NOUN":
                    for child in token.children:
                        if child.dep_ in ["amod", "det"]:
                            for gpt2_idx in spacy_to_gpt2[child.i]:
                                if gpt2_idx < i:
                                    attention[i, gpt2_idx] += 0.15
                
                # Adjectives attend to the nouns they modify
                if token.pos_ == "ADJ":
                    if token.head and token.head.pos_ == "NOUN":
                        for gpt2_idx in spacy_to_gpt2[token.head.i]:
                            if gpt2_idx < i:
                                attention[i, gpt2_idx] += 0.2
                
                # Prepositions attend to their objects
                if token.pos_ == "ADP":
                    for child in token.children:
                        if child.dep_ == "pobj":
                            for gpt2_idx in spacy_to_gpt2[child.i]:
                                if gpt2_idx < i:
                                    attention[i, gpt2_idx] += 0.15
                
                # Pronouns attend to potential antecedents (earlier nouns)
                if token.pos_ == "PRON":
                    for j in range(i):
                        spacy_j = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                        for sj in spacy_j:
                            if sj < len(doc) and doc[sj].pos_ == "NOUN":
                                attention[i, j] += 0.1
                
                # Particles and adverbs attend to nearby verbs
                if token.pos_ in ["PART", "ADV"]:
                    if token.head and token.head.pos_ == "VERB":
                        for gpt2_idx in spacy_to_gpt2[token.head.i]:
                            if gpt2_idx < i:
                                attention[i, gpt2_idx] += 0.25
        
        # Special handling for common patterns based on token text
        token_text = tokens[i].lower().strip()
        
        # "to" tokens (infinitives) attend to preceding verbs
        if token_text == "to" and i > 0:
            for j in range(max(0, i-4), i):
                spacy_j = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                for sj in spacy_j:
                    if sj < len(doc) and doc[sj].pos_ == "VERB":
                        attention[i, j] += 0.3
        
        # Punctuation attends to recent important tokens
        if token_text in [".", ",", "!"]:
            for j in range(max(0, i-5), i):
                spacy_j = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                for sj in spacy_j:
                    if sj < len(doc) and doc[sj].pos_ in ["VERB", "NOUN"]:
                        attention[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 6, Head 1
def decaying_content_focus_L6H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Get spacy analysis for semantic relationships
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
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
    
    # Add special case: conjunctions and function words attending to nearby content words
    for i in range(n):
        token_text = tokens[i].strip().lower()
        
        # Check if current token is a conjunction or similar function word
        if token_text in ['and', 'but', 'or', ',']:
            # Look for content words (adjectives, verbs, nouns) in recent context
            for j in range(max(0, i-4), i):  # Look back up to 4 tokens
                if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                    spacy_idx = gpt2_to_spacy[j][0]
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        # If it's a content word (adj, verb, noun), boost attention
                        if spacy_token.pos_ in ['ADJ', 'VERB', 'NOUN']:
                            # Strong attention to nearby content words
                            distance = i - j
                            if distance == 1:
                                attention[i, j] = 0.6  # Very strong for adjacent
                            elif distance == 2:
                                attention[i, j] = 0.3  # Strong for distance 2
                            else:
                                attention[i, j] = 0.15  # Moderate for further
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_content_focus_L6", attention


# Layer 6, Head 2
def first_token_bias_content_focus_punctuation_L6H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention distribution
        base_weights = np.zeros(i + 1)  # Can only attend to tokens <= i
        
        if i == 0:
            # First token attends to itself
            base_weights[0] = 1.0
        else:
            # Strong first-token bias (especially for early positions)
            first_token_weight = 0.9 if i <= 3 else max(0.3, 0.8 - 0.1 * i)
            
            # Self-attention component
            self_weight = 0.03
            
            # Recent context bias
            recent_weight = 0.05
            
            # Distribute remaining weight
            remaining = 1.0 - first_token_weight - self_weight - recent_weight
            
            # First token gets strong weight
            base_weights[0] = first_token_weight
            
            # Self attention
            base_weights[i] = self_weight
            
            # Recent important tokens (within last 3-5 positions)
            recent_start = max(1, i - 4)
            recent_positions = list(range(recent_start, i))
            
            # Look for important tokens to attend to
            important_positions = []
            
            # Check if current token's spacy alignment gives us syntactic info
            current_spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
            
            for j in range(1, i):
                token_text = tokens[j].strip()
                
                # High attention to verbs, conjunctions, and punctuation
                j_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                
                is_important = False
                if j_spacy_indices:
                    spacy_token = doc[j_spacy_indices[0]]
                    if spacy_token.pos_ in ['VERB', 'AUX', 'CCONJ'] or token_text in [',', '.', '?', '!', '"']:
                        is_important = True
                
                # Also boost attention to tokens that are syntactically related
                if current_spacy_indices and j_spacy_indices:
                    current_spacy_token = doc[current_spacy_indices[0]]
                    j_spacy_token = doc[j_spacy_indices[0]]
                    
                    # Attend to syntactic heads
                    if j_spacy_token in [current_spacy_token.head] + list(current_spacy_token.ancestors):
                        is_important = True
                
                if is_important:
                    important_positions.append(j)
            
            # Distribute recent weight among important recent positions
            if recent_positions:
                recent_per_pos = recent_weight / len(recent_positions)
                for pos in recent_positions:
                    if pos in important_positions:
                        base_weights[pos] += recent_per_pos * 2  # Boost important tokens
                    else:
                        base_weights[pos] += recent_per_pos * 0.5
            
            # Distribute remaining weight uniformly among all available positions
            uniform_weight = max(0, remaining) / (i + 1)
            base_weights += uniform_weight
        
        attention[i, :i+1] = base_weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 6, Head 3
def first_token_bias_content_focus_L6H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (dominant pattern)
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly
        
        # Self-attention (moderate)
        if i > 0:
            attention[i, i] = 0.3
        
        # Attention to previous token (local context)
        if i > 1:
            attention[i, i-1] = 0.2
        
        # For tokens near the end, add attention to content words
        if i >= n // 2:  # Second half of sentence
            for j in range(1, i):
                # Add small attention to intermediate tokens
                attention[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 6, Head 4
def first_token_bias_content_focus_L6H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify important entities/subjects
    important_tokens = set()
    for spacy_token in doc:
        # Add subjects, proper nouns, and important entities
        if (spacy_token.dep_ in ["nsubj", "nsubjpass"] or 
            spacy_token.pos_ in ["PROPN"] or
            spacy_token.ent_type_ in ["PERSON", "ORG", "GPE"]):
            # Find corresponding GPT2 tokens
            for i, spacy_indices in enumerate(gpt2_to_spacy):
                if spacy_token.i in spacy_indices:
                    important_tokens.add(i)
    
    # NEW: Identify key verbs for longer sentences
    key_verbs = set()
    if n > 15:  # Only for longer sentences where this pattern matters
        for spacy_token in doc:
            # Look for main verbs, especially past tense narrative verbs
            if (spacy_token.pos_ == "VERB" and 
                spacy_token.dep_ in ["ROOT", "conj"] and
                spacy_token.tag_ in ["VBD", "VBN"]):  # Past tense/participle verbs
                # Find corresponding GPT2 tokens
                for i, spacy_indices in enumerate(gpt2_to_spacy):
                    if spacy_token.i in spacy_indices:
                        key_verbs.add(i)
    
    for i in range(n):
        # Strong attention to first token
        attention[i, 0] = 0.8 if i > 0 else 1.0
        
        # Self attention
        if i > 0:
            attention[i, i] = 0.15
        
        # Attention to important entities/subjects
        for important_idx in important_tokens:
            if important_idx <= i and important_idx != 0:
                attention[i, important_idx] = 0.3
        
        # NEW: Attention to key verbs in longer sentences
        for verb_idx in key_verbs:
            if verb_idx <= i and verb_idx != 0 and verb_idx not in important_tokens:
                attention[i, verb_idx] = 0.25
        
        # Some attention to recent tokens (positional bias)
        for j in range(max(0, i-3), i):
            if j != 0 and j not in important_tokens and j not in key_verbs:
                # Get spacy info for this token
                spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                is_content_word = False
                if spacy_indices:
                    spacy_token = doc[spacy_indices[0]]
                    is_content_word = spacy_token.pos_ in ["VERB", "NOUN", "ADJ"]
                
                if is_content_word:
                    attention[i, j] = 0.1
                else:
                    attention[i, j] = 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 6, Head 5
def first_token_bias_content_focus_stochastic_L6H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Get spacy parse and alignment
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (except for first token itself)
        if i > 0:
            attention[i, 0] = 0.85
        
        # Self attention
        attention[i, i] = 0.06
        
        # Previous token attention
        if i > 0:
            attention[i, i-1] = 0.04
        
        # Handle specific token patterns
        token = tokens[i].strip()
        
        # Punctuation patterns
        if token in ['.', '!', '?']:
            # Periods attend more to nearby content
            for j in range(max(0, i-3), i):
                if tokens[j].strip() not in [',', '.', '!', '?', '"', "'", '(', ')']:
                    attention[i, j] += 0.02
        
        elif token in [',', ',"', ',"']:
            # Commas attend to previous content word
            for j in range(i-1, max(-1, i-4), -1):
                if tokens[j].strip() not in [',', '.', '!', '?', '"', "'", '(', ')']:
                    attention[i, j] += 0.03
                    break
        
        elif token.startswith('"') and i > 0:
            # Quote tokens attend to first token less, more to context
            attention[i, 0] = 0.7
            
        # Content words attend to related words
        if i > 0 and gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0] if gpt2_to_spacy[i] else None
            if spacy_idx is not None and spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Verbs attend to subjects/objects
                if spacy_token.pos_ == 'VERB':
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                if other_token.pos_ in ['NOUN', 'PRON'] and other_token.dep_ in ['nsubj', 'dobj']:
                                    attention[i, j] += 0.03
                
                # Adjectives attend to nouns they modify
                elif spacy_token.pos_ == 'ADJ':
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                if other_token.pos_ == 'NOUN' and abs(other_spacy_idx - spacy_idx) <= 2:
                                    attention[i, j] += 0.02
        
        # NEW: Enhanced verb-complement attention patterns
        if i > 0 and gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0] if gpt2_to_spacy[i] else None
            if spacy_idx is not None and spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Auxiliary verbs attend strongly to main verbs and their complements
                if spacy_token.pos_ == 'AUX' or (spacy_token.pos_ == 'VERB' and spacy_token.dep_ == 'aux'):
                    for j in range(max(0, i-6), i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                # Strong attention to main verbs, complements, or objects
                                if (other_token.pos_ == 'VERB' and other_token.dep_ in ['ROOT', 'ccomp', 'xcomp']) or \
                                   (other_token.dep_ in ['dobj', 'attr', 'ccomp']):
                                    attention[i, j] += 0.05
                
                # Main verbs attend to their complements and clausal objects
                elif spacy_token.pos_ == 'VERB' and spacy_token.dep_ in ['ROOT', 'ccomp']:
                    for j in range(max(0, i-5), i):
                        if gpt2_to_spacy[j]:
                            other_spacy_idx = gpt2_to_spacy[j][0]
                            if other_spacy_idx < len(doc):
                                other_token = doc[other_spacy_idx]
                                # Attend to complements, objects, or related verbs
                                if other_token.dep_ in ['dobj', 'ccomp', 'xcomp', 'nsubj'] or \
                                   (other_token.pos_ == 'VERB' and abs(other_spacy_idx - spacy_idx) <= 3):
                                    attention[i, j] += 0.04
        
        # Add small random variations to nearby tokens
        for j in range(max(0, i-3), i):
            attention[i, j] += np.random.uniform(0.01, 0.025)
    
    # Special case for first token - full self attention
    if n > 0:
        attention[0, 0] = 1.0
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 6, Head 6
def decaying_first_token_bias_stochastic_L6H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Get spacy parse for syntactic features
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    doc = spacy_parse(sentence)
    
    for i in range(n):
        # Strong attention to first token (BOS)
        if i == 0:
            attention[i, 0] = 1.0
        else:
            # High attention to first token
            attention[i, 0] = 0.85 + 0.1 * np.random.random()
            
            # Self-attention with moderate weight
            attention[i, i] = 0.05 + 0.05 * np.random.random()
            
            # Attention to previous token
            if i > 0:
                attention[i, i-1] = 0.02 + 0.03 * np.random.random()
            
            # Small attention to other previous tokens with decay
            for j in range(1, min(i, 5)):  # Look back up to 5 tokens
                if i - j > 0:
                    decay_factor = 0.5 ** j
                    attention[i, i-j] += 0.01 * decay_factor * np.random.random()
            
            # ADD: Syntactic relationship attention
            if gpt2_to_spacy[i]:  # If this GPT2 token aligns to spacy tokens
                for spacy_idx in gpt2_to_spacy[i]:
                    if spacy_idx < len(doc):
                        spacy_token = doc[spacy_idx]
                        
                        # Look for syntactic heads/children
                        syntactic_targets = []
                        
                        # Add head if it exists
                        if spacy_token.head != spacy_token:
                            syntactic_targets.append(spacy_token.head)
                        
                        # Add direct objects, prepositional objects, adjectival modifiers
                        for child in spacy_token.children:
                            if child.dep_ in ["dobj", "pobj", "amod"]:
                                syntactic_targets.append(child)
                        
                        # Convert spacy targets back to GPT2 indices and add attention
                        for target in syntactic_targets:
                            target_gpt2_indices = []
                            for gpt2_idx in range(i):  # Only look at previous tokens (causal)
                                if gpt2_to_spacy[gpt2_idx]:
                                    for target_spacy_idx in gpt2_to_spacy[gpt2_idx]:
                                        if target_spacy_idx < len(doc) and doc[target_spacy_idx] == target:
                                            target_gpt2_indices.append(gpt2_idx)
                            
                            # Add syntactic attention
                            for target_idx in target_gpt2_indices:
                                attention[i, target_idx] += 0.03 + 0.02 * np.random.random()
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 6, Head 7
def first_token_bias_content_focus_punctuation_L6H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_gpt2 = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token for early positions
        if i < 4:
            attention[i, 0] = 0.9 - (i * 0.1)
        else:
            attention[i, 0] = 0.1
        
        # Self-attention
        attention[i, i] = 0.1
        
        # Get spacy token info if available
        spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
        
        if spacy_indices:
            spacy_tok = doc[spacy_indices[0]]
            
            # Find syntactic head
            if spacy_tok.head != spacy_tok:
                head_gpt2_indices = []
                for spacy_idx in range(len(doc)):
                    if doc[spacy_idx] == spacy_tok.head:
                        head_gpt2_indices = spacy_to_gpt2[spacy_idx]
                        break
                
                for head_idx in head_gpt2_indices:
                    if head_idx <= i:
                        attention[i, head_idx] = 0.3
            
            # Attend to recent content words
            for j in range(max(0, i-5), i):
                j_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                if j_spacy_indices:
                    j_spacy_tok = doc[j_spacy_indices[0]]
                    if j_spacy_tok.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        distance = i - j
                        attention[i, j] = max(0.05, 0.2 - distance * 0.03)
        
        # Special handling for punctuation
        if i == n - 1 and tokens[i] in ['.', '!', '?']:
            # Final punctuation attends to last content word
            for j in range(i-1, max(0, i-6), -1):
                j_spacy_indices = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                if j_spacy_indices:
                    j_spacy_tok = doc[j_spacy_indices[0]]
                    if j_spacy_tok.pos_ in ['NOUN', 'VERB']:
                        attention[i, j] = 0.4
                        break
        
        # Add some general recency bias
        for j in range(max(0, i-3), i):
            if attention[i, j] < 0.05:
                attention[i, j] = 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 6, Head 8
def first_token_bias_L6H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (very consistent pattern)
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0  # Self-attention for first token
        
        # Self-attention (moderate)
        if i > 0:
            attention[i, i] = 0.15
        
        # Detect subword continuation pattern (BPE tokens that continue previous token)
        if i > 0:
            current_token = tokens[i]
            prev_token = tokens[i-1]
            # Check if current token looks like a continuation (doesn't start with space and previous doesn't end with space)
            if (not current_token.startswith(' ') and not prev_token.endswith(' ') and 
                len(current_token) > 0 and len(prev_token) > 0):
                # This looks like a subword continuation, boost attention to previous token
                attention[i, i-1] += 0.4
        
        # Get spacy token info for current GPT2 token
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            current_spacy = doc[spacy_indices[0]]
            
            # Syntactic dependency attention
            # Attend to syntactic head
            if current_spacy.head != current_spacy and current_spacy.head.i < len(doc):
                head_idx = current_spacy.head.i
                # Find GPT2 tokens that overlap with this spacy token
                for j in range(i):
                    j_spacy = gpt2_to_spacy[j]
                    if j_spacy and head_idx in j_spacy:
                        attention[i, j] += 0.4
            
            # Special patterns for specific dependencies
            # Prepositions attend strongly to their objects' heads
            if current_spacy.pos_ == "ADP":  # preposition
                for child in current_spacy.children:
                    if child.dep_ == "pobj":  # prepositional object
                        child_head_idx = child.i
                        for j in range(i):
                            j_spacy = gpt2_to_spacy[j]
                            if j_spacy and child_head_idx in j_spacy:
                                attention[i, j] += 0.3
            
            # Objects attend to their verbs
            if current_spacy.dep_ in ["dobj", "pobj", "iobj"]:
                head_idx = current_spacy.head.i
                for j in range(i):
                    j_spacy = gpt2_to_spacy[j]
                    if j_spacy and head_idx in j_spacy:
                        attention[i, j] += 0.5
            
            # Adjectives attend to their head nouns
            if current_spacy.pos_ == "ADJ" and current_spacy.dep_ == "amod":
                head_idx = current_spacy.head.i
                for j in range(i):
                    j_spacy = gpt2_to_spacy[j]
                    if j_spacy and head_idx in j_spacy:
                        attention[i, j] += 0.3
        
        # Local attention to previous token (moderate)
        if i > 0:
            attention[i, i-1] += 0.2
        
        # Decay attention to nearby tokens
        for j in range(max(0, i-3), i):
            if j != i-1 and j != 0:  # Don't double-count previous token or first token
                attention[i, j] += 0.1 * (1.0 - (i-j) * 0.2)
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L6H8", attention


# Layer 6, Head 9
def first_token_bias_L6H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            # First token attends only to itself
            attention_matrix[i, 0] = 1.0
        else:
            # All other tokens attend primarily to first token with small self-attention
            attention_matrix[i, 0] = 0.99  # Strong attention to first token
            attention_matrix[i, i] = 0.01  # Small self-attention
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L6H9", attention_matrix


# Layer 6, Head 10
def first_token_bias_punctuation_L6H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Very strong attention to first token (position 0)
        if i > 0:
            attention[i, 0] = 0.9
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly
            
        # Self-attention (moderate)
        if i > 0:
            attention[i, i] = 0.04
            
        # Distribute remaining weight to nearby tokens and punctuation
        remaining_weight = 1.0 - attention[i].sum()
        
        if remaining_weight > 0 and i > 0:
            # Small weights to other accessible positions
            accessible_positions = list(range(1, i))  # Exclude position 0 and self
            
            if accessible_positions:
                # Give slightly more weight to recent tokens and punctuation
                weights = np.ones(len(accessible_positions)) * 0.01
                
                # Boost punctuation and recent positions
                for idx, pos in enumerate(accessible_positions):
                    token = tokens[pos]
                    if token in [',', '.', ';', ':', '!', '?']:
                        weights[idx] *= 1.5
                    elif pos >= i - 3:  # Recent tokens
                        weights[idx] *= 1.2
                
                # Normalize to use remaining weight
                if weights.sum() > 0:
                    weights = weights * (remaining_weight / weights.sum())
                    
                for idx, pos in enumerate(accessible_positions):
                    attention[i, pos] = weights[idx]
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 6, Head 11
def first_token_bias_content_focus_L6H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Find "time" token if it exists
    time_token_idx = None
    for i, token in enumerate(tokens):
        if token.strip().lower() == "time":
            time_token_idx = i
            break
    
    # Find comma positions
    comma_positions = []
    for i, token in enumerate(tokens):
        if token.strip() == ",":
            comma_positions.append(i)
    
    for i in range(n):
        # Self-attention (moderate weight)
        attention_matrix[i, i] = 0.1
        
        # Strong attention to first token (except for first token itself)
        if i > 0:
            attention_matrix[i, 0] = 0.6
        else:
            attention_matrix[i, i] = 1.0  # First token attends to itself strongly
        
        # Attention to previous token
        if i > 0:
            attention_matrix[i, i-1] = 0.3
            
        # Additional attention patterns based on position and content
        if i > 1:
            # Some attention to token 2 positions back
            attention_matrix[i, i-2] = 0.05
            
        # Distribute some attention to other earlier tokens
        for j in range(max(0, i-5), i):
            if j != i-1 and j != 0:  # Don't double-count prev token and first token
                attention_matrix[i, j] += 0.02
        
        # Special case: Strong attention to "time" token for tokens after comma in narrative contexts
        if time_token_idx is not None and comma_positions:
            # Check if current token is after a comma that comes after "time"
            for comma_pos in comma_positions:
                if comma_pos > time_token_idx and i > comma_pos:
                    # Boost attention to "time" token significantly
                    attention_matrix[i, time_token_idx] = 0.4
                    # Reduce attention to first token to compensate
                    if i > 0:
                        attention_matrix[i, 0] = 0.3
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_content_", attention_matrix


# Layer 7, Head 0
def first_token_bias_content_focus_L7H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (0.6-0.9+ range)
        if i == 0:
            attention[i, 0] = 1.0  # Self-attention for first token
        else:
            attention[i, 0] = 0.7  # Strong first-token bias
        
        # Self-attention (moderate weight)
        if i > 0:
            attention[i, i] = 0.15
        
        # Previous token attention (local context)
        if i > 0:
            attention[i, i-1] = 0.1
        
        # NEW: Strong determiner-to-preceding-verb/preposition pattern
        if gpt2_to_spacy[i]:
            spacy_indices = gpt2_to_spacy[i]
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.pos_ == "DET" and i > 0:
                        # Check if previous token is a verb or preposition
                        prev_spacy_indices = gpt2_to_spacy[i-1] if i-1 < len(gpt2_to_spacy) else []
                        for prev_spacy_idx in prev_spacy_indices:
                            if prev_spacy_idx < len(doc):
                                prev_spacy_token = doc[prev_spacy_idx]
                                if prev_spacy_token.pos_ in ["VERB", "ADP"]:
                                    attention[i, i-1] += 0.4  # Strong boost for det->verb/prep
        
        # Look for syntactic dependencies using spacy
        if gpt2_to_spacy[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_indices = gpt2_to_spacy[i]
            for spacy_idx in spacy_indices:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    
                    # Head dependency (attend to syntactic head)
                    if spacy_token.head != spacy_token:  # Not root
                        head_text = spacy_token.head.text
                        # Find GPT2 tokens that might correspond to the head
                        for j in range(i):
                            if tokens[j].strip() == head_text or tokens[j].strip().lower() == head_text.lower():
                                attention[i, j] += 0.2
                    
                    # Special patterns for prepositions and their objects
                    if spacy_token.dep_ == "pobj" and spacy_token.head.pos_ == "ADP":
                        # Object of preposition attends to preposition
                        prep_text = spacy_token.head.text
                        for j in range(i):
                            if tokens[j].strip() == prep_text:
                                attention[i, j] += 0.3
                    
                    # Determiner-noun relationships
                    if spacy_token.pos_ == "NOUN":
                        for child in spacy_token.children:
                            if child.pos_ == "DET":
                                det_text = child.text
                                for j in range(i):
                                    if tokens[j].strip() == det_text:
                                        attention[i, j] += 0.15
    
    # Apply causal mask first
    attention = apply_causal_mask(attention)
    
    # Normalize to make row-stochastic
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 7, Head 1
def decaying_first_token_bias_content_focus_L7H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Get content word positions (nouns, verbs, adjectives)
    content_positions = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    content_positions.add(i)
    
    for i in range(n):
        # Strong attention to first token (dominant pattern)
        attention[i, 0] = 0.9
        
        # Self-attention (weaker)
        if i > 0:
            attention[i, i] = 0.05
        
        # Weak attention to other content words
        for j in range(min(i + 1, n)):
            if j != 0 and j != i and j in content_positions:
                attention[i, j] = 0.02
        
        # NEW: Add recency bias for longer sequences
        if n > 15:  # Only apply to longer sequences where this pattern is more important
            # Give extra attention to recent tokens (within last 5 positions)
            recent_window = min(5, i)
            for j in range(max(0, i - recent_window), i):
                if j != 0:  # Don't interfere with first-token attention
                    # Add extra weight that decays with distance
                    distance = i - j
                    extra_weight = 0.03 * (1.0 / distance)
                    attention[i, j] += extra_weight
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 7, Head 2
def first_token_bias_L7H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention_matrix = np.zeros((n, n))
    
    # First token attends to itself with weight 1.0
    attention_matrix[0, 0] = 1.0
    
    # All other tokens attend primarily to first token
    for i in range(1, n):
        # Strong attention to first token
        attention_matrix[i, 0] = 0.99
        
        # Small self-attention weight
        attention_matrix[i, i] = 0.01
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L7H2", attention_matrix


# Layer 7, Head 3
def decaying_first_token_bias_content_focus_L7H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Find subjects, main verbs, and other important tokens
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
                # Self-attention baseline
                attention[i, j] = 0.1
            elif j == 0:
                # Very strong first-token attention, especially for early tokens
                if i <= 3:
                    attention[i, j] = 0.9 - 0.1 * i
                else:
                    attention[i, j] = 0.4
            else:
                base_weight = 0.05
                
                # Check if current token should attend to subjects
                if spacy_indices:
                    current_spacy = spacy_indices[0]
                    current_token = doc[current_spacy] if current_spacy < len(doc) else None
                    
                    if current_token and current_token.pos_ == "VERB":
                        # Verbs attend strongly to subjects
                        target_spacy_indices = alignment[j] if j < len(alignment) else []
                        for target_idx in target_spacy_indices:
                            if target_idx in subjects:
                                base_weight += 0.3
                
                # Check if target is a subject or main verb
                target_spacy_indices = alignment[j] if j < len(alignment) else []
                for target_idx in target_spacy_indices:
                    if target_idx in subjects:
                        base_weight += 0.15
                    if target_idx in main_verbs:
                        base_weight += 0.1
                
                # Distance decay
                distance = i - j
                distance_factor = 1.0 / (1.0 + 0.1 * distance)
                
                # Recent token bias
                if distance <= 2:
                    base_weight += 0.05
                
                attention[i, j] = base_weight * distance_factor
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 7, Head 4
def first_token_bias_content_focus_punctuation_L7H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base attention - strong first token attention
        attention_matrix[i, 0] = 0.8
        
        # Self attention (moderate)
        attention_matrix[i, i] = 0.1
        
        # Previous token attention (moderate)
        if i > 0:
            attention_matrix[i, i-1] = 0.2
            
        # Special handling for punctuation tokens
        token = tokens[i]
        if token in ['.', '!', '?']:
            # End punctuation attends differently
            attention_matrix[i, 0] = 0.3  # Reduced first token attention
            
            # Find important content words to attend to
            for j in range(i):
                prev_token = tokens[j]
                # Attend to verbs, nouns, and clause boundaries
                if prev_token.strip() in ['said', 'was', 'wanted', 'liked', 'walked']:
                    attention_matrix[i, j] += 0.4
                elif prev_token in [',', '!', '"']:
                    attention_matrix[i, j] += 0.3
                elif j > 0 and prev_token.strip() and not prev_token.isspace():
                    attention_matrix[i, j] += 0.1
                    
        elif token == ',':
            # Commas have very strong first token attention
            attention_matrix[i, 0] = 0.9
            
        elif token.startswith('"') or token == '"':
            # Quotes have maximum first token attention
            attention_matrix[i, 0] = 1.0
            
        else:
            # Regular tokens - try to identify syntactic relationships
            spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
            
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                
                # If this is a verb, attend to its object/complement
                if spacy_token.pos_ == 'VERB':
                    for child in spacy_token.children:
                        if child.dep_ in ['dobj', 'pobj', 'comp']:
                            # Find corresponding GPT2 token
                            for j in range(i):
                                j_spacy = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                                if j_spacy and child.i in j_spacy:
                                    attention_matrix[i, j] += 0.3
                                    
                # If this is a noun, attend to its modifiers
                elif spacy_token.pos_ == 'NOUN':
                    for child in spacy_token.children:
                        if child.dep_ in ['amod', 'det']:
                            for j in range(i):
                                j_spacy = gpt2_to_spacy[j] if j < len(gpt2_to_spacy) else []
                                if j_spacy and child.i in j_spacy:
                                    attention_matrix[i, j] += 0.2
            
            # Add some recency bias for content words
            for j in range(max(0, i-3), i):
                if tokens[j].strip() and not tokens[j] in [',', '.', '!', '?', '"']:
                    attention_matrix[i, j] += 0.05
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_content_", attention_matrix


# Layer 7, Head 5
def first_token_bias_content_focus_L7H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Find verb positions in GPT2 tokens
    verb_positions = set()
    for i, spacy_indices in enumerate(alignment):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['VERB', 'AUX']:
                verb_positions.add(i)
    
    for i in range(n):
        # Strong attention to first token (position 0)
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0
        
        # Self-attention
        attention[i, i] = 0.1
        
        # Attention to nearby verbs
        for j in verb_positions:
            if j <= i and j != 0:  # Respect causal mask and not first token
                distance = i - j
                if distance <= 3:  # Local context
                    attention[i, j] = 0.2 / (1 + distance * 0.5)
        
        # Attention to immediately preceding token
        if i > 1:  # Not first or second token
            attention[i, i-1] = 0.15
        
        # For tokens near verbs, attend to surrounding context
        spacy_indices = alignment[i] if i < len(alignment) else []
        is_near_verb = any(spacy_idx < len(doc) and 
                          any(child.pos_ in ['VERB', 'AUX'] or child.head.pos_ in ['VERB', 'AUX']
                              for child in [doc[spacy_idx]] + list(doc[spacy_idx].children))
                          for spacy_idx in spacy_indices)
        
        if is_near_verb:
            # Attend more to recent context
            for j in range(max(0, i-3), i):
                if j not in verb_positions and j != 0:
                    attention[i, j] += 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 7, Head 6
def first_token_bias_content_focus_punctuation_stochastic_L7H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        token = tokens[i]
        
        # Strong first-token attention for early positions
        if i <= 3:
            attention_matrix[i, 0] = 0.9 + 0.1 * (4 - i) / 4
        else:
            # Decreasing first-token attention for later positions
            first_token_weight = max(0.02, 0.3 * np.exp(-0.3 * (i - 3)))
            attention_matrix[i, 0] = first_token_weight
        
        # Self-attention
        if i == 0:
            attention_matrix[i, i] = 1.0  # First token always attends to itself fully
        else:
            attention_matrix[i, i] = 0.05 + 0.03 * np.random.random()
        
        # Local context attention for non-first positions
        if i > 0:
            # Attend to previous token
            if i - 1 >= 0:
                attention_matrix[i, i-1] = 0.02 + 0.02 * np.random.random()
            
            # Attend to tokens 2-3 positions back with decreasing weight
            for j in range(max(0, i-3), i-1):
                if j != 0:  # Don't double-count first token
                    distance = i - j
                    weight = 0.01 * np.exp(-0.5 * distance) * (1 + 0.5 * np.random.random())
                    attention_matrix[i, j] = weight
        
        # Special patterns for punctuation and conjunctions
        if token in ['.', '!', '?', ',', ';', ':', '"']:
            # Punctuation attends more to content words
            for j in range(i):
                if tokens[j] not in [' ', '\n', '"', "'", ',', '.', '!', '?']:
                    attention_matrix[i, j] *= 1.5
        
        if token.strip().lower() in ['and', 'or', 'but', 'so', 'then']:
            # Conjunctions attend to previous clause elements
            for j in range(max(0, i-5), i):
                attention_matrix[i, j] *= 1.3
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_content_", attention_matrix


# Layer 7, Head 7
def decaying_first_token_bias_punctuation_L7H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token
        attention_matrix[i, 0] = 0.85 if i > 0 else 1.0
        
        # Self attention (moderate)
        if i > 0:
            attention_matrix[i, i] = 0.08
        
        # Distribute remaining attention to previous tokens
        remaining_weight = 1.0 - attention_matrix[i, :].sum()
        
        if i > 1 and remaining_weight > 0:
            # Get spacy info for current token if available
            spacy_indices = gpt2_to_spacy[i] if i < len(gpt2_to_spacy) else []
            current_token_text = tokens[i].strip().lower()
            
            # Calculate weights for each previous token (excluding 0 and i)
            weights = np.zeros(i)
            
            for j in range(1, i):
                if j == i:
                    continue
                    
                weight = 0.01  # base weight
                
                # Distance decay
                distance = i - j
                weight *= (1.0 / (1 + distance * 0.3))
                
                # Boost for conjunctions and function words
                prev_token_text = tokens[j].strip().lower()
                if prev_token_text in ['and', 'or', 'but', 'that', 'with', 'to', 'of', 'in']:
                    weight *= 2.0
                
                # Boost for punctuation attention patterns
                if tokens[i] == '.':
                    # Punctuation attends more broadly
                    weight *= 1.5
                
                # Boost if current token is a conjunction
                if current_token_text in ['and', 'or', 'but', 'then', 'finally']:
                    weight *= 1.2
                
                weights[j] = weight
            
            # Normalize and apply remaining weight
            if weights.sum() > 0:
                weights = weights * (remaining_weight / weights.sum())
                attention_matrix[i, 1:i] = weights[1:i]
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 7, Head 8
def decaying_first_token_bias_punctuation_L7H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token for most positions
        if i > 0:
            attention[i, 0] = 0.6
        
        # Self-attention
        attention[i, i] = 0.1
        
        # Attention to previous token
        if i > 0:
            attention[i, i-1] = 0.15
        
        # Check if current token is punctuation
        token_text = tokens[i].strip()
        is_punct = token_text in '.,;:!?"\'()[]{}' or ('"' in token_text and len(token_text) <= 3)
        
        if is_punct:
            # Punctuation gets attention from nearby tokens
            for j in range(max(0, i-3), i):
                if j != i:
                    attention[i, j] += 0.1
        else:
            # Non-punctuation tokens: look for syntactic relationships
            spacy_indices = gpt2_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                
                # Find syntactic head
                head = spacy_token.head
                if head != spacy_token:  # Not root
                    head_idx = head.i
                    # Find corresponding GPT2 token
                    for j in range(i):
                        j_spacy_indices = gpt2_to_spacy[j]
                        if j_spacy_indices and head_idx in j_spacy_indices:
                            attention[i, j] += 0.2
                            break
                
                # Find children that this token should attend to
                for child in spacy_token.children:
                    child_idx = child.i
                    # Find corresponding GPT2 token
                    for j in range(i):
                        j_spacy_indices = gpt2_to_spacy[j]
                        if j_spacy_indices and child_idx in j_spacy_indices:
                            attention[i, j] += 0.15
                            break
        
        # Special handling for tokens that attend to punctuation
        for j in range(i):
            j_token_text = tokens[j].strip()
            j_is_punct = j_token_text in '.,;:!?"\'()[]{}' or ('"' in j_token_text and len(j_token_text) <= 3)
            
            if j_is_punct and abs(i - j) <= 3:
                attention[i, j] += 0.1
        
        # Add some distance decay
        for j in range(i):
            if j != 0:  # Don't modify first token attention
                distance = i - j
                if distance <= 3:
                    attention[i, j] += max(0, 0.05 - 0.01 * distance)
    
    # Handle first token specially
    attention[0, 0] = 1.0
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 7, Head 9
def first_token_bias_content_focus_punctuation_L7H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token bias
        if i == 0:
            attention_matrix[i, i] = 1.0
        else:
            attention_matrix[i, 0] = 0.8
            
            # Punctuation attraction (especially commas)
            for j in range(i + 1):
                if tokens[j] in [',', '.', '"', "'", ';', ':', '!', '?']:
                    if j < i - 1:  # Not immediate previous token
                        attention_matrix[i, j] += 0.3
                    elif j == i - 1:  # Immediate previous punctuation
                        attention_matrix[i, j] += 0.1
            
            # Self-attention
            attention_matrix[i, i] = 0.05
            
            # Local context - moderate attention to nearby tokens
            for j in range(max(0, i - 3), i):
                if j != 0:  # Don't double-count first token
                    distance = i - j
                    weight = 0.03 / distance if distance > 0 else 0
                    attention_matrix[i, j] += weight
            
            # Syntactic relationships using spacy
            spacy_indices = gpt2_to_spacy[i]
            if spacy_indices:
                spacy_token = doc[spacy_indices[0]]
                
                # Verb to subject relationship
                if spacy_token.pos_ == 'VERB':
                    for child in spacy_token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            for k in range(i):
                                k_spacy = gpt2_to_spacy[k]
                                if k_spacy and child.i in k_spacy:
                                    attention_matrix[i, k] += 0.15
                
                # Preposition to object relationship
                if spacy_token.pos_ == 'ADP':  # Preposition
                    for child in spacy_token.children:
                        if child.dep_ == 'pobj':
                            for k in range(i + 1, min(n, i + 4)):
                                k_spacy = gpt2_to_spacy[k]
                                if k_spacy and child.i in k_spacy:
                                    attention_matrix[i, k] += 0.1
                
                # Modifier relationships
                if spacy_token.dep_ in ['amod', 'det']:
                    head = spacy_token.head
                    for k in range(i + 1, min(n, i + 3)):
                        k_spacy = gpt2_to_spacy[k]
                        if k_spacy and head.i in k_spacy:
                            attention_matrix[i, k] += 0.08
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_content_", attention_matrix


# Layer 7, Head 10
def first_token_bias_L7H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            # First token attends to itself with weight 1.0
            attention_matrix[i, i] = 1.0
        else:
            # All other tokens attend primarily to first token
            # Base attention to first token (very high)
            attention_matrix[i, 0] = 0.97
            
            # Small self-attention
            attention_matrix[i, i] = 0.02
            
            # Tiny residual attention to adjacent tokens
            if i > 0:
                attention_matrix[i, i-1] = 0.01
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L7H10", attention_matrix


# Layer 7, Head 11
def first_token_bias_stochastic_L7H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            # First token attends to itself
            attention_matrix[i, 0] = 1.0
        else:
            # All other tokens attend very strongly to first token
            attention_matrix[i, 0] = 0.92 + 0.07 * np.random.random()
            
            # Small self-attention
            if np.random.random() < 0.3:
                attention_matrix[i, i] = 0.01 + 0.02 * np.random.random()
            
            # Very sparse attention to a few other positions
            num_other = min(2, i)
            if num_other > 0:
                other_positions = np.random.choice(range(1, i), size=num_other, replace=False)
                for pos in other_positions:
                    if np.random.random() < 0.4:
                        attention_matrix[i, pos] = 0.005 + 0.02 * np.random.random()
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_stochast", attention_matrix


# Layer 8, Head 0
def first_token_bias_content_focus_L8H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for early tokens
        if i < 4:
            attention[i, 0] = 0.9 - 0.1 * i
        else:
            attention[i, 0] = 0.05
        
        # Self-attention
        attention[i, i] = 0.08
        
        # Find quote markers and dialogue content
        is_quote_marker = tokens[i] in [' "', '"', "'", " '"]
        in_quotes = False
        quote_start = -1
        
        # Simple quote detection
        for j in range(i):
            if tokens[j] in [' "', '"']:
                if quote_start == -1:
                    quote_start = j
                    in_quotes = True
                else:
                    in_quotes = False
                    quote_start = -1
        
        # Special attention patterns for quote-related tokens
        if is_quote_marker or in_quotes:
            for j in range(i + 1):
                if tokens[j] in [' "', '"', 'I', "'m", ' said', ':']:
                    attention[i, j] += 0.15
        
        # Attention to clause markers and important function words
        clause_words = [' that', ' when', ' which', ' where', ' who', ' what']
        for j in range(i):
            if tokens[j] in clause_words:
                attention[i, j] += 0.1
        
        # Attention to auxiliary verbs and "to"
        aux_words = [' to', ' be', ' was', ' were', ' is', ' are']
        for j in range(max(0, i-5), i):
            if tokens[j] in aux_words:
                attention[i, j] += 0.05
        
        # Local attention to previous few tokens
        for j in range(max(0, i-3), i):
            attention[i, j] += 0.02
        
        # Use spacy for syntactic relationships if available
        if i < len(gpt2_to_spacy) and gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to head of current token
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    # Find corresponding GPT2 token
                    for j in range(i):
                        if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                            if head_idx in gpt2_to_spacy[j]:
                                attention[i, j] += 0.08
                                break
        
        # Boost attention for specific patterns observed in examples
        token_text = tokens[i].lower()
        
        # Conjunctions attend to earlier content
        if token_text in [' and', ' or', ' but']:
            for j in range(max(0, i-8), i):
                attention[i, j] += 0.03
        
        # Pronouns attend to recent nouns/names
        if token_text in [' she', ' he', ' it', ' they']:
            for j in range(max(0, i-6), i):
                if tokens[j].strip().istitle() or tokens[j] in [' girl', ' boy', ' bird', ' cat']:
                    attention[i, j] += 0.06
        
        # Final token often attends to key content words
        if i == n - 1:
            for j in range(i):
                if tokens[j].strip().istitle() or len(tokens[j]) > 4:
                    attention[i, j] += 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 8, Head 1
def first_token_bias_L8H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            # First token attends to itself
            attention[i, 0] = 1.0
        else:
            # All other tokens attend primarily to first token
            # with very high weight (0.95-0.99 range observed)
            attention[i, 0] = 0.97
            
            # Small amount of self-attention
            attention[i, i] = 0.02
            
            # Tiny residual attention to a few nearby previous tokens
            for j in range(max(0, i-3), i):
                if j != 0:  # Don't double-count first token
                    attention[i, j] = 0.01 / max(1, i-1)
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L8H1", attention


# Layer 8, Head 2
def first_token_bias_content_focus_punctuation_stochastic_L8H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Identify proper nouns and named entities
    proper_nouns = set()
    named_entities = set()
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            for i in range(ent.start, ent.end):
                named_entities.add(i)
    
    for i, token in enumerate(doc):
        if token.pos_ == "PROPN" or token.ent_type_ in ["PERSON", "ORG", "GPE"]:
            proper_nouns.add(i)
    
    # Find GPT2 tokens that correspond to proper nouns/named entities
    gpt2_proper_nouns = set()
    gpt2_named_entities = set()
    
    for gpt2_idx, spacy_indices in enumerate(alignment):
        for spacy_idx in spacy_indices:
            if spacy_idx in proper_nouns:
                gpt2_proper_nouns.add(gpt2_idx)
            if spacy_idx in named_entities:
                gpt2_named_entities.add(gpt2_idx)
    
    # Main attention computation
    for i in range(n):
        for j in range(i + 1):  # Causal mask
            weight = 0.0
            
            # 1. Strong first-token attention (dominant pattern)
            if j == 0:
                if i <= 4:  # First few tokens attend very strongly to first token
                    weight += 0.8 + 0.1 * (4 - i) / 4
                else:
                    weight += 0.1
            
            # 2. Self-attention
            if i == j:
                if tokens[i].strip() in [".", ",", "?", "!", ":", ";"]:  # Punctuation
                    weight += 0.15
                else:
                    weight += 0.08
            
            # 3. Attention to proper nouns and named entities
            if j in gpt2_proper_nouns or j in gpt2_named_entities:
                if i != j:  # Don't double-count self-attention
                    weight += 0.12
            
            # 4. Special patterns for punctuation and sentence endings
            if tokens[i].strip() in [".", "!", "?"]:  # End punctuation
                if j in gpt2_proper_nouns or j in gpt2_named_entities:
                    weight += 0.05
                # Attend to certain content words
                if j > 0 and j < i:
                    token_text = tokens[j].strip().lower()
                    if token_text in ["to", "with", "and", "the"]:
                        weight += 0.03
            
            # 5. Local context - attend to recent important tokens
            if i - j <= 3 and j > 0:
                token_text = tokens[j].strip().lower()
                if token_text in ["to", "and", "with", "from", "that", "the"]:
                    weight += 0.02
            
            # 6. Quote and dialogue patterns
            if '"' in tokens[i] or "'" in tokens[i]:
                # Attend to quote boundaries and dialogue markers
                if '"' in tokens[j] or "'" in tokens[j]:
                    weight += 0.05
                # Attend to names in dialogue
                if j in gpt2_proper_nouns:
                    weight += 0.03
            
            # 7. Comma patterns - attend to nearby important tokens
            if tokens[i].strip() == ",":
                if j == 0:  # Already handled above
                    pass
                elif j in gpt2_proper_nouns and j < i:
                    weight += 0.03
            
            attention[i, j] = weight
    
    # Add small random baseline to prevent zero rows
    attention += np.random.random((n, n)) * 0.005
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 8, Head 3
def first_token_bias_content_focus_L8H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    attention_matrix = np.zeros((n, n))
    
    # Detect quoted speech regions
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
        
        # Base attention weights
        for j in range(i + 1):  # Only attend to previous tokens and self
            if j == 0:  # First token gets very high attention
                attention_matrix[i, j] = 10.0
            elif j == i:  # Self-attention
                attention_matrix[i, j] = 0.3
            else:
                # Base attention decreases with distance
                distance = i - j
                attention_matrix[i, j] = 0.1 / (1 + 0.3 * distance)
        
        # Special handling for tokens in quoted speech
        if in_quotes[i] and i > 5:  # Only apply to longer sentences with quotes
            # Reduce first-token dominance for tokens in quotes
            attention_matrix[i, 0] *= 0.3
            
            # Increase attention to recent tokens within the quote
            for j in range(max(0, i - 8), i + 1):
                if j != 0 and in_quotes[j]:  # Recent tokens also in quotes
                    attention_matrix[i, j] *= 2.5
        
        # Special cases for high-attention tokens
        for j in range(i + 1):
            target_token = tokens[j]
            
            # Sentence boundary tokens (periods, quotes) get extra attention
            if target_token in ['.', '."', '"', '!"', '?"']:
                attention_matrix[i, j] *= 3.0
            
            # Conjunctions get extra attention from later tokens
            elif target_token.strip().lower() in ['and', 'or', 'but']:
                if i > j + 2:  # Only from tokens that are not immediately following
                    attention_matrix[i, j] *= 2.0
            
            # "The" gets some extra attention as a common function word
            elif target_token.strip().lower() == 'the' and j > 0:
                attention_matrix[i, j] *= 1.5
        
        # Special handling for specific token types as queries
        if token.strip() in ['.', '."', '"']:  # Sentence endings attend more to content words
            # Find content words using spacy alignment
            if alignment[i]:
                spacy_idx = alignment[i][0]
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    # Period/quote tokens distribute attention more broadly
                    for j in range(i + 1):
                        if j != 0:  # Don't reduce first token attention
                            attention_matrix[i, j] *= 0.8
                            
        elif token.strip().lower() in ['and', 'or']:  # Conjunctions
            # Conjunctions attend more to nearby content
            for j in range(max(0, i - 3), i):
                if j != 0:  # Don't reduce first token attention
                    attention_matrix[i, j] *= 1.5
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_content_", attention_matrix


# Layer 8, Head 4
def first_token_bias_content_focus_L8H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong self-attention for first token
        if i == 0:
            attention[i, i] = 1.0
            continue
            
        # Strong attention to first token (BOS-like behavior)
        attention[i, 0] = 0.7
        
        # Get spacy tokens that align with current GPT2 token
        spacy_indices = alignment[i] if i < len(alignment) else []
        
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Find syntactic head
            head_idx = None
            if spacy_token.head != spacy_token and spacy_token.head.i < len(doc):
                # Find which GPT2 tokens correspond to the head
                for j in range(i):
                    head_spacy_indices = alignment[j] if j < len(alignment) else []
                    if spacy_token.head.i in head_spacy_indices:
                        head_idx = j
                        break
            
            # Strong attention to syntactic head
            if head_idx is not None:
                attention[i, head_idx] = 0.4
            
            # Special attention patterns for specific POS/dependencies
            if spacy_token.pos_ in ['NOUN', 'PROPN']:
                # Nouns attend to recent verbs
                for j in range(max(0, i-3), i):
                    j_spacy = alignment[j] if j < len(alignment) else []
                    if j_spacy and doc[j_spacy[0]].pos_ == 'VERB':
                        attention[i, j] = 0.3
                        break
            
            elif spacy_token.pos_ == 'VERB':
                # Verbs get attention from their objects/complements
                for child in spacy_token.children:
                    if child.dep_ in ['dobj', 'pobj', 'nsubj']:
                        # This will be handled when we process the child token
                        pass
                
            elif spacy_token.pos_ == 'ADP':  # Prepositions
                # Prepositions attend to their heads
                if head_idx is not None:
                    attention[i, head_idx] = 0.5
        
        # Punctuation gets special treatment
        token_text = tokens[i].strip()
        if token_text in [',', '.', '!', '?', ';', ':', '"', '"']:
            # Punctuation attends to recent content words
            for j in range(max(0, i-3), i):
                j_spacy = alignment[j] if j < len(alignment) else []
                if j_spacy and doc[j_spacy[0]].pos_ in ['VERB', 'NOUN', 'ADJ']:
                    attention[i, j] = 0.4
                    break
        
        # Add some recency bias
        if i > 0:
            attention[i, i-1] = 0.15
        
        # Add weak self-attention
        attention[i, i] = 0.08
        
        # Special handling for quoted speech
        if '"' in tokens[i] or "'" in tokens[i]:
            # Find the main verb in the clause
            for j in range(i):
                j_spacy = alignment[j] if j < len(alignment) else []
                if j_spacy and doc[j_spacy[0]].pos_ == 'VERB':
                    attention[i, j] = 0.3
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 8, Head 5
def first_token_bias_content_focus_punctuation_L8H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Create mapping from GPT2 tokens to spacy features
    token_poses = []
    token_deps = []
    token_heads = []
    
    for i, gpt2_overlaps in enumerate(alignment):
        if gpt2_overlaps:
            spacy_tok = doc[gpt2_overlaps[0]]  # Use first overlapping spacy token
            token_poses.append(spacy_tok.pos_)
            token_deps.append(spacy_tok.dep_)
            # Find head token's GPT2 index
            head_idx = None
            if spacy_tok.head != spacy_tok:  # Not root
                for j, other_overlaps in enumerate(alignment):
                    if spacy_tok.head.i in other_overlaps:
                        head_idx = j
                        break
            token_heads.append(head_idx)
        else:
            token_poses.append('')
            token_deps.append('')
            token_heads.append(None)
    
    for i in range(n):
        # Strong attention to first token for early tokens
        if i < 4:
            attention[i, 0] = 0.8 - (i * 0.15)
        else:
            attention[i, 0] = 0.1
        
        # Self-attention (moderate)
        attention[i, i] = 0.08
        
        # Attention to syntactic head
        head_idx = token_heads[i]
        if head_idx is not None and head_idx <= i:
            attention[i, head_idx] += 0.15
        
        # Verbs get extra attention from later tokens
        for j in range(i):
            if token_poses[j] in ['VERB', 'AUX']:
                attention[i, j] += 0.12
        
        # Local context - attend to previous few tokens
        for j in range(max(0, i-3), i):
            attention[i, j] += 0.05
        
        # Special handling for punctuation and quotes
        if tokens[i] in ['.', '!', '?', '"', "'", ',"', '."', '!"']:
            # Punctuation attends more to verbs and important content
            for j in range(i):
                if token_poses[j] in ['VERB', 'NOUN']:
                    attention[i, j] += 0.08
        
        # Nouns attend to their modifiers
        if token_poses[i] == 'NOUN':
            for j in range(i):
                if token_poses[j] in ['ADJ', 'DET'] and j >= i-3:
                    attention[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 8, Head 6
def first_token_bias_content_focus_punctuation_L8H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token for most positions
        if i > 0:
            attention_matrix[i, 0] = 0.8
        
        # Self-attention
        attention_matrix[i, i] = 0.1
        
        # Attention to previous token
        if i > 0:
            attention_matrix[i, i-1] = 0.05
        
        # For punctuation and end tokens, distribute attention more broadly
        token = tokens[i]
        if token in ['.', '?', '!', ','] or i == n-1:
            # Reduce first token attention and distribute to earlier tokens
            if i > 0:
                attention_matrix[i, 0] = 0.4
            attention_matrix[i, i] = 0.1
            
            # Add attention to content tokens in earlier positions
            remaining_weight = 0.5
            valid_positions = list(range(i))
            if valid_positions:
                weight_per_pos = remaining_weight / len(valid_positions)
                for j in valid_positions:
                    attention_matrix[i, j] += weight_per_pos
    
    # Special case for first token (attends only to itself)
    attention_matrix[0, 0] = 1.0
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_content_", attention_matrix


# Layer 8, Head 7
def first_token_bias_content_focus_L8H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for early tokens
        if i <= 3:  # First few tokens
            attention[i, 0] = 0.8 - (i * 0.15)  # Decreasing strength
        else:
            attention[i, 0] = 0.1  # Weaker for later tokens
        
        # Self-attention
        attention[i, i] = 0.1
        
        # Previous token attention (general recency bias)
        if i > 0:
            attention[i, i-1] = 0.2
        if i > 1:
            attention[i, i-2] = 0.1
            
        # Get spacy information for current token
        spacy_indices = gpt2_to_spacy[i]
        
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Preposition to object pattern
            if spacy_token.pos_ == "ADP":  # Preposition
                # Look for the next noun/object
                for j in range(i+1, min(i+3, n)):
                    next_spacy = gpt2_to_spacy[j]
                    if next_spacy:
                        next_token = doc[next_spacy[0]]
                        if next_token.pos_ in ["NOUN", "PROPN", "PRON"]:
                            attention[i, j] = 0.4
                            break
            
            # Adjective to noun pattern
            if spacy_token.pos_ == "ADJ":
                # Find the noun this modifies
                if spacy_token.head.pos_ in ["NOUN", "PROPN"]:
                    head_idx = spacy_token.head.i
                    # Find corresponding GPT2 token
                    for j in range(max(0, i-3), min(i+3, n)):
                        if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                            if head_idx in gpt2_to_spacy[j]:
                                attention[i, j] = 0.3
                                break
            
            # Noun to modifying adjective
            if spacy_token.pos_ in ["NOUN", "PROPN"]:
                for child in spacy_token.children:
                    if child.dep_ == "amod":  # Adjectival modifier
                        child_idx = child.i
                        for j in range(max(0, i-3), i):
                            if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                                if child_idx in gpt2_to_spacy[j]:
                                    attention[i, j] = 0.25
                                    break
            
            # Verb to subject/object patterns
            if spacy_token.pos_ == "VERB":
                for child in spacy_token.children:
                    if child.dep_ in ["nsubj", "dobj"]:
                        child_idx = child.i
                        for j in range(max(0, i-5), i):
                            if j < len(gpt2_to_spacy) and gpt2_to_spacy[j]:
                                if child_idx in gpt2_to_spacy[j]:
                                    attention[i, j] = 0.2
                                    break
        
        # Special handling for specific token patterns
        token_text = tokens[i].strip()
        
        # Punctuation tends to attend to nearby content
        if token_text in [".", ",", "?", "!"]:
            if i > 0:
                attention[i, i-1] = 0.3
            if i > 1:
                attention[i, i-2] = 0.2
        
        # Special case: Commas have strong attention to recent quotation marks
        if token_text == ",":
            for j in range(max(0, i-10), i):  # Look back up to 10 tokens
                quote_text = tokens[j].strip()
                if quote_text in ['"', "'", ',"', '?"', '!"']:  # Various quote marks
                    attention[i, j] = 0.4
                    break  # Use the most recent quote mark
        
        # Articles and determiners attend to following nouns
        if token_text.lower() in ["the", "a", "an"]:
            for j in range(i+1, min(i+3, n)):
                next_spacy = gpt2_to_spacy[j]
                if next_spacy:
                    next_token = doc[next_spacy[0]]
                    if next_token.pos_ in ["NOUN", "PROPN"]:
                        attention[i, j] = 0.3
                        break
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 8, Head 8
def first_token_bias_content_focus_L8H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    if n == 1:
        return tokens, np.array([[1.0]])
    
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Helper to check if a token is content word
    def is_content_word(spacy_indices):
        if not spacy_indices:
            return False
        for idx in spacy_indices:
            if idx < len(doc):
                token = doc[idx]
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop:
                    return True
        return False
    
    # Helper to check if token is conjunction/coordination
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
        # Strong first token attention
        if i > 0:
            attention[i, 0] = 0.7
        else:
            attention[i, i] = 0.8
        
        # Self attention
        if i > 0:
            attention[i, i] = 0.15
        
        # Content word attraction
        for j in range(i):
            if j == 0:
                continue  # Already handled first token
                
            spacy_j = alignment[j] if j < len(alignment) else []
            spacy_i = alignment[i] if i < len(alignment) else []
            
            # Higher attention to content words
            if is_content_word(spacy_j):
                attention[i, j] += 0.2
                
                # Extra boost if current token is also content word
                if is_content_word(spacy_i):
                    attention[i, j] += 0.1
            
            # Conjunction patterns - conjunctions attract attention from later tokens
            if is_conjunction(spacy_j):
                attention[i, j] += 0.15
            
            # Local attention bias (prefer recent tokens)
            if i - j <= 3:
                attention[i, j] += 0.05
            
            # Special boost for verb-noun relationships
            if spacy_j and spacy_i:
                j_pos = doc[spacy_j[0]].pos_ if spacy_j[0] < len(doc) else ''
                i_pos = doc[spacy_i[0]].pos_ if spacy_i[0] < len(doc) else ''
                
                if (j_pos == 'NOUN' and i_pos == 'VERB') or (j_pos == 'VERB' and i_pos == 'NOUN'):
                    attention[i, j] += 0.1
        
        # Add small baseline attention to all previous tokens
        for j in range(i):
            if attention[i, j] == 0:
                attention[i, j] = 0.01
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 8, Head 9
def decaying_first_token_bias_content_focus_L8H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Identify content words (nouns, verbs, adjectives)
    content_word_indices = set()
    for i, spacy_indices in enumerate(alignment):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    content_word_indices.add(i)
    
    for i in range(n):
        # Strong first-token attention for early tokens
        if i <= 3:
            attention[i, 0] = 0.9 - (i * 0.1)
        else:
            attention[i, 0] = 0.05
        
        # Self-attention
        if i < 4:
            attention[i, i] = 0.05 + (i * 0.02)
        else:
            attention[i, i] = 0.08
        
        # Content word attention
        for j in range(i + 1):
            if j in content_word_indices and j != i:
                # Distance decay for content words
                distance = i - j
                if distance <= 3:
                    attention[i, j] += 0.12 - (distance * 0.02)
                else:
                    attention[i, j] += 0.04
        
        # Special patterns for sentence-final tokens
        if i == n - 1 and n > 1:
            # Final token attends more to recent content words
            for j in range(max(0, i - 5), i):
                if j in content_word_indices:
                    attention[i, j] += 0.05
        
        # Adjacent token attention (mild)
        if i > 0:
            attention[i, i - 1] += 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 8, Head 10
def decaying_first_token_bias_content_focus_punctuation_L8H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Base attention distribution
        for j in range(i + 1):  # Only attend to previous tokens + self
            if j == 0:  # First token
                # Strong first-token attention, decreasing with position
                if i <= 3:
                    attention_matrix[i, j] = 0.85 + 0.1 * (3 - i) / 3
                else:
                    # Exponential decay for later positions
                    attention_matrix[i, j] = 0.6 * np.exp(-0.3 * (i - 3))
                    
            elif j == i:  # Self-attention
                if i == 0:
                    attention_matrix[i, j] = 1.0  # First token always attends to itself
                else:
                    # Moderate self-attention for other tokens
                    attention_matrix[i, j] = 0.1 + 0.05 * min(i, 5)
                    
            else:  # Other tokens
                # Local attention with bias toward recent tokens
                distance = i - j
                if distance == 1:  # Previous token
                    attention_matrix[i, j] = 0.08
                elif distance <= 3:  # Nearby tokens
                    attention_matrix[i, j] = 0.04 / distance
                else:  # Distant tokens
                    attention_matrix[i, j] = 0.02 / distance
        
        # Special adjustments for punctuation and sentence structure
        token = tokens[i]
        
        # Punctuation tokens (commas, periods) get enhanced attention patterns
        if token in [',', '.', '!', '?', ':"', '."']:
            # Redistribute some attention from first token to content words
            for j in range(i):
                if j > 0 and tokens[j] not in [',', '.', '!', '?', ' the', ' a', ' an', ' and', ' or']:
                    attention_matrix[i, j] *= 1.5
            attention_matrix[i, 0] *= 0.7  # Reduce first-token attention slightly
        
        # Conjunctions like "and" get special patterns
        if token == ' and':
            # Attend more to the token being conjoined (often the previous content word)
            for j in range(max(0, i-3), i):
                if tokens[j] not in [' the', ' a', ' an', ',', '.']:
                    attention_matrix[i, j] *= 2.0
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 8, Head 11
def first_token_bias_punctuation_L8H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Very strong attention to first token (dominant pattern)
        attention[i, 0] = 0.85 if i > 0 else 1.0
        
        # Moderate self-attention
        if i > 0:
            attention[i, i] = 0.04
        
        # Get spacy alignment for syntactic features
        spacy_indices = gpt2_to_spacy[i]
        
        if spacy_indices and i > 0:
            current_spacy_idx = spacy_indices[0]
            current_token = doc[current_spacy_idx]
            
            # Find syntactic relationships
            # Attend to head of current token
            if current_token.head != current_token and current_token.head.i < len(doc):
                head_idx = current_token.head.i
                # Find corresponding GPT2 tokens
                for j in range(min(i, n)):  # causal mask
                    j_spacy_indices = gpt2_to_spacy[j]
                    if j_spacy_indices and head_idx in j_spacy_indices:
                        attention[i, j] += 0.08
            
            # Attend to children/modifiers
            for child in current_token.children:
                if child.i < len(doc):
                    child_idx = child.i
                    for j in range(min(i, n)):  # causal mask
                        j_spacy_indices = gpt2_to_spacy[j]
                        if j_spacy_indices and child_idx in j_spacy_indices:
                            attention[i, j] += 0.06
            
            # Special attention to nearby punctuation and function words
            for j in range(max(0, i-5), i):
                j_spacy_indices = gpt2_to_spacy[j]
                if j_spacy_indices:
                    j_token = doc[j_spacy_indices[0]]
                    if j_token.pos_ in ['PUNCT'] or j_token.text in [',', '.', '?', '!']:
                        attention[i, j] += 0.03
                    elif j_token.pos_ in ['ADP', 'CONJ', 'CCONJ', 'DET']:
                        attention[i, j] += 0.02
        
        # Add small attention to recent tokens (recency bias)
        for j in range(max(0, i-3), i):
            if j != 0:  # first token already handled
                attention[i, j] += 0.01 * (1.0 / (i - j + 1))
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 9, Head 0
def first_token_bias_content_focus_punctuation_L9H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Get spacy parse for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base weights
        weights = np.zeros(n)
        
        # Strong attention to first few tokens (especially token 0)
        for j in range(min(4, i + 1)):
            if j == 0:
                weights[j] = 10.0  # Very strong attention to first token
            elif j <= 3:
                weights[j] = 3.0 - j * 0.5  # Decreasing attention to early tokens
        
        # Self-attention
        weights[i] = 1.0
        
        # Attention to immediate predecessor
        if i > 0:
            weights[i-1] += 0.5
        
        # Content-based attention - look for noun/verb relationships
        if gpt2_to_spacy[i]:
            current_spacy_idx = gpt2_to_spacy[i][0]
            if current_spacy_idx < len(doc):
                current_token = doc[current_spacy_idx]
                
                # If current token is a verb, attend to nearby nouns
                if current_token.pos_ in ['VERB', 'AUX']:
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            j_spacy_idx = gpt2_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_token = doc[j_spacy_idx]
                                if j_token.pos_ == 'NOUN':
                                    weights[j] += 0.3
                
                # If current token is related to objects/instruments, attend to action words
                if current_token.pos_ == 'NOUN':
                    for j in range(i):
                        if gpt2_to_spacy[j]:
                            j_spacy_idx = gpt2_to_spacy[j][0]
                            if j_spacy_idx < len(doc):
                                j_token = doc[j_spacy_idx]
                                if j_token.pos_ in ['VERB', 'NOUN'] and j_token.lemma_ != current_token.lemma_:
                                    weights[j] += 0.2
        
        # Add small attention to all valid positions
        for j in range(i + 1):
            weights[j] += 0.05
        
        # Special handling for punctuation and function words
        if i < n and tokens[i].strip() in [',', '.', ':', ';', '"', "'", '?', '!']:
            # Punctuation attends more to content words
            for j in range(i):
                if gpt2_to_spacy[j]:
                    j_spacy_idx = gpt2_to_spacy[j][0]
                    if j_spacy_idx < len(doc):
                        j_token = doc[j_spacy_idx]
                        if j_token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            weights[j] += 0.2
        
        attention[i] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 9, Head 1
def first_token_bias_L9H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            # First token attends to itself
            attention[i, i] = 1.0
        else:
            # All other tokens attend primarily to first token with some self-attention
            attention[i, 0] = 0.95  # Strong attention to first token
            attention[i, i] = 0.05  # Small self-attention
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L9H1", attention


# Layer 9, Head 2
def decaying_first_token_bias_content_focus_punctuation_L9H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify important tokens
    punctuation_indices = set()
    content_word_indices = set()
    
    for i, token in enumerate(tokens):
        # Identify punctuation
        if any(c in token for c in '.,!?;:'):
            punctuation_indices.add(i)
        
        # Identify content words using spacy alignment
        if gpt2_to_spacy[i]:
            spacy_token = doc[gpt2_to_spacy[i][0]]
            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not spacy_token.is_stop:
                content_word_indices.add(i)
    
    for i in range(n):
        # Base attention pattern
        for j in range(i + 1):  # Causal mask
            if j == 0:
                # Very strong attention to first token
                attention_matrix[i, j] = 0.8
            elif j == i:
                # Moderate self-attention
                attention_matrix[i, j] = 0.05
            elif j in punctuation_indices:
                # Higher attention to punctuation
                attention_matrix[i, j] = 0.1
            else:
                # Base attention with distance decay
                distance = i - j
                attention_matrix[i, j] = 0.02 / (1 + 0.1 * distance)
        
        # Special cases for content words
        if i in content_word_indices:
            # Content words attend more to other content words
            for j in range(i):
                if j in content_word_indices:
                    attention_matrix[i, j] *= 2.0
        
        # Boost attention to nearby punctuation
        for j in punctuation_indices:
            if j < i and i - j <= 3:
                attention_matrix[i, j] *= 3.0
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 9, Head 3
def first_token_bias_L9H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    # Parse with spacy for syntactic information
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_gpt2 = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (very consistent pattern)
        attention_matrix[i, 0] = 0.6
        
        # Self-attention (moderate but consistent)
        attention_matrix[i, i] = 0.1
        
        # Previous token attention (recency bias)
        if i > 0:
            attention_matrix[i, i-1] = 0.15
        
        # Syntactic dependencies
        spacy_indices = gpt2_to_spacy[i]
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attention to syntactic head
                if spacy_token.head != spacy_token:
                    head_gpt2_indices = spacy_to_gpt2[spacy_token.head.i]
                    for head_idx in head_gpt2_indices:
                        if head_idx <= i:  # Causal constraint
                            attention_matrix[i, head_idx] += 0.2
                
                # Attention from modifiers to their heads
                for child in spacy_token.children:
                    if child.dep_ in ['amod', 'nmod', 'prep', 'pobj']:
                        child_gpt2_indices = spacy_to_gpt2[child.i]
                        for child_idx in child_gpt2_indices:
                            if child_idx <= i and child_idx < n:
                                attention_matrix[child_idx, i] += 0.15
    
    # Additional pattern: Strong attention to function words that act as discourse markers
    function_words = set([' but', ' can', ' could', ' will', ' would', ' should', ' may', ' might', 
                         ' and', ' or', ' so', ' because', ' since', ' while', ' if', ' when',
                         ' "', '"', ',"', ' said', ':', ' that'])
    
    for j in range(n):
        if tokens[j] in function_words or (tokens[j].strip() in ['"', ',"', ':']):
            # Tokens within a window attend strongly to these function words
            window_size = min(5, n - j)
            for k in range(j + 1, min(j + window_size + 1, n)):
                attention_matrix[k, j] += 0.3
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L9H3", attention_matrix


# Layer 9, Head 4
def first_token_bias_punctuation_L9H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Very strong attention to first token for all positions
        attention_matrix[i, 0] = 0.9
        
        # Self-attention boost for punctuation tokens
        token = tokens[i]
        if token.strip() in '.!?;:,':
            attention_matrix[i, i] = 0.15
            attention_matrix[i, 0] = 0.75  # Reduce first-token attention slightly
        else:
            # Small self-attention for non-punctuation
            attention_matrix[i, i] = 0.02
        
        # Small attention to nearby previous tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                attention_matrix[i, j] = 0.03 / distance
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_punctuat", attention_matrix


# Layer 9, Head 5
def first_token_bias_L9H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base weights
        weights = np.zeros(n)
        
        # Strong first-token attention (dominant pattern)
        weights[0] = 0.8
        
        # Self-attention
        weights[i] = 0.15
        
        # Previous token attention (recency bias)
        if i > 0:
            weights[i-1] = 0.1
            
        # Syntactic dependencies - find head relationships
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]  # Use first aligned spacy token
            
            # Attend to syntactic head
            if spacy_token.head != spacy_token:
                head_char_start = spacy_token.head.idx
                # Find GPT2 token that starts at or near this position
                char_pos = 0
                for j in range(min(i+1, n)):  # Only look at previous tokens due to causal mask
                    if char_pos <= head_char_start < char_pos + len(tokens[j]):
                        weights[j] += 0.2
                        break
                    char_pos += len(tokens[j])
            
            # Attend to syntactic children (modifiers, objects, etc.)
            for child in spacy_token.children:
                child_char_start = child.idx
                char_pos = 0
                for j in range(min(i+1, n)):  # Only look at previous tokens
                    if char_pos <= child_char_start < char_pos + len(tokens[j]):
                        weights[j] += 0.1
                        break
                    char_pos += len(tokens[j])
        
        # Add small uniform attention to other previous tokens
        for j in range(i):
            if weights[j] == 0:
                weights[j] = 0.02
        
        attention[i] = weights
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L9H5", attention


# Layer 9, Head 6
def first_token_bias_L9H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    attention = np.zeros((n, n))
    
    # Strong attention to first token from all positions
    for i in range(n):
        attention[i, 0] = 0.95  # Very high base attention to first token
    
    # Small self-attention for non-first tokens
    for i in range(1, n):
        attention[i, i] = 0.02
    
    # Very small residual attention distributed to earlier tokens
    for i in range(1, n):
        remaining_mass = 1.0 - attention[i, 0] - attention[i, i]
        # Distribute small amounts to tokens between first and self
        if i > 1:
            per_token = remaining_mass / (i - 1)
            for j in range(1, i):
                attention[i, j] = per_token
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L9H6", attention


# Layer 9, Head 7
def decaying_first_token_bias_content_focus_L9H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Get content word mask
    content_words = set()
    for i, spacy_indices in enumerate(alignment):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token = doc[spacy_idx]
                if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
                    content_words.add(i)
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # First token gets very high base attention
        if i < n:
            attention_matrix[i, 0] = 10.0
        
        # Strong recency bias for positions 1-2 back
        if i >= 1:
            attention_matrix[i, i-1] = 5.0
        if i >= 2:
            attention_matrix[i, i-2] = 2.0
            
        # Self attention
        attention_matrix[i, i] = 1.0
        
        # Content word attraction
        for j in range(i + 1):  # Only look backwards (causal)
            if j in content_words:
                distance = i - j + 1
                # Boost content words with distance decay
                boost = 3.0 / (distance ** 0.5)
                attention_matrix[i, j] += boost
        
        # Additional distance-based decay for all positions
        for j in range(i + 1):
            if j != 0:  # Don't decay first token attention
                distance = i - j + 1
                base_decay = 0.3 / (distance ** 0.8)
                attention_matrix[i, j] += base_decay
    
    # Special handling for first few tokens - they should attend very strongly to first token
    for i in range(min(4, n)):
        attention_matrix[i, 0] = max(attention_matrix[i, 0], 15.0)
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 9, Head 8
def decaying_first_token_bias_content_focus_punctuation_L9H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify content words (nouns, verbs, adjectives) at GPT2 token level
    content_words = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                    content_words.add(i)
    
    for i in range(n):
        # Strong attention to first token (very high baseline)
        attention[i, 0] = 3.0
        
        # Content word attraction - earlier content words get higher attention
        for j in range(i + 1):
            if j in content_words and j < i:
                # Distance decay for content words
                distance = i - j
                content_weight = 1.5 / (1 + 0.3 * distance)
                attention[i, j] += content_weight
        
        # Self attention (moderate)
        attention[i, i] += 0.3
        
        # Local context (previous token)
        if i > 0:
            attention[i, i-1] += 0.2
        
        # Special handling for punctuation and sentence boundaries
        token = tokens[i]
        if token.strip() in ['.', '!', '?', ',']:
            # Punctuation tends to attend more to recent content
            for j in range(max(0, i-3), i):
                if j in content_words:
                    attention[i, j] += 0.5
            # Less attention to first token for punctuation
            attention[i, 0] *= 0.6
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 9, Head 9
def first_token_bias_stochastic_L9H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize matrix
    attention_matrix = np.zeros((n, n))
    
    # Very strong first-token bias for all positions
    for i in range(n):
        attention_matrix[i, 0] = 0.95  # High baseline attention to first token
    
    # Add some self-attention
    for i in range(n):
        attention_matrix[i, i] = 0.04  # Moderate self-attention
    
    # Add small amount of random attention to other valid positions
    for i in range(n):
        for j in range(1, i):  # Skip first token (already high) and self (already set)
            attention_matrix[i, j] = 0.01 / max(1, i-1)  # Small distributed attention
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_stochast", attention_matrix


# Layer 9, Head 10
def first_token_bias_L9H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            # First token attends only to itself with weight 1.0
            attention[i, 0] = 1.0
        else:
            # Strong first-token bias for early tokens, decreasing for later ones
            first_token_weight = max(0.3, 0.95 - 0.1 * i)
            attention[i, 0] = first_token_weight
            
            # Adjacent token attention (previous token)
            if i > 0:
                adjacent_weight = max(0.1, 0.4 - 0.05 * i)
                attention[i, i-1] = adjacent_weight
            
            # Self attention
            self_weight = 0.05 + 0.02 * min(i, 5)
            attention[i, i] = self_weight
            
            # Moderate attention to tokens 1-3 for early positions
            if i <= 5:
                for j in range(1, min(4, i)):
                    if j != i-1:  # Don't double-count adjacent
                        attention[i, j] = max(0.02, 0.15 - 0.02 * (i + j))
            
            # Decay attention to other previous tokens
            for j in range(1, i-1):
                if j not in [0, i-1] and j not in range(1, min(4, i)):
                    distance = i - j
                    attention[i, j] = max(0.01, 0.08 / (1 + 0.3 * distance))
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L9H10", attention


# Layer 9, Head 11
def first_token_bias_punctuation_L9H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention_matrix = np.zeros((n, n))
    
    # Each token attends primarily to the first token and itself
    for i in range(n):
        if i == 0:
            # First token attends only to itself
            attention_matrix[i, 0] = 1.0
        else:
            # Other tokens attend heavily to first token with small self-attention
            attention_matrix[i, 0] = 0.97  # Very high attention to first token
            attention_matrix[i, i] = 0.025  # Small self-attention
            
            # Add tiny amounts of attention to nearby tokens for realism
            for j in range(max(0, i-2), i):
                if j != 0:  # Don't double-count first token
                    attention_matrix[i, j] = 0.005 / max(1, i-1)
            
            # Special case: for longer sentences (>15 tokens), add some attention to dialogue/punctuation context
            if n > 15:
                # Look for quote-related tokens or punctuation that might be contextually important
                for j in range(i):
                    token = tokens[j]
                    # If this is a quote or dialogue-related token, add slight attention
                    if '"' in token or "'" in token or token in [',', '.', '?', '!']:
                        # Add small amount of attention, but don't override first token dominance
                        if j != 0:  # Don't modify first token attention
                            attention_matrix[i, j] += 0.003
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_punctuat", attention_matrix


# Layer 10, Head 0
def first_token_bias_content_focus_punctuation_L10H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Get noun positions for content word attention
    noun_positions = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ in ['NOUN', 'PROPN']:
                noun_positions.add(i)
    
    # Get important content word positions (proper nouns, main verbs, key nouns)
    important_positions = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                token_spacy = doc[spacy_idx]
                # Include proper nouns, main verbs, and important nouns
                if (token_spacy.pos_ == 'PROPN' or 
                    (token_spacy.pos_ == 'VERB' and token_spacy.dep_ in ['ROOT', 'conj']) or
                    (token_spacy.pos_ == 'NOUN' and len(token_spacy.text) > 3)):
                    important_positions.add(i)
    
    for i in range(n):
        # Strong attention to first token (dominant pattern)
        attention[i, 0] = 0.95
        
        # Self-attention (moderate)
        attention[i, i] = 0.03
        
        # Content word attention for some tokens
        if i > 0:
            # Find nearby nouns to attend to
            for noun_pos in noun_positions:
                if noun_pos <= i and noun_pos != 0:  # Can only attend to previous tokens, not first
                    distance = i - noun_pos
                    if distance <= 3:  # Only nearby nouns
                        weight = 0.02 / (1 + distance * 0.5)
                        attention[i, noun_pos] += weight
        
        # NEW: Enhanced attention to important content words
        if i > 0:
            for imp_pos in important_positions:
                if imp_pos < i and imp_pos != 0:  # Can only attend to previous tokens, not first
                    distance = i - imp_pos
                    # Stronger attention to important content words, with longer range
                    if distance <= 8:
                        weight = 0.08 / (1 + distance * 0.3)
                        attention[i, imp_pos] += weight
        
        # Special handling for punctuation (attend more to content)
        token_text = tokens[i].strip()
        if token_text in ['.', ',', '!', '?']:
            # Reduce first-token bias slightly for punctuation
            attention[i, 0] = 0.85
            # Increase attention to nouns
            for noun_pos in noun_positions:
                if noun_pos < i:
                    attention[i, noun_pos] += 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 10, Head 1
def first_token_bias_content_focus_punctuation_L10H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Strong first-token attention for all tokens
    for i in range(n):
        attention[i, 0] = 0.9  # Very high base attention to first token
    
    # Self-attention
    for i in range(n):
        attention[i, i] = 0.05  # Moderate self-attention
    
    # Add some contextual attention to recent tokens
    for i in range(1, n):
        # Attend to previous token with small weight
        if i > 0:
            attention[i, i-1] += 0.02
        # Attend to token 2 positions back with smaller weight  
        if i > 1:
            attention[i, i-2] += 0.01
    
    # Special handling for punctuation - they often attend more to content words
    for i in range(n):
        token = tokens[i]
        if token in ['.', '!', '?', ',', ';']:
            # Punctuation attends less to first token, more to nearby content
            attention[i, 0] *= 0.7
            for j in range(max(0, i-3), i):
                if tokens[j] not in ['.', '!', '?', ',', ';', ' ', "'s", "'t"]:
                    attention[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 10, Head 2
def first_token_bias_L10H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (except for first token itself)
        if i > 0:
            attention_matrix[i, 0] = 0.9
        
        # Self-attention
        attention_matrix[i, i] = 0.05
        
        # Very weak uniform attention to other previous tokens
        for j in range(1, i):
            if j != 0:  # Already set first token attention
                attention_matrix[i, j] = 0.01
    
    # Special case for first token - it attends to itself with weight 1.0
    attention_matrix[0, 0] = 1.0
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "first_token_bias_L10H2", attention_matrix


# Layer 10, Head 3
def decaying_first_token_bias_content_focus_punctuation_L10H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (except for first token itself)
        if i > 0:
            attention[i, 0] = 0.8
        
        # Self attention
        attention[i, i] = 0.3
        
        # Attention to previous tokens with decay
        for j in range(max(0, i-3), i):
            if j != 0:  # Already handled first token
                distance = i - j
                attention[i, j] = 0.1 / distance
        
        # Special handling for punctuation and end tokens
        token = tokens[i]
        if token in ['.', '!', '?', ',']:
            # Punctuation attends less to first token, more to recent content
            if i > 0:
                attention[i, 0] *= 0.3
            # Find recent content words
            for j in range(max(0, i-5), i):
                if tokens[j].strip() and not tokens[j] in [',', '.', '!', '?']:
                    attention[i, j] += 0.2
        
        # First token attends only to itself
        if i == 0:
            attention[i, :] = 0
            attention[i, i] = 1.0
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 10, Head 4
def first_token_bias_L10H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for most tokens
        if i > 0:
            attention[i, 0] = 0.7 + 0.2 * np.exp(-i * 0.1)  # Decay slightly with distance
        else:
            attention[i, 0] = 1.0
        
        # Self-attention
        attention[i, i] += 0.1
        
        # Syntactic relationships via spacy alignment
        if alignment[i]:  # If this GPT2 token aligns to spacy tokens
            for spacy_idx in alignment[i]:
                if spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    
                    # Find syntactic dependencies
                    syntactic_targets = []
                    
                    # Head relationships
                    if spacy_token.head != spacy_token:
                        syntactic_targets.append(spacy_token.head)
                    
                    # Children relationships  
                    for child in spacy_token.children:
                        syntactic_targets.append(child)
                    
                    # Map back to GPT2 tokens and add attention
                    for target in syntactic_targets:
                        target_idx = target.i
                        if target_idx < len(alignment):
                            # Find which GPT2 tokens this spacy token maps to
                            for gpt2_idx in range(n):
                                if gpt2_idx <= i and alignment[gpt2_idx] and target_idx in alignment[gpt2_idx]:
                                    attention[i, gpt2_idx] += 0.15
        
        # Local attention bias - prefer recent tokens
        for j in range(max(0, i-3), i):
            attention[i, j] += 0.05 * (1.0 - (i - j) * 0.1)
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L10H4", attention


# Layer 10, Head 5
def first_token_bias_content_focus_L10H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for early tokens
        if i <= 3:
            attention[i, 0] = 0.8 - 0.15 * i
        else:
            # Moderate first-token attention for later tokens
            attention[i, 0] = 0.1
        
        # Self-attention
        attention[i, i] = 0.05
        
        # Get spacy token info if available
        spacy_indices = gpt2_to_spacy[i]
        current_spacy_token = doc[spacy_indices[0]] if spacy_indices else None
        
        if current_spacy_token:
            # Syntactic attention patterns
            
            # If this is a verb, attend to nearby nouns (subjects/objects)
            if current_spacy_token.pos_ == "VERB":
                for j in range(max(0, i-5), i):
                    spacy_j = gpt2_to_spacy[j]
                    if spacy_j:
                        spacy_token_j = doc[spacy_j[0]]
                        if spacy_token_j.pos_ in ["NOUN", "PRON"]:
                            attention[i, j] += 0.15
            
            # If this is a noun, attend to modifying adjectives and determiners
            if current_spacy_token.pos_ == "NOUN":
                for j in range(max(0, i-3), i):
                    spacy_j = gpt2_to_spacy[j]
                    if spacy_j:
                        spacy_token_j = doc[spacy_j[0]]
                        if spacy_token_j.pos_ in ["ADJ", "DET"]:
                            attention[i, j] += 0.1
                        # Attend to prepositions
                        if spacy_token_j.pos_ == "ADP":
                            attention[i, j] += 0.05
            
            # Attend to conjunctions
            if current_spacy_token.pos_ == "CCONJ" or tokens[i].strip() in ["and", "but", "or"]:
                for j in range(i):
                    attention[i, j] += 0.02
            
            # From conjunctions, attend to coordinated elements
            for j in range(i):
                spacy_j = gpt2_to_spacy[j]
                if spacy_j:
                    spacy_token_j = doc[spacy_j[0]]
                    if spacy_token_j.pos_ == "CCONJ" or tokens[j].strip() in ["and", "but", "or"]:
                        attention[i, j] += 0.08
        
        # Token-specific patterns based on content
        token_text = tokens[i].strip().lower()
        
        # Punctuation attends to recent content
        if token_text in [".", ",", "!", "?"]:
            for j in range(max(0, i-5), i):
                attention[i, j] += 0.02
        
        # Prepositions and articles attend to nearby nouns
        if token_text in ["to", "with", "of", "in", "on", "the", "a", "an"]:
            for j in range(max(0, i-3), i+1):
                if j < n:
                    j_spacy = gpt2_to_spacy[j]
                    if j_spacy:
                        j_token = doc[j_spacy[0]]
                        if j_token.pos_ == "NOUN":
                            attention[i, j] += 0.05
        
        # Local attention - attend to previous few tokens
        for j in range(max(0, i-2), i):
            attention[i, j] += 0.03
        
        # Distant tokens get small uniform attention
        for j in range(i):
            attention[i, j] += 0.01
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 10, Head 6
def first_token_bias_L10H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # First token attends only to itself
    attention[0, 0] = 1.0
    
    # All other tokens attend primarily to first token with some self-attention
    for i in range(1, n):
        # Strong attention to first token (around 0.9-0.95)
        attention[i, 0] = 0.93
        
        # Weak self-attention (around 0.05-0.1)
        attention[i, i] = 0.07
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L10H6", attention


# Layer 10, Head 7
def first_token_bias_punctuation_coreference_L10H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Helper to check if a token is a name/pronoun
    def is_salient_entity(token_idx):
        spacy_indices = gpt2_to_spacy[token_idx]
        if not spacy_indices:
            return False
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                if spacy_token.pos_ in ['PROPN', 'PRON'] or spacy_token.ent_type_:
                    return True
        return False
    
    # Helper to check if token is punctuation
    def is_punctuation(token):
        return token.strip() in ['.',  ',', '!', '?', ':', ';', '"', "'", '(', ')']
    
    for i in range(n):
        # Base attention weights
        base_first_token = 0.8  # Strong first token attention
        base_self = 0.1
        base_prev = 0.05
        base_entity = 0.3
        
        # First token attention (very strong pattern)
        attention[i, 0] = base_first_token
        
        # Self attention
        attention[i, i] = base_self
        
        # Previous token attention
        if i > 0:
            attention[i, i-1] += base_prev
        
        # Attention to salient entities (names, pronouns)
        for j in range(i + 1):
            if j != 0 and j != i and is_salient_entity(j):
                attention[i, j] += base_entity
        
        # Special handling for punctuation tokens
        if is_punctuation(tokens[i]):
            # Punctuation tends to attend more to recent salient entities
            for j in range(max(0, i-5), i):
                if is_salient_entity(j):
                    attention[i, j] += 0.2
        
        # Local context (small attention to nearby tokens)
        for j in range(max(0, i-3), i):
            if j != 0 and j != i-1:  # Don't double-count first token and prev token
                attention[i, j] += 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 10, Head 8
def first_token_bias_punctuation_L10H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Very strong attention to first token (primary pattern)
        attention[i, 0] = 0.95
        
        # Moderate self-attention
        attention[i, i] = 0.03
        
        # Small attention to punctuation and conjunctions
        for j in range(i + 1):
            if j != 0 and j != i:  # Skip first token and self (already set)
                token = tokens[j].strip()
                if token in [',', '.', 'and', 'or']:
                    attention[i, j] = 0.015
                else:
                    attention[i, j] = 0.005
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 10, Head 9
def first_token_bias_content_focus_punctuation_stochastic_L10H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for most tokens
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0  # Self-attention for first token
        
        # Self-attention (moderate)
        if i > 0:
            attention[i, i] = 0.15
        
        # Previous token attention
        if i > 0:
            attention[i, i-1] = 0.3
        
        # For tokens beyond position 2, add some attention to position 1
        if i > 2:
            attention[i, 1] = 0.2
        
        # Special handling for punctuation
        token_text = tokens[i].strip()
        if token_text in ['!', '.', '?', ',']:
            # Punctuation attends more to recent content words
            for j in range(max(0, i-3), i):
                if tokens[j].strip().isalpha() and len(tokens[j].strip()) > 2:
                    attention[i, j] = 0.4
        
        # Semantic relationships via spacy
        spacy_indices = gpt2_to_spacy[i]
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # If this is a noun, attend to its modifiers
            if spacy_token.pos_ == "NOUN":
                for child in spacy_token.children:
                    if child.dep_ in ["amod", "det", "poss"]:
                        for k in range(i):
                            k_spacy = gpt2_to_spacy[k]
                            if k_spacy and child.i in k_spacy:
                                attention[i, k] = 0.6
            
            # If this is an adjective or determiner, attend to the noun it modifies
            if spacy_token.pos_ in ["ADJ", "DET"] and spacy_token.head.pos_ == "NOUN":
                for k in range(i+1, n):
                    k_spacy = gpt2_to_spacy[k]
                    if k_spacy and spacy_token.head.i in k_spacy:
                        attention[i, k] = 0.3
            
            # Verb-object relationships
            if spacy_token.pos_ == "VERB":
                for child in spacy_token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        for k in range(i+1, n):
                            k_spacy = gpt2_to_spacy[k]
                            if k_spacy and child.i in k_spacy:
                                attention[i, k] = 0.4
        
        # Add some random local attention for remaining positions
        for j in range(max(0, i-2), i):
            if attention[i, j] < 0.1:
                attention[i, j] += 0.05
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 10, Head 10
def first_token_bias_content_focus_L10H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Get important tokens (proper nouns, main verbs, content words)
    important_tokens = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        if spacy_indices:
            for s_idx in spacy_indices:
                if s_idx < len(doc):
                    token = doc[s_idx]
                    # Mark as important if proper noun, main verb, or key content
                    if (token.pos_ in ['PROPN', 'NOUN'] or 
                        (token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp']) or
                        token.ent_type_ in ['PERSON', 'ORG', 'GPE']):
                        important_tokens.add(i)
    
    for i in range(n):
        # Very strong attention to first token (dominant pattern)
        attention[i, 0] = 0.95
        
        # Self-attention (moderate)
        attention[i, i] = 0.02
        
        # Attention to important semantic tokens
        for j in important_tokens:
            if j <= i and j != 0:  # Respect causal mask, don't double-count first token
                attention[i, j] += 0.02
        
        # Small amount of local attention to previous tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                attention[i, j] += 0.005
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 10, Head 11
def decaying_first_token_bias_L10H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token (very dominant pattern)
        attention[i, 0] = 0.8
        
        # Self-attention with moderate weight
        attention[i, i] = 0.15
        
        # Local attention to previous tokens (decaying)
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                attention[i, j] = 0.05 / distance
        
        # Syntactic attention if we can align to spacy
        if alignment[i]:  # If this GPT2 token aligns to spacy tokens
            spacy_idx = alignment[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attention to syntactic head
                if spacy_token.head != spacy_token:
                    head_idx = spacy_token.head.i
                    # Find GPT2 tokens that align to this head
                    for k in range(i):
                        if alignment[k] and head_idx in [doc[idx].i for idx in alignment[k] if idx < len(doc)]:
                            attention[i, k] += 0.1
                
                # Attention from modifiers to heads
                for child in spacy_token.children:
                    if child.dep_ in ["amod", "compound"]:
                        child_idx = child.i
                        for k in range(i+1, n):
                            if k < len(alignment) and alignment[k] and child_idx in [doc[idx].i for idx in alignment[k] if idx < len(doc)]:
                                attention[k, i] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 11, Head 0
def first_token_bias_content_focus_L11H0(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse sentence for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    spacy_to_gpt2 = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong attention to first token for most positions
        if i > 0:
            attention[i, 0] = 0.3
        else:
            attention[i, 0] = 1.0  # First token attends to itself strongly
        
        # Self-attention
        attention[i, i] = 0.2
        
        # Local attention patterns - attend to previous tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                if distance == 1:
                    attention[i, j] = 0.25  # Strong attention to previous token
                elif distance == 2:
                    attention[i, j] = 0.15
                elif distance == 3:
                    attention[i, j] = 0.1
        
        # Syntactic attention patterns
        if gpt2_to_spacy[i]:
            spacy_idx = gpt2_to_spacy[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Attend to syntactic head
                if spacy_token.head != spacy_token and spacy_token.head.i < len(spacy_to_gpt2):
                    head_gpt2_indices = spacy_to_gpt2[spacy_token.head.i]
                    for head_idx in head_gpt2_indices:
                        if head_idx < i:  # Respect causal mask
                            attention[i, head_idx] += 0.15
                
                # If this is a verb, attend to subject
                if spacy_token.pos_ == "VERB":
                    for child in spacy_token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"] and child.i < len(spacy_to_gpt2):
                            subj_gpt2_indices = spacy_to_gpt2[child.i]
                            for subj_idx in subj_gpt2_indices:
                                if subj_idx < i:
                                    attention[i, subj_idx] += 0.12
                
                # If this is a modifier, attend to what it modifies
                if spacy_token.dep_ in ["amod", "advmod", "compound"]:
                    if spacy_token.head.i < len(spacy_to_gpt2):
                        head_gpt2_indices = spacy_to_gpt2[spacy_token.head.i]
                        for head_idx in head_gpt2_indices:
                            if head_idx < i:
                                attention[i, head_idx] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 11, Head 1
def decaying_first_token_bias_content_focus_punctuation_L11H1(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    attention_matrix = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token
        attention_matrix[i, 0] = 0.9
        
        # Moderate attention to early tokens (positions 1-3)
        for j in range(1, min(4, i + 1)):
            if j < n:
                attention_matrix[i, j] = max(0.1 - 0.02 * j, 0.02)
        
        # Self-attention (moderate)
        attention_matrix[i, i] = 0.05
        
        # Weak attention to other previous tokens with decay
        for j in range(4, i):
            if j < n:
                decay = max(0.01, 0.05 * np.exp(-0.3 * (j - 3)))
                attention_matrix[i, j] = decay
        
        # Special handling for punctuation and last tokens
        if i == n - 1:  # Last token (often punctuation)
            # Redistribute some attention to middle tokens
            mid_start = max(1, n // 3)
            mid_end = min(n - 1, 2 * n // 3)
            for j in range(mid_start, mid_end):
                attention_matrix[i, j] *= 2
            
            # Boost attention to specific content positions
            if n > 5:
                attention_matrix[i, min(5, n-1)] *= 3
    
    # Apply causal mask and normalize
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 11, Head 2
def first_token_bias_content_focus_punctuation_L11H2(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Get spacy alignment for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Identify important tokens (proper nouns, nouns, entities)
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
        # Base attention to first token (very strong pattern)
        if i == 0:
            attention[i, 0] = 1.0  # First token attends to itself completely
        else:
            attention[i, 0] = 0.8  # Strong attention to first token
            
            # Self-attention
            attention[i, i] = 0.1
            
            # Attention to important tokens
            for j in range(i):  # Only previous tokens due to causal mask
                if j in important_tokens and j != 0:
                    attention[i, j] = 0.05
                elif j != 0:  # Small residual attention to other tokens
                    attention[i, j] = 0.01
            
            # Special case for punctuation - higher self-attention
            current_token = tokens[i]
            if current_token in ['.', ',', '!', '?', ';', ':']:
                attention[i, i] = 0.15
                attention[i, 0] = 0.7  # Reduce first-token attention slightly
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 11, Head 3
def decaying_first_token_bias_content_focus_L11H3(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong self-attention baseline
        attention[i, i] = 0.1
        
        # Very strong attention to first token from early positions
        if i <= 3:
            attention[i, 0] = 0.8 - 0.15 * i
        else:
            attention[i, 0] = 0.05
        
        # Add syntactic and positional patterns
        spacy_indices = gpt2_to_spacy[i]
        
        for j in range(i):  # Only attend to previous tokens (causal)
            if j == 0:
                continue  # Already handled first token
                
            # Distance-based decay
            dist = i - j
            base_weight = 0.1 / (1 + 0.3 * dist)
            
            # Boost for immediate predecessor
            if dist == 1:
                base_weight *= 2.0
            
            # Syntactic boosting
            if spacy_indices and gpt2_to_spacy[j]:
                spacy_i = spacy_indices[0]
                spacy_j = gpt2_to_spacy[j][0]
                
                if spacy_i < len(doc) and spacy_j < len(doc):
                    tok_i = doc[spacy_i]
                    tok_j = doc[spacy_j]
                    
                    # Verb-subject relationships
                    if tok_i.pos_ == 'VERB' and tok_j.pos_ in ['NOUN', 'PRON'] and tok_j.dep_ in ['nsubj', 'nsubjpass']:
                        base_weight *= 3.0
                    
                    # Adjective-noun modification
                    elif tok_i.pos_ == 'ADJ' and tok_j.pos_ == 'NOUN' and tok_j.head == tok_i:
                        base_weight *= 2.5
                    elif tok_i.pos_ == 'NOUN' and tok_j.pos_ == 'ADJ' and tok_i.head == tok_j:
                        base_weight *= 2.5
                    
                    # Preposition-object relationships
                    elif tok_i.pos_ == 'ADP' and tok_j.dep_ == 'pobj':
                        base_weight *= 2.0
                    
                    # Conjunction relationships
                    elif tok_i.pos_ == 'CCONJ' and j > 0:
                        base_weight *= 1.5
                    
                    # Punctuation attending to clause boundaries
                    elif tokens[i] in ['.', ',', '!', '?']:
                        if tok_j.pos_ == 'VERB' or j == i - 1:
                            base_weight *= 2.0
            
            # Special handling for common patterns
            if tokens[j] in [',', '.'] and dist <= 3:
                base_weight *= 1.5
            
            attention[i, j] += base_weight
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 11, Head 4
def first_token_bias_punctuation_L11H4(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Very strong attention to first token (position 0)
        if i > 0:
            attention[i, 0] = 0.8  # High base weight for first token
        else:
            attention[i, 0] = 1.0  # Self-attention for first token
        
        # Self-attention (moderate weight)
        if i > 0:
            attention[i, i] = 0.1
        
        # Attention to previous tokens with recency bias
        for j in range(max(0, i-3), i):  # Look at up to 3 previous tokens
            if j > 0:  # Don't double-count first token
                distance = i - j
                weight = 0.05 / distance  # Decaying weight based on distance
                attention[i, j] += weight
        
        # Extra attention to punctuation tokens
        for j in range(i):
            token = tokens[j]
            if token in [',', '.', '!', '?', ';', ':']:
                attention[i, j] += 0.03
        
        # Attention to immediately previous token (if not first token)
        if i > 1:
            attention[i, i-1] += 0.02
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_punctuat", attention


# Layer 11, Head 5
def first_token_bias_content_focus_punctuation_L11H5(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    attention = np.zeros((n, n))
    
    for i in range(n):
        # Strong attention to first token (position 0)
        if i > 0:
            attention[i, 0] = 0.8
        else:
            attention[i, 0] = 1.0
        
        # Self-attention (moderate weight)
        if i > 0:
            attention[i, i] = 0.1
        
        # Backward attention to recent tokens
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                if distance == 1:
                    attention[i, j] = 0.05
                elif distance == 2:
                    attention[i, j] = 0.03
                else:
                    attention[i, j] = 0.02
        
        # Special handling for punctuation
        token = tokens[i]
        if token in [',', '.', '!', '?']:
            # Punctuation tends to attend more to nearby content words
            for j in range(max(0, i-5), i):
                if j != 0 and tokens[j].strip() and tokens[j] not in [',', '.', '!', '?']:
                    attention[i, j] += 0.02
        
        # For conjunction "and", slightly boost attention to nearby verbs/content
        if token.lower().strip() == 'and':
            for j in range(max(0, i-3), i):
                if j != 0:
                    attention[i, j] += 0.01
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 11, Head 6
def first_token_bias_L11H6(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    # Find proper nouns
    proper_noun_indices = set()
    for i, spacy_indices in enumerate(gpt2_to_spacy):
        for spacy_idx in spacy_indices:
            if spacy_idx < len(doc) and doc[spacy_idx].pos_ == 'PROPN':
                proper_noun_indices.add(i)
    
    for i in range(n):
        # Strong attention to first token (dominant pattern)
        attention[i, 0] = 0.95
        
        # Self attention (weaker)
        attention[i, i] = 0.02
        
        # Small attention to proper nouns if they exist and are accessible
        for prop_idx in proper_noun_indices:
            if prop_idx <= i and prop_idx != 0:  # causal and not first token
                attention[i, prop_idx] = 0.08
        
        # Very small local attention to previous tokens
        for j in range(max(0, i-2), i):
            if j != 0 and j not in proper_noun_indices:  # not first token or proper noun
                attention[i, j] = 0.01
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_L11H6", attention


# Layer 11, Head 7
def decaying_first_token_bias_content_focus_punctuation_stochastic_L11H7(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    for i in range(n):
        token = tokens[i]
        
        # Very strong attention to first token (except first token itself)
        if i > 0:
            attention[i, 0] = 0.85 + 0.1 * np.random.random()
        
        # Self-attention
        if token.strip() in [',', '.', '!', '?']:
            # Punctuation has lower self-attention
            attention[i, i] = 0.04 + 0.03 * np.random.random()
        else:
            attention[i, i] = 0.08 + 0.05 * np.random.random()
        
        # Recent context attention (exponential decay)
        for j in range(max(0, i-5), i):
            if j == 0:
                continue  # Already handled first token
            distance = i - j
            base_weight = 0.15 * np.exp(-0.5 * (distance - 1))
            
            # Boost for certain patterns
            curr_token = tokens[i].strip().lower()
            prev_token = tokens[j].strip().lower()
            
            # Punctuation patterns
            if curr_token in [',', '.'] and prev_token not in [',', '.']:
                base_weight *= 1.5
            
            # Add some randomness
            base_weight *= (0.8 + 0.4 * np.random.random())
            attention[i, j] = base_weight
        
        # Add small attention to syntactically related words
        if alignment[i]:  # If this GPT2 token aligns with spacy tokens
            spacy_idx = alignment[i][0]  # Take first aligned spacy token
            if spacy_idx < len(doc):
                spacy_token = doc[spacy_idx]
                
                # Find syntactic head
                if spacy_token.head != spacy_token and spacy_token.head.i < len(doc):
                    # Find GPT2 tokens that align with the syntactic head
                    for k in range(i):
                        if alignment[k] and spacy_token.head.i in alignment[k]:
                            attention[i, k] += 0.02 + 0.01 * np.random.random()
                
                # Find children/dependents
                for child in spacy_token.children:
                    if child.i < len(doc):
                        for k in range(i):
                            if alignment[k] and child.i in alignment[k]:
                                attention[i, k] += 0.015 + 0.01 * np.random.random()
        
        # Special handling for sentence-ending punctuation
        if token.strip() in ['.', '!', '?'] and i > 0:
            # Boost attention to important content words throughout the sentence
            for j in range(i):
                if j == 0:
                    continue  # Skip first token (already handled)
                if alignment[j]:  # If GPT2 token aligns with spacy tokens
                    for spacy_idx in alignment[j]:
                        if spacy_idx < len(doc):
                            spacy_token = doc[spacy_idx]
                            # Boost attention to nouns, verbs, and adjectives
                            if spacy_token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                                # Distance-based decay but with minimum boost
                                distance = i - j
                                boost = max(0.02, 0.06 * np.exp(-0.1 * distance))
                                boost *= (0.8 + 0.4 * np.random.random())
                                attention[i, j] += boost
    
    # Special handling for first token
    attention[0, 0] = 1.0
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 11, Head 8
def first_token_bias_content_focus_punctuation_L11H8(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    attention = np.zeros((n, n))
    
    # Parse with spacy for linguistic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    # Find important structural tokens
    comma_positions = []
    quote_positions = []
    first_comma = None
    
    for i, token in enumerate(tokens):
        if ',' in token:
            comma_positions.append(i)
            if first_comma is None:
                first_comma = i
        if '"' in token or "'" in token:
            quote_positions.append(i)
    
    # NEW: Identify quoted speech regions for enhanced attention
    quoted_regions = []
    if len(quote_positions) >= 2:
        for i in range(0, len(quote_positions) - 1, 2):
            start_quote = quote_positions[i]
            if i + 1 < len(quote_positions):
                end_quote = quote_positions[i + 1]
                quoted_regions.append((start_quote, end_quote))
    
    for i in range(n):
        token = tokens[i]
        
        # Strong self-attention (especially for early tokens and structural elements)
        if i < 3 or any(punct in token for punct in [',', '"', "'"]):
            attention[i, i] = 0.8
        else:
            attention[i, i] = 0.3
        
        # Attend to commas (especially first comma)
        for comma_pos in comma_positions:
            if comma_pos < i:
                if comma_pos == first_comma:
                    attention[i, comma_pos] = 0.4
                else:
                    attention[i, comma_pos] = 0.2
        
        # Attend to quotes
        for quote_pos in quote_positions:
            if quote_pos < i:
                attention[i, quote_pos] = 0.3
        
        # NEW: Enhanced attention within quoted speech regions
        for start_quote, end_quote in quoted_regions:
            if start_quote < i <= end_quote:
                # Tokens within quotes attend more strongly to the opening quote
                attention[i, start_quote] = 0.5
                # Also attend to other tokens within the same quoted region
                for j in range(start_quote + 1, i):
                    if start_quote < j < end_quote:
                        attention[i, j] += 0.1
        
        # Syntactic attention using spacy
        spacy_indices = alignment[i] if i < len(alignment) else []
        if spacy_indices:
            spacy_token = doc[spacy_indices[0]]
            
            # Attend to syntactic head
            if spacy_token.head != spacy_token:
                head_char_start = spacy_token.head.idx
                # Find GPT2 token that contains this character position
                char_pos = 0
                for j in range(i):
                    if char_pos <= head_char_start < char_pos + len(tokens[j]):
                        attention[i, j] += 0.2
                        break
                    char_pos += len(tokens[j])
            
            # Special patterns for different POS tags
            if spacy_token.pos_ == 'VERB':
                # Verbs attend to subjects and earlier clause elements
                for j in range(max(0, i-5), i):
                    attention[i, j] += 0.1
            
            elif spacy_token.pos_ in ['NOUN', 'PROPN']:
                # Nouns attend to their modifiers and determiners
                for j in range(max(0, i-3), i):
                    attention[i, j] += 0.05
        
        # Positional bias - attend to earlier tokens
        for j in range(i):
            distance_weight = 1.0 / (1.0 + (i - j) * 0.1)
            attention[i, j] += 0.02 * distance_weight
        
        # Special handling for sentence-final tokens
        if i == n - 1 or any(punct in token for punct in ['.', '!', '?']):
            # Attend back to key content words
            for j in range(i):
                if j in comma_positions or j in quote_positions:
                    attention[i, j] += 0.1
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "first_token_bias_content_", attention


# Layer 11, Head 9
def decaying_first_token_bias_L11H9(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    if n == 1:
        return tokens, np.array([[1.0]])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # First token attends to itself with weight 1.0
    attention[0, 0] = 1.0
    
    # All other tokens attend primarily to first token
    for i in range(1, n):
        # Strong attention to first token (base weight around 0.95)
        base_first_attention = 0.95
        
        # Slight decay for later tokens
        decay = min(0.05, i * 0.005)
        first_attention = base_first_attention - decay
        
        attention[i, 0] = first_attention
        
        # Distribute remaining attention
        remaining = 1.0 - first_attention
        
        # Self attention (small amount)
        self_attention = min(0.03, remaining * 0.3)
        attention[i, i] = self_attention
        remaining -= self_attention
        
        # Local attention to nearby previous tokens
        if remaining > 0:
            # Attend to previous 1-2 tokens with decreasing weights
            local_positions = []
            if i >= 1:
                local_positions.append(i - 1)
            if i >= 2:
                local_positions.append(i - 2)
            
            if local_positions:
                # Distribute remaining attention with preference for more recent tokens
                weights = [0.7, 0.3][:len(local_positions)]
                weights = np.array(weights)
                weights = weights * (remaining / weights.sum())
                
                for j, pos in enumerate(local_positions):
                    attention[i, pos] = weights[j]
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention


# Layer 11, Head 10
def decaying_first_token_bias_content_focus_punctuation_L11H10(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 1:
        return tokens, np.array([[1.0]])
    
    attention_matrix = np.zeros((n, n))
    
    # Get spacy parse for linguistic features
    doc = spacy_parse(sentence)
    gpt2_to_spacy = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Base weights
        weights = np.zeros(i + 1)  # Can only attend to positions 0 to i
        
        # Strong attention to first token
        if i > 0:
            weights[0] = 0.7
        
        # Self-attention
        weights[i] = 0.15
        
        # Attention to previous tokens with decay
        for j in range(max(0, i-3), i):
            if j != 0:  # Don't double-count first token
                distance = i - j
                weight = 0.1 / distance
                weights[j] = weight
        
        # Special handling for punctuation
        token_text = tokens[i].strip()
        if token_text in ['.', '!', '?', ',', ':', ';']:
            # Punctuation attends more to content words
            weights = np.zeros(i + 1)
            weights[i] = 0.2  # Self attention for punctuation
            
            # Find important content words to attend to
            for j in range(i):
                token_j = tokens[j].strip()
                if j == 0:
                    weights[j] = 0.3  # Still some first-token bias
                elif token_j and not token_j in [' ', 'the', 'a', 'an', 'and', 'or', 'but']:
                    # Content words get more attention
                    distance_factor = 1.0 / (i - j) if i > j else 1.0
                    weights[j] = 0.1 * distance_factor
        
        # Handle contractions and special tokens
        if "'" in tokens[i]:  # Contractions like "'t", "'s"
            if i > 1:
                weights = np.zeros(i + 1)
                weights[i-1] = 0.4  # Strong attention to preceding word
                weights[i] = 0.2    # Self attention
                weights[0] = 0.3    # First token
                
                # Distribute remaining weight
                remaining = 0.1
                for j in range(1, i-1):
                    weights[j] = remaining / max(1, i-2)
        
        # Normalize and add to matrix
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights[i] = 1.0  # Fallback to self-attention
            
        attention_matrix[i, :i+1] = weights
    
    # First token always attends to itself
    attention_matrix[0, 0] = 1.0
    
    # Apply causal mask and ensure row-stochastic
    attention_matrix = apply_causal_mask(attention_matrix)
    attention_matrix = make_row_stochastic(attention_matrix)
    
    return "decaying_first_token_bias", attention_matrix


# Layer 11, Head 11
def decaying_first_token_bias_L11H11(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    import numpy as np
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors="pt").input_ids[0])
    n = len(tokens)
    
    if n == 0:
        return tokens, np.array([])
    
    # Initialize attention matrix
    attention = np.zeros((n, n))
    
    # Parse with spacy for syntactic features
    doc = spacy_parse(sentence)
    alignment = _align_to_spacy(sentence, tokens)
    
    for i in range(n):
        # Strong first-token attention for early positions (0-3)
        if i <= 3:
            attention[i, 0] = 0.8 + 0.2 * (4 - i) / 4  # 0.8 to 1.0
        else:
            attention[i, 0] = 0.05  # Weak first-token attention for later positions
        
        # Self-attention
        attention[i, i] = 0.05 + 0.05 * (1.0 / (i + 1))  # Decreasing self-attention
        
        # Get spacy tokens for current GPT2 token
        spacy_indices = alignment[i] if i < len(alignment) else []
        
        for j in range(min(i, n)):  # Causal mask
            if i == j or j == 0:  # Skip self and first token (already handled)
                continue
                
            # Base attention with distance decay
            base_weight = 0.02 / (i - j + 1)
            
            # Syntactic relationships
            syntactic_boost = 0.0
            
            if spacy_indices and j < len(alignment):
                target_spacy_indices = alignment[j]
                
                for si in spacy_indices:
                    if si < len(doc):
                        current_token = doc[si]
                        
                        for tj in target_spacy_indices:
                            if tj < len(doc):
                                target_token = doc[tj]
                                
                                # Head-dependent relationships
                                if current_token.head == target_token:
                                    syntactic_boost = max(syntactic_boost, 0.15)
                                elif target_token.head == current_token:
                                    syntactic_boost = max(syntactic_boost, 0.10)
                                
                                # Adjacent token relationships
                                if abs(si - tj) == 1:
                                    syntactic_boost = max(syntactic_boost, 0.05)
            
            # Previous token attention (common pattern)
            if j == i - 1:
                attention[i, j] = base_weight + 0.03 + syntactic_boost
            else:
                attention[i, j] = base_weight + syntactic_boost
    
    # Apply causal mask and normalize
    attention = apply_causal_mask(attention)
    attention = make_row_stochastic(attention)
    
    return "decaying_first_token_bias", attention

