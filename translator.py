"""
HDC Polysemy Translator (English -> French)
===========================================
High-level neuro-symbolic translation service that translates complex
or ambiguous English sentences into French by applying 10,000-D
Associative Self-Attention (W = X @ X.T / D) to contextualize word manifolds.
"""

import re
import time
from typing import Dict, List, Tuple
import numpy as np
from hdc_interlingua import HDCInterlinguaCore


class HDCPolysemyTranslator:
    """
    Translates English sentences into French while resolving polysemic words
    using genuine 10,000-D Associative Self-Attention across word hypervectors.
    """

    def __init__(self, dim: int = 10000):
        self.core = HDCInterlinguaCore(dim=dim)

    def translate(self, sentence: str) -> Dict:
        """
        Translate an English sentence into French using HDC Self-Attention.
        Returns translation, attention matrix, pre/post attention resonance shifts,
        and sub-millisecond execution latency.
        """
        t0 = time.perf_counter()

        # Tokenize words cleanly
        raw_words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
        if not raw_words:
            return {
                "source_english": sentence,
                "translated_french": "",
                "primary_domain": "GENERAL",
                "domain_resonance": {},
                "polysemic_resolutions": [],
                "attention_matrix": [],
                "words": [],
                "latency_ms": 0.0,
            }

        # 1. Build base 10,000-D word vectors X
        base_vectors = [self.core.get_word_vector(w) for w in raw_words]

        # 2. Apply 10,000-D Associative Self-Attention: X_ctx = sign(X + beta * W @ X)
        contextualized_vectors, W = self.core.apply_self_attention(base_vectors, beta=3.0)

        # Build sentence context manifold from contextualized vectors
        sentence_hv = self.core.bundle(contextualized_vectors)

        # Compute overall sentence domain resonance
        domain_resonance: Dict[str, float] = {}
        for dom, vec in self.core.domain_vectors.items():
            sim = self.core.cosine_similarity(sentence_hv, vec)
            domain_resonance[dom] = round(sim, 4)

        sorted_domains = sorted(domain_resonance.items(), key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0]

        # 3. Translate words & trace Pre-Attention vs Post-Attention Polysemy Disambiguation
        translated_words = []
        polysemic_resolutions = []

        for idx, word in enumerate(raw_words):
            pre_vec = base_vectors[idx]
            post_vec = contextualized_vectors[idx]

            if word in self.core.polysemic_map:
                candidates = self.core.polysemic_map[word]

                # Evaluate pre-attention vs post-attention cosine similarity for each candidate sense domain
                pre_scores = {}
                post_scores = {}
                for dom in candidates.keys():
                    if dom in self.core.domain_vectors:
                        pre_sim = self.core.cosine_similarity(pre_vec, self.core.domain_vectors[dom])
                        post_sim = self.core.cosine_similarity(post_vec, self.core.domain_vectors[dom])
                        pre_scores[dom] = round(pre_sim, 4)
                        post_scores[dom] = round(post_sim, 4)

                # Winning domain is the candidate sense domain with highest post-attention resonance
                best_dom = max(post_scores.items(), key=lambda x: x[1])[0] if post_scores else primary_domain
                resolved_fr = candidates.get(best_dom, candidates.get("GENERAL", word))
                translated_words.append(resolved_fr)

                polysemic_resolutions.append(
                    {
                        "english_word": word,
                        "position": idx,
                        "selected_french": resolved_fr,
                        "winning_domain": best_dom,
                        "pre_attention_scores": pre_scores,
                        "post_attention_scores": post_scores,
                        "resonance_gain": round(
                            post_scores.get(best_dom, 0.0) - pre_scores.get(best_dom, 0.0), 4
                        ),
                    }
                )
            elif word in self.core.standard_translations:
                translated_words.append(self.core.standard_translations[word])
            else:
                translated_words.append(word)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "source_english": sentence,
            "translated_french": " ".join(translated_words),
            "primary_domain": primary_domain,
            "domain_resonance": domain_resonance,
            "polysemic_resolutions": polysemic_resolutions,
            "attention_matrix": np.round(W, 3).tolist(),
            "words": raw_words,
            "latency_ms": round(latency_ms, 3),
        }
