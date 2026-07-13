#!/usr/bin/env python3
"""
HDC 10,000-D Interlingua Polysemy Showcase (CLI Demo)
=====================================================
Demonstrates sub-2ms polysemic disambiguation using genuine 10,000-D
Associative Self-Attention across word manifolds.
"""

import sys
from translator import HDCPolysemyTranslator


def run_cli_demo():
    print("====================================================================")
    print("   O-MACHINE HDC INTERLINGUA — POLYSEMY TRANSLATOR SHOWCASE         ")
    print("   Based on the Neuro-Semiotic Architecture by Martin Trajkow       ")
    print("   https://o-machine.com                                            ")
    print("====================================================================")
    print("-> Initializing 10,000-D Bipolar Manifold & Associative Attention...")
    translator = HDCPolysemyTranslator(dim=10000)
    print("✓ 10,000-D Vector-Symbolic Self-Attention Core ready.\n")

    test_sentences = [
        "I deposited money and capital in the bank account.",
        "We sat on the green grass by the river bank and watched the water.",
        "The prison guard locked the inmate in the cell.",
        "The organism tissue microscope revealed a healthy cell.",
        "We need to scale the server compute cloud infrastructure.",
        "We watched the construction crane lift steel and concrete building material.",
    ]

    for idx, sentence in enumerate(test_sentences, 1):
        out = translator.translate(sentence)
        print(f"[{idx}] English : {out['source_english']}")
        print(f"    French  : {out['translated_french']}")
        print(f"    Domain  : {out['primary_domain']} (Latency: {out['latency_ms']:.3f} ms)")

        if out["polysemic_resolutions"]:
            for poly in out["polysemic_resolutions"]:
                wdom = poly["winning_domain"]
                pre_s = poly["pre_attention_scores"].get(wdom, 0.0)
                post_s = poly["post_attention_scores"].get(wdom, 0.0)
                gain = poly["resonance_gain"]
                print(
                    f"    ⚡ Self-Attention Disambiguation: '{poly['english_word']}' -> '{poly['selected_french']}'\n"
                    f"       Resonance ({wdom}): Pre-Attention [{pre_s:.3f}] -> Post-Attention [{post_s:.3f}] (Gain: +{gain:.3f})"
                )
        print("-" * 68)


if __name__ == "__main__":
    run_cli_demo()
