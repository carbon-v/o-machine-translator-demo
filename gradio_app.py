"""
O-Machine 10,000-D Interlingua Polysemy Engine — Gradio Showcase App
=====================================================================
Designed for Hugging Face Spaces (Free Gradio CPU Tier).
Demonstrates sub-2ms polysemy disambiguation using 10,000-D Vector-Symbolic
Associative Self-Attention (Zero-LLM, $0 cost).
"""

import gradio as gr
from translator import HDCPolysemyTranslator

translator = HDCPolysemyTranslator(dim=10000)


def run_translation_ui(sentence: str):
    if not sentence or not sentence.strip():
        return "--", "DOMAIN: GENERAL | Latency: 0.000 ms", "Please enter an English sentence."

    out = translator.translate(sentence)

    french_text = out["translated_french"]
    header_info = f"⚡ DOMAIN: {out['primary_domain']}  |  CPU Latency: {out['latency_ms']:.3f} ms (Zero-LLM)"

    # Format polysemic resolutions table/markdown
    resolutions = out["polysemic_resolutions"]
    if not resolutions:
        details_md = "*No ambiguous polysemous target words detected in this sentence.*"
    else:
        rows = [
            "| English Word | Translated French | Winning Domain | Pre-Attention | Post-Attention | Self-Attention Gain |",
            "| :--- | :--- | :--- | :---: | :---: | :---: |",
        ]
        for p in resolutions:
            wdom = p["winning_domain"]
            pre_s = p.get("pre_attention_scores", {}).get(wdom, 0.0)
            post_s = p.get("post_attention_scores", {}).get(wdom, 0.0)
            gain = p["resonance_gain"]
            gain_str = f"+{gain:.3f}" if gain >= 0 else f"{gain:.3f}"
            rows.append(
                f"| **\"{p['english_word']}\"** | `{p['selected_french']}` | **{wdom}** | `{pre_s:.3f}` | `{post_s:.3f}` | **`{gain_str}`** |"
            )

        details_md = (
            "### 10,000-D Associative Self-Attention Disambiguation\n\n"
            + "\n".join(rows)
            + "\n\n*Polysemy resolved via cosine resonance shift along invariant bipolar manifolds.*"
        )

    return french_text, header_info, details_md


# Custom styling for premium dark look
CUSTOM_CSS = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
}
"""

with gr.Blocks(title="O-Machine 10,000-D Interlingua Polysemy Engine", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        """
        # ⚡ O-Machine 10,000-D Interlingua Polysemy Engine
        **Based on the Neuro-Semiotic Architecture developed by [Martin Trajkow](https://www.linkedin.com/in/martin-trajkow/)** • [O-Machine Enterprise Causal Graph](https://o-machine.com)
        
        Disambiguate ambiguous English sentences and translate to French in **<2 milliseconds** using **10,000-D Vector-Symbolic Associative Self-Attention** ($W = X X^T / D$). Zero LLMs, zero autoregressive tokens, $0 inference cost.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            sentence_input = gr.Textbox(
                label="English Input Sentence",
                placeholder="Enter an English sentence with polysemic words (bank, cell, scale, crane)...",
                value="I deposited money and capital in the bank account.",
                lines=2,
            )
            submit_btn = gr.Button("⚡ Translate via 10,000-D VSA Algebra", variant="primary")

            gr.Examples(
                examples=[
                    ["I deposited money and capital in the bank account."],
                    ["We sat on the green grass by the river bank and watched the water."],
                    ["The prison guard locked the inmate in the cell."],
                    ["The organism tissue microscope revealed a healthy cell."],
                    ["We need to scale the server compute cloud infrastructure."],
                    ["We watched the construction crane lift steel and concrete building material."],
                ],
                inputs=[sentence_input],
                label="Quick Polysemy Benchmark Examples",
            )

        with gr.Column(scale=1):
            header_output = gr.Textbox(label="Manifold Classification & Speed", interactive=False)
            french_output = gr.Textbox(label="Translated French Output", interactive=False, lines=2)
            details_output = gr.Markdown(label="Self-Attention Diagnostic")

    submit_btn.click(
        fn=run_translation_ui,
        inputs=[sentence_input],
        outputs=[french_output, header_output, details_output],
    )
    sentence_input.submit(
        fn=run_translation_ui,
        inputs=[sentence_input],
        outputs=[french_output, header_output, details_output],
    )

    gr.Markdown(
        """
        ---
        © 2026 **[Martin Trajkow](https://www.linkedin.com/in/martin-trajkow/)** and **[O-Machine](https://o-machine.com)**. Licensed under the Apache License 2.0.
        """
    )

if __name__ == "__main__":
    demo.launch()
