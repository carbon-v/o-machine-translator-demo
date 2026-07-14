"""
O-Machine 10,000-D Interlingua Polysemy Engine — Gradio Showcase App
=====================================================================
Designed for Hugging Face Spaces.
Demonstrates sub-2ms polysemy disambiguation using 10,000-D Vector-Symbolic
Associative Self-Attention (Zero-LLM, $0 cost).
"""

# Backward-compatibility shim for Hugging Face Spaces environments
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolderShim:
            @staticmethod
            def save_token(token):
                pass
            @staticmethod
            def get_token():
                return None
            @staticmethod
            def delete_token():
                pass
        huggingface_hub.HfFolder = _HfFolderShim
except Exception:
    pass

# Backward-compatibility patch for Starlette >= 0.40 / FastAPI >= 0.115 with Gradio 4.x
try:
    import starlette.templating
    _orig_TemplateResponse = starlette.templating.Jinja2Templates.TemplateResponse
    def _patched_TemplateResponse(self, *args, **kwargs):
        if args and len(args) >= 1 and isinstance(args[0], str):
            request = kwargs.pop("request", {})
            return _orig_TemplateResponse(self, request, args[0], *args[1:], **kwargs)
        return _orig_TemplateResponse(self, *args, **kwargs)
    starlette.templating.Jinja2Templates.TemplateResponse = _patched_TemplateResponse
except Exception:
    pass

import gradio as gr
from translator import HDCPolysemyTranslator

try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    GPU_DECORATOR = lambda fn: fn

translator = HDCPolysemyTranslator(dim=10000)


@GPU_DECORATOR
def _zero_gpu_warmup():
    """Dummy decorated function to satisfy Hugging Face ZeroGPU startup health check."""
    return True


def run_translation_ui(sentence: str):
    if not sentence or not sentence.strip():
        return "DOMAIN: GENERAL | Latency: 0.000 ms", "--", "Please enter an English sentence."

    out = translator.translate(sentence)

    header_info = f"⚡ DOMAIN: {out['primary_domain']}  |  CPU Latency: {out['latency_ms']:.3f} ms (Zero-LLM)"
    french_text = out["translated_french"]

    # Format polysemic resolutions table/markdown
    resolutions = out["polysemic_resolutions"]
    if not resolutions:
        details_md = "### 10,000-D Associative Self-Attention Diagnostic\n*No ambiguous polysemous target words detected in this sentence.*"
    else:
        rows = [
            "| English Word | Selected French | Winning Domain Manifold | Pre-Attention Resonance | Post-Attention Resonance | Self-Attention Gain |",
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
            "### ⚡ 10,000-D Associative Self-Attention Disambiguation\n\n"
            + "\n".join(rows)
            + "\n\n*Polysemy resolved via cosine resonance shift along invariant bipolar manifolds ($W = X X^T / D$).*"
        )

    return header_info, french_text, details_md


# Crisp, high-contrast Native Light UI styling matching O-Machine aesthetic
CUSTOM_CSS = """
body, .gradio-container {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.omachine-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.omachine-header h1 {
    color: #0f172a !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    margin-bottom: 8px !important;
}

.omachine-header p {
    color: #475569 !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
}

.omachine-badge {
    display: inline-block;
    background: #e0f2fe;
    color: #0284c7;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}
"""

with gr.Blocks(title="O-Machine 10,000-D Interlingua Polysemy Engine", css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <div class="omachine-header">
            <span class="omachine-badge">⚡ o-machine Native Translation Demo</span>
            <h1>10,000-D Interlingua Polysemy Engine</h1>
            <p style="margin-top: 6px !important; font-weight: 600;">
                Based on the Neuro-Semiotic Architecture developed by 
                <a href="https://www.linkedin.com/in/martin-trajkow/" target="_blank" style="color: #0284c7; text-decoration: none;">Martin Trajkow</a> • 
                <a href="https://o-machine.com" target="_blank" style="color: #0f172a; text-decoration: underline;">o-machine</a>
            </p>
            <p style="margin-top: 10px !important; color: #334155 !important;">
                Disambiguate ambiguous English sentences and translate to French in <strong>&lt;2 milliseconds</strong> using 
                <strong>10,000-D Vector-Symbolic Associative Self-Attention</strong> (<i>W = X X<sup>T</sup> / D</i>). 
                Zero LLMs, zero autoregressive tokens, $0 inference cost.
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=11):
            sentence_input = gr.Textbox(
                label="English Input Sentence",
                placeholder="Enter an English sentence with polysemic words (bank, cell, scale, crane)...",
                value="I deposited money and capital in the bank account.",
                lines=3,
            )
            submit_btn = gr.Button("⚡ Translate via 10,000-D VSA Algebra", variant="primary")

            header_output = gr.Textbox(
                label="Manifold Classification & Speed",
                interactive=False,
            )
            french_output = gr.Textbox(
                label="Translated French Output",
                interactive=False,
                lines=2,
            )

        with gr.Column(scale=13):
            details_output = gr.Markdown(
                label="Self-Attention Diagnostic",
                value="### ⚡ 10,000-D Associative Self-Attention Diagnostic\n*Click an example below or enter a sentence to inspect pre/post attention resonance shifts.*",
            )

    examples_block = gr.Examples(
        examples=[
            [
                "Here from the king's mountain view, here from a wild dream come true, feast like a sultan I do, of treasures and flesh never few."
            ],
            ["I deposited money and capital in the bank account."],
            ["We sat on the green grass by the river bank and watched the water."],
            ["The prison guard locked the inmate in the cell."],
            ["The organism tissue microscope revealed a healthy cell."],
            ["We need to scale the server compute cloud infrastructure."],
            ["We watched the construction crane lift steel and concrete building material."],
        ],
        inputs=[sentence_input],
        outputs=[header_output, french_output, details_output],
        fn=run_translation_ui,
        run_on_click=True,
        label="Click Any Benchmark Phrase to Translate Immediately",
    )

    submit_btn.click(
        fn=run_translation_ui,
        inputs=[sentence_input],
        outputs=[header_output, french_output, details_output],
    )
    sentence_input.submit(
        fn=run_translation_ui,
        inputs=[sentence_input],
        outputs=[header_output, french_output, details_output],
    )

    gr.HTML(
        """
        <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b; font-size: 0.88rem;">
            © 2026 <strong><a href="https://www.linkedin.com/in/martin-trajkow/" target="_blank" style="color: #0284c7; text-decoration: none;">Martin Trajkow</a></strong> and 
            <strong><a href="https://o-machine.com" target="_blank" style="color: #0f172a; text-decoration: none;">O-Machine</a></strong>. 
            Licensed under the Apache License 2.0.
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch()
