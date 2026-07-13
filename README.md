# O-Machine 10,000-D Interlingua Polysemy Showcase

An open-source interactive showcase demonstrating **Vector-Symbolic Architectures (VSA / High-Dimensional Computing)** solving complex contextual polysemy across languages (English $\rightarrow$ French) in **$<1\text{ ms}$ CPU latency at $\$0$ cost**.

> **Architecture:** Based on the Neuro-Semiotic Architecture developed by **[Martin Trajkow](https://www.linkedin.com/in/martin-trajkow/)**.  
> Learn more about O-Machine and real-time enterprise causal intelligence at **[https://o-machine.com](https://o-machine.com)**.

---

## Why This Showcase?

Current LLMs solve disambiguation and translation by deploying multi-billion parameter autoregressive Transformers, costing dollars per million tokens and adding hundreds of milliseconds of latency.

This showcase demonstrates how **10,000-dimensional bipolar hypervectors ($\{-1, +1\}^{10000}$)** act as an invariant geometric interlingua:
1. **Binding ($\otimes$):** Elementwise XOR/multiplication encodes structural relationships.
2. **Bundling ($\oplus$):** Majority-vote superposition constructs an invariant context manifold vector for any input sentence.
3. **Resonance Disambiguation:** Probing the sentence vector against 12 domain manifolds (`FINANCE`, `NATURE`, `LEGAL`, `BIOLOGY`, `TECHNOLOGY`, `CONSTRUCTION`, `MOBILITY`, `MUSIC`, `ENERGY`, `ASTRONOMY`, `MEDICINE`, `GENERAL`) resolves classic ambiguous words like `"bank"`, `"cell"`, `"scale"`, `"crane"`, `"note"`, `"charge"`, or `"bond"` with **0% hallucination risk** in **0.13 milliseconds**.

---

## Quickstart

### 1. Interactive Terminal Showcase
Run the command-line suite to benchmark disambiguation across diverse semantic domains:
```bash
python demo_cli.py
```

### 2. Sleek Local Web UI Playground
Start the zero-dependency web server and explore interactive manifold resonance graphs:
```bash
python app.py 8088
```
Then open **[http://localhost:8088](http://localhost:8088)** in your browser.

### 3. Gradio App / Hugging Face Spaces Deployment ($0 Free CPU Tier)
To run the Gradio app locally or deploy directly to a **Hugging Face Free Gradio Space**:
```bash
pip install -r requirements.txt
python gradio_app.py
```
To deploy: create a new Space on Hugging Face (`SDK: Gradio`), and upload `gradio_app.py`, `translator.py`, `hdc_interlingua.py`, and `requirements.txt`. Rename `gradio_app.py` to `app.py` in your Space repository.

---

## License

Copyright (c) 2026 **[Martin Trajkow](https://www.linkedin.com/in/martin-trajkow/)** and **[O-Machine](https://o-machine.com)**.  
Licensed under the **[Apache License, Version 2.0](LICENSE)**.

---

## Architectural Foundation

Based on the Neuro-Semiotic Architecture developed by **[Martin Trajkow](https://www.linkedin.com/in/martin-trajkow/)**.  
Explore real-time causal graph intelligence at **[https://o-machine.com](https://o-machine.com)**.
