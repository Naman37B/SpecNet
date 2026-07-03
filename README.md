# SpecNet: Distributed Speculative Validation

A decentralized peer-to-peer LLM inference engine designed to bypass network latency constraints via speculative decoding.

## Architecture

This monorepo is split into three core domains:
1. **`ai_core/`**: The Python/PyTorch inference engine handling token drafting, target verification, and rejection sampling.
2. **`p2p_network/`**: *(WIP)* The Go/libp2p daemon handling decentralized mesh networking.
3. **`control_plane/`**: *(WIP)* The React observability dashboard.

## Milestone 1: Local Speculative Decoding

Currently, the `ai_core` implements the speculative decoding rejection-sampling loop locally, bypassing the network entirely to validate the underlying math. We utilize Meta's OPT family (`opt-125m` as the drafter, `opt-350m` as the target).

### Quickstart (Linux / HP Omen)

1. **Navigate to the AI Core:**
```bash
cd SpecNet/ai_core
```

2. **Setup the Environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Run the Model:**
```bash
./run.sh
```

With these two files committed to GitHub, Milestone 1 is packaged and ready for your teammates to reproduce locally.

---

Next: Milestone 2 — initialize the Go module and scaffold the libp2p peer daemon, or map the libp2p network architecture first.
>>>>>>> a707cd7 (Milestone 1: add ai_core scipts, README, and run helper)
