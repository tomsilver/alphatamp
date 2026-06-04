# Archive — historical SPECTRE spec documents

These are the original spec documents, moved here **byte-unchanged** from the
repo root during the 2026-06 spectre silo refactor. They are historical: the
living, consolidated version is [`../proposal.md`](../proposal.md). Where a
document below disagrees with `proposal.md`, the proposal wins. Section
references like "METHOD §4.1.4" in code docstrings resolve against these files.

| File | One-line historical note |
|---|---|
| `SPECTRE_METHOD_SPEC.md` | Original method spec (Φ/Ψ/σ, PL loss, F-subset rules) targeting the five kinder envs; partially superseded by the RT2D spec. |
| `SPECTRE_RT2D_METHOD_SPEC.md` | RT2D-adapted method/training spec (v1.0); authoritative over the original method spec for the RT2D evaluation; introduced fixes 1–5. |
| `SPECTRE_TRAINING_PIPELINE_SPEC.md` | Original three-layer data-pipeline design (raw episodes / parquet derived / online examples). |
| `SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` | What was actually built vs the pipeline spec — Layer 2 collapsed, live-object schema, divergence log (last synced 2026-04-24). |
| `SPECTRE_EDA_SPEC.md` | Pre-training EDA gates: Group 1 sanity, baselines B1–B5, adaptive premium Δ and headroom H, pass bar. |
| `ROUTED_TRANSPORT2D_SPEC.md` | RoutedTransport2D environment spec (v1): K₃,₃ topology, scene latent, per-problem tags, three-gate refiner. |
| `SYNTHETIC_ENVIRONMENT.md` | Motivation memo for building RT2D — why the kinder kinematic2d envs let a lookup-table baseline win, and the required properties of a replacement env. |
