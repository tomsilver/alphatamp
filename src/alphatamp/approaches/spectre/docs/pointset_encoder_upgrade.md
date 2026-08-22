# PointSetEncoder upgrade — implementation guide

**Status: IMPLEMENTED 2026-08-18** (code + tests + smoke-verify; the 3-seed retrain of §7 is
deferred). As-built + the three deviations from this doc are in
[`decisions/07` 2026-08-18](decisions/07-stickbutton2d.md#2026-08-18-pointsetencoder-upgrade-per-point-differential-features-edgeconv)
/ [`notebook/07` 2026-08-18](notebook/07-stickbutton2d.md#2026-08-18-pointsetencoder-upgrade-built-smoke-verified-dd2d-sb2d).
**Deviations:** (a) config-off equivalence is by *module selection* (config-off builds the v1
`FootprintEncoder`), so T1 is exact; (b) EdgeConv keeps the residual LayerNorm, so **T7 is
relaxed** to "zeroed `out_proj` ⇒ output = `LayerNorm(h)`", not the exact identity; (c) the 3D
orientation oracle is **away-from-origin** (convex origin-centered box), the doc's sensor
viewpoint being unavailable. `pma_seeds` defaults to 1 (not 4) to preserve config-off
byte-identity; and (d) the **3D `edgeconv_k` default is 6, not this doc's 16** — measured
on the real analytic box clouds, `k=16` spans opposite walls and corrupts the normal/`f`,
while `k=6` cleans the well-proportioned boxes (the tall block is PCA-degenerate at 32 pts
regardless, but its F3 signal is height-in-coords, not curvature). Everything below is
standard technique (PCA normals: PCL / Rusu et al.;
EdgeConv: DGCNN, Wang et al. 2019; multi-seed PMA: Lee et al. 2019); the only registered
*predictions* are the expected test values in §6 and the guardrail in §7.

**Goal.** Close the two gaps in `FootprintEncoder`: (1) the information gap — ring
adjacency is discarded at tensorization, so area/curvature are not well-posed functions of
the input; (2) the inductive-bias gap — PMA-over-φ(points) with no interaction stage caps
the encoder at weighted low-order moments. Fix: per-point **local differential features**
(computed in the tensorizer) + one **EdgeConv** interaction layer + **multi-seed PMA**.
Dimension-generic by construction (no ring order used anywhere in the deployed path).

---

## 0. Design constraints (binding)

1. **Additive, not destructive.** `FootprintEncoder` (v1) stays untouched; new module
   `PointSetEncoder` selected by config. Config-off must reproduce the v1 forward exactly.
2. **Dimension-generic.** Every feature and layer must have a stated 2D and 3D form (§5).
   No ring-order dependence in the deployed path (order is used only in dev-time
   cross-checks, §6).
3. **Canonicalization from the point set itself.** No simulator frames, no family labels.
   The one permitted oracle is an *inside test* (§2.3), which deployment supplies from a
   segmentation mask (2D) or sensor viewpoint (3D).
4. **Scene interface frozen.** `D_DESCRIPTOR = 32` and the scene-token schema are
   unchanged — the upgrade is entirely inside the per-object descriptor pathway.
5. **Every addition independently switchable** (`use_pca_feats`, `use_edgeconv`,
   `use_point_sab`, `pma_seeds`) so the paper ablation is free later; all switches recorded
   on the checkpoint and read back by `inference.load_checkpoint` (§4), never from the
   caller.

---

## 1. What changes at a glance

| Component | Change | Where |
|---|---|---|
| Tensorizer | per-point features `[p; n; κ̂; f]` + precomputed `knn_idx` | `dataset` (`build_example` path) |
| Batch schema | `point_feats (B,N,P,C_pt)`, `knn_idx (B,N,P,k)` — trailing, nullable | batch dataclass |
| Encoder | new `PointSetEncoder`: lift MLP → EdgeConv ×1 → [SAB ×1] → PMA(k=4) → 32-d | `encoders.py` (additive) |
| Config | 4 switches + `edgeconv_k` | model config + checkpoint plumbing |
| Tests | §6 | new test module |

Parameter cost (approximate): lift ≈ 2k, EdgeConv ≈ 12k, optional SAB ≈ 50k, 4-seed PMA +
projection ≈ 10k → **~25k added without SAB, ~75k with**, against the 324k deployed model.

---

## 2. Tensorizer: per-point differential features

All computed **per object, in the item frame, from the raw point set** — identical code
path for a 2D boundary ring and a 3D cloud except where §5 says otherwise. Coordinates
stay **absolute item-frame** (not scale-normalized): object size is signal in a packing
domain.

### 2.1 kNN

Euclidean kNN over the object's own points. `k = 4` in 2D (P = 32), `k = 16` in 3D.
Precompute indices in the tensorizer → `knn_idx (B, N, P, k)` int tensor. Rationale for
Euclidean (not arc) neighborhoods: on a dense ring they coincide with arc-adjacency
*except at pinches and narrow openings*, where the neighborhood pulls in the opposite wall
— which makes narrow openings directly visible to EdgeConv. (This side effect is a
prediction, not a measured result.)

### 2.2 Local PCA frame

For point $p_i$ with neighborhood $N_i$ (the k neighbors, including $p_i$): center, form
the covariance $C_i$, eigendecompose, eigenvalues $\lambda_1 \ge \dots \ge \lambda_d$.

- **2D:** tangent $t_i = v_1$, normal $n_i = v_2$; flatness $f_i = \lambda_2 / (\lambda_1 + \lambda_2) \in [0, 0.5]$.
- **3D:** normal $n_i = v_3$; surface variation $f_i = \lambda_3 / \sum_j \lambda_j \in [0, 1/3]$ (Pauly et al. 2002).

Guard: `eps = 1e-9` on eigenvalue sums; duplicate-point neighborhoods fall back to
$f_i = 0$, $n_i$ from the previous valid point (assert this never fires on clean sim data).

### 2.3 Orientation (sign disambiguation)

PCA gives $n_i$ up to sign. Fix it **outward** with an inside test:
flip $n_i$ if $p_i + \varepsilon\, n_i$ is inside the object, $\varepsilon = 0.5\,\bar h_i$
where $\bar h_i$ = mean distance to the kNN set.

- Sim / DD2D: Shapely `contains` on the source polygon (tensorizer-time only; the polygon
  never enters the model input).
- Deployment 2D: segmentation-mask interior lookup.
- Deployment 3D: orient toward the sensor viewpoint (the standard PCL convention), i.e.
  flip if $n_i \cdot (v_{\text{cam}} - p_i) < 0$.

Then in 2D set $t_i = \mathrm{rot}_{90°\,\text{CCW}}(n_i)$ so the frame orientation is
consistent everywhere. **Do not** derive orientation from ring winding — that violates
constraint 3 and breaks on unordered deployment clouds.

### 2.4 Signed curvature (2D)

Local quadric fit in the PCA frame: for each neighbor $p_j$, coordinates
$u_j = (p_j - p_i)\cdot t_i$, $v_j = (p_j - p_i)\cdot n_i$; least-squares fit
$v \approx b\,u + a\,u^2$ (the linear term absorbs tangent misestimate); define

$$\kappa_i = -2a.$$

**Sign convention (stated once, tested in §6):** with the outward normal, **convex
segments have $\kappa > 0$, reflex/pocket segments have $\kappa < 0$.** Sanity derivation:
circle of radius $r$, point $(r, 0)$, $n = (1,0)$, $t = (0,1)$; a neighbor at angle
$\varphi$ has $u = r\sin\varphi$, $v = r\cos\varphi - r \approx -u^2/2r$, so
$a = -1/2r$ and $\kappa = 1/r > 0$. ✓

Normalize dimensionless and bounded: $\hat\kappa_i = \tanh(\kappa_i \,\bar h_i)$.
(For the P = 32 ring, $\kappa \bar h$ ≈ the per-step turning angle: a circle gives
$\approx 2\pi/32 \approx 0.2$ everywhere — comfortably in tanh's linear range; corner
spikes saturate gracefully.)

3D minimal version: **no signed curvature** — ship coordinates + oriented normal +
surface variation only. (A signed mean-curvature proxy via full quadric fit is a
later, optional column; do not block on it.)

### 2.5 Assembled per-point feature vector

$$\text{2D } (C_{pt}=6):\ [\,x,\ y,\ n_x,\ n_y,\ \hat\kappa,\ f\,] \qquad
  \text{3D } (C_{pt}=8):\ [\,x,\ y,\ z,\ n_x,\ n_y,\ n_z,\ f,\ 0\,]$$

(Pad 3D to a common width if a shared schema is wanted; otherwise let `C_pt` vary with
`d` — either is fine, pick one and record it in the checkpoint config.)

### 2.6 Padding

Padded objects: `point_feats = 0`, `knn_idx = 0`, existing object mask governs. Keep the
`pmask` pathway alive even though P = 32 is currently all-real — 3D partial clouds will
need variable P.

---

## 3. Encoder: `PointSetEncoder` (additive module)

```python
class PointSetEncoder(nn.Module):
    # input: point_feats (B*N, P, C_pt), knn_idx (B*N, P, k), pmask (B*N, P)
    # output: descriptor (B*N, D_DESCRIPTOR=32)

    lift    = MLP(C_pt -> 32 -> D_MODEL=64, GELU)      # per-point, shared
    edge    = EdgeConv(D_MODEL, k)                      # if use_edgeconv
    sab     = SetAttentionBlock(D_MODEL, heads=4)       # if use_point_sab
    pma     = PMA(D_MODEL, heads=4, seeds=pma_seeds)    # default seeds=4
    proj    = Linear(D_MODEL * pma_seeds -> 32)
```

**EdgeConv** (DGCNN form, fixed precomputed graph, one layer):

```python
def edgeconv(h, knn_idx):                    # h: (B*N, P, 64)
    h_j = gather(h, knn_idx)                 # (B*N, P, k, 64)
    msg = mlp(cat([h_i.expand, h_j - h_i]))  # 128 -> 64 -> 64, GELU
    agg = msg.max(dim=k)                     # max-aggregation (DGCNN standard)
    return layer_norm(h + out_proj(agg))     # residual; out_proj zero-init
```

Notes, in order of importance:

1. **Fixed graph.** `knn_idx` comes from the tensorizer (coordinate-space kNN), *not*
   recomputed in feature space per layer as in full DGCNN. One layer, one fixed graph —
   deterministic, cheap, and sufficient at P = 32.
2. **Zero-init the residual projection** (`out_proj`), so at initialization the EdgeConv
   branch is the identity and the function class strictly nests the no-EdgeConv model.
   Training is from scratch (no v1 warm-start — matches project convention), so this is a
   stability measure, not a fine-tuning trick.
3. **Masking.** Neighbors that are padding contribute `-inf` before the max (2D never
   triggers this today; 3D will).
4. **SAB stays optional and off by default.** It restores the canonical Set-Transformer
   encoder-then-PMA form and covers *global* relational features (hull gap, diameter),
   but it is the largest parameter line item; turn it on only if the EdgeConv-only
   variant regresses (§7).
5. **PMA seeds = 4**, outputs concatenated then projected to 32. `seeds = 1` +
   `use_pca_feats = False` + `use_edgeconv = False` must reproduce the v1 forward
   bit-for-bit modulo naming (§6, test T1).

Downstream is untouched: the 32-d descriptor drops into the existing scene-token slot.

---

## 4. Schema and checkpoint plumbing

Follow the existing deploy-switch discipline: everything that changes what
`dataset.build_example` *emits* (`use_pca_feats`, `edgeconv_k`, and the `C_pt` layout)
is recorded in the checkpoint config and read back by `inference.load_checkpoint` — a
model is never scored on features it was not trained on. Batch fields `point_feats` /
`knn_idx` are **trailing nullable** additions (the established additive-schema pattern);
old pickles load via the existing migration-shim mechanism with the new fields `None`,
and the v1 path ignores them.

---

## 5. 2D → 3D lift table

| Ingredient | 2D (boundary ring) | 3D (surface cloud) |
|---|---|---|
| points | 32-pt arc-length ring, item frame | P points via farthest-point sampling |
| kNN | k = 4, Euclidean | k = 16, Euclidean |
| frame | tangent $v_1$, normal $v_2$ | normal $v_3$ |
| bending | signed $\hat\kappa$ (quadric fit) | surface variation $\lambda_3/\sum\lambda$ (unsigned) |
| orientation | inside test (polygon / mask) | sensor-viewpoint (PCL convention) |
| EdgeConv / SAB / PMA | identical code | identical code |

The deployed path consumes only the point set + the inside/viewpoint oracle — no ring
order, no mesh connectivity — which is what makes the module the concrete
`PointSetEncoder(d, N)` contract.

---

## 6. Tests

- **T1 — config-off equivalence.** All switches off + `seeds = 1` ⇒ output equals the v1
  `FootprintEncoder` forward on identical inputs.
- **T2 — feature correctness on analytic shapes.** Circle radius $r$: $\hat\kappa \approx \tanh(2\pi/32)$
  at every point, normals radial-outward, $f \approx 0$. Axis-aligned square: $\hat\kappa \approx 0$
  mid-edge, positive spikes at corners. Horseshoe: $\hat\kappa < 0$ strictly inside the pocket, $> 0$
  on the outer arc.
- **T3 — PCA vs exact ring cross-check (dev-only, may use ring order).** PCA tangents
  within tolerance of exact ring tangents on all 7 DD2D families; discrete Gauss–Bonnet
  on exact ring turning angles ($\sum \theta_i = 2\pi$) as a tensorizer data-validation
  assert.
- **T4 — end-to-end permutation invariance.** Permute the input points (permuting
  `knn_idx` contents consistently) ⇒ descriptor unchanged. Holds because PCA, Euclidean
  kNN, the inside test, and max-aggregation are all order-free.
- **T5 — orientation robustness.** Randomly pre-flip PCA eigenvector signs ⇒ identical
  features after the inside-test disambiguation.
- **T6 — existing regressions retained.** Object-order invariance of scene logits and
  the anti-collapse test, re-run against the new encoder.
- **T7 — nesting.** With `out_proj` weights zeroed at runtime, EdgeConv-on equals
  EdgeConv-off exactly.

---

## 7. Retrain and guardrail

Retrain from scratch under the standard protocol (3 checksum-distinct seeds, existing
rollout-based checkpoint selection, same data). Single comparison against the deployed
model: rollout FP with paired CIs, reported per stratum. **Guardrail:** any regression on
the easy/convex-dominated strata beyond CI reverts the responsible switch (they are
independently revertible by design). Re-run the elimination ladder once on the final
config to confirm the residual signal did not collapse into a slack/area proxy — the
upgrade makes area *easier* to compute, so the within-length loss + ladder remain the
license for it.

Expected but unverified: the `[area, sin θ, cos θ]` scalars become redundant under this
encoder. Do **not** remove them in the same change — that is a second variable; schedule
removal as its own one-line follow-up after the retrain lands.

---

## 8. Explicitly deferred (out of scope here)

- Per-point convex-hull-gap feature (non-local concavity) — only if pinch-sensitive
  failures persist after EdgeConv.
- Circular 1D conv over the ring — 2D-only, breaks the dimension-generic contract;
  permanently deprioritized.
- Self-supervised pretraining of the encoder on procedurally generated shapes —
  separate proposal, separate ablation.
- Signed curvature in 3D (quadric fit) — optional later column.
- Multi-stage / hierarchical grouping (PointNet++ depth), Point-Transformer backbones —
  unjustified at ~550 episodes (PointNeXt: recipe, not architecture, closed that gap).
