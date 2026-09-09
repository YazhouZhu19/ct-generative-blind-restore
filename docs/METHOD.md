# Method and implementation

## 1. Scope

The workflow targets single-channel industrial CT images containing two dense,
approximately vertical lamella stacks separated by a central block. The goal is
strong visual denoising without silently changing the source canvas or freely
inventing measurable boundaries.

The generated image is treated as an appearance proposal. A blind-denoised image
is treated as a geometry measurement guide. Neither one is declared ground truth.

## 2. Ordered workflow

### 2.1 Immediate source-size lock

The external generated candidate is converted to grayscale and, if necessary,
resampled exactly once with Lanczos to `(source_width, source_height)`. There is
no aspect-ratio crop, padding, enhancement, or geometry operation before this
step. A relative aspect-ratio mismatch above 1% is rejected by default rather
than stretched silently. If dimensions already match, the 8-bit pixels are
copied exactly.

### 2.2 Blind-denoised measurement guide

The fast path accepts a previously computed native-size guide. The full path
trains `sota_geometry_blind.py` on the current CT image. Its core is a
self-supervised re-visible blind denoiser with:

- Poisson-Gaussian noise estimation;
- sub-lattice blind masking;
- a clipped low-frequency generator prior in reliable low-gradient regions;
- transverse edge/width and longitudinal endpoint losses;
- post-inference data/geometry projection;
- candidate rejection using measurement and boundary guardrails.

Generated pixels are not written into the measurement guide. The generated
candidate contributes only through a deliberately low-frequency training term.

### 2.3 Geometry measurement

For each left/right lamella, the guide supplies:

- center coordinate and local pitch;
- a tracked ridge path over image rows;
- top and bottom endpoints;
- body length;
- median plate width;
- inter-layer spacing and confidence/uncertainty values.

Detections in the guide and generated image are matched with dynamic programming.
The correspondence is monotonic, preventing ridge identities from crossing.
When the generated detector has extra ridges, only generated detections may be
skipped; every reliable guide ridge receives a match.

### 2.4 Native-canvas continuous 2-D guidance

For each scanline, matched guide paths are fixed control points `f_i` and the
generated paths are moving control points `m_i`. The inverse sampling map is
piecewise linear:

```text
x_source(x) = interp(x; f_i, m_i)
x_sample(x) = x + strength * clip(x_source(x) - x, ±d_x)
```

Top/bottom endpoint pairs provide an analogous piecewise vertical mapping. The
two mappings are sampled together with bilinear interpolation. Soft Voronoi-like
stack masks taper the operation at layer and body boundaries, avoiding seams.

The operation is intentionally conservative:

- default full displacement cap: `5 px`;
- candidate strengths: `0.12, 0.16, 0.20, 0.24`;
- therefore the accepted `0.16` candidate applies no more than about `0.8 px`;
- the central block is explicitly restored from the size-locked generation;
- every pixel outside the writable stack mask is explicitly restored;
- no raw/guide intensity writeback and no analytic layer redraw are allowed.

### 2.5 Candidate selection and rejection

Each strength is quantized through the actual 16-bit export path and evaluated.
A candidate is eligible only when:

- global SSIM to the accepted size-locked generation is at least the configured
  threshold (`0.965` by default);
- maximum absolute change outside the writable mask is no more than half a
  16-bit code value;
- median and P95 guide-to-result ridge offsets do not exceed baseline values;
- mid-frequency appearance correlation remains within a bounded tolerance.

Among eligible candidates, the score balances guide correlation, median ridge
offset, and P95 ridge offset. If all candidates fail, the program stops rather
than silently emitting an unconstrained result.

## 3. Geometry profile

`config/reference_geometry.json` defines all coordinate-dependent assumptions.
Coordinates use `[y0, y1, x0, x1]`; ranges use `[start, stop]`. By default the
coordinates scale independently in x/y from the `2200 × 1600` reference frame.
The horizontal and vertical displacement caps scale by the same axis-specific
factors.

Automatic scaling handles resolution changes, not arbitrary repositioning. For a
different scanner pose or object layout, verify ROIs using the audit overlay and
create a dedicated profile.

## 4. Reproducibility and auditability

Every run records:

- source and input hashes;
- input/output dimensions and size-lock operation;
- blind-guide origin;
- selected candidate and all guardrail metrics;
- whether the exact identity fallback was used because no nonzero warp improved
  alignment safely;
- ridge/endpoint/width summaries;
- final output hashes;
- explicit invariants and measurement warning.

The PNG is for viewing. The 16-bit TIFF is the preferred downstream artifact.
Quantitative claims should still be derived from the archived source/guide
constraint table and validated against calibrated data.

## 5. Known limits

- The external generator can hallucinate structures. The small warp improves
  coordinate agreement but cannot make missing or invented content truthful.
- Structure guidance is allowed to select an exact no-op when all nonzero
  candidates fail the alignment guardrails; this is a safety outcome, not a
  failed run.
- Default ROIs assume a two-stack/central-block layout.
- A precomputed guide must have exactly the native source dimensions; it is never
  resized because that would invalidate measurement coordinates.
- The bundled blind stage trains per image and is compute-intensive on CPU.
- “Blind2Sound-style” identifies the research design adapted here; it does not
  claim official authorship, identical weights, or universal state-of-the-art
  performance on every CT modality.
