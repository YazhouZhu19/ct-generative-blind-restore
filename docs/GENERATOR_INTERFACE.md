# External generator interface

The repository starts from an already generated enhancement candidate. Keeping
this boundary explicit makes the code reproducible without committing API keys,
service-specific calls, licenses, or proprietary model weights.

## Input contract

The candidate may be PNG, JPEG, or integer TIFF and may have any dimensions.
It should:

- depict the same complete CT field of view as the source;
- retain both lamella stacks and the complete central bright block;
- preserve plate count and ordering as closely as possible;
- keep plates as measurable strips, not sharpen them into needle-like lines;
- remove haze and stochastic noise without adding labels, borders, or crops;
- avoid perspective changes, rotations, object replacement, and background edits.

Recommended prompt template:

```text
Enhance this industrial CT image as a conservative restoration. Remove grain,
haze, and low-frequency fog while preserving the complete field of view, central
bright block, left/right lamella count, plate-like widths, endpoints, spacing,
and order. Do not crop, pad, rotate, redraw layers as needles, invent structures,
or add annotations. Return a grayscale image with the same composition.
```

Do not rely on the prompt alone for geometry. The pipeline immediately locks the
candidate to the source canvas, measures a blind-denoised native guide, and only
permits a bounded coordinate correction.

## Adding a local generator

Generate the candidate before invoking `app/run_pipeline.py`, then pass its path
with `--generated`. A future model adapter should remain a separate executable
and must not bypass `app/size_lock.py`. Its output must always enter the same
audited pipeline boundary.
