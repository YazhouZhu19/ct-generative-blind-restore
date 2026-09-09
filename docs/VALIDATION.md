# Validation

## Unit tests

```bash
python -m unittest discover -s tests -v
```

Tests cover source-size locking, geometry-profile scaling, protected-output
cleanup, blind-denoiser primitives, monotonic ridge matching, bounded coordinate
mapping, central/background locks, and continuous 2-D warp behavior.

## Docker smoke test

Prepare `input/source_16bit.tif`, `input/generative_candidate.png`, and
`input/measurement_guide_16bit.tif`, then run:

```bash
docker compose build enhance
docker compose run --rm enhance
```

Verify:

```bash
docker compose run --rm tests
```

The run is valid only when:

- `RUN_MANIFEST.json` reports `completed: true`;
- final dimensions equal the source dimensions;
- the structure stage reports zero change outside the writable mask;
- central-block maximum change is at most half a 16-bit code value;
- a nonzero candidate passed all guardrails, or the audited exact identity
  fallback was selected because no nonzero warp did;
- PNG and TIFF hashes are present.

## Golden-sample reproduction

For the accepted project sample, use the original raw CT, its saved blind guide,
and the accepted generated image immediately after `2200 × 1600` size correction.
The expected selector chooses strength `0.16`. Input and exact output hashes are
stored in `config/golden_sample_reference.json`; image data remains excluded from
Git. The accepted outputs are:

```text
PNG:  e58ab1d2b553b55604ccab151388d31365c36d23af5ef54dd9809507981dee2d
TIFF: 952e852a50bab113fbe7bc405fce9e13c33d2e9a30695c27f7aa17b406096f5e
```

The reference result previously measured approximately:

- median/P95/max applied x displacement: `0.80 / 0.80 / 0.80 px`;
- median/P95/max applied y displacement: `0.49 / 0.80 / 0.80 px`;
- central-block maximum absolute change: `0`;
- outside-mask maximum absolute change: `0`;
- median ridge offset: `2.6267 → 2.6137 px`;
- median width relative error: `32.35% → 25.29%`.

These are sample-specific regression values, not general performance guarantees.
