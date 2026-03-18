# Descriptor Design

## Purpose

Infrared and point-cloud channels are not treated as side inputs in their raw form inside the first detector. Instead, they are transformed into structured descriptors that can condition diffusion, support localisation, and provide explanation evidence.

The design has two descriptor families:

- spatial descriptor maps
- global descriptor vectors

## Infrared Descriptor Candidates

Planned infrared-derived spatial features:

- normalised thermal intensity
- thermal gradient magnitude
- hotspot response map
- local thermal contrast
- thermal edge density

Planned infrared-derived global features:

- peak temperature proxy
- average thermal deviation
- hotspot area ratio
- thermal entropy or spread
- category-normalised thermal severity

## Point-Cloud Descriptor Candidates

Planned point-cloud spatial features after projection or rasterisation:

- depth or height map
- surface normal deviation
- local curvature
- roughness estimate
- geometric discontinuity strength

Planned point-cloud global features:

- global roughness score
- curvature dispersion
- estimated deformation severity
- symmetry or shape deviation proxy
- geometric irregularity score

## Cross-Modal Descriptor Candidates

Descriptors that combine infrared and geometry:

- thermal-structural agreement map
- hotspot-to-geometry overlap score
- descriptor consistency index
- anomaly support agreement between thermal and geometric cues

These are especially useful for explanation because they show whether multiple sensor channels support the same anomaly hypothesis.

## Output Shape Strategy

For implementation simplicity:

- spatial descriptors should be resampled into a common 2D grid aligned with RGB space
- global descriptors should be stored as a fixed-length vector per sample

That gives a sample form like:

- `x_rgb`: RGB image
- `M`: stacked spatial descriptor maps
- `g`: compact global descriptor vector

## Engineering Constraints

Descriptor extraction should be:

- deterministic
- cached under `data/processed/`
- independent of test labels
- robust to missing GT assets

The first version should prioritise interpretable descriptors over exotic learned descriptor generators.

## Recommended First Descriptor Set

For a stable initial implementation, start with:

- infrared intensity
- infrared gradient magnitude
- projected point-cloud depth
- projected point-cloud curvature or roughness
- one cross-modal agreement map
- a compact global vector with thermal severity, geometric irregularity, and cross-modal consistency

This is enough to test the descriptor-conditioned hypothesis without overcomplicating preprocessing.
