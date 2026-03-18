# MulSenDiff-X: Descriptor-Conditioned Diffusion for Unsupervised Multi-Sensor Industrial Anomaly Detection and Evidence-Grounded Explanation

MulSenDiff-X is the implementation repo for an IEEE IES Generative AI Challenge project built on the MulSen-AD dataset. The project uses RGB as the learned appearance branch and turns infrared thermography plus 3D point clouds into descriptor-based conditioning signals for anomaly detection, localisation, and explanation.

## Project Direction

The core idea is:

- learn normal RGB appearance with a conditional diffusion model
- derive thermal and geometric evidence from aligned infrared and point-cloud data
- use those descriptors to condition detection and support localisation
- package the resulting evidence for a grounded explanation module

This keeps the detector fully unsupervised during training while still using all three aligned sensing modalities at inference and evaluation time.

## Dataset Status

The dataset is present locally at:

- `data/raw/MulSen_AD`

Verified local structure:

- 15 object categories
- 3 aligned modalities per category: `RGB`, `Infrared`, `Pointcloud`
- normal-only training splits under `train/`
- evaluation data under `test/`
- modality-specific ground truth under `GT/`

Observed aggregate counts from the extracted dataset:

- RGB: `1391` train files, `644` test files, `448` GT files
- Infrared: `1391` train files, `644` test files, `388` GT files
- Pointcloud: `1391` train files, `644` test files, `342` GT files
- extra metadata assets: `87` CSV files in RGB GT folders

## Current Repo Status

The repository now has the expected top-level structure for:

- dataset documentation
- architecture notes
- descriptor design
- explanation design
- experiment planning
- split management

Most Python modules are still scaffolding, so the next implementation focus should be the data manifest, loaders, preprocessing, and a first trainable baseline.

## Recommended Build Order

1. Build a dataset index that aligns RGB, infrared, point cloud, split, defect label, and GT paths.
2. Implement modality loaders and preprocessing.
3. Generate descriptor maps and global descriptor vectors from infrared and point clouds.
4. Train a per-category conditional RGB diffusion baseline.
5. Add anomaly scoring and localisation.
6. Add evidence packaging and explanation generation.

## Hardware Note

CUDA is not currently available on this machine because Secure Boot key enrollment was not completed after driver installation. The Ubuntu driver stack is now installed cleanly with `nvidia-driver-535`, but the MOK approval step still needs to be accepted during reboot before NVIDIA modules can load.

Until that is completed, assume:

- CPU-first development
- small-scale preprocessing and indexing locally
- conservative training settings
- optional later migration to a stronger GPU machine for full experiments

## Documentation Map

- `docs/01_dataset.md`: dataset structure, counts, and usage rules
- `docs/02_architecture.md`: end-to-end model design
- `docs/03_descriptors.md`: infrared and point-cloud descriptor plan
- `docs/04_explainer.md`: evidence-grounded explanation pipeline
- `docs/literature_review.md`: implementation-oriented review framing
- `data/splits/README.md`: split and manifest conventions
- `src/experiments/README.md`: experiment roadmap
