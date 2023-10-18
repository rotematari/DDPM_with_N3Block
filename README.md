
# DDPM_with_N3Block: Integrating N3 Blocks with DhariwalUNet for Enhanced Performance

## Overview
This repository contains the implementation of two distinct neural network architectures aimed at improving existing models:
1. A modified version of the baseline model from the [Exemplar Diffusion Machine (EDM)](https://github.com/NVlabs/edm), specifically the DhariwalUNet. Modifications were made to accommodate hardware constraints.
2. An advanced version of the DhariwalUNet that incorporates N3 Blocks from [N3Net](https://github.com/visinf/n3net), replacing the attention mechanisms at the initial two layers.

## Directory Structure
- `src_denoising/`: Contains the codebase from the N3Net repository.
- Remaining Directories: Comprise the transformed EDM code, adapted by our team to integrate the N3 Block into the DhariwalUNet architecture.

## Modifications
### Baseline Model
- Origin: Adapted from [EDM's DhariwalUNet](https://github.com/NVlabs/edm).
- Changes: Network dimensions were altered to fit the available hardware resources.

### Enhanced Model
- Origin: DhariwalUNet from EDM.
- Enhancement: Replaced attention mechanisms at the first two layers with N3 Blocks.
- Reference: [N3Net GitHub Repository](https://github.com/visinf/n3net).

## Contributors
- Modified EDM Code by Shadi.
