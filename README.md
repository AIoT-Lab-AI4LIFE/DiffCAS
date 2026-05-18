# DiffCAS: Inference-time CT-free Diffusion Model for Physics-aware Multi-slice Attenuation Correction in Cardiac SPECT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DiffCAS is an inference-time CT-free diffusion model for physics-aware multi-slice 
attenuation correction in cardiac SPECT. The method generates attenuation-corrected 
(AC) images directly from non-attenuation-corrected (NAC) inputs without requiring 
CT input at inference time.

## Key Features

- **Physics-aware reconstruction**: Integrates a Brownian Bridge diffusion process 
  with physics-guided supervision via a learnable Attenuation Correction Factor (ACF).
- **Multi-slice contextual learning**: Captures cross-slice anatomical dependencies 
  to improve spatial coherence in reconstructed volumes.
- **3D CT-ViT backbone**: Leverages a Transformer-based volumetric feature extractor 
  initialized from a medical foundation model for long-range attenuation prior modeling.
- **Teacher-student distillation**: Transfers physics-informed knowledge from 
  CT-conditioned training to a CT-free student network for inference-time deployment.

## Citation

If you find this work useful, please cite:

```bibtex
@article{vu2025diffcas,
  title={DiffCAS: Inference-time CT-free Diffusion Model for Physics-aware 
         Multi-slice Attenuation Correction in Cardiac SPECT},
  author={Vu, Hoang Minh and Pham, Trung Kien and Nguyen, Thi Ha Chi and 
          Nguyen, Hai Dang and Nguyen, Dac Thai and Son, Mai Hong and 
          Nguyen, Thanh Trung and Nguyen, Trung Thanh and Nguyen, Phi Le},
  year={2026}
}
```

## Code Release

The source code will be released upon acceptance of the paper.  
Stay tuned by starring or watching this repository.

## Contact

For questions or collaborations, please open an issue or contact the corresponding 
authors.
