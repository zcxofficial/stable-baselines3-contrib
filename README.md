# SB3 for Alphagen
This repository provides contributions to the Stable Baselines3 (SB3) library. We provide multiple models which can applied in [AlphaGen](https://github.com/RL-MLDM/alphagen/) environment for formulatic quant factor generation.

To fully replicate the project, plkease follow installation instructions in [AlphaGen](https://github.com/RL-MLDM/alphagen/), and import models from [sb3_contrib](./sb3_contrib). Relevant paper is available through [Arxiv](https://arxiv.org/abs/2409.05144). 

## Models
- [**QFR**](./sb3_contrib/qfr) : A maskable REINFORCE-based model which we mainly introduce in the paper.

- [**PPOFD**](./sb3_contrib/ppo_reawrd_shaping) : A maskable PPO model with reward shaping.

- [**A3C**](./sb3_contrib/A3C_mask), [**TRPO**](./sb3_contrib/TRPO_mask),[**REINFORCE**](./sb3_contrib/A3C_mask) : Models used for baselines.

## Performances
