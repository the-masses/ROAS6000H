# Humanoid Motion Retargeting from Video to Unitree H1 (GVHMR + ProtoMotions)

> **TL;DR**: This repo takes monocular human videos → reconstructs SMPL / SMPL-X motion with **GVHMR** → **retargets** them to Unitree H1 → trains a **full-body motion tracker**  → trains **MaskedMimic** in IsaacLab → runs a controllable H1 policy that imitates your own walking.

https://github.com/the-masses/ROAS6000H

---

## 🎬 Final Results

<p align="center">
  <!-- Replace with your teaser video / gif -->
  <img src="media/hmr.gif" width="70%">
</p>

<p align="center">
  <!-- Replace with your teaser video / gif -->
  <img src="media/training.gif" width="70%">
</p>

---

## Environment Setup

We use **two conda environments**:

- `gvhmr`: GVHMR inference + Genesis full-body tracker training  
- `env_isaaclab`: IsaacLab-based MaskedMimic training

Both are exported in `envs/`.

---

### Clone Repo & Initialize Submodules

```bash
git clone https://github.com/the-masses/ROAS6000H.git
cd ROAS6000H
git submodule update --init --recursive
```

To refresh submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### Install the Two Conda Environments

GVHMR + Genesis environment:

```bash
conda env create -f envs/gvhmr_env.yaml
conda activate gvhmr
```

IsaacLab environment:

```bash
conda env create -f envs/env_isaaclab.yaml
conda activate env_isaaclab
```

### Additional Install Steps

Depending on your local setup, you may need to:

* Download SMPL-X / SMPL / H1 model files and place them in the expected folders (TODO: describe paths).
* Configure IsaacLab / Genesis assets (e.g., set `ISAAC_NUCLEUS_DIR`, etc.).
You can follow ProtoMotions’ and GVHMR’s official READMEs inside:
```bash
cd GVHMR
# (Follow GVHMR README)
cd ../ProtoMotions
# (Follow ProtoMotions README for assets / models)
```

## Full Pipeline: From Video to Trained H1 Policy

This section gives a **script-level** overview of the entire pipeline. All paths / config names are examples; adjust them to your setup.

### Step 0 – Record Video

Record a monocular RGB video of human motion (e.g., walking in front view).
* Resolution: e.g., 720p / 1080p
* Frame rate: e.g., 30 FPS
* Save as: `data/videos/mywalk.mp4`, etc.
We assume multiple videos like:
```text
data/videos/
├── jz_walk1.mp4
├── jz_walk2.mp4
└── ...
```
---

### Step 1 – Video to SMPL-X

**Environment:** `gvhmr`

```bash
cd GVHMR \
python tools/demo/demo.py --video={YOUR_VIDEO_PATH} -s
```

Inside `tools/demo/demo.py`, you typically get **Output**:
* Save human motion capture data in the form of video recordings (`.mp4`) and reconstructed 3D human motion (`.pt`).
* Save per-sequence SMPL-X parameters (`.npz`) in `../ProtoMotions/data/amassx`.


### Step 2 – Retarget SMPL-X to H1

**Environment:** `gvhmr`

```bash
cd ../ProtoMotions \
python data/scripts/convert_amass_to_isaac.py     data/amassx     --robot-type=h1     --humanoid-type=smplx     --force-retarget
```

The above script can convert all `.npz` files in `data/amassx` into retargeted `.npy` files. Each file will generate its own folder with the suffix `-h1_retargeted`.

Next, create a file named `mywalk_train_h1.yaml` in the `data/yaml_files` folder to package all the retargeted dataset sequences.

Here is the simplest example of a single action `.yaml` file:
```bash
motions:
  - id: 0
    name: "jz_walk1"
    file: "1-h1_retargeted/1.npy"
    fps: 30.0
    weight: 1.0
    subject: "jz"
    gender: "male"
```

After creating the packaged file for your custom action library `mywalk_train_h1.yaml`, run the script:
```bash
python data/scripts/package_motion_lib.py     data/yaml_files/mywalk_train_h1.yaml   data/amassx     data/motions/h1_mywalk.npy    --humanoid-type=h1
```
All `.npy` files with the suffix `-h1_retargeted` can be packaged into the `./data/motions/` folder to generate the `h1_mywalk.npy` file.

### Step 3 – Full Body Tracker Training

**Environment:** `env_isaaclab`

This step is used to train the prior model for maskedmimic. Since we only need to achieve retargeting on a single action, using an MLP structure as the prior network makes it easier to train successfully.

Training script:
```bash
PYTHONPATH=. python protomotions/train_agent.py \
+exp=full_body_tracker/mlp_single_motion_flat_terrain.yaml \
+robot=h1 \
+simulator=isaaclab \
motion_file=data/motions/h1_mywalk.pt \
+terrain=flat \
+experiment_name=h1
```

To view the latest saved training results in real time, run the validation visualization script:
```bash
PYTHONPATH=. python protomotions/eval_agent.py \
+robot=h1 \
+simulator=isaaclab \
+checkpoint=results/h1/last.ckpt
```

The parameters provided by the official source may have been adjusted on IsaacLab; you'll need to retune them yourself on Genesis.

### Step 4 – MaskedMimic Training

**Environment:** `env_isaaclab`

Using the results from step three as the network prior for this step enables rapid completion of the final training step.
Training script:
```bash
PYTHONPATH=. python protomotions/train_agent.py \
+exp=masked_mimic/flat_terrain \
+robot=h1 \
+simulator=isaaclab \
motion_file=./data/motions/h1_mywalk.pt \
agent.config.expert_model_path=results/h1 \
+experiment_name=h1_maskedmimic
```

Evaluation script:
```bash
PYTHONPATH=. python protomotions/eval_agent.py \
+robot=h1 \
+simulator=isaaclab \
+checkpoint=results/h1_maskedmimic/last.ckpt \
+headless=False
```

# Acknowledgments

This project repository are combined with [GVHMR](https://github.com/zju3dv/GVHMR) and [ProtoMotions](https://github.com/NVlabs/ProtoMotions.git).

## Citation

This codebase builds upon prior work from NVIDIA and external collaborators. Please adhere to the relevant licensing in the respective repositories.
If you use this code in your work, please consider citing our works:
```bibtex
@misc{ProtoMotions,
  title = {ProtoMotions: Physics-based Character Animation},
  author = {Tessler, Chen and Juravsky, Jordan and Guo, Yunrong and Jiang, Yifeng and Coumans, Erwin and Peng, Xue Bin},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVLabs/ProtoMotions/}},
}

@inproceedings{tessler2024masked,
  title={MaskedMimic: Unified Physics-Based Character Control Through Masked Motion},
  author={Tessler, Chen and Guo, Yunrong and Nabati, Ofir and Chechik, Gal and Peng, Xue Bin},
  booktitle={ACM Transactions On Graphics (TOG)},
  year={2024},
  publisher={ACM New York, NY, USA}
}

@inproceedings{tessler2023calm,
  title={CALM: Conditional adversarial latent models for directable virtual characters},
  author={Tessler, Chen and Kasten, Yoni and Guo, Yunrong and Mannor, Shie and Chechik, Gal and Peng, Xue Bin},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={1--9},
  year={2023},
}
```

Also consider citing these prior works that helped contribute to this project:
```bibtex
@inproceedings{juravsky2024superpadl,
  title={SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation},
  author={Juravsky, Jordan and Guo, Yunrong and Fidler, Sanja and Peng, Xue Bin},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
}

@inproceedings{luo2024universal,
    title={Universal Humanoid Motion Representations for Physics-Based Control},
    author={Zhengyi Luo and Jinkun Cao and Josh Merel and Alexander Winkler and Jing Huang and Kris M. Kitani and Weipeng Xu},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=OrOd8PxOO2}
}

@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}            

@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
} 

@inproceedings{hassan2023synthesizing,
  title={Synthesizing physical character-scene interactions},
  author={Hassan, Mohamed and Guo, Yunrong and Wang, Tingwu and Black, Michael and Fidler, Sanja and Peng, Xue Bin},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={1--9},
  year={2023}
}
```

If you find GVHMR useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

## References and Thanks
This project repository builds upon the shoulders of giants. 
* [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) for reference IsaacGym code. For example, terrain generation code.
* [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs) for reference IsaacSim code.
* [DeepMimic](https://github.com/xbpeng/DeepMimic) our full body tracker (Mimic) can be seen as a direct extension of DeepMimic.
* [ASE/AMP](https://github.com/nv-tlabs/ASE) for adversarial motion generation reference code.
* [PACER](https://github.com/nv-tlabs/pacer) for path generator code.
* [PADL/SuperPADL](https://github.com/nv-tlabs/PADL2) and Jordan Juravsky for initial code structure with PyTorch lightning
* [PHC](https://github.com/ZhengyiLuo/PHC) for AMASS preprocessing and conversion to Isaac (PoseLib) and reference on working with SMPL robotic humanoid.
* [SMPLSim](https://github.com/ZhengyiLuo/SMPLSim) for SMPL and SMPL-X simulated humanoid.
* [OmniH2O](https://omni.human2humanoid.com/) and [PHC-H1](https://github.com/ZhengyiLuo/PHC/tree/h1_phc) for AMASS to Isaac H1 conversion script.
* [rl_games](https://github.com/Denys88/rl_games) for reference PPO code.
* [Mink](https://github.com/kevinzakka/mink/) and Kevin Zakka for help with the retargeting.

The following people have contributed to this project:
* Chen Tessler, Yifeng Jiang, Xue Bin Peng, Erwin Coumans, Kelly Guo, and Jordan Juravsky.

We thank the authors of
[WHAM](https://github.com/yohanshin/WHAM),
[4D-Humans](https://github.com/shubham-goel/4D-Humans),
and [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch) for their great works, without which our project/code would not be possible.

## Dependencies
This project uses the following packages:
* PyTorch, [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE)
* PyTorch Lightning, [LICENSE](https://github.com/Lightning-AI/pytorch-lightning/blob/master/LICENSE)
* IsaacGym, [LICENSE](https://developer.download.nvidia.com/isaac/NVIDIA_Isaac_Gym_Pre-Release_Evaluation_EULA_19Oct2020.pdf)
* IsaacSim, [LICENSE](https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html)
* IsaacLab, [LICENSE](https://isaac-sim.github.io/IsaacLab/main/source/refs/license.html)
* Genesis, [LICENSE](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/LICENSE)
* SMPLSim, [LICENSE](https://github.com/ZhengyiLuo/SMPLSim/blob/0ec11c8dd3115792b8cf0bfeaef64e8c81be592a/LICENSE)
* Mink, [LICENSE](https://github.com/kevinzakka/mink/blob/main/LICENSE)