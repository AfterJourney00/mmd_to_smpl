## MMD to SMPL

> **TL;DR**  
> This is an automated workflow for composing, rendering, and retargeting any humanoid MikuMikuDance (MMD) characters driven by any MMD animations in any blender scenes.

### ✨ Key Features

- Composing any combination of MMD characters in `.pmx` and animations in `.vmd` within any `.blend` scenes.
- Toon and manga manuscript style rendering with tracking camera.
- Automated marker annotation tool for any humanoid MMD character.
- Shape and pose retargeting from humanoid MMD character to SMPL(-H).

### 🚀 Getting Started

#### 1. Environment Setup

```bash
conda create python=3.10 --name mmd2smpl
conda activate mmd2smpl

pip install -r requirements.txt

cp -r mmd_tools <your_python_env>/site_packages/bpy/3.6/scripts/addons
```

#### 2. Run Workflow

```shell
#!/bin/bash

export TMPDIR=<your_tmp_directory_for_cycles>

VERSION="mmd_to_smpl"

CUDA_VISIBLE_DEVICES=0 python -m .workflow \
--random_seed 42 \
--bake \
--no_physics \
--render \
--engine "CYCLES" \
--res_x 1280 \
--res_y 720 \
--focal 50 \
--start_frame 150 \
--duration 10 \
--style None \
--samples 64 \
--use_gpu \
--retarget \
--sf_standard_height 1.7 \
--sf_learning_rate 0.1 \
--sf_max_iter 300 \
--sf_w_shape_reg 0.0005 \
--sf_w_pose_reg 0.001 \
--pf_batch_size 1 \
--pf_num_iters 100 \
--version ${VERSION}
```

Explanation of key arguments:

- `--bake`: remove it to disable baking
- `--no_physics`: remove it to enable physical simulation
- `--render`: remove it to disable rendering
- `--style`: `"None"` for toon style, `"sketch"` for manga manuscript style
- `--retarget`: remove it to disable retargeting from MMD to SMPL(-H)

> [!NOTE]
> If you encounter the following error, just ignore it.
> ```bash
> Exception in module unregister(): '_ops.py'
> Traceback (most recent call last):
>   File ...
>     mod.unregister()
> TypeError: '_OpNamespace' object is not callable
> Exception in module unregister(): '_classes.py'
> Traceback (most recent call last):
>   File ...
>     mod.unregister()
> TypeError: '_ClassNamespace' object is not callable
> ```

### 🏄‍♂️ Contributors

- Chengfeng Zhao - [AfterJourney00](https://github.com/AfterJourney00)
- Junbo Qi - [jumbo-q](https://github.com/jumbo-q)
- Wangpok Tse - [JerryTseee](https://github.com/JerryTseee)
- Zekai Gu - [skygoo2000](https://github.com/skygoo2000)

### Acknowledgments

Thanks to the following work that we refer to and benefit from:
- [mmd_tools](https://github.com/sugiany/blender_mmd_tools): blender api for manipulating MMD contents;
- [joints2smpl](https://github.com/wangsen1312/joints2smpl): SMPL parameter solver according to 3D joints;
- [MMDMC](https://github.com/rongakowang/MMDMC): code reference