# Eval_MCM-LDM
Evaluation Code for MCM-LDM

## Installation
Download the required files in [Google Drive](https://drive.google.com/drive/folders/13d2wWLlJ8MuJCV6bmlenKcTEpmuo3suj?usp=sharing), and place them in './'

## Quick Evaluation
Please use the same environment in [MCM-LDM](https://github.com/XingliangJin/MCM-LDM). 

Run the following commands:
```bash
python eval_FMD_CRA.py
python eval_SRA.py
```
The results should match those in the paper.

## Generate your own pkl files for Evaluation
You should run the following command in [MCM-LDM](https://github.com/XingliangJin/MCM-LDM) repository.
```bash
python demo_transfer_crafmd.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --style_motion_dir demo/content_test_feats --content_motion_dir demo/content_test_feats --scale 2.5
python demo_transfer_sra.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --style_motion_dir demo/style_test_feats --content_motion_dir demo/style_test_feats --scale 2.5
```
With these generated pkl files, after running the evaluation, you will get results similar to those in the papers. (The differences in evaluation results each time might be due to not fixing the seed during our testing.)

## Acknowledgements
This code is mostly taken from [ACTOR](https://github.com/Mathux/ACTOR)
