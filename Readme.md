# RoMER

**RoMER: Query-Driven Evidence Decoupling for Robust Multi-label Website Fingerprinting**

*Official implementation. The paper is currently under submission.*

[English](README.md) | [中文](README_zh.md)

RoMER is a website fingerprinting (WF) attack framework for multi-tab traffic. It supports both closed-world and open-world settings, as well as common traffic defenses (WTF-PAD, FRONT, and RegulaTor).

## Repository Structure

```
.
├── data_process/            # dataset split
│   └── dataset_split.py
├── defense_npz/             # traffic defenses (operate on npz directly)
│   ├── wtfpad/              #   WTF-PAD
│   ├── front/               #   FRONT
│   └── regulartor/          #   RegulaTor
├── wfa_main/
│   ├── Model_Dataset/       # RoMER model & dataset
│   │   ├── model_romer.py   #   RoMER model (EM1 / EM3 variants)
│   │   ├── model_base.py
│   │   └── dataset.py       #   TAM feature extraction
│   └── run/                 # train / test entry points
│       ├── main.py          #   training
│       ├── test.py          #   evaluation
│       ├── config/Romer.ini
│       └── ...
├── lxj_utils_sys/           # shared utilities
├── requirements.txt
└── README.md
```

## 1. Dependency Install

```shell
conda create -n romer python=3.10
conda activate romer
# PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# Other dependencies
pip install -r requirements.txt
```

After installation, add the repository root to `PYTHONPATH`:

```shell
cd <repo_root>
export PYTHONPATH=$PWD
```

All attack scripts in Section 4 are run from the `wfa_main/run` directory:

```shell
cd wfa_main/run
```

## 2. Dataset

### 2.1 Download the Dataset

The multi-tab traffic is collected based on the ARES multi-tab website fingerprinting dataset:

- **ARES Dataset**: [https://github.com/Xinhao-Deng/Multitab-WF-Datasets](https://github.com/Xinhao-Deng/Multitab-WF-Datasets)

Download the raw ARES data for each setting and place the `data.npz` file into `npz_dataset/{dataset}/data.npz`. The raw `data.npz` contains `direction`, `time`, and `label`. Convert it into the `(X, y)` format used by RoMER, where `X` are traffic traces with shape `(N, L, 2)` (`timestamp`, `packet length`) and `y` is the multi-label indicator matrix with shape `(N, C)`:

```shell
cd data_process
for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab Open_2tab Open_3tab Open_4tab Open_5tab
do
  python convert_multi_tab_npz.py --dataset ${dataset}
done
```

**All normalized datasets can be downloaded from [here](https://zenodo.org/uploads/21769044).**

### 2.2 Dataset Split

Split each dataset into `train.npz`, `valid.npz`, and `test.npz`:

```shell
cd data_process
for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab Open_2tab Open_3tab Open_4tab Open_5tab
do
  python dataset_split.py --dataset ${dataset} --use_stratify False
done
```

## 3. Defense Dataset

Generate defended datasets with WTF-PAD, FRONT, and RegulaTor. Each defense reads `npz_dataset/{dataset}/data.npz` and writes a defended `data.npz`.

### 3.1 Generate Defended Traces

```shell
# WTF-PAD
cd defense_npz/wtfpad
python main.py --traces_path ../../npz_dataset/Closed_2tab --output_path ../results/wtfpad_Closed_2tab
python main.py --traces_path ../../npz_dataset/Open_2tab   --output_path ../results/wtfpad_Open_2tab

# FRONT
cd ../front
python main.py --p ../../npz_dataset/Closed_2tab --output_path ../results/front_Closed_2tab
python main.py --p ../../npz_dataset/Open_2tab   --output_path ../results/front_Open_2tab

# RegulaTor
cd ../regulartor
python regulator_sim.py --source_path ../../npz_dataset/Closed_2tab --output_path ../results/regulator_Closed_2tab
python regulator_sim.py --source_path ../../npz_dataset/Open_2tab   --output_path ../results/regulator_Open_2tab
```

### 3.2 Copy and Split

Copy the defended `data.npz` files into `npz_dataset/` and split them:

```shell
cd ../..
for ds in wtfpad front regulator
do
  for scen in Closed_2tab Open_2tab
  do
    mkdir -p npz_dataset/${ds}_${scen}
    cp defense_npz/results/${ds}_${scen}/data.npz npz_dataset/${ds}_${scen}/data.npz
  done
done

cd data_process
for dataset in wtfpad_Closed_2tab wtfpad_Open_2tab front_Closed_2tab front_Open_2tab regulator_Closed_2tab regulator_Open_2tab
do
  python dataset_split.py --dataset ${dataset} --use_stratify False
done
```

## 4. Website Fingerprinting Attack Scripts

Train and evaluate RoMER on the multi-tab datasets. Each dataset is trained with `main.py` and evaluated with `test.py` (the two share the same `--note`, so the checkpoint is reused).

### 4.1 Undefended Traffic (Closed-world & Open-world)

```shell
cd wfa_main/run

# Closed-world
for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab
do
  python main.py --dataset ${dataset} --config config/Romer.ini \
      --checkpoint_path ../../checkpoints --file_base_dir ../../npz_dataset \
      --note romer --Model_name EM3 --TAM_type ED3 --test_flag False \
      --train_epochs 30 --overlap_ratio 0.1 --valid_name valid --remove_size False
  python test.py --dataset ${dataset} --config config/Romer.ini \
      --checkpoint_path ../../checkpoints --file_base_dir ../../npz_dataset \
      --note romer --Model_name EM3 --TAM_type ED3 --test_flag False \
      --load_ratio 100 --overlap_ratio 0.1 --is_pr_auc True --first_check_ratio True --remove_size False
done

# Open-world
for dataset in Open_2tab Open_3tab Open_4tab Open_5tab
do
  python main.py --dataset ${dataset} --config config/Romer.ini \
      --checkpoint_path ../../checkpoints --file_base_dir ../../npz_dataset \
      --note romer --Model_name EM3 --TAM_type ED3 --test_flag False \
      --train_epochs 30 --overlap_ratio 0.1 --valid_name valid --remove_size False
  python test.py --dataset ${dataset} --config config/Romer.ini \
      --checkpoint_path ../../checkpoints --file_base_dir ../../npz_dataset \
      --note romer --Model_name EM3 --TAM_type ED3 --test_flag False \
      --load_ratio 100 --overlap_ratio 0.1 --is_pr_auc True --first_check_ratio True --remove_size False
done
```

### 4.2 Defended Traffic

```shell
cd wfa_main/run

for dataset in wtfpad_Closed_2tab wtfpad_Open_2tab front_Closed_2tab front_Open_2tab regulator_Closed_2tab regulator_Open_2tab
do
  python main.py --dataset ${dataset} --config config/Romer.ini \
      --checkpoint_path ../../checkpoints --file_base_dir ../../npz_dataset \
      --note romer --Model_name EM3 --TAM_type ED3 --test_flag False \
      --train_epochs 30 --overlap_ratio 0.1 --valid_name valid --remove_size False
  python test.py --dataset ${dataset} --config config/Romer.ini \
      --checkpoint_path ../../checkpoints --file_base_dir ../../npz_dataset \
      --note romer --Model_name EM3 --TAM_type ED3 --test_flag False \
      --load_ratio 100 --overlap_ratio 0.1 --is_pr_auc True --first_check_ratio True --remove_size False
done
```

## Results

The figures below summarize the performance of RoMER on the multi-tab datasets, covering closed-world and open-world settings as well as defended traffic (WTF-PAD, FRONT, RegulaTor).

### Closed-world

![Closed-world results](ref/closed.png)

### Open-world

![Open-world results](ref/open.png)

## Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@misc{romer2026query,
  title = {RoMER: Query-Driven Evidence Decoupling for Robust Multi-label Website Fingerprinting},
  author = {TBD},
  year = {2026},
  note = {Under submission}
}
```

The paper is currently under submission; the BibTeX entry will be updated once it is accepted.
