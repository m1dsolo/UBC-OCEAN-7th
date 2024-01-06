# UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN) 7th place solution.

This code repository mainly includes data preprocessing and multiple instance learning(MIL) model training.

- For detailed competition information, please refer to the [competition link](https://www.kaggle.com/competitions/UBC-OCEAN).
- For detailed solution information, please refer to the [discussion link](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/465697).
- For detailed submission notebook, please refer to the [notebook link](https://www.kaggle.com/code/m1dsolo/ubc-ocean-7th-submission).

## Requirements

```txt
python>=3.10
```

## Install

```bash
git clone --recurse-submodules https://github.com/m1dsolo/UBC-OCEAN-7th.git
pip install UBC-OCEAN-7th/yangdl
```

## Pipeline

1. Tiling patches

    ```bash
    python png2patches.py
    ```

2. Download [ctranspath checkpoints](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view)
3. Extract features

    ```bash
    python gen_features.py -t ctrans
    python gen_features.py -t vits16
    ```

4. train MIL

    ```bash
    python train_dsmil.py -t ctrans
    python train_dsmil.py -t vits16
    python train_perceiver.py -t ctrans
    python train_perceiver.py -t vits16
    ```

model will save to `./res/*/ckpt`

### For more information please refer to:

1. [CTransPath, MIA2022](https://github.com/Xiyue-Wang/TransPath)
2. [LunitDINO, CVPR2023](https://github.com/lunit-io/benchmark-ssl-pathology)
3. [DSMIL, CVPR2021](https://github.com/binli123/dsmil-wsi)
4. [Perceiver, BMVA2023](https://github.com/cgtuebingen/DualQueryMIL)

