# LSTTN
------  

A PyTorch implementation of LSTTN
### Abstract
<p align="justify">
Accurate traffic forecasting is a fundamental problem in intelligent transportation systems and learning long-range traffic representations with key information through spatiotemporal graph neural networks (STGNNs) is a basic assumption of current traffic flow prediction models. However, due to structural limitations, existing STGNNs can only utilize short-range traffic flow data; therefore, the models cannot adequately learn the complex trends and periodic features in traffic flow. Besides, it is challenging to extract the key temporal information from the long historical traffic series and obtain a compact representation. To solve the above problems, we propose a novel LSTTN (Long-Short Term Transformer-based Network) framework comprehensively considering the long- and short-term features in historical traffic flow. First, we employ a masked subseries Transformer to infer the content of masked subseries from a small portion of unmasked subseries and their temporal context in a pretraining manner, forcing the model to efficiently learn compressed and contextual subseries temporal representations from long historical series. Then, based on the learned representations, long-term trend is extracted by using stacked 1D dilated convolution layers, and periodic features are extracted by dynamic graph convolution layers. For the difficulties in making time-step level predictions, LSTTN adopts a short-term trend extractor to learn fine-grained short-term temporal features. Finally, LSTTN fuses the long-term trend, periodic features, and short-term features to obtain the prediction results. Experiments on four real-world datasets show that in 60-minute-ahead long-term forecasting, the LSTTN model achieves a minimum improvement of 5.63% and a maximum improvement of 16.78% over baseline models. The source code is available at https://github.com/GeoX-Lab/LSTTN.

You can visit the document from [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0950705124002727) or [arXiv](https://arxiv.org/abs/2403.16495)

### Datasets
Download the datasets from Google Drive URL: https://drive.google.com/file/d/1GHQ071AICZW6rSsXBsjQGaUNPh047xqN/view?usp=drive_link.

Detailed statistics of datasets:
|  Dataset | Nodes    | Edges  | Time interval | Samples        |
| ---- | ------ | ------ | ----------- | ---------- |
| METR-LA  | 207 | 1722 | 5 mins      | 34272      |
| PEMS-BAY  | 325 | 2694 | 5 mins      | 52116      |
| PEMS-04  | 307 | 987 | 5 mins      | 16992      |
| PEMS-08  | 170 | 718 | 5 mins      | 17856      |

## Install

All experiments were implemented on 1 piece of A100 with 40G RAM. The `requirements.txt` file contains all the project's dependent installation packages, which can be quickly installed using the following command.

```
$ pip install -r requirements.txt
```

### Pretraining and forecasting

Training the model is handled by the `main.py` script. All the configurations related to pretraining and forecasting is provided in `/configs`. 

You can simply import CFG from corresponding config files in `\main.py` and run it for training.

```
from configs.PEMS04 import CFG

launch_training(CFG)
```
  
### Citation information
If our repo is useful to you, please cite our published paper as follow:
```
Bibtex
@article{luo2024lsttn,
  title={LSTTN: A Long-Short Term Transformer-based spatiotemporal neural network for traffic flow forecasting},
  author={Luo, Qinyao and He, Silu and Han, Xing and Wang, Yuhan and Li, Haifeng},
  journal={Knowledge-Based Systems},
  year={2024},
  volume={293},
  pages={111637},
  DOI = {10.1016/j.knosys.2024.111637},
  publisher={Elsevier}
}
```
