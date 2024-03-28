# LSTTN
==========================================
A PyTorch implementation of LSTTN
### Abstract
<p align="justify">
Accurate traffic forecasting is a fundamental problem in intelligent transportation systems and learning long- range traffic representations with key information through spatiotemporal graph neural networks (STGNNs) is a basic assumption of current traffic flow prediction models. However, due to structural limitations, existing STGNNs can only utilize short-range traffic flow data; therefore, the models cannot adequately learn the complex trends and periodic features in traffic flow. Besides, it is challenging to extract the key temporal information from the long historical traffic series and obtain a compact representation. To solve the above problems, we propose a novel LSTTN (Long-Short Term Transformer-based Network) framework comprehensively considering the long- and short-term features in historical traffic flow. First, we employ a masked subseries Transformer to infer the content of masked subseries from a small portion of unmasked subseries and their temporal context in a pretraining manner, forcing the model to efficiently learn compressed and contextual subseries temporal representations from long historical series. Then, based on the learned representations, long-term trend is extracted by using stacked 1D dilated convolution layers, and periodic features are extracted by dynamic graph convolution layers. For the difficulties in making time-step level prediction, LSTTN adopts a short-term trend extractor to learn fine-grained short-term temporal features. Finally, LSTTN fuses the long-term trend, periodic features and short-term features to obtain the prediction results. Experiments on four real-world datasets show that in 60-minute-ahead long-term forecasting, the LSTTN model achieves a minimum improvement of 5.63% and a maximum improvement of 16.78% over baseline models. The source code is availble at https://github.com/GeoX-Lab/LSTTN.

### Datasets
All of datasets is loaded and processed by [Pytorch-Geometric](https://github.com/pyg-team/pytorch_geometric). Note that the version of Pytorch-Geometric is `1.5.0`, which has a slight difference with the latest version on loading these dataset.  
The Ricci Curvature of these datasets is saved on `data/Ricci`. To compute curvature, please refer to the Python library [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature). 

### Options

Training the model is handled by the `main.py` script which provides the following command line arguments.  

```
  --data_path        STRING    Path of saved processed data files.                  Required is False    Default is ./data.
  --dataset          STRING    Name of the datasets.                                Required is True.
  --NCTM             STRING    Type of Negative Curvature Transformation Module.    Required is True     Choices are ['linear', 'exp'].
  --CNM              STRING    Type of Curvature Normalization Module.              Required is True     Choices are ['symmetry-norm', '1-hop-norm', '2-hop-norm'].
  --d_hidden         INT       Dimension of the hidden node features.               Required is False    Default is 64.
  --epochs           INT       The maximum iterations of training.                  Required is False    Default is 200.
  --num_expriment    INT       The number of the repeating expriments.              Required is False    Default is 50.
  --early_stop       INT       Early stop.                                          Required is False    Default is 20.
  --dropout          FLOAT     Dropout.                                             Required is False    Default is 0.5.
  --lr               FLOAT     Learning rate.                                       Required is False    Default is 0.005.
  --weight_decay     FLOAT     Weight decay.                                        Required is False    Default is 0.0005.
```

### Examples
The following commands learn the weights of a curvature graph neural network.
```commandline
python main.py --dataset Cora --NCTM linear --CNM symmetry-norm
```
Another examples is that the following commands learn the weights of the curvature graph neural network with 2-hop normalization on Citeseer.
```commandline
python main.py --dataset Citeseer --NCTM linear --CNM 2-hop-norm
```
  
### Citation information
If our repo is useful to you, please cite our published paper as follow:
```
Bibtex
@article{luo2024lsttn,
  title={LSTTN: A Long-Short Term Transformer-based spatiotemporal neural network for traffic flow forecasting},
  author={Luo, Qinyao and He, Silu and Han, Xing and Wang, Yuhan and Li, Haifeng},
  journal={Knowledge-Based Systems},
  pages={111637},
  year={2024},
  publisher={Elsevier}
}
```
