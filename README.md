# Graph Learning-Convolutional Networks

This is a TensorFlow implementation of Graph Learning-Convolutional Networks for the task of (semi-supervised) classification of nodes in a graph, as described in our paper:
 
Bo Jiang, Ziyan Zhang, [Semi-supervised Learning with Graph Learning-Convolutional Networks](http://http://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) (CVPR 2019)


## Introduction

In this repo, we provide GLCN's code with the Cora and Citeseer datasets as example. The graph convolution method used in this code is provided by Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017).

## Requirements
The codebase is implemented in Python 3.6.8. package versions used for development are just below
* tensorflow-gpu (1.13.1)
* scipy (1.4.1)
* numpy (1.16.2)

## Run the demo

```bash
cd glcn
python run_cora.py
```

## Data

There are three entries for the code.
* Feature matrix (feature.mat): An n * p sparse matrix, where n represents the number of nodes, and p represents the feature dimension of each node.
* Adjacency matrix (adj.mat): An n * n sparse matrix, where n represents the number of nodes.
* Label matrix (label.mat): An n * c matrix, where n represents the number of nodes, c represents the number of classes, and the label of the node is represented by onehot.

We provide the Cora and Citeseer datasets as example. The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/. In our version (see `data` folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016). 

If you want to use your own dataset, please process the data into the above state, and look at the `load_data()` function in `utils.py` for an example.


## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{jiang2019semi,
  title={Semi-supervised learning with graph learning-convolutional networks},
  author={Jiang, Bo and Zhang, Ziyan and Lin, Doudou and Tang, Jin and Luo, Bin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11313--11320},
  year={2019}
}
```
