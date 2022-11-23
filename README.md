# HARDVS
A large-scale benchmark dataset for Human Activity Recognition with Dynamic Vision Sensors

[Revisiting Color-Event based Tracking: A Unified Network, Dataset, and Metric](https://arxiv.org/abs/2211.11010), Chuanming Tang, Xiao Wang, Ju Huang, Bo Jiang, Lin Zhu, Jianlin Zhang, Yaowei Wang, Yonghong Tian 
[[Paper](https://arxiv.org/abs/2211.11010)] 
[[DemoVideo](https://youtu.be/_ROv09rvi2k)] 
[[Project](https://sites.google.com/view/coesot/)]



## TODO List 
- [x] Paper (arXiv) release (2022-12-01)
- [x] COESOT dataset release (2022-12-01)
- [x] Evaluation Toolkit release (2022-12-01)
- [x] Source Code release (2022-12-01)
- [x] Tracking Models release (2022-12-01)


### Demo Video: 
* [[YouTube](https://youtu.be/_ROv09rvi2k)]


### Dataset Download: 
* [[Baidu](链接：链接：https://pan.baidu.com/s/1OhlhOBHY91W2SwE6oWjDwA?pwd=1234 
提取码：1234 ] 

Install env
```
conda create -n event python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate event
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .

```

## Train & Test & Evaluation
```
    # train
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/recognition/hardvs_ESTF/hardvs_ESTF.py --work-dir path_to_checkpoint --validate --seed 0 --deterministic --gpu-ids=0

    # test
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/recognition/puke_SNNCNN/puke_SNNCNN.py  path_to_checkpoint --eval top_k_accuracy

```



### Citation: 
```bibtex
@misc{https://doi.org/10.48550/arxiv.2211.09648,
  title = {HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors},
  url = {https://arxiv.org/abs/2211.09648},
  author = {Wang, Xiao and Wu, Zongzhen and Jiang, Bo and Bao, Zhimin and Zhu, Lin and Li, Guoqi and Wang, Yaowei and Tian, Yonghong},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

