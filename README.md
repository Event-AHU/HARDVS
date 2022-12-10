<div align="center">

<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/HARDVS_logo.png" width="350px">
  
**HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors**

------

<p align="center">
  <a href="https://sites.google.com/view/hardvs/">Project</a> •
  <a href="https://arxiv.org/abs/2211.09648">Paper</a> • 
  <a href="https://youtu.be/AgYjh-pfUT0">Demo</a> •
</p>

</div>




### Demo Video: 
* [[YouTube](https://youtu.be/_ROv09rvi2k)]


<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/HARDVS_all_samples.jpg" width="700px" align="center">



### Dataset Download: 
* [Event Images] 链接：https://pan.baidu.com/s/1OhlhOBHY91W2SwE6oWjDwA?pwd=1234    提取码：1234  

* [Compact Event file] 百度网盘链接： 链接：https://pan.baidu.com/s/1iw214Aj5ugN-arhuxjmfOw?pwd=1234 提取码：1234  

* [Raw Event file] To be updated 



### Environment 
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
  CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/recognition/hardvs_ESTF/hardvs_ESTF.py  path_to_checkpoint --eval top_k_accuracy
```



### Citation: 
If you find this work useful for your research, please cite the following paper and give us a :star2:.  
```bibtex
@article{wang2022hardvs,
  title={HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors},
  author={Wang, Xiao and Wu, Zongzhen and Jiang, Bo and Bao, Zhimin and Zhu, Lin and Li, Guoqi and Wang, Yaowei and Tian, Yonghong},
  journal={arXiv preprint arXiv:2211.09648},
  url={https://arxiv.org/abs/2211.09648}, 
  year={2022}
}
```

