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

### Paper
Wang, Xiao and Wu, Zongzhen and Jiang, Bo and Bao, Zhimin and Zhu, Lin and Li, Guoqi and Wang, Yaowei and Tian, Yonghong. "**HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors.**" arXiv preprint arXiv:2211.09648 (2022). 

### Abstract
The main streams of human activity recognition (HAR) algorithms are developed based on RGB cameras which are suffered from illumination, fast motion, privacy-preserving, and large energy consumption. Meanwhile, the biologically inspired event cameras attracted great interest due to their unique features, such as high dynamic range, dense temporal but sparse spatial resolution, low latency, low power, etc. As it is a newly arising sensor, even there is no realistic large-scale dataset for HAR. Considering its great practical value, in this paper, we propose a large-scale benchmark dataset to bridge this gap, termed HARDVS, which contains 300 categories and more than 100K event sequences. We evaluate and report the performance of multiple popular HAR algorithms, which provide extensive baselines for future works to compare. More importantly, we propose a novel spatial-temporal feature learning and fusion framework, termed ESTF, for event stream based human activity recognition. It first projects the event streams into spatial and temporal embeddings using StemNet, then, encodes and fuses the dual-view representations using Transformer networks. Finally, the dual features are concatenated and fed into a classification head for activity prediction. Extensive experiments on multiple datasets fully validated the effectiveness of our model. 

<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/eventcompare.jpg" width="700px" align="center">


### Demo Video
* [[YouTube](https://youtu.be/_ROv09rvi2k)]


<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/HARDVS_all_samples.jpg" width="700px" align="center">






### Dataset Download

<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/compareEventdatasets.png" width="700px" align="center">


* [Event Images] 链接：https://pan.baidu.com/s/1OhlhOBHY91W2SwE6oWjDwA?pwd=1234    提取码：1234  

* [Compact Event file] 链接：https://pan.baidu.com/s/1iw214Aj5ugN-arhuxjmfOw?pwd=1234 提取码：1234  

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

### Our Proposed Approach 
<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/spatialtempHAR.jpg" width="700px" align="center"> 

**An overview of our proposed ESTF framework for event-based human action recognition.** It transforms the event streams into spatial and temporal tokens and learns the dual features using multi-head self-attention layers. Further, a FusionFormer is proposed to realize message passing between the spatial and temporal features. The aggregated features are added with dual features as the input for subsequent TF and SF blocks, respectively. The outputs will be concatenated and fed into MLP layers for action prediction.


### Train & Test & Evaluation
```
# train
  CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/recognition/hardvs_ESTF/hardvs_ESTF.py --work-dir path_to_checkpoint --validate --seed 0 --deterministic --gpu-ids=0

# test
  CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/recognition/hardvs_ESTF/hardvs_ESTF.py  path_to_checkpoint --eval top_k_accuracy
```





### Citation
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


### Acknowledgement and Other Useful Materials 
* **MMAction2**: [https://github.com/open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) 

* **SpikingJelly**: [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)








