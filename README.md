[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.05830)

<div align="center">

<img src="https://github.com/Event-AHU/HARDVS/blob/HARDVSv2/pictures/hardvsv2_log.png" width="350px">
  
**Human Activity Recognition using RGB-Event based Sensors: A Multi-modal Heat Conduction Model and A Benchmark Dataset**

------

<p align="center">
</p>

</div>

### Paper
Shiao Wang, Xiao Wang, Bo Jiang, Lin Zhu, Guoqi Li, Yaowei Wang, Yonghong Tian, Jin Tang. "**Human Activity Recognition using RGB-Event based Sensors: A Multi-modal Heat Conduction Model and A Benchmark Dataset.**" arXiv preprint 	arXiv:2504.05830 (2025). 
  [[arXiv](https://arxiv.org/abs/2504.05830)] 



### Abstract
The main streams of human activity recognition (HAR) algorithms are developed based on RGB cameras which are suffered from illumination, fast motion, privacy-preserving, and large energy consumption. Meanwhile, the biologically inspired event cameras attracted great interest due to their unique features, such as high dynamic range, dense temporal but sparse spatial resolution, low latency, low power, etc. As it is a newly arising sensor, even there is no realistic large-scale dataset for HAR. Considering its great practical value, in this paper, we propose a large-scale benchmark dataset to bridge this gap, termed HARDVS, which contains 300 categories and more than 100K event sequences. We evaluate and report the performance of multiple popular HAR algorithms, which provide extensive baselines for future works to compare. More importantly, we propose a novel spatial-temporal feature learning and fusion framework, termed ESTF, for event stream based human activity recognition. It first projects the event streams into spatial and temporal embeddings using StemNet, then, encodes and fuses the dual-view representations using Transformer networks. Finally, the dual features are concatenated and fed into a classification head for activity prediction. Extensive experiments on multiple datasets fully validated the effectiveness of our model. 

<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/eventcompare.jpg" width="800px" align="center"> 


### News 
* :fire: [2023.12.09] Our paper is accepted by AAAI-2024 !!!
* :fire: [2023.05.29] The class label (i.e., category name) is available at [[HARDVS_300_class.txt](https://github.com/Event-AHU/HARDVS/blob/main/HARDVS_300_class.txt)]
* :fire: [2022.12.14] HARDVS dataset is integrated into the SNN toolkit [[SpikingJelly](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/hardvs.py)]  

 


### Demo Videos 

* **A demo video for the HARDVS dataset can be found by clicking the image below:** 
<p align="center">
  <a href="https://youtu.be/AZsbCAfYzac?si=_Y6i17Pt-bg2v0WR">
    <img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/hardvs_demo_screenshot.png" alt="DemoVideo" width="700"/>
  </a>
</p>


* **Video Tutorial for this work can be found by clicking the image below:**
<p align="center">
  <a href="https://youtu.be/OvE53dJWzoo?si=BEHpiuqpNURV3JaG">
    <img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/hardvs_tutorial_screeshot.png" alt="Tutorials" width="700"/>
  </a>
</p>


* **Representative samples of HARDVS can be found below:**
<p align="center">
  <a href="">
    <img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/HARDVS_all_samples.jpg" alt="Tutorials" width="700"/>
  </a>
</p>




### Dataset Download

<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/compareEventdatasets.png" width="800px" align="center">


* **Download from Baidu Disk**: 
```
  [Event Images] 链接：https://pan.baidu.com/s/1OhlhOBHY91W2SwE6oWjDwA?pwd=1234    提取码：1234
  [Compact Event file] 链接：https://pan.baidu.com/s/1iw214Aj5ugN-arhuxjmfOw?pwd=1234 提取码：1234
  [RGB Event Images] 链接：https://pan.baidu.com/s/1w-z86PH7mGY0CqVBj_MpNA?pwd=1234 提取码：1234
  [Raw Event file] To be updated 
```

* **Download from DropBox:**
```
  To be updated ... 
```




### Environment 
```
conda create -n event python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate event
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
pip3 install -e .
```



Details of each package: 

<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/package.jpg" 
  width="400px" >


### Our Proposed Approach 
<img src="https://github.com/Event-AHU/HARDVS/blob/main/figures/spatialtempHAR.jpg" width="800px" align="center"> 

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








