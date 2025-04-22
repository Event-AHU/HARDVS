[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.05830)

<div align="center">

<img src="https://github.com/Event-AHU/HARDVS/blob/HARDVSv2/pictures/HARDVSv2_log.png" width="350px">
  
**A Multi-modal Heat Conduction Model and A Benchmark Dataset for RGB-Event based HAR**

------

<p align="center">
</p>

</div>

### Paper
Shiao Wang, Xiao Wang, Bo Jiang, Lin Zhu, Guoqi Li, Yaowei Wang, Yonghong Tian, Jin Tang. 
"**Human Activity Recognition using RGB-Event based Sensors: A Multi-modal Heat Conduction Model and A Benchmark Dataset.**
" arXiv preprint 	arXiv:2504.05830 (2025).  [[arXiv](https://arxiv.org/abs/2504.05830)] 



### Abstract
Human Activity Recognition (HAR) has long been a fundamental research direction in the field of computer vision. Previous studies have primarily relied on traditional RGB cameras to achieve high-performance activity recognition. However, the challenging factors in real-world scenarios, such as insufficient lighting and rapid movements, inevitably degrade the performance of RGB cameras. To address these challenges, biologically inspired event cameras offer a promising solution to overcome the limitations of traditional RGB cameras. In this work, we rethink human activity recognition by combining the RGB and event cameras. The first contribution is the proposed large-scale multi-modal RGB-Event human activity recognition benchmark dataset, termed HARDVS 2.0, which bridges the dataset gaps. It contains 300 categories of everyday real-world actions with a total of 107,646 paired videos covering various challenging scenarios. Inspired by the physics-informed heat conduction model, we propose a novel multi-modal heat conduction operation framework for effective activity recognition, termed MMHCO-HAR. More in detail, given the RGB frames and event streams, we first extract the feature embeddings using a stem network. Then, multi-modal Heat Conduction blocks are designed to fuse the dual features, the key module of which is the multi-modal Heat Conduction Operation (HCO) layer. We integrate RGB and event embeddings through a multi-modal DCT-IDCT layer while adaptively incorporating the thermal conductivity coefficient via FVEs (Frequency Value Embeddings) into this module. After that, we propose an adaptive fusion module based on a policy routing strategy for high-performance classification. We conduct comprehensive experiments comparing our proposed method with baseline methods on the HARDVS 2.0 dataset and other public datasets. These results demonstrate that our method consistently performs well, validating its effectiveness and robustness.

<img src="https://github.com/Event-AHU/HARDVS/blob/HARDVSv2/pictures/first_image.png" width="800px" align="center"> 


### News 


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








