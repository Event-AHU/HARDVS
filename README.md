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

**(a).** Comparison between existing datasets and our proposed HARDVS 2.0 dataset for video classification. **(b).** A simple schematic diagram of our framework.

### News 


### HAR Datasets

<img src="https://github.com/Event-AHU/HARDVS/blob/HARDVSv2/pictures/har_dataset.png" width="800px" align="center">


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
conda create -n mmhco python=3.8 pytorch=1.12.1 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate mmhco
pip install -U openmim
mim install mmengine
mim install mmcv
pip install -e .
```

**Details of each package:**


### Our Proposed Approach 
<img src="https://github.com/Event-AHU/HARDVS/blob/HARDVSv2/pictures/HARDVS_extension_framework.jpg" width="800px" align="center"> 

**An overview of our proposed MMHCO-HAR framework for multi-modal human action recognition.** we propose a novel heat conduction-based multi-modal learning framework for efficient and effective RGB-Event based human activity recognition. Concretely, we first adopt a stem network to transform the input RGB frames and event streams into corresponding feature embeddings. Then, the multi-modal HCO blocks are proposed to achieve RGB and event feature learning and interaction simultaneously. The core operation is the DCT-IDCT transformation network equipped with modality-specific continuous Frequency Value Embeddings (FVEs). After that, we explore a multi-modal fusion method with a policy routing mechanism to facilitate adaptive feature fusion. Finally, a classification head is employed to obtain the recognition results. Compared with existing mainstream multi-modal fusion algorithm frameworks, such as Transformer, our adoption of the computationally less complex heat conduction model achieves high accuracy while offering better computational efficiency and physical interpretability. Additionally, our newly proposed routing mechanism-guided multi-modal fusion strategy enables more effective integration of RGB-Event features.


### Train & Test & Evaluation
```
# train
  python tools/train.py configs/recognition/mmhco/mmhco.py --seed 0 --deterministic --work-dir work_dirs/mmhco_train

# test
  python tools/test.py configs/recognition/mmhco/mmhco.py  path_to_checkpoint --eval top_k_accuracy
```





### Citation
If you find this work useful for your research, please cite the following paper and give us a :star2:.  
```bibtex
@article{wang2025human,
  title={Human Activity Recognition using RGB-Event based Sensors: A Multi-modal Heat Conduction Model and A Benchmark Dataset},
  author={Wang, Shiao and Wang, Xiao and Jiang, Bo and Zhu, Lin and Li, Guoqi and Wang, Yaowei and Tian, Yonghong and Tang, Jin},
  journal={arXiv preprint arXiv:2504.05830},
  year={2025}
}
```


### Acknowledgement and Other Useful Materials 
* **MMAction2**: [https://github.com/open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) 
* **SpikingJelly**: [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)
* **vHeat**: [https://github.com/MzeroMiko/vHeat](https://github.com/MzeroMiko/vHeat)








