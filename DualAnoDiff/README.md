# <center>Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation<center>
<!-- ::--:DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation::--: -->
[Ying Jin<sup>1#</sup>](),
[Jinlong Peng<sup>2#</sup>](https://scholar.google.com/citations?user=i5I-cIEAAAAJ&hl=zh-CN&oi=sra),
[Qingdong He<sup>2#</sup>](https://scholar.google.com/citations?user=gUJWww0AAAAJ&hl=zh-CN&oi=sra),
[Teng Hu<sup>3</sup>](https://scholar.google.com/citations?user=Jm5qsAYAAAAJ&hl=zh-CN&oi=sra),
[Hao Chen<sup>1</sup>](),
[Haoxuan Wang<sup>1</sup>](), 
[Jiafu Wu<sup>2</sup>](https://scholar.google.com/citations?user=tiQ_rv0AAAAJ&hl=zh-CN&oi=sra),
[wenbing zhu<sup>1</sup>](),
[Mingmin Chi<sup>1*</sup>](https://scholar.google.com/citations?user=Y8b1W00AAAAJ&hl=zh-CN&oi=sra),
[Jun Liu<sup>2</sup>](),
[Yabiao Wang<sup>2,4</sup>]()

(#Equal contribution,*Corresponding author)

<sup>1</sup>Fudan University, <sup>2</sup>Youtu Lab, Tencent, <sup>3</sup>Shanghai Jiao Tong University, <sup>4</sup>Zhe Jiang University

[[paper](https://arxiv.org/abs/2408.13509)]

## Abstract
The performance of anomaly inspection in industrial manufacturing is constrained by the scarcity of anomaly data. To overcome this challenge, researchers have started employing anomaly generation approaches to augment the anomaly dataset.
However, existing anomaly generation methods suffer from limited diversity in the generated anomalies and struggle to achieve a seamless blending of this anomaly with the original image. Moreover, the generated mask is usually not aligned with the generated anomaly. In this paper, we overcome these challenges from a new perspective, simultaneously generating a pair of the overall image and the corresponding anomaly part.
We propose **DualAnoDiff**, a novel diffusion-based few-shot anomaly image generation model, which can generate diverse and realistic anomaly images by using a dual-interrelated diffusion model, where one of them is employed to generate the whole image while the other one generates the anomaly part.
Moreover, we extract background and shape information to mitigate the distortion and blurriness phenomenon in few-shot image generation. 
Extensive experiments demonstrate the superiority of our proposed model over state-of-the-art methods in terms of diversity, realism and the accuracy of mask. Overall, our approach significantly improves the performance of downstream anomaly inspection tasks, including anomaly detection, anomaly localization, and anomaly classification tasks. Code will be made available.

# ✨Overview
<img width="720" alt="image" src="https://github.com/user-attachments/assets/27bd1be9-726a-4257-a160-5816317e1d43" />


# Getting Started
```
pip -r requirments.txt
cd dual-interrelated_diff # or bcm-dual-interrelated_diff
sh run_mvtec_split.py
# run_mvtec_split.py includes operations for training, inference, and mask generation. Since our method involves training the model for a single category, it is necessary to modify the name in the run_mvtec_split.py file, which represents the category to be generated from Mvtec.
```

# Data and checkpoints



# Result
![image](https://github.com/user-attachments/assets/7128b95d-3a35-4838-ad88-c2150afdee2d)

## Comparison in Anomaly Generation
### Anomaly Generation Quality
![image](https://github.com/user-attachments/assets/196d6147-f010-4c69-a5d5-89df94a80bb6)
### Anomaly Generation for Anomaly Detection and Localization
![image](https://github.com/user-attachments/assets/18e29fe2-b613-4fc2-98e3-1a5f2860b8a1)

## Comparison with Anomaly Detection Models
![image](https://github.com/user-attachments/assets/f793f984-e746-4d2d-bc1b-8d50144a0eb2)


# Citation
```
@article{jin2024dualanodiff,
  title={DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation},
  author={Jin, Ying and Peng, Jinlong and He, Qingdong and Hu, Teng and Chen, Hao and Wu, Jiafu and Zhu, Wenbing and Chi, Mingmin and Liu, Jun and Wang, Yabiao and others},
  journal={arXiv preprint arXiv:2408.13509},
  year={2024}
}
```


