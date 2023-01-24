# Weakly-supervised Anomaly Detection: A Survey

This repo is constructed for collecting and categorizing papers about weakly supervised anomaly detection models according to our survey paper——*Weakly Supervised Anomaly Detection: A Survey*

## Summary
| Method | Reference | Venue | Backbone | Modalities | Key Idea | Official Code |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|OE  |\cite{micenkova2014learning}       | KDD'14  |    -     | Tabular |  Anomaly feature representation learning | ×|
|XGBOD  |\cite{zhao2018xgbod}       | IJCNN'18  |   -      | Tabular |  Anomaly feature representation learning | √|
|DeepSAD  |\cite{ruff2019deep}       | ICLR'20  | MLP        | Tabular |  Anomaly feature representation learning | √|
|ESAD      |\cite{huang2020esad}      |    Preprint      |      MLP      |   Tabular      |  Anomaly feature representation learning                  | ×|
|DSSAD        |\cite{feng2021learning}   | ICASSP'21         |    CNN        | Image/Video        |          Anomaly feature representation learning          | ×|
|REPEN    | \cite{pang2018learning}  | KDD'18   | MLP        | Tabular |      Anomaly feature representation learning                   |  × |
|AA-BiGAN | \cite{tian2022anomaly}   | IJCAI'22 | GAN        | Tabular |         Anomaly feature representation learning                | √ |
|Dual-MGAN      |\cite{li2022dual}         |      TKDD'22     |    GAN       |    Tabular     |     Anomaly feature representation learning                    | √|
|DevNet  | \cite{pang2019deepdevnet} | KDD'19   | MLP       | Tabular |   Anomaly score learning  |  √|
|PReNet | \cite{pang2019deepprenet} |  Preprint        | MLP       | Tabular |  Anomaly score learning                         |  ×   |
|FEAWAD | \cite{zhou2021feature}   | TNNLS'21  | AE        | Tabular |     Anomaly score learning                       | √    |
|SNARE    | \cite{mcglohon2009snare} | KDD'09    | -          | Graph   |  Graph learning and label propagation |   ×  |
|AESOP    | \cite{tamersoy2014guilt} | KDD'14    |  -         | Graph   |  Graph learning and label propagation  |   ×  |
|SemiGNN  | \cite{wang2019semi}      | ICDM'19   | MLP+Attention | Graph   |  Graph learning and label propagation | ×    |
|SemiGAD   | \cite{kumagai2021semi}   | IJCNN'21  | GNN       | Graph   |  Graph learning and label propagation  | × |
|Meta-GDN | \cite{ding2021few}       | WWW'21    | GNN       | Graph   |   Graph learning and label propagation |  √   |
|SemiADC  | \cite{meng2021semi}      | IS Journal'21 | GAN   | Graph | Graph learning and label propagation|  × |
|SSAD     | \cite{gornitz2013toward} | JAIR'13   |  -         | Tabular |   Active learning                  |  ×   |
|AAD      | \cite{das2016incorporating} | ICDM'16 | -         | Tabular |  Active learning   |   √  |
|SLA-VAE    | \cite{huang2022semi}      |   WWW'22        |    VAE      |     Time series    |          Active learning                    | × |
|Meta-AAD | \cite{zha2020meta}       | ICDM'20   | MLP       | Tabular |  Reinforcement learning |   √  |
|DPLAN    | \cite{pang2021toward}     | KDD'21   | MLP       | Tabular |  Reinforcement learning  |  ×   |
|GraphUCB | \cite{ding2019interactive} | WSDM'19 |   -        | Graph   |  Reinforcement learning  |  √   |







## Catalogue
### AD with Incomplete Supervision
![](figures/Incomplete.png)
#### Anomaly Feature Representation Learning
**OE**\
[Learning outlier ensembles:The best of both worlds–supervised and unsupervised](https://www.semanticscholar.org/paper/Learning-Outlier-Ensembles%3A-The-Best-of-Both-Worlds-Micenkov%C3%A1-McWilliams/a517bc0a071be0982150c38c8ac3c52093995106)\
**XGBOD**\
[Xgbod: improving supervised outlier detection with unsupervised representation learning](https://ieeexplore.ieee.org/document/8489605)\
[Code Link](https://github.com/yzhao062/XGBOD)\
**DeepSAD**\
[Deep semi-supervised anomaly detection](https://iclr.cc/virtual_2020/poster_HkgH0TEYwH.html)\
[Code Link](https://github.com/lukasruff/Deep-SAD-PyTorch)\
**ESAD**\
[Esad: End-to-end deep semi-supervised anomaly detection](https://arxiv.org/abs/2012.04905v1)\
**REPEN**\
[Learning representations of ultrahigh-dimensional data for random distance-based outlier detection](https://dl.acm.org/doi/10.1145/3219819.3220042)\
**DSSAD**\
[Learning discriminative features for semi-supervised anomaly detection](https://ieeexplore.ieee.org/document/9414285)\
**AA-BiGAN**\
[Anomaly detection by leveraging incomplete anomalous knowledge with anomaly-aware bidirectional gans](https://www.ijcai.org/proceedings/2022/0313.pdf)\
[Code Link](https://github.com/tbw162/AA-BiGAN)\
**Dual-MGAN**\
[Dual-mgan: An efficient approach for semi-supervised outlier detection with few identified anomalies](https://dl.acm.org/doi/10.1145/3522690)\
[Code Link](https://github.com/leibinghe/Dual-MGAN)
#### Anomaly Score Learning
**DevNet**\
[Deep anomaly detection with deviation networks](https://dl.acm.org/doi/10.1145/3292500.3330871)\
[Code Link](https://github.com/GuansongPang/deviation-network)\
**PReNet**\
[Deep weakly-supervised anomaly detection](https://arxiv.org/abs/1910.13601)\
**FEAWAD**\
[Feature encoding with autoencoders for weakly supervised anomaly detection](https://ieeexplore.ieee.org/abstract/document/9465358/)\
[Code Link](https://github.com/yj-zhou/Feature_Encoding_with_AutoEncoders_for_Weakly-supervised_Anomaly_Detection)
#### Graph Learning
**SNARE**\
[Snare: a link analytic system for graph labeling and risk detection](https://www.cs.cmu.edu/~mmcgloho/pubs/snare.pdf)\
**AESOP**\
[Guilt by association: large scale malware detection by mining file-relation graphs](https://dl.acm.org/doi/abs/10.1145/2623330.2623342)\
**SemiGNN**\
[A semi-supervised graph attentive network for financial fraud detection](https://ieeexplore.ieee.org/document/8970829)\
**SemiGAD**\
[Semi-supervised anomaly detection on attributed graphs](https://ieeexplore.ieee.org/document/9533507)\
**Meta-GDN**\
[Few-shot network anomaly detection via cross-network meta-learning](https://dl.acm.org/doi/10.1145/3442381.3449922)\
[Code Link](https://github.com/kaize0409/Meta-GDN_AnomalyDetection)\
**SemiADC**\
[Semi-supervised anomaly detection in dynamic communication networks](www.sciencedirect.com/science/article/pii/S0020025521003856)\
**SSAD**\
[Toward supervised anomaly detection](https://dl.acm.org/doi/10.5555/2512538.2512545)\
**AAD**\
[Incorporating expert feedback into active anomaly discover](https://ieeexplore.ieee.org/document/7837915)\
[Code Link](https://github.com/shubhomoydas/pyaad)\
**GraphUCB**\
[Interactive anomaly detection on attributed networks](https://dl.acm.org/doi/10.1145/3289600.3290964)\
[Code Link](https://github.com/kaize0409/GraphUCB_AnomalyDetection)
#### Active learning and reinforcement learning
**Meta-AAD**\
[Meta-aad: Active anomaly detection with deep reinforcement learning](https://ieeexplore.ieee.org/document/9338270)\
[Code Link](https://github.com/daochenzha/Meta-AAD)\
**DPLAN**\
[Toward deep supervised anomaly detection: Reinforcement learning from partially labeled anomaly data](https://dl.acm.org/doi/10.1145/3447548.3467417)\
**SLA-VAE**\
[A semi-supervised vae based active anomaly detection framework in multivariate time series for online systems](https://dl.acm.org/doi/abs/10.1145/3485447.3511984)
### AD with Inexact Supervision
![](figures/Inexact.png)
#### MIL-based
**MIL**\
[Real-world anomaly detection in surveillance videos](https://ieeexplore.ieee.org/document/8578776)\
[Code Link](https://github.com/vantage-vision-vv/Anomaly-Detection-in-Surveillance-Videos)\
**AR-Net**\
[Weakly supervised video anomaly detection via center-guided discriminative learning](https://ieeexplore.ieee.org/document/9102722)\
[Code Link](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020)\
**TCN-IBL**\
[Temporal convolutional network with complementary inner bag loss for weakly supervised anomaly detection](https://ieeexplore.ieee.org/document/8803657)\
**RTFM**\
[Weakly-supervised video anomaly detection with robust temporal feature magnitude learning](https://ieeexplore.ieee.org/document/9710957)\
[Code Link](https://github.com/tianyu0207/RTFM)\
**Motion-Aware**\
[Motion-aware feature for improved video anomaly detection](https://arxiv.org/abs/1907.1021)\
**CRF-Attention**\
[Dance with self-attention: A new look of conditional random fields on anomaly detection in videos](https://ieeexplore.ieee.org/document/9710685)\
**MPRF**\
[Weakly-supervised spatio-temporal anomaly detection in surveillance video](https://www.ijcai.org/proceedings/2021/0162.pdf)\
**MCR**\
[Multi-scale continuity-aware refinement network for weakly supervised video anomaly detection](https://ieeexplore.ieee.org/abstract/document/9860012)\
**XEL**\
[Cross-epoch learning for weakly supervised anomaly detection in surveillance videos](https://ieeexplore.ieee.org/document/9560033)\
[Code Link](https://github.com/sdjsngs/XEL-WSAD)\
**MIST**\
[MIST: Multiple instance self-training framework for video anomaly detection](https://ieeexplore.ieee.org/document/9578773)\
[Code Link](https://github.com/fjchange/MIST_VAD)\
**MSLNet**\
[Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection](https://ojs.aaai.org/index.php/AAAI/article/download/20028/version/18325/19787)\
[Code Link](https://github.com/xidianai/MSL)
#### Non MIL-based
**SRF**\
[A self-reasoning framework for anomaly detection using video-level labels](https://ieeexplore.ieee.org/document/9204830)\
**WETAS**\
[Weakly supervised temporal anomaly seg- mentation with dynamic time warping](https://ieeexplore.ieee.org/document/9710920)\
#### Evaluating and Selecting Unsupervised methods
**Inexact AUC**\
[Anomaly detection with inexact labels](https://dl.acm.org/doi/10.1007/s10994-020-05880-w)\
**Isudra**\
[Indirectly supervised anomaly detection of clinically meaningful health events from smart home data](https://dl.acm.org/doi/10.1145/3439870)\
[Code Link](https://github.com/jb3dahmen/indirectsupervision)
### AD with Inaccurate Supervision
![](figures/Inaccurate.png)
#### Ensemble Learning
**LAC**\
[Fraud Detection under Multi-Sourced Extremely Noisy Annotations](https://dl.acm.org/doi/10.1145/3459637.3482433)\
**ADMoE**\
[ADMoE: Anomaly detection with mixture-of-experts from noisy labels](https://arxiv.org/abs/2208.11290)\
[Code Link](https://github.com/microsoft/admoe)
#### Denosing Network
**BGPAD**\
[Isp self-operated bgp anomaly detection based on weakly supervised learning](https://ieeexplore.ieee.org/document/9651957)\
[Code Link](https://github.com/universetao/A-General-Framework-BGP-Anomaly-Detection)\
**SemiADC**\
[Semi-supervised anomaly detection in dynamic communication networks](https://www.sciencedirect.com/science/article/pii/S0020025521003856)\
#### Graph Learning
**TSN**\
[Graph Convolutional Label Noise Cleaner: Train a Plug-And-Play Action Classifier for Anomaly Detection](https://ieeexplore.ieee.org/document/8953791)\
[Code Link](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)


## Experiment