# 👏 Survey of Deep Metric Learning

Traditionally, they have defined metrics in a variety of ways, including Euclidean distance and cosine similarity.


💡 I hope that many people will learn about metric learning through this repository.

🔔 Updated frequently.

<p align="center">
  <img width="820" height="250" src="/pic/Pedigree_of_metric_learning.png">
</p>

---
### 1️⃣ Euclidean-based metric

- Dimensionality Reduction by Learning an Invariant Mapping (Contrastive) (CVPR 2006) [[Paper]](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)[[Caffe]](https://github.com/wujiyang/Contrastive-Loss)[[Tensorflow]](https://github.com/ardiya/siamesenetwork-tensorflow)[[Keras]](https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py)[[Pytorch1]](https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)

- FaceNet: A Unified Embedding for Face Recognition and Clustering (Triplet) (CVPR 2015) [[Paper]](https://arxiv.org/abs/1503.03832)[[Tensorflow]](https://github.com/omoindrot/tensorflow-triplet-loss)[[Pytorch]](https://github.com/bnulihaixia/Deep_metric)

- Deep Metric Learning via Lifted Structured Feature Embedding (LSSS) (CVPR 2016) [[Paper]](https://arxiv.org/abs/1511.06452)[[Chainer]](https://github.com/ronekko/deep_metric_learning)[[Caffe]](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)[[Pytorch1]](https://github.com/zhengxiawu/pytorch_deep_metric_learning)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)

- Improved Deep Metric Learning with Multi-class N-pair Loss Objective (N-pair) (NIPS 2016) [[Paper]](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)[[Pytorch]](https://github.com/ChaofWang/Npair_loss_pytorch)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Beyond triplet loss: a deep quadruplet network for person re-identification (Quadruplet) (CVPR 2017) [[Paper]](https://cvip.computing.dundee.ac.uk/papers/Chen_CVPR_2017_paper.pdf)

- Deep Metric Learning via Facility Location (CVPR 2017) [[Paper]](https://arxiv.org/abs/1612.01213)[[Tensorflow]](https://github.com/CongWeilin/cluster-loss-tensorflow)

- No Fuss Distance Metric Learning using Proxies (Proxy NCA) (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf)[[Pytorch1]](https://github.com/dichotomies/proxy-nca)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Deep Metric Learning with Angular Loss (Angular) (CVPR 2017) [[Paper]](https://arxiv.org/abs/1708.01682)[[Tensorflow]](https://github.com/geonm/tf_angular_loss)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Ranked List Loss for Deep Metric Learning (RLL) (CVPR 2019) [[Paper]](https://arxiv.org/abs/1903.03238)

- Hardness-Aware Deep Metric Learning (HDML) (CVPR 2019) [[Paper]](https://arxiv.org/abs/1903.05503)[[Tensorflow]](https://github.com/wzzheng/HDML)

- Deep Metric Learning to Rank (CVPR 2019) [[Paper]](http://cs-people.bu.edu/hekun/papers/CVPR2019FastAP.pdf)

- Deep Metric Learning Beyond Binary Supervision (CVPR 2019) [[Paper]](https://arxiv.org/pdf/1904.09626.pdf)

- Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning (CVPR 2019) [[Paper]](https://arxiv.org/pdf/1904.02616.pdf)

- Density Aware Deep Metric Learning (CVPR 2019) [[Paper]](https://arxiv.org/pdf/1904.03911.pdf)

- A Theoretically Sound Upper Bound on the Triplet Loss for Improving the Efficiency of Deep Distance Metric Learning (CVPR 2019) [[paper]](https://arxiv.org/pdf/1904.08720.pdf)

- Deep Metric Learning by Online Soft Mining and Class-Aware Attention (AAAI 2019) [[Paper]](https://arxiv.org/pdf/1811.01459.pdf)



---
### 2️⃣ Similarity-based metric

- Deep Metric Learning for Practical Person Re-Identification [[Paper]](https://arxiv.org/abs/1407.4979)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)[[Pytorch]](https://github.com/bnulihaixia/Deep_metric)

- Learning Deep Embeddings with Histogram Loss (NIPS 2016) [[Paper]](https://papers.nips.cc/paper/6464-learning-deep-embeddings-with-histogram-loss)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)[[Pytorch]](https://github.com/valerystrizh/pytorch-histogram-loss)[[Caffe]](https://github.com/madkn/HistogramLoss)

- Learning Deep Disentangled Embeddings With the F-Statistic Loss (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7303-learning-deep-disentangled-embeddings-with-the-f-statistic-loss)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)

---
### 3️⃣ Integrated framework

- Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7293-adapted-deep-embeddings-a-synthesis-of-methods-for-k-shot-inductive-transfer-learning)[[Tensorflow]](https://github.com/tylersco/adapted_deep_embeddings)

---
### 4️⃣ Ensemble methods

- BIER-Boosting Independent Embeddings Robustly (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Opitz_BIER_-_Boosting_ICCV_2017_paper.pdf)[[Tensorflow]](https://github.com/mop/bier)

- Hard-Aware Deeply Cascaded Embedding (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)[[Caffe]](https://github.com/PkuRainBow/HardAwareDeeplyCascadedEmbedding.caffe)

- Deep Adversarial Metric Learning (CVPR 2018) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)[[Chainer]](https://github.com/duanyq14/DAML)

- Deep Randomized Ensembles for Metric Learning (ECCV 2018) [[Paper]](https://arxiv.org/abs/1808.04469)[[Pytorch]](https://github.com/littleredxh/DREML)

- Attention-based Ensemble for Deep Metric Learning (ECCV 2018) [[Paper]](https://arxiv.org/abs/1804.00382)

- Deep Metric Learning with Hierarchical Triplet Loss (ECCV 2018) [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf)

- Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning (CVPR 2019) [[Paper]](https://arxiv.org/pdf/1904.06627.pdf)




---
### 5️⃣ Applications
#### Person re-identification

- Person Re-Identification using Kernel-based Metric Learning Methods (ECCV 2014) [[Paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.673.1976&rep=rep1&type=pdf)[[Matlab]](https://github.com/NEU-Gou/kernel-metric-learning-reid)

- Similarity Learning on an Explicit Polynomial Kernel Feature Map for Person Re-Identification (CVPR 2015) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Chen_Similarity_Learning_on_2015_CVPR_paper.pdf)

- Learning to rank in person re-identification with metric ensembles (CVPR 2015) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Paisitkriangkrai_Learning_to_Rank_2015_CVPR_paper.pdf)

- Person Re-identification by Local Maximal Occurrence Representation and Metric Learning (CVPR 2015) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Liao_Person_Re-Identification_by_2015_CVPR_paper.pdf)[[Matlab]](https://github.com/liangzheng06/MARS-evaluation/tree/master/LOMO_XQDA)

- Learning a Discriminative Null Space for Person Re-identification (CVPR 2016) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Learning_a_Discriminative_CVPR_2016_paper.pdf)[[Matlab]](https://github.com/lzrobots/NullSpace_ReID)

- Similarity Learning with Spatial Constraints for Person Re-identification (CVPR 2016) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_Similarity_Learning_With_CVPR_2016_paper.pdf)[[Matlab]](https://github.com/dapengchen123/SCSP)

- Consistent-Aware Deep Learning for Person Re-identification in a Camera Network (CVPR 2016) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Consistent-Aware_Deep_Learning_CVPR_2017_paper.pdf)

- Re-ranking Person Re-identification with k-reciprocal Encoding (CVPR 2017) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf)[[Caffe]](https://github.com/zhunzhong07/person-re-ranking)

- Scalable Person Re-identification on Supervised Smoothed Manifold (CVPR 2017) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bai_Scalable_Person_Re-Identification_CVPR_2017_paper.pdf)

- One-Shot Metric Learning for Person Re-identification (CVPR 2017) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bak_One-Shot_Metric_Learning_CVPR_2017_paper.pdf)

- Point to Set Similarity Based Deep Feature Learning for Person Re-identification (CVPR 2017) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Point_to_Set_CVPR_2017_paper.pdf)

- Consistent-Aware Deep Learning for Person Re-identification in a Camera Network (CVPR 2017) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Consistent-Aware_Deep_Learning_CVPR_2017_paper.pdf)

- Cross-view Asymmetric Metric Learning for Unsupervised Person Re-identification (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yu_Cross-View_Asymmetric_Metric_ICCV_2017_paper.pdf)[[Matlab]](https://github.com/KovenYu/CAMEL)

- Efficient Online Local Metric Adaptation via Negative Samples for Person Re-Identification (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Efficient_Online_Local_ICCV_2017_paper.pdf)

- Mask-guided Contrastive Attention Model for Person Re-Identification (CVPR 2018) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Mask-Guided_Contrastive_Attention_CVPR_2018_paper.pdf)[[Caffe]](https://github.com/developfeng/MGCAM)

- Efficient and Deep Person Re-Identification using Multi-Level Similarity (CVPR 2018) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Guo_Efficient_and_Deep_CVPR_2018_paper.pdf)

- Group Consistent Similarity Learning via Deep CRF for Person Re-Identification (CVPR 2018) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Group_Consistent_Similarity_CVPR_2018_paper.pdf)[[Pytorch]](https://github.com/dapengchen123/crf_affinity)


#### Face verification

- Discriminative Deep Metric Learning for Face Verification in the Wild (CVPR 2014) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Hu_Discriminative_Deep_Metric_2014_CVPR_paper.pdf)


#### Face recognition

- Fusing Robust Face Region Descriptors via Multiple Metric Learning for Face Recognition in the Wild (CVPR 2013) [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Cui_Fusing_Robust_Face_2013_CVPR_paper.pdf)


#### Point-cloud segmentation

- Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning (CVPR 2019) [[Paper]](https://arxiv.org/pdf/1904.02113.pdf)


#### Image registration

- Metric Learning for Image Registration (CVPR 2019) [[Paper]](https://arxiv.org/pdf/1904.09524.pdf)


---
### 6️⃣ Related works

#### NIPS

- Distance Metric Learning for Large Margin Nearest Neighbor Classification (NIPS 2005) [[Paper]](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification)[[Journal]](http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf)[[Code]](https://github.com/johny-c/pylmnn)

- Metric Learning by Collapsing Classes (NIPS 2005) [[Paper]](https://papers.nips.cc/paper/2947-metric-learning-by-collapsing-classes)

- Online Metric Learning and Fast Similarity Search (NIPS 2008) [[Paper]](https://papers.nips.cc/paper/3446-online-metric-learning-and-fast-similarity-search)

- Sparse Metric Learning via Smooth Optimization (NIPS 2009) [[Paper]](https://papers.nips.cc/paper/3847-sparse-metric-learning-via-smooth-optimization)

- Metric Learning with Multiple Kernels (NIPS 2011) [[Paper]](https://papers.nips.cc/paper/4399-metric-learning-with-multiple-kernels)

- Hamming Distance Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4808-hamming-distance-metric-learning)[[Matlab]](https://github.com/norouzi/hdml)

- Parametric Local Metric Learning for Nearest Neighbor Classification (NIPS 2012) [[Paper]](http://papers.nips.cc/paper/4818-parametric-local-metric-learning-for-nearest-neighbor-classification)

- Non-linear Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4840-non-linear-metric-learning)

- Latent Coincidence Analysis: A Hidden Variable Model for Distance Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4634-latent-coincidence-analysis-a-hidden-variable-model-for-distance-metric-learning)
  - They deal with probabilistic model based on EM algorithm

- Semi-Crowdsourced Clustering: Generalizing Crowd Labeling by Robust Distance Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4688-semi-crowdsourced-clustering-generalizing-crowd-labeling-by-robust-distance-metric-learning)

- Discriminative Metric Learning by Neighborhood Gerrymandering (NIPS 2014) [[Paper]](https://papers.nips.cc/paper/5385-discriminative-metric-learning-by-neighborhood-gerrymandering)

- Log-Hilbert-Schmidt metric between positive definite operators on Hilbert spaces (NIPS 2014) [[Paper]](https://papers.nips.cc/paper/5457-log-hilbert-schmidt-metric-between-positive-definite-operators-on-hilbert-spaces)

- Metric Learning for Temporal Sequence Alignment (NIPS 2014) [[paper]](http://papers.nips.cc/paper/5383-metric-learning-for-temporal-sequence-alignment)

- Sample complexity of learning Mahalanobis distance metrics (NIPS 2015) [[Paper]](https://arxiv.org/abs/1505.02729)

- Regressive Virtual Metric Learning (NIPS 2015) [[Paper]](https://papers.nips.cc/paper/5687-regressive-virtual-metric-learning)

- Improved Error Bounds for Tree Representations of Metric Spaces (NIPS 2016) [[Paper]](https://papers.nips.cc/paper/6431-improved-error-bounds-for-tree-representations-of-metric-spaces)

- What Makes Objects Similar: A Unified Multi-Metric Learning Approach (NIPS 2016) [[Paper]](https://papers.nips.cc/paper/6192-what-makes-objects-similar-a-unified-multi-metric-learning-approach)

- Learning Low-Dimensional Metrics (NIPS 2017) [[Paper]](https://papers.nips.cc/paper/7002-learning-low-dimensional-metrics)

- Generative Local Metric Learning for Kernel Regression (NIPS 2017) [[Paper]](https://papers.nips.cc/paper/6839-generative-local-metric-learning-for-kernel-regression)

- Persistence Fisher Kernel: A Riemannian Manifold Kernel for Persistence Diagrams (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/8205-persistence-fisher-kernel-a-riemannian-manifold-kernel-for-persistence-diagrams)[[Matlab]](https://github.com/lttam/PersistenceFisher)

- Bilevel Distance Metric Learning for Robust Image Recognition (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7674-bilevel-distance-metric-learning-for-robust-image-recognition)


#### ICLR

- Deep Metric Learning Using Triplet Network (ICLR 2015 workshop) [[Paper]](https://arxiv.org/abs/1412.6622)[[Keras]](https://github.com/Ariel-Perez/triplet-net)[[Torch]](https://github.com/eladhoffer/TripletNet)

- Metric Learning with Adaptive Density Discrimination (Magnet loss) (ICLR 2016) [[Paper]](https://arxiv.org/abs/1511.05939)[[Pytorch1]](https://github.com/vithursant/MagnetLoss-PyTorch)[[Pytorch2]](https://github.com/mbanani/pytorch-magnet-loss)[[Tensorflow]](https://github.com/pumpikano/tf-magnet-loss)

- Learning Wasserstein Embedding (ICLR 2018) [[Paper]](https://openreview.net/pdf?id=SJyEH91A-)[[Keras]](https://github.com/mducoffe/Learning-Wasserstein-Embeddings)

#### ICCV

- From Point to Set: Extend the Learning of Distance Metrics (ICCV 2013) [[Paper]](http://openaccess.thecvf.com/content_iccv_2013/papers/Zhu_From_Point_to_2013_ICCV_paper.pdf)


---
### 7️⃣ Study materials

#### Tutorial

- Visual Search (Image Retrieval) and Metric Learning (CVPR 2018) [[Video]](https://www.youtube.com/watch?v=iW_9fvw6YtI)


#### Lecture

- Topology and Manifold (International Winter School on Gravity and Light 2015) [[Video]](https://www.youtube.com/watch?v=7G4SqIboeig&list=PLFeEvEPtX_0S6vxxiiNPrJbLu9aK1UVC_)

#### Repository

- Person re-identification in Pytorch [[Site]](https://github.com/KaiyangZhou/deep-person-reid)





---
### Milestone

- [x] Add Euclidean-based metric

- [x] Add Similarity-based metric

- [x] Add Ensemble-based metric

- [x] Add applications

- [x] Add study materials

- [ ] Add brief descriptions