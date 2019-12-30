# 👏 Survey of Deep Metric Learning

Traditionally, they have defined metrics in a variety of ways, including pairwise distance, similarity, and probability distribution.


💡 Note that most of all approaches in survey are included in `supervised distance metric learning`.

🔔 Updated frequently.

<p align="center">
  <img width="820" height="250" src="/pic/pedigree.png">
</p>


---
## Contents

- [Pairwise cost methods](#pcm)
- [Distribution or other variant methods](#dvm)
- [Probabilistic methods](#pm)
- [Boost-like methods](#bm)
- [Applications](#app)
  - [Re-identification](#reid)
  - [Face verification](#face_ver)
  - [Face recognition](#face_rec)
  - [Segmentation](#seg)
  - [Image registration](#regist)
  - [Few-shot approach](#few-shot)
  - [3D reconstruction](#3d_recon)
- [Related works](#related)
  - [Neurips](#neurips)
  - [ICLR](#iclr)
  - [ICML](#icml)
  - [ICIP](#icip)
- [Study materials](#study)
  - [Tutorial](#tutorial)
  - [Lecture](#lecture)
  - [Repository](#repo)
  
  
---
<a name="pcm" />

### 1️⃣ Pairwise cost methods

- Dimensionality Reduction by Learning an Invariant Mapping (__Contrastive__) (CVPR 2006) [[Paper]](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)[[Caffe]](https://github.com/wujiyang/Contrastive-Loss)[[Tensorflow]](https://github.com/ardiya/siamesenetwork-tensorflow)[[Keras]](https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py)[[Pytorch1]](https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)

- From Point to Set: Extend the Learning of Distance Metrics (ICCV 2013) [[Paper]](http://openaccess.thecvf.com/content_iccv_2013/papers/Zhu_From_Point_to_2013_ICCV_paper.pdf)

- FaceNet: A Unified Embedding for Face Recognition and Clustering (__Triplet__) (CVPR 2015) [[Paper]](https://arxiv.org/abs/1503.03832)[[Tensorflow]](https://github.com/omoindrot/tensorflow-triplet-loss)[[Pytorch]](https://github.com/bnulihaixia/Deep_metric)

- Deep Metric Learning via Lifted Structured Feature Embedding (__LSSS__) (CVPR 2016) [[Paper]](https://arxiv.org/abs/1511.06452)[[Chainer]](https://github.com/ronekko/deep_metric_learning)[[Caffe]](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)[[Pytorch1]](https://github.com/zhengxiawu/pytorch_deep_metric_learning)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)

- Improved Deep Metric Learning with Multi-class N-pair Loss Objective (__N-pair__) (NIPS 2016) [[Paper]](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)[[Pytorch]](https://github.com/ChaofWang/Npair_loss_pytorch)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Beyond triplet loss: a deep quadruplet network for person re-identification (__Quadruplet__) (CVPR 2017) [[Paper]](https://cvip.computing.dundee.ac.uk/papers/Chen_CVPR_2017_paper.pdf)

- Deep Metric Learning via Facility Location (CVPR 2017) [[Paper]](https://arxiv.org/abs/1612.01213)[[Tensorflow]](https://github.com/CongWeilin/cluster-loss-tensorflow)

- No Fuss Distance Metric Learning using Proxies (__Proxy NCA__) (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf)[[Pytorch1]](https://github.com/dichotomies/proxy-nca)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Deep Metric Learning with Angular Loss (__Angular__) (CVPR 2017) [[Paper]](https://arxiv.org/abs/1708.01682)[[Tensorflow]](https://github.com/geonm/tf_angular_loss)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Deep Metric Learning by Online Soft Mining and Class-Aware Attention (AAAI 2019) [[Paper]](https://arxiv.org/pdf/1811.01459.pdf)

- Deep Metric Learning Beyond Binary Supervision (__Log-loss__) (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Deep_Metric_Learning_Beyond_Binary_Supervision_CVPR_2019_paper.pdf)

- Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning (__DSML__) (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

- Ranked List Loss for Deep Metric Learning (__RLL__) (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

- Deep Metric Learning to Rank (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf)

- SoftTriple Loss: Deep Metric Learning Without Triplet Sampling (__Soft-Trip__) (ICCV 2019) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)[[Tensorflow]](https://github.com/geonm/tf_SoftTriple_loss)

- Curvilinear Distance Metric Learning (__CDML__) (NIPS 2019) [[Paper]](https://papers.nips.cc/paper/8675-curvilinear-distance-metric-learning.pdf)

---
<a name="dvm" />

### 2️⃣ Distribution or other variant methods

- Image Set Classification Using Holistic Multiple Order Statistics Features and Localized Multi-Kernel Metric Learning (ICCV 2013) [[Paper]](http://openaccess.thecvf.com/content_iccv_2013/papers/Lu_Image_Set_Classification_2013_ICCV_paper.pdf)

- Deep Metric Learning for Practical Person Re-Identification (__Binomial deviance__) (ICPR 2014) [[Paper]](https://arxiv.org/abs/1407.4979)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)[[Pytorch]](https://github.com/bnulihaixia/Deep_metric)

- Learning Deep Embeddings with Histogram Loss (__Histogram__) (NIPS 2016) [[Paper]](https://papers.nips.cc/paper/6464-learning-deep-embeddings-with-histogram-loss)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)[[Pytorch]](https://github.com/valerystrizh/pytorch-histogram-loss)[[Caffe]](https://github.com/madkn/HistogramLoss)

- Learning Deep Disentangled Embeddings With the F-Statistic Loss (__F-stat__) (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7303-learning-deep-disentangled-embeddings-with-the-f-statistic-loss)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)

- Deep Asymmetric Metric Learning via Rich Relationship Mining (__DAML__) (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.pdf)

- Hardness-Aware Deep Metric Learning (__HDML__) (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.pdf)[[Tensorflow]](https://github.com/wzzheng/HDML)

- Deep Meta Metric Learning (__DMML__) (ICCV 2019) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.pdf)[[Pytorch]](https://github.com/CHENGY12/DMML)


---
<a name="pm" />

### 3️⃣ Probabilistic methods

- Latent Coincidence Analysis: A Hidden Variable Model for Distance Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4634-latent-coincidence-analysis-a-hidden-variable-model-for-distance-metric-learning.pdf)

- Information-theoretic Semi-supervised Metric Learning via Entropy Regularization (ICML 2014) [[Paper]](https://icml.cc/Conferences/2012/papers/74.pdf)

- Learning Deep Disentangled Embeddings With the F-Statistic Loss (__F-stat__) (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7303-learning-deep-disentangled-embeddings-with-the-f-statistic-loss)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)


---
<a name="bm" />

### 4️⃣ Boost-like methods

- BIER-Boosting Independent Embeddings Robustly (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Opitz_BIER_-_Boosting_ICCV_2017_paper.pdf)[[Tensorflow]](https://github.com/mop/bier)

- Hard-Aware Deeply Cascaded Embedding (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)[[Caffe]](https://github.com/PkuRainBow/HardAwareDeeplyCascadedEmbedding.caffe)

- Learning Spread-out Local Feature Descriptors (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Learning_Spread-Out_Local_ICCV_2017_paper.pdf)

- Deep Adversarial Metric Learning (CVPR 2018) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)[[Chainer]](https://github.com/duanyq14/DAML)

- Deep Randomized Ensembles for Metric Learning (ECCV 2018) [[Paper]](https://arxiv.org/abs/1808.04469)[[Pytorch]](https://github.com/littleredxh/DREML)

- Attention-based Ensemble for Deep Metric Learning (ECCV 2018) [[Paper]](https://arxiv.org/abs/1804.00382)

- Deep Metric Learning with Hierarchical Triplet Loss (ECCV 2018) [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf)

- Hybrid-Attention based Decoupled Metric Learning for Zero-Shot Image Retrieval (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid-Attention_Based_Decoupled_Metric_Learning_for_Zero-Shot_Image_Retrieval_CVPR_2019_paper.pdf) [[Caffe]](https://github.com/chenbinghui1/Hybrid-Attention-based-Decoupled-Metric-Learning)

- Divide and Conquer the Embedding Space for Metric Learning (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf) [[Code]](https://github.com/CompVis/metric-learning-divide-and-conquer) (__Not released yet__)

- Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

- Stochastic Class-based Hard Example Mining for Deep Metric Learning (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

- Distilled Person Re-identification: Towards a More Scalable System (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Distilled_Person_Re-Identification_Towards_a_More_Scalable_System_CVPR_2019_paper.pdf)

- Deep Metric Learning with Tuplet Margin Loss (ICCV 2019) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)

- Metric Learning with HORDE: High-Order Regularizer for Deep Embeddings (ICCV 2019) [[Paper]](https://arxiv.org/pdf/1908.02735.pdf)[[Code]](https://github.com/pierre-jacob/ICCV2019-Horde)

- MIC: Mining Interclass Characteristics for Improved Metric Learning (ICCV 2019) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.pdf)[[Pytorch]](https://github.com/Confusezius/metric-learning-mining-interclass-characteristics)



---
<a name="app" />

### 5️⃣ Applications

<a name="reid" />

#### Re-identification

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

- Perceive Where to Focus: Learning Visibility-aware Part-level Features for Partial Person Re-identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Perceive_Where_to_Focus_Learning_Visibility-Aware_Part-Level_Features_for_Partial_CVPR_2019_paper.pdf)

- Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification (CVPR 2019) [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Invariance_Matters_Exemplar_Memory_for_Domain_Adaptive_Person_Re-Identification_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/zhunzhong07/ECN)

- Learning to Reduce Dual-level Discrepancy for Infrared-Visible Person Re-identification (CVPR2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_to_Reduce_Dual-Level_Discrepancy_for_Infrared-Visible_Person_Re-Identification_CVPR_2019_paper.pdf)

- Densely Semantically Aligned Person Re-Identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Densely_Semantically_Aligned_Person_Re-Identification_CVPR_2019_paper.pdf)

- Generalizable Person Re-identification by Domain-Invariant Mapping Network (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Song_Generalizable_Person_Re-Identification_by_Domain-Invariant_Mapping_Network_CVPR_2019_paper.pdf)

- Re-ranking via Metric Fusion for Object Retrieval and Person Re-identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Bai_Re-Ranking_via_Metric_Fusion_for_Object_Retrieval_and_Person_Re-Identification_CVPR_2019_paper.pdf)

- Weakly Supervised Person Re-Identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Meng_Weakly_Supervised_Person_Re-Identification_CVPR_2019_paper.pdf)

- Towards Rich Feature Discovery with Class Activation Maps Augmentation for Person Re-Identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Towards_Rich_Feature_Discovery_With_Class_Activation_Maps_Augmentation_for_CVPR_2019_paper.pdf)

- Joint Discriminative and Generative Learning for Person Re-identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Joint_Discriminative_and_Generative_Learning_for_Person_Re-Identification_CVPR_2019_paper.pdf)

- Unsupervised Person Re-identification by Soft Multilabel Learning (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Unsupervised_Person_Re-Identification_by_Soft_Multilabel_Learning_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/KovenYu/MAR)

- Patch-based Discriminative Feature Learning for Unsupervised Person Re-identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Patch-Based_Discriminative_Feature_Learning_for_Unsupervised_Person_Re-Identification_CVPR_2019_paper.pdf)

- Attribute-Driven Feature Disentangling and Temporal Aggregation for Video Person Re-Identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Attribute-Driven_Feature_Disentangling_and_Temporal_Aggregation_for_Video_Person_Re-Identification_CVPR_2019_paper.pdf)

- AANet: Attribute Attention Network for Person Re-Identifications (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tay_AANet_Attribute_Attention_Network_for_Person_Re-Identifications_CVPR_2019_paper.pdf)

- VRSTC: Occlusion-Free Video Person Re-Identification (CVPR 2019) [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_VRSTC_Occlusion-Free_Video_Person_Re-Identification_CVPR_2019_paper.pdf)

- Adaptive Transfer Network for Cross-Domain Person Re-Identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_Transfer_Network_for_Cross-Domain_Person_Re-Identification_CVPR_2019_paper.pdf)

- Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Pyramidal_Person_Re-IDentification_via_Multi-Loss_Dynamic_Training_CVPR_2019_paper.pdf)

- Interaction-and-Aggregation Network for Person Re-identification (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Interaction-And-Aggregation_Network_for_Person_Re-Identification_CVPR_2019_paper.pdf)

- Vehicle Re-identification with Viewpoint-aware Metric Learning (ICCV 2019) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_Vehicle_Re-Identification_With_Viewpoint-Aware_Metric_Learning_ICCV_2019_paper.pdf)[[Code]](https://github.com/RyanRuihanG/VANet_labels)


<a name="face_ver" />

#### Face verification

- Discriminative Deep Metric Learning for Face Verification in the Wild (CVPR 2014) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Hu_Discriminative_Deep_Metric_2014_CVPR_paper.pdf)

- Fantope Regularization in Metric Learning (CVPR 2014) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Law_Fantope_Regularization_in_2014_CVPR_paper.pdf)

- Deep Transfer Metric Learning (CVPR 2015) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Hu_Deep_Transfer_Metric_2015_CVPR_paper.pdf)


<a name="face_rec" />

#### Face recognition

- Large Scale Metric Learning from Equivalence Constraints (CVPR 2012) [[Paper]](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/lrs/pubs/koestinger_cvpr_2012.pdf)

- Fusing Robust Face Region Descriptors via Multiple Metric Learning for Face Recognition in the Wild (CVPR 2013) [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Cui_Fusing_Robust_Face_2013_CVPR_paper.pdf)

- Similarity Metric Learning for Face Recognition (ICCV 2013) [[Paper]](http://openaccess.thecvf.com/content_iccv_2013/papers/Cao_Similarity_Metric_Learning_2013_ICCV_paper.pdf)

- Projection Metric Learning on Grassmann Manifold with Application to Video based Face Recognition (CVPR 2015) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Huang_Projection_Metric_Learning_2015_CVPR_paper.pdf)


<a name="seg" />

#### Segmentation

- Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Landrieu_Point_Cloud_Oversegmentation_With_Graph-Structured_Deep_Metric_Learning_CVPR_2019_paper.pdf)

- 3D Instance Segmentation via Multi-Task Metric Learning (ICCV 2019) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf)


<a name="regist" />

#### Image registration

- Metric Learning for Image Registration (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Niethammer_Metric_Learning_for_Image_Registration_CVPR_2019_paper.pdf)


<a name="few-shot" />

#### Few-shot approach

- RepMet: Representative-based metric learning for classification and few-shot object detection (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karlinsky_RepMet_Representative-Based_Metric_Learning_for_Classification_and_Few-Shot_Object_Detection_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/HaydenFaulkner/pytorch.repmet)

- Revisiting Metric Learning for Few-Shot Image Classification (arXiv) [[Paper]](https://arxiv.org/pdf/1907.03123.pdf)


<a name="3d_recon" />

#### 3D reconstruction

- Learning Embedding of 3D models with Quadric Loss (BMVC 2019) [[Paper]](https://bmvc2019.org/wp-content/uploads/papers/0452-paper.pdf)[[Pytorch]](https://github.com/nitinagarwal/QuadricLoss)

---
<a name="related" />

### 6️⃣ Related works

<a name="neurips" />

#### Neurips

- Distance Metric Learning for Large Margin Nearest Neighbor Classification (NIPS 2005) [[Paper]](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification)[[Journal]](http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf)[[Code]](https://github.com/johny-c/pylmnn)
 - First approach of local metric learning

- Metric Learning by Collapsing Classes (NIPS 2005) [[Paper]](https://papers.nips.cc/paper/2947-metric-learning-by-collapsing-classes)

- Online Metric Learning and Fast Similarity Search (NIPS 2008) [[Paper]](https://papers.nips.cc/paper/3446-online-metric-learning-and-fast-similarity-search)

- Sparse Metric Learning via Smooth Optimization (NIPS 2009) [[Paper]](https://papers.nips.cc/paper/3847-sparse-metric-learning-via-smooth-optimization)

- Metric Learning with Multiple Kernels (NIPS 2011) [[Paper]](https://papers.nips.cc/paper/4399-metric-learning-with-multiple-kernels)

- Hamming Distance Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4808-hamming-distance-metric-learning)[[Matlab]](https://github.com/norouzi/hdml)

- Parametric Local Metric Learning for Nearest Neighbor Classification (NIPS 2012) [[Paper]](http://papers.nips.cc/paper/4818-parametric-local-metric-learning-for-nearest-neighbor-classification)
 * A representative approach of local metric learning

- Non-linear Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4840-non-linear-metric-learning)

- Latent Coincidence Analysis: A Hidden Variable Model for Distance Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4634-latent-coincidence-analysis-a-hidden-variable-model-for-distance-metric-learning)
  - They deal with probabilistic model based on EM algorithm

- Semi-Crowdsourced Clustering: Generalizing Crowd Labeling by Robust Distance Metric Learning (NIPS 2012) [[Paper]](https://papers.nips.cc/paper/4688-semi-crowdsourced-clustering-generalizing-crowd-labeling-by-robust-distance-metric-learning)

- Similarity Component Analysis (NIPS 2013) [[Paper]](https://papers.nips.cc/paper/5015-similarity-component-analysis.pdf)

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

- Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7293-adapted-deep-embeddings-a-synthesis-of-methods-for-k-shot-inductive-transfer-learning)[[Tensorflow]](https://github.com/tylersco/adapted_deep_embeddings)

- A Theoretically Sound Upper Bound on the Triplet Loss for Improving the Efficiency of Deep Distance Metric Learning (CVPR 2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Do_A_Theoretically_Sound_Upper_Bound_on_the_Triplet_Loss_for_CVPR_2019_paper.pdf)

<a name="iclr" />

#### ICLR

- Deep Metric Learning Using Triplet Network (ICLR 2015 workshop) [[Paper]](https://arxiv.org/abs/1412.6622)[[Keras]](https://github.com/Ariel-Perez/triplet-net)[[Torch]](https://github.com/eladhoffer/TripletNet)

- Metric Learning with Adaptive Density Discrimination (Magnet loss) (ICLR 2016) [[Paper]](https://arxiv.org/abs/1511.05939)[[Pytorch1]](https://github.com/vithursant/MagnetLoss-PyTorch)[[Pytorch2]](https://github.com/mbanani/pytorch-magnet-loss)[[Tensorflow]](https://github.com/pumpikano/tf-magnet-loss)

- Semi-supervised Deep Learning by Metric Embedding (ICLRW 2017) [[Paper]](https://openreview.net/pdf?id=r1R5Z19le)[[Torch(Lua)]](https://github.com/eladhoffer/SemiSupContrast)

- Learning Wasserstein Embedding (ICLR 2018) [[Paper]](https://openreview.net/pdf?id=SJyEH91A-)[[Keras]](https://github.com/mducoffe/Learning-Wasserstein-Embeddings)

- Smoothing the Geometry of Probabilistic Box Embeddings (ICLR 2019) [[Paper]](https://openreview.net/pdf?id=H1xSNiRcF7)[[Tensorflow]](https://github.com/Lorraine333/smoothed_box_embedding)
  - New type of embedding method

- Unsupervised Domain Adaptation for Distance Metric Learning (ICLR 2019) [[Paper]](https://openreview.net/pdf?id=BklhAj09K7)

- ROTATE: Knowledge Graph Embedding bt Relational Rotation in Complex Space (ICLR 2019) [[Paper]](https://openreview.net/pdf?id=HkgEQnRqYQ)[[Pytorch]](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
  - Define relationship by using rotation in vector space

- Conditional Network Embeddings (ICLR 2019) [[Paper]](https://openreview.net/pdf?id=ryepUj0qtX)[[Matlab]](https://bitbucket.org/ghentdatascience/cne)
  - Add additional information with respect to given structural properties

<a name="icml" />

#### ICML

- Gromov-Wasserstein Learning for Graph Matching and Node Embedding (ICML 2019) [[Paper]](https://arxiv.org/pdf/1901.06003.pdf)[[Pytorch]](https://github.com/HongtengXu/gwl)
  - Propose novel framework btw. relation graph and embedding space

- Hyperbolic Disk Embeddings for Directed Acyclic Graphs (ICML 2019) [[Paper]](https://arxiv.org/pdf/1902.04335.pdf)[[Code]](https://github.com/lapras-inc/disk-embedding)
  - Propose embedding framework into quasi-metric space
  

<a name="icip" />

#### ICIP

- Quadruplet Selection Methods for Deep Embedding Learning (ICIP 2019) [[Paper]](https://arxiv.org/pdf/1907.09245.pdf)



---
<a name="study" />

### 7️⃣ Study materials

<a name="tutorial" />

#### Tutorial

- Metric learning tutorial (ICML 2010) [[Video]](https://www.youtube.com/watch?v=06BscIm7TwY)

- Metric Learning and Manifolds: Preserving the Intrinsic Geometry (MS research 2016) [[VIdeo]](https://www.youtube.com/watch?v=kTJoFLcdtn8)

- Visual Search (Image Retrieval) and Metric Learning (CVPR 2018) [[Video]](https://www.youtube.com/watch?v=iW_9fvw6YtI)


<a name="lecture" />

#### Lecture

- Topology and Manifold (International Winter School on Gravity and Light 2015) [[Video]](https://www.youtube.com/watch?v=7G4SqIboeig&list=PLFeEvEPtX_0S6vxxiiNPrJbLu9aK1UVC_)

- Metric learning lecture (Waterloo University) [[Video]](https://www.youtube.com/watch?v=GhsHPY3-1zY)

- Understanding of Mahalanobis distance [[Video]](https://www.youtube.com/watch?v=3IdvoI8O9hU)

- Metric Learning by Caltech (2018) [[Video]](https://www.youtube.com/watch?v=M0EjrFQH49o)


<a name="repo" />

#### Repository

- Person re-identification in Pytorch [[Site]](https://github.com/KaiyangZhou/deep-person-reid)


---
### Milestone

- [x] Add Pairwise cost methods

- [x] Add Distribution or other variant methods

- [x] Add Probabilistic methods

- [x] Add Boost-like methods

- [x] Add applications

- [x] Add study materials

- [ ] Add brief descriptions