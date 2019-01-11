# NLP-EHR
Analysing clinical semantic textual similarity on big data. 

Clinical Semantic Textual Similarity (ClinicalSTS) measures the degree of similarity between two sentences. The key challenges of ClinicalSTS are how to extract medical concepts and identify templates. This project has leveraged three deep learning models, they are fully connected neural network, Attention-based CNN (ABCNN) and LSTM. The comparison between the three models has illustrated that even their accuracy is quite identical, each of them has its own strength and weakness. The comparison between embedding methods has shown the influence of embedding methods. After partially removed templates, the performance of ABCNN is improved, it proves the template is a distraction for deep learning models. 

Current three models are transplanted into a distributed system to cope with the big data challenge. This project has brought up two types of mechanism, one is data parallelism, and another is model parallelism. Data parallelism is deployed on Spark, model parallelism is on HPC (Spartan). 

===The dataset is confidential, it is not allowed to share with public===


本项目为分析医疗系统中语义文本相似性，采用了全连接深度学习网络，ABCNN和LSTM。三种模型精度大致分布于80%~85%。由于样本数量级偏小，模型的选择不够明朗，最终采用集成学习（ensemble）的方式，将三种网络的输出整合在一起。


ABCNN.py -> 文件预处理，tokenizer，三种词嵌入，搭建ABCNN regression模型
BiLSTM.py -> 按照Siamese孪生卷积网络搭建双向LSTM模型
FuzzyMatch.py -> 用模糊算法大致处理原文本中的术语模版。
HPC文件中涵盖了在Spartan上放置模型的脚本，用.slurm文件安排不同模型布置在不同节点上，以达到效率最大化。


数据集由BioCreative/OHNLP所提供，不开放给公众。
