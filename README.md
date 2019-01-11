# NLP-EHR
Analysing clinical semantic textual similarity on big data. 

Clinical Semantic Textual Similarity (ClinicalSTS) measures the degree of similarity between two sentences. The key challenges of ClinicalSTS are how to extract medical concepts and identify templates. This project has leveraged three deep learning models, they are fully connected neural network, Attention-based CNN (ABCNN) and LSTM. The comparison between the three models has illustrated that even their accuracy is quite identical, each of them has its own strength and weakness. The comparison between embedding methods has shown the influence of embedding methods. After partially removed templates, the performance of ABCNN is improved, it proves the template is a distraction for deep learning models. 

Current three models are transplanted into a distributed system to cope with the big data challenge. This project has brought up two types of mechanism, one is data parallelism, and another is model parallelism. Data parallelism is deployed on Spark, model parallelism is on HPC (Spartan). 

===The dataset is confidential, it is not allowed to share with public===
