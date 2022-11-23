# HuBMAP-HPA-Hacking-the-Human-Body
*** Bronze Medal Awarded, Top 6% (69/1175) ***

References:
1. HubMap Inference TF TPU EfficientNet B7 640x640 https://www.kaggle.com/code/markwijkhuizen/hubmap-inference-tf-tpu-efficientnet-b7-640x640
2. torch official clip+timm embeddings https://www.kaggle.com/code/hmendonca/torch-official-clip-timm-embeddings
3. coat-parallel-small https://www.kaggle.com/competitions/hubmap-organ-segmentation/discussion/332941
Coat model: Staining tool: refer to https://www.kaggle.com/code/gray98/stain-normalization-color-transfer for installation Use {hubmap2022-stainnorm1-for-coat.ipynb} and {hubmap2022-stainnorm2-for-coat.ipynb} to perform two kinds of staining respectively, as the candidate set data for dynamic training (the staining template is the picture in the stain_pics folder)

The coat/dataset.py includes above dyeing image and original data, respectively.
1. {run_train_fold0_dynamic.py} trains the coat-small fold0 model (modify the corresponding fold parameters for other folds)
2. {run_train_fold0_dynamic_plus.py} trains the coat-small-plus fold0 model (modify the corresponding fold parameters for other folds)

Inference:
1. coat_small_infer.py
2. coat_small_plus_infer.py

TF model:
1. {hubmap-patched-tfrecord-generation-v1-negative.ipynb} generates TFRecord format data
2. {hubmap-training-tf-tpu-v1-negative.ipynb} training model
3. {hubmap-inference-v1-negative.ipynb} inference

Model eensembling: ensemble.py

Final submission notebook is {Final_ens4-hubmap-test.ipynb}

