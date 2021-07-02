# LongShortNetworkBipolar

Code implementation of the work described in "Long-Short Ensemble Network for Bipolar Manic-Euthymic State Recognition
Based on Wrist-worn Sensors"

DataProcessing folder contains the methods used to generate the feature sets for all considered modalities (EDA,
Actigraphy, PPG and HR).

DeepLearning folder contains the Short and ShortLong network, and the training loop use to train the networks as well as
the bucket implementation to handle batch of data of different sizes. To train the ensemble networks, simply train 
N networks independently. 