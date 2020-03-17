# DeepNut: A Finding Nemo Task

### Description

The project is done for the subject 'Computer Vision' Winter semester 2019 at Hochschule Bonn-Rhein-Sieg.

An extensive paper on the project work is provided [here](DeepNut-CV2019.pdf).

### Abstract

Nut detection is an imperative aspect given the substantial health benefits of nuts and the increasing growth of the edible nut industry. To characterize and analyze a nut, the process of detecting and classifying the nuts is an imperative task.  The pivotal aspect of this project is to detect three classes of nuts, namely, peanut, walnut, and hazelnut. The proposed solution is robust to variations in the illumination conditions, viewpoints, frame rates, backgrounds, and the number of distractors in the video input. This particular work aims to deploy a deep learning based object detection methods to detect nuts. In addition, the project utilizes classical methods such as Contrast Limited Adaptive Histogram Equalization (CLAHE) for preprocessing and frame differencing for stable frame extraction from the video input. The nuts are detected within a confined rectangular area of a stable frame extracted from video input. MobileNet-SSD is the deep learning model trained. The model achieves a mean Average Precision (mAP) of 88.12\% on the validation set generated from the CV 2019 dataset with an average inference time of 1.72 seconds. The developed DeepNut detector is evaluated using a localization efficiency metric and mean F-score of all the nut classes for localization and classification tasks, respectively. DeepNut detector localizes the nuts with an efficiency of 83.33\% and classifies the nuts correctly with a mean F-score of 0.921.

### Pipeline

![CV_strategy](https://user-images.githubusercontent.com/43172178/76835379-4777b300-682f-11ea-9c6c-698e873aaff2.jpg)
