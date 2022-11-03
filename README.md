# Cross-corpus stress detection
Implementation of the paper **On the Generalizability of ECG-based Stress Detection Models** - *Prajod, P., & André, E. (2022)*

Stress is prevalent in many aspects of everyday life including work, healthcare, and social interactions.
Handcrafted features from various bio-signals are a popular method of detecting stress.
In the recent years, deep learning models have been gaining traction.
Typically, stress models are trained and validated on the same dataset, often involving one stressful scenario.
But, a lot of factors have to be considered while deploying such stress models in a different scenario - e.g. intensity of stress, brand of sensor, etc.
It is not practical to collect stress data for every scenario.
So, it is crucial to study the generalizability of these models and determine to what extent they can be used in other scenarios.
As part of the paper, we explored the generalization capabilities of Electrocardiogram (ECG)-based deep learning models and 
models based on handcrafted ECG features, i.e.,  Heart Rate Variability (HRV) features.
We implemented :
1. Deep ECGNet - Hwang, Bosun, et al. (2018) "Deep ECGNet: An optimal deep learning framework for monitoring mental stress using ultra short-term ECG signals."
2. ECG Emotion Model - Sarkar, Pritam, and Ali Etemad. (2020) "Self-supervised ECG representation learning for emotion recognition."
3. Multi-Layer Perceptron (Simple feed-forward neural network using HRV features)
4. Random Forest Classifier
5. Support Vector Machine

We use ECG signals from two popular stress datasets differing in terms of stressors and recording devices:
- WESAD - Schmidt, Philip, et al. (2018) "Introducing wesad, a multimodal dataset for wearable stress and affect detection."
- SWELL-KW - Koldijk, Saskia, et al. (2014) "The swell knowledge work dataset for stress and user modeling research."

The implementation also includes the pre-processing steps that we followed for both datasets. 

# Dependencies
- Numpy
- Pandas
- Scipy
- Tensorflow
- Pickle
- Sklearn
- Neurokit2

# License
The code is under the license of [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/)

# Citation
If you use any of the resources in your research, please cite the following publication:

```
@article{prajod2022generalizability,
  title={On the Generalizability of ECG-based Stress Detection Models},
  author={Prajod, Pooja and Andr{\'e}, Elisabeth},
  journal={arXiv preprint arXiv:2210.06225},
  year={2022}
}
```
**Note:** The paper is accepted to ICMLA 2022 conference and will appear in the IEEE Xplore by early 2023. Please update the citation from pre-print to published version (we will update the information here)

