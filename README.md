# scotopic
Train and test scotopic classifiers on MNIST and CIFAR
(c) 2016 [Bo Chen](http://vision.caltech.edu/~bchen3/_site2/)

## About
This codebase corresponds to the following publication:

[1] Bo Chen and Pietro Perona, Seeing into Darkness: Scotopic Visual Recognition, ArXiv 2016

- Train and test WaldNet, photoic classifier, ensemble classifier, and WaldNet with dynamic light level estimation. 

- All models are tested against different camera noises. 

- Includes plotting code to reproduce figures in [1]. 

## Installation
- This codebase depends on and modifies [MatConvNet](http://www.vlfeat.org/matconvnet/) (beta18). The MatConvNet is included
and must be installed as explained [here](http://www.vlfeat.org/matconvnet/install/). Use different compile options depending on whether your
machine has GPUs. 

- In `Matlab`, go to the ``scotopic`` directory and type
```shell
addpath(genpath(pwd))
```

- Modify `getScotopicConfig.m` to have `data_path` point to your custom data directory

## Training scotopic classifiers and evaluate on MNIST and CIFAR
- Now you can train and test scotopic classifiers. To start, run
```shell
cnn_scotopic_mnist_demo(1)
```
to train a WaldNet on the MNIST dataset. The argument to the function indicates what type of scotopic model you are training: 1: WaldNet; 
2: Rate classifier; 3-6: Specialists at PPP=[.22, 2.2, 22, 220], respectively; 7: Ensemble; and 8: WaldNet with light level estimation. Each model
takes from 20 minutes to a day to train depending on your workstation's config. 

- After you have trained all of them (training in order is recommended), visualize the results by running
```shell
plot_SAT_mnist(FIGURE_FOLDER)
```
where `FIGURE_FOLDER` should be a folder you created in advance to store the result figures. You should be able to regenerate figures in [1] this way.

- The results from CIFAR10 may be obtained par simile. 

## Testing scotopic classifiers under camera noises
After all the models are trained, run
```shell
cnn_scotopic_noise(DATASET)
```
where `DATASET` is either "mnist" or "cifar". The evaluation should take about four hours if your machine support Matlab's parpool environment. 

After that's finished you may visualize the robustness analysis by running
```shell
plot_sensor_noise_effect(FIGURE_FOLDER)
```
where `FIGURE_FOLDER` should be a folder you created in advance to store the result figures. 

## Contact
Questions please email bchen3@caltech.edu








