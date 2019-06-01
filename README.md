# SSD-Keras [![Build Status](https://travis-ci.org/schoettner/ssd-keras.svg?branch=master)](https://travis-ci.org/schoettner/ssd-keras)
Implementation of [SSD](https://arxiv.org/abs/1512.02325) (Single Shot MultiBox Detector) with [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) backend.
The implementation is also heavily motivated by [YOLOv3](https://arxiv.org/abs/1804.02767).

To make use of python's @dataclass, make sure to use python >= 3.7!

### Pre-Processor ###
The pre processor is responsible to read image files and the labels. They are then converted to *(x, y_true)* with the
help of an encoder.

### Training ###
For the moment, the implementation supports only a SSD300. 


### Image format ###
I decided to use a PIL based image loading due to the fact that the *(width, height, channel)* shape appears more natural to me
and hence helps me to keep up with the dimensions of the network. And with [Pillow-SIMD](https://github.com/uploadcare/pillow-simd#pillow-simd)
the performance gap to OpenCV is not that big anymore.

### Todo ###
- [x] Implement SSD300
- [x] Implement first version of loss function
- [x] Implement random data generator
- [ ] add scale of label depending on the image (see tf_dataset vs batch_loader)
- [ ] Implement pre-processor
- [ ] Implement post-processor
- [ ] Implement prediction
- [ ] Implement full version of loss function
- [ ] Verify that the model trains and predicts correct without priori box
- [ ] Add documentation


### References ###
[SSD Paper](https://arxiv.org/abs/1512.02325)  
[SSD-Keras example](https://github.com/pierluigiferrari/ssd_keras)  
[SSD-Keras example fork](https://github.com/lvaleriu/ssd_keras-1)  
[SSD with tensorflow blog](https://lambdalabs.com/blog/how-to-implement-ssd-object-detection-in-tensorflow/)  
[Netscope to visualize caffe model](https://dgschwend.github.io/netscope/#/editor)  
[PIL vs OpenCV](https://www.kaggle.com/vfdev5/pil-vs-opencv)
[Pillow Performance](https://python-pillow.org/pillow-perf/)   
[Tensorflow dataset](https://cs230-stanford.github.io/tensorflow-input-data.html)