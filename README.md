# PicoDet-Backbone

PyTorch Implementation of **Backbone** of [PicoDet](https://arxiv.org/abs/2111.00902)

[Original Implementation](https://github.com/PaddlePaddle/PaddleDetection/blob/6e65f11d8ec0169655b5ec7614a6360b25090ae3/ppdet/modeling/backbones/esnet.py) is implemented on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).

![image](https://user-images.githubusercontent.com/35001605/144397123-9ceb1316-7ff2-41f7-9294-2822fa8f96fb.png)

# Example


```python
picodet_l_backbone = ESNet(scale=1.25, 
                     feature_maps=[4, 11, 14], 
                     channel_ratio=[0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 0.625, 0.5, 0.625, 1.0, 0.625, 0.75])
...
c3, c4, c5 = picodet_l_backbone(input) 
```
