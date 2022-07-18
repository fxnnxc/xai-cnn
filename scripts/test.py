from xai_cnn.wrappers import ActivationHookWrapper
import torch 
import torchvision 
import torchvision.models as models

vgg16 = models.vgg16(pretrained=True)
vgg_wrapper = ActivationHookWrapper(vgg16)
result = vgg_wrapper.explain(torch.rand(3, 224,224))
print(result.keys())

