from xai_cnn.wrappers import ActivationHookWrapper
import torch 
import torchvision 
import torchvision.models as models

# TODO : define your named modules and class modules 
googlenet_xai_modules ={
    "named_list" : ['inception5b'],
    "class_names" : ['Inception', 'Conv2d', 'ReLU']
}
vgg16_xai_moduels ={
    "named_list" : ['features'],
    "class_names" : ['Conv2d', 'ReLU', 'Linear']
}


googlenet = models.googlenet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
inception = models.inception_v3(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)


vgg16_wrapper     = ActivationHookWrapper(vgg16, **vgg16_xai_moduels)
googlenet_wrapper = ActivationHookWrapper(googlenet, **googlenet_xai_modules)

save_dir = "results"

# TODO : load ImageSet
for image in range(1):
    image = torch.rand(3,224,224)
    for wrapper in [googlenet_wrapper]:
        output = wrapper.forward_with_register(image)
        print("----")
        for k in wrapper.tensors_named:
            for v in wrapper.tensors_named[k]:
                # TODO implmente how to handle it
                print(k, v[1].size()) # output 
                # save 
        print("----")
        for k in wrapper.tensors_class:
            for v in wrapper.tensors_class[k]:
                # TODO implmente how to handle it
                print(k, v[1].size())   # output 
                # save 

