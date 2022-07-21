
import torch 
import torch.nn as nn 

class ActivationHookWrapper():
    tensors_class = {}  # Tree structure for activations storage
    tensors_named = {}
    @staticmethod
    def forward_hook_class(module, input, output):
        name = module.__class__.__name__
        ActivationHookWrapper.tensors_class[name].append((input[0], output[0]))
    
    @staticmethod
    def forward_hook_named(module, input, output):
        name = module.__class__.__name__
        ActivationHookWrapper.tensors_named[name].append((input[0], output[0]))
    
    def reset_static_variables(self):
        ActivationHookWrapper.tensors = {}

    def __init__(self, model, named_list=[], class_names=[]):
        self.model = model 
        for name, module in self.model.named_children():
            if name in named_list:
                module.register_forward_hook(ActivationHookWrapper.forward_hook_named)      
                ActivationHookWrapper.tensors_named[module.__class__.__name__] = []

        for module in self.model.modules():
            name = module.__class__.__name__
            if name in class_names:
                module.register_forward_hook(ActivationHookWrapper.forward_hook_class)      
                ActivationHookWrapper.tensors_class[name] = []



    def forward_with_register(self, single_image):
        assert single_image.size() == (3, 224,224)
        x = single_image.unsqueeze(0)
        self.reset_static_variables()
        x = self.model(x)		
        return x

    def explain(self, input):
        self.forward_with_register(input)
        return ActivationHookWrapper.tensors


    def register_forward(self, module:nn.Module, function):
        hook_handler = module.register_forward_hook(function)
        self.handler = hook_handler
        

if __name__ == "__main__":
    import torchvision 
    import torchvision.models as models

    vgg16 = models.vgg16(pretrained=True)
    vgg_wrapper = ActivationHookWrapper(vgg16)
    result = vgg_wrapper.explain(torch.rand(3, 224,224))
    print(result.keys())

        