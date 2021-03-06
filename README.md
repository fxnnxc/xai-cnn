# xai-cnn

**eXplainable Artificial Intelligence (XAI) for CNN networks**



## Wrapper

A wrapper is a class which contains model and the XAI method to be used. The code below shows an example of explain method to be used for the model and XAI methods. The XAI methods are implemented in a way of model-agnostic. If a method requires model-agnostic explaination, it is implemented with a tag such as `VGG`.

```python
img = ...
model = VGG16()

# Model-agnostic method
wrapper = ActivationHookWrapper(model)
result = wrapper.explain(img)

# Model-specific method
wrapper = InceptionHookWrapper(model)
result = wrapper.explain(img)
```


## Running 

```bash
pip install -e .

bash shells/test.sh
``` 

## File Structure

```bash
π¦xai-cnn
 β£ πxai_cnn
 β β£ πplot
 β β β π utils.py
 β β πwrappers
 β β β£ πlrp_wrapper
 β β β π activation_hook_wrapper.py
 β£ πscripts
 β β π  test.py
 β£ πshells
 β β π¦ͺ test.sh
 β£ βοΈ .gitignore
 β£ πͺ LICENSE
 β£ π README.md
 β π setup.py
``` 


