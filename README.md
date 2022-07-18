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
📦xai-cnn
 ┣ 📂xai_cnn
 ┃ ┣ 📂plot
 ┃ ┃ ┗ 🐍 utils.py
 ┃ ┗ 📂wrappers
 ┃ ┃ ┣ 📂lrp_wrapper
 ┃ ┃ ┗ 🐍 activation_hook_wrapper.py
 ┣ 📂scripts
 ┃ ┗ 🐍  test.py
 ┣ 📂shells
 ┃ ┗ 🦪 test.sh
 ┣ ❄️ .gitignore
 ┣ 🪙 LICENSE
 ┣ 📖 README.md
 ┗ 🐍 setup.py
``` 


