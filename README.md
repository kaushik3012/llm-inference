# LLM Inference using Modal

### Requirements
```bash
pip install modal
```

Make sure you have created a [HuggingFace access token](https://huggingface.co/settings/tokens).<br>
To access the token in a Modal function, we can create a secret on the [secrets page](https://modal.com/secrets).<br>
Remember to the secret in the inference.py file.<br>

### Instructions
To run the model on the validation set and get accuracy, enter following command in terminal
<br>
```bash
modal run inference
```
