# # Fast inference with vLLM
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
from modal import Image, Secret, Stub, method

MODEL_DIR = "/model"
BASE_MODEL = "satpalsr/llama2-translation-filter-full"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# Make sure you have created a [HuggingFace access token](https://huggingface.co/settings/tokens).
# To access the token in a Modal function, we can create a secret on the [secrets page](https://modal.com/secrets).
# Now the token will be available via the environment variable named `HF_TOKEN`. Functions that inject this secret will have access to the environment variable.
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# Tip: avoid using global variables in this function. Changes to code outside this function will not be detected and the download step will not re-run.
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


# ### Image definition
# We’ll start from a recommended Dockerhub image and install `vLLM`.
# Then we’ll use run_function to run the function defined above to ensure the weights of
# the model are saved within the container image.
image = (
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
    )
    .pip_install("vllm==0.2.5", "huggingface_hub==0.19.4", "hf-transfer==0.1.4", "datasets")
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("my-huggingface-secret"),
        timeout=60 * 20,
    )
)

stub = Stub("example-vllm-inference", image=image)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean.
@stub.cls(gpu="A100", secret=Secret.from_name("my-huggingface-secret"))
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)
        self.template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
"""

    @method()
    def generate(self, questions):
        from vllm import SamplingParams

        prompts = [
            self.template.format(system=question["conversations"][0]["value"], user=question["conversations"][1]["value"]) for question in questions
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1,
            max_tokens=180,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        
        pred_outputs = []
        for output in result:
            output_text = output.outputs[0].text.strip()
            if output_text.find('False') != -1:
                pred_outputs.append(False)
            elif output_text.find('True') != -1:
                pred_outputs.append(True)
            else:
                print("Error: Output not recognized\n", output_text)

        true_outputs = []
        for question in questions:
            output_text = question["conversations"][2]["value"].strip()
            if output_text == '{"translate": False}':
                true_outputs.append(False)
            elif output_text == '{"translate": True}':
                true_outputs.append(True)

        if len(pred_outputs) != len(true_outputs):
            print("Error: Length of predictions and true outputs do not match")
        else:
            accuracy_score = sum([1 if pred_outputs[i] == true_outputs[i] else 0 for i in range(len(pred_outputs))])/len(pred_outputs)
            print("Accuracy Score: ", accuracy_score)   


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@stub.local_entrypoint()
def main():
    from datasets import load_dataset

    dataset = load_dataset("satpalsr/chatml-translation-filter")
    questions = [dataset["validation"][i] for i in range(len(dataset["validation"]))]

    model = Model()
    model.generate.remote(questions)