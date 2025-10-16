# Session 3 - Practice

Today's goal is to add to the previous Kubeflow pipeline 3 steps:
1. Fine-tuning Phi-3 with LoRA
2. Computing predictions on the test set using the fine-tuned model
3. Evaluating the predictions using ragas

Let's go step by step! The suggested structure of your repository is, at the end of the session:

```
├── README.md
├── pyproject.toml
├── .venv/  # created by uv
├── .gitignore
├── .env.example
├── .env  # not committed to git
├── src/
│   ├── constants.py  # constants, loads from environment variables
│   ├── pipeline_components/
│   │   ├── fine_tuning_component.py  # the fine-tuning component
│   │   ├── inference_component.py  # the prediction component
│   │   ├── evaluation_component.py  # the evaluation component
│   │   └── data_transformation_component.py  # the Kubeflow Pipeline component
│   ├── pipelines/
│   │   └── model_training_pipeline.py  # the Kubeflow Pipeline
├── scripts/
│   ├── validate_gcp_setup.py  # from session 1
│   └── pipeline_runner.py  # compiles and submits the pipeline job
```

As before, we'll be adapting code from this [Hugging Face article](https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face)!

> ⚠ Warning - to benefit from pre-built serving Docker images on Vertex AI, we need to downgrade the version of `peft` to `0.13.2`, and `transformers` to `4.46.*` (different from the versions of the article)!


# 1. Fine-tuning component

The goal of this section is to add a fine-tuning step to the pipeline, using LoRA to reduce the number of trainable parameters. This step should take as input the training dataset produced by the data transformation step, and output the fine-tuned model and training metrics.

## Make it work locally

1. Add the additional dependencies to your project by using the `uv add` command.
> Note: `BitsAndBytes` can't be installed on MacOS, so you won't be able to test the fine-tuning with the quantization component locally. 
2. Test locally in a notebook that you can create the different configurations needed for the fine tuning:
    - `LoraConfig`
    - `BitsAndBytesConfig` (if not on MacOS)
    - `SFTConfig`
        -  To visualize training metrics, we ask the trainer to output them for [TensorBoard](https://www.tensorflow.org/tensorboard), a common training visualization tool. 
        - To do this, set the `report_to` parameter of the `SFTConfig` to `["tensorboard"]`, and pass a `logging_dir` to decide where the log files will be stored.
        - You may need to add `tensorboard` to your dependencies.
3. Isolate key hyper-parameters in a dictionary - you'll want to log those values with your metrics or as parameters of your component!
4. In the same notebook, test that you can load the `microsoft/Phi-3-mini-4k-instruct`, and that you can prepare the model for training (`prepare_model_for_kbit_training` and `get_peft_model`).
    - Like in the article, check that you are indeed training only a subset of parameters!
    - You should also take a look at the LoRA adapters that were added to the model - do they make sense given your LoRA configuration?
5. In the same notebook, test that you can load the training dataset from GCS, and convert to a Hugging Face `datasets.Dataset` object (similar to what's expected in the article).
    - Make sure to take a look at your dataset to see if it's correctly formatted!
6. To have validation metrics during training, you can split the training dataset into a training and a validation set using `datasets.Dataset.train_test_split`.
    - The evaluation set can be passed to the `SFTTrainer` using the `eval_dataset` parameter.
7. Test locally that you can launch the training process.

At the end of this section, you should be able to load the pre-trained model, and start a fine-tuning locally!

## Adding it to the pipeline

1. When all seems to work locally, create a new file `src/pipeline_components/fine_tuning_component.py`, and a `fine_tuning_component` function to wrap your code.
    - This function should have an `Input[Dataset]`  parameter for the training dataset, and an `Output[Model]` and `Output[Metrics]` parameters for the fine-tuned model and the training metrics.
    - For the base image of the component, we need an image that comes with `torch` and necessary GPU drivers installed (CUDA on most of GCP's virtual machines). We can use the `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` image in our case!
    - For the dependencies of the component, don't forget to add everything you need to import in the function!
    - Don't forget to save your model at `model.path`, and log relevant metrics to Kubeflow!
        - In particular, you should adapt the `logging_dir` of TensorBoard to write logs to the output metrics path of the component by setting it to `metrics.path`, for example.
2. Adapt your pipeline code in `src/pipelines/model_training_pipeline.py` to add the fine-tuning step after the data transformation step.
    - Don't forget to pass the output of the data transformation step as input to the fine-tuning step!
    - To make sure that a GPU is used for the fine-tuning step, you can pass additional configuraion to your task:
    ```python
    fine_tuning_task = fine_tuning_component(...)
    fine_tuning_task.set_accelerator_type("NVIDIA_TESLA_T4")
    fine_tuning_task.set_cpu_limit("16")
    fine_tuning_task.set_memory_limit("50G")
    ```
3. Launch the pipeline using your existing pipeline runner script!
    - Make sure to use a region where you requested a GPU quota increase in session 1!

At the end of this step, you should have a successful pipeline run with a fine-tuning step in the Vertex Pipelines console!

## Checking the results

Once the pipeline has run successfully, we should check that it has produced the expected results.
1. Follow the artifact URI in the Vertex Pipelines console to check that the fine-tuned model is in your GCS bucket.
2. Download the model locally, and test that you can load it an generate a quick prediction! Does it do a good Yoda impression?
3. Take a look at the training metrics in TensorBoard locally.
    - Download the `events.out.tfevents.*` file from the `metrics.path` of the fine-tuning step to a local `logs/` directory.
    - Launch TensorBoard, and specify where you've stored the logs with: `tensorboard --logdir logs/`.
    - Do you see the training and validation loss decreasing over time? Are your hyper-parameters adapted?
4. Check that the metrics you've logged to Kubeflow are visible in the Vertex Pipelines console.


# 2. Inference component

The goal of this section is to add an inference step to the pipeline, using the fine-tuned model to compute predictions on the test set. This step should take as input the fine-tuned model and the test dataset produced by the data transformation step, and output the predictions.

For computing metrics with `ragas`, the following predictions CSV format is advised:
```
user_input,reference,extracted_response
"He knew the skill of the great young actress.","The skill of the great young actress, he knew."," The skill of the great young actress, he knew."
```

## Make it work locally

1. Create a function that can download all model files from a GCS directory URI to a local directory.
    - You can use the `google-cloud-storage` library to do this.
```python
    def download_model(model_uri: str, local_dir: str):
        """Download model from GCS to local directory."""
        # TODO
        pass
```
2. Based on the article code, create a function that given a sentence from the test set, applies the chat template included in the `tokenizer` for inference.
```python
    def build_prompt(tokenizer: AutoTokenizer, sentence: str):
        """Build a prompt from a sentence applying the chat template.
        
        Output string should look like:
            <|user|>
            The Force is strong in you!<|end|>
            <|assistant|>
        """
        # TODO
        pass

```
3. Based on the article code, create a function that, given a model, a tokenizer, a prompt that respects the chat template, and generation parameters, computes a prediction.
```python
    def generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model given a prompt."""
        # TODO
        pass
```
4. Write a function that extracts from the model output the actual response of the model, without the prompt or special tokens. You can use a simple regex!
```python
    def extract_response(model_output: str) -> str:
        """Extract the actual response from the model output.
        
        If input string looks like this:
            <|user|>
            The Force is strong in you!<|end|>
            <|assistant|>
            Strong in you, the Force is. Yes, hrrmmm.<|end|>

        The output should be:
            Strong in you, the Force is. Yes, hrrmmm.
        """
        # TODO
        pass
```
5. Let's tie it all together! In a notebook:
    - Call the `download_model` function to download the fine-tuned model from GCS to a local directory.
    - Load the model and tokenizer using `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`.
    - Load the test dataset from GCS.
    - Loop through the rows of the test dataset, and for the first 3 rows:
        - Build the prompt using the `build_prompt` function.
        - Generate a response using the `generate_response` function.
        - Extract the actual response using the `extract_response` function.
    - Output the predictions in a CSV file with the expected format.

## Adding it to the pipeline
1. When all seems to work locally, create a new file `src/pipeline_components/inference_component.py`, and a `inference_component` function to wrap your code.
    - This function should have an `Input[Model]` parameter for the fine-tuned model, an `Input[Dataset]` parameter for the test dataset, and an `Output[Dataset]` parameter for the predictions.
    - For the base image of the component, we can use the same image as for the fine-tuning step: `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`.
    - For the dependencies of the component, don't forget to add everything you need to import in the function!
    - Don't forget to save your predictions at `predictions.path`.
2. Adapt your pipeline code in `src/pipelines/model_training_pipeline.py` to add the inference step after the fine-tuning step.
    - Don't forget to pass the output of the fine-tuning step as input to the inference step!
    - Also, make sure a GPU is used for the inference step, like for the fine-tuning step!
3. Compile and launch the pipeline using your existing pipeline runner script!

At the end of this step, you should have a successful pipeline run with an inference step in the Vertex Pipelines console!

## Checking the results

Once the pipeline has run successfully, we should check that it has produced the expected results.
1. Follow the artifact URI in the Vertex Pipelines console to check that the predictions CSV file is in your GCS bucket.
2. Download the predictions CSV file locally, and take a look at the predictions. Is it in the right format, and do they look like sensible Yoda answers?

# 3. Evaluation component

The objective of this section is to add an evaluation step to the pipeline, using `ragas` to compute metrics on the predictions. This step should take as input the predictions produced by the inference step, and output the evaluation metrics per sample in a CSV file, as well as aggregated metrics in a Metrics artifact.

## Make it work locally

1. In a notebook, load the predictions CSV file produced by the inference step.
2. Add the `ragas` dependency to your project using `uv add`.
2. Using `ragas`, define in a list the different metrics you want to compute on the predictions.
    - Let's start with non-LLM metrics like `Rouge` (it may need additional packages to work).
3. Write a function that, given a row of the predictions CSV file, creates a `SingleTurnSample` object, and returns a dictionary with a score for each defined metric:
```python
   def compute_metrics(
        user_input: str,
        response: str,
        reference: str,
        metric_definitions: list[SingleTurnMetric],
    ) -> dict[str, float | int]:
        """Compute metrics for a single prediction."""
        # TODO
        pass
```
4. Loop through the rows of the predictions CSV file, and for each row, call the `compute_metrics` function.
    - Store the results in a list of dictionaries, and convert it to a Pandas DataFrame.
5. Write a function that computes aggregated metrics from the per-sample metrics DataFrame we just created.
```python
    def aggregate_metrics(per_sample_metrics: pd.DataFrame) -> dict[str, float]:
        """Aggregate metrics from per-sample metrics DataFrame."""
        # TODO
        pass
```
6. Check your granular and aggregated metrics - do they make sense?

## Adding it to the pipeline
1. When all seems to work locally, create a new file `src/pipeline_components/evaluation_component.py`, and an `evaluation_component` function to wrap your code.
    - This function should have an `Input[Dataset]` parameter for the predictions, an `Output[Dataset]` parameter for the per-sample `evaluation_results`, and an `Output[Metrics]` parameter for the aggregated metrics.
    - `ragas` needs a non-Python dependency under the hood - `git`. The standard Python Docker images don't come with `git` installed, but thankfully the `cicirello/pyaction:3.11` image does! We can use it as the base image of the component.
    - Don't forget to add all the dependencies you need to import in the function!
    - Don't forget to save your per-sample metrics at `evaluation_results.path`!
2. Adapt your pipeline code in `src/pipelines/model_training_pipeline.py` to add the evaluation step after the inference step.
    - Don't forget to pass the output of the inference step as input to the evaluation step!
3. Compile and launch the pipeline using your existing pipeline runner script!

At the end of this step, you should have a successful pipeline run with an evaluation step in the Vertex Pipelines console!

## Checking the results
Once the pipeline has run successfully, we should check that it has produced the expected results.
1. Follow the artifact URI in the Vertex Pipelines console to check that the per-sample evaluation results CSV file is in your GCS bucket.
2. Download the per-sample evaluation results CSV file locally, and take a look at the results. Do they look like sensible scores?
3. Check that the aggregated metrics you've logged to Kubeflow are visible in the Vertex Pipelines console.