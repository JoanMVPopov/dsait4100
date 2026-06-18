# WORKFLOW (running the model query pipeline)

1) install ollama. make sure the gpu is detectable
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
> to make sure the GPU is detected, you might have to run other commands
> (e.g. apt-get update | apt-get install -y pciutils lshw)

> useful FAQ
https://docs.ollama.com/faq

3) models
```bash
ollama pull qwen3.5:9b-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull deepseek-r1:8b-0528-qwen3-q4_K_M
```

4) env vars
```bash
export OLLAMA_MODELS=/folder_for_models/ollama_models
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_MAX_QUEUE=1024
```

although for linux, you should use systemctl edit ollama.service
as per the docs (check the faq). Using the above exports, you have to do them each time

5) start ollama as a background service that does not get interrupted if you close the console
nohup ollama serve > /workspace/ollama.log 2>&1 &
---

## Environment

Managed by conda

```bash
conda env create -f environment.yml
conda activate nlp-dsait4100
```

## Processing

If you want to change the preprocessing and filtering steps for the different datasets, you may do so in the `rq1/preprocess_hatexplain.py` and `rq3/dataset_processing.py` and then run 
```bash
python <filename> 
```
to apply it and save it in the respective folders

## Running Experiments

To run the hatespeech classification experiments with the exact same setup as the paper (you may change the output path and queries in the file), run
```bash
python workflow.py
```
With the current setup, this may take around 10 hours to run due to the larger size of the sets. 

## Notes

If sufficient resources are available, using the non-quantized models could provide different results than the ones presented in the paper. 



