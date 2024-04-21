# Mini LLM Challenge(Team-13)


## Result

| Model                     | Perplexity | Accuracy (%)   | Comments                       |
|---------------------------|------------|----------------|--------------------------------|
| **Our Best model (166M)** | **24.2**   | **43.3**       | Achieved at 173min(26.2K iters) |
| GPT-baseline (208M)       | 26.4       | 41.7           | Given by the organizers        |
| llama-baseline (208M)     | 27.1       | 41.2           | Given by the organizers        |



## Reproduce 

Install dependencies: 

```
pip install -r requirements.txt
python scripts/best_run.sh
```

The above command trains a 166M parameters model with the following parameters:
- n_layer **18**
- n_embd 768 
- n_head 12

It trains for 23.2k iterations with a batch size of 66 (No gradient accumulation steps), using default hyperparameters with dropout **enabled**. 
The model is trained on the `slimpajama` dataset. 
The training takes roughly 3 hours on a single A100 GPU.

You can check out the wandb run for yourself [here](https://api.wandb.ai/links/lauzhack-llm/udvei49j).
You find the [model checkpoint](https://drive.google.com/file/d/1Byj1gQRN-Lf2XqFmvCQMiNA_4aTjmDkO/view?usp=sharing) and the [model config](assets/best_run_summary_config.json). 

## Our Approach

We worked on 3 directions to improve the training efficiency given the allocated 3 hours of training: hyperparameter search and model scaling, model architecture, and data selection. 

We tried various tricks but most of them didn't result in a significant improvement in the final performance.
Some of them looked promising at the beginning but didn't converge to a better model in the end.

The tricks that helped us to improve the performance are:
- Enable dropout
- Using scaling law to find the optimal model size
- Using a smaller batch size for more iterations

### Scaling Law with fixed compute

The setting of the challenge is to train a model with a fixed 3h compute budget.
The FLOPS of A100 GPU is 3.12 * 10^14 flops/s, so the total FLOPS is 3.12 * 10^14 * 3 * 3600 ~ 3.4 * 10^18 flops.

However, we may not be able to fully utilize the FLOPS

We interpolate the [scaling law from Deep Mind](https://arxiv.org/pdf/2203.15556.pdf) and the optimal config would be:
- 170M parameters
- 3.4B tokens

Considering we may not fully utilize the FLOPS, the optimal model size may be a bit smaller.

However, as the scaling law is generally for LLM with parameters > 1B, we decide to validate it empirically by training model with different sizes.

The plots are [here](https://api.wandb.ai/links/lauzhack-llm/df512dsi)

Takeaways:
- The smaller the model, the faster it converges but the final performance is not guaranteed to be better.
- 12Layer and 18Layer models have similar performance, but 24Layer model(given by the organizers) has a 2 ppl gap.
- Model smaller than 8 Layer has significantly worse performance.
  
### Larger batch size

There is a trade-off between batch size and the number of iterations.

Traditionally, people use large batch size to train LLM such as 256 or 512. 
However, in our settings, smaller batch size is better as it allows more iterations.

This is why we removed **gradient accumulation** and used a batch size of 66.

### Dropout

Previous work has shown that dropout is not necessary for LLM training.
However, we found that dropout is beneficial in our setting, boosting the performance by 1.5 ppl.

Takeaways:
- Dropout is beneficial in our setting.

### Model architecture

After finding the optimal model size, we want to know if using model with **wider layers** would help.
Given the`N = 12*D^2*L`, we increase the width of the model and reduce the depth to keep the number of parameters the same.

The plots are [here](https://wandb.ai/lauzhack-llm/width_depth_3h/reports/width-VS-depth--Vmlldzo3NjM1NTkz)

Takeaways:
- Wider models converge faster but the final performance is slightly worse than the deeper models.

### Large Learning Rate

As we have limited time, we want to know if using a large learning rate would accelerate the training process.

We did not investigate this further as the default learning rate is already large, i.e. 0.001

### Brainformer architecture: 
The [brainformer block](https://arxiv.org/pdf/2306.00008.pdf) is composed of self-attention + MoE layer + MLP layer + MoE layer + MLP layer + MoE layer or little variations depending on the brainformer version - Brainformer paper

### Architecture modification from BabyLM challenge winners:
We applied BabyLM challenge winners architecture modification on LlaMA 2 architecture: allowing all layers within the architecture to have a weighted residual connection to all previous layers 
[Not all layers are equally as important](https://arxiv.org/pdf/2311.02265.pdf)

### LR scheduling:

- Composing lr-schedulers (e.g. warm restarts with cosine schedule on plateau)
- Schedule-free optimizers (https://github.com/facebookresearch/schedule_free)

### Data selection:
Data selection using (papers considered Dataset Cartography, SemDeDup, RETSIM) all approaches required steps that turned out to be too expensive for us within this 2 days timeframe



### Other directions we considered:
- MLP-Mixer inspired models like HyperMixer (HyperMixer paper) but required generalization to the autoregressive setting (which with our naive extension would require the same computational complexity as the transformer's self-attention).
- 8-bit optimizers (https://huggingface.co/docs/bitsandbytes/main/en/optimizers)
