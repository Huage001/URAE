# URAE: ~~Your Free FLUX Pro Ultra~~

<img src='teaser.jpg' width='100%' />
<br>

<a href="https://arxiv.org/abs/2503.16322"><img src="https://img.shields.io/badge/arXiv-2503.16322-A42C25.svg" alt="arXiv"></a> 
<a href="https://huggingface.co/Huage001/URAE"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/Yuanshi/URAE"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/Yuanshi/URAE_dev"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>

> ***U*ltra-*R*esolution *A*daptation with *E*ase**
> <br>
> [Ruonan Yu*](https://scholar.google.com/citations?user=UHP95egAAAAJ&hl=en), 
> [Songhua Liu*](http://121.37.94.87/), 
> [Zhenxiong Tan](https://scholar.google.com/citations?user=HP9Be6UAAAAJ&hl=en), 
> and 
> [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)
> <br>
> [xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore
> <br>

## 🪶Features

* **Easy-to-Use High-Quality and High-Resolution Generation😊**: Ultra-Resolution Adaptation with Ease, or URAE in short, generates high-resolution images with FLUX, with minimal code modifications.
* **Easy Training🚀**: URAE tames light-weight adapters with a handful of synthetic data from [FLUX1.1 Pro Ultra](https://blackforestlabs.ai/ultra-home/).

## 🔥News

**[2025/03/20]** We release models and codes for both training and inference of URAE.

## Introduction

Text-to-image diffusion models have achieved remarkable progress in recent years. However, training models for high-resolution image generation remains challenging, particularly when training data and computational resources are limited. In this paper, we explore this practical problem from two key perspectives: data and parameter efficiency, and propose a set of key guidelines for ultra-resolution adaptation termed *URAE*. For data efficiency, we theoretically and empirically demonstrate that synthetic data generated by some teacher models can significantly promote training convergence. For parameter efficiency, we find that tuning minor components of the weight matrices outperforms widely-used low-rank adapters when synthetic data are unavailable, offering substantial performance gains while maintaining efficiency. Additionally, for models leveraging guidance distillation, such as FLUX, we show that disabling classifier-free guidance, *i.e.*, setting the guidance scale to 1 during adaptation, is crucial for satisfactory performance. Extensive experiments validate that URAE achieves comparable 2K-generation performance to state-of-the-art closed-source models like FLUX1.1 [Pro] Ultra with only 3K samples and 2K iterations, while setting new benchmarks for 4K-resolution generation.

## Quick Start

* If you have not, install [PyTorch](https://pytorch.org/get-started/locally/), [diffusers](https://huggingface.co/docs/diffusers/index), [transformers](https://huggingface.co/docs/transformers/index), and [peft](https://huggingface.co/docs/peft/index).

* Clone this repo to your project directory:

  ``` bash
  git clone https://github.com/Huage001/URAE.git
  cd URAE
  ```

* **You only need minimal modifications!**

  ```diff
  import torch
  - from diffusers import FluxPipeline
  + from pipeline_flux import FluxPipeline
  + from transformer_flux import FluxTransformer2DModel
  
  bfl_repo = "black-forest-labs/FLUX.1-dev"
  + transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=torch.bfloat16)
  - pipe = FluxPipeline.from_pretrained(bfl_repo, torch_dtype=torch.bfloat16)
  + pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=torch.bfloat16)
  pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
  
  + pipe.load_lora_weights("Huage001/URAE", weight_name="urae_2k_adapter.safetensors")
  
  prompt = "An astronaut riding a green horse"
  image = pipe(
      prompt,
  -    height=1024,
  -    width=1024,
  +    height=2048,
  +    width=2048,
      guidance_scale=3.5,
      num_inference_steps=50,
      max_sequence_length=512,
      generator=torch.Generator("cpu").manual_seed(0)
  ).images[0]
  image.save("flux-urae.png")
  ```
  ⚠️ **FLUX requires at least 28GB of GPU memory to operate at a 2K resolution.** A 48GB GPU is recommended for the full functionalities of URAE, including both 2K and 4K. We are actively integrating model lightweighting strategies into URAE! If you have a good idea, feel free to submit a PR!

* Do not want to run the codes? No worry! Try the model on Huggingface Space!
  
  * [URAE w. FLUX1.schnell](https://huggingface.co/spaces/Yuanshi/URAE) (Faster)
  * [URAE w. FLUX1.dev](https://huggingface.co/spaces/Yuanshi/URAE_dev) (Higher Quality)

## Installation

* Clone this repo to your project directory:

  ``` bash
  git clone https://github.com/Huage001/URAE.git
  cd URAE
  ```
  
* URAE has been tested on ``torch==2.5.1`` and ``diffusers==0.31.0``, but it should also be compatible to similar variants. You can set up an new environment if you wish and install packages listed in ``requirements.txt``:

  ```bash
  conda create -n URAE python=3.12
  conda activate URAE
  pip install -r requirements.txt
  ```

## Inference

#### 2K Resolution, or Use Cases Similar to FLUX Pro Ultra

* We use **LoRA adapters** to adapt FLUX 1.dev to 2K resolution. You can try the corresponding URAE model in `inference_2k.ipynb`.

* We also support FLUX 1.schnell for a **faster inference process**. Please refer to `inference_2k_schnell.ipynb`.

#### 4K Resolution or Higher

* Instead of LoRA, we use **minor-component adapters** at the 4K stage. You can try the models for FLUX 1.dev and FLUX 1.schnell in `inference_4k.ipynb` and `inference_4k_schnell.ipynb`.
* Alternatively, although these adapters are different from LoRA, it can also be converted to the LoRA format, so that we can apply interfaces provided by `peft` for a more convenient usage. **If you only want to try the models instead of understanding the principle,** `inference_4k_lora_conversion.ipynb` **and** `inference_4k_lora_conversion_schnell.ipynb` **are what you want!** Their outputs should be equivalent to the above counterparts.

🚧 The 4K model is still in beta and the performance may not be stable. A more recommended use case for the current 4K model is to integrate it with some training-free high-resolution generation pipelines based on coarse-to-fine strategies, such as [SDEdit](https://arxiv.org/abs/2108.01073) (refer to [this repo](https://github.com/Huage001/CLEAR/blob/main/inference_t2i_highres.ipynb) for a sample usage) and [I-Max](https://github.com/PRIS-CV/I-Max), and load the 4K adapter at high resolution stages.


## Training

1. Prepare training data.

   * To train the 2K model, we collect 3,000 images generated by [FLUX1.1 Pro Ultra](https://blackforestlabs.ai/ultra-home/). You can use your API and follow the instructions on acquiring images.

   * To train the 4K model, we use ~16,000 images with resolution greater than 4K from [LAION-High-Resolution](https://huggingface.co/datasets/laion/laion-high-resolution).

   * The training data folder should be organized to the following format:

     ```
     train_data
     ├── image_0.jpg
     ├── image_0.json
     ├── image_1.jpg
     ├── image_1.json
     ├── ...
     ```

     The json file should contain a dictionary with entries `prompt` and/or `generated_prompt`, to specify the original caption and generated caption with detailed descriptions by VLM such as GPT-4o or Llama3.

   * The T5 and VAE can take a large amount of GPU memory, which can trigger OOM when training at high resolutions. Therefore, we pre-cache the T5 and VAE features instead of computing them online:

     ```bash
     bash cache_prompt_embeds.sh
     bash cache_latent_codes.sh
     ```

     Make sure to modify these bash files beforehand and configure the number of processes in parallel (`$NUM_WORKERS`), training data folder (`$DATA_DIR`), the target resolution (`--resolution`), and the key to the prompt in json files (`--column`, can be `prompt` or `generated_prompt`).

   * The final format of the training data folder should be:

     ```
     train_data
     ├── image_0_generated_prompt_embed.safetensors
     ├── image_0.jpg
     ├── image_0.json
     ├── image_0_latent_code.safetensors
     ├── image_0_prompt_embed.safetensors
     ├── image_1_generated_prompt_embed.safetensors
     ├── image_1.jpg
     ├── image_1.json
     ├── image_1_latent_code.safetensors
     ├── image_1_prompt_embed.safetensors
     ├── ...
     ```

2. Let's start training! For 2K model, make sure to modify `train_2k.sh` and configure the training data folder (`$DATA_DIR`), output folder (`$OUTPUT_DIR`), and the number of GPUs (`--num_processes`).

   ```bash
   bash train_2k.sh
   ```

3. 4K model is based on the 2K LoRA trained previously. Make sure to modify `train_4k.sh` to configure the path (`--pretrained_lora`) in addition to the aforementioned items.

   ```bash
   bash train_4k.sh
   ```

   ⚠️ Currently, if the training images have different resolutions, the `--train_batch_size` can only be 1 because we did not customize data sampler to handle this case. Nevertheless, in practice, with such large resolutions, the maximal batch size per GPU (before gradient accumulation) can only be 1 even on 80G GPUs 😂.

## Acknowledgement

* [FLUX](https://blackforestlabs.ai/announcing-black-forest-labs/) for the source models.
* [diffusers](https://github.com/huggingface/diffusers), [CLEAR](https://github.com/Huage001/CLEAR), and [I-Max](https://github.com/PRIS-CV/I-Max) for the code base.
* [patch_conv](https://github.com/mit-han-lab/patch_conv) for solution to VAE OOM error at high resolutions.
* [@Xinyin Ma](https://github.com/horseee) for valuable discussions.
* NUS IT’s Research Computing group using grant numbers NUSREC-HPC-00001.

## Citation

If you finds this repo is helpful, please consider citing:

```bib
@article{yu2025urae,
  title     = {Ultra-Resolution Adaptation with Ease},
  author    = {Yu, Ruonan and Liu, Songhua and Tan, Zhenxiong and Wang, Xinchao},
  year      = {2025},
  eprint    = {2503.16322},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
