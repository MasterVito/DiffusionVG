# Exploring Iterative Refinement with Diffusion Models for Video Grounding  
Implementation for DiffusionVG: Exploring Iterative Refinement with Diffusion Models for Video Grounding

Once our paper is accepted, we will release the full codes.



## Updates

+ 2023/10/26: A toy example of our DiffusionVG model is updated (DiffusionVG.py).



## Abstract

Video grounding aims to localize the target moment in an untrimmed video corresponding to a given sentence query. Existing methods typically select the best prediction from a set of predefined proposals or directly regress the target span in a one-shot manner, resulting in the absence of prediction evaluation and refinement process. In this paper, we propose DiffusionVG, a novel framework with diffusion models that formulates video grounding as a conditioned generation task, where the target span is generated from Gaussian noise and interatively refined in the denoising diffusion process. During training, DiffusionVG progressively adds noise to the target span with a fixed forward diffusion process and learns to recover the target span in the denoising diffusion process. In inference, DiffusionVG can generate the target span from input Gaussian noise by the learned denoising diffusion process conditioned on the video-sentence representations. Our DiffusionVG follows the encoder-decoder architecture, which firstly encodes the video-sentence features and iteratively denoise the target span in the denoising decoder. Without bells and whistles, our DiffusionVG demonstrates competitive or even superior performance compared to existing well-established models on the mainstream Charades-STA and ActivityNet Captions benchmarks.



The full paper can be found at: [https://arxiv.org/abs/2310.xxxxx](https://arxiv.org/abs/2310.xxxxx)



## Pipeline

![model](https://github.com/MasterVito/DiffusionVG/blob/main/images/model.png)



## Citation

If you find our work helpful to your research, please consider citing our paper using the following format: 

```bibtex
@misc{liang2023diffusionvg,
      title={Exploring Iterative Refinement with Diffusion Models for Video Grounding}, 
      author={Xiao Liang and Tao Shi and Yaoyuan Liang and Te Tao and Shao-Lun Huang},
      year={2023},
      eprint={2310.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
