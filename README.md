# Optimization Visualization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Interactive visualizations and paper analyses of optimization algorithms and deep learning concepts, built with Python (Matplotlib, Manim) and HTML/Three.js.

## Paper Analyses

| Paper | Topic | HTML | Manim Code | Manim Video |
|---|---|---|---|---|
| **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2015) | Adaptive moment estimation optimization | [adam.md](./adam/adam.md) | [adam_manim.py](./adam/adam_manim.py) | [MP4](./adam/adam_manim.mp4) |
| **Deep Residual Learning for Image Recognition** (He et al., CVPR 2016) | Residual networks (ResNet) | [resnet.md](./resnet/resnet.md) | [resnet_manim.py](./resnet/resnet_manim.py) | [MP4](./resnet/resnet_manim.mp4) |
| **Why does deep and cheap learning work so well?** (Lin, Tegmark & Rolnick, 2017) | Physics of deep learning | [cheap.md](./cheap/cheap.md) | [cheap_manim.py](./cheap/cheap_manim.py) | [MP4](./cheap/cheap_manim.mp4) |
| **Attention Is All You Need** (Vaswani et al., NIPS 2017) | Transformer architecture | [attention.md](./attention/attention.md) | [attention_tofix_01.py](./attention/attention_manim_gpt.py) | [MP4](./attention/attention_tofix_01.mp4) |

## Interactive Demos

- [Adam Slow Motion Comparison](./adam/adam_slow.html) - Step-by-step Adam optimization visualization
- [Adam Integration](./adam/adam_integrate.html) - Interactive Adam parameter integration

## Features

- Full paper analyses with mathematical derivations (TeX/MathJax)
- Python code with highlighting and line numbers
- Manim animation videos for key concepts
- Interactive HTML demos

## Deployment

This repository automatically deploys to GitHub Pages via GitHub Actions on every push to `main`.

## License

[MIT](LICENSE)
