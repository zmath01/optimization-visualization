# Optimization Visualization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Interactive visualizations and paper analyses of optimization algorithms and deep learning concepts, built with Python (Matplotlib, Manim) and HTML/Three.js.

## Paper Analyses

| Paper | Topic | Source | Manim Code | Manim Video |
|---|---|---|---|---|
| **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2015) | Adaptive moment estimation optimization | [adam.md](./adam/adam.md) | [adam_manim.py](./adam/adam_manim.py) | [MP4](./adam/adam_manim.mp4) |
| **Deep Residual Learning for Image Recognition** (He et al., CVPR 2016) | Residual networks (ResNet) | [resnet.md](./resnet/resnet.md) | [resnet_manim.py](./resnet/resnet_manim.py) | [MP4](./resnet/resnet_manim.mp4) |
| **Why does deep and cheap learning work so well?** (Lin, Tegmark & Rolnick, 2017) | Physics of deep learning | [cheap.md](./cheap/cheap.md) | [cheap_manim.py](./cheap/cheap_manim.py) | [MP4](./cheap/cheap_manim.mp4) |
| **Attention Is All You Need** (Vaswani et al., NIPS 2017) | Transformer architecture | [attention.md](./attention/attention.md) | [attention_tofix_01.py](./attention/attention_tofix_01.py) | [MP4](./attention/attention_tofix_01.mp4) |

## Interactive Demos

- [Adam Slow Motion Comparison](./adam/adam_slow.html) - Step-by-step Adam optimization visualization
- [Adam Integration](./adam/adam_integrate.html) - Interactive Adam parameter integration

## Render the Manim videos

With FFmpeg, LaTeX and dvisvgm installed, install Manim Community 0.19.0 and run the scenes at 1080p and 30 fps:

```sh
python -m pip install manim==0.19.0
python -m manim -qh --fps 30 adam/adam_manim.py AdamOptimizerScene
python -m manim -qh --fps 30 resnet/resnet_manim.py ResNetVisualization
python -m manim -qh --fps 30 cheap/cheap_manim.py CheapLearningScene
python -m manim -qh --fps 30 attention/attention_tofix_01.py TransformerAttentionScene
```

Encode each rendered MP4 at its linked path with `ffmpeg -i rendered.mp4 -an -c:v libx264 -preset slow -tune animation -crf 19 -pix_fmt yuv420p -movflags +faststart published.mp4`.

## Features

- Full paper analyses with mathematical derivations (TeX/MathJax)
- Collapsible Python code with syntax highlighting in each article
- Manim animation videos for key concepts
- Interactive HTML demos

## Deployment

This repository automatically deploys to GitHub Pages via GitHub Actions on every push to `main`. The build converts the four Markdown analyses to HTML and preserves their TeX for MathJax.

To preview the same artifact locally:

```sh
python -m pip install -r requirements-site.txt
python build_site.py
python -m http.server --directory site 8000
```

Then open http://localhost:8000/. The Adam demos use Three.js from a CDN and the paper pages use MathJax from a CDN.

## License

[MIT](LICENSE)
