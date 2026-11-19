"""Render the paper analyses and copy the static site assets into site/."""
from html import escape
from pathlib import Path
from shutil import copy2

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
PAPERS = {
    "adam": "Adam: A Method for Stochastic Optimization",
    "resnet": "Deep Residual Learning for Image Recognition",
    "cheap": "Why does deep and cheap learning work so well?",
    "attention": "Attention Is All You Need",
}


def math_text(content, context):
    # Restore delimiters after Markdown parsing so it cannot consume TeX escapes.
    return "$" + escape(content) + "$"


def render_article(source, title, key):
    parser = (
        MarkdownIt("commonmark", {"html": False})
        .enable(["table", "strikethrough"])
        .use(dollarmath_plugin, renderer=math_text, allow_labels=False)
    )

    def render_display(tokens, index, options, env):
        return '<div class="math-display">$$' + escape(tokens[index].content) + '$$</div>\n'

    parser.renderer.rules["math_block"] = render_display
    article = parser.render(source)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} | Optimization Visualization</title>
<style>
  body {{ background: #10121b; color: #e9ebf4; font: 1rem/1.65 system-ui, sans-serif; }}
  main {{ max-width: 76ch; margin: 2rem auto; padding: 0 1rem; overflow-wrap: anywhere; }}
  a {{ color: #9dbdff; }}
  pre, .math-display {{ overflow-x: auto; padding: .5rem; }}
  pre {{ background: #1d2233; border-radius: .5rem; }}
  table {{ display: block; overflow-x: auto; border-collapse: collapse; }}
  th, td {{ padding: .4rem .75rem; border: 1px solid #525c74; }}
  img {{ max-width: 100%; }}
  mjx-container {{ max-width: 100%; overflow-x: auto; overflow-y: hidden; vertical-align: middle; }}
  article > p, article li {{ overflow-x: auto; }}
</style>
<script>
window.MathJax = {{tex: {{inlineMath: [['$', '$']], tags: 'none'}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
</head>
<body><main>
<nav><a href="../">All visualizations</a> · <a href="{key}.md">Markdown source</a></nav>
<article>
{article}
</article>
</main></body>
</html>
"""


def main():
    SITE.mkdir(exist_ok=True)
    copy2(ROOT / "index.html", SITE)
    (SITE / ".nojekyll").touch()
    for key, title in PAPERS.items():
        dest = SITE / key
        dest.mkdir(exist_ok=True)
        for path in (ROOT / key).iterdir():
            if path.is_file() and path.suffix in {".md", ".html", ".js", ".py", ".png", ".mp4"}:
                copy2(path, dest)
        source = (ROOT / key / f"{key}.md").read_text(encoding="utf-8")
        (dest / f"{key}.html").write_text(render_article(source, title, key), encoding="utf-8")


if __name__ == "__main__":
    main()
