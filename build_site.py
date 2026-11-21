"""Render the paper analyses and copy the static site assets into site/."""
import re
from html import escape
from pathlib import Path
from shutil import copy2

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
PAPERS = {
    "adam": "Adam: A Method for Stochastic Optimization",
    "resnet": "Deep Residual Learning for Image Recognition",
    "cheap": "Why does deep and cheap learning work so well?",
    "attention": "Attention Is All You Need",
}
CODE_FORMATTER = HtmlFormatter(nowrap=True, style="github-dark")


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
    env = {}
    tokens = parser.parse(source, env)
    article = render_sections(parser, tokens, env)
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
  .visualization-code {{
    margin: 1.6rem 0;
    background: #182337;
    border: 1px solid #48618c;
    border-inline-start: .25rem solid #9dbdff;
    border-radius: .5rem;
  }}
  .visualization-code summary {{
    padding: .8rem 1rem;
    color: #d7e5ff;
    cursor: pointer;
    font-weight: 700;
    scroll-margin-top: 1rem;
  }}
  .visualization-code summary:hover {{ background: #223250; }}
  .visualization-code summary:focus-visible {{ outline: 2px solid #9dbdff; outline-offset: 2px; }}
  .visualization-code[open] summary {{ border-bottom: 1px solid #48618c; }}
  .visualization-code p {{ margin: .8rem 1rem 0; }}
  .visualization-code .highlight {{ margin: .85rem; border-radius: .5rem; }}
  .visualization-code pre {{ margin: 0; padding: .85rem; tab-size: 4; }}
  {CODE_FORMATTER.get_style_defs(".visualization-code .highlight")}
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


def render_sections(parser, tokens, env):
    def render(items):
        return parser.renderer.render(items, parser.options, env)

    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        title = tokens[index + 1].content
        if not re.fullmatch(r"(?:\d+\. )?Visualization Code", title):
            continue

        for end in range(index + 3, len(tokens)):
            fence = tokens[end]
            if fence.type == "heading_open":
                break
            if fence.type == "fence" and fence.info.strip() == "python":
                code = highlight(fence.content, PythonLexer(), CODE_FORMATTER)
                return (
                    render(tokens[:index])
                    + '<details class="visualization-code" id="visualization-code">\n'
                    + f"<summary>{escape(title)}</summary>\n"
                    + render(tokens[index + 3:end])
                    + f'<pre class="highlight"><code class="language-python">{code}</code></pre>\n'
                    + "</details>\n"
                    + render(tokens[end + 1:])
                )
    return render(tokens)


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
