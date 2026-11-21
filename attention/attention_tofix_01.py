import numpy as np
from manim import *


class TransformerAttentionScene(Scene):
    def construct(self):
        self.camera.background_color = "#101722"

        # One query, three keys of width d_k = 4, and three values of width 2.
        q = np.array([1, 1, 0, 0])
        keys = np.array([[1, 1, 0, 0], [1, -1, 0, 0], [-1, -1, 0, 0]])
        values = np.array([[1, 0], [0, 1], [1, 1]])
        scores = keys @ q
        scaled = scores / np.sqrt(keys.shape[1])
        exp_scores = np.exp(scaled - scaled.max())
        weights = exp_scores / exp_scores.sum()
        context = weights @ values

        self.query_key_value(q, keys, values)
        self.scale_and_normalize(scores, scaled, weights, keys.shape[1])
        self.weighted_values(context)
        self.multi_head()

    def heading(self, text):
        title = Text(text, font_size=44).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

    def clear_panel(self):
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.45)

    @staticmethod
    def vector(values, decimals=0):
        entries = " & ".join(f"{value:.{decimals}f}" for value in values)
        return r"\begin{bmatrix}" + entries + r"\end{bmatrix}"

    def query_key_value(self, q, keys, values):
        self.heading("Scaled dot-product attention")
        formula = MathTex(
            r"\operatorname{Attention}(Q,K,V)=\operatorname{softmax}"
            r"\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V",
            font_size=44,
        ).move_to(UP * 1.94)
        groups = VGroup(
            self.labeled_matrix("Query", [q], BLUE_A),
            self.labeled_matrix("Keys", keys, GOLD_A),
            self.labeled_matrix("Values", values, TEAL_A),
        ).arrange(RIGHT, buff=0.95, aligned_edge=UP).move_to(DOWN * 0.72)
        caption = Text("Follow one query as it reads from three key–value pairs.",
                       font_size=26, color=GREY_A).to_edge(DOWN, buff=0.53)
        self.play(Write(formula))
        self.play(LaggedStart(*[FadeIn(group) for group in groups], lag_ratio=0.24))
        self.play(FadeIn(caption))
        self.wait(3)
        self.clear_panel()

    @staticmethod
    def labeled_matrix(label, rows, color):
        expr = r"\\".join(" & ".join(str(entry) for entry in row) for row in rows)
        matrix = MathTex(r"\begin{bmatrix}" + expr + r"\end{bmatrix}", font_size=42, color=color)
        return VGroup(Text(label, font_size=29, color=color), matrix).arrange(DOWN, buff=0.34)

    def scale_and_normalize(self, scores, scaled, weights, width):
        self.heading("Score, scale, then normalize")
        rows = [
            ("Dot products", r"qK^{\mathsf T}=" + self.vector(scores), GOLD_A),
            ("Scaled scores", rf"qK^{{\mathsf T}}/\sqrt{{{width}}}=" + self.vector(scaled), BLUE_A),
            ("Softmax weights", r"w\approx" + self.vector(weights, 3), TEAL_A),
        ]
        for i, (label, equation, color) in enumerate(rows):
            y = 1.65 - 1.57 * i
            name = Text(label, font_size=31, color=color).move_to([-3.87, y, 0])
            tex = MathTex(equation, font_size=43, color=color).move_to([1.75, y, 0])
            self.play(FadeIn(name), Write(tex))
            self.wait(1.2)
        note = Text("Softmax runs across the keys. The exact weights sum to 1.",
                    font_size=26, color=GREY_A).to_edge(DOWN, buff=0.59)
        self.play(FadeIn(note))
        self.wait(3)
        self.clear_panel()

    def weighted_values(self, context):
        self.heading("Mix the values using the attention weights")
        equation = MathTex(
            r"wV=w_1\begin{bmatrix}1&0\end{bmatrix}"
            r"+w_2\begin{bmatrix}0&1\end{bmatrix}"
            r"+w_3\begin{bmatrix}1&1\end{bmatrix}",
            font_size=47,
        ).move_to(UP * 1.10)
        result = MathTex(r"wV\approx" + self.vector(context, 3), font_size=53, color=TEAL_A)
        result.move_to(DOWN * 0.46)
        note = Text("Each output component is a weighted average of the value components.",
                    font_size=27, color=GREY_A).to_edge(DOWN, buff=0.64)
        self.play(Write(equation))
        self.wait(1)
        self.play(FadeIn(result), FadeIn(note))
        self.wait(4)
        self.clear_panel()

    def multi_head(self):
        self.heading("Multi-head attention")
        formula = MathTex(
            r"\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)",
            font_size=43,
        ).move_to(UP * 2.00)
        heads = VGroup(*[
            MathTex(rf"\operatorname{{head}}_{i}", font_size=42, color=color).move_to([x, 0.75, 0])
            for i, color, x in [(1, BLUE_A, -3.6), (2, GOLD_A, 0), ("h", TEAL_A, 3.6)]
        ])
        ellipsis = MathTex(r"\cdots", font_size=40).move_to([1.8, 0.75, 0])
        concat = MathTex(
            r"\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O",
            font_size=44,
            color=PURPLE_A,
        ).move_to(DOWN * 1.47)
        links = VGroup(*[
            Arrow(head.get_bottom(), [0.72 * head.get_x(), -1.02, 0], buff=0.16,
                  color=head.get_color(), stroke_width=3)
            for head in heads
        ])
        note = Text("The original Transformer base model uses eight heads.",
                    font_size=27, color=GREY_A).to_edge(DOWN, buff=0.56)
        self.play(Write(formula))
        self.play(FadeIn(heads), FadeIn(ellipsis))
        self.play(Create(links), Write(concat))
        self.play(FadeIn(note))
        self.wait(4)
