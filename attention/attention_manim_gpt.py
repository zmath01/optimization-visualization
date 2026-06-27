from manim import *
import numpy as np

# -----------------------------
# Helper function to create matrix grids
# -----------------------------
def create_matrix(rows, cols, color, label_text):
    squares = VGroup()
    for i in range(rows):
        row = VGroup()
        for j in range(cols):
            sq = Square(side_length=0.4, stroke_width=1)
            sq.set_fill(color, opacity=0.5)
            row.add(sq)
        row.arrange(RIGHT, buff=0.05)
        squares.add(row)
    squares.arrange(DOWN, buff=0.05)

    label = Text(label_text, font_size=30).next_to(squares, UP)
    return VGroup(squares, label)


# -----------------------------
# Main Scene
# -----------------------------
class TransformerAttentionScene(Scene):

    def construct(self):

        # =============================
        # Scene 1: Q, K, V Introduction
        # =============================
        Q = create_matrix(4, 4, BLUE, "Q")
        K = create_matrix(4, 4, GREEN, "K")
        V = create_matrix(4, 4, RED, "V")

        matrices = VGroup(Q, K, V).arrange(RIGHT, buff=1.5).scale(0.8)
        self.play(FadeIn(matrices))
        self.wait(1)

        # Highlight one query row
        q_row = Q[0][0][1]  # second row
        q_rect = SurroundingRectangle(q_row, color=BLUE, buff=0.05)

        # Highlight all key rows
        k_rects = VGroup(*[
            SurroundingRectangle(K[0][i], color=GREEN, buff=0.05)
            for i in range(4)
        ])

        self.play(Create(q_rect))
        self.play(LaggedStart(*[Create(r) for r in k_rects], lag_ratio=0.2))
        self.wait(1)

        formula = MathTex(
            r"\text{Attention}(Q,K,V)="
            r"\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
        ).scale(0.8).to_edge(DOWN)

        self.play(Write(formula))
        self.wait(2)

        # =============================
        # Scene 2: Dot Product
        # =============================
        self.play(FadeOut(k_rects), FadeOut(q_rect))

        q_vec = q_row.copy().move_to(LEFT * 3)
        k_vec = K[0][0].copy().move_to(RIGHT * 3)

        self.play(TransformFromCopy(q_row, q_vec),
                  TransformFromCopy(K[0][0], k_vec))
        self.wait(1)

        # Element-wise multiplication visualization
        products = VGroup()
        for i in range(4):
            bar = Rectangle(height=0.2, width=0.5)
            bar.set_fill(YELLOW, opacity=0.8)
            bar.move_to(DOWN * i * 0.4)
            products.add(bar)

        products.next_to(q_vec, RIGHT, buff=1)

        self.play(LaggedStart(*[Create(p) for p in products], lag_ratio=0.2))
        self.wait(1)

        sum_text = MathTex("s_i").set_color(YELLOW).next_to(products, RIGHT)
        self.play(Write(sum_text))
        self.wait(1)

        # Repeat for all keys (fast)
        scores = VGroup()
        for i in range(4):
            bar = Rectangle(height=0.3, width=0.6)
            bar.set_fill(YELLOW, opacity=0.8)
            scores.add(bar)

        scores.arrange(DOWN, buff=0.2).move_to(RIGHT * 3)

        self.play(Transform(products, scores), FadeOut(sum_text))
        self.wait(2)

        # =============================
        # Scene 3: Scaling + Softmax
        # =============================
        scale_text = MathTex(r"\div \sqrt{d_k}").next_to(scores, UP)
        self.play(Write(scale_text))
        self.wait(1)

        # Softmax normalization
        probs = scores.copy()
        for i, bar in enumerate(probs):
            bar.stretch(0.5 + 0.2 * i, dim=1)

        probs.set_color(GOLD)

        self.play(Transform(scores, probs))
        self.wait(1)

        label = Text("Attention Weights", font_size=28).next_to(probs, RIGHT)
        self.play(Write(label))
        self.wait(2)

        # =============================
        # Scene 4: Weighted Sum
        # =============================
        self.play(FadeOut(scale_text), FadeOut(label))

        V_matrix = V.copy().to_edge(RIGHT)
        weights = scores.copy().to_edge(LEFT)

        self.play(FadeIn(V_matrix), Transform(scores, weights))
        self.wait(1)

        weighted_rows = VGroup()
        for i in range(4):
            row = V_matrix[0][i].copy()
            row.scale(0.7 + i * 0.1)
            weighted_rows.add(row)

        weighted_rows.arrange(DOWN, buff=0.2).move_to(ORIGIN)

        self.play(LaggedStart(*[
            TransformFromCopy(V_matrix[0][i], weighted_rows[i])
            for i in range(4)
        ], lag_ratio=0.2))

        self.wait(1)

        # Sum into context vector
        context = Square(color=PURPLE, fill_opacity=0.8).scale(0.7)
        context_label = Text("Context", font_size=28).next_to(context, DOWN)

        self.play(Transform(weighted_rows, context))
        self.play(Write(context_label))
        self.wait(2)

        # =============================
        # Scene 5: Multi-Head Attention
        # =============================
        self.play(FadeOut(context_label))

        heads = VGroup()
        for i in range(4):
            head = Square(color=BLUE, fill_opacity=0.5).scale(0.5)
            heads.add(head)

        heads.arrange(RIGHT, buff=0.5).to_edge(UP)

        self.play(FadeIn(heads))
        self.wait(1)

        outputs = VGroup()
        for i in range(4):
            out = Square(color=PURPLE, fill_opacity=0.6).scale(0.4)
            outputs.add(out)

        outputs.arrange(RIGHT, buff=0.5).next_to(heads, DOWN)

        self.play(LaggedStart(*[
            TransformFromCopy(heads[i], outputs[i])
            for i in range(4)
        ], lag_ratio=0.2))

        self.wait(1)

        concat = Rectangle(width=3, height=0.6, color=PURPLE)
        concat_label = Text("Concat + Linear", font_size=28).next_to(concat, DOWN)

        self.play(Transform(outputs, concat))
        self.play(Write(concat_label))

        final = Square(color=PURPLE, fill_opacity=1).scale(0.7).next_to(concat, DOWN)

        self.play(Transform(concat, final))
        self.wait(2)
