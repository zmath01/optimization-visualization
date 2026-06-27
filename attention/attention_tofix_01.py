from manim import *
import numpy as np

np.random.seed(42)

CELL = 0.35
BAR_W = 0.22
BAR_GAP = 0.06
BAR_H_MAX = 0.55


class TransformerAttentionScene(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        # --- data ---
        n = 4
        d = 4

        Q_data = np.array([
            [0.8, 0.3, 0.6, 0.1],
            [0.2, 0.7, 0.4, 0.9],
            [0.6, 0.5, 0.1, 0.3],
            [0.4, 0.8, 0.9, 0.2],
        ])
        K_data = np.array([
            [0.7, 0.2, 0.5, 0.9],
            [0.1, 0.8, 0.2, 0.3],
            [0.9, 0.1, 0.7, 0.2],
            [0.3, 0.6, 0.1, 0.8],
        ])
        V_data = np.array([
            [0.5, 0.9, 0.2, 0.4],
            [0.7, 0.1, 0.8, 0.3],
            [0.1, 0.6, 0.3, 0.9],
            [0.8, 0.4, 0.7, 0.2],
        ])

        q_row = Q_data[0]
        scores = []
        for i in range(n):
            scores.append(float(np.dot(q_row, K_data[i])))
        scores = np.array(scores)                    # [1.01, 0.47, 1.19, 0.56]
        scores_scaled = scores / np.sqrt(d)          # [0.505, 0.235, 0.595, 0.28]
        e = np.exp(scores_scaled - np.max(scores_scaled))
        attn_weights = e / e.sum()                   # softmax

        # =====================================================================
        # SCENE 1: Q, K, V Introduction
        # =====================================================================
        title1 = Text("Scaled Dot-Product Attention", font_size=30, color=WHITE)
        title1.to_edge(UP, buff=0.1)

        formula1 = MathTex(
            r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=26, color=LIGHT_GRAY,
        )
        formula1.next_to(title1, DOWN, buff=0.15)

        self.play(Write(title1), run_time=0.8)
        self.play(Write(formula1), run_time=1.2)
        self.wait(0.5)

        # add explanatory text
        intro_text = Text(
            "Each token has Query, Key, Value vectors",
            font_size=18, color=LIGHT_GRAY,
        )
        intro_text.to_edge(DOWN, buff=0.25)
        self.play(Write(intro_text), run_time=0.8)
        self.wait(0.3)

        # matrices
        mat_y = 0.4
        Q_mat = self.make_grid(Q_data, BLUE, [-4.5, mat_y, 0], "Q", BLUE)
        K_mat = self.make_grid(K_data, GREEN, [-0.5, mat_y, 0], "K", GREEN)
        V_mat = self.make_grid(V_data, RED, [3.5, mat_y, 0], "V", RED)

        self.play(
            FadeIn(Q_mat, shift=UP * 0.3),
            FadeIn(K_mat, shift=UP * 0.3),
            FadeIn(V_mat, shift=UP * 0.3),
            run_time=1.5,
        )
        self.wait(0.8)

        self.play(FadeOut(intro_text))
        self.wait(0.2)

        # highlight query row in Q
        q_label = Text("Query vector q", font_size=18, color=BLUE)
        q_label.next_to(Q_mat, DOWN, buff=0.25)
        q_highlight = SurroundingRectangle(
            Q_mat[0:4], color=YELLOW, buff=0.04, stroke_width=2.5,
        )
        self.play(Create(q_highlight), Write(q_label), run_time=0.8)
        self.wait(0.5)

        # explain query
        q_explain = Text(
            "Query: 'What am I looking for?'",
            font_size=16, color=BLUE,
        )
        q_explain.next_to(q_label, DOWN, buff=0.1).shift(RIGHT * 0.5)
        self.play(Write(q_explain), run_time=0.6)
        self.wait(0.8)
        self.play(FadeOut(q_explain), run_time=0.3)
        self.wait(0.2)

        # highlight all keys in K
        k_label = Text("Key vectors k_i", font_size=18, color=GREEN)
        k_label.next_to(K_mat, DOWN, buff=0.25)
        k_highlights = VGroup()
        for i in range(n):
            rect = SurroundingRectangle(
                K_mat[i * d:(i + 1) * d], color=YELLOW, buff=0.03, stroke_width=1.5, stroke_opacity=0.6,
            )
            k_highlights.add(rect)

        self.play(
            FadeIn(k_highlights, lag_ratio=0.1),
            Write(k_label),
            run_time=1,
        )
        self.wait(0.5)

        # explain keys
        k_explain = Text(
            "Keys: 'What do I contain?'",
            font_size=16, color=GREEN,
        )
        k_explain.next_to(k_label, DOWN, buff=0.1).shift(RIGHT * 0.5)
        self.play(Write(k_explain), run_time=0.6)
        self.wait(0.8)
        self.play(FadeOut(k_explain), run_time=0.3)

        v_explain = Text(
            "Values: 'What information do I pass?'",
            font_size=16, color=RED,
        )
        v_explain.next_to(V_mat, DOWN, buff=0.25)
        self.play(Write(v_explain), run_time=0.6)
        self.wait(1)
        self.play(FadeOut(v_explain), run_time=0.3)

        scene1 = VGroup(title1, formula1, Q_mat, K_mat, V_mat,
                        q_highlight, q_label, k_highlights, k_label)

        # =====================================================================
        # SCENE 2: Dot Product (Similarity)
        # =====================================================================
        self.play(FadeOut(scene1), run_time=0.6)
        self.wait(0.15)

        title2 = Text("Computing Similarity: q \u00b7 k_i", font_size=28, color=YELLOW)
        title2.to_edge(UP, buff=0.1)
        self.play(Write(title2), run_time=0.6)

        # show q and k_i bars
        bar_area_y = 0.5
        dim_labels = VGroup(*[
            MathTex(f"d_{{{j+1}}}", font_size=14, color=LIGHT_GRAY)
            for j in range(d)
        ])
        for j, dl in enumerate(dim_labels):
            dl.move_to([-1.5 + j * (BAR_W + BAR_GAP), bar_area_y + 1.0, 0])

        # q bars
        q_bars, _ = self.make_bars(q_row, BLUE, [-1.5, bar_area_y, 0])
        q_bar_label = Text("q", font_size=18, color=BLUE).next_to(q_bars, LEFT, buff=0.2)

        self.play(Write(dim_labels), FadeIn(q_bars), Write(q_bar_label), run_time=0.8)
        self.wait(0.2)

        # iterate over keys
        score_bars_list = []
        score_vals = []

        for key_idx in range(n):
            is_first = key_idx == 0
            run = 1.2 if is_first else 0.6

            k_row = K_data[key_idx]
            k_bars, _ = self.make_bars(k_row, GREEN, [-1.5, bar_area_y - 0.5, 0])
            k_bar_label = MathTex(f"k_{{{key_idx+1}}}", font_size=16, color=GREEN)
            k_bar_label.next_to(k_bars, LEFT, buff=0.2)

            if is_first:
                self.play(FadeIn(k_bars), Write(k_bar_label), run_time=0.5)

            # element-wise product bars
            prod = q_row * k_row
            prod_bars, _ = self.make_bars(prod, YELLOW, [-1.5, bar_area_y - 1.0, 0])
            prod_label = MathTex(r"q \cdot k_{" + str(key_idx + 1) + "}", font_size=14, color=YELLOW)
            prod_label.next_to(prod_bars, LEFT, buff=0.2)

            if is_first:
                # animate multiplication: transform each pair into product
                self.play(
                    *[ReplacementTransform(
                        VGroup(q_bars[j].copy(), k_bars[j].copy()),
                        prod_bars[j],
                    ) for j in range(d)],
                    Write(prod_label),
                    run_time=1.2,
                )
            else:
                self.play(
                    FadeOut(k_bars, k_bar_label),
                    FadeIn(k_bars, k_bar_label),
                    FadeIn(prod_bars),
                    Write(prod_label),
                    run_time=run,
                )

            # sum → score scalar
            s = float(np.sum(prod))
            score_vals.append(s)

            if is_first:
                sum_arrow = Arrow(
                    prod_bars.get_center() + DOWN * 0.2,
                    prod_bars.get_center() + DOWN * 0.6,
                    color=YELLOW, stroke_width=3,
                )
                score_display = MathTex(f"s_{{{key_idx+1}}} = {s:.2f}", font_size=18, color=YELLOW)
                score_display.next_to(sum_arrow, DOWN, buff=0.1)

                # animate bars merging into sum
                self.play(
                    LaggedStart(
                        *[b.animate.move_to(sum_arrow.get_start()).scale(0.2) for b in prod_bars],
                        lag_ratio=0.05,
                    ),
                    GrowArrow(sum_arrow),
                    run_time=1.2,
                )
                self.play(Write(score_display), run_time=0.4)
                score_bars_list.append(score_display)
            else:
                score_display = MathTex(f"s_{{{key_idx+1}}} = {s:.2f}", font_size=16, color=YELLOW)
                score_display.next_to(score_bars_list[-1], RIGHT, buff=0.2)
                self.play(
                    FadeOut(prod_bars, prod_label),
                    Write(score_display),
                    run_time=run,
                )
                score_bars_list.append(score_display)

            if key_idx == 0:
                self.wait(0.5)
            self.wait(0.1)

        # score vector
        score_vector_label = Text("Score vector s", font_size=20, color=YELLOW)
        score_vector_label.next_to(score_bars_list[0], DOWN, buff=0.5).shift(RIGHT * 0.5)
        score_brace = Brace(VGroup(*score_bars_list), DOWN, buff=0.2)
        self.play(
            Write(score_vector_label),
            GrowFromCenter(score_brace),
            run_time=0.8,
        )
        self.wait(0.5)

        scene2 = VGroup(
            title2, dim_labels, q_bars, q_bar_label,
            *score_bars_list, score_vector_label, score_brace,
        )

        # =====================================================================
        # SCENE 3: Scaling & Softmax
        # =====================================================================
        self.play(FadeOut(scene2), run_time=0.6)
        self.wait(0.15)

        title3 = Text("Scaling \u00f7 \u221ad\u2096  +  Softmax", font_size=28, color=YELLOW)
        title3.to_edge(UP, buff=0.1)
        self.play(Write(title3), run_time=0.6)

        # raw score bars
        raw_y = 1.2
        raw_bars, raw_rects = self.make_bars(scores, YELLOW, [-2.0, raw_y, 0], max_h=0.7)
        raw_label = Text("Raw scores", font_size=18, color=YELLOW).next_to(raw_bars, LEFT, buff=0.2)

        self.play(FadeIn(raw_bars), Write(raw_label), run_time=0.6)
        self.wait(0.2)

        # scale by sqrt(d_k)
        scale_text = MathTex(r"\div \sqrt{d_k} = 2", font_size=22, color=YELLOW)
        scale_text.next_to(raw_bars, DOWN, buff=0.4)

        scaled_bars, _ = self.make_bars(scores_scaled, YELLOW, [-2.0, raw_y - 1.0, 0], max_h=0.7)
        scaled_label = Text("Scaled scores", font_size=18, color=YELLOW).next_to(scaled_bars, LEFT, buff=0.2)

        self.play(Write(scale_text), run_time=0.5)
        self.wait(0.3)

        scale_explain = Text(
            "Prevents softmax gradients from vanishing",
            font_size=16, color=YELLOW,
        )
        scale_explain.next_to(scale_text, DOWN, buff=0.15)
        self.play(Write(scale_explain), run_time=0.6)
        self.wait(0.5)

        self.play(
            Transform(raw_bars, scaled_bars),
            Transform(raw_label, scaled_label),
            run_time=1,
        )
        self.play(FadeOut(scale_explain), run_time=0.3)
        self.wait(0.5)

        # softmax
        softmax_label = Text("Softmax \u2192 Probabilities", font_size=22, color=YELLOW)
        softmax_label.next_to(raw_label, DOWN, buff=0.5).shift(RIGHT * 0.5)

        prob_bars, _ = self.make_bars(attn_weights, YELLOW, [-2.0, raw_y - 2.0, 0], max_h=0.7)
        prob_label = Text("Attention weights", font_size=18, color=YELLOW).next_to(prob_bars, LEFT, buff=0.2)

        sum_label = Text("Sum = 1.00", font_size=16, color=GREEN).next_to(prob_bars, DOWN, buff=0.2)

        self.play(Write(softmax_label), run_time=0.5)
        self.play(
            Transform(raw_bars, prob_bars),
            Transform(raw_label, prob_label),
            run_time=1.5,
        )
        self.play(Write(sum_label), run_time=0.5)
        self.wait(0.5)

        soft_explain = Text(
            "Softmax converts scores to probabilities summing to 1",
            font_size=16, color=YELLOW,
        )
        soft_explain.next_to(sum_label, DOWN, buff=0.25)
        self.play(Write(soft_explain), run_time=0.8)
        self.wait(1)
        self.play(FadeOut(soft_explain), run_time=0.3)

        scene3 = VGroup(
            title3, raw_bars, raw_label, scale_text, softmax_label, sum_label,
        )

        # =====================================================================
        # SCENE 4: Weighted Sum of Values
        # =====================================================================
        self.play(FadeOut(scene3), run_time=0.6)
        self.wait(0.15)

        title4 = Text("Weighted Sum of Values \u2192 Context Vector", font_size=26, color=PURPLE)
        title4.to_edge(UP, buff=0.1)
        self.play(Write(title4), run_time=0.6)

        # show V matrix and attention weights
        V_small = self.make_grid(V_data, RED, [-4.0, 0.5, 0], "V", RED, cell=0.28)
        self.play(FadeIn(V_small), run_time=0.6)

        # weight indicators next to V rows
        weight_dots = VGroup()
        for i in range(n):
            frac = attn_weights[i]
            bar = Rectangle(
                width=frac * 0.6, height=0.12,
                color=YELLOW, fill_opacity=0.8, stroke_width=0,
            )
            bar.next_to(V_small[i * d:(i + 1) * d], RIGHT, buff=0.15)
            bar.align_to(V_small[i * d:(i + 1) * d], UP)
            wl = MathTex(f"{frac:.2f}", font_size=11, color=YELLOW)
            wl.next_to(bar, RIGHT, buff=0.04)
            weight_dots.add(VGroup(bar, wl))

        weight_title = Text("Attention weights", font_size=16, color=YELLOW)
        weight_title.next_to(weight_dots, RIGHT, buff=0.2)

        self.play(
            LaggedStart(*[FadeIn(w) for w in weight_dots], lag_ratio=0.1),
            Write(weight_title),
            run_time=1.2,
        )
        self.wait(0.3)

        # weight * V rows (scaled V rows)
        context = np.zeros(d)
        scaled_row_groups = VGroup()
        for i in range(n):
            w = attn_weights[i]
            context += w * V_data[i]
            scaled_row = self.make_row_bar(w * V_data[i], RED if w > 0.25 else RED, 0.3)
            scaled_row.next_to(weight_dots, RIGHT, buff=0.5 + i * 0.3)
            scaled_row.scale(0.5)
            scaled_row_groups.add(scaled_row)

        # show weighted rows
        self.play(
            *[FadeIn(sr, shift=DOWN * 0.2) for sr in scaled_row_groups],
            run_time=1.5, lag_ratio=0.15,
        )
        self.wait(0.3)

        # context vector
        cv_y = -1.5
        cv_bars, _ = self.make_bars(context, PURPLE, [-1.5, cv_y, 0], max_h=0.55)
        cv_label = Text("Context Vector (output)", font_size=18, color=PURPLE)
        cv_label.next_to(cv_bars, LEFT, buff=0.2)
        cv_box = SurroundingRectangle(cv_bars, color=PURPLE, buff=0.1, stroke_width=2)

        sum_arrow2 = Arrow(
            scaled_row_groups.get_center() + RIGHT * 0.5,
            cv_bars.get_center() + UP * 0.5,
            color=PURPLE, stroke_width=3,
        )

        self.play(
            GrowArrow(sum_arrow2),
            FadeIn(cv_bars, shift=UP * 0.2),
            Write(cv_label),
            Create(cv_box),
            run_time=1.5,
        )

        attention_text = Text(
            "High weight \u2192 model attends to that token",
            font_size=17, color=YELLOW,
        )
        attention_text.to_edge(DOWN, buff=0.2)
        self.play(Write(attention_text), run_time=0.8)
        self.wait(1)

        attended_label = Text(
            "Token 3 has highest attention weight",
            font_size=16, color=YELLOW,
        )
        attended_label.next_to(attention_text, UP, buff=0.15)
        self.play(Write(attended_label), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(attended_label), run_time=0.3)

        scene4 = VGroup(
            title4, V_small, weight_dots, weight_title,
            scaled_row_groups, sum_arrow2, cv_bars, cv_label, cv_box,
            attention_text,
        )

        # =====================================================================
        # SCENE 5: Multi-Head Attention
        # =====================================================================
        self.play(FadeOut(scene4), run_time=0.6)
        self.wait(0.15)

        title5 = Text("Multi-Head Attention (h = 4)", font_size=28, color=PURPLE)
        title5.to_edge(UP, buff=0.1)

        mh_formula = MathTex(
            r"\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O",
            font_size=22, color=LIGHT_GRAY,
        )
        mh_formula.next_to(title5, DOWN, buff=0.15)

        self.play(Write(title5), Write(mh_formula), run_time=0.8)
        self.wait(0.3)

        # show h=4 parallel attention heads
        h = 4
        heads = VGroup()
        head_center_y = -0.3
        head_spacing = 1.6
        start_x = -(h - 1) * head_spacing / 2

        for hi in range(h):
            head = VGroup()

            # smaller Q, K, V grids
            hq = self.make_grid(
                Q_data * (0.7 + 0.3 * np.random.rand(*Q_data.shape)),
                BLUE_B if hi % 2 == 0 else BLUE_D,
                [start_x + hi * head_spacing, head_center_y + 0.6, 0],
                "", BLUE, cell=0.18, rows=3, cols=3,
            )
            hk = self.make_grid(
                K_data * (0.7 + 0.3 * np.random.rand(*K_data.shape)),
                GREEN_B if hi % 2 == 0 else GREEN_D,
                [start_x + hi * head_spacing, head_center_y - 0.2, 0],
                "", GREEN, cell=0.18, rows=3, cols=3,
            )
            hv = self.make_grid(
                V_data * (0.7 + 0.3 * np.random.rand(*V_data.shape)),
                RED_B if hi % 2 == 0 else RED_D,
                [start_x + hi * head_spacing, head_center_y - 1.0, 0],
                "", RED, cell=0.18, rows=3, cols=3,
            )

            # attention arrow through head
            attn_arrow = Arrow(
                [start_x + hi * head_spacing, head_center_y - 1.3, 0],
                [start_x + hi * head_spacing, head_center_y + 1.0, 0],
                color=PURPLE, stroke_width=2, buff=0.08,
            )

            head_label = MathTex(f"h_{{{hi+1}}}", font_size=14, color=PURPLE)
            head_label.next_to(attn_arrow, LEFT, buff=0.08)

            head.add(hq, hk, hv, attn_arrow, head_label)
            heads.add(head)

        # parallel computation animation
        self.play(
            LaggedStart(*[
                FadeIn(heads[hi], shift=UP * 0.2) for hi in range(h)
            ], lag_ratio=0.15),
            run_time=2,
        )
        self.wait(0.3)

        # animate attention arrows flowing
        self.play(
            *[GrowArrow(heads[hi][3]) for hi in range(h)],
            run_time=1, lag_ratio=0.1,
        )
        self.wait(0.3)

        # concatenation
        concat_bar = Rectangle(
            width=3.2, height=0.35,
            color=PURPLE, fill_opacity=0.15, stroke_width=2,
        )
        concat_bar.to_edge(DOWN, buff=0.5)

        concat_label = MathTex(
            r"\text{Concat}(", font_size=18, color=PURPLE
        )
        concat_label.next_to(concat_bar, LEFT, buff=0.1)

        # arrows from heads to concat
        concat_arrows = VGroup()
        for hi in range(h):
            src = heads[hi][3].get_center() + DOWN * 0.3
            dst = concat_bar.get_top() + RIGHT * (-1.5 + hi * 1.0)
            arr = Arrow(src, dst, color=PURPLE, stroke_width=1.5, buff=0.05)
            concat_arrows.add(arr)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in concat_arrows], lag_ratio=0.08),
            FadeIn(concat_bar, shift=UP * 0.3),
            Write(concat_label),
            run_time=1.5,
        )
        self.wait(0.3)

        # W^O
        wo_arrow = Arrow(
            concat_bar.get_right() + RIGHT * 0.3,
            concat_bar.get_right() + RIGHT * 1.8,
            color=PURPLE, stroke_width=3,
        )
        wo_label = MathTex(r"W^O", font_size=22, color=PURPLE).next_to(wo_arrow, UP, buff=0.05)

        output_box = Rectangle(
            width=1.2, height=0.5,
            color=PURPLE, fill_opacity=0.2, stroke_width=2.5,
        )
        output_box.next_to(wo_arrow, RIGHT, buff=0.1)
        out_text = Text("Output", font_size=16, color=PURPLE).move_to(output_box)

        self.play(
            GrowArrow(wo_arrow), Write(wo_label),
            FadeIn(output_box, shift=RIGHT * 0.3), Write(out_text),
            run_time=1,
        )

        subspaces_text = Text(
            "Each head captures different semantic subspaces",
            font_size=18, color=YELLOW,
        )
        subspaces_text.to_edge(DOWN, buff=0.1)
        self.play(Write(subspaces_text), run_time=0.8)
        self.wait(1)

        mh_explain = Text(
            "h=8 in the original Transformer paper",
            font_size=16, color=LIGHT_GRAY,
        )
        mh_explain.next_to(subspaces_text, UP, buff=0.15)
        self.play(Write(mh_explain), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(mh_explain), run_time=0.3)
        self.wait(0.5)

        # recap summary
        recap = VGroup(
            Text("Recap:", font_size=24, color=YELLOW),
            Text("1. Q\u00b7K \u2192 Similarity scores", font_size=20, color=WHITE),
            Text("2. Scale & Softmax \u2192 Attention weights", font_size=20, color=WHITE),
            Text("3. Weights \u00d7 V \u2192 Context vector", font_size=20, color=WHITE),
            Text("4. Multi-head \u2192 Parallel subspaces", font_size=20, color=WHITE),
        )
        recap.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        recap.move_to(title5.get_center() + DOWN * 1.5)

        self.play(
            *[FadeOut(m) for m in self.mobjects if m not in [title5]],
            FadeIn(recap, shift=UP * 0.3),
            run_time=1.5,
        )
        self.wait(2)

        self.play(FadeOut(recap), FadeOut(title5), run_time=1)
        self.wait(1)

    # =====================================================================
    # helpers
    # =====================================================================
    def make_grid(self, values, color, pos, label, label_color, cell=CELL, rows=None, cols=None):
        if rows is None:
            rows = values.shape[0]
        if cols is None:
            cols = values.shape[1]
        grid = VGroup()
        for i in range(rows):
            for j in range(cols):
                v = values[i, j]
                sq = Square(
                    side_length=cell * 0.88,
                    color=color,
                    fill_color=color,
                    fill_opacity=min(1.0, v * 0.85 + 0.1),
                    stroke_width=1.5,
                )
                sq.move_to([
                    pos[0] + j * cell,
                    pos[1] - i * cell,
                    0,
                ])
                grid.add(sq)
        if label:
            lbl = MathTex(label, font_size=20, color=label_color)
            lbl.next_to(grid, DOWN, buff=0.1)
            grid.add(lbl)
        return grid

    def make_bars(self, values, color, pos, max_h=BAR_H_MAX):
        bars = VGroup()
        rects = VGroup()
        for i, v in enumerate(values):
            h = max(0.02, v * max_h)
            bar = Rectangle(
                width=BAR_W, height=h,
                color=color, fill_opacity=0.85, stroke_width=1,
            )
            bar.move_to([pos[0] + i * (BAR_W + BAR_GAP), pos[1] + h / 2, 0])
            bars.add(bar)

            outline = Rectangle(
                width=BAR_W, height=max_h,
                color=color, stroke_width=0.8, fill_opacity=0,
            )
            outline.move_to([pos[0] + i * (BAR_W + BAR_GAP), pos[1] + max_h / 2, 0])
            rects.add(outline)
        return bars, rects

    def make_row_bar(self, values, color, max_h=0.3):
        parts = VGroup()
        for i, v in enumerate(values):
            h = max(0.02, v * max_h)
            bar = Rectangle(
                width=BAR_W * 0.6, height=h,
                color=color, fill_opacity=0.85, stroke_width=1,
            )
            bar.move_to([i * (BAR_W * 0.6 + 0.03), h / 2, 0])
            parts.add(bar)
        return parts
