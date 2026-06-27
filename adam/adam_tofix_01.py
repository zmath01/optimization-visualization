from manim import *
import numpy as np

np.random.seed(42)


class AdamOptimizerScene(Scene):
    def construct(self):
        BETA1 = 0.9
        BETA2 = 0.999
        EPS = 1e-8
        LR = 0.3
        START_X = -3.0

        def f(x):
            return 0.5 * x * x

        def df(x):
            return x

        # --- persistent elements ---
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 6, 1],
            x_length=7.5,
            y_length=4.8,
            axis_config={"include_numbers": True, "font_size": 20, "color": LIGHT_GRAY},
        )
        axes.shift(LEFT * 2.5 + DOWN * 0.15)

        curve = axes.plot(f, x_range=[-3.9, 3.9], color=LIGHT_GRAY, stroke_width=3)
        curve_label = MathTex(r"f(\theta)", font_size=22, color=LIGHT_GRAY).next_to(
            axes.c2p(0, 5.5), UP, buff=0.1
        )

        title = Text(
            "Adam: A Method for Stochastic Optimization",
            font_size=30,
            color=BLUE,
        )
        title.to_edge(UP, buff=0.15)

        theta_tracker = ValueTracker(START_X)

        def get_param_pt():
            return axes.c2p(theta_tracker.get_value(), f(theta_tracker.get_value()))

        param_dot = always_redraw(
            lambda: Dot(get_param_pt(), color=BLUE, radius=0.08)
        )
        param_label = always_redraw(
            lambda: MathTex(r"\theta", font_size=24, color=BLUE).next_to(
                param_dot, UP, buff=0.05
            )
        )

        # --- helpers ---
        def tangent_line(x, length=0.8):
            y = f(x)
            s = df(x)
            dx = length * 0.5
            return Line(
                axes.c2p(x - dx, y - s * dx),
                axes.c2p(x + dx, y + s * dx),
                color=YELLOW,
                stroke_width=2.5,
            )

        def grad_arrow(x, color=RED):
            y = f(x)
            g = df(x)
            if abs(g) < 0.01:
                return Vector(RIGHT * 0.01, color=color)
            direction = -np.sign(g)
            length = min(1.0, max(0.35, abs(g) * 0.18))
            start = axes.c2p(x, y)
            end = start + direction * length * RIGHT
            return Arrow(start, end, color=color, stroke_width=5, buff=0.05)

        # ========== SCENE 1 ==========
        self.play(Write(title, run_time=1.5))
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(param_dot, scale=0.5), Write(param_label))
        self.wait(0.5)

        start_label = MathTex(r"\theta_0", font_size=22, color=BLUE).next_to(
            param_dot, DOWN, buff=0.15
        )
        self.play(Write(start_label))
        self.wait(0.5)

        update_rule = MathTex(
            r"\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
            font_size=24,
            color=GRAY,
        )
        update_rule.to_edge(DOWN, buff=0.15).to_edge(RIGHT, buff=0.3)
        update_rule.set_opacity(0.3)
        self.play(Write(update_rule, run_time=1))
        self.wait(0.5)

        highlight = update_rule.copy().set_color(YELLOW).set_opacity(1)
        self.play(Transform(update_rule, highlight, run_time=0.8))
        self.wait(1.5)
        faint = update_rule.copy().set_color(GRAY).set_opacity(0.3)
        self.play(Transform(update_rule, faint, run_time=0.5))
        self.wait(0.5)

        # ========== SCENE 2: Gradient ==========
        x0 = theta_tracker.get_value()

        tangent = tangent_line(x0)
        g_arrow = grad_arrow(x0)
        g_label = MathTex(r"g_t", font_size=28, color=RED).next_to(
            g_arrow.get_end(), RIGHT, buff=0.1
        )

        g_formula = MathTex(
            r"g_t = \nabla_\theta f_t(\theta)", font_size=26, color=RED
        )
        g_formula.next_to(axes, RIGHT, buff=0.5).shift(UP * 1.8)

        self.play(Create(tangent), run_time=0.8)
        self.play(GrowArrow(g_arrow), Write(g_label), run_time=1.2)
        self.play(Write(g_formula))
        self.wait(1.5)

        self.play(
            FadeOut(tangent, shift=UP * 0.1),
            FadeOut(g_arrow, shift=UP * 0.1),
            FadeOut(g_label, shift=UP * 0.1),
            FadeOut(g_formula, shift=UP * 0.1),
        )
        self.wait(0.3)

        # ========== SCENE 3: First Moment (Momentum) ==========
        panel_center = axes.get_right() + RIGHT * 0.9 + UP * 0.3

        m_formula = MathTex(
            r"m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t",
            font_size=26,
            color=GREEN,
        )
        m_formula.move_to(panel_center + UP * 1.6)

        beta1_text = MathTex(r"\beta_1 = 0.9", font_size=24, color=GREEN).next_to(
            m_formula, DOWN, buff=0.25
        )

        self.play(Write(m_formula, run_time=1.5))
        self.play(Write(beta1_text))
        self.wait(1)

        # vector addition visual
        vec_center = panel_center + DOWN * 0.5

        m_prev_arrow = Arrow(
            vec_center + LEFT * 1.6,
            vec_center,
            color=GREEN,
            stroke_width=8,
            stroke_opacity=0.3,
        )
        m_prev_label = MathTex(r"m_{t-1}", font_size=18, color=GREEN).next_to(
            m_prev_arrow, UP, buff=0.05
        )

        g_contrib_arrow = Arrow(
            vec_center + LEFT * 0.35,
            vec_center,
            color=RED,
            stroke_width=4,
        )
        g_contrib_label = MathTex(
            r"(1-\beta_1) g_t", font_size=16, color=RED
        ).next_to(g_contrib_arrow, UP, buff=0.05)

        plus_sign = MathTex("+", font_size=24, color=WHITE).move_to(
            vec_center - LEFT * 0.95
        )

        eq_sign = MathTex("=", font_size=24, color=WHITE).next_to(
            vec_center, DOWN, buff=0.8
        )

        m_res_arrow = Arrow(
            vec_center + DOWN * 1.2 + LEFT * 1.7,
            vec_center + DOWN * 1.2,
            color=GREEN,
            stroke_width=10,
        )
        m_res_label = MathTex(r"m_t", font_size=20, color=GREEN).next_to(
            m_res_arrow, UP, buff=0.05
        )

        self.play(FadeIn(m_prev_arrow), Write(m_prev_label))
        self.play(Write(plus_sign))
        self.play(GrowArrow(g_contrib_arrow), Write(g_contrib_label))
        self.wait(0.8)
        self.play(
            FadeOut(m_prev_arrow, m_prev_label, plus_sign, g_contrib_arrow, g_contrib_label),
            GrowArrow(m_res_arrow),
            Write(eq_sign),
            Write(m_res_label),
        )

        smooth_text = Text("Smooths noisy gradients", font_size=18, color=GREEN).next_to(
            m_res_arrow, DOWN, buff=0.3
        )
        self.play(Write(smooth_text))
        self.wait(1.5)
        self.play(
            FadeOut(m_formula, beta1_text, m_res_arrow, m_res_label, eq_sign, smooth_text)
        )
        self.wait(0.3)

        # ========== SCENE 4: Second Moment (Scaling) ==========
        v_formula = MathTex(
            r"v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2",
            font_size=26,
            color=ORANGE,
        )
        v_formula.move_to(panel_center + UP * 1.6)

        beta2_text = MathTex(r"\beta_2 = 0.999", font_size=24, color=ORANGE).next_to(
            v_formula, DOWN, buff=0.25
        )

        self.play(Write(v_formula, run_time=1.5))
        self.play(Write(beta2_text))
        self.wait(1)

        # g² squaring
        g_sq_display = VGroup()
        g_val = MathTex(r"g_t = -3.0", font_size=22, color=RED).move_to(
            panel_center + DOWN * 0.1
        )
        arrow1 = Arrow(
            g_val.get_right() + RIGHT * 0.3,
            g_val.get_right() + RIGHT * 1.5,
            color=ORANGE,
            stroke_width=3,
        )
        g_sq_val = MathTex(r"g_t^2 = 9.0", font_size=22, color=ORANGE).next_to(
            arrow1, RIGHT, buff=0.15
        )

        self.play(Write(g_val))
        self.play(GrowArrow(arrow1))
        self.play(Write(g_sq_val))
        self.wait(0.5)

        # bar chart for v_t
        bar_y = panel_center[1] - 1.2
        bar_x = panel_center[0]

        bar_bg = Rectangle(width=2.8, height=0.35, color=ORANGE, stroke_width=1.5, fill_opacity=0.1)
        bar_bg.move_to([bar_x, bar_y, 0])

        bar_fill = Rectangle(
            width=0.01, height=0.35, color=ORANGE, fill_opacity=0.8, stroke_width=0
        ).move_to(bar_bg.get_left(), aligned_edge=LEFT)

        v_bar_label = MathTex(r"v_t", font_size=20, color=ORANGE).next_to(
            bar_bg, LEFT, buff=0.2
        )

        self.play(Create(bar_bg), Write(v_bar_label))
        self.play(bar_fill.animate.set_width(2.5), run_time=1.2)
        self.wait(0.5)

        expl_text = Text(r"Large gradients \rightarrow large v_t \rightarrow smaller steps", font_size=16, color=ORANGE).next_to(
            bar_bg, DOWN, buff=0.3
        )
        self.play(Write(expl_text))
        self.wait(1.5)
        self.play(
            FadeOut(v_formula, beta2_text, g_val, arrow1, g_sq_val, bar_bg, bar_fill, v_bar_label, expl_text)
        )
        self.wait(0.3)

        # ========== SCENE 5: Bias Correction ==========
        m_hat_formula = MathTex(
            r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}",
            font_size=26,
            color=GREEN,
        )
        m_hat_formula.move_to(panel_center + UP * 0.8)

        v_hat_formula = MathTex(
            r"\hat{v}_t = \frac{v_t}{1 - \beta_2^t}",
            font_size=26,
            color=ORANGE,
        )
        v_hat_formula.next_to(m_hat_formula, DOWN, buff=0.4)

        self.play(Write(m_hat_formula), Write(v_hat_formula), run_time=1.5)
        self.wait(0.5)

        # at t=1
        t1_line = MathTex(
            r"t = 1:", font_size=22, color=WHITE
        )
        t1_line.move_to(panel_center + DOWN * 0.6)

        m_biased_1 = MathTex(r"m_1 = 0.1 \, g_1", font_size=20, color=GREEN)
        m_biased_1.next_to(t1_line, DOWN, buff=0.15).align_to(t1_line, LEFT)

        m_corrected_1 = MathTex(
            r"\hat{m}_1 = \frac{0.1}{0.1} g_1 = g_1", font_size=20, color=GREEN
        )
        m_corrected_1.next_to(m_biased_1, DOWN, buff=0.1).align_to(m_biased_1, LEFT)

        v_biased_1 = MathTex(r"v_1 = 0.001 \, g_1^2", font_size=20, color=ORANGE)
        v_biased_1.next_to(m_corrected_1, DOWN, buff=0.1).align_to(m_corrected_1, LEFT)

        v_corrected_1 = MathTex(
            r"\hat{v}_1 = \frac{0.001}{0.001} g_1^2 = g_1^2", font_size=20, color=ORANGE
        )
        v_corrected_1.next_to(v_biased_1, DOWN, buff=0.1).align_to(v_biased_1, LEFT)

        bias_label = Text(
            "Fixing Initialization Bias", font_size=22, color=YELLOW
        )
        bias_label.next_to(panel_center + DOWN * 2.0, DOWN, buff=0.2)

        self.play(Write(t1_line))
        self.play(Write(m_biased_1), Write(v_biased_1))
        self.wait(0.5)
        self.play(
            Transform(m_biased_1, m_corrected_1),
            Transform(v_biased_1, v_corrected_1),
            run_time=1.2,
        )
        self.play(Write(bias_label))
        self.wait(1.5)
        self.play(
            FadeOut(
                m_hat_formula,
                v_hat_formula,
                t1_line,
                m_biased_1,
                v_biased_1,
                bias_label,
            )
        )
        self.wait(0.3)

        # ========== SCENE 6: Parameter Update ==========
        # precompute Adam trajectory
        theta = START_X
        m = 0.0
        v = 0.0
        adam_traj = [theta]
        adam_meta = []

        for t in range(1, 10):
            g = df(theta)
            m = BETA1 * m + (1 - BETA1) * g
            v = BETA2 * v + (1 - BETA2) * g * g
            m_h = m / (1 - BETA1**t)
            v_h = v / (1 - BETA2**t)
            step = LR * m_h / (np.sqrt(v_h) + EPS)
            theta = theta - step
            adam_traj.append(theta)
            adam_meta.append((g, m, v, m_h, v_h, step))

        # precompute noisy SGD trajectory
        sgd_theta = START_X
        sgd_traj = [sgd_theta]
        for _ in range(20):
            g = df(sgd_theta) + np.random.normal(0, 0.6)
            sgd_theta = sgd_theta - LR * g
            sgd_traj.append(sgd_theta)

        # fade in update elements
        step_formula = MathTex(
            r"\Delta\theta = \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
            font_size=24,
            color=PURPLE,
        )
        step_formula.next_to(axes, RIGHT, buff=0.3).shift(DOWN * 0.5)
        self.play(Write(step_formula))
        self.wait(0.5)

        # gradient arrow and tangent for scene 6 (always_redraw)
        g_arrow_dyn = always_redraw(
            lambda: grad_arrow(theta_tracker.get_value(), RED)
        )
        tangent_dyn = always_redraw(
            lambda: tangent_line(theta_tracker.get_value(), 0.8)
        )
        g_label_dyn = always_redraw(
            lambda: MathTex(r"g_t", font_size=24, color=RED).next_to(
                g_arrow_dyn.get_end(), RIGHT, buff=0.05
            )
        )

        # SGD dot
        sgd_dot = Dot(color=RED, radius=0.055, stroke_opacity=0.4, fill_opacity=0.4)

        # Draw SGD trace faintly in background
        sgd_points = [axes.c2p(x, f(x)) for x in sgd_traj]
        sgd_path = VMobject(color=RED, stroke_width=1.5, stroke_opacity=0.25)
        sgd_path.set_points_smoothly(sgd_points)
        self.play(Create(sgd_path), run_time=1.5)
        self.play(FadeIn(sgd_dot))
        self.wait(0.3)

        # Animate SGD noisy trajectory first (as comparison)
        for i in range(1, min(15, len(sgd_traj))):
            x_prev = sgd_traj[i - 1]
            x_next = sgd_traj[i]
            self.play(
                sgd_dot.animate.move_to(axes.c2p(x_next, f(x_next))),
                run_time=0.15,
                rate_func=linear,
            )
        self.wait(0.3)

        # Reset theta
        theta_tracker.set_value(START_X)
        g_arrow_dyn.set_opacity(0)
        tangent_dyn.set_opacity(0)
        g_label_dyn.set_opacity(0)

        self.play(
            FadeIn(g_arrow_dyn),
            FadeIn(tangent_dyn),
            FadeIn(g_label_dyn),
            sgd_dot.animate.move_to(axes.c2p(sgd_traj[-1], f(sgd_traj[-1]))),
            run_time=0.5,
        )

        step_counter = Text("Step 0", font_size=24, color=YELLOW)
        step_counter.to_corner(UR)
        self.play(Write(step_counter))
        self.wait(0.3)

        # Animate Adam steps
        step_value_text = (
            MathTex(
                r"\Delta\theta = \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
                font_size=22,
                color=PURPLE,
            )
            .next_to(axes, RIGHT, buff=0.3)
            .shift(DOWN * 1.2)
        )
        self.play(Write(step_value_text))
        self.wait(0.2)

        for i in range(1, len(adam_traj)):
            g, m_val, v_val, m_h, v_h, step = adam_meta[i - 1]
            new_x = adam_traj[i]

            step_count = MathTex(f"\\text{{Step {i}}}", font_size=22, color=YELLOW)
            step_count.move_to(step_counter.get_center())
            self.play(Transform(step_counter, step_count), run_time=0.3)

            values_str = (
                f"g_t = {g:.2f}\\quad "
                f"\\hat{{m}}_t = {m_h:.2f}\\quad "
                f"\\hat{{v}}_t = {v_h:.2f}"
            )
            new_values = MathTex(values_str, font_size=18, color=PURPLE)
            new_values.move_to(step_value_text.get_center())

            self.play(
                theta_tracker.animate.set_value(new_x),
                Transform(step_value_text, new_values),
                run_time=1.2,
            )

            # Show step magnitude indicator
            if i <= 3:
                step_mag = Rectangle(
                    width=min(2.0, abs(step) * 3),
                    height=0.2,
                    color=PURPLE,
                    fill_opacity=0.8,
                    stroke_width=0,
                )
                step_mag.next_to(step_value_text, DOWN, buff=0.15).align_to(
                    step_value_text, LEFT
                )
                mag_label = Text(
                    f"\u0394\u03b8 = {step:.3f}",
                    font_size=16,
                    color=PURPLE,
                ).next_to(step_mag, RIGHT, buff=0.1)
                self.play(
                    GrowFromPoint(step_mag, step_mag.get_left()),
                    Write(mag_label),
                    run_time=0.5,
                )
                self.wait(0.3)
                self.play(FadeOut(step_mag, mag_label), run_time=0.2)

            self.wait(0.2)

        # Final convergence
        self.play(FadeOut(sgd_dot, sgd_path), run_time=0.5)

        converge = Text("Converged!", font_size=36, color=GREEN)
        converge.next_to(axes, RIGHT, buff=0.5).shift(UP * 0.5)
        self.play(Write(converge))
        self.wait(1)

        # highlight final position
        final_label = MathTex(
            r"\theta^*", font_size=26, color=GREEN
        ).next_to(param_dot, DOWN, buff=0.15)
        self.play(Write(final_label))
        self.wait(2)

        # fade everything out
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
        self.wait(1)
