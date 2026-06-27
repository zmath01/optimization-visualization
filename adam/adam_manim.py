from manim import *
import numpy as np

class AdamOptimizerScene(Scene):
    def construct(self):
        # Styling & Colors
        C_LOSS = LIGHT_GREY
        C_PARAM = BLUE
        C_GRAD = RED
        C_M = GREEN
        C_V = ORANGE
        C_UPDATE = PURPLE

        # ==========================================
        # Scene 1: The Loss Landscape & Initialization
        # ==========================================
        title = Text("Adam: A Method for Stochastic Optimization", font_size=36).to_edge(UP)
        self.play(Write(title))

        # 2D Convex Loss Function: f(x) = x^2
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 10, 2],
            x_length=8,
            y_length=5,
            axis_config={"color": WHITE}
        ).shift(DOWN * 0.5)
        
        loss_curve = axes.plot(lambda x: x**2, color=C_LOSS)
        
        # Initialization tracker for theta
        theta_tracker = ValueTracker(2.5)
        theta_dot = always_redraw(
            lambda: Dot(axes.c2p(theta_tracker.get_value(), theta_tracker.get_value()**2), color=C_PARAM)
        )

        self.play(Create(axes), Create(loss_curve))
        self.play(FadeIn(theta_dot))

        # Faint background formula
        bg_formula = MathTex(
            r"\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
            color=WHITE, fill_opacity=0.15
        ).scale(1.5).center()
        
        self.play(FadeIn(bg_formula))
        self.wait(1.5)

        # ==========================================
        # Scene 2: Computing the Gradient (g_t)
        # ==========================================
        def get_grad_vector():
            x = theta_tracker.get_value()
            y = x**2
            slope = 2 * x
            start_pt = axes.c2p(x, y)
            
            # Scale gradient representation for visual clarity
            dx = -0.5 if slope > 0 else 0.5
            dy = slope * dx
            end_pt = axes.c2p(x + dx, y + dy)
            return Arrow(start_pt, end_pt, color=C_GRAD, buff=0)

        grad_arrow = always_redraw(get_grad_vector)
        grad_label = MathTex("g_t", color=C_GRAD).next_to(grad_arrow, RIGHT)

        self.play(Create(grad_arrow), Write(grad_label))
        self.wait(1.5)

        # ==========================================
        # Scene 3: First Moment Estimate (Momentum)
        # ==========================================
        self.play(FadeOut(title), FadeOut(bg_formula))
        
        m_formula = MathTex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t", color=C_M).to_edge(UP)
        self.play(Write(m_formula))

        # Visualizing momentum as a smoothed, longer vector pointing towards minimum
        m_arrow = Arrow(
            axes.c2p(2.5, 2.5**2),
            axes.c2p(2.5 - 0.8, 2.5**2 - 0.8*(2*2.5)),
            color=C_M, buff=0
        )
        m_label = MathTex("m_t", color=C_M).next_to(m_arrow, LEFT)
        
        self.play(
            TransformFromCopy(grad_arrow, m_arrow),
            Write(m_label)
        )
        
        beta1_text = Text("Memory factor: ", font_size=24).next_to(m_formula, DOWN)
        beta1_val = MathTex(r"\beta_1 = 0.9", color=C_M, font_size=36).next_to(beta1_text, RIGHT)
        self.play(Write(beta1_text), Write(beta1_val))
        self.wait(1.5)

        # ==========================================
        # Scene 4: Second Raw Moment Estimate (Scaling)
        # ==========================================
        v_formula = MathTex(r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2", color=C_V).next_to(beta1_text, DOWN, buff=0.5)
        self.play(Write(v_formula))

        # Magnitude indicator bar for v_t
        v_bar = Rectangle(height=1.0, width=0.4, color=C_V, fill_opacity=0.8).to_corner(UR).shift(DOWN)
        v_label = MathTex("v_t", color=C_V).next_to(v_bar, DOWN)
        
        self.play(FadeIn(v_bar), Write(v_label))
        
        beta2_text = Text("Scaling factor: ", font_size=24).next_to(v_formula, DOWN)
        beta2_val = MathTex(r"\beta_2 = 0.999", color=C_V, font_size=36).next_to(beta2_text, RIGHT)
        self.play(Write(beta2_text), Write(beta2_val))
        self.wait(1.5)

        # ==========================================
        # Scene 5: Bias Correction
        # ==========================================
        self.play(
            FadeOut(m_formula), FadeOut(v_formula),
            FadeOut(beta1_text), FadeOut(beta1_val),
            FadeOut(beta2_text), FadeOut(beta2_val)
        )

        m_hat_formula = MathTex(r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}", color=C_M).to_edge(UP).shift(LEFT * 2)
        v_hat_formula = MathTex(r"\hat{v}_t = \frac{v_t}{1 - \beta_2^t}", color=C_V).next_to(m_hat_formula, RIGHT, buff=1.5)

        self.play(Write(m_hat_formula), Write(v_hat_formula))

        fix_label = Text("Fixing Initialization Bias", font_size=28, color=YELLOW).next_to(m_hat_formula, DOWN, buff=0.5).shift(RIGHT * 1.5)
        self.play(Write(fix_label))

        # Animate scaling up (bias correction dynamically increasing early magnitudes)
        m_hat_arrow = Arrow(
            axes.c2p(2.5, 2.5**2),
            axes.c2p(2.5 - 1.2, 2.5**2 - 1.2*(2*2.5)),
            color=C_M, buff=0
        )
        v_hat_bar = Rectangle(height=2.0, width=0.4, color=C_V, fill_opacity=0.8).move_to(v_bar).align_to(v_bar, DOWN)

        self.play(
            Transform(m_arrow, m_hat_arrow),
            Transform(v_bar, v_hat_bar)
        )
        self.wait(1.5)

        # ==========================================
        # Scene 6: The Parameter Update
        # ==========================================
        self.play(
            FadeOut(m_hat_formula), FadeOut(v_hat_formula), FadeOut(fix_label),
            FadeOut(m_arrow), FadeOut(m_label), FadeOut(grad_arrow), FadeOut(grad_label),
            FadeOut(v_bar), FadeOut(v_label)
        )

        update_formula = MathTex(r"\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}", color=C_UPDATE).to_edge(UP)
        self.play(Write(update_formula))

        # Math parameters for simulated convergence
        alpha = 0.5
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        m = 0
        v = 0
        theta_val = 2.5
        
        path_lines = VGroup()

        # Iterative update steps along the curve
        for t in range(1, 15):
            g = 2 * theta_val 
            
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g**2)
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            step = alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            new_theta_val = theta_val - step

            line = Line(
                axes.c2p(theta_val, theta_val**2),
                axes.c2p(new_theta_val, new_theta_val**2),
                color=C_UPDATE, stroke_width=4
            )
            path_lines.add(line)

            self.play(
                theta_tracker.animate.set_value(new_theta_val),
                Create(line),
                run_time=0.4
            )
            theta_val = new_theta_val

        self.wait(2)
