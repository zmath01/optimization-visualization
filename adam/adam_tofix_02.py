from manim import *
import numpy as np

class AdamOptimizerScene(Scene):
    def f(self, x):
        # Slightly rippled convex valley
        return (x - 2)**2 + 0.2 * np.sin(10 * (x - 2))
    
    def df(self, x):
        return 2 * (x - 2) + 2 * np.cos(10 * (x - 2))
    
    def construct(self):
        # -------------- Scene 1: Loss Landscape --------------
        # Axes and curve
        axes = Axes(
            x_range=[-3, 5, 1],
            y_range=[0, 6, 1],
            x_length=8,
            y_length=4.5,
            axis_config={"include_tip": False, "font_size": 20},
        ).to_edge(DOWN, buff=0.5)
        
        # Manually create labels to avoid Unicode issues in get_axis_labels
        x_label = MathTex(r"\theta", font_size=30).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = MathTex(r"\text{Loss}", font_size=30).next_to(axes.y_axis, LEFT, buff=0.3).rotate(PI/2)
        labels = VGroup(x_label, y_label)
        
        curve = axes.plot(lambda x: self.f(x), color=WHITE, stroke_width=3)
        self.play(Create(axes), Write(labels), Create(curve))
        
        # Title and update rule
        title = Text("Adam: Adaptive Moment Estimation", font_size=36).to_edge(UP)
        self.play(Write(title))
        update_rule = MathTex(
            r"\theta_t = \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
            font_size=28
        ).next_to(title, DOWN, buff=0.2)
        self.play(Write(update_rule))
        
        # Initial parameter dot (Adam)
        theta_tracker = ValueTracker(-2.0)  # start at x = -2
        adam_dot = always_redraw(
            lambda: Dot(
                axes.coords_to_point(theta_tracker.get_value(), self.f(theta_tracker.get_value())),
                color=BLUE, radius=0.1, z_index=5
            )
        )
        self.add(adam_dot)
        # Initial SGD dot (will animate separately)
        sgd_x = -2.0
        sgd_dot = Dot(
            axes.coords_to_point(sgd_x, self.f(sgd_x)),
            color=RED, radius=0.08, fill_opacity=0.6, z_index=4
        )
        self.add(sgd_dot)
        # Legend
        legend = VGroup(
            Dot(color=BLUE, radius=0.08).set_opacity(1),
            Text("Adam", font_size=20, color=BLUE),
            Dot(color=RED, radius=0.08).set_opacity(0.6),
            Text("SGD", font_size=20, color=RED)
        ).arrange(RIGHT, buff=0.3).to_corner(UR, buff=0.3).shift(DOWN*0.3)
        self.play(FadeIn(legend))
        self.wait(0.3)
        
        # Precompute trajectories
        alpha = 0.1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        m, v = 0.0, 0.0
        x = -2.0
        adam_path = [x]
        for t in range(1, 8):
            g = self.df(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
            adam_path.append(x)
        
        sgd_lr = 0.3
        sgd_path = [sgd_x]
        for _ in range(1, len(adam_path)):
            g = self.df(sgd_path[-1])
            sgd_path.append(sgd_path[-1] - sgd_lr * g)
        
        # -------------- Moment Bars Setup --------------
        bar_zero = axes.coords_to_point(-3, 0)  # reference line at bottom left
        # Create two bar containers (rectangles with height managed by updater)
        m_bar = Rectangle(
            width=0.3,
            height=0,
            fill_color=GREEN,
            fill_opacity=0.8,
            stroke_color=GREEN,
            stroke_width=1
        ).move_to(bar_zero, aligned_edge=DOWN)
        v_bar = Rectangle(
            width=0.3,
            height=0,
            fill_color=ORANGE,
            fill_opacity=0.8,
            stroke_color=ORANGE,
            stroke_width=1
        ).next_to(m_bar, RIGHT, buff=0.5)
        m_label = Text("m", font_size=20, color=GREEN).next_to(m_bar, UP, buff=0.1)
        v_label = Text("v", font_size=20, color=ORANGE).next_to(v_bar, UP, buff=0.1)
        # Use always_redraw for dynamic values (capture current_m, current_v from outer scope)
        current_m = 0.0
        current_v = 0.0
        m_value_tex = always_redraw(lambda: MathTex(f"{current_m:.3f}", font_size=18, color=GREEN).next_to(m_bar, DOWN, buff=0.1))
        v_value_tex = always_redraw(lambda: MathTex(f"{current_v:.3f}", font_size=18, color=ORANGE).next_to(v_bar, DOWN, buff=0.1))
        bar_group = VGroup(m_bar, v_bar, m_label, v_label, m_value_tex, v_value_tex)
        bar_group.to_edge(DOWN, buff=0.8).shift(LEFT*2.5)
        self.play(FadeIn(bar_group))
        
        # Gradient arrow object (recreated each step)
        grad_arrow = None
        
        max_bar_height = 1.5
        
        def update_bars(m_val, v_val):
            nonlocal current_m, current_v
            current_m, current_v = m_val, v_val
            m_height = np.clip(abs(m_val) * 0.8, 0.1, max_bar_height)
            v_height = np.clip(abs(v_val) * 0.5, 0.1, max_bar_height)
            self.play(
                m_bar.animate.stretch_to_fit_height(m_height, about_edge=DOWN),
                v_bar.animate.stretch_to_fit_height(v_height, about_edge=DOWN),
                run_time=0.3
            )
        
        # Display formulas at bottom
        formula_m = MathTex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t", font_size=26, color=GREEN)
        formula_v = MathTex(r"v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2", font_size=26, color=ORANGE)
        formula_bc = MathTex(
            r"\hat{m}_t = \frac{m_t}{1-\beta_1^t},\ \hat{v}_t = \frac{v_t}{1-\beta_2^t}",
            font_size=26, color=PURPLE
        )
        formulas = VGroup(formula_m, formula_v, formula_bc).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        formulas.to_edge(DOWN, buff=0.5).shift(RIGHT*2.5)
        
        # Show formulas gradually
        self.play(Write(formula_m))
        self.wait(0.2)
        self.play(Write(formula_v))
        self.wait(0.2)
        self.play(Write(formula_bc))
        self.wait(0.5)
        
        # Helper to show gradient arrow
        def show_gradient(x_val):
            nonlocal grad_arrow
            g_val = self.df(x_val)
            dot_pos = axes.coords_to_point(x_val, self.f(x_val))
            sign = -1 if g_val > 0 else 1
            step_x = 0.3 * min(1, abs(g_val) / 5.0)
            end_x = x_val + sign * step_x
            end_pos = axes.coords_to_point(end_x, self.f(end_x))
            if grad_arrow:
                self.remove(grad_arrow)
            grad_arrow = Arrow(start=dot_pos, end=end_pos, color=RED, buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.2)
            return grad_arrow
        
        # -------------- Step-by-step animation --------------
        for i in range(1, 7):  # i corresponds to t
            x_prev = adam_path[i-1]
            x_new = adam_path[i]
            g = self.df(x_prev)
            # Compute raw moments for this step
            if i == 1:
                m_raw = (1 - beta1) * g
                v_raw = (1 - beta2) * (g**2)
            else:
                m_raw = beta1 * current_m + (1 - beta1) * g
                v_raw = beta2 * current_v + (1 - beta2) * (g**2)
            
            # Bias correction
            m_hat = m_raw / (1 - beta1**i)
            v_hat = v_raw / (1 - beta2**i)
            
            # Animate gradient arrow
            new_grad_arrow = show_gradient(x_prev)
            self.play(Create(new_grad_arrow))
            self.wait(0.2)
            
            # Update bars with raw moments
            update_bars(m_raw, v_raw)
            
            # Highlight bias correction (flash)
            bc_flash = SurroundingRectangle(formula_bc, color=YELLOW, buff=0.1)
            self.play(Create(bc_flash))
            self.wait(0.3)
            self.play(FadeOut(bc_flash))
            
            # Show update direction and magnitude
            delta = alpha * m_hat / (np.sqrt(v_hat) + eps)
            update_arrow = Arrow(
                start=axes.coords_to_point(x_prev, self.f(x_prev)),
                end=axes.coords_to_point(x_new, self.f(x_new)),
                color=PURPLE, buff=0, stroke_width=5
            )
            delta_label = MathTex(f"\\Delta\\theta={delta:.3f}", font_size=20, color=PURPLE)
            delta_label.next_to(update_arrow, LEFT, buff=0.2)
            self.play(Create(update_arrow), Write(delta_label))
            self.wait(0.2)
            
            # Move Adam dot
            self.play(
                theta_tracker.animate.set_value(x_new),
                run_time=0.4
            )
            # Move SGD dot (parallel)
            sgd_new = sgd_path[i]
            self.play(
                sgd_dot.animate.move_to(axes.coords_to_point(sgd_new, self.f(sgd_new))),
                run_time=0.4
            )
            
            # Clean up arrows for next step
            self.remove(new_grad_arrow, update_arrow)
            self.play(FadeOut(delta_label))
            self.wait(0.3)
        
        # Final emphasis: convergence comparison
        final_text = Text("Adam converges smoothly, SGD oscillates", font_size=24, color=WHITE)
        final_text.next_to(legend, DOWN, buff=0.3).align_to(legend, LEFT)
        self.play(Write(final_text))
        self.wait(1)
