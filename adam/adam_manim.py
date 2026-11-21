from math import sqrt

from manim import *


class AdamOptimizerScene(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        heading = Text("Adam optimizer", font_size=46).to_edge(UP, buff=0.4)
        subtitle = Text("A one-dimensional example", font_size=24, color=GREY_B)
        subtitle.next_to(heading, DOWN, buff=0.18)
        self.play(FadeIn(heading), FadeIn(subtitle))

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 9, 3],
            x_length=5.5,
            y_length=4.0,
            tips=False,
            axis_config={"color": GREY_B, "include_numbers": True, "font_size": 19},
        ).move_to(LEFT * 3.55 + DOWN * 0.3)
        curve = axes.plot(lambda x: x**2, color=TEAL_A)
        x_label = MathTex(r"\theta", font_size=30).next_to(axes.x_axis, RIGHT, buff=0.14)
        loss_label = MathTex(r"f(\theta)=\theta^2", font_size=32, color=TEAL_A)
        loss_label.move_to([-3.55, 2.48, 0])

        theta = ValueTracker(2.5)
        dot = Dot(color=YELLOW, radius=0.10).set_z_index(2)

        def follow_theta(mob):
            value = theta.get_value()
            mob.move_to(axes.c2p(value, value**2))

        dot.add_updater(follow_theta, call_updater=True)
        self.play(Create(axes), Create(curve), FadeIn(x_label), FadeIn(loss_label))
        self.play(FadeIn(dot))

        def panel(section, *rows):
            header = Text(section, font_size=31, color=BLUE_A)
            block = VGroup(header, *rows).arrange(DOWN, buff=0.36, aligned_edge=LEFT)
            block.move_to(RIGHT * 0.28 + UP * 2.05, aligned_edge=UL)
            return block

        gradient = panel(
            "Gradient at the current point",
            MathTex(r"g_t=2\theta_{t-1}", font_size=44),
            MathTex(r"\theta_0=2.5,\quad g_1=5", font_size=38, color=YELLOW),
        )
        self.play(FadeIn(gradient))
        self.wait(2)

        moments = panel(
            "Track two moving averages",
            MathTex(r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t", font_size=37, color=GREEN_A),
            Text("First moment  ·  signed gradient", font_size=23),
            MathTex(r"v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2", font_size=37, color=GOLD_A),
            Text("Second raw moment  ·  squared gradient", font_size=23),
            MathTex(r"\beta_1=0.9,\quad \beta_2=0.999", font_size=30, color=GREY_B),
        )
        self.play(FadeOut(gradient))
        self.play(FadeIn(moments))
        self.wait(3)

        bias = panel(
            "Correct the initial bias",
            MathTex(r"m_0=v_0=0", font_size=38, color=GREY_B),
            MathTex(r"\hat m_t=\frac{m_t}{1-\beta_1^t}", font_size=42, color=GREEN_A),
            MathTex(r"\hat v_t=\frac{v_t}{1-\beta_2^t}", font_size=42, color=GOLD_A),
            MathTex(r"\hat m_1=g_1,\qquad \hat v_1=g_1^2", font_size=32),
        )
        self.play(FadeOut(moments))
        self.play(FadeIn(bias))
        self.wait(3)

        update = panel(
            "Update the parameter",
            MathTex(
                r"\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}",
                font_size=42,
                color=PURPLE_A,
            ),
            MathTex(r"\alpha=0.5,\quad\epsilon=10^{-8}", font_size=34),
            Text("Large step size chosen to show overshoot", font_size=22, color=GREY_B),
        )
        readout = VGroup(
            Text("Step", font_size=25),
            Integer(0, font_size=34, color=YELLOW),
        ).arrange(RIGHT, buff=0.22).next_to(update, DOWN, buff=0.48, aligned_edge=LEFT)
        self.play(FadeOut(bias))
        self.play(FadeIn(update), FadeIn(readout))
        self.wait(1)

        m = v = 0.0
        for t in range(1, 15):
            current = theta.get_value()
            g = 2 * current
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)
            next_theta = current - 0.5 * m_hat / (sqrt(v_hat) + 1e-8)

            marker = Dot(axes.c2p(current, current**2), color=YELLOW, radius=0.036)
            self.add(marker)
            self.play(theta.animate.set_value(next_theta), run_time=0.53)
            readout[1].set_value(t)

        self.wait(2)
