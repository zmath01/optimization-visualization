from manim import *


class ResNetVisualization(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        self.degradation()
        self.residual_block()
        self.gradient()
        self.projection()

    def heading(self, text):
        title = Text(text, font_size=44).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

    def clear_panel(self):
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.45)

    @staticmethod
    def block(label, color, width=2.6):
        box = RoundedRectangle(width=width, height=0.66, corner_radius=0.12,
                               color=color, fill_opacity=0.17)
        return VGroup(box, Text(label, font_size=24, color=color).move_to(box))

    def degradation(self):
        self.heading("ResNet  ·  The degradation problem")
        labels = VGroup(
            Text("Shallower plain network", font_size=27, color=BLUE_A).move_to(LEFT * 3.3 + UP * 1.75),
            Text("Deeper plain network", font_size=27, color=GOLD_A).move_to(RIGHT * 3.3 + UP * 1.75),
        )
        stacks = VGroup(*[
            VGroup(*[Rectangle(width=2.4, height=0.26, color=color, fill_opacity=0.24)
                     for _ in range(count)]).arrange(UP, buff=0.16).move_to([x, -0.08, 0])
            for count, color, x in [(3, BLUE_A, -3.3), (6, GOLD_A, 3.3)]
        ])
        caption = Text("The deeper plain models in the paper had higher training error.",
                       font_size=28).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(labels), Create(stacks))
        self.play(FadeIn(caption))
        self.wait(3)
        self.clear_panel()

    def residual_block(self):
        self.heading("Learn a residual around the identity")
        start = Dot([-4.95, 0, 0], color=BLUE_A, radius=0.09)
        source = MathTex(r"\mathbf{x}", color=BLUE_A).next_to(start, LEFT, buff=0.28)
        weights = self.block("Weight layers", GOLD_A).move_to([-2.35, 0, 0])
        add = Circle(radius=0.28, color=PURPLE_A).move_to([0.15, 0, 0])
        plus = MathTex("+", color=PURPLE_A, font_size=34).move_to(add)
        activation = self.block("ReLU", GREY_A, width=1.6).move_to([2.45, 0, 0])
        output = MathTex(r"\sigma(\mathbf{y})", font_size=36).move_to([4.55, 0, 0])
        branch = VGroup(
            Arrow(start.get_center(), weights.get_left(), buff=0.11, color=GOLD_A),
            Arrow(weights.get_right(), add.get_left(), buff=0.1, color=GOLD_A),
        )
        path = VGroup(
            Line(start.get_center(), [-4.95, 1.45, 0], color=BLUE_A),
            Line([-4.95, 1.45, 0], [0.15, 1.45, 0], color=BLUE_A),
            Arrow([0.15, 1.45, 0], add.get_top(), buff=0, color=BLUE_A),
        )
        skip_label = Text("Identity shortcut", font_size=25, color=BLUE_A)
        skip_label.next_to(path[1], UP, buff=0.19)
        forward = VGroup(
            Arrow(add.get_right(), activation.get_left(), buff=0.1, color=GREY_A),
            Arrow(activation.get_right(), output.get_left(), buff=0.1, color=GREY_A),
        )
        residual_label = MathTex(r"\mathcal F(\mathbf{x})", font_size=33, color=GOLD_A)
        residual_label.next_to(weights, DOWN, buff=0.28)
        equation = MathTex(r"\mathbf{y}=\mathcal F(\mathbf{x})+\mathbf{x}",
                           font_size=49).move_to(DOWN * 1.79)
        self.play(FadeIn(source), FadeIn(start), FadeIn(weights), Create(branch), FadeIn(residual_label))
        self.play(Create(path), FadeIn(skip_label), FadeIn(add), FadeIn(plus))
        self.play(Create(forward), FadeIn(activation), FadeIn(output), FadeIn(equation))
        self.wait(2)
        zero = MathTex(r"\mathcal F(\mathbf{x})\approx0\quad\Longrightarrow\quad\mathbf{y}\approx\mathbf{x}",
                       font_size=35, color=BLUE_A).move_to(DOWN * 2.75)
        self.play(FadeIn(zero), VGroup(weights, branch).animate.set_opacity(0.3))
        self.wait(2.5)
        self.clear_panel()

    def gradient(self):
        self.heading("The shortcut contributes an identity Jacobian")
        equation = MathTex(
            r"J_{\mathbf{y}}=", r"J_{\mathcal F}", "+", "I", font_size=63,
        ).move_to(UP * 1.06)
        equation[1].set_color(GOLD_A)
        equation[3].set_color(BLUE_A)
        chain = MathTex(
            r"\nabla_{\mathbf{x}}L=(J_{\mathcal F}+I)^{\mathsf T}\nabla_{\mathbf{y}}L",
            font_size=46,
        ).move_to(DOWN * 0.46)
        note = VGroup(
            Text("The identity term gives the gradient a direct route.", font_size=27),
            Text("This derivative is for the sum before the block activation.", font_size=24, color=GREY_B),
        ).arrange(DOWN, buff=0.24).move_to(DOWN * 2.42)
        self.play(Write(equation))
        self.play(Circumscribe(equation[3], color=BLUE_A, buff=0.16))
        self.play(FadeIn(chain), FadeIn(note))
        self.wait(4)
        self.clear_panel()

    def projection(self):
        self.heading("Project the shortcut when dimensions differ")
        specs = [
            ("Residual branch", r"\mathcal F(\mathbf{x})\in\mathbb{R}^{128}", GOLD_A, 1.08),
            ("Projection shortcut", r"W_s\mathbf{x}\in\mathbb{R}^{128}", BLUE_A, -0.68),
        ]
        branches = VGroup(*[
            VGroup(Text(label, font_size=26, color=color), MathTex(tex, font_size=39, color=color))
            .arrange(DOWN, buff=0.20).move_to([0, y, 0])
            for label, tex, color, y in specs
        ])
        source = MathTex(r"\mathbf{x}\in\mathbb{R}^{64}", font_size=39).move_to([-5.05, 0.20, 0])
        add = MathTex("+", font_size=55, color=PURPLE_A).move_to([4.7, 0.20, 0])
        links = VGroup(*[
            arrow
            for group, (_, _, color, _) in zip(branches, specs)
            for arrow in (
                Arrow(source.get_right(), group.get_left(), buff=0.19, color=color),
                Arrow(group.get_right(), add.get_left(), buff=0.19, color=color),
            )
        ])
        equation = MathTex(r"\mathbf{y}=\mathcal F(\mathbf{x})+W_s\mathbf{x}", font_size=48)
        equation.move_to(DOWN * 2.46)
        self.play(FadeIn(source), FadeIn(branches), Create(links), FadeIn(add))
        self.play(Write(equation))
        self.wait(4)
