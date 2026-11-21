from manim import *


class CheapLearningScene(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        self.image_space()
        self.locality()
        self.coarse_graining()
        self.multiplication()

    def heading(self, text):
        title = Text(text, font_size=44).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

    def clear_panel(self):
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.45)

    def image_space(self):
        self.heading("Why can learning be cheap?")
        count = MathTex(r"256^{1{,}000{,}000}", font_size=78, color=GOLD_A).move_to(UP * 1.24)
        label = Text("Possible 1,000 × 1,000 grayscale images", font_size=30)
        label.next_to(count, DOWN, buff=0.32)
        contrast = VGroup(
            Text("Natural images have structure", font_size=35, color=BLUE_A),
            Text("Nearby pixels often carry related information.", font_size=26),
        ).arrange(DOWN, buff=0.34).move_to(DOWN * 2.06)
        self.play(Write(count), FadeIn(label))
        self.wait(1.5)
        self.play(FadeIn(contrast))
        self.wait(3)
        self.clear_panel()

    def locality(self):
        self.heading("Locality and shared interactions")
        grid = VGroup(*[Dot(radius=0.09, color=BLUE_A) for _ in range(16)])
        grid.arrange_in_grid(rows=4, cols=4, buff=0.57).move_to(LEFT * 3.95)
        edges = VGroup(*[
            Line(grid[4*i+j], grid[4*(i+di)+j+dj], buff=0.09, color=GOLD_A, stroke_width=3)
            for i in range(4) for j in range(4) for di, dj in [(1, 0), (0, 1)]
            if i + di < 4 and j + dj < 4
        ])
        math = VGroup(
            MathTex(r"H(\mathbf{x})=J\!\sum_{\langle i,j\rangle}x_i x_j", font_size=49),
            Text("Only neighboring sites interact.", font_size=28),
            Text("Each bond has the same coupling J.", font_size=28),
        ).arrange(DOWN, buff=0.42, aligned_edge=LEFT).move_to(RIGHT * 2.18)
        note = Text("A quarter-turn preserves this square lattice and its couplings.",
                    font_size=27, color=GREY_A).to_edge(DOWN, buff=0.59)
        lattice = VGroup(edges, grid)
        self.play(FadeIn(grid), Create(edges))
        self.play(FadeIn(math))
        self.play(Rotate(lattice, angle=PI / 2), FadeIn(note), run_time=1.6)
        self.wait(3)
        self.clear_panel()

    def coarse_graining(self):
        self.heading("A hierarchy of coarser descriptions")
        fine = VGroup()
        coarse = VGroup()
        for i in range(3):
            for j in range(3):
                color = [TEAL_A, BLUE_A, GREEN_A][(i + j) % 3]
                patch = VGroup(*[Dot(radius=0.075, color=color) for _ in range(4)])
                fine.add(patch.arrange_in_grid(rows=2, cols=2, buff=0.13))
                coarse.add(Dot(radius=0.16, color=color))
        fine.arrange_in_grid(rows=3, cols=3, buff=0.35).move_to(LEFT * 3.8)
        coarse.arrange_in_grid(rows=3, cols=3, buff=0.48).move_to(RIGHT * 3.8)
        arrow = Arrow(LEFT * 1.38, RIGHT * 1.38, buff=0, color=GOLD_A)
        operation = Text("Aggregate each block", font_size=25, color=GOLD_A).next_to(arrow, UP, buff=0.25)
        fine_label = Text("Fine scale", font_size=27).move_to([-3.8, -1.85, 0])
        coarse_label = Text("Coarse scale", font_size=27).move_to([3.8, -1.85, 0])
        note = Text("Coarse-graining in physics and pooling in neural networks share this idea.",
                    font_size=26, color=GREY_A).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(fine), FadeIn(fine_label))
        self.play(GrowArrow(arrow), FadeIn(operation))
        self.play(*[TransformFromCopy(patch, dot) for patch, dot in zip(fine, coarse)],
                  FadeIn(coarse_label), run_time=1.5)
        self.play(FadeIn(note))
        self.wait(3)
        self.clear_panel()

    def multiplication(self):
        self.heading("Depth can save neurons")
        gate = MathTex(r"uv=\frac{(u+v)^2-(u-v)^2}{4}", font_size=57, color=TEAL_A)
        gate.move_to(UP * 1.38)
        gate_note = Text("A two-input multiplication gate with square units.", font_size=27)
        gate_note.next_to(gate, DOWN, buff=0.28)
        self.play(Write(gate), FadeIn(gate_note))
        self.wait(2)

        # Two squares suffice for this identity. The separate width bound
        # assumes the generic activation specified in the paper's Appendix A.
        self.play(FadeOut(gate), FadeOut(gate_note))
        product = MathTex(r"x_1x_2\cdots x_n", font_size=58).move_to(UP * 1.87)
        deep = VGroup(
            Text("Tree of pairwise multiplications", font_size=28, color=TEAL_A),
            MathTex(r"O(n)\text{ neurons}", font_size=53, color=TEAL_A),
        ).arrange(DOWN, buff=0.3).move_to([-3.25, -0.14, 0])
        shallow = VGroup(
            Text("One hidden layer", font_size=28, color=GOLD_A),
            MathTex(r"2^n\text{ neurons}", font_size=53, color=GOLD_A),
        ).arrange(DOWN, buff=0.3).move_to([3.25, -0.14, 0])
        conditions = VGroup(
            Text("The shallow width bound uses the paper’s smooth activation assumptions.", font_size=26),
            MathTex(r"\sigma^{(k)}(0)\ne 0\quad(0\le k\le n)", color=GREY_A, font_size=32),
        ).arrange(DOWN, buff=0.22).to_edge(DOWN, buff=0.44)
        self.play(FadeIn(product))
        self.play(FadeIn(deep), FadeIn(shallow))
        self.play(FadeIn(conditions))
        self.wait(5)
