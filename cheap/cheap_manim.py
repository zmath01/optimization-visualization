from manim import *

class CheapLearningScene(Scene):
    def construct(self):
        # Configuration
        C_HARD = RED
        C_EASY = BLUE
        C_WEIGHTS = GOLD
        C_RG = GREEN
        C_HIERARCHY = PURPLE

        # ==========================================
        # SCENE 1: The "Swindle" (Exponential vs Linear)
        # ==========================================
        title_1 = Text("1. The Combinatorial Swindle").to_edge(UP)
        self.play(Write(title_1))

        # Exponential Explosion Space
        all_funcs = MathTex(r"256^{1,000,000}", color=C_HARD).scale(2.5)
        all_funcs_label = Text("All Possible Images (Pure Chaos)", color=C_HARD).next_to(all_funcs, DOWN)
        self.play(FadeIn(all_funcs, shift=UP), Write(all_funcs_label))
        self.wait(1.5)

        # Collapse to Physical Space
        phys_dot = Dot(color=C_EASY).scale(2)
        phys_label = Text("Images of the Physical World", color=C_EASY).next_to(phys_dot, DOWN)
        self.play(
            ReplacementTransform(all_funcs, phys_dot),
            ReplacementTransform(all_funcs_label, phys_label)
        )
        self.wait(1)

        # Equations
        exp_eq = MathTex(r"N_{params} \approx v^n", color=C_HARD).scale(1.5)
        lin_eq = MathTex(r"N_{params} \approx v \times n", color=C_EASY).scale(1.5)
        
        self.play(FadeOut(phys_dot), FadeOut(phys_label))
        self.play(Write(exp_eq))
        self.wait(1)
        self.play(
            Transform(exp_eq, lin_eq),
            Circumscribe(lin_eq, color=C_WEIGHTS, run_time=2)
        )
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # ==========================================
        # SCENE 2: Physics Properties (Locality)
        # ==========================================
        title_2 = Text("2. Locality & Symmetry").to_edge(UP)
        self.play(Write(title_2))

        # Grid of pixels
        grid = VGroup(*[Dot(color=C_EASY, radius=0.1) for _ in range(16)]).arrange_in_grid(rows=4, cols=4, buff=0.8)
        self.play(Create(grid))

        # Locality: Local connections
        local_edges = VGroup()
        for i in range(4):
            for j in range(4):
                if i < 3: local_edges.add(Line(grid[i*4+j], grid[(i+1)*4+j], color=C_WEIGHTS, stroke_width=2))
                if j < 3: local_edges.add(Line(grid[i*4+j], grid[i*4+j+1], color=C_WEIGHTS, stroke_width=2))
        
        self.play(Create(local_edges), run_time=2)
        
        # Hamiltonian Equation
        hamiltonian = MathTex(r"H(x) = \sum h_i x_i + \sum h_{ij} x_i x_j + \dots").to_edge(DOWN)
        self.play(Write(hamiltonian))
        self.wait(1.5)

        # Symmetry Rotation
        sym_text = Text("Symmetry Invariant", color=C_EASY, font_size=24).next_to(grid, UP)
        self.play(FadeIn(sym_text, shift=DOWN))
        self.play(Rotate(grid, angle=PI/2), Rotate(local_edges, angle=PI/2), run_time=2)
        self.wait(1.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # ==========================================
        # SCENE 3: Compositionality & Depth
        # ==========================================
        title_3 = Text("3. Hierarchical Compositionality").to_edge(UP)
        self.play(Write(title_3))

        # Helper to create layers
        def get_layer(size, spacing):
            return VGroup(*[Dot(color=C_HIERARCHY) for _ in range(size)]).arrange(RIGHT, buff=spacing)

        def connect(l1, l2, color=GRAY):
            return VGroup(*[Line(n1, n2, color=color, stroke_width=1, stroke_opacity=0.5) for n1 in l1 for n2 in l2])

        # Deep Network (Hierarchical)
        l1 = get_layer(1, 1).shift(UP * 1.5 + LEFT * 3)
        l2 = get_layer(2, 1).shift(UP * 0 + LEFT * 3)
        l3 = get_layer(4, 0.5).shift(DOWN * 1.5 + LEFT * 3)
        
        deep_edges = VGroup(connect(l1, l2, C_WEIGHTS), connect(l2, l3, C_WEIGHTS))
        deep_net = VGroup(l1, l2, l3, deep_edges)
        deep_label = Text("Deep (Sparse & Structured)", font_size=24).next_to(deep_net, DOWN)

        # Shallow Network (Flat)
        s1 = get_layer(1, 1).shift(UP * 1.5 + RIGHT * 3)
        s2 = get_layer(8, 0.3).shift(DOWN * 1.5 + RIGHT * 3)
        shallow_edges = connect(s1, s2, C_HARD)
        shallow_net = VGroup(s1, s2, shallow_edges)
        shallow_label = Text("Shallow (Dense & Tangled)", font_size=24).next_to(shallow_net, DOWN)

        self.play(Create(deep_net), Write(deep_label))
        self.wait(1)
        self.play(Create(shallow_net), Write(shallow_label))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # ==========================================
        # SCENE 4: The Renormalization Group (RG)
        # ==========================================
        title_4 = Text("4. The Renormalization Group (RG)").to_edge(UP)
        self.play(Write(title_4))

        # High-res lattice
        lattice_fine = VGroup(*[Dot(color=WHITE, radius=0.08) for _ in range(36)]).arrange_in_grid(rows=6, cols=6, buff=0.4).shift(LEFT * 3)
        
        # Coarse-grained lattice
        lattice_coarse = VGroup(*[Dot(color=C_RG, radius=0.15) for _ in range(9)]).arrange_in_grid(rows=3, cols=3, buff=0.8).shift(RIGHT * 3)

        flow_arrow = Arrow(start=LEFT, end=RIGHT, color=C_RG).scale(1.5)
        flow_math = MathTex(r"J \to J'", color=C_RG).next_to(flow_arrow, UP)

        self.play(Create(lattice_fine))
        self.wait(1)
        
        # Animate Decimation / Pooling
        self.play(GrowArrow(flow_arrow), Write(flow_math))
        self.play(TransformFromCopy(lattice_fine, lattice_coarse))
        
        rg_label = Text("Physics: Coarse-graining", font_size=24).next_to(lattice_coarse, DOWN)
        cnn_label = Text("Deep Learning: CNN Pooling", font_size=24, color=C_WEIGHTS).next_to(lattice_coarse, DOWN)
        
        self.play(Write(rg_label))
        self.wait(1.5)
        self.play(Transform(rg_label, cnn_label))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # ==========================================
        # SCENE 5: No-Flattening Theorem
        # ==========================================
        title_5 = Text("5. The No-Flattening Theorem").to_edge(UP)
        self.play(Write(title_5))

        task_text = MathTex(r"\text{Task: } x_1 \times x_2 \times \dots \times x_n").shift(UP * 2)
        self.play(Write(task_text))

        # Deep comparison
        deep_res = MathTex(r"\text{Deep Network: } \approx 2n \text{ Neurons}", color=C_EASY).shift(UP * 0.5)
        self.play(FadeIn(deep_res, shift=UP))
        self.wait(1)

        # Shallow explosion
        shallow_res = MathTex(r"\text{Shallow Network: } \ge 2^n \text{ Neurons}", color=C_HARD).shift(DOWN * 0.5)
        self.play(FadeIn(shallow_res, shift=UP))
        
        # Exploding animation effect
        explosion_box = SurroundingRectangle(shallow_res, color=C_HARD, buff=0.2)
        self.play(Create(explosion_box), shallow_res.animate.scale(1.2))
        
        conclusion = Text("Multiplication fundamentally requires depth.", font_size=32, color=C_WEIGHTS).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(3)
