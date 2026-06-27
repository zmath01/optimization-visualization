from manim import *
import numpy as np

class CheapLearningScene(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        
        # Scene 1: The Combinatorial Swindle
        self.scene_swindle()
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)
        
        # Scene 2: Physics Properties (Symmetry & Locality)
        self.scene_physics()
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)
        
        # Scene 3: Compositionality & Depth
        self.scene_compositionality()
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)
        
        # Scene 4: The Renormalization Group (RG) Connection
        self.scene_rg()
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)
        
        # Scene 5: No-Flattening Theorem (Multiplication)
        self.scene_no_flattening()
        self.wait(2)

    # ========== SCENE 1: THE SWINDLE ==========
    def scene_swindle(self):
        # Title
        title = Text("The Combinatorial Swindle", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Massive possible space: 256^1,000,000
        huge_number = MathTex("256^{1,000,000}", font_size=48, color=RED)
        huge_number.move_to(UP)
        self.play(Write(huge_number))
        self.wait(0.5)
        
        # Label: "All possible images"
        all_label = Text("All Possible Functions", font_size=24, color=RED).next_to(huge_number, DOWN, buff=0.3)
        self.play(Write(all_label))
        self.wait(0.5)
        
        # Transform into a tiny dot (physical images)
        dot = Dot(color=BLUE, radius=0.15).move_to(ORIGIN)
        self.play(Transform(huge_number, dot), FadeOut(all_label))
        self.wait(0.3)
        
        # Physical label
        physical_label = Text("Physical Data (Order)", font_size=24, color=BLUE).next_to(dot, DOWN, buff=0.3)
        self.play(Write(physical_label))
        self.wait(0.5)
        self.play(FadeOut(dot), FadeOut(physical_label))
        
        # Generic parameter formula: N_params ~ v^n
        generic_formula = MathTex(r"N_{\text{params}} \approx v^n", font_size=40, color=RED)
        generic_formula.next_to(title, DOWN, buff=0.8)
        self.play(Write(generic_formula))
        self.wait(0.5)
        
        # Transform to linear scaling: N_params ~ v * n
        cheap_formula = MathTex(r"N_{\text{params}} \approx v \times n", font_size=40, color=BLUE)
        cheap_formula.move_to(generic_formula.get_center())
        self.play(Transform(generic_formula, cheap_formula))
        self.wait(0.5)
        
        # Add "The Combinatorial Swindle" as label
        swindle_label = Text("The Combinatorial Swindle", font_size=28, color=YELLOW)
        swindle_label.next_to(cheap_formula, DOWN, buff=0.5)
        self.play(Write(swindle_label))
        self.wait(0.8)
        
        # Animate network growing linearly while function space grows exponentially (concept)
        # Show a small network (3 layers) on left, and a huge number of functions on right expanding.
        # For simplicity, use text.
        left_net = Text("Linear growth\nof parameters", font_size=24, color=GOLD).to_edge(LEFT, buff=1)
        right_exp = Text("Exponential growth\nof representable functions", font_size=24, color=PURPLE).to_edge(RIGHT, buff=1)
        self.play(Write(left_net), Write(right_exp))
        self.wait(1)
        self.play(FadeOut(cheap_formula), FadeOut(swindle_label), FadeOut(left_net), FadeOut(right_exp))

    # ========== SCENE 2: PHYSICS PROPERTIES ==========
    def scene_physics(self):
        title = Text("Physics of Data: Symmetry & Locality", font_size=34, color=WHITE).to_edge(UP)
        self.play(Write(title))
        
        # Grid of pixels (4x4)
        grid = VGroup(*[Square(side_length=0.6, fill_color=BLUE, fill_opacity=0.3, stroke_color=WHITE) 
                        for _ in range(16)]).arrange_in_grid(4, 4, buff=0.1)
        grid.move_to(LEFT * 3 + DOWN * 0.5)
        pixel_labels = VGroup(*[MathTex(f"x_{{{i+1}}}", font_size=20, color=WHITE).move_to(sq) 
                                 for i, sq in enumerate(grid)])
        self.play(Create(grid), Write(pixel_labels))
        
        # Highlight locality: show connections between neighbors (sparse)
        # Draw lines between neighbors
        neighbor_lines = VGroup()
        for i in range(4):
            for j in range(4):
                idx = i*4 + j
                # right neighbor
                if j < 3:
                    line = Line(grid[idx].get_right(), grid[idx+1].get_left(), color=GOLD, stroke_width=2)
                    neighbor_lines.add(line)
                # down neighbor
                if i < 3:
                    line = Line(grid[idx].get_bottom(), grid[idx+4].get_top(), color=GOLD, stroke_width=2)
                    neighbor_lines.add(line)
        self.play(Create(neighbor_lines, lag_ratio=0.1))
        
        # Label "Locality"
        locality_text = Text("Locality: only neighboring pixels interact", font_size=22, color=GOLD)
        locality_text.next_to(grid, DOWN, buff=0.5).align_to(grid, LEFT)
        self.play(Write(locality_text))
        
        # Show long-range connections fading out
        long_lines = VGroup()
        # a few far away pairs
        far_pairs = [(0, 15), (3, 12), (1, 14)]
        for a, b in far_pairs:
            line = Line(grid[a].get_center(), grid[b].get_center(), color=RED, stroke_width=2, stroke_opacity=0.7)
            long_lines.add(line)
        self.play(Create(long_lines))
        self.wait(0.5)
        self.play(long_lines.animate.set_stroke(opacity=0.1))
        
        # Hamiltonian expansion on the right
        hamiltonian = MathTex(r"H(x) = \sum h_i x_i + \sum h_{ij} x_i x_j + \dots",
                              font_size=30, color=WHITE)
        hamiltonian.move_to(RIGHT * 2.5 + DOWN * 1.0)
        self.play(Write(hamiltonian))
        self.wait(0.5)
        
        # Show that most coefficients vanish due to locality: cross out higher-order terms
        # Create a new MathTex with zeros.
        local_ham = MathTex(r"H(x) \approx \sum h_i x_i + \sum_{\text{neighbors}} h_{ij} x_i x_j",
                            font_size=30, color=GOLD)
        local_ham.move_to(hamiltonian.get_center())
        self.play(Transform(hamiltonian, local_ham))
        self.wait(0.5)
        
        # Symmetry: rotate grid and show invariance
        sym_label = Text("Symmetry: Rotational Invariance", font_size=22, color=BLUE).next_to(locality_text, DOWN, buff=0.3)
        self.play(Write(sym_label))
        
        # Rotate the grid (animate rotation)
        self.play(Rotate(grid, angle=PI/2, about_point=grid.get_center()), 
                  Rotate(pixel_labels, angle=PI/2, about_point=grid.get_center()),
                  Rotate(neighbor_lines, angle=PI/2, about_point=grid.get_center()),
                  run_time=1)
        # Show output probability staying flat (a flat line graph)
        axes = Axes(x_range=[0, 5], y_range=[0, 1], x_length=2, y_length=1.5,
                    axis_config={"include_numbers": False}).to_edge(RIGHT, buff=0.5).shift(DOWN*0.5)
        flat_line = axes.plot(lambda x: 0.5, color=BLUE)
        prob_label = MathTex("p(y|x)", font_size=24, color=BLUE).next_to(axes, UP, buff=0.1)
        self.play(Create(axes), Create(flat_line), Write(prob_label))
        self.wait(1)

    # ========== SCENE 3: COMPOSITIONALITY & DEPTH ==========
    def scene_compositionality(self):
        title = Text("Depth = Hierarchical Compositionality", font_size=34, color=WHITE).to_edge(UP)
        self.play(Write(title))
        
        # Hierarchical tree: Face -> parts -> edges -> pixels
        root = Text("Face", font_size=28, color=PURPLE).move_to(UP*2)
        left = Text("Eyes", font_size=24, color=BLUE).next_to(root, DOWN + LEFT*0.5, buff=0.6)
        right = Text("Nose", font_size=24, color=BLUE).next_to(root, DOWN + RIGHT*0.5, buff=0.6)
        edges1 = Text("Edges", font_size=20, color=GOLD).next_to(left, DOWN, buff=0.6).shift(LEFT*0.3)
        edges2 = Text("Edges", font_size=20, color=GOLD).next_to(right, DOWN, buff=0.6).shift(RIGHT*0.3)
        pixels1 = Text("Pixels", font_size=20, color=GRAY).next_to(edges1, DOWN, buff=0.5)
        pixels2 = Text("Pixels", font_size=20, color=GRAY).next_to(edges2, DOWN, buff=0.5)
        
        tree = VGroup(root, left, right, edges1, edges2, pixels1, pixels2)
        tree.arrange_in_grid(rows=4, cols=2, cell_alignment=ORIGIN).scale(0.7).center().shift(UP*0.5)
        # Adjust positions manually for a nicer tree look
        tree.remove(root, left, right, edges1, edges2, pixels1, pixels2)
        root.move_to(UP*2)
        left.move_to(LEFT*1.5 + UP*1)
        right.move_to(RIGHT*1.5 + UP*1)
        edges1.move_to(LEFT*1.5 + DOWN*0.2)
        edges2.move_to(RIGHT*1.5 + DOWN*0.2)
        pixels1.move_to(LEFT*1.5 + DOWN*1.5)
        pixels2.move_to(RIGHT*1.5 + DOWN*1.5)
        
        connections = VGroup(
            Line(root.get_bottom(), left.get_top(), color=PURPLE, stroke_width=3),
            Line(root.get_bottom(), right.get_top(), color=PURPLE, stroke_width=3),
            Line(left.get_bottom(), edges1.get_top(), color=BLUE, stroke_width=3),
            Line(right.get_bottom(), edges2.get_top(), color=BLUE, stroke_width=3),
            Line(edges1.get_bottom(), pixels1.get_top(), color=GOLD, stroke_width=3),
            Line(edges2.get_bottom(), pixels2.get_top(), color=GOLD, stroke_width=3),
        )
        self.play(Write(tree), Create(connections))
        self.wait(0.8)
        
        # Deep Network mirroring hierarchy
        deep_net_text = Text("Deep Network", font_size=24, color=GOLD).next_to(tree, RIGHT, buff=1.5).align_to(tree, UP)
        self.play(Write(deep_net_text))
        # Stack of rectangles representing layers
        deep_layers = VGroup(*[
            Rectangle(width=1, height=0.4, fill_color=GOLD, fill_opacity=0.6, stroke_color=WHITE)
            for _ in range(4)
        ]).arrange(DOWN, buff=0.3).next_to(deep_net_text, DOWN, buff=0.3)
        self.play(Create(deep_layers))
        
        # Shallow Network: tangled mess
        shallow_text = Text("Shallow Network", font_size=24, color=RED).next_to(deep_net_text, DOWN, buff=2.2)
        shallow_rect = Rectangle(width=3, height=0.8, fill_color=RED, fill_opacity=0.4, stroke_color=RED)
        shallow_rect.next_to(shallow_text, DOWN, buff=0.3)
        # Tangled lines: draw many crossing lines inside
        tangled_lines = VGroup()
        for _ in range(8):
            start = shallow_rect.get_left() + UP*np.random.uniform(-0.3, 0.3) + RIGHT*np.random.uniform(0, 0.3)
            end = shallow_rect.get_right() + UP*np.random.uniform(-0.3, 0.3) + LEFT*np.random.uniform(0, 0.3)
            line = Line(start, end, stroke_color=RED, stroke_width=2, stroke_opacity=0.6)
            tangled_lines.add(line)
        self.play(Write(shallow_text), Create(shallow_rect), Create(tangled_lines))
        self.wait(1)

    # ========== SCENE 4: RENORMALIZATION GROUP ==========
    def scene_rg(self):
        title = Text("Deep Learning = Variational RG", font_size=34, color=WHITE).to_edge(UP)
        self.play(Write(title))
        
        # Lattice of spins (4x4 dots)
        lattice = VGroup(*[Dot(radius=0.1, color=GREEN) for _ in range(16)]).arrange_in_grid(4,4,buff=0.6).move_to(LEFT*3.5)
        for dot in lattice:
            dot.set_fill(GREEN, opacity=1)
        self.play(Create(lattice))
        label = Text("Scale L", font_size=24, color=GREEN).next_to(lattice, DOWN, buff=0.3)
        self.play(Write(label))
        
        # Decimation: group 4 spins into 1 effective spin
        # Show green boxes grouping 2x2
        groups = VGroup()
        for i in range(0,4,2):
            for j in range(0,4,2):
                indices = [i*4+j, i*4+j+1, (i+1)*4+j, (i+1)*4+j+1]
                box = Polygon(
                    lattice[indices[0]].get_center() + LEFT*0.3 + UP*0.3,
                    lattice[indices[1]].get_center() + RIGHT*0.3 + UP*0.3,
                    lattice[indices[3]].get_center() + RIGHT*0.3 + DOWN*0.3,
                    lattice[indices[2]].get_center() + LEFT*0.3 + DOWN*0.3,
                    color=YELLOW, stroke_width=2, fill_opacity=0
                )
                groups.add(box)
        self.play(Create(groups))
        self.wait(0.5)
        
        # Coarse-grain: create new lattice with 4 effective spins
        coarse_lattice = VGroup(*[Dot(radius=0.15, color=YELLOW) for _ in range(4)]).arrange_in_grid(2,2,buff=0.6)
        coarse_lattice.move_to(LEFT*3.5 + DOWN*2)
        coarse_label = Text("Scale L/2", font_size=24, color=YELLOW).next_to(coarse_lattice, DOWN, buff=0.3)
        self.play(TransformFromCopy(lattice, coarse_lattice), Write(coarse_label))
        self.wait(0.5)
        
        # Hamiltonian parameters flowing: J -> J'
        flow_arrow = Arrow(
            start=LEFT*1 + UP*0.5,
            end=LEFT*1 + DOWN*1.5,
            color=GREEN, stroke_width=4
        )
        j_label = MathTex("J", font_size=30, color=GREEN).next_to(flow_arrow, LEFT, buff=0.2)
        j_prime = MathTex("J'", font_size=30, color=YELLOW).next_to(coarse_lattice, LEFT, buff=0.2)
        self.play(GrowArrow(flow_arrow), Write(j_label), Write(j_prime))
        self.wait(0.5)
        
        # Morph into CNN pooling: Stride-2 convolution
        cnn_box = Rectangle(width=3, height=1.5, color=BLUE, stroke_width=3).to_edge(RIGHT, buff=0.5).shift(UP*1)
        cnn_label = Text("CNN Pooling\n(Stride 2)", font_size=24, color=BLUE).move_to(cnn_box)
        self.play(TransformFromCopy(coarse_lattice, cnn_box), Write(cnn_label))
        self.wait(0.5)
        # Show arrow from RG to CNN
        morph_arrow = Arrow(
            start=LEFT*1.5,
            end=RIGHT*1.5,
            color=PURPLE, stroke_width=4
        )
        morph_text = Text("Variational RG", font_size=24, color=PURPLE).next_to(morph_arrow, UP, buff=0.1)
        self.play(GrowArrow(morph_arrow), Write(morph_text))
        self.wait(1)

    # ========== SCENE 5: NO-FLATTENING THEOREM ==========
    def scene_no_flattening(self):
        title = Text("The No-Flattening Theorem: Multiplication", font_size=32, color=WHITE).to_edge(UP)
        self.play(Write(title))
        
        # Task: multiply n variables
        task = MathTex(r"x_1 \times x_2 \times \dots \times x_n", font_size=36, color=WHITE)
        task.next_to(title, DOWN, buff=0.5)
        self.play(Write(task))
        
        # Deep network: binary tree multiplication (log depth)
        deep_label = Text("Deep Net (log depth)", font_size=24, color=GOLD).next_to(task, DOWN, buff=1).align_to(LEFT)
        # Draw binary tree of multiplication nodes
        node_radius = 0.15
        root = Circle(radius=node_radius, fill_color=GOLD, fill_opacity=0.8, stroke_color=WHITE).move_to(DOWN*1.5 + LEFT*1)
        left_child = Circle(radius=node_radius, fill_color=GOLD, fill_opacity=0.8, stroke_color=WHITE).move_to(DOWN*2 + LEFT*2)
        right_child = Circle(radius=node_radius, fill_color=GOLD, fill_opacity=0.8, stroke_color=WHITE).move_to(DOWN*2 + LEFT*0)
        leaf1 = Circle(radius=node_radius, fill_color=BLUE, fill_opacity=0.8).move_to(DOWN*3 + LEFT*3)
        leaf2 = Circle(radius=node_radius, fill_color=BLUE, fill_opacity=0.8).move_to(DOWN*3 + LEFT*2)
        leaf3 = Circle(radius=node_radius, fill_color=BLUE, fill_opacity=0.8).move_to(DOWN*3 + LEFT*1)
        leaf4 = Circle(radius=node_radius, fill_color=BLUE, fill_opacity=0.8).move_to(DOWN*3 + LEFT*0)
        lines = VGroup(
            Line(root.get_bottom(), left_child.get_top(), color=GOLD),
            Line(root.get_bottom(), right_child.get_top(), color=GOLD),
            Line(left_child.get_bottom(), leaf1.get_top(), color=GOLD),
            Line(left_child.get_bottom(), leaf2.get_top(), color=GOLD),
            Line(right_child.get_bottom(), leaf3.get_top(), color=GOLD),
            Line(right_child.get_bottom(), leaf4.get_top(), color=GOLD),
        )
        deep_tree = VGroup(root, left_child, right_child, leaf1, leaf2, leaf3, leaf4, lines)
        deep_tree.move_to(LEFT*3.5 + DOWN*1.5)
        deep_label.next_to(deep_tree, LEFT, buff=0.3).align_to(deep_tree, UP)
        self.play(Write(deep_label), Create(deep_tree))
        # Show total neurons approx 2n
        neuron_count = MathTex(r"\approx 2n \text{ neurons}", font_size=28, color=GOLD).next_to(deep_tree, RIGHT, buff=0.5)
        self.play(Write(neuron_count))
        
        # Shallow network: one hidden layer must have 2^n neurons
        shallow_label = Text("Shallow Net (1 hidden layer)", font_size=24, color=RED).next_to(deep_label, DOWN, buff=1.5).align_to(LEFT)
        shallow_rect = Rectangle(width=4, height=0.8, fill_color=RED, fill_opacity=0.3, stroke_color=RED)
        shallow_rect.next_to(shallow_label, DOWN, buff=0.3).align_to(shallow_label, LEFT)
        explosion_text = MathTex(r"\text{Needed: } \ge 2^n \text{ neurons}", font_size=28, color=RED).next_to(shallow_rect, RIGHT, buff=0.5)
        self.play(Write(shallow_label), Create(shallow_rect), Write(explosion_text))
        self.wait(0.5)
        
        # Animate the explosion: show many small dots popping out of the shallow rectangle
        dots = VGroup(*[Dot(radius=0.05, color=RED) for _ in range(50)])
        dots.arrange_in_grid(5, 10, buff=0.15).move_to(shallow_rect.get_center())
        self.play(LaggedStart(*[FadeIn(dot, scale=0.5) for dot in dots], lag_ratio=0.02))
        self.wait(0.5)
        # Emphasize exponential vs linear
        compare = MathTex(r"\text{Deep: } O(n) \quad \text{vs} \quad \text{Shallow: } O(2^n)",
                          font_size=32, color=WHITE).to_edge(DOWN, buff=0.5)
        self.play(Write(compare))
        self.wait(1)
