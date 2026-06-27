from manim import *
import numpy as np

np.random.seed(42)


class CheapLearningScene(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        self.scene_1_swindle()
        self.scene_2_physics()
        self.scene_3_compositionality()
        self.scene_4_rg()
        self.scene_5_no_flattening()

    # ---------- helpers ----------

    def _fade_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.15)

    def _big_brace(self, text, direction=DOWN, buff=0.1):
        t = MathTex(text, font_size=26, color=WHITE)
        br = Brace(t, direction=direction, buff=buff)
        return VGroup(t, br)

    # =================================================================
    # SCENE 1: The Combinatorial Swindle
    # =================================================================
    def scene_1_swindle(self):
        title = Text(
            "The Combinatorial Swindle", font_size=32, color=RED
        ).to_edge(UP, buff=0.15)
        self.play(Write(title), run_time=0.8)

        # all possible functions
        all_funcs = MathTex(
            r"256^{1\,000\,000}", font_size=64, color=RED
        )
        all_funcs.shift(UP * 0.5)
        self.play(Write(all_funcs), run_time=1)
        self.wait(0.5)

        all_label = Text(
            "All possible 256\u00d71000\u00d71000 images", font_size=16, color=RED
        ).next_to(all_funcs, DOWN, buff=0.15)
        self.play(Write(all_label))
        self.wait(0.5)

        # dramatically shrink to a dot
        phys_dot = Dot(color=BLUE, radius=0.08)
        phys_dot.move_to(all_funcs.get_center())
        phys_label = Text(
            "Physical images (tiny subset)", font_size=16, color=BLUE
        )
        phys_label.next_to(phys_dot, DOWN, buff=0.4)

        self.play(
            all_funcs.animate.scale(0.04).set_opacity(0.2).set_color(BLUE),
            all_label.animate.set_opacity(0),
            FadeIn(phys_dot, scale=5),
            Write(phys_label),
            run_time=2,
        )
        self.wait(0.5)

        # formula transformation
        generic_formula = MathTex(
            r"N_{\text{params}} \approx v^n",
            font_size=32, color=RED,
        )
        generic_formula.shift(DOWN * 0.8 + LEFT * 2)
        generic_label = Text(
            "Generic functions", font_size=16, color=RED
        ).next_to(generic_formula, DOWN, buff=0.1)

        cheap_formula = MathTex(
            r"N_{\text{params}} \approx v \times n",
            font_size=32, color=BLUE,
        )
        cheap_formula.shift(DOWN * 0.8 + RIGHT * 2)
        cheap_label = Text(
            "Physical functions", font_size=16, color=BLUE
        ).next_to(cheap_formula, DOWN, buff=0.1)

        swindle_arrow = Arrow(
            generic_formula.get_right() + RIGHT * 0.3,
            cheap_formula.get_left() - LEFT * 0.3,
            color=YELLOW, stroke_width=3,
        )
        swindle_text = Text(
            "The Swindle", font_size=20, color=YELLOW
        ).next_to(swindle_arrow, UP, buff=0.05)

        self.play(
            Write(generic_formula), Write(generic_label),
            Write(cheap_formula), Write(cheap_label),
            GrowArrow(swindle_arrow), Write(swindle_text),
            run_time=2,
        )
        self.wait(0.5)

        # network visualization – linear growth
        net_title = Text(
            "Network grows linearly with n", font_size=18, color=GOLD
        )
        net_title.to_edge(DOWN, buff=0.5)

        net_layers = VGroup()
        for i in range(6):
            y = -2.0 + i * 0.35
            rect = Rectangle(
                width=0.8 + i * 0.06, height=0.25,
                color=GOLD, fill_opacity=0.15,
            ).move_to([0, y, 0])
            net_layers.add(rect)

        self.play(Write(net_title), run_time=0.5)
        self.play(LaggedStart(*[Create(r) for r in net_layers], lag_ratio=0.15), run_time=2)
        self.wait(1)

        self._fade_all()

    # =================================================================
    # SCENE 2: Physics Properties – Symmetry & Locality
    # =================================================================
    def scene_2_physics(self):
        title = Text(
            "Physics Properties: Symmetry & Locality", font_size=30, color=BLUE
        ).to_edge(UP, buff=0.15)
        self.play(Write(title), run_time=0.8)

        # --- Locality ---
        loc_label = Text("Locality: nearby pixels interact", font_size=22, color=WHITE)
        loc_label.shift(UP * 1.5)
        self.play(Write(loc_label), run_time=0.5)

        # 5×5 grid of pixels
        grid = VGroup()
        n = 5
        spacing = 0.35
        for i in range(n):
            for j in range(n):
                sq = Square(
                    side_length=spacing * 0.85,
                    color=LIGHT_GRAY,
                    fill_opacity=0.15,
                    stroke_width=1,
                )
                sq.move_to([(i - n / 2) * spacing, (j - n / 2) * spacing - 0.3, 0])
                grid.add(sq)
        grid.shift(LEFT * 3)
        self.play(Create(grid), run_time=1)

        # neighborhood connections (local only)
        local_conns = VGroup()
        for i in range(n):
            for j in range(n):
                cx = (i - n / 2) * spacing - 3
                cy = (j - n / 2) * spacing - 0.3
                for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        nx = (ni - n / 2) * spacing - 3
                        ny = (nj - n / 2) * spacing - 0.3
                        line = Line(
                            [cx, cy, 0], [nx, ny, 0],
                            color=BLUE, stroke_width=1, stroke_opacity=0.4,
                        )
                        local_conns.add(line)

        # fade in local connections
        self.play(
            *[Create(c) for c in local_conns],
            run_time=2, lag_ratio=0.02,
        )
        self.wait(0.3)

        # Hamiltonian expansion
        hamiltonian = MathTex(
            r"H(x) = \sum_i h_i x_i + \sum_{ij} h_{ij} x_i x_j + \sum_{ijk} h_{ijk} x_i x_j x_k + \dots",
            font_size=26, color=WHITE,
        )
        hamiltonian.next_to(loc_label, DOWN, buff=0.1).shift(RIGHT * 0.5)

        locality_note = Text(
            "Most coefficients = 0\n(locality constraint)",
            font_size=18, color=BLUE, line_spacing=0.3,
        )
        locality_note.next_to(hamiltonian, DOWN, buff=0.3)

        self.play(Write(hamiltonian), run_time=1.5)
        self.play(Write(locality_note), run_time=1)
        self.wait(0.5)

        # --- Symmetry ---
        sym_label = Text("Symmetry: physics is invariant under rotation", font_size=22, color=WHITE)
        sym_label.to_edge(DOWN, buff=0.3)
        self.play(Write(sym_label), run_time=0.5)

        # create a simple pattern on a small grid
        pattern_grid = VGroup()
        pg = 3
        vals = np.array([
            [0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.1],
        ])
        ps = 0.5
        for i in range(pg):
            for j in range(pg):
                sq = Square(
                    side_length=ps * 0.9,
                    color=BLUE,
                    fill_opacity=vals[i, j],
                    stroke_width=1,
                )
                sq.move_to([(i - pg / 2) * ps + 3.5, (j - pg / 2) * ps - 0.3, 0])
                pattern_grid.add(sq)

        self.play(FadeIn(pattern_grid), run_time=0.8)

        # rotate animation
        rot_label = Text("Rotate 90\u00b0", font_size=18, color=YELLOW)
        rot_label.next_to(pattern_grid, DOWN, buff=0.2)

        self.play(Write(rot_label))
        self.wait(0.3)
        self.play(
            Rotate(pattern_grid, angle=PI / 2, about_point=pattern_grid.get_center()),
            run_time=1.5,
        )
        self.wait(0.3)

        # output stays same
        out_eq = MathTex(
            r"P(\text{output}) = P(\text{rotated output})",
            font_size=22, color=GREEN,
        )
        out_eq.next_to(sym_label, UP, buff=0.3)
        self.play(Write(out_eq), run_time=1)

        weight_label = Text(
            "Result: Network weights must share symmetry",
            font_size=18, color=YELLOW,
        )
        weight_label.next_to(out_eq, UP, buff=0.2)
        self.play(Write(weight_label), run_time=1)
        self.wait(1)

        self._fade_all()

    # =================================================================
    # SCENE 3: Compositionality & Depth
    # =================================================================
    def scene_3_compositionality(self):
        title = Text(
            "Compositionality: The Power of Depth", font_size=30, color=PURPLE
        ).to_edge(UP, buff=0.15)
        self.play(Write(title), run_time=0.8)

        # --- Deep Hierarchical Network ---
        deep_label = Text("Deep Network (Hierarchical)", font_size=22, color=PURPLE)
        deep_label.shift(LEFT * 3 + UP * 1.5)
        self.play(Write(deep_label))

        # Tree structure
        # Level 3 (top): Face
        # Level 2: Eyes, Nose, Mouth (3 nodes)
        # Level 1: Edges (5 nodes)
        # Level 0: Pixels (8 nodes)

        tree_nodes = VGroup()
        tree_edges = VGroup()

        node_positions = {
            "face": np.array([-3, 1.5, 0]),
            "eyes": np.array([-4.5, 0, 0]),
            "nose": np.array([-3, -0.2, 0]),
            "mouth": np.array([-1.5, 0, 0]),
            "e1": np.array([-5.5, -1.5, 0]),
            "e2": np.array([-4.5, -1.5, 0]),
            "e3": np.array([-3, -1.5, 0]),
            "e4": np.array([-1.5, -1.5, 0]),
            "e5": np.array([-0.5, -1.5, 0]),
        }

        # create nodes
        face_node = Text("Face", font_size=16, color=PURPLE).move_to(node_positions["face"])
        eyes_node = Text("Eyes", font_size=13, color=PURPLE).move_to(node_positions["eyes"])
        nose_node = Text("Nose", font_size=13, color=PURPLE).move_to(node_positions["nose"])
        mouth_node = Text("Mouth", font_size=13, color=PURPLE).move_to(node_positions["mouth"])

        edge_nodes = VGroup()
        for k in ["e1", "e2", "e3", "e4", "e5"]:
            d = Dot(radius=0.04, color=PURPLE).move_to(node_positions[k])
            edge_nodes.add(d)

        tree_nodes.add(face_node, eyes_node, nose_node, mouth_node, edge_nodes)

        # tree edges
        pairs = [
            ("face", "eyes"), ("face", "nose"), ("face", "mouth"),
            ("eyes", "e1"), ("eyes", "e2"),
            ("nose", "e3"),
            ("mouth", "e4"), ("mouth", "e5"),
        ]
        for p in pairs:
            line = Line(
                node_positions[p[0]], node_positions[p[1]],
                color=PURPLE, stroke_width=2, stroke_opacity=0.6,
            )
            tree_edges.add(line)

        self.play(
            *[Create(e) for e in tree_edges],
            *[Write(n) for n in [face_node, eyes_node, nose_node, mouth_node]],
            *[FadeIn(d) for d in edge_nodes],
            run_time=2,
        )
        self.wait(0.3)

        # abstraction arrows
        abstr_arrows = VGroup()
        for y in [-0.6, 0.7]:
            arr = Arrow(
                [-6, y, 0], [0.5, y, 0],
                color=PURPLE, stroke_width=2, stroke_opacity=0.4,
            )
            abstr_arrows.add(arr)
        self.play(*[GrowArrow(a) for a in abstr_arrows], run_time=0.5)

        # --- Shallow Network ---
        shallow_label = Text("Shallow Network (Flat)", font_size=22, color=RED)
        shallow_label.shift(RIGHT * 3 + UP * 1.5)
        self.play(Write(shallow_label))

        # shallow net: input (8 pixels) → hidden → output
        inputs = VGroup()
        for i in range(8):
            d = Dot(radius=0.035, color=LIGHT_GRAY)
            d.move_to([2, -2 + i * 0.5, 0])
            inputs.add(d)

        hidden_shallow = VGroup()
        for i in range(6):
            d = Dot(radius=0.04, color=RED)
            d.move_to([4, -1.7 + i * 0.55, 0])
            hidden_shallow.add(d)

        output_s = Text("Face?", font_size=14, color=RED).move_to([5.8, 0, 0])

        # tangled connections
        tangles = VGroup()
        for inp in inputs:
            for hid in hidden_shallow:
                line = Line(
                    inp.get_center(), hid.get_center(),
                    color=RED, stroke_width=0.5, stroke_opacity=0.2,
                )
                tangles.add(line)
        # hidden to output
        for hid in hidden_shallow:
            line = Line(
                hid.get_center(), output_s.get_center(),
                color=RED, stroke_width=0.5, stroke_opacity=0.2,
            )
            tangles.add(line)

        self.play(
            FadeIn(inputs),
            FadeIn(hidden_shallow),
            Write(output_s),
            *[Create(t) for t in tangles],
            run_time=2.5, lag_ratio=0.02,
        )
        self.wait(0.3)

        # text explanation
        depth_text = Text(
            "Depth = Hierarchical Compositionality",
            font_size=24, color=YELLOW,
        )
        depth_text.to_edge(DOWN, buff=0.3)
        self.play(Write(depth_text), run_time=1)
        self.wait(1.5)

        self._fade_all()

    # =================================================================
    # SCENE 4: Renormalization Group Connection
    # =================================================================
    def scene_4_rg(self):
        title = Text(
            "Deep Learning = Variational Renormalization Group",
            font_size=28, color=GREEN,
        ).to_edge(UP, buff=0.15)
        self.play(Write(title), run_time=0.8)

        # --- Lattice of spins ---
        lat_label = Text("Spin Lattice (Scale L)", font_size=20, color=WHITE)
        lat_label.shift(LEFT * 3 + UP * 1.3)
        self.play(Write(lat_label))

        lat_n = 6
        lat_spacing = 0.3
        lattice = VGroup()
        for i in range(lat_n):
            for j in range(lat_n):
                up = np.random.choice([True, False])
                color = GREEN if up else WHITE
                d = Dot(radius=0.05, color=color)
                d.move_to([
                    (i - lat_n / 2) * lat_spacing - 2.5,
                    (j - lat_n / 2) * lat_spacing + 0.2,
                    0,
                ])
                lattice.add(d)

        self.play(LaggedStart(*[FadeIn(d) for d in lattice], lag_ratio=0.02), run_time=1.5)
        self.wait(0.3)

        # --- Coarse-graining (decimation) ---
        cg_label = Text("Coarse-graining (Decimation)", font_size=18, color=GREEN)
        cg_label.next_to(lat_label, DOWN, buff=0.3)

        # highlight 2×2 blocks
        blocks = VGroup()
        cg_dots = VGroup()
        for bi in range(0, lat_n, 2):
            for bj in range(0, lat_n, 2):
                cx = (bi - lat_n / 2 + 0.5) * lat_spacing - 2.5
                cy = (bj - lat_n / 2 + 0.5) * lat_spacing + 0.2
                rect = Square(
                    side_length=lat_spacing * 1.05,
                    color=GREEN, stroke_width=2, stroke_opacity=0.6,
                ).move_to([cx, cy, 0])
                blocks.add(rect)

                # majority vote spin
                spins = []
                for di in range(2):
                    for dj in range(2):
                        idx = (bi + di) * lat_n + (bj + dj)
                        if idx < len(lattice):
                            spins.append(lattice[idx].get_color() == GREEN)
                maj_up = sum(spins) >= 2
                d = Dot(
                    radius=0.08, color=GREEN if maj_up else WHITE,
                ).move_to([cx + 4, cy, 0])
                cg_dots.add(d)

        self.play(Write(cg_label), run_time=0.5)
        self.play(*[Create(b) for b in blocks], run_time=1, lag_ratio=0.05)
        self.wait(0.3)

        # move coarse-grained dots to the right
        self.play(
            LaggedStart(*[FadeIn(d) for d in cg_dots], lag_ratio=0.03),
            run_time=1.5,
        )
        self.wait(0.3)

        # RG flow label
        rg_flow = Text(
            "RG Flow: J \u2192 J'",
            font_size=22, color=GREEN,
        )
        rg_flow.shift(RIGHT * 4 + UP * 0.5)
        self.play(Write(rg_flow), run_time=0.5)
        self.wait(0.3)

        # --- Morph into CNN ---
        morph_label = Text("Morph into CNN ...", font_size=20, color=YELLOW)
        morph_label.shift(DOWN * 1.5)
        self.play(Write(morph_label), run_time=0.5)

        # show CNN layers
        cnn_label = Text("CNN: Conv + Pooling", font_size=22, color=GREEN)
        cnn_label.to_edge(DOWN, buff=0.3)

        # input layer → conv → pool → conv → pool → fc
        cnn_layers = VGroup()
        layer_specs = [
            ("Input", 1.6, 1.0, LIGHT_GRAY),
            ("Conv", 1.4, 0.5, ORANGE),
            ("Pool", 1.0, 0.5, GREEN),
            ("Conv", 1.2, 0.5, ORANGE),
            ("Pool", 0.8, 0.5, GREEN),
            ("FC", 0.6, 0.5, PURPLE),
        ]
        x_pos = -5.5
        for name, w, h, color in layer_specs:
            rect = Rectangle(width=w, height=h, color=color, fill_opacity=0.15)
            rect.move_to([x_pos, 0, 0])
            txt = Text(name, font_size=12, color=color).move_to(rect)
            cnn_layers.add(VGroup(rect, txt))
            x_pos += w + 0.6

        # arrows between layers
        cnn_arrows = VGroup()
        for i in range(len(layer_specs) - 1):
            x1 = -5.5 + sum(layer_specs[k][1] for k in range(i)) + 0.6 * i + layer_specs[i][1] / 2
            x2 = x1 + layer_specs[i][1] / 2 + 0.6 + layer_specs[i + 1][1] / 2
            arr = Arrow([x1, 0, 0], [x2, 0, 0], color=WHITE, stroke_width=2, buff=0.08)
            cnn_arrows.add(arr)

        self.play(FadeOut(lat_label, cg_label, morph_label, rg_flow, blocks, lattice, cg_dots))
        self.play(
            LaggedStart(
                *[Create(l) for l in cnn_layers],
                *[GrowArrow(a) for a in cnn_arrows],
                lag_ratio=0.1,
            ),
            Write(cnn_label),
            run_time=2.5,
        )
        self.wait(1.5)

        self._fade_all()

    # =================================================================
    # SCENE 5: No-Flattening Theorem
    # =================================================================
    def scene_5_no_flattening(self):
        title = Text(
            "No-Flattening Theorem: Multiplication", font_size=30, color=GOLD
        ).to_edge(UP, buff=0.15)
        self.play(Write(title), run_time=0.8)

        # task
        task = MathTex(r"f(x) = x_1 \times x_2 \times \dots \times x_n", font_size=28, color=WHITE)
        task.shift(UP * 1)
        self.play(Write(task), run_time=1)
        self.wait(0.3)

        # --- Deep Network ---
        deep_label = Text("Deep Network", font_size=22, color=GOLD)
        deep_label.shift(LEFT * 3 + UP * 0.5)
        self.play(Write(deep_label))

        # Multiplication tree for n=4
        # Level 0: x1, x2, x3, x4
        x_positions = [[-3.5, -2, -0.5, 1], [-2.75, 0.25], [-1.25]]
        y_levels = [-0.5, -1.5, -2.5]

        deep_nodes = VGroup()
        deep_edges = VGroup()

        # positions shifted left by 2
        shift_vec = LEFT * 2

        for level, xs in enumerate(x_positions):
            y = y_levels[level]
            for i, x in enumerate(xs):
                if level == 0:
                    t = MathTex(f"x_{{{i + 1}}}", font_size=16, color=GOLD)
                else:
                    t = MathTex(r"\times", font_size=16, color=GOLD)
                t.move_to(np.array([x, y, 0]) + shift_vec)
                deep_nodes.add(t)

        # edges (multiplication tree)
        # Level 0 → Level 1: x1,x2 → p1; x3,x4 → p2
        for src_idx in range(0, 4, 2):
            for off in range(2):
                start = np.array([x_positions[0][src_idx + off], y_levels[0], 0]) + shift_vec
                end = np.array([x_positions[1][src_idx // 2], y_levels[1], 0]) + shift_vec
                line = Line(start, end, color=GOLD, stroke_width=1.5, stroke_opacity=0.5)
                deep_edges.add(line)
        # Level 1 → Level 2
        for src_idx in range(2):
            start = np.array([x_positions[1][src_idx], y_levels[1], 0]) + shift_vec
            end = np.array([x_positions[2][0], y_levels[2], 0]) + shift_vec
            line = Line(start, end, color=GOLD, stroke_width=1.5, stroke_opacity=0.5)
            deep_edges.add(line)

        out_deep = Text("= \u220f x_i", font_size=16, color=GOLD)
        out_deep.move_to(np.array([x_positions[2][0] + 0.6, y_levels[2], 0]) + shift_vec)

        self.play(
            *[Write(n) for n in deep_nodes],
            *[Create(e) for e in deep_edges],
            Write(out_deep),
            run_time=2,
        )
        self.wait(0.3)

        neuron_count = Text("Neurons \u2248 2n", font_size=18, color=GOLD)
        neuron_count.next_to(out_deep, DOWN, buff=0.3)
        self.play(Write(neuron_count))
        self.wait(0.5)

        # --- Shallow Network ---
        shallow_label = Text("Shallow Network", font_size=22, color=RED)
        shallow_label.shift(RIGHT * 3 + UP * 0.5)
        self.play(Write(shallow_label))

        # input nodes
        n_vars = 4
        inp_nodes = VGroup()
        for i in range(n_vars):
            t = MathTex(f"x_{{{i + 1}}}", font_size=14, color=RED)
            t.move_to([1.5, 0.5 - i * 0.6, 0])
            inp_nodes.add(t)

        # hidden layer (2^n = 16 neurons for n=4)
        n_hidden = 16
        hidden_nodes = VGroup()
        for i in range(min(n_hidden, 14)):
            d = Dot(radius=0.03, color=RED)
            d.move_to([3.5, 1.0 - i * 0.14, 0])
            hidden_nodes.add(d)

        hidden_dots = Text("...", font_size=14, color=RED).move_to([3.5, -1.0, 0])

        output = Text("= \u220f x_i ?", font_size=14, color=RED).move_to([5.5, 0, 0])

        # connections (tangled)
        shallow_conn = VGroup()
        for inp in inp_nodes:
            for hid in hidden_nodes:
                line = Line(
                    inp.get_center(), hid.get_center(),
                    color=RED, stroke_width=0.4, stroke_opacity=0.15,
                )
                shallow_conn.add(line)
        for hid in hidden_nodes:
            line = Line(
                hid.get_center(), output.get_center(),
                color=RED, stroke_width=0.4, stroke_opacity=0.15,
            )
            shallow_conn.add(line)

        self.play(
            *[Write(n) for n in inp_nodes],
            *[FadeIn(h) for h in hidden_nodes],
            Write(hidden_dots),
            Write(output),
            *[Create(c) for c in shallow_conn],
            run_time=2, lag_ratio=0.02,
        )
        self.wait(0.3)

        explosion_text = MathTex(
            r"\ge 2^n \text{ neurons needed}", font_size=20, color=RED,
        )
        explosion_text.next_to(output, DOWN, buff=0.5)
        self.play(Write(explosion_text))
        self.wait(0.3)

        # Theorem
        theorem = VGroup()
        theorem_line1 = Text(
            "No-Flattening Theorem:",
            font_size=26, color=YELLOW,
        )
        theorem_line2 = Text(
            "Multiplying n variables requires \u2265 2\u207f neurons",
            font_size=22, color=YELLOW,
        )
        theorem_line3 = Text(
            "in a single hidden layer.",
            font_size=22, color=YELLOW,
        )
        theorem.add(theorem_line1, theorem_line2, theorem_line3)
        theorem.arrange(DOWN, buff=0.1)
        theorem.to_edge(DOWN, buff=0.15)

        # fade out previous elements for cleaner display
        self.play(
            FadeOut(inp_nodes, hidden_nodes, hidden_dots, output, shallow_conn, explosion_text),
            Write(theorem),
            run_time=1.5,
        )
        self.wait(2)

        self._fade_all()
