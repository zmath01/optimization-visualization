from manim import *

class ResNetVisualization(Scene):
    def construct(self):
        # Configuration & Styling
        C_INPUT = BLUE
        C_RES = ORANGE
        C_ADD = PURPLE
        C_GRAD = GREEN
        C_WEIGHTS = LIGHT_GREY
        
        # ==========================================
        # SCENE 1: The Problem with Plain Networks
        # ==========================================
        title_1 = Text("1. Plain Network: Gradients Vanish").to_edge(UP)
        self.play(Write(title_1))

        # Build a stack of plain layers
        layers = VGroup(*[
            Rectangle(height=0.4, width=3, color=C_WEIGHTS, fill_opacity=0.2)
            for _ in range(8)
        ]).arrange(UP, buff=0.2).shift(DOWN * 1)

        self.play(Create(layers))

        # Forward Pass (Input x)
        x_label = MathTex("x", color=C_INPUT).next_to(layers, DOWN)
        forward_arrow = Arrow(start=DOWN, end=UP, color=C_INPUT).scale(1.5).next_to(layers[0], DOWN)
        
        self.play(FadeIn(x_label), GrowArrow(forward_arrow))
        self.play(forward_arrow.animate.next_to(layers[-1], UP), run_time=2)
        self.play(FadeOut(forward_arrow))

        # Backward Pass (Vanishing Gradient)
        grad_label = MathTex(r"\frac{\partial L}{\partial y}", color=C_GRAD).next_to(layers[-1], UP)
        grad_arrow = Arrow(start=UP, end=DOWN, color=C_GRAD, stroke_width=8).next_to(layers[-1], UP)
        
        self.play(FadeIn(grad_label), GrowArrow(grad_arrow))

        # Animate the gradient shrinking as it flows backwards
        for i in range(len(layers)-1, -1, -1):
            shrink_factor = 0.7 ** (len(layers) - i)
            self.play(
                grad_arrow.animate.next_to(layers[i], DOWN).set_stroke(width=8 * shrink_factor),
                run_time=0.3
            )
        
        vanish_text = Text("Gradient disappears before reaching early layers", font_size=24, color=RED).next_to(grad_arrow, DOWN)
        self.play(Write(vanish_text))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # ==========================================
        # SCENE 2: The Residual Block Concept
        # ==========================================
        title_2 = Text("2. The Residual Block").to_edge(UP)
        self.play(Write(title_2))

        # Nodes
        start_node = Dot(color=C_INPUT).shift(DOWN * 2.5)
        x_start_label = MathTex(r"\mathbf{x}", color=C_INPUT).next_to(start_node, DOWN)
        
        add_node = Circle(radius=0.3, color=C_ADD).shift(UP * 1.5)
        plus_sign = MathTex("+", color=C_ADD).move_to(add_node)
        add_group = VGroup(add_node, plus_sign)

        # Path A: Residual Path (F(x))
        res_layer_1 = Rectangle(height=0.8, width=2.5, color=C_RES, fill_opacity=0.2).shift(DOWN * 0.8)
        res_layer_1_text = Text("Weight Layer", font_size=20, color=C_RES).move_to(res_layer_1)
        res_layer_2 = Rectangle(height=0.8, width=2.5, color=C_RES, fill_opacity=0.2).shift(UP * 0.5)
        res_layer_2_text = Text("Weight Layer", font_size=20, color=C_RES).move_to(res_layer_2)
        
        res_path_group = VGroup(res_layer_1, res_layer_1_text, res_layer_2, res_layer_2_text)
        
        res_arrow_1 = Arrow(start_node.get_top(), res_layer_1.get_bottom(), color=C_RES)
        res_arrow_2 = Arrow(res_layer_1.get_top(), res_layer_2.get_bottom(), color=C_RES)
        res_arrow_3 = Arrow(res_layer_2.get_top(), add_node.get_bottom(), color=C_RES)
        
        fx_label = MathTex(r"\mathcal{F}(\mathbf{x})", color=C_RES).next_to(res_layer_2, RIGHT)

        # Path B: Skip Connection (Identity)
        skip_line_1 = Line(start_node.get_center(), start_node.get_center() + LEFT * 3, color=C_INPUT)
        skip_line_2 = Line(skip_line_1.get_end(), skip_line_1.get_end() + UP * 4.0, color=C_INPUT)
        skip_line_3 = Arrow(skip_line_2.get_end(), add_node.get_left(), color=C_INPUT)
        skip_path = VGroup(skip_line_1, skip_line_2, skip_line_3)
        
        identity_label = MathTex(r"\mathbf{x} \text{ (Identity)}", color=C_INPUT).next_to(skip_line_2, LEFT)

        # Output
        out_arrow = Arrow(add_node.get_top(), add_node.get_top() + UP * 1, color=C_WEIGHTS)
        out_label = MathTex(r"\text{ReLU}", font_size=24).next_to(out_arrow, UP)

        self.play(FadeIn(start_node), Write(x_start_label))
        
        # Animate splitting
        self.play(
            Create(res_arrow_1), Create(res_path_group), Create(res_arrow_2), Create(res_arrow_3),
            Create(skip_path),
            run_time=2
        )
        self.play(Write(fx_label), Write(identity_label))
        
        # Animate merging
        self.play(FadeIn(add_group))
        self.play(GrowArrow(out_arrow), Write(out_label))

        # Core Formula
        formula = MathTex(r"\mathbf{y} = ", r"\mathcal{F}(\mathbf{x}, \{W_i\})", r" + ", r"\mathbf{x}").to_edge(DOWN)
        formula[1].set_color(C_RES)
        formula[3].set_color(C_INPUT)
        self.play(Write(formula))
        self.wait(2)

        # ==========================================
        # SCENE 3: Visualizing the Identity Mapping
        # ==========================================
        self.play(FadeOut(title_2))
        title_3 = Text("3. Easier to learn zero than identity").to_edge(UP)
        self.play(Write(title_3))

        # Fade out residual path to simulate F(x) approaching 0
        zero_fx = MathTex(r"\mathcal{F}(\mathbf{x}) \approx 0", color=C_RES).move_to(fx_label)
        self.play(
            res_path_group.animate.set_opacity(0.1),
            res_arrow_1.animate.set_opacity(0.2),
            res_arrow_2.animate.set_opacity(0.2),
            res_arrow_3.animate.set_opacity(0.2),
            Transform(fx_label, zero_fx)
        )
        
        # Highlight Skip
        self.play(
            skip_path.animate.set_stroke(width=8, color=YELLOW),
            run_time=1.5
        )
        self.wait(2)

        # Restore for next scene
        self.play(
            res_path_group.animate.set_opacity(1),
            res_arrow_1.animate.set_opacity(1),
            res_arrow_2.animate.set_opacity(1),
            res_arrow_3.animate.set_opacity(1),
            skip_path.animate.set_stroke(width=4, color=C_INPUT)
        )

        # ==========================================
        # SCENE 4: Gradient Highway (The "Why")
        # ==========================================
        self.play(FadeOut(title_3), FadeOut(formula))
        title_4 = Text("4. The Gradient Highway").to_edge(UP)
        self.play(Write(title_4))

        # Incoming Gradient
        grad_in_arrow = Arrow(out_arrow.get_top() + UP, add_node.get_top(), color=C_GRAD, stroke_width=8)
        grad_in_label = MathTex(r"\frac{\partial L}{\partial \mathbf{y}}", color=C_GRAD).next_to(grad_in_arrow, RIGHT)
        self.play(GrowArrow(grad_in_arrow), Write(grad_in_label))

        # Split Gradient
        grad_res_arrow = Arrow(add_node.get_bottom(), res_layer_2.get_top(), color=C_GRAD, stroke_width=4)
        grad_skip_line_1 = Line(add_node.get_left(), skip_line_2.get_end(), color=C_GRAD, stroke_width=8)
        grad_skip_line_2 = Line(skip_line_2.get_end(), skip_line_1.get_end(), color=C_GRAD, stroke_width=8)
        grad_skip_line_3 = Arrow(skip_line_1.get_end(), start_node.get_center(), color=C_GRAD, stroke_width=8)
        grad_skip_path = VGroup(grad_skip_line_1, grad_skip_line_2, grad_skip_line_3)

        self.play(
            GrowArrow(grad_res_arrow),
            Create(grad_skip_path),
            run_time=2
        )

        # Math logic
        deriv_eq = MathTex(
            r"\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = ", 
            r"\frac{\partial \mathcal{F}}{\partial \mathbf{x}}", 
            r" + ", 
            r"1"
        ).to_edge(DOWN)
        deriv_eq[1].set_color(C_RES)
        deriv_eq[3].set_color(C_GRAD)

        chain_rule = MathTex(
            r"\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + 1 \right)", 
            color=C_GRAD
        ).next_to(deriv_eq, UP)

        self.play(Write(deriv_eq))
        self.play(Circumscribe(deriv_eq[3], color=YELLOW, time_width=2))
        self.wait(1)
        self.play(Write(chain_rule))
        
        highway_text = Text("The '+1' guarantees gradients never vanish!", font_size=24, color=YELLOW).next_to(chain_rule, UP)
        self.play(Write(highway_text))
        self.wait(2.5)

        # ==========================================
        # SCENE 5: Projection Shortcuts
        # ==========================================
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in [
                add_node, plus_sign, add_group, start_node, res_layer_1, res_layer_2, 
                res_layer_1_text, res_layer_2_text, res_path_group, res_arrow_1, res_arrow_2, res_arrow_3
            ]]
        )
        
        title_5 = Text("5. Projection Shortcuts (Dimension Mismatch)").to_edge(UP)
        self.play(Write(title_5))

        # Show Dimensions
        dim_x = MathTex(r"\mathbf{x} \in \mathbb{R}^{64}", color=C_INPUT).next_to(start_node, DOWN)
        dim_fx = MathTex(r"\mathcal{F}(\mathbf{x}) \in \mathbb{R}^{128}", color=C_RES).next_to(res_layer_2, RIGHT)
        self.play(Write(dim_x), Write(dim_fx))

        # Re-draw standard skip connection
        skip_line_1 = Line(start_node.get_center(), start_node.get_center() + LEFT * 3, color=C_INPUT)
        skip_line_2 = Line(skip_line_1.get_end(), skip_line_1.get_end() + UP * 4.0, color=C_INPUT)
        skip_line_3 = Arrow(skip_line_2.get_end(), add_node.get_left(), color=C_INPUT)
        skip_path = VGroup(skip_line_1, skip_line_2, skip_line_3)
        self.play(Create(skip_path))

        # Cross out addition (Mismatch)
        cross = Cross(add_node, stroke_color=RED, stroke_width=6)
        mismatch_text = Text("Mismatch!", color=RED, font_size=24).next_to(cross, UP)
        self.play(Create(cross), Write(mismatch_text))
        self.wait(1)

        # Introduce 1x1 Conv (Ws)
        self.play(FadeOut(cross), FadeOut(mismatch_text))
        
        ws_box = Rectangle(height=0.6, width=1.5, color=YELLOW, fill_opacity=0.8).move_to(skip_line_2.get_center())
        ws_text = Text("1x1 Conv", font_size=16, color=BLACK).move_to(ws_box)
        ws_group = VGroup(ws_box, ws_text)
        
        self.play(Create(ws_group))
        
        ws_dim = MathTex(r"W_s \mathbf{x} \in \mathbb{R}^{128}", color=YELLOW).next_to(ws_box, LEFT)
        self.play(Write(ws_dim))

        # Update Formula
        final_formula = MathTex(r"\mathbf{y} = \mathcal{F}(\mathbf{x}) + W_s \mathbf{x}").to_edge(DOWN)
        self.play(Write(final_formula))
        self.wait(3)
