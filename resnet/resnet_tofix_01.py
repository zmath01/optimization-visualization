from manim import *
import numpy as np


class ResNetVisualization(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"
        np.random.seed(42)

        # =====================================================================
        # SCENE 1: The Problem with Plain Networks
        # =====================================================================
        title1 = Text("Deep Plain Networks: Vanishing Gradients", font_size=28, color=RED)
        title1.to_edge(UP, buff=0.15)

        num_layers = 8
        layer_height = 0.32
        layer_spacing = 0.28
        total_h = num_layers * (layer_height + layer_spacing) - layer_spacing
        start_y = total_h / 2

        plain_layers = VGroup()
        for i in range(num_layers):
            y = start_y - i * (layer_height + layer_spacing)
            rect = Rectangle(
                width=2.2, height=layer_height,
                color=LIGHT_GRAY, fill_opacity=0.1, stroke_width=1.5,
            )
            rect.move_to([0, y, 0])
            lbl = Text(
                f"Layer {num_layers - i}", font_size=13, color=LIGHT_GRAY
            ).move_to(rect)
            plain_layers.add(VGroup(rect, lbl))

        # forward arrows (bottom → top)
        fwd_arrows = VGroup()
        for i in range(num_layers - 1):
            bot = start_y - (i + 1) * (layer_height + layer_spacing) + layer_height / 2
            top = start_y - i * (layer_height + layer_spacing) - layer_height / 2
            a = Arrow(
                [0, bot, 0], [0, top, 0],
                color=BLUE, stroke_width=3, buff=0.04,
            )
            fwd_arrows.add(a)

        # input / output labels
        inp_label = MathTex("x", font_size=22, color=BLUE).next_to(
            [0, -total_h / 2 - 0.35, 0], DOWN, buff=0.05
        )
        out_label = MathTex(r"\mathcal{H}(x)", font_size=22, color=WHITE).next_to(
            [0, total_h / 2 + 0.35, 0], UP, buff=0.05
        )

        self.play(Write(title1), run_time=1)
        self.play(Create(plain_layers), run_time=1.5)
        self.play(Write(inp_label), Write(out_label))
        self.play(
            *[GrowArrow(a) for a in fwd_arrows],
            run_time=1.5,
            lag_ratio=0.15,
        )
        self.wait(0.5)

        # backward pass – vanishing gradients
        vanishing_arrows = VGroup()
        vanishing_label = Text(
            "Gradient vanishes", font_size=20, color=GREEN
        )
        vanishing_label.to_corner(DR, buff=0.4)

        for i in range(num_layers - 1):
            idx = num_layers - 2 - i
            bot = start_y - (idx + 1) * (layer_height + layer_spacing) + layer_height / 2
            top = start_y - idx * (layer_height + layer_spacing) - layer_height / 2
            fade = max(0.08, 1.0 - (i + 1) * 0.13)
            thick = max(0.8, 5.0 - i * 0.55)
            a = Arrow(
                [0, top, 0], [0, bot, 0],
                color=GREEN, stroke_width=thick, stroke_opacity=fade, buff=0.04,
            )
            vanishing_arrows.add(a)
            self.play(GrowArrow(a), run_time=0.25)

        gradient_mags = VGroup()
        for i in range(num_layers - 1):
            idx = num_layers - 2 - i
            fade = max(0.08, 1.0 - (i + 1) * 0.13)
            val = f"{fade:.1f}"
            mag = MathTex(val, font_size=12, color=GREEN, stroke_opacity=fade)
            y_pos = start_y - (idx + 0.5) * (layer_height + layer_spacing)
            mag.next_to([0.7, y_pos, 0], RIGHT, buff=0.05)
            gradient_mags.add(mag)

        self.play(
            *[Write(m) for m in gradient_mags],
            Write(vanishing_label),
            run_time=1,
        )
        self.wait(1)

        # =====================================================================
        # transition → SCENE 2
        # =====================================================================
        scene1_all = VGroup(
            title1, plain_layers, inp_label, out_label,
            fwd_arrows, vanishing_arrows, gradient_mags, vanishing_label,
        )
        self.play(FadeOut(scene1_all), run_time=1)
        self.wait(0.3)

        # =====================================================================
        # SCENE 2: The Residual Block Concept
        # =====================================================================
        title2 = Text("Residual Block: y = F(x) + x", font_size=28, color=ORANGE)
        title2.to_edge(UP, buff=0.15)

        # positions
        input_pos = np.array([-5.8, 0, 0])
        conv_pos = np.array([-3.0, 1.3, 0])
        bn_pos = np.array([-0.5, 1.3, 0])
        relu_pos = np.array([2.0, 1.3, 0])
        adder_pos = np.array([4.5, 0, 0])
        output_pos = np.array([6.5, 0, 0])

        rect_w, rect_h = 1.6, 0.5

        def make_layer(pos, label, color, fill=0.15):
            rect = Rectangle(
                width=rect_w, height=rect_h,
                color=color, fill_opacity=fill, stroke_width=2,
            ).move_to(pos)
            txt = Text(label, font_size=14, color=color).move_to(rect)
            return VGroup(rect, txt)

        conv_layer = make_layer(conv_pos, "Conv", ORANGE, 0.2)
        bn_layer = make_layer(bn_pos, "BN", ORANGE, 0.15)
        relu_layer = make_layer(relu_pos, "ReLU", ORANGE, 0.2)

        # addition node
        adder_circle = Circle(radius=0.3, color=PURPLE, fill_opacity=0.15, stroke_width=2.5)
        plus = MathTex("+", font_size=22, color=PURPLE).move_to(adder_circle)
        adder_node = VGroup(adder_circle, plus).move_to(adder_pos)

        # arrows
        arrow_in_conv = Arrow(input_pos, conv_pos, color=ORANGE, stroke_width=3, buff=0.12)
        arrow_conv_bn = Arrow(conv_pos, bn_pos, color=ORANGE, stroke_width=3, buff=0.12)
        arrow_bn_relu = Arrow(bn_pos, relu_pos, color=ORANGE, stroke_width=3, buff=0.12)
        arrow_relu_add = Arrow(relu_pos, adder_pos, color=ORANGE, stroke_width=3, buff=0.12)

        # skip connection (curved, going below)
        skip_arrow = CurvedArrow(
            input_pos, adder_pos - np.array([0.15, 0, 0]),
            angle=-TAU / 7,
            color=BLUE, stroke_width=4,
        )

        arrow_add_out = Arrow(adder_pos, output_pos, color=PURPLE, stroke_width=3.5, buff=0.12)

        # input / output labels
        input_label = MathTex("x", font_size=24, color=BLUE)
        input_label.next_to(input_pos, LEFT, buff=0.2)

        output_label = MathTex(r"y = \mathcal{F}(x) + x", font_size=22, color=PURPLE)
        output_label.next_to(output_pos, RIGHT, buff=0.15)

        # residual path label
        residual_label = Text("Residual Path F(x)", font_size=16, color=ORANGE)
        residual_label.next_to(conv_pos, UP * 0.5 + RIGHT * 0.5, buff=0)

        skip_label = MathTex("x", font_size=20, color=BLUE)
        skip_label.next_to(skip_arrow, DOWN, buff=0.1)

        self.play(Write(title2), run_time=0.8)
        self.play(FadeIn(conv_layer), FadeIn(bn_layer), FadeIn(relu_layer), run_time=1)
        self.play(Create(adder_node), run_time=0.5)
        self.play(GrowArrow(arrow_in_conv), run_time=0.5)
        self.play(
            GrowArrow(arrow_conv_bn),
            GrowArrow(arrow_bn_relu),
            GrowArrow(arrow_relu_add),
            run_time=1,
        )
        self.play(Create(skip_arrow), run_time=0.8)
        self.play(GrowArrow(arrow_add_out), run_time=0.5)
        self.play(Write(input_label), Write(output_label))
        self.play(Write(residual_label), Write(skip_label))
        self.wait(0.5)

        # core formula (bottom)
        core_formula = MathTex(
            r"\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}",
            font_size=28, color=ORANGE,
        )
        core_formula.to_edge(DOWN, buff=0.25)
        self.play(Write(core_formula), run_time=1.5)
        self.wait(1)

        # animate data flow through both paths
        flow_dot_skip = Dot(color=BLUE, radius=0.06)
        flow_dot_skip.move_to(input_pos)
        flow_dot_res = flow_dot_skip.copy()
        flow_dot_add = Dot(color=PURPLE, radius=0.06).move_to(adder_pos)
        flow_dot_add.set_opacity(0)

        path_res = [input_pos, conv_pos, bn_pos, relu_pos, adder_pos]
        path_skip = [input_pos, adder_pos]

        self.play(FadeIn(flow_dot_skip), FadeIn(flow_dot_res))
        for i in range(len(path_res) - 1):
            self.play(
                flow_dot_res.animate.move_to(path_res[i + 1]),
                run_time=0.4,
            )
        self.play(
            flow_dot_skip.animate.move_to(skip_arrow.get_end()),
            run_time=0.6,
        )
        self.play(
            FadeOut(flow_dot_res),
            FadeOut(flow_dot_skip),
            FadeIn(flow_dot_add),
            run_time=0.3,
        )
        self.play(flow_dot_add.animate.move_to(output_pos), run_time=0.5)
        self.play(FadeOut(flow_dot_add))
        self.wait(0.5)

        scene2_block = VGroup(
            conv_layer, bn_layer, relu_layer, adder_node,
            arrow_in_conv, arrow_conv_bn, arrow_bn_relu, arrow_relu_add,
            skip_arrow, arrow_add_out,
            input_label, output_label, residual_label, skip_label,
        )

        # =====================================================================
        # transition → SCENE 3
        # =====================================================================
        self.play(
            FadeOut(scene2_block),
            FadeOut(core_formula),
            run_time=1,
        )
        # keep title2 for continuity

        # =====================================================================
        # SCENE 3: Visualizing the Identity Mapping
        # =====================================================================
        title3 = Text("Learning Residual: F(x) → 0 is easier", font_size=28, color=ORANGE)
        title3.move_to(title2.get_center())
        self.play(Transform(title2, title3), run_time=0.8)

        # recreate block (slightly different positions for scene 3)
        conv_3 = make_layer(conv_pos, "Conv", ORANGE, 0.2)
        bn_3 = make_layer(bn_pos, "BN", ORANGE, 0.15)
        relu_3 = make_layer(relu_pos, "ReLU", ORANGE, 0.2)
        adder_3 = adder_node.copy()
        skip_3 = skip_arrow.copy()
        input_3 = MathTex("x", font_size=24, color=BLUE).next_to(input_pos, LEFT, buff=0.2)
        output_3 = MathTex(
            r"\mathcal{H}(x) = \mathcal{F}(x) + x", font_size=20, color=PURPLE
        ).next_to(output_pos, RIGHT, buff=0.15)

        arrow_conv_bn_3 = Arrow(conv_pos, bn_pos, color=ORANGE, stroke_width=3, buff=0.12)
        arrow_bn_relu_3 = Arrow(bn_pos, relu_pos, color=ORANGE, stroke_width=3, buff=0.12)
        arrow_relu_add_3 = Arrow(relu_pos, adder_pos, color=ORANGE, stroke_width=3, buff=0.12)

        self.play(
            FadeIn(conv_3), FadeIn(bn_3), FadeIn(relu_3),
            FadeIn(adder_3), FadeIn(skip_3),
            FadeIn(input_3), FadeIn(output_3),
            FadeIn(arrow_conv_bn_3), FadeIn(arrow_bn_relu_3), FadeIn(arrow_relu_add_3),
            run_time=1,
        )

        # fade residual path weights to near-zero
        weight_text = Text("Weights → 0", font_size=16, color=ORANGE)
        weight_text.next_to(conv_pos, UP, buff=0.2)

        self.play(Write(weight_text), run_time=0.5)
        self.play(
            conv_3.animate.set_opacity(0.15),
            bn_3.animate.set_opacity(0.1),
            relu_3.animate.set_opacity(0.15),
            arrow_conv_bn_3.animate.set_stroke_opacity(0.15),
            arrow_bn_relu_3.animate.set_stroke_opacity(0.15),
            arrow_relu_add_3.animate.set_stroke_opacity(0.15),
            run_time=1.5,
        )

        # highlight skip connection
        glow_rect = SurroundingRectangle(
            skip_3, color=BLUE, buff=0.15, stroke_width=3, stroke_opacity=0.8
        )
        self.play(Create(glow_rect), run_time=0.5)

        zero_text = Text(
            "Easier to learn residual (≈0)\nthan identity from scratch",
            font_size=18, color=WHITE,
            line_spacing=0.3,
        )
        zero_text.to_corner(DR, buff=0.4)
        self.play(Write(zero_text), run_time=1)
        self.wait(1.5)

        scene3_all = VGroup(
            conv_3, bn_3, relu_3, adder_3, skip_3,
            input_3, output_3,
            arrow_conv_bn_3, arrow_bn_relu_3, arrow_relu_add_3,
            weight_text, glow_rect, zero_text,
        )

        # =====================================================================
        # transition → SCENE 4
        # =====================================================================
        self.play(
            FadeOut(scene3_all),
            run_time=1,
        )
        self.wait(0.2)

        # =====================================================================
        # SCENE 4: Gradient Highway (The "Why")
        # =====================================================================
        title4 = Text("Gradient Highway: The +1", font_size=28, color=GREEN)
        title4.move_to(title2.get_center())
        self.play(Transform(title2, title4), run_time=0.8)

        # recreate block
        conv_4 = make_layer(conv_pos, "Conv", ORANGE, 0.15)
        bn_4 = make_layer(bn_pos, "BN", ORANGE, 0.12)
        relu_4 = make_layer(relu_pos, "ReLU", ORANGE, 0.15)
        adder_4 = adder_node.copy()
        skip_4 = skip_arrow.copy()
        input_4 = MathTex("x", font_size=22, color=BLUE).next_to(input_pos, LEFT, buff=0.2)

        arrow_conv_bn_4 = Arrow(conv_pos, bn_pos, color=ORANGE, stroke_width=2, buff=0.12)
        arrow_bn_relu_4 = Arrow(bn_pos, relu_pos, color=ORANGE, stroke_width=2, buff=0.12)
        arrow_relu_add_4 = Arrow(relu_pos, adder_pos, color=ORANGE, stroke_width=2, buff=0.12)

        self.play(
            FadeIn(conv_4), FadeIn(bn_4), FadeIn(relu_4),
            FadeIn(adder_4), FadeIn(skip_4), FadeIn(input_4),
            FadeIn(arrow_conv_bn_4), FadeIn(arrow_bn_relu_4), FadeIn(arrow_relu_add_4),
            run_time=1,
        )

        # gradient arrival
        grad_from_top = Arrow(
            output_pos + RIGHT * 0.3, adder_pos,
            color=GREEN, stroke_width=6, buff=0.1,
        )
        grad_label = MathTex(
            r"\frac{\partial L}{\partial \mathbf{y}}", font_size=20, color=GREEN
        )
        grad_label.next_to(grad_from_top, RIGHT, buff=0.05)

        self.play(GrowArrow(grad_from_top), Write(grad_label), run_time=1)
        self.wait(0.3)

        # split – skip path (thick, full gradient)
        grad_skip_path = CurvedArrow(
            adder_pos - np.array([0.15, 0, 0]),
            input_pos,
            angle=TAU / 7,
            color=GREEN, stroke_width=5.5,
        )
        grad_skip_label = MathTex(r"+1", font_size=24, color=GREEN)
        grad_skip_label.next_to(grad_skip_path, DOWN, buff=0.1)

        # FIX: Use Create instead of GrowArrow for CurvedArrow
        self.play(Create(grad_skip_path), Write(grad_skip_label), run_time=0.8)

        # split – residual path (thin, vanishing)
        grad_res_path = Arrow(
            adder_pos, relu_pos,
            color=GREEN, stroke_width=2.5, stroke_opacity=0.35, buff=0.1,
        )
        grad_res_label = MathTex(
            r"\frac{\partial \mathcal{F}}{\partial \mathbf{x}}",
            font_size=16, color=GREEN,
        )
        grad_res_label.next_to(adder_pos, UP, buff=0.1)

        # also through BN and Conv
        grad_res_bn = Arrow(
            relu_pos, bn_pos,
            color=GREEN, stroke_width=2, stroke_opacity=0.25, buff=0.1,
        )
        grad_res_conv = Arrow(
            bn_pos, conv_pos,
            color=GREEN, stroke_width=1.5, stroke_opacity=0.15, buff=0.1,
        )

        self.play(GrowArrow(grad_res_path), Write(grad_res_label), run_time=0.8)
        self.play(
            GrowArrow(grad_res_bn),
            GrowArrow(grad_res_conv),
            run_time=0.5,
        )

        # derivative equation
        deriv_parts = MathTex(
            r"\frac{\partial L}{\partial \mathbf{x}}",
            r"=",
            r"\frac{\partial L}{\partial \mathbf{y}}",
            r"\bigl(",
            r"\frac{\partial \mathcal{F}}{\partial \mathbf{x}}",
            r"+",
            r"\mathbf{1}",
            r"\bigr)",
            font_size=26, color=GREEN,
        )
        deriv_parts.to_edge(DOWN, buff=0.25)

        self.play(Write(deriv_parts), run_time=1.5)

        # highlight the "+1"
        plus_one = VGroup(deriv_parts[5], deriv_parts[6])
        highlight_rect = SurroundingRectangle(plus_one, color=YELLOW, buff=0.08, stroke_width=2.5)
        self.play(Create(highlight_rect), run_time=0.5)

        emphasis = Text(
            "Gradient flows directly through skip connection!",
            font_size=18, color=YELLOW,
        )
        emphasis.next_to(deriv_parts, DOWN, buff=0.15)
        self.play(Write(emphasis), run_time=1)
        self.wait(1.5)

        scene4_grad = VGroup(
            grad_from_top, grad_label, grad_skip_path, grad_skip_label,
            grad_res_path, grad_res_label, grad_res_bn, grad_res_conv,
        )
        scene4_block = VGroup(
            conv_4, bn_4, relu_4, adder_4, skip_4, input_4,
            arrow_conv_bn_4, arrow_bn_relu_4, arrow_relu_add_4,
        )

        # =====================================================================
        # transition → SCENE 5
        # =====================================================================
        self.play(
            FadeOut(scene4_block),
            FadeOut(scene4_grad),
            FadeOut(deriv_parts),
            FadeOut(highlight_rect),
            FadeOut(emphasis),
            FadeOut(title2),
            run_time=1,
        )
        self.wait(0.2)

        # =====================================================================
        # SCENE 5: Projection Shortcuts (Dimension Mismatch)
        # =====================================================================
        title5 = Text("Projection Shortcuts: Dimension Mismatch", font_size=28, color=PURPLE)
        title5.to_edge(UP, buff=0.15)
        self.play(Write(title5), run_time=0.8)

        # block with dimension labels
        conv_5 = make_layer(conv_pos, "Conv", ORANGE, 0.2)
        bn_5 = make_layer(bn_pos, "BN", ORANGE, 0.15)
        relu_5 = make_layer(relu_pos, "ReLU", ORANGE, 0.2)
        adder_5 = adder_node.copy()
        input_5 = MathTex("x", font_size=22, color=BLUE).next_to(input_pos, LEFT, buff=0.15)

        arrow_conv_bn_5 = Arrow(conv_pos, bn_pos, color=ORANGE, stroke_width=3, buff=0.12)
        arrow_bn_relu_5 = Arrow(bn_pos, relu_pos, color=ORANGE, stroke_width=3, buff=0.12)
        arrow_relu_add_5 = Arrow(relu_pos, adder_pos, color=ORANGE, stroke_width=3, buff=0.12)

        skip_5 = CurvedArrow(
            input_pos, adder_pos - np.array([0.15, 0, 0]),
            angle=-TAU / 7, color=BLUE, stroke_width=4,
        )

        self.play(
            FadeIn(conv_5), FadeIn(bn_5), FadeIn(relu_5),
            FadeIn(adder_5), FadeIn(input_5),
            FadeIn(arrow_conv_bn_5), FadeIn(arrow_bn_relu_5), FadeIn(arrow_relu_add_5),
            FadeIn(skip_5),
            run_time=1.2,
        )

        # dimension labels
        dim_in = MathTex("64", font_size=18, color=BLUE).next_to(input_pos, DOWN, buff=0.35)
        dim_out = MathTex("128", font_size=18, color=ORANGE).next_to(relu_pos, UP, buff=0.15).shift(RIGHT * 0.5)
        dim_skip_fail = MathTex(
            r"64 \neq 128", font_size=18, color=RED
        ).next_to(skip_5, DOWN, buff=0.15)

        self.play(Write(dim_in), Write(dim_out))
        self.play(Write(dim_skip_fail))
        self.wait(0.5)

        # show the mismatch visually – red X on the skip connection
        fail_x = Cross(stroke_color=RED, stroke_width=4)
        fail_x.move_to(skip_5.get_center())
        self.play(Create(fail_x), run_time=0.5)
        self.wait(0.3)

        # introduce 1×1 Conv on skip path
        proj_conv_pos = np.array([0.5, -0.9, 0])
        proj_conv = make_layer(proj_conv_pos, "1×1\nConv", BLUE, 0.25)
        proj_conv.scale(0.75)

        proj_label = MathTex(r"W_s", font_size=20, color=BLUE)
        proj_label.next_to(proj_conv, DOWN, buff=0.1)

        # new skip path through the projection
        new_skip_in = Arrow(input_pos, proj_conv_pos, color=BLUE, stroke_width=3.5, buff=0.08)
        new_skip_out = Arrow(proj_conv_pos, adder_pos - np.array([0.15, 0, 0]), color=BLUE, stroke_width=3.5, buff=0.08)

        # updated formula
        updated_formula = MathTex(
            r"\mathbf{y} = \mathcal{F}(\mathbf{x}) + W_s \mathbf{x}",
            font_size=26, color=PURPLE,
        )
        updated_formula.to_edge(DOWN, buff=0.25)

        self.play(FadeIn(proj_conv), Write(proj_label), run_time=1)
        self.play(
            FadeOut(skip_5),
            FadeOut(fail_x),
            FadeOut(dim_skip_fail),
            GrowArrow(new_skip_in),
            GrowArrow(new_skip_out),
            run_time=1,
        )

        # dimension after projection
        dim_proj = MathTex("128", font_size=18, color=BLUE).next_to(
            proj_conv, DOWN, buff=0.35
        )
        self.play(Write(dim_proj), run_time=0.3)
        self.wait(0.3)

        self.play(Write(updated_formula), run_time=1.5)

        # success indicator
        success = Text("Dimensions match! ✓", font_size=20, color=GREEN)
        success.next_to(adder_pos, UP, buff=0.3)
        self.play(Write(success), run_time=0.5)
        self.wait(1.5)

        # final fade-out
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )
        self.wait(1)
