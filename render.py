import bpy
import argparse
import math
import mathutils
import time
from pathlib import Path

from .base import AnimerBase
from .utils import format_duration


class AnimerRenderer(AnimerBase):
    def __init__(
        self,
        input_path: str,
        engine: str,
        res_x: int,
        res_y: int,
        focal: int,
        start_frame: int,
        duration: float,
        style: str,
        samples: int,
        use_gpu: bool,
        version: str = "debug"
    ):
        super().__init__()
        
        self.base_path = Path(__file__).resolve().parent
        self.input_path = Path(input_path)
        self.engine = engine
        self.res_x = res_x
        self.res_y = res_y
        self.focal = focal
        self.start_frame = start_frame
        self.duration = duration
        self.style = style
        self.samples = samples
        self.use_gpu = use_gpu

        self.output_path = self.base_path / "outputs" / version / f"retargeting_{style}" / (self.input_path.stem + f"_w{res_x}_h{res_y}_f{focal}_s{start_frame}_d{duration}_{style}_e{engine}_sample{samples}.mp4")

        self.style_setup_func = {
            "sketch": self._setup_sketch_style,
            "toon": self._setup_toon_style
        }

    def _setup_sketch_style(self, scene):
        """
        Configure manga/sketch manuscript style (黑白漫画手稿风格).
        
        Creates a black and white line art look similar to:
        - Manga manuscripts (漫画原稿)
        - Pencil/ink sketches
        - Comic book line art
        
        Key features:
        - White background with black line art
        - Variable line thickness for hand-drawn feel
        - Optional halftone/crosshatch shading
        - Clean, high-contrast output
        """
        self.logger.info("Setting up sketch/manga manuscript style")
        
        # ============ 1. Configure Freestyle for Line Art ============
        scene.render.use_freestyle = True
        scene.render.line_thickness = 1.5  # Base line thickness
        
        for view_layer in scene.view_layers:
            view_layer.use_freestyle = True
            fs = view_layer.freestyle_settings
            fs.use_culling = True
            fs.crease_angle = 2.0  # ~115 degrees, catches more edges
            
            # Clear existing linesets and create fresh ones
            while fs.linesets:
                fs.linesets.remove(fs.linesets[0])
            
            # ----- Main Outline Lineset -----
            main_lineset = fs.linesets.new("MainOutline")
            main_lineset.select_silhouette = True
            main_lineset.select_border = True
            main_lineset.select_contour = True
            main_lineset.select_crease = False  # Separate lineset for creases
            main_lineset.select_edge_mark = True
            
            main_style = main_lineset.linestyle
            main_style.color = (0.0, 0.0, 0.0)  # Pure black
            main_style.thickness = 2.0  # Thicker main outlines
            
            # Add thickness variation for hand-drawn feel
            self._add_sketch_line_modifiers(main_style, variation_strength=0.3)
            
            # ----- Detail/Crease Lineset (thinner lines) -----
            detail_lineset = fs.linesets.new("DetailLines")
            detail_lineset.select_silhouette = False
            detail_lineset.select_border = False
            detail_lineset.select_contour = False
            detail_lineset.select_crease = True
            detail_lineset.select_material_boundary = True
            
            detail_style = detail_lineset.linestyle
            detail_style.color = (0.1, 0.1, 0.1)  # Slightly lighter
            detail_style.thickness = 1.0  # Thinner detail lines
            
            self._add_sketch_line_modifiers(detail_style, variation_strength=0.2)
        
        # ============ 2. Convert Materials to Sketch Style ============
        for mat in bpy.data.materials:
            if not mat.use_nodes or mat.library:
                continue
            self._convert_material_to_sketch(mat)
        
        # ============ 3. Set up Compositor for Final Look ============
        self._setup_sketch_compositor(scene)
    
    def _add_sketch_line_modifiers(self, linestyle, variation_strength=0.3):
        """
        Add modifiers to Freestyle linestyle for hand-drawn effect.
        
        Args:
            linestyle: The Freestyle linestyle to modify
            variation_strength: How much thickness variation (0.0-1.0)
        """
        # Clear existing modifiers
        while linestyle.thickness_modifiers:
            linestyle.thickness_modifiers.remove(linestyle.thickness_modifiers[0])
        
        # 1. Along Stroke - taper at ends like a brush stroke
        along_stroke = linestyle.thickness_modifiers.new(name="AlongStroke", type="ALONG_STROKE")
        along_stroke.blend = "MULTIPLY"
        along_stroke.influence = variation_strength
        # Configure curve for tapered ends
        along_stroke.mapping = "CURVE"
        curve = along_stroke.curve
        # Curve points: start thin, thick in middle, thin at end
        curve.curves[0].points[0].location = (0.0, 0.7)
        curve.curves[0].points[-1].location = (1.0, 0.7)
        # Add middle point for thickness
        curve.curves[0].points.new(0.5, 1.0)
        curve.update()
        
        # 2. Noise - slight randomness for organic feel
        noise_mod = linestyle.thickness_modifiers.new(name="Noise", type="NOISE")
        noise_mod.blend = "ADD"
        noise_mod.influence = variation_strength * 0.5
        noise_mod.amplitude = 1.5
        noise_mod.period = 15.0
        
        # 3. Distance from Camera - thinner lines for distant objects
        distance_mod = linestyle.thickness_modifiers.new(name="Distance", type="DISTANCE_FROM_CAMERA")
        distance_mod.blend = "MULTIPLY"
        distance_mod.influence = 0.3
        distance_mod.range_min = 0.5
        distance_mod.range_max = 20.0
        distance_mod.value_min = 1.0
        distance_mod.value_max = 0.5
    
    def _convert_material_to_sketch(self, mat):
        """
        Convert material to sketch/manuscript style.
        
        Options:
        - Pure white (clean line art)
        - Grayscale shading (adds depth)
        - Halftone dots (manga screentone effect)
        """
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find output node
        output_node = None
        tex_node = None
        for node in nodes:
            if node.type == "OUTPUT_MATERIAL":
                output_node = node
            elif node.type == "TEX_IMAGE" and tex_node is None:
                tex_node = node
        
        if not output_node:
            return
        
        # Clear existing connections
        for link in list(output_node.inputs["Surface"].links):
            links.remove(link)
        
        # Remove old shader nodes
        nodes_to_remove = [n for n in nodes if n.type in [
            "BSDF_PRINCIPLED", "BSDF_DIFFUSE", "BSDF_GLOSSY", 
            "BSDF_TOON", "EMISSION", "SHADER_TO_RGB"
        ]]
        for node in nodes_to_remove:
            nodes.remove(node)
        
        # ============ Sketch Material Setup ============
        
        # 1. Get shading info from geometry
        geometry = nodes.new(type="ShaderNodeNewGeometry")
        geometry.location = (-600, 200)
        
        # 2. Dot product for basic shading
        dot_product = nodes.new(type="ShaderNodeVectorMath")
        dot_product.location = (-400, 200)
        dot_product.operation = "DOT_PRODUCT"
        links.new(geometry.outputs["Normal"], dot_product.inputs[0])
        links.new(geometry.outputs["Incoming"], dot_product.inputs[1])
        
        # 3. Adjust range to 0-1
        math_add = nodes.new(type="ShaderNodeMath")
        math_add.location = (-200, 200)
        math_add.operation = "ADD"
        math_add.inputs[1].default_value = 1.0
        math_add.use_clamp = True
        links.new(dot_product.outputs["Value"], math_add.inputs[0])
        
        math_multiply = nodes.new(type="ShaderNodeMath")
        math_multiply.location = (-50, 200)
        math_multiply.operation = "MULTIPLY"
        math_multiply.inputs[1].default_value = 0.5
        math_multiply.use_clamp = True
        links.new(math_add.outputs["Value"], math_multiply.inputs[0])
        
        # 4. ColorRamp for sketch shading (white with gray shadows)
        color_ramp = nodes.new(type="ShaderNodeValToRGB")
        color_ramp.location = (150, 200)
        color_ramp.color_ramp.interpolation = "CONSTANT"  # Sharp manga-style
        
        cr = color_ramp.color_ramp
        # Three levels: dark shadow, light shadow, white
        cr.elements[0].position = 0.0
        cr.elements[0].color = (0.85, 0.85, 0.85, 1.0)  # Light gray shadow
        cr.elements[1].position = 0.3
        cr.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # White (main area)
        
        links.new(math_multiply.outputs["Value"], color_ramp.inputs["Fac"])
        
        # 5. Optional: Add halftone/screentone effect for shadows
        # Uncomment below for manga screentone effect
        # halftone = self._create_halftone_node(nodes, links)
        # ... mix with color_ramp output
        
        # 6. Emission shader (flat, no lighting influence)
        emission = nodes.new(type="ShaderNodeEmission")
        emission.location = (400, 200)
        emission.inputs["Strength"].default_value = 1.0
        links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
        
        # 7. Handle transparency
        if tex_node and "Alpha" in tex_node.outputs:
            transparent = nodes.new(type="ShaderNodeBsdfTransparent")
            transparent.location = (400, 0)
            
            mix_shader = nodes.new(type="ShaderNodeMixShader")
            mix_shader.location = (600, 100)
            
            links.new(tex_node.outputs["Alpha"], mix_shader.inputs["Fac"])
            links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
            links.new(emission.outputs["Emission"], mix_shader.inputs[2])
            links.new(mix_shader.outputs["Shader"], output_node.inputs["Surface"])
            
            mat.blend_method = "HASHED"
            mat.shadow_method = "NONE"  # No shadows for sketch style
        else:
            links.new(emission.outputs["Emission"], output_node.inputs["Surface"])
        
        # Disable shadows for clean look
        mat.shadow_method = "NONE"
    
    def _setup_sketch_compositor(self, scene, use_paper_texture=True, paper_intensity=0.15):
        """
        Set up compositor nodes for final sketch effect.
        Adds contrast, paper texture, and post-processing.
        
        Args:
            scene: Blender scene object
            use_paper_texture: Whether to add paper texture overlay
            paper_intensity: Strength of paper texture (0.0-1.0)
        """
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links
        
        # Clear default nodes
        for node in nodes:
            nodes.remove(node)
        
        # ============ 1. Render Input ============
        render_layers = nodes.new(type="CompositorNodeRLayers")
        render_layers.location = (0, 400)
        
        # ============ 2. Convert to Black & White ============
        rgb_to_bw = nodes.new(type="CompositorNodeRGBToBW")
        rgb_to_bw.location = (200, 400)
        links.new(render_layers.outputs["Image"], rgb_to_bw.inputs["Image"])
        
        # ============ 3. Contrast Adjustment ============
        color_ramp = nodes.new(type="CompositorNodeValToRGB")
        color_ramp.location = (400, 400)
        color_ramp.color_ramp.interpolation = "EASE"
        
        cr = color_ramp.color_ramp
        cr.elements[0].position = 0.08  # Crush blacks
        cr.elements[0].color = (0.02, 0.02, 0.02, 1)  # Near black
        cr.elements[1].position = 0.55  # Lift whites
        cr.elements[1].color = (0.98, 0.97, 0.95, 1)  # Slightly warm white (paper tone)
        
        links.new(rgb_to_bw.outputs["Val"], color_ramp.inputs["Fac"])
        
        # Current output for chaining
        current_output = color_ramp.outputs["Image"]
        current_x = 600
        
        # ============ 4. Paper Texture (Procedural) ============
        if use_paper_texture:
            paper_output, paper_x = self._create_paper_texture_nodes(
                nodes, links, current_x, paper_intensity
            )
            
            # Mix paper with line art
            paper_mix = nodes.new(type="CompositorNodeMixRGB")
            paper_mix.location = (paper_x + 200, 400)
            paper_mix.blend_type = "MULTIPLY"
            paper_mix.inputs[0].default_value = paper_intensity  # Fac
            
            # Compositor MixRGB uses indices: [0]=Fac, [1]=Image1, [2]=Image2
            links.new(current_output, paper_mix.inputs[1])
            links.new(paper_output, paper_mix.inputs[2])
            
            current_output = paper_mix.outputs["Image"]
            current_x = paper_x + 400
        
        # ============ 5. Final Adjustments ============
        # Slight vignette for traditional feel (optional)
        # vignette = self._create_vignette_nodes(nodes, links, current_x)
        
        # ============ 6. Output ============
        composite = nodes.new(type="CompositorNodeComposite")
        composite.location = (current_x, 400)
        links.new(current_output, composite.inputs["Image"])
        
        viewer = nodes.new(type="CompositorNodeViewer")
        viewer.location = (current_x, 200)
        links.new(current_output, viewer.inputs["Image"])
    
    def _create_paper_texture_nodes(self, nodes, links, start_x, intensity=0.15):
        """
        Create procedural paper texture using compositor nodes.
        
        Paper texture consists of:
        - Fine grain noise (paper fiber)
        - Larger scale variation (paper thickness)
        - Subtle color tint (aged paper)
        
        Returns:
            tuple: (output_socket, end_x_position)
        """
        x = start_x
        y_offset = -200  # Below main chain
        
        # ============ Fine Grain (Paper Fiber) ============
        # Use STUCCI texture for fine paper grain (has noise_scale property)
        
        paper_tex_name = "PaperGrainTexture"
        if paper_tex_name not in bpy.data.textures:
            paper_tex = bpy.data.textures.new(paper_tex_name, type="STUCCI")
            paper_tex.noise_scale = 0.08  # Fine grain for paper fiber
            paper_tex.turbulence = 5.0
            paper_tex.stucci_type = "PLASTIC"  # Gives paper-like variation
        else:
            paper_tex = bpy.data.textures[paper_tex_name]
        
        # Texture node in compositor
        tex_node = nodes.new(type="CompositorNodeTexture")
        tex_node.location = (x, y_offset)
        tex_node.texture = paper_tex
        
        x += 200
        
        # ============ Adjust Texture Levels ============
        # ColorRamp to control paper texture contrast
        paper_ramp = nodes.new(type="CompositorNodeValToRGB")
        paper_ramp.location = (x, y_offset)
        paper_ramp.color_ramp.interpolation = "LINEAR"
        
        pr = paper_ramp.color_ramp
        # Create subtle paper variation (mostly white with slight texture)
        pr.elements[0].position = 0.0
        pr.elements[0].color = (0.92, 0.90, 0.87, 1)  # Slightly darker (paper shadow)
        pr.elements[1].position = 1.0
        pr.elements[1].color = (1.0, 0.99, 0.97, 1)   # Paper highlight (warm white)
        
        links.new(tex_node.outputs["Value"], paper_ramp.inputs["Fac"])
        
        x += 200
        
        # ============ Add Larger Scale Variation ============
        # Create CLOUDS texture for broader paper thickness variation
        broad_tex_name = "PaperBroadTexture"
        if broad_tex_name not in bpy.data.textures:
            broad_tex = bpy.data.textures.new(broad_tex_name, type="CLOUDS")
            broad_tex.noise_scale = 1.5  # Larger scale variation
            broad_tex.noise_depth = 2
            broad_tex.noise_type = "SOFT_NOISE"  # Softer for paper
        else:
            broad_tex = bpy.data.textures[broad_tex_name]
        
        broad_node = nodes.new(type="CompositorNodeTexture")
        broad_node.location = (start_x, y_offset - 200)
        broad_node.texture = broad_tex
        
        # Adjust broad variation
        broad_ramp = nodes.new(type="CompositorNodeValToRGB")
        broad_ramp.location = (start_x + 200, y_offset - 200)
        broad_ramp.color_ramp.interpolation = "LINEAR"
        
        br = broad_ramp.color_ramp
        br.elements[0].position = 0.3
        br.elements[0].color = (0.95, 0.94, 0.92, 1)
        br.elements[1].position = 0.7
        br.elements[1].color = (1.0, 0.99, 0.98, 1)
        
        links.new(broad_node.outputs["Value"], broad_ramp.inputs["Fac"])
        
        # ============ Combine Fine and Broad Textures ============
        combine_mix = nodes.new(type="CompositorNodeMixRGB")
        combine_mix.location = (x, y_offset - 100)
        combine_mix.blend_type = "MULTIPLY"
        combine_mix.inputs[0].default_value = 0.5  # Fac: Balance between textures
        
        # Compositor MixRGB uses indices: [0]=Fac, [1]=Image1, [2]=Image2
        links.new(paper_ramp.outputs["Image"], combine_mix.inputs[1])
        links.new(broad_ramp.outputs["Image"], combine_mix.inputs[2])
        
        x += 200
        
        return combine_mix.outputs["Image"], x
    
    def _create_vignette_nodes(self, nodes, links, start_x):
        """
        Create vignette effect for traditional manuscript feel.
        Optional enhancement for sketch style.
        
        Returns:
            output_socket
        """
        x = start_x
        y_offset = -400
        
        # Ellipse mask for vignette
        ellipse = nodes.new(type="CompositorNodeEllipseMask")
        ellipse.location = (x, y_offset)
        ellipse.width = 1.0
        ellipse.height = 1.0
        
        # Blur the mask edges
        blur = nodes.new(type="CompositorNodeBlur")
        blur.location = (x + 200, y_offset)
        blur.size_x = 200
        blur.size_y = 200
        blur.use_relative = True
        blur.factor_x = 0.3
        blur.factor_y = 0.3
        
        links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
        
        # Invert for vignette (dark edges)
        invert = nodes.new(type="CompositorNodeInvert")
        invert.location = (x + 400, y_offset)
        links.new(blur.outputs["Image"], invert.inputs["Color"])
        
        # Adjust vignette strength
        vignette_ramp = nodes.new(type="CompositorNodeValToRGB")
        vignette_ramp.location = (x + 600, y_offset)
        
        vr = vignette_ramp.color_ramp
        vr.elements[0].position = 0.0
        vr.elements[0].color = (1, 1, 1, 1)
        vr.elements[1].position = 0.8
        vr.elements[1].color = (0.9, 0.88, 0.85, 1)  # Subtle darkening
        
        links.new(invert.outputs["Color"], vignette_ramp.inputs["Fac"])
        
        return vignette_ramp.outputs["Image"]

    def _setup_toon_style(self, scene):
        """
        Configure anime/toon style shading with support for both EEVEE and Cycles.
        
        - EEVEE: Uses Shader to RGB + ColorRamp (best quality)
        - Cycles: Uses Toon BSDF + custom shading setup
        
        Key features:
        - 2-3 level cel-shading with sharp transitions
        - Rim lighting for edge highlights (anime style)
        - Preserves original textures
        - Handles transparency properly
        - Enables Freestyle outlines
        """
        render_engine = scene.render.engine  # "CYCLES" or "BLENDER_EEVEE"
        self.logger.info(f"Setting up toon style for render engine: {render_engine}")
        
        # ============ 1. Enable Freestyle for outlines ============
        # great line arts but very slow (for rendering speed, disable it)
        # scene.render.use_freestyle = True
        # scene.render.line_thickness = 1.2
        
        # for view_layer in scene.view_layers:
        #     view_layer.material_override = None
        #     view_layer.use_freestyle = True
        #     fs = view_layer.freestyle_settings
        #     fs.use_culling = True
        #     fs.crease_angle = 2.5  # ~143 degrees
            
        #     if not fs.linesets:
        #         lineset = fs.linesets.new("ToonOutline")
        #     else:
        #         lineset = fs.linesets[0]
            
        #     # Configure outline detection
        #     lineset.select_silhouette = True
        #     lineset.select_crease = True
        #     lineset.select_border = True
        #     lineset.select_contour = True
            
        #     # Black outlines with slight thickness variation
        #     linestyle = lineset.linestyle
        #     linestyle.color = (0.05, 0.05, 0.05)  # Near black
        #     linestyle.thickness = 1.5
        
        # ============ 2. Configure materials for cel-shading ============
        use_cycles = (render_engine == "CYCLES")
        
        for mat in bpy.data.materials:
            if not mat.use_nodes or mat.library:
                continue
            
            if use_cycles:
                self._convert_material_to_toon_cycles(mat)
            else:
                self._convert_material_to_toon_eevee(mat)

        # ============ 3. apply inverted hull to draw outlines ============
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                self._apply_inverted_hull_outline(obj)
    
    def _get_material_base_info(self, mat):
        """Extract base color and texture info from existing material."""
        nodes = mat.node_tree.nodes
        
        tex_node = None
        output_node = None
        principled_node = None
        base_color = (0.8, 0.8, 0.8, 1.0)
        
        for node in nodes:
            if node.type == "TEX_IMAGE" and tex_node is None:
                tex_node = node
            elif node.type == "OUTPUT_MATERIAL":
                output_node = node
            elif node.type == "BSDF_PRINCIPLED":
                principled_node = node
        
        if principled_node and not principled_node.inputs["Base Color"].is_linked:
            base_color = tuple(principled_node.inputs["Base Color"].default_value)
        
        return tex_node, output_node, principled_node, base_color
    
    def _cleanup_material_nodes(self, mat, output_node):
        """Remove old shader nodes and clear connections."""
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing surface connections
        for link in list(output_node.inputs["Surface"].links):
            links.remove(link)
        
        # Remove old BSDF nodes
        nodes_to_remove = [n for n in nodes if n.type in [
            "BSDF_PRINCIPLED", "BSDF_DIFFUSE", "BSDF_GLOSSY", "BSDF_TOON",
            "SHADER_TO_RGB", "EMISSION"
        ]]
        for node in nodes_to_remove:
            nodes.remove(node)
                
    def _create_rim_lighting(self, nodes, links, x_offset=0, y_offset=-100):
        """Create rim lighting nodes (works for both EEVEE and Cycles)."""
        # Fresnel for edge detection
        fresnel = nodes.new(type="ShaderNodeFresnel")
        fresnel.location = (x_offset - 200, y_offset)
        fresnel.inputs["IOR"].default_value = 1.15
        
        # ColorRamp to control rim sharpness
        rim_ramp = nodes.new(type="ShaderNodeValToRGB")
        rim_ramp.location = (x_offset, y_offset)
        rim_ramp.color_ramp.interpolation = "EASE"
        rim_ramp.color_ramp.elements[0].position = 0.0
        rim_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        rim_ramp.color_ramp.elements[1].position = 0.65
        rim_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
        links.new(fresnel.outputs["Fac"], rim_ramp.inputs["Fac"])
        
        return rim_ramp
    
    def _handle_transparency(self, mat, nodes, links, tex_node, final_shader, output_node):
        """Handle alpha transparency for hair, accessories, etc."""
        if tex_node and "Alpha" in tex_node.outputs:
            transparent = nodes.new(type="ShaderNodeBsdfTransparent")
            transparent.location = (600, -100)
            
            mix_shader = nodes.new(type="ShaderNodeMixShader")
            mix_shader.location = (800, 0)
            
            links.new(tex_node.outputs["Alpha"], mix_shader.inputs["Fac"])
            links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
            links.new(final_shader.outputs[0], mix_shader.inputs[2])
            links.new(mix_shader.outputs["Shader"], output_node.inputs["Surface"])
            
            # Enable alpha blending
            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
            return True
        else:
            links.new(final_shader.outputs[0], output_node.inputs["Surface"])
            return False

    def _convert_material_to_toon_eevee(self, mat):
        """
        Convert material to toon style for EEVEE using Shader to RGB.
        This provides the best cel-shading quality in EEVEE.
        """
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        mat.metallic = 0.0
        mat.specular_intensity = 0.0
        
        tex_node, output_node, principled_node, base_color = self._get_material_base_info(mat)
        if not output_node:
            return
        
        self._cleanup_material_nodes(mat, output_node)
        
        # ============ EEVEE Cel-Shading Setup ============
        
        # 1. Diffuse BSDF for shading calculation
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse.location = (-400, 200)
        diffuse.inputs["Color"].default_value = base_color
        diffuse.inputs["Roughness"].default_value = 1.0
        
        # 2. Shader to RGB (EEVEE only)
        shader_to_rgb = nodes.new(type="ShaderNodeShaderToRGB")
        shader_to_rgb.location = (-200, 200)
        links.new(diffuse.outputs["BSDF"], shader_to_rgb.inputs["Shader"])
        
        # 3. ColorRamp for cel-shading bands
        color_ramp = nodes.new(type="ShaderNodeValToRGB")
        color_ramp.location = (0, 200)
        color_ramp.color_ramp.interpolation = "CONSTANT"
        
        cr = color_ramp.color_ramp
        cr.elements[0].position = 0.0
        cr.elements[0].color = (0.3, 0.3, 0.35, 1.0)  # Shadow
        cr.elements[1].position = 0.4
        cr.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # Lit
        highlight = cr.elements.new(0.85)
        highlight.color = (1.1, 1.1, 1.15, 1.0)  # Highlight
        
        links.new(shader_to_rgb.outputs["Color"], color_ramp.inputs["Fac"])
        
        # 4. Mix shading with texture
        mix_color = nodes.new(type="ShaderNodeMixRGB")
        mix_color.location = (200, 200)
        mix_color.blend_type = "MULTIPLY"
        mix_color.inputs["Fac"].default_value = 1.0
        links.new(color_ramp.outputs["Color"], mix_color.inputs["Color1"])
        
        if tex_node:
            links.new(tex_node.outputs["Color"], mix_color.inputs["Color2"])
        else:
            mix_color.inputs["Color2"].default_value = base_color
        
        # 5. Rim lighting
        rim_ramp = self._create_rim_lighting(nodes, links, x_offset=0, y_offset=-100)
        
        add_rim = nodes.new(type="ShaderNodeMixRGB")
        add_rim.location = (400, 100)
        add_rim.blend_type = "ADD"
        add_rim.inputs["Fac"].default_value = 0.15
        links.new(mix_color.outputs["Color"], add_rim.inputs["Color1"])
        links.new(rim_ramp.outputs["Color"], add_rim.inputs["Color2"])
        
        # 6. Emission shader for flat output
        emission = nodes.new(type="ShaderNodeEmission")
        emission.location = (600, 100)
        emission.inputs["Strength"].default_value = 1.0
        links.new(add_rim.outputs["Color"], emission.inputs["Color"])
        
        # 7. Handle transparency
        self._handle_transparency(mat, nodes, links, tex_node, emission, output_node)

    def _convert_material_to_toon_cycles(self, mat):
        """
        Convert material to toon style for Cycles.
        Uses Toon BSDF + Diffuse with Light Path for cel-shading effect.
        
        Cycles doesn't support Shader to RGB, so we use:
        - Toon BSDF for basic cel-shading
        - Diffuse + dot product with light direction for custom shading
        - Light Path node to separate direct/indirect lighting
        """
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        mat.metallic = 0.0
        mat.specular_intensity = 0.0
        
        tex_node, output_node, principled_node, base_color = self._get_material_base_info(mat)
        if not output_node:
            return
        
        self._cleanup_material_nodes(mat, output_node)
        
        # ============ Cycles Cel-Shading Setup ============
        
        # Method: Toon BSDF + custom shading with Geometry/Light info
        
        # 1. Get shading information using Geometry node
        geometry = nodes.new(type="ShaderNodeNewGeometry")
        geometry.location = (-600, 300)
        
        # 2. Create a simple light direction (use incoming ray as approximation)
        # For better results, you can use a specific light's direction
        vector_math = nodes.new(type="ShaderNodeVectorMath")
        vector_math.location = (-400, 300)
        vector_math.operation = "DOT_PRODUCT"
        # Dot product of normal and incoming direction gives shading factor
        links.new(geometry.outputs["Normal"], vector_math.inputs[0])
        links.new(geometry.outputs["Incoming"], vector_math.inputs[1])
        
        # 3. Math node to adjust and clamp the shading value
        math_add = nodes.new(type="ShaderNodeMath")
        math_add.location = (-200, 300)
        math_add.operation = "ADD"
        math_add.inputs[1].default_value = 0.5  # Shift to 0-1 range
        links.new(vector_math.outputs["Value"], math_add.inputs[0])
        
        math_clamp = nodes.new(type="ShaderNodeMath")
        math_clamp.location = (-50, 300)
        math_clamp.operation = "MULTIPLY"
        math_clamp.inputs[1].default_value = 1.0
        math_clamp.use_clamp = True
        links.new(math_add.outputs["Value"], math_clamp.inputs[0])
        
        # 4. ColorRamp for cel-shading bands
        color_ramp = nodes.new(type="ShaderNodeValToRGB")
        color_ramp.location = (100, 300)
        color_ramp.color_ramp.interpolation = "CONSTANT"
        
        cr = color_ramp.color_ramp
        cr.elements[0].position = 0.0
        cr.elements[0].color = (0.35, 0.35, 0.4, 1.0)  # Shadow (slightly cool)
        cr.elements[1].position = 0.45
        cr.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # Lit area
        
        links.new(math_clamp.outputs["Value"], color_ramp.inputs["Fac"])
        
        # 5. Mix shading with texture color
        mix_color = nodes.new(type="ShaderNodeMixRGB")
        mix_color.location = (300, 200)
        mix_color.blend_type = "MULTIPLY"
        mix_color.inputs["Fac"].default_value = 1.0
        links.new(color_ramp.outputs["Color"], mix_color.inputs["Color1"])
        
        if tex_node:
            links.new(tex_node.outputs["Color"], mix_color.inputs["Color2"])
        else:
            mix_color.inputs["Color2"].default_value = base_color
        
        # 6. Rim lighting
        rim_ramp = self._create_rim_lighting(nodes, links, x_offset=100, y_offset=-100)
        
        add_rim = nodes.new(type="ShaderNodeMixRGB")
        add_rim.location = (500, 100)
        add_rim.blend_type = "ADD"
        add_rim.inputs["Fac"].default_value = 0.12  # Slightly less for Cycles
        links.new(mix_color.outputs["Color"], add_rim.inputs["Color1"])
        links.new(rim_ramp.outputs["Color"], add_rim.inputs["Color2"])
        
        # 7. Create final shader using Diffuse BSDF (for proper light interaction)
        # Option A: Use Emission for flat look (ignores scene lighting)
        # Option B: Use Diffuse for some light interaction
        
        # Using Emission for consistent flat anime look
        emission = nodes.new(type="ShaderNodeEmission")
        emission.location = (700, 100)
        emission.inputs["Strength"].default_value = 1.0
        links.new(add_rim.outputs["Color"], emission.inputs["Color"])
        
        # Alternative: Toon BSDF for Cycles-native toon shading
        # Uncomment below and comment emission if you want Cycles lighting interaction
        # toon_bsdf = nodes.new(type="ShaderNodeBsdfToon")
        # toon_bsdf.location = (700, 100)
        # toon_bsdf.component = "DIFFUSE"
        # toon_bsdf.inputs["Size"].default_value = 0.3
        # toon_bsdf.inputs["Smooth"].default_value = 0.1
        # links.new(add_rim.outputs["Color"], toon_bsdf.inputs["Color"])
        
        # 8. Light Path for handling camera rays vs shadow rays differently
        light_path = nodes.new(type="ShaderNodeLightPath")
        light_path.location = (700, -150)
        
        # Mix between emission (for camera) and diffuse (for shadows/GI)
        diffuse_for_shadows = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse_for_shadows.location = (700, -50)
        links.new(add_rim.outputs["Color"], diffuse_for_shadows.inputs["Color"])
        
        mix_shader = nodes.new(type="ShaderNodeMixShader")
        mix_shader.location = (900, 50)
        links.new(light_path.outputs["Is Camera Ray"], mix_shader.inputs["Fac"])
        links.new(diffuse_for_shadows.outputs["BSDF"], mix_shader.inputs[1])
        links.new(emission.outputs["Emission"], mix_shader.inputs[2])
        
        # 9. Handle transparency
        self._handle_transparency(mat, nodes, links, tex_node, mix_shader, output_node)

    def _apply_inverted_hull_outline(self, obj, thickness=2e-3, color=(0.05, 0.05, 0.05, 1)):
        """
        Adds a Solidify Modifier to create a fast, GPU-accelerated outline.
        """
        # 1. Create the Outline Material (Emission Black)
        mat_name = "Outline_Material_CYCLES"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            mat.use_backface_culling = True
            mat.blend_method = "BLEND"
            mat.shadow_method = "NONE" 
            
            # Set to Emission Black (Unlit)
            tree = mat.node_tree
            tree.nodes.clear()
            
            # Output
            node_out = tree.nodes.new(type='ShaderNodeOutputMaterial')
            mix_shader = tree.nodes.new('ShaderNodeMixShader')
            mix_shadow = tree.nodes.new('ShaderNodeMixShader') # Second mix for shadows
            
            transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
            node_emit = tree.nodes.new(type='ShaderNodeEmission')
            node_emit.inputs['Color'].default_value = color
            node_emit.inputs['Strength'].default_value = 1.0

            geo_node = tree.nodes.new('ShaderNodeNewGeometry')
            light_path = tree.nodes.new('ShaderNodeLightPath')
            
            # 1. Backface Culling Logic:
            #    If 'Backfacing' is TRUE (Face pointing at camera due to flip), use Transparent.
            #    If 'Backfacing' is FALSE (Face on the other side), use Emission (The Outline).
            tree.links.new(geo_node.outputs['Backfacing'], mix_shader.inputs['Fac'])
            tree.links.new(node_emit.outputs['Emission'], mix_shader.inputs[1])     # False -> Black
            tree.links.new(transparent.outputs['BSDF'], mix_shader.inputs[2])      # True -> Invisible

            # 2. Shadow Culling Logic:
            #    If 'Is Shadow Ray' is TRUE, use Transparent (Don't cast shadow).
            #    Else, use the result of the previous mix.
            tree.links.new(light_path.outputs['Is Shadow Ray'], mix_shadow.inputs['Fac'])
            tree.links.new(mix_shader.outputs['Shader'], mix_shadow.inputs[1])
            tree.links.new(transparent.outputs['BSDF'], mix_shadow.inputs[2])
            
            tree.links.new(mix_shadow.outputs['Shader'], node_out.inputs['Surface'])

        # 2. Assign Material to Object (Last Slot)
        # Check if already assigned to avoid duplicates
        if obj.data.materials and obj.data.materials[-1].name == mat_name:
            mat_index = len(obj.data.materials) - 1
        else:
            obj.data.materials.append(mat)
            mat_index = len(obj.data.materials) - 1

        # 3. Add Solidify Modifier
        mod_name = "Outline_Solidify"
        if mod_name in obj.modifiers:
            mod = obj.modifiers[mod_name]
        else:
            mod = obj.modifiers.new(name=mod_name, type='SOLIDIFY')
        
        mod.thickness = thickness
        mod.offset = 1.0            # Push OUTWARDS
        mod.use_flip_normals = True # The magic trick: Inverse Normals
        mod.material_offset = mat_index # Use the black material we just added
        mod.use_quality_normals = True # Better corners
        # mod.material_offset_param = mat_index # For newer blender versions, ensures correct slot
        
        # Vertex Group Masking (Optional)
        # If MMD model has a 'Outline' vertex group (common), use it to vary thickness
        if "Outline" in obj.vertex_groups:
            mod.vertex_group = "Outline"

    def _set_render_configs(self, scene, fps=30):
        scene.frame_start = self.start_frame
        scene.frame_end = self.start_frame + int(fps * self.duration)

        render = scene.render
        render.engine = self.engine
        render.resolution_x = self.res_x
        render.resolution_y = self.res_y
        render.resolution_percentage = 100

        if self.engine == "CYCLES":      
            if self.use_gpu:
                scene.cycles.device = "GPU"
            
                # ensure CUDA/OPTIX is actually enabled in preferences
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.get_devices_for_type("CUDA")
                prefs.compute_device_type = "CUDA"
        else:
            if self.engine == "CYCLES":
                scene.cycles.device = "CPU"

        # use Noise Threshold to stop rendering when a frame is "clean enough"
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.samples = self.samples 
        scene.cycles.denoiser = "OPENIMAGEDENOISE" # high quality denoising
        
        # output settings
        render.image_settings.file_format = "FFMPEG"
        render.ffmpeg.format = "MPEG4"
        render.ffmpeg.codec = "H264"
        render.filepath = str(self.output_path)

    def _set_active_camera_portrait(self, scene):
        if not scene.camera:
            self.logger.info("No active camera found. Creating one...")

            cam_data = bpy.data.cameras.new("Camera")
            cam_obj = bpy.data.objects.new("Camera", cam_data)
            scene.collection.objects.link(cam_obj)
            scene.camera = cam_obj        
        cam = scene.camera
        cam.data.lens = self.focal

        # frame the character
        # move camera back along Y and up to hip/chest height
        armature_obj, armature_center, armature_height = self._calculate_armature_anim_range()
        fov = cam.data.angle
        distance = (armature_height * 1.2) / (2 * math.tan(fov / 2))
        cam.location = (0, -distance * 1.5, armature_height * 0.6)
        cam.rotation_euler = (math.radians(90), 0, 0)

        constraint = cam.constraints.new(type='TRACK_TO')
        constraint.target = armature_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

    def _set_active_camera_landscape(self, scene):
        if not scene.camera:
            self.logger.info("No active camera found. Creating one...")

            cam_data = bpy.data.cameras.new("Camera")
            cam_obj = bpy.data.objects.new("Camera", cam_data)
            scene.collection.objects.link(cam_obj)
            scene.camera = cam_obj
        cam = scene.camera
        cam.data.lens = self.focal
        # cam.data.sensor_fit = "HORIZONTAL"

        # frame the character
        # move camera back along Y and up to hip/chest height
        armature_obj, armature_z_coords, armature_height = self._calculate_armature_anim_range()
        aspect_ratio = self.res_x / self.res_y
        horizontal_fov = cam.data.angle
        vertical_fov = 2 * math.atan(math.tan(horizontal_fov / 2) / aspect_ratio)
        distance = (armature_height * 1.2) / (2 * math.tan(vertical_fov / 2))
        
        cam.location = (0, -distance, armature_height*0.6)  # min(armature_z_coords)+(armature_height*0.6)
        cam.rotation_euler = (math.radians(90), 0, 0)   # will be overrided by constraint.target

        constraint = cam.constraints.new(type='TRACK_TO')
        constraint.target = armature_obj
        constraint.subtarget = "上半身"
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"

    def _calculate_armature_anim_range(self):
        armature_name = None
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature_name = obj.name
                break
        armature = bpy.data.objects.get(armature_name)
        if not armature:
            self.logger.error(f"Armature object not found: {armature_name}")
            raise RuntimeError(f"Armature object not found: {armature_name}")
        target_obj = bpy.data.objects.get(armature_name)
        
        # get the bounding box in world space, center, and character height
        bbox_corners = [target_obj.matrix_world @ mathutils.Vector(corner) for corner in target_obj.bound_box]
        z_coords = [v.z for v in bbox_corners]
        height = max(c.z for c in bbox_corners) - min(c.z for c in bbox_corners)

        return target_obj, z_coords, height
    
    def load_blend_file(self):
        if self.input_path.exists():
            bpy.ops.wm.open_mainfile(filepath=str(self.input_path), load_ui=False)
            self.logger.info(f"Successfully loaded: {self.input_path}")
        else:
            self.logger.error(f"Error: File not found at {self.input_path}")
            raise RuntimeError(f"Error: File not found at {self.input_path}")
        return

    def render_animation(self):
        # set up render configs
        scene = bpy.context.scene
        scene.cycles.use_tiling = False
        setup_func = self.style_setup_func.get(self.style, None)
        if setup_func:
            setup_func(scene)
        self._set_render_configs(scene)

        # setup camera
        if self.res_x < self.res_y:
            self._set_active_camera_portrait(scene)
        else:
            self._set_active_camera_landscape(scene)

        # render
        self.logger.info(f"rendering with {self.engine}, saving to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(animation=True)

    def execute(self):
        start_time = time.time()
        try:
            self.load_blend_file()
            self.render_animation()
            elapsed_time = format_duration(time.time() - start_time)
            
            self.logger.critical(f"✅ Rendering finished, check: {self.output_path}")
            self.logger.critical(f"Time elapsed: {elapsed_time}. File: {self.output_path}")
        except Exception as e:
            self.logger.error(f"Rendering failed: {str(e)}")
            raise RuntimeError(f"Rendering failed: {str(e)}")
        finally:
            self._quit()
            self.logger.info("Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--res_x", type=int, default=1920)
    parser.add_argument("--res_y", type=int, default=1080)
    parser.add_argument("--focal", type=int, default=65)
    parser.add_argument("--start_frame", type=int, default=60)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    renderer = AnimerRenderer(
        args.input_path,
        args.engine,
        args.res_x,
        args.res_y,
        args.focal,
        args.start_frame,
        args.duration,
        args.style,
        args.samples,
        args.use_gpu,
    )
    renderer.execute()

    # bpy.ops.outliner.orphans_purge() this code do memory clear