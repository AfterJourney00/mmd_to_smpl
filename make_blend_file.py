import bpy
import argparse
import time
from pathlib import Path

from .base import AnimerBase
from .utils import standardize_file_name, format_duration

class AnimerBlendComposer(AnimerBase):
    def __init__(
        self,
        tri_path: dict,
        skip_animation_morph: int = 60,
        pre_roll_morph: int = 60,
        fps: int = 30,
        duration: float = 10.0,
        bake: int = True,
        no_physics: bool = False,
        disable_ik: bool =False,
        version: str = "debug"
    ):
        super().__init__()

        self.base_path = Path(__file__).resolve().parent
        self.pmx_path = tri_path["pmx_path"]
        self.vmd_path = tri_path["vmd_path"]
        self.blend_path = tri_path["blend_path"]
        self.skip_animation_morph = skip_animation_morph
        self.pre_roll_morph = pre_roll_morph
        self.fps = fps
        self.duration = duration

        self.character_name = standardize_file_name(f"{self.pmx_path.stem}")
        self.motion_name = standardize_file_name(f"{self.vmd_path.stem}")
        self.scene_name = standardize_file_name(f"{self.blend_path.stem}")

        # config baking
        self.bake = bake
        self.no_physics = no_physics
        self.disable_ik = disable_ik
        
        output_path = f"M_{self.motion_name}_C_{self.character_name}_S_{self.scene_name}"
        if self.no_physics:
            output_path += "_nophys"
        else:
            output_path += "_phys"
        
        if self.disable_ik:
            output_path += "_disableik"
        else:
            output_path += "_ik"
        output_path += f"_skip{self.skip_animation_morph}frames_duration{self.duration}s.blend"

        self.output_path = self.base_path / "outputs" / version / f"blend_files" / output_path

    def setup_blender_scene(self):
        self.mmd_tools_module = "mmd_tools"
        self.mbts_module = "MBTs_NG"
        
        try:
            # load scene and clean existing assets may cause conflicts
            self._load_initial_scene(self.blend_path)
            self._cleanup_existing_objects()
            scene = bpy.context.scene
            scene.render.fps = self.fps
            scene.render.fps_base = 1.0
            self.logger.info(f"Set scene frame rate to {self.fps}fps (fps={scene.render.fps}, fps_base={scene.render.fps_base})")
            
            # enable adds on
            self._ensure_addon_loaded(self.mmd_tools_module, required=True) # enables blender extensions (mmd_tools is true)
            mbts_loaded = self._ensure_addon_loaded(self.mbts_module, required=False) # (mbts_ng is false, no need this time)

            # load the first character in the character queue
            self._import_mmd_model()
            armature = self._select_character_armature()
            
            # load MMD animation
            self._import_vmd_animation()

            # inject pre-roll morph transition
            if self.pre_roll_morph > 0:
                self._inject_preroll_transition_v2(armature)

        except Exception as e:
            self._quit()
            self.logger.error(f"Processing failed: {str(e)}")
            raise RuntimeError(f"Processing failed: {str(e)}")

    def select_character_mesh(self):
        mesh_name = None
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                mesh_name = obj.name
                break
        mesh_obj = bpy.data.objects.get(mesh_name)
        
        if not mesh_obj:
            self.logger.error(f"Armature object not found: {mesh_name}")
            raise RuntimeError(f"Armature object not found: {mesh_name}")
        if mesh_obj.name not in bpy.context.view_layer.objects:
            self.logger.error(f"Object '{mesh_name}' exists but not in current view layer")
            raise RuntimeError(f"Object '{mesh_name}' exists but not in current view layer")
        
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        self.logger.info(f"Selected object: {mesh_name}")

    def configure_physics_settings(self, start_frame, end_frame):
        """
        Configure physics settings and set frame range for ALL point caches.
        
        This ensures that bpy.ops.ptcache.bake_all only bakes the specified range.
        """
        scene = bpy.context.scene
        scene.frame_start = start_frame
        scene.frame_end = end_frame
        
        rbw = scene.rigidbody_world
        if not rbw:
            # Try to create rigidbody world
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.rigidbody.world_add()
                rbw = scene.rigidbody_world  # Re-get after creation
                self.logger.info("Created new rigidbody world")
            except Exception as e:
                self.logger.warning(f"Could not create rigidbody world: {e}")
        
        if rbw:
            rbw.enabled = True
            rbw.substeps_per_frame = 1
            rbw.solver_iterations = 60
            
            # Set rigidbody world cache frame range
            cache = rbw.point_cache
            cache.frame_start = start_frame
            cache.frame_end = end_frame
            self.logger.info(
                f"Rigidbody world configured: frame range {cache.frame_start}-{cache.frame_end}, "
                f"{rbw.substeps_per_frame} substeps, {rbw.solver_iterations} iterations"
            )

        configured_caches = 0
        for obj in bpy.data.objects:
            # Cloth modifier cache
            for modifier in obj.modifiers:
                if modifier.type == 'CLOTH' and hasattr(modifier, 'point_cache'):
                    modifier.point_cache.frame_start = start_frame
                    modifier.point_cache.frame_end = end_frame
                    configured_caches += 1
                    
                elif modifier.type == 'SOFT_BODY' and hasattr(modifier, 'point_cache'):
                    modifier.point_cache.frame_start = start_frame
                    modifier.point_cache.frame_end = end_frame
                    configured_caches += 1
                    
                elif modifier.type == 'DYNAMIC_PAINT' and hasattr(modifier, 'canvas_settings'):
                    if modifier.canvas_settings:
                        for surface in modifier.canvas_settings.canvas_surfaces:
                            if hasattr(surface, 'point_cache'):
                                surface.point_cache.frame_start = start_frame
                                surface.point_cache.frame_end = end_frame
                                configured_caches += 1
            
            # Particle system caches
            if hasattr(obj, 'particle_systems'):
                for psys in obj.particle_systems:
                    if hasattr(psys, 'point_cache'):
                        psys.point_cache.frame_start = start_frame
                        psys.point_cache.frame_end = end_frame
                        configured_caches += 1
        
        self.logger.info(
            f"Physics settings configured: frame range {start_frame}-{end_frame}, "
            f"configured {configured_caches} additional point caches"
        )

    def _disable_all_physics_objects(self):
        for obj in bpy.data.objects:
            if not hasattr(obj, "rigid_body") or obj.rigid_body is None:
                continue
            if not hasattr(obj, "mmd_rigid") or obj.mmd_rigid is None:
                continue

            obj.mmd_rigid.type = "0"
            obj.rigid_body.type = "ACTIVE"
            obj.rigid_body.enabled = True
            obj.rigid_body.kinematic = True

    def _set_mmd_timeline_range(self, scene, start_frame, end_frame):
        """
        Set MMD Tools Timeline and Rigid Body Physics frame range.
        
        This is critical for controlling mmd_tools physics baking range.
        The settings appear in MMD Tools panel under:
        - Timeline: Start/End (uses scene.frame_start/end)
        - Rigid Body Physics: Start/End (uses rigidbody_world.point_cache)
        
        Args:
            scene: Blender scene object
            start_frame: Start frame for timeline
            end_frame: End frame for timeline
        """
        # 1. Set scene frame range (this controls MMD Tools "Timeline" Start/End)
        scene.frame_start = start_frame
        scene.frame_end = end_frame
        self.logger.info(f"Set scene/timeline frame range: {start_frame} - {end_frame}")
        
        # 2. Set rigidbody world cache frame range 
        # (this controls MMD Tools "Rigid Body Physics" Start/End)
        if scene.rigidbody_world and scene.rigidbody_world.point_cache:
            cache = scene.rigidbody_world.point_cache
            cache.frame_start = start_frame
            cache.frame_end = end_frame
            self.logger.info(f"Set rigidbody world cache range: {start_frame} - {end_frame}")
        
        # 3. Find MMD Root object and log its available properties (for debugging)
        mmd_root_obj = None
        for obj in bpy.data.objects:
            if hasattr(obj, 'mmd_type') and obj.mmd_type == 'ROOT':
                mmd_root_obj = obj
                break

        
        if mmd_root_obj and hasattr(mmd_root_obj, 'mmd_root'):
            root_settings = mmd_root_obj.mmd_root
            self.logger.info(f"Found MMD root: {mmd_root_obj.name}")
            
            # Try known property names for different mmd_tools versions
            physics_props = [
                'physics_mode',  # Some versions have this
                'show_rigid_bodies',
                'show_joints',
            ]
            
            for prop in physics_props:
                if hasattr(root_settings, prop):
                    self.logger.info(f"  MMD root has: {prop} = {getattr(root_settings, prop)}")
        else:
            self.logger.info("No MMD root object found (normal for non-MMD scenes)")

    def _trim_animation_action(self, obj, start_frame, end_frame):
        """
        Trim the animation action to only include keyframes within the specified range.
        This is crucial for limiting physics baking to a specific range.
        
        Args:
            obj: The object with animation data
            start_frame: First frame to keep
            end_frame: Last frame to keep
        """
        if not obj.animation_data or not obj.animation_data.action:
            return
        
        action = obj.animation_data.action
        original_range = self._get_animation_frame_range(obj)
        self.logger.info(f"Trimming action '{action.name}' from {original_range} to ({start_frame}, {end_frame})")
        
        keyframes_removed = 0
        
        # Iterate through all fcurves (animation channels)
        for fcurve in action.fcurves:
            # Collect keyframe points to remove (iterate in reverse to avoid index issues)
            points_to_remove = []
            for i, keyframe in enumerate(fcurve.keyframe_points):
                frame = keyframe.co[0]
                if frame < start_frame or frame > end_frame:
                    points_to_remove.append(i)
            
            # Remove keyframes outside range (in reverse order)
            for i in reversed(points_to_remove):
                fcurve.keyframe_points.remove(fcurve.keyframe_points[i])
                keyframes_removed += 1
        
        # Update action frame range
        action.frame_range = (start_frame, end_frame)
        
        self.logger.info(f"Removed {keyframes_removed} keyframes outside range")
        self.logger.info(f"New action frame range for {obj.name}: {action.frame_range[0]} - {action.frame_range[1]}")
    
    def _trim_shape_key_animations(self, start_frame, end_frame):
        """Trim shape key (morph) animations for all mesh objects."""
        trimmed_count = 0
        
        for obj in bpy.data.objects:
            if obj.type != 'MESH' or not obj.data.shape_keys:
                continue
            
            shape_keys = obj.data.shape_keys
            if not shape_keys.animation_data or not shape_keys.animation_data.action:
                continue
            
            action = shape_keys.animation_data.action
            
            for fcurve in action.fcurves:
                points_to_remove = []
                for i, kp in enumerate(fcurve.keyframe_points):
                    if kp.co[0] < start_frame or kp.co[0] > end_frame:
                        points_to_remove.append(i)
                
                for i in reversed(points_to_remove):
                    fcurve.keyframe_points.remove(fcurve.keyframe_points[i])
                    trimmed_count += 1
        
        if trimmed_count > 0:
            self.logger.info(f"Trimmed {trimmed_count} shape key keyframes")

    def bake_physics(self, armature):
        """
        Bake physics simulation for a specified frame range.
        
        Args:
            armature: The armature object with animation data
        
        Note: Physics simulation is sequential - Blender simulates from frame 1.
              To truly limit baking, we need to trim the animation action itself.
        """
        if not bpy.ops.ptcache.bake_all.poll():
            self.logger.error("Physics baking failed: no valid physics objects")
            raise RuntimeError("Physics baking failed: no valid physics objects")

        # Free existing bake data
        try:
            bpy.ops.ptcache.free_bake()
            bpy.ops.mmd_tools.ptcache_rigid_body_delete_bake()
            self.logger.info("Freed existing bake data")
        except Exception as e:
            self.logger.warning(f"Could not free existing bake: {e}")
        
        # Get full animation range for reference
        anim_start, anim_end = self._get_animation_frame_range(armature)
        
        # Calculate desired bake range
        frames_to_bake = int(self.fps * self.duration) - 1
        bake_start_frame = anim_start
        start_after_morph = self.skip_animation_morph
        if start_after_morph + frames_to_bake <= anim_end:
            bake_end_frame = start_after_morph + frames_to_bake
        else:
            bake_end_frame = anim_end
        
        self.logger.info(f"Full animation range: {anim_start} - {anim_end}")
        self.logger.info(f"Target bake range: {bake_start_frame} - {bake_end_frame}")
        
        # Trim all objects that have animation to specific simulation range
        if bake_end_frame < anim_end:
            self.logger.info("Trimming animation to limit physics baking...")
            for obj in bpy.data.objects:
                self._trim_animation_action(obj, bake_start_frame, bake_end_frame)
            
            # Also trim all shape key animations
            self._trim_shape_key_animations(bake_start_frame, bake_end_frame)

        bpy.context.view_layer.objects.active = armature
        
        # Configure ALL caches with the desired frame range
        self.configure_physics_settings(bake_start_frame, bake_end_frame)
        
        # NOTE: Force scene frame range IMMEDIATELY before baking, ensuring the physics simulation in specified range
        # don't know why the scene.frame_end will be overwritten by which command......
        bpy.context.scene.frame_end = bake_end_frame
        bpy.context.scene.frame_set(bake_start_frame)
        bpy.context.view_layer.update()
        self.logger.info(f"Forced scene frame range to {bake_start_frame}-{bake_end_frame}")
        
        # Bake physics
        self.logger.info("Starting physics baking...")
        bpy.ops.mmd_tools.ptcache_rigid_body_bake()
        bpy.ops.ptcache.bake_all(bake=True)
        self.logger.info(f"Physics baking completed for frames {bake_start_frame}-{bake_end_frame}")

    def apply_shader_preset(self, preset_index=0):
        if not hasattr(bpy.context.scene, "mbts_mode"):
            self.logger.warning("MBTs_NG addon not loaded, skipping material preset")
            return
        mesh_obj = bpy.context.active_object
        if not mesh_obj or mesh_obj.type != 'MESH':
            self.logger.warning("No active mesh object, cannot apply material preset")
            return
        if bpy.ops.mbts.apply_preset.poll():
            bpy.ops.mmd_tools.convert_materials()
            bpy.context.scene.mbts_mode = "REPLACE_MODEL"
            bpy.ops.mbts.apply_preset(preset_index=0)
            self.logger.info(f"Applied material preset {preset_index}")
        else:
            self.logger.warning("Failed to apply material preset")
    
    def execute(self):
        start_time = time.time()

        try:
            self.setup_blender_scene()
            armature = self._select_character_armature()

            if self.bake:
                # physics simulation
                if self.no_physics:
                    self._disable_all_physics_objects()
                self.logger.info("Initializing physics system...")
                if bpy.ops.mmd_tools.build_rig.poll():
                    bpy.ops.mmd_tools.build_rig()
                self.bake_physics(armature) # bake physics simulation
                
                # ik
                if self.disable_ik: # disable IK controllers if specified
                    ik_bones = ["つま先ＩＫ.R", "つま先ＩＫ.L", "足ＩＫ.R", "足ＩＫ.L"]
                    for bone in ik_bones:
                        if bone in armature.pose.bones:
                            armature.pose.bones[bone].mmd_ik_toggle = False # turn off these japanese-named IK bones
                    self.logger.info("IK controllers disabled")
                else:
                    self.logger.info("Skipped IK disable (DISABLE_IK=False)")
                
                # if mbts_loaded:
                #     self.select_character_mesh()
                #     self.apply_shader_preset(preset_index=0)
                
                if bpy.ops.file.pack_all.poll():
                    try:
                        bpy.ops.file.pack_all() # pack all files into .blend
                        self.logger.info("All packable files successfully packed")
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "source path" in error_msg:
                            missing_files = [path.strip("'") for path in error_msg.split("'")[1::2] if path.endswith(("png", "jpg", "bmp", "mp4", "wav"))]
                            if missing_files:
                                self.logger.info(f"Warning: Missing files skipped during packing: {', '.join(missing_files)}")
                            else:
                                self.logger.info(f"Warning: File missing encountered during packing - {error_msg}")
                        else:
                            raise
                else:
                    self.logger.info("Warning: Cannot execute external file packing (pack_all unavailable)")
                
                # save the composed .blend file
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                bpy.ops.wm.save_as_mainfile(filepath=str(self.output_path), compress=True)
                self.logger.critical(f"✅ Composed .blend file saved to: {self.output_path}")
            else:
                self._quit()

            elapsed_time = format_duration(time.time() - start_time)        
            self.logger.info(f"Blender file finished, check {self.output_path}")
            self.logger.critical(f"Time elapsed: {elapsed_time}. File: {self.output_path}")
        except Exception as e:
            if "Unable to pack file" in str(e):
                pass
            else:
                self.logger.error(f"Processing failed: {str(e)}")
                raise RuntimeError(f"Processing failed: {str(e)}")
        finally:
            self._quit()
            self.logger.info("Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", type=str, required=True)
    parser.add_argument("--motion", type=str, default="zyy") # animation file
    parser.add_argument("--scene", type=str, default="example_scene") # base scene file
    parser.add_argument("--pre_roll_morph", type=int, default=60)
    parser.add_argument("--no_physics", action="store_true")
    parser.add_argument("--disable_ik", action="store_true")
    args = parser.parse_args()

    processor = AnimerBlendComposer(
        tri_path={
            "pmx_path": Path(args.character),
            "vmd_path": Path(args.motion),
            "blend_path": Path(args.scene)
        },
        pre_roll_morph=args.pre_roll_morph,
        no_physics=args.no_physics,
        disable_ik=args.disable_ik
    )
    processor.execute()