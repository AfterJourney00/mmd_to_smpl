import bpy
import logging
import inspect
from pathlib import Path

import numpy as np
import torch

from .utils import config_logging, get_project_root

class AnimerBase:
    def __init__(self):
        self._make_logger()
        logging.getLogger("bpy").setLevel(logging.ERROR)

    def _make_logger(self):
        # get the file of the actual class being instantiated
        source_file = inspect.getfile(self.__class__)
        
        current_file = Path(source_file).resolve()
        project_root = get_project_root()
        rel_path = current_file.relative_to(project_root).with_suffix("")
        log_path = project_root / "logs" / rel_path
        log_path = log_path.with_stem(f"{log_path.stem}.log")
        config_logging(log_path)
        self.logger = logging.getLogger(self.__class__.__module__)

    def _load_initial_scene(self, blend_path):
        if blend_path.exists():
            bpy.ops.wm.open_mainfile(filepath=str(blend_path))
            self.logger.info(f"Loaded initial scene: {blend_path}")
        else:
            self.logger.info(f"Initial scene not found, using empty scene: {blend_path}")
            bpy.ops.wm.read_factory_settings(use_empty=True)

    def _cleanup_existing_objects(self):
        """
        Remove and delete existing cameras, armatures, and objects linked to external
        .abc (Alembic) files from the loaded scene.
        This prevents conflicts when importing new MMD models and setting up cameras.
        """
        cameras_removed = 0
        armatures_removed = 0
        mesh_removed = 0
        alembic_objects_removed = 0

        def _recurse_and_delete(col):
            for child in list(col.children):
                _recurse_and_delete(child)
            
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            
            bpy.data.collections.remove(col)
        
        # Delete exisitng but undesirable collections
        col_to_delete = []
        if self.scene_name == "B9XNCaRBQ1O8":
            col_to_delete.append(bpy.data.collections.get("角色"))
        elif self.scene_name == "kzLvGBFHvQu6":
            col_to_delete.append(bpy.data.collections.get("小陈"))
            col_to_delete.append(bpy.data.collections.get("GUN"))
        elif self.scene_name == "o5tTmlncJ8YE":
            col_to_delete.append(bpy.data.collections.get("Collection 5"))
        elif self.scene_name == "SgMbXVSclrLF":
            col_to_delete.append(bpy.data.collections.get("Collection 4"))
        for col in col_to_delete:
            _recurse_and_delete(col)

        # Collect objects to remove (can't modify while iterating)
        objects_to_remove = []
        for obj in bpy.data.objects:
            should_remove = False
            
            # Check for camera
            if obj.type == 'CAMERA':
                should_remove = True
                cameras_removed += 1
            
            # Check for armature
            elif obj.type == 'ARMATURE':
                should_remove = True
                armatures_removed += 1

            # Check monkey head
            elif obj.type == 'MESH':
                if obj.name == "猴头":
                    should_remove = True
                    mesh_removed += 1
            
            # Check for objects linked to external .abc files
            elif self._has_alembic_dependency(obj):
                should_remove = True
                alembic_objects_removed += 1
            
            if should_remove:
                objects_to_remove.append(obj)
        
        # Remove objects from all collections and delete them
        for obj in objects_to_remove:
            # Unlink from all collections
            for collection in obj.users_collection:
                collection.objects.unlink(obj)
            
            # Delete the object data block
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Clean up orphan data blocks
        self._cleanup_orphan_data_blocks()
        
        # Clean up Alembic cache files from bpy.data
        self._cleanup_alembic_caches()
        
        if cameras_removed > 0 or armatures_removed > 0 or alembic_objects_removed > 0:
            self.logger.info(
                f"Cleaned up existing objects: {cameras_removed} cameras, "
                f"{armatures_removed} armatures, {alembic_objects_removed} alembic-linked objects removed"
            )

    def _has_alembic_dependency(self, obj) -> bool:
        """
        Check if an object has dependencies on external .abc (Alembic) files.
        
        This includes:
        - Mesh Cache modifier with .abc file
        - Alembic cache constraints
        - Objects imported from Alembic files
        """
        # Check modifiers for Mesh Cache pointing to .abc
        for modifier in obj.modifiers:
            if modifier.type == 'MESH_CACHE':
                if hasattr(modifier, 'filepath') and modifier.filepath:
                    if modifier.filepath.lower().endswith('.abc'):
                        return True
            
            # Check for Mesh Sequence Cache modifier (used for Alembic)
            if modifier.type == 'MESH_SEQUENCE_CACHE':
                if hasattr(modifier, 'cache_file') and modifier.cache_file:
                    return True
        
        # Check constraints for Transform Cache (Alembic)
        for constraint in obj.constraints:
            if constraint.type == 'TRANSFORM_CACHE':
                if hasattr(constraint, 'cache_file') and constraint.cache_file:
                    return True
        
        # Check if object's mesh data is from Alembic cache
        if obj.type == 'MESH' and obj.data:
            # Check if mesh has cache file reference
            if hasattr(obj.data, 'shape_keys') and obj.data.shape_keys:
                if hasattr(obj.data.shape_keys, 'animation_data'):
                    anim_data = obj.data.shape_keys.animation_data
                    if anim_data and anim_data.action:
                        # Check action name for alembic references
                        if 'alembic' in anim_data.action.name.lower():
                            return True
        
        return False
    
    def _cleanup_orphan_data_blocks(self):
        """Clean up orphan data blocks (data that lost their objects)."""
        # Remove orphan camera data
        for camera_data in list(bpy.data.cameras):
            if camera_data.users == 0:
                bpy.data.cameras.remove(camera_data)
        
        # Remove orphan armature data
        for armature_data in list(bpy.data.armatures):
            if armature_data.users == 0:
                bpy.data.armatures.remove(armature_data)
        
        # Remove orphan mesh data
        for mesh_data in list(bpy.data.meshes):
            if mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
    
    def _cleanup_alembic_caches(self):
        """Remove Alembic cache files from bpy.data.cache_files."""
        if not hasattr(bpy.data, 'cache_files'):
            return
        
        cache_files_to_remove = []
        
        # Collect cache files to remove
        for cache_file in bpy.data.cache_files:
            if hasattr(cache_file, 'filepath'):
                # Check if it's an Alembic cache or has no users
                if cache_file.filepath.lower().endswith('.abc') or cache_file.users == 0:
                    cache_files_to_remove.append(cache_file)
        
        if not cache_files_to_remove:
            return
        
        # Try different removal methods
        cache_files_removed = 0
        
        # Method 1: Try batch_remove if available (Blender 2.93+)
        if hasattr(bpy.data, 'batch_remove'):
            try:
                bpy.data.batch_remove(cache_files_to_remove)
                cache_files_removed = len(cache_files_to_remove)
            except Exception as e:
                self.logger.warning(f"batch_remove failed: {e}")
        
        # Method 2: If batch_remove didn't work, try orphan purge
        if cache_files_removed == 0:
            try:
                # Use orphans_purge to clean up unused data blocks
                bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
                cache_files_removed = len(cache_files_to_remove)  # Approximate
                self.logger.info("Used orphans_purge to clean up cache files")
            except Exception as e:
                self.logger.warning(f"orphans_purge failed: {e}")
        
        if cache_files_removed > 0:
            self.logger.info(f"Cleaned up {cache_files_removed} Alembic cache files")

    def _ensure_addon_loaded(self, addon_module, required=True):
        try:
            if not bpy.context.preferences.addons.get(addon_module):
                bpy.ops.preferences.addon_enable(module=addon_module)
                self.logger.info(f"Enabled addon: {addon_module}")
            
            # disable preference save
            bpy.context.preferences.filepaths.temporary_directory = "tmp/blender_pref"
            bpy.context.preferences.use_preferences_save = False
            
            # save preference, (dont need this actually)
            # bpy.context.preferences.is_dirty = True
            # bpy.ops.wm.save_userpref()
            return True
        except Exception as e:
            if required:
                self.logger.error(f"Failed to load required addon {addon_module}: {str(e)}")
                raise RuntimeError(f"Failed to load required addon {addon_module}: {str(e)}")
            else:
                self.logger.warning(f"Failed to load **unrequired** optional addon {addon_module}: {str(e)}")
                return False

    def _import_mmd_model(self, scale=0.08):
        if not self.pmx_path.exists():
            self.logger.error(f"Model file not found: {self.pmx_path}")
            raise FileNotFoundError(f"Model file not found: {self.pmx_path}")
        if not bpy.ops.mmd_tools.import_model.poll():
            self.logger.error(f"mmd_tools.import_model not available")
            raise RuntimeError("mmd_tools.import_model not available")
        
        # Get existing objects before import
        objects_before = set(bpy.data.objects)
        bpy.ops.mmd_tools.import_model(
            filepath=str(self.pmx_path),
            files=[{"name": f"{self.pmx_path}"}],
            directory=str(self.pmx_path.parent),
            log_level="INFO",
            scale=scale
        )
        objects_after = set(bpy.data.objects)
        imported_objects = objects_after - objects_before
        self.logger.info(f"Model imported (scaled {scale}): {self.pmx_path}")
        
        # Move imported MMD character model to the root scene collection
        scene_collection = bpy.context.scene.collection
        for obj in imported_objects:
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            if obj.name not in scene_collection.objects:
                scene_collection.objects.link(obj)

    def _import_vmd_animation(self):
        if not self.vmd_path.exists():
            self.logger.error(f"VMD file not found: {self.vmd_path}")
            raise FileNotFoundError(f"VMD file not found: {self.vmd_path}")
        if not bpy.ops.mmd_tools.import_vmd.poll():
            self.logger.error(f"mmd_tools.import_vmd not available")
            raise RuntimeError("mmd_tools.import_vmd not available")
        bpy.ops.mmd_tools.import_vmd(
            filepath=str(self.vmd_path),
            files=[{"name": self.vmd_path.name}],
            directory=str(self.vmd_path.parent)
        )
        self.logger.info(f"Animation imported: {self.vmd_path}")

    def _inject_preroll_transition(self, armature):
        """
        Inserts a 'Rest Pose' at frame -60 and interpolates it to Frame 0
        to allow physics to settle gently.
        """
        self.logger.info(f"Injecting {self.pre_roll_morph}-frame pre-roll for: {armature.name}")
        
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")
        
        # Ensure there is an animation data block
        if not armature.animation_data or not armature.animation_data.action:
            self.logger.error(f"No animation found on armature {armature.name}. Import VMD first.")
            raise RuntimeError(f"No animation found on armature {armature.name}. Import VMD first.")

        # 2. Move the first keyframe of all animation curves to -60 frames
        action = armature.animation_data.action
        target_frame = -self.pre_roll_morph  # -60
        processed_fcurves = 0
        
        for fcurve in action.fcurves:
            if len(fcurve.keyframe_points) == 0:
                continue
            
            # Find the first keyframe (earliest frame)
            first_keyframe = min(fcurve.keyframe_points, key=lambda kf: kf.co.x)
            original_frame = first_keyframe.co.x
            
            # Move it to target frame (-60)
            first_keyframe.co.x = target_frame
            first_keyframe.interpolation = "LINEAR"
            
            processed_fcurves += 1
        self.logger.info(f"Processed {processed_fcurves} fcurves: moved first keyframes to frame {target_frame}")
        
        # 3. Move all keyframes forward to ensure first keyframe lies at frame 1
        # Find the earliest keyframe across all fcurves
        for fcurve in action.fcurves:
            if len(fcurve.keyframe_points) > 0:
                min_frame = min(kf.co.x for kf in fcurve.keyframe_points)
                if min_frame != -self.pre_roll_morph:
                    self.logger.error("After adding preroll, the first keyframe should be at -60frame")
                    raise AssertionError("After adding preroll, the first keyframe should be at -60frame")
        
        # Calculate offset to move earliest keyframe to frame 1
        offset = 1 + self.pre_roll_morph 
        self.logger.info(f"Moving all keyframes forward by {offset} frames")
        
        # Move all keyframes forward
        for fcurve in action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.co.x += offset
        
        # Recalculate action frame range based on actual keyframe positions
        min_frame = float("inf")
        max_frame = float("-inf")
        for fcurve in action.fcurves:
            if len(fcurve.keyframe_points) > 0:
                for kf in fcurve.keyframe_points:
                    min_frame = min(min_frame, kf.co.x)
                    max_frame = max(max_frame, kf.co.x)
                    if min_frame != 1:
                        self.logger.error(f"After moving all keyframes, the first keyframe should be at 1st frame, got {min_frame} frame")
                        raise AssertionError(f"After moving all keyframes, the first keyframe should be at 1st frame, got {min_frame} frame")
        if max_frame == float("-inf"):
            self.logger.error("The action has infinite timeline")
            raise AssertionError("The action has infinite timeline")
        action.frame_range = (1, max_frame)
        self.logger.info(f"Updated action frame range to: {action.frame_range[0]} - {action.frame_range[1]}")
        
        # Force scene update and evaluate at target frame
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        self.logger.info(f"Set scene to frame 1st frame and forced update")

        # switch back to object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        
        self.logger.info(f"Successfully set physics simulation warm-up with {self.pre_roll_morph}-frame pre-roll transitions.")

    def _inject_preroll_transition_v2(self, armature):
        """
        Inserts a 'Rest Pose' at frame -60 and interpolates it to Frame 0
        to allow physics to settle gently.
        """
        self.logger.info(f"Injecting {self.pre_roll_morph}-frame pre-roll for: {armature.name}")
        
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")
        
        # Ensure there is an animation data block
        if not armature.animation_data or not armature.animation_data.action:
            self.logger.error(f"No animation found on armature {armature.name}. Import VMD first.")
            raise RuntimeError(f"No animation found on armature {armature.name}. Import VMD first.")

        # Move all non-1st-keyframes forward
        action = armature.animation_data.action
        for fcurve in action.fcurves:
            if len(fcurve.keyframe_points) < 2:
                continue
            ordered_keyframes = sorted(fcurve.keyframe_points, key=lambda kf: kf.co.x)
            for kf in ordered_keyframes[1:]:
                    kf.co.x += self.pre_roll_morph
        
        # Iterate on all keyframes, set interpolation attribute, and recalculate action frame range based on actual keyframe positions
        min_frame = float("inf")
        max_frame = float("-inf")
        for fcurve in action.fcurves:
            if len(fcurve.keyframe_points) > 0:
                for kf in fcurve.keyframe_points:
                    kf.interpolation = "BEZIER"
                    kf.easing = "AUTO"
                    kf.handle_left_type = "AUTO_CLAMPED"
                    kf.handle_right_type = "AUTO_CLAMPED"

                    min_frame = min(min_frame, kf.co.x)
                    max_frame = max(max_frame, kf.co.x)
                    if min_frame != 1:
                        self.logger.error(f"After moving all keyframes, the first keyframe should be at 1st frame, got {min_frame} frame")
                        raise AssertionError(f"After moving all keyframes, the first keyframe should be at 1st frame, got {min_frame} frame")
            fcurve.update()
        if max_frame == float("-inf"):
            self.logger.error("The action has infinite timeline")
            raise AssertionError("The action has infinite timeline")
        action.frame_range = (1, max_frame)
        self.logger.info(f"Updated action frame range to: {action.frame_range[0]} - {action.frame_range[1]}")
        
        # Force scene update and evaluate at target frame
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        self.logger.info(f"Set scene to frame 1st frame and forced update")

        # switch back to object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        
        self.logger.info(f"Successfully set physics simulation warm-up with {self.pre_roll_morph}-frame pre-roll transitions.")

    def _select_character_armature(self):
        armature_name = None
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature_name = obj.name
                break
        armature = bpy.data.objects.get(armature_name)
        if not armature:
            self.logger.error(f"Armature object not found: {armature_name}")
            raise RuntimeError(f"Armature object not found: {armature_name}")
        
        bpy.ops.object.select_all(action='DESELECT')
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        self.logger.info(f"Selected armature: {armature_name}")
        return armature

    def _get_animation_frame_range(self, armature):
        if not armature.animation_data or not armature.animation_data.action:
            self.logger.error(f"No animation data found on armature")
            raise RuntimeError("No animation data found on armature")
        action = armature.animation_data.action
        start_frame = int(action.frame_range[0])
        end_frame = int(action.frame_range[1])
        self.logger.info(f"Animation frame range: {start_frame} - {end_frame}")
        return start_frame, end_frame

    def _to_device(self, dict_data, torch_device):
        if hasattr(torch_device, "device"):
            device = torch_device.device
        else:
            device = torch_device

        data_device = {}
        for key, val in dict_data.items():
            if torch.is_tensor(val):
                data_device[key] = val.to(device, non_blocking=True)
            elif isinstance(val, np.ndarray):
                data_device[key] = torch.from_numpy(val).float().to(device, non_blocking=True)
            elif isinstance(val, dict):
                data_device[key] = self.to_device(val, torch_device)
            else:
                data_device[key] = val
        return data_device

    def _quit(self):
        bpy.ops.wm.quit_blender()