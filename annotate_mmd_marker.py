import bpy
import mathutils
import torch
import numpy as np

import json
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from bodymodels import SMPLMesh
from .base import AnimerBase
from .config.joint_correspondence import (
    MMD_BONE_MAPPING,
    SMPL_BONE_JOINTS,
    SMPL_FALLBACK_BONE_DIRECTION,
    MARKER_SYMMETRIC_PAIRS,
)
from .utils import standardize_file_name, get_project_root

class AnimerMarkerAnnotator(AnimerBase):
    def __init__(
        self,
        input_path: str,
        validate: bool = False,
        visualize: bool = False,
        marker_size: float = 0.01,
        version: str = "debug"
    ):
        super().__init__()

        self.base_path = Path(__file__).resolve().parent
        self.pmx_path = Path(input_path)
        self.do_validate = validate
        self.do_visualize = visualize
        self.marker_size = marker_size

        # Initialize SMPL-H body model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.body_model = SMPLMesh().to(self.device)
        with open("config/smpl_marker_index.json", "r") as f:
            self.smpl_marker_index = json.load(f)  # {marker_name: vertex_index}
        self.expected_marker_name = list(self.smpl_marker_index.keys())
        self.expected_marker_coords = self._compute_marker_coords()
        self.logger.info(f"Computed {len(self.expected_marker_coords)} marker definitions from SMPL model")
        
        pmx_name = standardize_file_name(self.pmx_path.stem)
        self.output_path = self.base_path / "outputs" / version / "auto_marker_annotate" / pmx_name / (pmx_name+".json")

    def _to_device(self, dict_data, torch_device):
        data_device = {}
        for key, val in dict_data.items():
            if torch.is_tensor(val):
                if hasattr(torch_device, "device"):
                    data_device[key] = val.to(torch_device.device, non_blocking=True)
                else:
                    data_device[key] = val.to(torch_device, non_blocking=True)
            elif isinstance(val, dict):
                data_device[key] = self.to_device(val, torch_device)
            else:
                data_device[key] = val
        return data_device

    @torch.no_grad()
    def _compute_marker_coords(self, apose_arm_angle: float=np.pi / 6) -> Dict[str, Tuple[str, float, float]]:
        """
        Run SMPL forward pass with zero pose/shape, read marker vertex positions
        via smpl_index, then reverse-engineer (bone_name, ratio, angle) for each marker.

        Returns:
            Dict mapping marker_name -> (bone_name, ratio_along_bone, angle_degrees)
        """
        rot6d = torch.tensor([[[1., 0., 0., 1., 0., 0.]]]).repeat(1, 52, 1).float()
        c, s = math.cos(apose_arm_angle), math.sin(apose_arm_angle)
        rot6d[:, 13] = torch.tensor([c, s, -s, c, 0., 0.]).float()
        rot6d[:, 14] = torch.tensor([c, -s, s, c, 0., 0.]).float()
        apose_param = {
            "rot6d": rot6d,
            "trans": torch.zeros(1, 3),
            "shapes": torch.zeros(1, 16),
        }
        apose_param = self._to_device(apose_param, self.device)
        output = self.body_model(apose_param)
        vertices = output["vertices"][0].cpu().numpy()
        joints = output["joints"][0, :22].cpu().numpy()
        self.smpl_bones = {}
        for bone_name, bone_info in SMPL_BONE_JOINTS.items():
            self.smpl_bones[bone_name] = {
                "bone_head": bone_info[0],
                "bone_tail": bone_info[1],
                "bone_vec": joints[bone_info[1]] - joints[bone_info[0]]
            }
            bone_len = np.linalg.norm(self.smpl_bones[bone_name]["bone_vec"])
            self.smpl_bones[bone_name].update({
                "bone_dir": self.smpl_bones[bone_name]["bone_vec"] / bone_len,
                "bone_len": bone_len
            })
        self.logger.info(f"SMPL T-pose: {vertices.shape[0]} vertices, {joints.shape[0]} joints")

        marker_pos = {}
        marker_coords = {}
        for marker_name, vertex_idx in self.smpl_marker_index.items():
            marker_pos = vertices[vertex_idx]
            bone_name, ratio, angle = self._reverse_bone_params(marker_pos, joints, marker_name)
            marker_coords[marker_name] = (bone_name, ratio, angle)
            self.logger.debug(f"  {marker_name}: bone={bone_name}, ratio={ratio:.3f}, angle={angle:.1f}")

        return marker_coords

    def _reverse_bone_params(self, marker_pos: np.ndarray, joints: np.ndarray, marker_name: str) -> Tuple[str, float, float]:
        """
        Given a 3D marker position and SMPL joint positions, find the best-matching
        bone and compute (bone_name, ratio, angle).

        For each candidate bone:
          - Project marker onto bone segment, clamp ratio to [0, 1]
          - Compute Euclidean distance from marker to the clamped projection point
          - Score = total distance (lower is better)
        Then re-compute unclamped ratio and angle for the best bone.
        """
        best_bone = None
        best_ratio = 0.0
        best_angle = 0.0
        best_dist = float('inf')
        for bone_name, (head_idx, tail_idx) in SMPL_BONE_JOINTS.items():
            bone_head = joints[head_idx]
            bone_tail = joints[tail_idx]

            # Handle single-joint bones
            if head_idx == tail_idx:
                if bone_name in SMPL_FALLBACK_BONE_DIRECTION:
                    fb_head, fb_tail = SMPL_FALLBACK_BONE_DIRECTION[bone_name]
                    bone_vec = joints[fb_tail] - joints[fb_head]
                else:
                    continue
                bone_origin = bone_head
            else:
                bone_vec = bone_tail - bone_head
                bone_origin = bone_head

            # Calculate direction and length of bone
            bone_len = np.linalg.norm(bone_vec)
            if bone_len < 1e-6:
                continue
            bone_dir = bone_vec / bone_len

            # Project marker onto bone axis
            offset = marker_pos - bone_origin
            proj_len = np.dot(offset, bone_dir)
            ratio = proj_len / bone_len

            # Clamp to bone segment for distance scoring
            clamped_ratio = float(np.clip(ratio, 0.0, 1.0))
            closest_point = bone_origin + bone_dir * (clamped_ratio * bone_len)
            dist = np.linalg.norm(marker_pos - closest_point)
            if dist < best_dist:
                best_dist = dist
                best_bone = bone_name
                # Store unclamped ratio for the actual parameterization
                best_ratio = ratio
                perp_vec = marker_pos - (bone_origin + bone_dir * proj_len)
                best_angle = self._compute_angle_around_bone(bone_dir, perp_vec)

        return best_bone, round(best_ratio, 4), round(best_angle, 1)

    @staticmethod
    def _compute_angle_around_bone(bone_dir: np.ndarray, perp_vec: np.ndarray) -> float:
        """
        Compute the angle (in degrees) of perp_vec around the bone axis.
        Convention: 0 = forward (+Z), 90 = left (+X), -90 = right (-X), 180 = back (-Z).

        Uses the same local coordinate frame as _calculate_ray_origin_and_direction:
          vec_y = bone_dir (along bone)
          world_forward = (0, 0, 1) unless near-parallel to bone
          vec_x = vec_y x world_forward  (right)
          vec_z = vec_x x vec_y          (forward)
        """
        perp_len = np.linalg.norm(perp_vec)
        if perp_len < 1e-8:
            return 0.0

        vec_y = bone_dir / np.linalg.norm(bone_dir)

        # Build the same local frame as the ray casting code
        world_forward = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(vec_y, world_forward)) > 0.95:
            raise RuntimeError(f"The given smpl bone has bone direction nearly parallel to world forward [0, 0, 1]")

        vec_x = np.cross(vec_y, world_forward)
        vec_x = vec_x / np.linalg.norm(vec_x)
        vec_z = np.cross(vec_x, vec_y)
        vec_z = vec_z / np.linalg.norm(vec_z)

        # Project perp_vec onto the local x-z plane
        perp_unit = perp_vec / perp_len
        comp_z = np.dot(perp_unit, vec_z)  # forward component
        comp_x = np.dot(perp_unit, vec_x)  # right component

        angle_rad = math.atan2(comp_x, comp_z)
        return math.degrees(angle_rad)

    def _calculate_ray_origin_and_direction(
        self,
        armature,
        smpl_bone_name: str,
        smpl_ratio: float,
        angle_deg: float
    ) -> Tuple[Optional[mathutils.Vector], Optional[mathutils.Vector]]:
        _mmd_bone_head_name = MMD_BONE_MAPPING[smpl_bone_name][0].split(".")
        _mmd_bone_tail_name = MMD_BONE_MAPPING[smpl_bone_name][1].split(".")
        mmd_bone_head_name = ".".join(_mmd_bone_head_name[:-1])
        mmd_bone_tail_name = ".".join(_mmd_bone_tail_name[:-1])
        mmd_bone_head = getattr(armature.pose.bones.get(mmd_bone_head_name), _mmd_bone_head_name[-1])
        mmd_bone_tail = getattr(armature.pose.bones.get(mmd_bone_tail_name), _mmd_bone_tail_name[-1])
        mmd_bone_len = (mmd_bone_tail - mmd_bone_head).length
        if mmd_bone_len < 1e-6:
            self.logger.error(f"The mmd bone {_mmd_bone_head_name} -> {_mmd_bone_tail_name} has invalid bone length")
            raise RuntimeError(f"The mmd bone {_mmd_bone_head_name} -> {_mmd_bone_tail_name} has invalid bone length")

        # refer to smpl bone vec direction
        smpl_bone_dir = self.smpl_bones[smpl_bone_name]["bone_dir"]
        mat = armature.matrix_world
        tail_ws = mat @ mmd_bone_tail
        head_ws = tail_ws - mathutils.Vector(smpl_bone_dir * mmd_bone_len)
        mmd_bone_vec = tail_ws - head_ws
        origin = head_ws + mmd_bone_vec * smpl_ratio
        vec_y = mmd_bone_vec.normalized()
        world_forward = mathutils.Vector((0, 0, 1))
        if abs(vec_y.dot(world_forward)) > 0.95:
            self.logger.error(f"The mmd bone {_mmd_bone_head_name} -> {_mmd_bone_tail_name} has bone direction nearly parallel to world forward [0, 0, 1]")
            raise RuntimeError(f"The mmd bone {_mmd_bone_head_name} -> {_mmd_bone_tail_name} has bone direction nearly parallel to world forward [0, 0, 1]")

        vec_x = vec_y.cross(world_forward).normalized()
        vec_z = vec_x.cross(vec_y).normalized()

        rot_mat = mathutils.Matrix.Rotation(math.radians(angle_deg), 4, vec_y)
        direction = (rot_mat @ vec_z).normalized()

        return origin, direction

    def _ray_cast_to_meshes(self, mesh_objects, origin, direction):
        """Ray cast against multiple meshes, return the closest hit with vertex info.

        Returns
        -------
        dict or None
            {
                "hit_position": [x, y, z],       # world-space hit point on triangle
                "mesh_name": str,                 # name of the mesh object hit
                "vertex_id": int,                 # index of the nearest vertex in that mesh
                "vertex_position": [x, y, z],     # world-space position of that vertex
            }
        """
        best_result = None
        best_dist_sq = float('inf')

        for mesh_obj in mesh_objects:
            mat_inv = mesh_obj.matrix_world.inverted()
            local_origin = mat_inv @ origin
            local_direction = (mat_inv.to_3x3() @ direction).normalized()

            success, location, _, face_index = mesh_obj.ray_cast(local_origin, local_direction)
            if success:
                hit_ws = mesh_obj.matrix_world @ location
                dist_sq = (hit_ws - origin).length_squared
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    
                    # Find nearest vertex to the hit point (in local space)
                    mesh_data = mesh_obj.data
                    poly = mesh_data.polygons[face_index]
                    nearest_vid = None
                    nearest_dist_sq = float('inf')
                    for vid in poly.vertices:
                        v_local = mesh_data.vertices[vid].co
                        d_sq = (v_local - location).length_squared
                        if d_sq < nearest_dist_sq:
                            nearest_dist_sq = d_sq
                            nearest_vid = vid
                    v_ws = mesh_obj.matrix_world @ mesh_data.vertices[nearest_vid].co
                    best_result = {
                        "hit_position": [round(v, 6) for v in hit_ws],
                        "mesh_name": mesh_obj.name,
                        "vertex_id": nearest_vid,
                        "vertex_position": [round(v, 6) for v in v_ws],
                    }

        return best_result

    def _annotate_single_model(self) -> Dict[str, List[float]]:
        self._load_initial_scene(Path("a_virtual_path_that_not_exists"))
        self._ensure_addon_loaded("mmd_tools")
        self._import_mmd_model()

        armature = self._select_character_armature()
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not armature or not mesh_objects:
            self.logger.error(f"Invalid model structure in {self.pmx_path.name}")
            raise RuntimeError(f"Invalid model structure in {self.pmx_path.name}")
        self.logger.info(f"Found {len(mesh_objects)} meshes of {self.pmx_path.name} for ray casting")

        # Rotate the entire MMD model -90° around X to align forward direction:
        # Blender/MMD rest pose forward = [0,-1,0] → SMPL rest pose forward = [0,0,1]
        rot_x = mathutils.Matrix.Rotation(math.radians(-90), 4, 'X')
        armature.matrix_world = rot_x @ armature.matrix_world
        bpy.context.view_layer.update()
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="OBJECT")

        markers = {}
        for marker_name, (smpl_bone_name, ratio, angle) in self.expected_marker_coords.items():
            origin, direction = self._calculate_ray_origin_and_direction(armature, smpl_bone_name, ratio, angle)
            if origin is not None:
                result = self._ray_cast_to_meshes(mesh_objects, origin, direction)
                if result:
                    markers[marker_name] = result

        # Rotate marker positions 90° around X back
        rot_x = mathutils.Matrix.Rotation(math.radians(90), 4, 'X')
        for marker_name in markers.keys():
            hit_pos = rot_x @ mathutils.Vector(markers[marker_name]["hit_position"])
            vert_pos = rot_x @ mathutils.Vector(markers[marker_name]["vertex_position"])
            markers[marker_name]["hit_position"] = [round(v, 6) for v in hit_pos]
            markers[marker_name]["vertex_position"] = [round(v, 6) for v in vert_pos]

        # Rotate the entire MMD model 90° around X back to original orientation:
        # SMPL rest pose forward = [0,0,1] -> Blender/MMD rest pose forward = [0,-1,0]
        armature.matrix_world = rot_x @ armature.matrix_world
        bpy.context.view_layer.update()
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="OBJECT")
        
        return markers

    def _create_marker_sphere(self, name: str, location: tuple, radius: float):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius,
            location=location,
            segments=16,
            ring_count=8,
        )
        sphere = bpy.context.active_object
        sphere.name = f"Marker_{name}"

        mat = bpy.data.materials.new(name=f"Mat_{name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get('Principled BSDF')
        if bsdf:
            if 'chest' in name or 'waist' in name or 'back' in name:
                bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)
            elif 'head' in name:
                bsdf.inputs['Base Color'].default_value = (0, 1, 0, 1)
            elif 'arm' in name or 'elbow' in name or 'wrist' in name:
                bsdf.inputs['Base Color'].default_value = (0, 0, 1, 1)
            elif 'leg' in name or 'knee' in name or 'ankle' in name or 'foot' in name:
                bsdf.inputs['Base Color'].default_value = (1, 1, 0, 1)
            else:
                bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
            bsdf.inputs['Emission'].default_value = (0.5, 0.5, 0.5, 1)
            bsdf.inputs['Emission Strength'].default_value = 0.5

        sphere.data.materials.append(mat)
        return sphere

    def _visualize_markers(self, markers):
        for name, coords in markers.items():
            self._create_marker_sphere(name, coords["vertex_position"], self.marker_size)

        # save current blender scene for debugging
        bpy.ops.wm.save_as_mainfile(filepath=str(self.output_path.parent / "visualize_marker.blend"), compress=True)

        # Compute bounding box of all marker positions for camera framing
        marker_positions = np.array([m["vertex_position"] for m in markers.values()])
        center = marker_positions.mean(axis=0)
        extent = marker_positions.max(axis=0) - marker_positions.min(axis=0)
        distance = float(max(extent)) * 2.5

        # Render four views (front, left, back, right)
        scene = bpy.context.scene
        scene.cycles.samples = 4
        render = scene.render
        render.engine = "CYCLES"
        render.resolution_x = 960
        render.resolution_y = 1080
        render.resolution_percentage = 100
        render.image_settings.file_format = "PNG"

        # Add a sun light
        light_data = bpy.data.lights.new("Sun", type="SUN")
        light_data.energy = 3.0
        light_obj = bpy.data.objects.new("Sun", light_data)
        scene.collection.objects.link(light_obj)
        light_obj.rotation_euler = (math.radians(45), 0, math.radians(30))

        # Camera
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
        cam_data.lens = 85

        # Render four views
        views = [
            ("front", 0),
            ("left", 90),
            ("back", 180),
            ("right", 270),
        ]
        for view_name, azimuth_deg in views:
            azimuth = math.radians(azimuth_deg)
            cam_obj.location = (
                center[0] + distance * math.sin(azimuth),
                center[1] - distance * math.cos(azimuth),
                center[2] + float(extent[2]) * 0.3,
            )
            direction = mathutils.Vector(center.tolist()) - cam_obj.location
            cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

            render.filepath = str(self.output_path.parent / f"marker_viz_{view_name}.png")
            bpy.ops.render.render(write_still=True)
            self.logger.info(f"Saved {view_name} view to {render.filepath}")

    def _validate_marker_symmetry(self, markers: Dict[str, List[float]], tolerance: float = 0.05) -> Dict[str, dict]:
        issues = {}

        for left_name, right_name in MARKER_SYMMETRIC_PAIRS:
            if left_name not in markers or right_name not in markers:
                issues[f"{left_name}/{right_name}"] = {
                    'status': 'missing',
                    'message': 'One or both markers missing',
                }
                continue

            left_pos = np.array(markers[left_name]["vertex_position"])
            right_pos = np.array(markers[right_name]["vertex_position"])

            y_diff = abs(left_pos[1] - right_pos[1])
            z_diff = abs(left_pos[2] - right_pos[2])
            x_mirror_diff = abs(left_pos[0] + right_pos[0])

            max_coord = max(abs(left_pos).max(), abs(right_pos).max())
            relative_y_diff = y_diff / max_coord if max_coord > 0 else 0
            relative_z_diff = z_diff / max_coord if max_coord > 0 else 0
            relative_x_diff = x_mirror_diff / abs(left_pos[0]) if abs(left_pos[0]) > 1e-6 else 0

            if relative_y_diff > tolerance or relative_z_diff > tolerance or relative_x_diff > tolerance:
                issues[f"{left_name}/{right_name}"] = {
                    'status': 'asymmetric',
                    'y_diff': float(y_diff),
                    'z_diff': float(z_diff),
                    'x_mirror_diff': float(x_mirror_diff),
                    'relative_error': float(max(relative_y_diff, relative_z_diff, relative_x_diff)),
                }

        return issues

    def _validate_marker_coverage(self, markers: Dict[str, List[float]]) -> Dict:
        present = set(markers.keys())
        expected = set(self.expected_marker_name)

        missing = expected - present
        extra = present - expected

        return {
            'coverage': len(present) / len(expected) if expected else 0.0,
            'present_count': len(present),
            'expected_count': len(expected),
            'missing': sorted(list(missing)),
            'extra': sorted(list(extra)),
        }

    def _compare_mmd_to_smpl(
        self,
        mmd_markers: Dict[str, List[float]],
        smpl_markers: Dict[str, List[float]],
        normalize: bool = True,
    ) -> Dict[str, float]:
        distances = {}

        if normalize:
            def get_scale(markers):
                if 'top_head' in markers and 'middle_waist' in markers:
                    return np.linalg.norm(
                        np.array(markers['top_head']) - np.array(markers['middle_waist'])
                    )
                return 1.0

            mmd_scale = get_scale(mmd_markers)
            smpl_scale = get_scale(smpl_markers)
            scale_ratio = smpl_scale / mmd_scale if mmd_scale > 0 else 1.0
        else:
            scale_ratio = 1.0

        common_markers = set(mmd_markers.keys()) & set(self.expected_marker_name)

        for marker_name in common_markers:
            mmd_pos = np.array(mmd_markers[marker_name]["vertex_position"]) * scale_ratio
            smpl_pos = np.array(self.expected_marker_coords[marker_name])
            distances[marker_name] = float(np.linalg.norm(mmd_pos - smpl_pos))

        return distances

    def _generate_validation_report(self, markers_json: Path, output_txt: Path):
        with open(self.output_path, "r") as f:
            markers = json.load(f)

        with open(output_txt, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MMD MARKER VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model: {self.output_path.stem}\n")
            f.write(f"Date: {self.output_path.stat().st_mtime}\n\n")

            # Coverage check
            f.write("\n" + "-" * 70 + "\n")
            f.write("MARKER COVERAGE\n")
            f.write("-" * 70 + "\n")

            coverage = self._validate_marker_coverage(markers)
            f.write(f"Coverage: {coverage['coverage']:.1%} ({coverage['present_count']}/{coverage['expected_count']})\n")

            if coverage["missing"]:
                f.write(f"\nMissing markers ({len(coverage['missing'])}):\n")
                for name in coverage["missing"]:
                    f.write(f"  - {name}\n")

            if coverage["extra"]:
                f.write(f"\nExtra markers ({len(coverage['extra'])}):\n")
                for name in coverage["extra"]:
                    f.write(f"  - {name}\n")

            # Symmetry check
            f.write("\n" + "-" * 70 + "\n")
            f.write("BILATERAL SYMMETRY\n")
            f.write("-" * 70 + "\n")

            symmetry_issues = self._validate_marker_symmetry(markers)

            if not symmetry_issues:
                f.write("All marker pairs are symmetric within tolerance\n")
            else:
                f.write(f"Found {len(symmetry_issues)} asymmetric pairs:\n\n")
                for pair_name, issue in symmetry_issues.items():
                    f.write(f"  {pair_name}:\n")
                    if issue['status'] == 'missing':
                        f.write(f"    {issue['message']}\n")
                    else:
                        f.write(f"    Y diff: {issue['y_diff']:.4f}\n")
                        f.write(f"    Z diff: {issue['z_diff']:.4f}\n")
                        f.write(f"    X mirror diff: {issue['x_mirror_diff']:.4f}\n")
                        f.write(f"    Relative error: {issue['relative_error']:.2%}\n")
                    f.write("\n")

            # SMPL marker reference check
            f.write("\n" + "-" * 70 + "\n")
            f.write("COMPARISON WITH SMPL REFERENCE\n")
            f.write("-" * 70 + "\n")

            distances = self._compare_mmd_to_smpl(markers, normalize=True)

            if distances:
                mean_dist = np.mean(list(distances.values()))
                max_dist = max(distances.values())
                max_dist_marker = max(distances, key=distances.get)

                f.write(f"Mean distance: {mean_dist:.4f}\n")
                f.write(f"Max distance: {max_dist:.4f} (at {max_dist_marker})\n\n")

                f.write("Per-marker distances (top 10 largest):\n")
                sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:10]
                for name, dist in sorted_distances:
                    f.write(f"  {name:25s}: {dist:.4f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        self.logger.info(f"Validation report saved to: {output_txt}")

    def execute(self):
        try:
            markers = self._annotate_single_model()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w') as f:
                json.dump(markers, f, indent=4)
            self.logger.info(f"  Found {len(markers)}/{len(self.expected_marker_coords)} markers")

            ### debug
            # marker_pos = np.array([markers[marker_name]["vertex_position"] for marker_name in markers.keys()])
            # print(marker_pos.shape)
            # np.savetxt("./check_mmd_marker_v4.txt", marker_pos)
            # exit()
            ###

            if self.do_validate:
                report_path = self.output_path.parent / "validation.txt"
                self._generate_validation_report(self.output_path, report_path)

            if self.do_visualize:
                self._visualize_markers(markers)
            
            self.logger.critical(f"✅ Marker annotation finished, check: {self.output_path.parent}")
        except Exception as e:
            self.logger.error(f"Marker annotation for {self.pmx_path.name} failed: {e}")
            raise RuntimeError(f"Marker annotation for {self.pmx_path.name} failed: {e}")
        finally:
            self._quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Directory containing .pmx files")
    parser.add_argument("--validate", action="store_true", help="Generate validation reports after annotation")
    parser.add_argument("--visualize", action="store_true", help="Visualize markers on model after annotation")
    parser.add_argument("--marker_size", type=float, default=0.01, help="Marker sphere size for visualization")
    args = parser.parse_args()

    # test single file processing
    annotator = AnimerMarkerAnnotator(
        input_path=args.input_path,
        validate=args.validate,
        visualize=args.visualize,
        marker_size=args.marker_size,
    )
    annotator.execute()