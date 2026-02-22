import bpy
import argparse
import math
import re
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

from bodymodels import SMPLMesh
from bodymodels.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from .base import AnimerBase
from .smplify import SMPLify3D
from .config.joint_correspondence import *
from .utils import quat

class AnimerRetargeter(AnimerBase):
    def __init__(
        self,
        input_path: Path,
        shape_path: Path,
        batch_size: int = 1,
        num_iters: int = 100,
        use_collision: bool = True,
        version: str = "debug"
    ):
        super().__init__()

        self.base_path = Path(__file__).resolve().parent
        self.input_path = input_path
        self.shape_path = shape_path
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.use_collision = use_collision

        # Initialize SMPL-H body model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.body_model = SMPLMesh().to(self.device)

        # load shape fitting result
        shape_info = np.load(str(shape_path), allow_pickle=True).item()
        self.smpl_shapes = torch.from_numpy(shape_info["shapes"]).float().to(self.device)
        self.mmd_to_standard_scale = 1 / shape_info["scale_to_mmd"]

        # Define A-pose
        apose_arm_angle = np.pi / 6
        c, s = math.cos(apose_arm_angle), math.sin(apose_arm_angle)
        self.left_arm_pose = quat.from_xform(np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]))
        self.right_arm_pose = quat.from_xform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

        # Initialize simplify solver
        smpl_info = np.load("bodymodels/human_model_files/smpl_mesh_info.npy", allow_pickle=True).item()
        self.smpl_faces = torch.from_numpy(smpl_info["faces"]).int().to(self.device)
        self.smplify = SMPLify3D(
            smplxmodel=self.body_model,
            smplxfaces=self.smpl_faces,
            batch_size=self.batch_size,
            joints_category="AMASS",
            num_iters=self.num_iters,
            use_collision=self.use_collision,
            device=self.device
        )

        self.pre_output_path = (self.base_path / "outputs" / version / "retargeting" / "body_pose_retargeting" / self.input_path.name).with_suffix(".npy")
        self.output_path = (self.base_path / "outputs" / version / "retargeting" / "body_hand_pose_retargeting" / self.input_path.name).with_suffix(".npy")

    def _get_frame_range(self):
        """
        Get valid frame range according to the .blend file name.
        """
        start_frame = int(re.search(f"{re.escape('_skip')}(.*?){re.escape('frames')}", self.input_path.stem).group(1))
        duration = float(re.search(f"{re.escape('_duration')}(.*?){re.escape('s')}", self.input_path.stem).group(1))
        end_frame = int(start_frame + duration * 30)
        return start_frame, end_frame

    def _extract_mmd_joints_3d(self, armature, frame):
        """
        Get the joint position at given frame.
        """
        scene = bpy.context.scene
        mat = armature.matrix_world
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        joint_positions = np.zeros((52, 3))
        for i in range(len(mmd_skeleton)):
            mmd_bone_name = mmd_skeleton[i]
            smpl_bone_idx = smpl_skeleton[i]
            mmd_bone = armature.pose.bones.get(mmd_bone_name, None)
            if mmd_bone:
                joint_positions[smpl_bone_idx] = np.array((mat @ mmd_bone.matrix).to_translation())

        return joint_positions.reshape(1, 52, 3)

    def _retarget_body(self, joints3d, start_frame, end_frame):
        """
        MMD skeleton is global joint free, needs to pre-fit a global rotation.
        """
        smpl_param = {
            "rot6d": [],
            "trans": [],
            "shapes": []
        }

        prev_pose = torch.tensor([[[1., 0., 0., 1., 0., 0.]]]).repeat(self.batch_size, 52, 1).float().to(self.device)
        prev_cam_t = torch.zeros(self.batch_size, 3).to(self.device)
        for idx in tqdm(range(start_frame, end_frame), desc=f"retargeting body pose..."):
            frame_target_joints3d = torch.Tensor(joints3d[idx-self.action_start_frame]).view(self.batch_size, 22, 3).to(self.device)
            frame_confidence = torch.ones(22).to(self.device)
            is_zero = (frame_target_joints3d[0].abs().sum(dim=-1) == 0)
            frame_confidence[is_zero] = 0.0

            fit_vertices, \
            fit_joints, \
            fit_pose, \
            fit_betas, \
            fit_cam_t = self.smplify(
                prev_pose.detach(),
                self.smpl_shapes,
                prev_cam_t.detach(),
                frame_target_joints3d,
                conf_3d=frame_confidence,
                seq_ind=idx
            )

            prev_pose = fit_pose.detach()
            prev_cam_t = fit_cam_t.detach()

            smpl_param["rot6d"].append(fit_pose.detach().cpu().numpy())
            smpl_param["trans"].append(fit_cam_t.detach().cpu().numpy())
            smpl_param["shapes"].append(self.smpl_shapes.detach().cpu().numpy())
        for k in smpl_param.keys():
            smpl_param[k] = np.concatenate(smpl_param[k], axis=0)

        smpl_param["scale_to_mmd"] = 1 / self.mmd_to_standard_scale
        
        self.pre_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.pre_output_path, smpl_param)
        self.logger.critical(f"✅ Body pose retargeting result has been saved to {self.pre_output_path}")

    @staticmethod
    def _rot6d_to_quat(rot6d):
        if isinstance(rot6d, np.ndarray):
            rot6d = torch.from_numpy(rot6d).float().cpu()
        elif isinstance(rot6d, torch.tensor):
            rot6d = rot6d.float().cpu()
        return quat.from_xform(rot6d_to_rotation_matrix(rot6d).numpy())

    @staticmethod
    def _quat_to_rot6d(q):
        return rotation_matrix_to_rot6d(torch.from_numpy(quat.to_xform(q)).float()).numpy()

    def _to_smpl(self, mmd_quat, smpl_index):
        left_arms = [13, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        right_arms = [14, 17, 19, 21, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        knees = [4, 5]
        ankles = [7, 8]
        if smpl_index in left_arms:
            return np.array([mmd_quat[0], mmd_quat[2], mmd_quat[3], mmd_quat[1]])
        elif smpl_index in right_arms:
            return np.array([mmd_quat[0], -1*mmd_quat[2], -1*mmd_quat[3], mmd_quat[1]])
        elif smpl_index in knees:
            return np.array([mmd_quat[0], mmd_quat[1], -1*mmd_quat[2], -1*mmd_quat[3]])
        elif smpl_index in ankles:
            return np.array([mmd_quat[0], mmd_quat[1], -1*mmd_quat[3], mmd_quat[2]])
        else:
            return mmd_quat

    def _retarget(self, armature, start_frame, end_frame):
        """
        Retarget MMD pose to SMPL at given frame.
        """
        smpl_param = {
            "rot6d": [],
            "trans": [],
            "shapes": []    # TODO: read from shape fitting result
        }
        mmd_joints = []
        pre_retarget_result = np.load(self.pre_output_path, allow_pickle=True).item()
        global_r = pre_retarget_result["rot6d"][:, :1, :]

        scene = bpy.context.scene
        mat = armature.matrix_world
        for frame in tqdm(range(start_frame, end_frame)):
            scene.frame_set(frame)
            bpy.context.view_layer.update()

            frame_global_r = self._rot6d_to_quat(global_r[frame-start_frame])
            joint_rotations = np.zeros((52, 4)) # w x y z
            joint_positions = np.zeros((52, 3))
            for i in range(len(mmd_skeleton)):
                mmd_bone_name = mmd_skeleton[i]
                smpl_bone_idx = smpl_skeleton[i]
                
                mmd_bone = armature.pose.bones.get(mmd_bone_name, None)
                if mmd_bone:
                    ik_driven = False
                    for c in mmd_bone.constraints:
                        if c.type == "IK":
                            ik_driven = True
                            break

                    if smpl_bone_idx == 0:
                        if ik_driven:
                            self.logger.error(f"MMD bone {mmd_bone_name} should not be IK driven")
                            raise RuntimeError(f"MMD bone {mmd_bone_name} should not be IK driven")
                        
                        mmd_bone_rotation = frame_global_r
                    elif smpl_bone_idx == 3:
                        if ik_driven:
                            self.logger.error(f"MMD bone {mmd_bone_name} should not be IK driven")
                            raise RuntimeError(f"MMD bone {mmd_bone_name} should not be IK driven")
                        
                        mmd_bone_rotation = np.array(mmd_bone.matrix_basis.to_quaternion())
                        mmd_bone_rotation = quat.mul(quat.inv(frame_global_r), mmd_bone_rotation)
                    elif smpl_bone_idx == 1 or smpl_bone_idx == 2:
                        if ik_driven:
                            self.logger.error(f"MMD bone {mmd_bone_name} should not be IK driven")
                            raise RuntimeError(f"MMD bone {mmd_bone_name} should not be IK driven")
                        
                        lower_root = armature.pose.bones.get("下半身")
                        lower_root_rotation = np.array(lower_root.matrix_basis.to_quaternion())
                        lower_root_rotation[2] *= -1; lower_root_rotation[3] *= -1
                        lower_root_rotation = quat.mul(quat.inv(frame_global_r), lower_root_rotation)

                        mmd_bone_rotation = np.array(mmd_bone.matrix_basis.to_quaternion())
                        mmd_bone_rotation[2] *= -1; mmd_bone_rotation[3] *= -1
                        mmd_bone_rotation = quat.mul(mmd_bone_rotation, lower_root_rotation)
                    else:
                        if ik_driven:
                            mmd_bone_rotation = np.array(mmd_bone.matrix.to_quaternion())
                            parent_bone_rotation = np.array(mmd_bone.parent.matrix.to_quaternion())
                            mmd_bone_rotation = quat.mul(quat.inv(parent_bone_rotation), mmd_bone_rotation)
                        else:
                            mmd_bone_rotation = np.array(mmd_bone.matrix_basis.to_quaternion())
                            mmd_bone_rotation = self._to_smpl(mmd_bone_rotation, smpl_bone_idx)
                    mmd_bone_translation = np.array(mmd_bone.matrix.to_translation())
                else:
                    mmd_bone_rotation = np.array([[1., 0., 0., 0.]])
                    mmd_bone_translation = np.array([[0., 0., 0.]])
                joint_rotations[smpl_bone_idx] = mmd_bone_rotation
                joint_positions[smpl_bone_idx] = mmd_bone_translation
            joint_rotations[13] = quat.mul(joint_rotations[13], self.left_arm_pose)
            joint_rotations[14] = quat.mul(joint_rotations[14], self.right_arm_pose)
            joint_rotations = self._quat_to_rot6d(joint_rotations)

            smpl_param["rot6d"].append(joint_rotations.reshape(1, 52, 6))
            smpl_param["trans"].append(np.array(armature.pose.bones.get("上半身").matrix.to_translation()).reshape(1, 3))
            mmd_joints.append(joint_positions.reshape(1, 52, 3))

        smpl_param["rot6d"] = np.concatenate(smpl_param["rot6d"], axis=0)
        smpl_param["trans"] = np.concatenate(smpl_param["trans"], axis=0)
        smpl_param["shapes"] = np.zeros((smpl_param["rot6d"].shape[0], 16))
        mmd_joints = np.concatenate(mmd_joints, axis=0)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.output_path, smpl_param)
        self.logger.critical(f"✅ Pose retargeting result has been saved to {self.output_path}")

        return smpl_param, mmd_joints

    def _retarget_hand(self, armature, start_frame, end_frame):
        """
        Retarget MMD pose to SMPL at given frame.
        """
        mmd_joints = []
        body_hand_pose_retarget = np.load(self.pre_output_path, allow_pickle=True).item()

        scene = bpy.context.scene
        mat = armature.matrix_world
        for frame in tqdm(range(start_frame, end_frame)):
            scene.frame_set(frame)
            bpy.context.view_layer.update()

            for hand_jid in hand_joint_idx:
                mmd_bone_name = mmd_skeleton[hand_jid]
                smpl_bone_idx = smpl_skeleton[hand_jid]
                
                mmd_bone = armature.pose.bones.get(mmd_bone_name, None)
                if mmd_bone:
                    ik_driven = False
                    for c in mmd_bone.constraints:
                        if c.type == "IK":
                            ik_driven = True
                            break
                    
                    if ik_driven:
                        mmd_bone_rotation = np.array(mmd_bone.matrix.to_quaternion())
                        parent_bone_rotation = np.array(mmd_bone.parent.matrix.to_quaternion())
                        mmd_bone_rotation = quat.mul(quat.inv(parent_bone_rotation), mmd_bone_rotation)
                    else:
                        mmd_bone_rotation = np.array(mmd_bone.matrix_basis.to_quaternion())
                        mmd_bone_rotation = self._to_smpl(mmd_bone_rotation, smpl_bone_idx)
                    body_hand_pose_retarget["rot6d"][frame-start_frame, smpl_bone_idx] = self._quat_to_rot6d(mmd_bone_rotation)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.output_path, body_hand_pose_retarget)
        self.logger.critical(f"✅ Body and hand pose retargeting result has been saved to {self.output_path}")

    def execute(self):
        try:
            self._load_initial_scene(self.input_path)
            armature = self._select_character_armature()
            
            # extract MMD 3D joints
            self.action_start_frame, self.action_end_frame = self._get_frame_range()
            joints3d = []
            for frame in range(self.action_start_frame, self.action_end_frame):
                joints3d.append(self._extract_mmd_joints_3d(armature, frame))
            joints3d = np.concatenate(joints3d, axis=0) # NOTE: if a given mmd bone name does not exist, the joint position is default [0,0,0]
            joints3d *= self.mmd_to_standard_scale

            start_frame = self.action_start_frame
            end_frame = self.action_end_frame

            if not self.pre_output_path.exists():
                self._retarget_body(joints3d[:, :22, :], start_frame, end_frame)
            self._retarget_hand(armature, start_frame, end_frame)
        except Exception as e:
            self.logger.error(f"Retargeting for {self.input_path} failed: {e}")
            raise RuntimeError(f"Retargeting for {self.input_path} failed: {e}")
        finally:
            self._quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--shape_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--use_collision", action="store_true")
    args = parser.parse_args()

    retargeter = AnimerRetargeter(
        input_path=Path(args.input_path),
        shape_path=Path(args.shape_path),
        batch_size=args.batch_size,
        num_iters=args.num_iters,
        use_collision=args.use_collision
    )
    retargeter.execute()