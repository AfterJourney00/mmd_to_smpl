import bpy
import torch
import numpy as np

import json
import math
import argparse
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

from .annotate_mmd_marker import AnimerMarkerAnnotator
from .utils import standardize_file_name, get_project_root

class AnimerShapeFitter(AnimerMarkerAnnotator):
    def __init__(
        self,
        input_path,
        standard_height: float = 1.7,
        learning_rate: float = 0.1,
        max_iter: int = 300,
        w_shape_reg: float = 1e-4,
        w_pose_reg: float = 1e-3,
        visualize: bool = False,
        version: str = "debug"
    ):
        super().__init__(input_path, version=version)

        self.standard_height = standard_height
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.do_visualize = visualize
        if self.do_visualize:
            self.vis_frames = []
            self.vis_interval = max(1, self.max_iter // 60) # Capture ~60 frames total (first/last iteration always included)

        self.smpl_marker_name = list(self.smpl_marker_index.keys())
        if self.output_path.exists():
            with open(str(self.output_path), "r") as f:
                self.mmd_marker_annot = json.load(f)
        else:
            self.logger.error(f"Marker annotation of {self.pmx_path} is missing")
            raise RuntimeError(f"Marker annotation of {self.pmx_path} is missing")
        self.mmd_marker_name = list(self.mmd_marker_annot.keys())
        self.common_marker_name = list(set(self.smpl_marker_name) & set(self.mmd_marker_name))
        smpl_info = np.load("bodymodels/human_model_files/smpl_mesh_info.npy", allow_pickle=True).item()
        self.smpl_faces = torch.from_numpy(smpl_info["faces"]).int()

        self._scale_mmd_markers()
        self.target_markers = torch.tensor([self.mmd_marker_annot[marker_name]["vertex_position"] for marker_name in self.common_marker_name]).float().to(self.device)
        
        apose_arm_angle = np.pi / 6
        c, s = math.cos(apose_arm_angle), math.sin(apose_arm_angle)
        self._prev_pose = torch.tensor([[[1., 0., 0., 1., 0., 0.]]]).repeat(1, 13, 1).float().to(self.device)   # joint 0 - joint 12
        self.left_arm_pose = torch.tensor([[[c, s, -s, c, 0., 0.]]]).float().to(self.device).requires_grad_()   # joint 13
        self.right_arm_pose = torch.tensor([[[c, -s, s, c, 0., 0.]]]).float().to(self.device).requires_grad_()  # joint 14
        self._post_pose = torch.tensor([[[1., 0., 0., 1., 0., 0.]]]).repeat(1, 37, 1).float().to(self.device)   # joint 15 - joint 52
        self.smpl_shape = torch.zeros([1, 16], dtype=torch.float32).to(self.device).requires_grad_()
        self.smpl_trans = torch.zeros([1, 3], dtype=torch.float32).to(self.device)

        self.left_arm_pose_reg = torch.tensor([[[c, s, -s, c, 0., 0.]]]).float().to(self.device)
        self.right_arm_pose_reg = torch.tensor([[[c, -s, s, c, 0., 0.]]]).float().to(self.device)
        
        self.optimizer = torch.optim.Adam([self.smpl_shape, self.left_arm_pose, self.right_arm_pose], lr=self.learning_rate)
        self.w_shape_reg = w_shape_reg
        self.w_pose_reg = w_pose_reg

        pmx_name = standardize_file_name(self.pmx_path.stem)
        self.output_path = self.base_path / "outputs" / version / "auto_marker_annotate" / pmx_name / (pmx_name+".json")

    def _scale_mmd_markers(self):
        """
        Normalize MMD character to a certain, normal height.
        """
        self._load_initial_scene(Path("a_virtual_path_that_not_exists"))
        self._ensure_addon_loaded("mmd_tools")
        self._import_mmd_model()

        armature = self._select_character_armature()
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not armature or not mesh_objects:
            self.logger.error(f"Invalid model structure in {self.pmx_path.name}")
            raise RuntimeError(f"Invalid model structure in {self.pmx_path.name}")
        self.logger.info(f"Found {len(mesh_objects)} meshes of {self.pmx_path.name} for ray casting")

        # Compute MMD character height
        global_min_z = float('inf')
        global_max_z = float('-inf')
        for mesh_obj in mesh_objects:
            mat = mesh_obj.matrix_world
            for v in mesh_obj.data.vertices:
                world_z = (mat @ v.co).z
                if world_z < global_min_z:
                    global_min_z = world_z
                if world_z > global_max_z:
                    global_max_z = world_z
        self.mmd_height = global_max_z - global_min_z
        self.scale_factor = self.standard_height / self.mmd_height
        self.logger.info(f"Character {self.pmx_path.name} has height: {self.mmd_height:.4f} (z range: {global_min_z:.4f} ~ {global_max_z:.4f})")
        self.logger.info(f"Character {self.pmx_path.name} needs scale {self.scale_factor} times to {self.standard_height} meters height")
        self._quit()

        # Scale MMD character markers
        for marker_name in self.mmd_marker_name:
            self.mmd_marker_annot[marker_name]["hit_position"] = (np.array(self.mmd_marker_annot[marker_name]["hit_position"]) * self.scale_factor).tolist()
            self.mmd_marker_annot[marker_name]["vertex_position"] = (np.array(self.mmd_marker_annot[marker_name]["vertex_position"]) * self.scale_factor).tolist()

    def _umeyama_alignment(self, X, Y):
        """
        Align two 3D point sets using Umeyama's method.
        """
        mu_X = X.mean(axis=0)
        mu_Y = Y.mean(axis=0)
        X0 = X - mu_X
        Y0 = Y - mu_Y
        U, S, Vt = np.linalg.svd(Y0.T @ X0)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        t = mu_Y - R @ mu_X
        return R, t

    def _apply_transform(self, vertices, R, t):
        """
        Apply the transformation to the mesh vertices.
        """
        return (vertices @ R.T) + t

    @torch.no_grad()
    def _render_fitting_frame(self, smpl_verts, smpl_markers, iteration, loss_dict):
        """
        Render a frame showing SMPL mesh + markers from front and side views.
        Returns an RGB numpy array (H, W, 3) as uint8.
        """
        fig = plt.figure(figsize=(16, 7))

        target_np = self.target_markers if isinstance(self.target_markers, np.ndarray) else self.target_markers.detach().cpu().numpy()
        smpl_m_np = smpl_markers if isinstance(smpl_markers, np.ndarray) else smpl_markers.detach().cpu().numpy()
        verts_np = smpl_verts if isinstance(smpl_verts, np.ndarray) else smpl_verts.detach().cpu().numpy()

        # Subsample faces for faster rendering
        face_step = max(1, len(self.smpl_faces) // 2000)
        faces_sub = self.smpl_faces[::face_step]

        views = [(0, 90, "Front View"), (0, 0, "Side View")]
        for idx, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
            # Draw mesh
            triangles = verts_np[faces_sub]
            poly = Poly3DCollection(triangles, alpha=0.15, facecolor="skyblue", edgecolor="gray", linewidth=0.1)
            ax.add_collection3d(poly)
            # Draw markers
            ax.scatter(*target_np.T, c="red", s=40, marker="o", label="Target (MMD)")
            ax.scatter(*smpl_m_np.T, c="blue", s=40, marker="^", label="Fitted (SMPL)")
            # Draw correspondence lines
            for t, s in zip(target_np, smpl_m_np):
                ax.plot([t[0], s[0]], [t[1], s[1]], [t[2], s[2]], c="green", alpha=0.4, linewidth=0.8)

            ax.view_init(elev=elev, azim=azim)
            # Set axis limits from mesh bounds
            margin = 0.1
            mins, maxs = verts_np.min(axis=0) - margin, verts_np.max(axis=0) + margin
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_title(title)
            if idx == 0:
                ax.legend(fontsize=8, loc="upper left")

        loss_str = " | ".join(f"{k}: {v:.6f}" for k, v in loss_dict.items())
        fig.suptitle(f"Iteration {iteration}  |  {loss_str}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        # Rasterize to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        plt.close(fig)
        return frame

    def execute(self, converge_thres=1e-8):
        """
        Run the optimization loop to fit the SMPL model to the MMD character.
        """
        try:
            # Initial forward
            with torch.no_grad():
                initial_pose = {
                    "rot6d": torch.cat([self._prev_pose, self.left_arm_pose, self.right_arm_pose, self._post_pose], dim=1),
                    "trans": self.smpl_trans,
                    "shapes": self.smpl_shape,
                }
                initial_pose = self._to_device(initial_pose, self.device)
                initial_vertices = self.body_model(initial_pose)["vertices"][0].cpu().numpy()
                initial_markers = np.array([initial_vertices[self.smpl_marker_index[marker_name]] for marker_name in self.common_marker_name])

            # Pre-alignment to eliminate orientation gap of SMPL and MMD character (+Y up <-> +Z up)
            R, t = self._umeyama_alignment(initial_markers, self.target_markers.cpu().numpy())
            R = torch.tensor(R, dtype=torch.float32).to(self.device)
            t = torch.tensor(t, dtype=torch.float32).to(self.device)

            # Shape fitting loop
            prev_loss = None
            pbar = tqdm(range(self.max_iter))
            for i in pbar:
                smpl_pose_i = {
                    "rot6d": torch.cat([self._prev_pose, self.left_arm_pose, self.right_arm_pose, self._post_pose], dim=1),
                    "trans": self.smpl_trans,
                    "shapes": self.smpl_shape,
                }
                smpl_vertices_i = self.body_model(smpl_pose_i)["vertices"][0]
                smpl_vertices_i = self._apply_transform(smpl_vertices_i, R, t)
                smpl_markers_i = smpl_vertices_i[[self.smpl_marker_index[marker_name] for marker_name in self.common_marker_name]]

                marker_loss = torch.mean((smpl_markers_i - self.target_markers) ** 2)
                shape_reg = self.w_shape_reg * torch.mean(self.smpl_shape ** 2)
                left_arm_reg = self.w_pose_reg * torch.mean((self.left_arm_pose - self.left_arm_pose_reg) ** 2)
                right_arm_reg = self.w_pose_reg * torch.mean((self.right_arm_pose - self.right_arm_pose_reg) ** 2)
                loss = marker_loss + shape_reg + left_arm_reg + right_arm_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(f"After {i} iterations, total_loss: {loss.item():4f} | marker_loss: {marker_loss.item():4f} | shape_reg: {shape_reg.item():4f} | left_arm_reg: {left_arm_reg.item():4f} | right_arm_reg: {right_arm_reg.item():4f}")

                cur_loss = loss.item()
                if prev_loss is not None and abs(prev_loss - cur_loss) < converge_thres:
                    self.logger.info(f"Converged at iteration {i} (loss delta < {converge_thres})")
                    break
                prev_loss = cur_loss

                if self.do_visualize:
                    if i % self.vis_interval == 0 or i == self.max_iter - 1:
                        loss_dict = {
                            "total": loss.item(),
                            "marker": marker_loss.item(),
                            "shape_reg": shape_reg.item(),
                        }
                        frame = self._render_fitting_frame(
                            smpl_vertices_i,
                            smpl_markers_i,
                            i,
                            loss_dict
                        )
                        self.vis_frames.append(frame)
            pbar.close()

            # Save optimization video
            if self.do_visualize:
                video_path = f"./shape_fitting_mi{self.max_iter}_lr{self.learning_rate}_ws{self.w_shape_reg}_wp{self.w_pose_reg}.mp4"
                imageio.mimwrite(video_path, self.vis_frames, fps=10, quality=8)
                self.logger.info(f"Shape fitting video saved to {video_path} ({len(self.vis_frames)} frames)")
            
            # Save result
            fitting_result = {
                "shapes": self.smpl_shape.detach().cpu().numpy(),
                "scale_to_mmd": 1 / self.scale_factor
            }
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.output_path, fitting_result)
            self.logger.critical(f"✅ Shape fitting result has been saved to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Shape fitting for {self.pmx_path.name} failed: {e}")
            raise RuntimeError(f"Shape fitting for {self.pmx_path.name} failed: {e}")
        finally:
            self._quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Directory containing .pmx files")
    parser.add_argument("--standard_height", type=float, default=1.7)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--w_shape_reg", type=float, default=5e-4)
    parser.add_argument("--w_pose_reg", type=float, default=1e-3)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--version", type=str, default="debug")
    args = parser.parse_args()

    # test single file processing
    shape_fitter = AnimerShapeFitter(
        input_path=args.input_path,
        standard_height=args.standard_height,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        w_shape_reg=args.w_shape_reg,
        w_pose_reg=args.w_pose_reg,
        visualize=args.visualize,
        version=args.version,
    )
    shape_fitter.execute()