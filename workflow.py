import argparse
import random
import json
import numpy as np
from collections import deque
from pathlib import Path
from collections import defaultdict

from .base import AnimerBase
from .make_blend_file import AnimerBlendComposer
from .annotate_mmd_marker import AnimerMarkerAnnotator
from .shape_fitting import AnimerShapeFitter
from .pose_retargeting import AnimerRetargeter
from .render import AnimerRenderer
from .utils import get_project_root

class AnimerWorkflow(AnimerBase):
    def __init__(
        self,
        global_size: int,
        local_rank: int,
        reverse: bool = False,
        random_seed: int = 42,

        bake: bool = True,
        no_physics: bool = False,
        disable_ik: bool = False,
        
        render: bool = True,
        engine: str = "CYCLES",
        res_x: int = 1920,
        res_y: int = 1080,
        focal: int = 65,
        start_frame: int = 60,
        duration: float = 10.0,
        style: list[str] = ["toon", "sketch"],
        samples: int = 8,
        use_gpu: bool = True,
        n_video_per_animation: int = 3,
        
        retarget: bool = True,
        ma_validate: bool = False,
        ma_visualize: bool = False,
        ma_marker_size: float = 0.01,

        sf_standard_height: float = 1.7,
        sf_learning_rate: float = 0.1,
        sf_max_iter: int = 300,
        sf_w_shape_reg: float = 5e-4,
        sf_w_pose_reg: float = 1e-3,
        sf_visualize: bool = False,

        pf_batch_size: int = 1,
        pf_num_iters: int = 100,
        pf_use_collision: bool = False,
        pf_visualize: bool = False,
        
        version: str = ""
    ):
        super().__init__()
        
        self.global_size = global_size
        self.local_rank = local_rank
        self.random_seed = random_seed

        # for composer
        self.bake = bake
        self.no_physics = no_physics
        self.disable_ik = disable_ik

        # for renderer
        self.render = render
        self.engine = engine
        self.res_x = res_x
        self.res_y = res_y
        self.focal = focal
        self.start_frame = start_frame
        self.duration = duration
        self.style = style
        self.samples = samples
        self.use_gpu = use_gpu

        # for retargeting, ma: marker annotation, sf: shape fitting, pf: pose fitting
        self.retarget = retarget
        self.ma_validate = ma_validate
        self.ma_visualize = ma_visualize
        self.ma_marker_size = ma_marker_size
        self.sf_standard_height = sf_standard_height
        self.sf_learning_rate = sf_learning_rate
        self.sf_max_iter = sf_max_iter
        self.sf_w_shape_reg = sf_w_shape_reg
        self.sf_w_pose_reg = sf_w_pose_reg
        self.sf_visualize = sf_visualize
        self.pf_batch_size = pf_batch_size
        self.pf_num_iters = pf_num_iters
        self.pf_use_collision = pf_use_collision
        self.pf_visualize = pf_visualize

        # meta control
        self.n_video_per_animation = n_video_per_animation
        self.version = version
        if not self.version:
            self.version = "debug"
        
        # load (character, animation, scene) file list
        character_list = self._load_file_list("example/character_file_list.txt")
        animation_list = self._load_file_list("example/motion_file_list.txt")
        scene_list = self._load_file_list("example/scene_file_list.txt")
        self.logger.info(f"Loaded {len(character_list)} characters, {len(animation_list)} animations, {len(scene_list)} scenes")
        
        # randomize character and scene lists, convert to queues
        self.character_queue = self._refresh_file_queue(character_list, animation_list)
        self.scene_queue = self._refresh_file_queue(scene_list, animation_list)

        # construct and split triplet file list for current rank
        self.tri_path_list = []
        for anim in animation_list:
            for _ in range(self.n_video_per_animation):
                self.tri_path_list.append({
                    "pmx_path": Path(self.character_queue.popleft()),
                    "vmd_path": Path(anim),
                    "blend_path": Path(self.scene_queue.popleft()),
                })
        
        # record global path tracer before reverse and splitting
        self.global_path_tracer = self.dry_execute()

        # reverse and split all paths to local ranks
        if reverse:
            self.tri_path_list = self.tri_path_list[::-1]
        self._split_for_rank()
        self.logger.info(f"Rank {local_rank}/{global_size}: assigned {len(self.tri_path_list)} compositions")
    
    def _load_file_list(self, filename: str) -> list:
        project_root = get_project_root()
        file_path = project_root / filename
        if not file_path.exists():
            self.logger.error(f"File list not found: {file_path}")
            raise FileNotFoundError(f"File list not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()][1:]    # the first row records file num
        return lines
    
    def _split_for_rank(self):
        self.tri_path_list = [
            self.tri_path_list[i] 
            for i in range(len(self.tri_path_list)) 
            if i % self.global_size == self.local_rank
        ]
    
    def _set_all_seed(self, loop: int):
        random.seed(self.random_seed + loop)

    def _get_next_from_scene_queue(self) -> str:
        return self.scene_queue.popleft()
    
    def _refresh_file_queue(self, file_list, animation_list):
        # suppose we randomly assign 3 different characters and 3 different scenes to each animation
        queue = []
        q_repeat_num = (len(animation_list) * self.n_video_per_animation) // len(file_list) + 1
        for i in range(q_repeat_num):
            self._set_all_seed(i)
            _queue = file_list[:]
            random.shuffle(_queue)
            queue.extend(_queue)
        return deque(queue)

    def _visualize(self, smpl_param):
        import torch
        import imageio
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras,
            PointLights,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader,
            TexturesVertex,
        )
        from bodymodels import SMPLMesh

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        smpl_info = np.load("bodymodels/human_model_files/smpl_mesh_info.npy", allow_pickle=True).item()
        smpl_faces = torch.from_numpy(smpl_info["faces"]).int().to(device)

        with torch.no_grad():
            body_model = SMPLMesh().to(device)
            _ = smpl_param.pop("scale_to_mmd")
            smpl_param = self._to_device(smpl_param, device)
            smpl_verts = body_model(smpl_param)["vertices"]

        # ensure smpl_verts is (N, V, 3)
        if smpl_verts.ndim == 2:
            smpl_verts = smpl_verts[None]

        # renderer setup
        R, T = look_at_view_transform(dist=2.5, elev=0, azim=0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
        lights = PointLights(device=device, location=[[0.0, 1.0, 2.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )

        frames = []
        for i in range(smpl_verts.shape[0]):
            verts_i = smpl_verts[i].unsqueeze(0)
            textures = TexturesVertex(verts_features=torch.ones_like(verts_i) * 0.7)
            mesh = Meshes(verts=verts_i, faces=smpl_faces.unsqueeze(0), textures=textures)
            image = renderer(mesh)  # (1, H, W, 4)
            frame = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)

        out_path = f"smpl_vis_{self.version}.mp4"
        imageio.mimwrite(out_path, frames, fps=30, quality=8)
        self.logger.info(f"Saved SMPL mesh visualization to {out_path}")
        exit()

    def dry_execute(self):
        fn = Path(f"./global_path_tracer_{self.version}.json")
        if fn.exists():
            with open(str(fn), "r", encoding="utf-8") as f:
                global_path_tracer = json.load(f)
        else:
            global_path_tracer = defaultdict(dict)
            for idx, tri_path in enumerate(self.tri_path_list):
                composer = AnimerBlendComposer(
                    tri_path,
                    skip_animation_morph=self.start_frame,
                    duration=self.duration,
                    bake=self.bake,
                    no_physics=self.no_physics,
                    disable_ik=self.disable_ik,
                    version=self.version
                )
                renderer = AnimerRenderer(
                    input_path=str(composer.output_path),
                    engine=self.engine,
                    res_x=self.res_x,
                    res_y=self.res_y,
                    focal=self.focal,
                    start_frame=self.start_frame,
                    duration=self.duration,
                    style=None,
                    samples=self.samples,
                    use_gpu=self.use_gpu,
                    version=self.version
                )
                for k in tri_path.keys():
                    global_path_tracer[composer.output_path.name][k] = str(tri_path[k])
                    global_path_tracer[renderer.output_path.name][k] = str(tri_path[k])
                global_path_tracer[composer.output_path.name]["video_path"] = str(renderer.output_path)
                global_path_tracer[renderer.output_path.name]["compose_path"] = str(composer.output_path)
            
            with open(str(fn), "w", encoding="utf-8") as f:
                json.dump(global_path_tracer, f, indent=4)

        return global_path_tracer
    
    def execute(self):
        self.logger.info(f"Starting workflow execution for rank {self.local_rank}")
        self.logger.info(f"{len(self.tri_path_list)} compositions in total")
        
        for idx, tri_path in enumerate(self.tri_path_list):
            self.logger.info(f"Processing composition {idx + 1}/{len(self.tri_path_list)}")
            self.logger.info(f">>> Character: {tri_path['pmx_path']}")
            self.logger.info(f">>> Animation: {tri_path['vmd_path']}")
            self.logger.info(f">>> Scene: {tri_path['blend_path']}")
            
            try:
                composer = AnimerBlendComposer(
                    tri_path,
                    skip_animation_morph=self.start_frame,
                    duration=self.duration,
                    bake=self.bake,
                    no_physics=self.no_physics,
                    disable_ik=self.disable_ik,
                    version=self.version
                )
                if composer.output_path.exists():
                    self.logger.info(f"Blender file already created, skip: {composer.output_path}")
                else:
                    composer.execute()
                    self.logger.info(f"Successfully processed current composition, check: {composer.output_path}")
                
                # For each rendering style, initialize a renderer, which takes the output .blend file of composer as input
                # NOTE: self.style only has one element
                for render_style in self.style:
                    renderer = AnimerRenderer(
                        input_path=str(composer.output_path),
                        engine=self.engine,
                        res_x=self.res_x,
                        res_y=self.res_y,
                        focal=self.focal,
                        start_frame=self.start_frame,
                        duration=self.duration,
                        style=render_style,
                        samples=self.samples,
                        use_gpu=self.use_gpu,
                        version=self.version
                    )
                    if renderer.output_path.exists():
                        self.logger.info(f"Video file already rendered, skip: {renderer.output_path}")
                    else:
                        if self.render:
                            renderer.execute()
                            self.logger.info(f"Successfully rendered current composition in {render_style} style, check: {renderer.output_path}")

                if self.retarget:
                    # only apply retargeting for existing composed blend files with rendered videos
                    if composer.output_path.exists() and renderer.output_path.exists():
                        annotator = AnimerMarkerAnnotator(
                            input_path=self.global_path_tracer[composer.output_path.name]["pmx_path"],
                            validate=self.ma_validate,
                            visualize=self.ma_visualize,
                            marker_size=self.ma_marker_size,
                            version=self.version
                        )
                        if annotator.output_path.exists():
                            self.logger.info(f"Marker already annotated, skip: {annotator.output_path}")
                        else:
                            annotator.execute()
                            self.logger.info(f"Successfully annotated mmd character marker, check: {annotator.output_path}")

                        shape_fitter = AnimerShapeFitter(
                            input_path=self.global_path_tracer[composer.output_path.name]["pmx_path"],
                            standard_height=self.sf_standard_height,
                            learning_rate=self.sf_learning_rate,
                            max_iter=self.sf_max_iter,
                            w_shape_reg=self.sf_w_shape_reg,
                            w_pose_reg=self.sf_w_pose_reg,
                            visualize=self.sf_visualize,
                            version=self.version
                        )
                        if shape_fitter.output_path.exists():
                            self.logger.info(f"Shape fitting already done, skip: {shape_fitter.output_path}")
                        else:
                            shape_fitter.execute()
                            self.logger.info(f"Successfully fit smpl shape parameter to mmd character, check: {shape_fitter.output_path}")

                        retargeter = AnimerRetargeter(
                            input_path=composer.output_path,
                            shape_path=shape_fitter.output_path,
                            batch_size=self.pf_batch_size,
                            num_iters=self.pf_num_iters,
                            use_collision=self.pf_use_collision,
                            version=self.version
                        )
                        if retargeter.output_path.exists():
                            self.logger.info(f"Pose retargeting already done, skip: {retargeter.output_path}")
                        else:
                            retargeter.execute()
                            self.logger.info(f"Successfully retarget mmd animation to smpl, check: {retargeter.output_path}")

                        if self.pf_visualize:
                            annotation = np.load(str(retargeter.output_path), allow_pickle=True).item()
                            self._visualize(annotation)
                            exit()

            except Exception as e:
                self.logger.error(f"Failed to process current composition: {str(e)}")
                exit()  # for debug
        self.logger.info(f"Workflow execution completed for rank {self.local_rank}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    
    parser.add_argument("--bake", action="store_true", help="if not for debug, always add --bake")
    parser.add_argument("--no_physics", action="store_true", help="disable physics simulation during baking")
    parser.add_argument("--disable_ik", action="store_true")
    
    parser.add_argument("--render", action="store_true", help="if not --render, the following settings will be discarded")
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--res_x", type=int, default=1280)
    parser.add_argument("--res_y", type=int, default=720)
    parser.add_argument("--focal", type=int, default=50)
    parser.add_argument("--start_frame", type=int, default=60, help="start from 60th-frame to discard the first 2 seconds morph")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--style", type=str, nargs="+")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--retarget", action="store_true")
    parser.add_argument("--ma_validate", action="store_true")
    parser.add_argument("--ma_visualize", action="store_true")
    parser.add_argument("--ma_marker_size", type=float, default=0.01)

    parser.add_argument("--sf_standard_height", type=float, default=1.7)
    parser.add_argument("--sf_learning_rate", type=float, default=0.1)
    parser.add_argument("--sf_max_iter", type=int, default=300)
    parser.add_argument("--sf_w_shape_reg", type=float, default=5e-4)
    parser.add_argument("--sf_w_pose_reg", type=float, default=1e-3)
    parser.add_argument("--sf_visualize", action="store_true")

    # parser.add_argument("--standard_height", type=float, default=1.6)
    parser.add_argument("--pf_batch_size", type=int, default=1)
    parser.add_argument("--pf_num_iters", type=int, default=100)
    parser.add_argument("--pf_use_collision", action="store_true")
    parser.add_argument("--pf_visualize", action="store_true")

    parser.add_argument("--version", type=str, default="")
    args = parser.parse_args()
    
    workflow = AnimerWorkflow(
        global_size=args.global_size,
        local_rank=args.local_rank,
        reverse=args.reverse,
        random_seed=args.random_seed,

        bake=args.bake,
        no_physics=args.no_physics,
        disable_ik=args.disable_ik,
        
        render=args.render,
        engine=args.engine,
        res_x=args.res_x,
        res_y=args.res_y,
        focal=args.focal,
        start_frame=args.start_frame,
        duration=args.duration,
        style=args.style,
        samples=args.samples,
        use_gpu=args.use_gpu,

        retarget=args.retarget,
        ma_validate=args.ma_validate,
        ma_visualize=args.ma_visualize,
        ma_marker_size=args.ma_marker_size,

        sf_standard_height=args.sf_standard_height,
        sf_learning_rate=args.sf_learning_rate,
        sf_max_iter=args.sf_max_iter,
        sf_w_shape_reg=args.sf_w_shape_reg,
        sf_w_pose_reg=args.sf_w_pose_reg,
        sf_visualize=args.sf_visualize,

        pf_batch_size=args.pf_batch_size,
        pf_num_iters=args.pf_num_iters,
        pf_use_collision=args.pf_use_collision,
        pf_visualize=args.pf_visualize,

        version=args.version
    )
    workflow.execute()

if __name__ == "__main__":
    main()