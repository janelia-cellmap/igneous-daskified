# %%
from funlib.persistence.arrays.datasets import _read_attrs, open_ds
from funlib.geometry import Roi, Coordinate
import numpy as np
import os
import logging
from zmesh import Mesher
from zmesh import Mesh as Zmesh
import pandas as pd
from igneous_daskified.util import dask_util, io_util
import dask.bag as db
from cloudvolume.mesh import Mesh as CloudVolumeMesh
import shutil
from igneous_daskified.process.downsample_numba import (
    downsample_labels_3d_suppress_zero,
)
import trimesh
import json
import pymeshlab

# import igneous_daskified.process.fixed_edge_meshes
# from importlib import reload

# reload(igneous_daskified.process.fixed_edge_meshes)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import fixed edge mesh utilities
try:
    from igneous_daskified.process.fixed_edge_meshes import (
        simplify_mesh,
    )

    FIXED_EDGE_AVAILABLE = True
except ImportError as e:
    FIXED_EDGE_AVAILABLE = False
    logger.warning(
        f"Fixed edge mesh utilities not available: {e}. Fixed edge simplification will not work."
    )


def staged_reductions(target_reduction_total, frac1, frac2):
    """
    Compute per-stage reductions (r1, r2) given how much of the total reduction
    each stage should contribute.

    target_reduction_total: overall target reduction (e.g. 0.99)
    frac1, frac2: relative fractions of total simplification (e.g. 0.25, 0.75)
    """
    assert abs(frac1 + frac2 - 1.0) < 1e-6, "fractions must sum to 1"
    keep_total = 1 - target_reduction_total

    r1 = 1 - keep_total**frac1
    r2 = 1 - keep_total**frac2
    return r1, r2


class Meshify:
    """Class to create meshes from a segmentation volume using dask and zmesh."""

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        total_roi: Roi = None,
        max_num_voxels=np.inf,  # 50 * 512**3,  # 500 blocks times 512*512*512
        max_num_blocks=np.inf,  # 200_000,
        read_write_block_shape_pixels: list = None,
        downsample_factor: int | None = None,
        target_reduction: float = 0.99,
        num_workers: int = 10,
        remove_smallest_components: bool = True,
        n_smoothing_iter: int = 10,
        default_aggressiveness: int = 0.3,
        check_mesh_validity: bool = True,  # useful if meshes will be used for things other than visualization
        do_simplification: bool = True,
        do_analysis: bool = True,
        do_legacy_neuroglancer=False,
        do_singleres_multires_neuroglancer=False,
        use_fixed_edge_simplification: bool = False,
        fixed_edge_merge_weld_epsilon: float = 1e-4,
        fixed_edge_seam_angle_deg: float = 35.0,
        fixed_edge_k_ring: int = 2,
        fixed_edge_taubin_iters: int = 12,
        fixed_edge_taubin_lambda: float = 0.5,
        fixed_edge_taubin_mu: float = -0.53,
        stage_1_reduction_fraction: float = 0.5,
    ):
        """
        Args:
            input_path (str): Path to the input segmentation volume.
            output_directory (str): Path to the output directory.
            total_roi (Roi): The total ROI to process. If None, the entire volume will be processed.
            max_num_voxels (int): Maximum number of voxels to process. Default is np.inf.
            max_num_blocks (int): Maximum number of blocks to process. Default is 20_000.
            read_write_block_shape_pixels (list): Shape of the blocks to read/write. If None, will use the chunk shape of the input volume.
            downsample_factor (int | None): Factor to downsample the input volume. If None, no downsampling will be applied.
            target_reduction (float): Target reduction factor for mesh simplification. Target faces will be (1-target_reduction)*num_faces Default is 0.99.
            num_workers (int): Number of workers to use for processing. Default is 10.
            remove_smallest_components (bool): Whether to remove the smallest components from the mesh. Default is True.
            n_smoothing_iter (int): Number of smoothing iterations to apply to the mesh. Default is 10.
            default_aggressiveness (int): Default aggressiveness for mesh simplification. Default is .3.
            check_mesh_validity (bool): Whether to check the validity of the mesh. This is useful most useful if doing downstream analysis. Default is True.
            do_simplification (bool): Whether to apply mesh simplification. Default is True.
            do_analysis (bool): Whether to perform analysis on the meshes. Default is True.
            do_legacy_neuroglancer (bool): Whether to create legacy neuroglancer files. Default is False.
            do_singleres_multires_neuroglancer (bool): Whether to create single resolution multi-resolution neuroglancer files. Default is False.
            use_fixed_edge_simplification (bool): Whether to use edge-constrained simplification during block processing and seam cleanup during merging. Default is False.
            fixed_edge_merge_weld_epsilon (float): Vertex welding tolerance when merging blocks with fixed edge approach. Default is 1e-4.
            fixed_edge_seam_angle_deg (float): Dihedral angle threshold (degrees) for seam detection with fixed edge approach. Default is 35.0.
            fixed_edge_k_ring (int): K-ring expansion for seam smoothing band with fixed edge approach. Default is 2.
            fixed_edge_taubin_iters (int): Taubin smoothing iterations for seam denoising with fixed edge approach. Default is 12.
            fixed_edge_taubin_lambda (float): Taubin smoothing lambda parameter for fixed edge approach. Default is 0.5.
            fixed_edge_taubin_mu (float): Taubin smoothing mu parameter for fixed edge approach. Default is -0.53.
            stage_1_reduction_fraction (float): Fraction of total reduction to perform in stage 1 of staged simplification. Default is 0.5.
        """

        for file_type in [".n5", ".zarr"]:
            if file_type in input_path:
                path_split = input_path.split(file_type + "/")
                break

        self.segmentation_array = open_ds(path_split[0] + file_type, path_split[1])
        self.output_directory = output_directory

        # NOTE: Currently true voxel size only works with zarr in certain order...funlib persistence forces voxel size to be integer otherwise:
        # NOTE: Funlib persistence does not support non-integer voxel sizes
        self.true_voxel_size, _, _ = _read_attrs(self.segmentation_array.data)
        self.true_voxel_size = np.array(self.true_voxel_size)
        if total_roi:
            self.total_roi = total_roi
        else:
            self.total_roi = self.segmentation_array.roi
        self.num_workers = num_workers

        if read_write_block_shape_pixels:
            self.read_write_block_shape_pixels = np.array(read_write_block_shape_pixels)
        else:
            self.read_write_block_shape_pixels = np.array(
                self.segmentation_array.chunk_shape
            )

        self.max_num_blocks = max_num_blocks  # np.prod(read_write_block_shape_pixels)
        self.base_voxel_size_funlib = self.segmentation_array.voxel_size

        # keep track of funlib voxel size, but make sure it as at least 1, which won't be the case if <1nm
        self.output_voxel_size_funlib = max(
            self.base_voxel_size_funlib, Coordinate(1, 1, 1)
        )
        self.downsample_factor = downsample_factor
        if self.downsample_factor:
            self.output_voxel_size_funlib = Coordinate(
                np.array(self.output_voxel_size_funlib) * self.downsample_factor
            )
            self.true_voxel_size *= self.downsample_factor
        self.target_reduction = target_reduction

        self.check_mesh_validity = check_mesh_validity
        self.remove_smallest_components = remove_smallest_components
        self.n_smoothing_iter = n_smoothing_iter
        self.do_analysis = do_analysis
        self.do_legacy_neuroglancer = do_legacy_neuroglancer
        self.do_singleres_multires_neuroglancer = do_singleres_multires_neuroglancer
        self.do_simplification = do_simplification
        self.default_aggressiveness = default_aggressiveness

        # Fixed edge simplification parameters
        self.use_fixed_edge_simplification = use_fixed_edge_simplification
        if self.use_fixed_edge_simplification and not FIXED_EDGE_AVAILABLE:
            raise RuntimeError(
                "Fixed edge simplification requested but dependencies not available. "
                "Ensure fixed_edge_meshes.py is present and pyfqmr is installed (`pip install pyfqmr`)."
            )

        self.fixed_edge_merge_weld_epsilon = fixed_edge_merge_weld_epsilon
        self.fixed_edge_seam_angle_deg = fixed_edge_seam_angle_deg
        self.fixed_edge_k_ring = fixed_edge_k_ring
        self.fixed_edge_taubin_iters = fixed_edge_taubin_iters
        self.fixed_edge_taubin_lambda = fixed_edge_taubin_lambda
        self.fixed_edge_taubin_mu = fixed_edge_taubin_mu

        self.stage_1_reduction_fraction = stage_1_reduction_fraction
        self.stage_2_reduction_fraction = 1 - self.stage_1_reduction_fraction

    @staticmethod
    def my_cloudvolume_concatenate(*meshes):
        # The default cloudvolume concatenate requires normals, which we do not
        vertex_ct = np.zeros(len(meshes) + 1, np.uint32)
        vertex_ct[1:] = np.cumsum([len(mesh) for mesh in meshes])

        vertices = np.concatenate([mesh.vertices for mesh in meshes])

        faces = np.concatenate(
            [mesh.faces + vertex_ct[i] for i, mesh in enumerate(meshes)]
        )

        # normals = np.concatenate([ mesh.normals for mesh in meshes ])
        normals = None

        return CloudVolumeMesh(vertices, faces, normals)

    def _get_chunked_mesh(self, block, tmpdirname):
        mesher = Mesher(self.output_voxel_size_funlib[::-1])  # anisotropy of image
        segmentation_block = self.segmentation_array.to_ndarray(block.roi, fill_value=0)
        if segmentation_block.dtype.byteorder == ">":
            # get the opposite-endian dtype
            swapped_dtype = segmentation_block.dtype.newbyteorder()
            # reinterpret the data as that dtype and swap the bytes
            segmentation_block = segmentation_block.view(swapped_dtype).byteswap()
        if self.downsample_factor:
            segmentation_block, _ = downsample_labels_3d_suppress_zero(
                segmentation_block, self.downsample_factor
            )
        # initial marching cubes pass
        # close controls whether meshes touching
        # the image boundary are left open or closed
        block_offset = np.array(block.roi.get_begin())
        # write out segmentation block as numpy array
        mesher.mesh(segmentation_block, close=False)
        for id in mesher.ids():
            mesh = mesher.get_mesh(id)
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)

            # If using fixed edge simplification, simplify per-block with border preservation
            if self.use_fixed_edge_simplification and self.do_simplification:
                # try:
                # Convert to trimesh
                mesh_tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                # Calculate target reduction based on target_reduction
                stage_1_reduction, _ = staged_reductions(
                    self.target_reduction,
                    self.stage_1_reduction_fraction,
                    self.stage_2_reduction_fraction,
                )

                # Calculate block size and offset for boundary removal
                # Block size in voxels (without halo)
                block_size_voxels = (
                    self.read_write_block_shape_pixels + 1
                )  # starts at 0
                # ROI offset for this block (in world coordinates, z,y,x order to match vertices)
                # Block size in world coordinates (z,y,x order)
                block_size_world = (block_size_voxels * self.output_voxel_size_funlib)[
                    ::-1
                ]

                # Simplify with positive-face boundary removal
                mesh_tri_simplified = simplify_mesh(
                    mesh_tri,
                    voxel_size=self.output_voxel_size_funlib,
                    target_reduction=stage_1_reduction,
                    block_size=block_size_world,
                    aggressiveness=self.default_aggressiveness,
                    verbose=False,
                    fix_edges=True,
                )
                mesh_tri_simplified.vertices += block_offset[::-1]

                # Create a new CloudVolumeMesh with simplified data and write using CloudVolumeMesh format
                # This ensures compatibility with CloudVolumeMesh.from_ply() during assembly
                mesh_simplified = CloudVolumeMesh(
                    mesh_tri_simplified.vertices,
                    mesh_tri_simplified.faces,
                    normals=None,
                )

                with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
                    fp.write(mesh_simplified.to_ply())
                # except Exception as e:
                #     logger.warning(
                #         f"Fixed edge simplification failed for block {block.index}, id {id}: {e}. Using unsimplified mesh."
                #     )
                #     # Fall back to original mesh
                #     with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
                #         fp.write(mesh.to_ply())
            else:
                mesh.vertices += block_offset[::-1]
                with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
                    fp.write(mesh.to_ply())

    @staticmethod
    def is_mesh_valid(mesh):
        valid_volume = (
            mesh.is_winding_consistent and mesh.is_watertight and mesh.volume > 0
        )
        return valid_volume

    def get_chunked_meshes(self, dirname):
        blocks = dask_util.create_blocks(
            self.total_roi,
            self.segmentation_array,
            self.read_write_block_shape_pixels.copy(),
            padding=self.output_voxel_size_funlib,
        )

        b = db.from_sequence(blocks, npartitions=self.num_workers * 10).map(
            self._get_chunked_mesh, dirname
        )

        with dask_util.start_dask(self.num_workers, "generate chunked meshes", logger):
            with io_util.Timing_Messager("Generating chunked meshes", logger):
                b.compute()

    @staticmethod
    def simplify_and_smooth_mesh(
        mesh,
        target_reduction=0.99,
        n_smoothing_iter=10,
        remove_smallest_components=True,
        aggressiveness=0.3,
        do_simplification=True,
        check_mesh_validity=True,
    ):

        def get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_reduction, aggressiveness, do_simplification
        ):
            if do_simplification:
                # logger.warning("simplifying mesh")
                simplified_mesh = simplify_mesh(
                    mesh,
                    voxel_size=None,
                    target_reduction=target_reduction,
                    aggressiveness=aggressiveness,
                    verbose=False,
                    fix_edges=False,
                )

                # do taubin smoothing inside simplify_mesh now

                # simplified_mesh = fast_simplification.simplify_mesh(
                #     mesh, target_count=target_count, agg=aggressiveness, verbose=False
                # )
                # logger.warning("simplified")
            else:
                simplified_mesh = mesh
            del mesh

            if n_smoothing_iter > 0:
                ms = pymeshlab.MeshSet()
                ms.add_mesh(
                    pymeshlab.Mesh(
                        vertex_matrix=simplified_mesh.vertices,
                        face_matrix=simplified_mesh.faces,
                    )
                )

                # Apply Taubin smoothing
                ms.apply_coord_taubin_smoothing(
                    lambda_=0.5,  # Smoothing factor
                    mu=-0.53,  # Inflation factor
                    stepsmoothnum=n_smoothing_iter,  # Number of iterations
                )
                m = ms.current_mesh()
                simplified_mesh = trimesh.Trimesh(
                    vertices=m.vertex_matrix(), faces=m.face_matrix()
                )
            # logger.warning("smoothed")
            # clean mesh, this helps ensure watertightness, but not guaranteed?
            # if remove_smallest_components:
            if not check_mesh_validity:
                return simplified_mesh
            cleaned_mesh = Meshify.repair_mesh_pymeshlab(
                simplified_mesh.vertices,
                simplified_mesh.faces,
                remove_smallest_components=remove_smallest_components,
            )
            del simplified_mesh

            return cleaned_mesh

        if remove_smallest_components:
            if type(mesh) != trimesh.base.Trimesh:
                mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

            components = mesh.split(only_watertight=check_mesh_validity)

            if len(components) > 0:
                # this will equal zero if not watertight?
                mesh = components[0]
                for m in components[1:]:
                    if len(m.faces) > len(mesh.faces):
                        # print(f"{len(m.faces)},{len(mesh.faces)}")
                        mesh = m

        com = mesh.vertices.mean(axis=0)
        # things seem better if center first otherwise for example taubin smoothing shifts things

        vertices = mesh.vertices - com
        faces = mesh.faces
        mesh = trimesh.Trimesh(vertices, faces)

        # print(f"Initial mesh has {len(faces)} faces")
        output_trimesh_mesh = mesh.copy()
        # print(f"output mesh")
        if check_mesh_validity and not Meshify.is_mesh_valid(output_trimesh_mesh):
            # write out failed mesh
            output_trimesh_mesh.export("failed_mesh.ply")
            raise Exception(
                f"Initial mesh is not valid, {output_trimesh_mesh.is_winding_consistent=},{output_trimesh_mesh.is_watertight=},{output_trimesh_mesh.volume=}."
            )

        if do_simplification:
            # check to make sure it is actually necessary
            do_simplification = output_trimesh_mesh.faces.shape[0] > 100
        trimesh_mesh = get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_reduction, aggressiveness, do_simplification
        )
        # logger.warning(f"got vclean and fclean: len(fclean)={len(fclean)}")
        min_faces = 100
        retry_simplification_for_validity = False
        if check_mesh_validity:
            if Meshify.is_mesh_valid(trimesh_mesh):
                output_trimesh_mesh = trimesh_mesh
            else:
                retry_simplification_for_validity = True
        else:
            output_trimesh_mesh = trimesh_mesh
        # logger.warning(f"trimeshed")

        aggressiveness -= 0.05
        # initially would redo only if fclean==0, but noticed that sometimes it simplifies weird structures to a single face which isnt good
        # also need to check if mesh is valid after simplification
        while (
            (
                len(output_trimesh_mesh.faces)
                < 0.5 * len(trimesh_mesh.faces) * (1 - target_reduction)
                or retry_simplification_for_validity
            )
            and aggressiveness >= -0.05
            and do_simplification
        ):
            print("aggressiveness:", aggressiveness)
            # if aggressiveness is <0, then we want to try without simplification
            trimesh_mesh = get_cleaned_simplified_and_smoothed_mesh(
                mesh,
                target_reduction,
                aggressiveness,
                do_simplification=aggressiveness >= 0,
            )
            aggressiveness -= 0.05
            if check_mesh_validity:
                if Meshify.is_mesh_valid(trimesh_mesh):
                    output_trimesh_mesh = trimesh_mesh
                    retry_simplification_for_validity = False
            else:
                output_trimesh_mesh = trimesh_mesh

        if do_simplification and aggressiveness < -0.05:
            logger.warning(
                f"Mesh with {len(output_trimesh_mesh.faces)} faces (min_faces={min_faces}) had to be processed unsimplified."
            )

        if len(output_trimesh_mesh.faces) == 0:
            raise Exception(
                f"Mesh with {len(output_trimesh_mesh.faces)} faces (min_faces={min_faces}) could not be smoothed and cleaned even without simplification."
            )

        output_trimesh_mesh.vertices += com
        output_trimesh_mesh.fix_normals()
        return output_trimesh_mesh

    @staticmethod
    def repair_mesh_pymeshlab(
        vertices,
        faces,
        remove_smallest_components=True,
        max_hole_size=30,
        verbose=False,
    ):
        """
        Repair mesh using PyMeshLab and return as trimesh.

        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices
            max_hole_size: Maximum hole size to fill (in number of edges), default 30

        Returns:
            tuple: (vertices, faces) - repaired or original if skipped
        """
        import pymeshlab
        import trimesh
        import time
        from datetime import datetime

        def print_with_time(msg, verbose=verbose):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if verbose:
                print(f"[{timestamp}] {msg}")

        t_start = time.time()
        print_with_time(
            f"Starting mesh repair with pymeshlab ({len(faces):,} faces)..."
        )

        # Create MeshSet and add mesh
        t0 = time.time()
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
        print_with_time(f"Added mesh to pymeshlab ({time.time() - t0:.2f}s)")

        # Repair pipeline
        t0 = time.time()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()
        print_with_time(
            f"Removed duplicates and unreferenced vertices ({time.time() - t0:.2f}s)"
        )

        # Repair non-manifold edges (split first, then remove)
        t0 = time.time()
        ms.meshing_repair_non_manifold_edges(method="Split Vertices")
        ms.meshing_repair_non_manifold_edges(method="Remove Faces")
        print_with_time(f"Repaired non-manifold edges ({time.time() - t0:.2f}s)")

        # Close holes
        t0 = time.time()
        ms.meshing_close_holes(maxholesize=max_hole_size)
        print_with_time(
            f"Closed holes (max size {max_hole_size}) ({time.time() - t0:.2f}s)"
        )

        # Repair non-manifold vertices
        t0 = time.time()
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        print_with_time(f"Repaired non-manifold vertices ({time.time() - t0:.2f}s)")

        # Keep only largest component - API changed between pymeshlab versions
        t0 = time.time()
        n_components_before = ms.current_mesh().face_number()
        if remove_smallest_components:
            try:
                # Try new API (>= 2022.2) with PercentageValue
                if hasattr(pymeshlab, "PercentageValue"):
                    ms.meshing_remove_connected_component_by_diameter(
                        mincomponentdiag=pymeshlab.PercentageValue(0)
                    )
                elif hasattr(pymeshlab, "Percentage"):
                    ms.meshing_remove_connected_component_by_diameter(
                        mincomponentdiag=pymeshlab.Percentage(0)
                    )
                else:
                    # Old API - plain number
                    ms.meshing_remove_connected_component_by_diameter(
                        mincomponentdiag=0
                    )
            except Exception as e:
                print_with_time(
                    f"Could not remove small components: {e}. Skipping component removal."
                )
            n_components_after = ms.current_mesh().face_number()
            print_with_time(
                f"Component filtering: {n_components_before:,} -> {n_components_after:,} faces ({time.time() - t0:.2f}s)"
            )

        # Fix face orientation
        t0 = time.time()
        ms.meshing_re_orient_faces_coherently()
        print_with_time(f"Re-oriented faces ({time.time() - t0:.2f}s)")

        # Convert back - extract from PyMeshLab's internal format
        t0 = time.time()
        m = ms.current_mesh()
        print_with_time(f"Got current_mesh reference ({time.time() - t0:.2f}s)")

        # Extract vertex and face matrices (this is the slow part for large meshes)
        t0 = time.time()
        verts_out = m.vertex_matrix()
        print_with_time(f"Extracted vertex matrix ({time.time() - t0:.2f}s)")

        t0 = time.time()
        faces_out = m.face_matrix()
        print_with_time(f"Extracted face matrix ({time.time() - t0:.2f}s)")

        print_with_time(f"Total repair time: {time.time() - t_start:.2f}s")
        return trimesh.Trimesh(vertices=verts_out, faces=faces_out)

    def _assemble_mesh(self, mesh_id):

        def store_skipped_info(self, mesh_id):
            skipped_path = f"{self.output_directory}/too_big_skipped"
            os.makedirs(skipped_path, exist_ok=True)
            f = open(f"{skipped_path}/{mesh_id}.txt", "a")
            f.write(
                f"Mesh {mesh_id} has too many blocks {len(mesh_files)}>{self.max_num_blocks}. Skipping.\n"
            )
            f.write(", ".join(mesh_files))
            f.close()

        block_meshes = []
        # check if directory exists. if it sucessfully made it through the rest of this function, it will remove the temporary blocked directory. but it may run the function multiple times:
        # https://stackoverflow.com/questions/44548706/dask-processes-tasks-twice. So we need to check if the directory exists before running the function even if it existed before calling it
        if not os.path.exists(f"{self.dirname}/{mesh_id}"):
            return

        mesh_files = [
            f for f in os.listdir(f"{self.dirname}/{mesh_id}") if f.endswith(".ply")
        ]
        if len(mesh_files) >= self.max_num_blocks:
            logger.warning(
                f"Mesh {mesh_id} has too many blocks {len(mesh_files)})>{self.max_num_blocks}. Skipping."
            )
            store_skipped_info(self, mesh_id)

            shutil.rmtree(f"{self.dirname}/{mesh_id}")
            return
        print("loading meshes")
        for mesh_file in mesh_files:
            # if self.use_fixed_edge_simplification:
            #     # When using fixed edge simplification, PLY files are written by trimesh
            #     # Load with trimesh and convert to zmesh-compatible format
            #     mesh_tri = trimesh.load(f"{self.dirname}/{mesh_id}/{mesh_file}", process=False)
            #     # Convert to CloudVolumeMesh-like object that can be concatenated
            #     mesh = Zmesh(mesh_tri.vertices, mesh_tri.faces)
            #     block_meshes.append(mesh)
            # else:
            #     # Standard path: PLY files written by Zmesh
            with open(f"{self.dirname}/{mesh_id}/{mesh_file}", "rb") as f:
                mesh = Zmesh.from_ply(f.read())
                block_meshes.append(mesh)
        num_blocks = len(block_meshes)  # Store for later use

        if len(block_meshes) > 1:
            print(f"concatenating {len(block_meshes)} meshes")
            try:
                mesh = Meshify.my_cloudvolume_concatenate(*block_meshes)
            except Exception as e:
                raise Exception(f"{mesh_id} failed, with error: {e}")
            del block_meshes
            print("concatenated meshes")
            # doing both of the following may be redundant
            mesh = mesh.consolidate()  # if you don't have self-contacts
            chunk_size = (
                self.read_write_block_shape_pixels * self.base_voxel_size_funlib
            )
            print("deduplicating chunk boundaries")
            print(f"chunk size: {chunk_size}, offset: {self.total_roi.offset}")
            mesh = mesh.deduplicate_chunk_boundaries(
                chunk_size=chunk_size[::-1],
                offset=self.total_roi.offset[::-1],
            )
            print("deduplicated chunk boundaries")

            # # If using fixed edge approach, weld vertices and denoise seams after merging
            # if self.use_fixed_edge_simplification:
            #     try:
            #         # Convert to trimesh for processing
            #         mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

            #         # Calculate block boundaries for targeted vertex welding
            #         # Only weld vertices near block boundaries to save computation
            #         block_size = (
            #             self.read_write_block_shape_pixels * self.base_voxel_size_funlib
            #         )[::-1]  # Convert to z,y,x order

            #         # Weld vertices at block boundaries (only near boundaries)
            #         mesh = weld_vertices(
            #             mesh,
            #             epsilon=self.fixed_edge_merge_weld_epsilon,
            #             block_size=block_size,
            #             roi_offset=self.total_roi.offset[::-1],
            #             verbose=False
            #         )

            #         # Detect and smooth seams
            #         denoise_seams_inplace(
            #             mesh,
            #             seam_angle_deg=self.fixed_edge_seam_angle_deg,
            #             k_ring=self.fixed_edge_k_ring,
            #             taubin_iters=self.fixed_edge_taubin_iters,
            #             lamb=self.fixed_edge_taubin_lambda,
            #             mu=self.fixed_edge_taubin_mu,
            #             verbose=False,
            #         )
            #     except Exception as e:
            #         logger.warning(
            #             f"Fixed edge welding/denoising failed for mesh {mesh_id}: {e}. Continuing with standard processing."
            #         )
            #         # Convert back to CloudVolumeMesh format if it was trimesh
            #         if isinstance(mesh, trimesh.Trimesh):
            #             mesh = CloudVolumeMesh(
            #                 vertices=mesh.vertices, faces=mesh.faces, normals=None
            #             )

        if self.check_mesh_validity:
            try:
                vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
                faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
                del mesh
                # mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
                mesh = Meshify.repair_mesh_pymeshlab(
                    vertices,
                    faces,
                    remove_smallest_components=self.remove_smallest_components,
                )
            except Exception as e:
                raise Exception(f"{mesh_id} failed, with error: {e}")

        # simplify and smooth the final mesh
        try:
            if self.use_fixed_edge_simplification and self.do_simplification:
                stage_2_reduction, _ = staged_reductions(
                    self.target_reduction,
                    self.stage_1_reduction_fraction,
                    self.stage_2_reduction_fraction,
                )
                mesh = Meshify.simplify_and_smooth_mesh(
                    mesh,
                    stage_2_reduction,
                    self.n_smoothing_iter,
                    self.remove_smallest_components,
                    self.default_aggressiveness,
                    self.do_simplification,
                    self.check_mesh_validity,
                )
            else:
                mesh = Meshify.simplify_and_smooth_mesh(
                    mesh,
                    self.target_reduction,
                    self.n_smoothing_iter,
                    self.remove_smallest_components,
                    self.default_aggressiveness,
                    self.do_simplification,
                    self.check_mesh_validity,
                )

            if len(mesh.faces) == 0:
                _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
                raise Exception(f"Mesh {mesh_id} contains no faces")
        except Exception as e:
            raise Exception(f"{mesh_id} failed, with error: {e}")

        # correct for difference between funlib voxel sizes (which are rounded) and the actual voxel sizes
        if list(self.true_voxel_size) != list(self.output_voxel_size_funlib):
            mesh.vertices -= self.total_roi.offset[::-1]
            mesh.vertices *= np.array(self.true_voxel_size[::-1]) / np.array(
                self.output_voxel_size_funlib[::-1]
            )
            mesh.vertices += self.total_roi.offset[::-1]

        if self.do_legacy_neuroglancer:
            io_util.write_ngmesh(
                mesh.vertices,
                mesh.faces,
                f"{self.output_directory}/meshes/{mesh_id}",
            )
            with open(f"{self.output_directory}/meshes/{mesh_id}:0", "w") as f:
                f.write(json.dumps({"fragments": [f"./{mesh_id}"]}))
        elif self.do_singleres_multires_neuroglancer:
            io_util.write_singleres_multires_files(
                mesh.vertices, mesh.faces, f"{self.output_directory}/meshes/{mesh_id}"
            )
        else:
            _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
        shutil.rmtree(f"{self.dirname}/{mesh_id}")

    def assemble_meshes(self, dirname):
        os.makedirs(f"{self.output_directory}/meshes/", exist_ok=True)
        self.dirname = dirname
        mesh_ids = os.listdir(dirname)
        b = db.from_sequence(
            mesh_ids,
            npartitions=dask_util.guesstimate_npartitions(mesh_ids, self.num_workers),
        ).map(self._assemble_mesh)
        with dask_util.start_dask(self.num_workers, "assemble meshes", logger):
            with io_util.Timing_Messager("Assembling meshes", logger):
                b.compute()
        if self.do_legacy_neuroglancer:
            io_util.write_ngmesh_metadata(f"{self.output_directory}/meshes")
        elif self.do_singleres_multires_neuroglancer:
            io_util.write_singleres_multires_metadata(f"{self.output_directory}/meshes")
        shutil.rmtree(dirname)

    def assign_mitos_to_cells(self):
        cells_ds = open_ds(
            "/nrs/cellmap/ackermand/cellmap/jrc_mus-liver-zon-1.n5", "cells"
        )
        cells = cells_ds.to_ndarray()

        df = pd.read_csv(
            "/nrs/cellmap/ackermand/cellmap/analysisResults/jrc_mus-liver-zon-1/mito/postprocessed_mito_fixed_filled_volumeFiltered.csv"
        )
        self.mito_coms = df[["COM X (nm)", "COM Y (nm)", "COM Z (nm)"]].to_numpy()
        mito_coms_voxels = np.floor(self.mito_coms / cells_ds.voxel_size).astype(int)
        self.mito_ids = df["Object ID"].to_numpy()

        self.mito_to_cell_ids = cells[
            mito_coms_voxels[:, 2], mito_coms_voxels[:, 1], mito_coms_voxels[:, 0]
        ]

        all_cells = np.unique(cells)
        self.all_cells = all_cells[all_cells > 0].tolist()

    def get_meshes(self):
        """Get the meshes using dask save them to disk in a temporary directory"""
        os.makedirs(self.output_directory, exist_ok=True)
        # with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_chunked_dir = self.output_directory + "/tmp_chunked"
        os.makedirs(tmp_chunked_dir, exist_ok=True)
        self.get_chunked_meshes(tmp_chunked_dir)
        self.assemble_meshes(tmp_chunked_dir)

        if self.do_analysis:
            from igneous_daskified.process.analyze_meshes import AnalyzeMeshes

            analyze = AnalyzeMeshes(
                self.output_directory + "/meshes", self.output_directory + "/metrics"
            )
            analyze.analyze()


# # %%
# if __name__ == "__main__":
#     m = Meshify(
#         input_path="/nrs/cellmap/zubovy/symlinks_mito/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/s0",
#         output_directory="/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/c-elegans/jrc_c-elegans-bw-1/mito_highres_fixed_edge",
#         read_write_block_shape_pixels=[448, 448, 448],
#         do_analysis=False,
#         use_fixed_edge_simplification=True,
#     )
#     # %%
#     dirname = f"{m.output_directory}/tmp_chunked/"
#     os.makedirs(dirname, exist_ok=True)
#     m.dirname = dirname
#     id = "2761737217"

#     # m._assemble_mesh(id)

# # %%
# %%
# mesher = Mesher([16, 16, 16])  # anisotropy of image
# print("loading segmentation block")
# segmentation_block = np.load(
#     "/groups/cellmap/cellmap/ackermand/new_meshes/scripts/single_resolution/c-elegans/jrc_c-elegans-bw-1/seg_0_55-20251114.133642/50.npy"
# )
# print(np.unique(segmentation_block))
# mesher.mesh(segmentation_block, close=False)
# for id in mesher.ids():
#     print("get_mesh for id:", id)
#     mesh = mesher.get_mesh(id)

#     # try:
#     # Convert to trimesh
#     mesh_tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
#     print("got trimesh mesh")
#     # Simplify with positive-face boundary removal
#     mesh_tri_simplified = simplify_mesh(
#         mesh_tri,
#         voxel_size=[16, 16, 16],
#         target_reduction=0.9,
#         block_size=np.array([449, 449, 449]) * np.array([16, 16, 16]),
#         aggressiveness=0.3,
#         verbose=True,
#         fix_edges=True,
#     )
#     print("got simplified mesh")

#     # Create a new CloudVolumeMesh with simplified data and write using CloudVolumeMesh format
#     # This ensures compatibility with CloudVolumeMesh.from_ply() during assembly
#     mesh_simplified = CloudVolumeMesh(
#         mesh_tri_simplified.vertices,
#         mesh_tri_simplified.faces,
#         normals=None,
#     )

# %%

mesh = trimesh.load_mesh(
    "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/cgal_skeletonize_mesh/failed.ply"
)
output = Meshify.repair_mesh_pymeshlab(
    mesh.vertices,
    mesh.faces,
)
output.export(
    "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/cgal_skeletonize_mesh/failed_repaired.ply"
)

# %%
