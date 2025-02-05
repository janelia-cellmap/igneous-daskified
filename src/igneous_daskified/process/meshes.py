import fast_simplification
from igneous_daskified.process.analyze import AnalyzeMeshes

# import pymeshlab

from funlib.persistence import open_ds
from funlib.persistence.arrays.datasets import _read_attrs
from funlib.geometry import Roi
import numpy as np
import os
import logging
from funlib.geometry import Roi
from zmesh import Mesher
from zmesh import Mesh as Zmesh
import pandas as pd
from igneous_daskified.util import dask_util, io_util
import dask.bag as db
from cloudvolume.mesh import Mesh as CloudVolumeMesh
import dask.dataframe as dd
import pyvista as pv
import pymeshfix
import shutil
from igneous_daskified.process.downsample_numba import (
    downsample_labels_3d_suppress_zero,
)
import pymeshlab
import trimesh
import json
from funlib.geometry import Coordinate

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Meshify:
    """Get a meshes from a zarr or n5 segmentation array"""

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        total_roi: Roi = None,
        max_num_voxels=np.inf,  # 50 * 512**3,  # 500 blocks times 512*512*512
        max_num_blocks=20_000,
        read_write_block_shape_pixels: list = None,
        downsample_factor: int | None = None,
        target_reduction: float = 0.99,
        num_workers: int = 10,
        remove_smallest_components: bool = True,
        n_smoothing_iter: int = 10,
        default_aggressiveness: int = 7,
        check_mesh_validity: bool = True,  # useful if meshes will be used for things other than visualization
        do_simplification: bool = True,
        do_analysis: bool = True,
        do_legacy_neuroglancer=False,
    ):

        for file_type in [".n5", ".zarr"]:
            if file_type in input_path:
                path_split = input_path.split(file_type + "/")
                break

        self.segmentation_array = open_ds(path_split[0] + file_type, path_split[1])
        self.output_directory = output_directory

        # NOTE: Currently true voxel size only works with zarr in certain order...funlib persistence forces voxel size to be integer otherwise:
        # NOTE: Funlib persistence does not support non-integer voxel sizes
        self.s, _, _ = _read_attrs(self.segmentation_array.data)
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
        self.output_voxel_size_funlib = self.base_voxel_size_funlib

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
        self.do_simplification = do_simplification
        self.default_aggressiveness = default_aggressiveness

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
        segmentation_block = self.segmentation_array.to_ndarray(block.roi)
        if segmentation_block.dtype.byteorder == ">":
            segmentation_block = segmentation_block.newbyteorder().byteswap()
        if self.downsample_factor:
            segmentation_block, _ = downsample_labels_3d_suppress_zero(
                segmentation_block, self.downsample_factor
            )
        # initial marching cubes pass
        # close controls whether meshes touching
        # the image boundary are left open or closed
        block_offset = np.array(block.roi.get_begin())

        mesher.mesh(segmentation_block, close=False)
        for id in mesher.ids():
            mesh = mesher.get_mesh(id)
            mesh.vertices += block_offset[::-1]
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
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
            self.read_write_block_shape_pixels,
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
        aggressiveness=7,
        do_simplification=True,
        check_mesh_validity=True,
    ):

        def get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_count, aggressiveness, do_simplification
        ):
            if do_simplification:
                # logger.warning("simplifying mesh")
                simplified_mesh = fast_simplification.simplify_mesh(
                    mesh, target_count=target_count, agg=aggressiveness, verbose=True
                )
                # logger.warning("simplified")
            else:
                simplified_mesh = mesh

            if n_smoothing_iter > 0:
                simplified_mesh = simplified_mesh.smooth_taubin(n_smoothing_iter)
            # logger.warning("smoothed")
            # clean mesh, this helps ensure watertightness, but not guaranteed?
            # if remove_smallest_components:
            if not check_mesh_validity:
                return (
                    simplified_mesh.points,
                    simplified_mesh.faces.reshape(-1, 4)[:, 1:],
                )
            vclean, fclean = pymeshfix.clean_from_arrays(
                simplified_mesh.points,
                simplified_mesh.faces.reshape(-1, 4)[:, 1:],
                remove_smallest_components=remove_smallest_components,
                verbose=True,
            )
            # logger.warning("cleaned")
            # else:
            #     return (
            #         simplified_mesh.points,
            #         simplified_mesh.faces.reshape(-1, 4)[:, 1:],
            #     )

            return vclean, fclean

        if remove_smallest_components:
            components = mesh.split()

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
        # print(f"Initial mesh has {len(faces)} faces")
        output_trimesh_mesh = trimesh.Trimesh(vertices, faces)
        # print(f"output mesh")
        if check_mesh_validity and not Meshify.is_mesh_valid(output_trimesh_mesh):
            raise Exception(
                f"Initial mesh is not valid, {output_trimesh_mesh.is_winding_consistent=},{output_trimesh_mesh.is_watertight=},{output_trimesh_mesh.volume=}."
            )

        # Create a new column filled with 3s since we need to tell pyvista that each face has 3 vertices in this mesh
        new_column = np.full((mesh.faces.shape[0], 1), 3)

        # Concatenate the new column with the original array
        faces = np.concatenate((new_column, mesh.faces), axis=1)
        mesh = pv.PolyData(vertices, faces[:])
        # print("polydataed")

        min_faces = 100
        # simplify mesh
        target_count = max(int(mesh.n_faces * (1 - target_reduction)), min_faces)
        if do_simplification:
            # check to make sure it is actually necessary
            do_simplification = mesh.n_faces > min_faces
        vclean, fclean = get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_count, aggressiveness, do_simplification
        )
        # logger.warning(f"got vclean and fclean: len(fclean)={len(fclean)}")
        trimesh_mesh = trimesh.Trimesh(vertices=vclean, faces=fclean)
        if check_mesh_validity:
            if Meshify.is_mesh_valid(trimesh_mesh):
                output_trimesh_mesh = trimesh_mesh
        else:
            output_trimesh_mesh = trimesh_mesh
        # logger.warning(f"trimeshed")

        aggressiveness -= 1
        # initially would redo only if fclean==0, but noticed that sometimes it simplifies weird structures to a single face which isnt good
        # also need to check if mesh is valid after simplification
        while (
            len(fclean) < 0.5 * target_count
            and aggressiveness >= -1
            and do_simplification
        ):
            # if aggressiveness is <0, then we want to try without simplification
            vclean, fclean = get_cleaned_simplified_and_smoothed_mesh(
                mesh,
                target_count,
                aggressiveness,
                do_simplification=aggressiveness >= 0,
            )
            aggressiveness -= 1
            trimesh_mesh = trimesh.Trimesh(vertices=vclean, faces=fclean)
            if check_mesh_validity:
                if Meshify.is_mesh_valid(trimesh_mesh):
                    output_trimesh_mesh = trimesh_mesh
            else:
                output_trimesh_mesh = trimesh_mesh

        if do_simplification and aggressiveness == -2:
            logger.warning(
                f"Mesh with {mesh.n_faces} faces (min_faces={min_faces}) had to be processed unsimplified."
            )

        if len(fclean) == 0:
            raise Exception(
                f"Mesh with {mesh.n_faces} faces (min_faces={min_faces}) could not be smoothed and cleaned even without simplificaiton."
            )

        output_trimesh_mesh.vertices += com
        output_trimesh_mesh.fix_normals()
        return output_trimesh_mesh

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

        mesh_files = os.listdir(f"{self.dirname}/{mesh_id}")
        if len(mesh_files) >= self.max_num_blocks:
            logger.warning(
                f"Mesh {mesh_id} has too many blocks {len(mesh_files)})>{self.max_num_blocks}. Skipping."
            )
            store_skipped_info(self, mesh_id)

            shutil.rmtree(f"{self.dirname}/{mesh_id}")
            return

        for mesh_file in mesh_files:
            with open(f"{self.dirname}/{mesh_id}/{mesh_file}", "rb") as f:
                mesh = Zmesh.from_ply(f.read())
                block_meshes.append(mesh)

        if len(block_meshes) > 1:
            try:
                mesh = Meshify.my_cloudvolume_concatenate(*block_meshes)
            except Exception as e:
                raise Exception(f"{mesh_id} failed, with error: {e}")

            # doing both of the following may be redundant
            mesh = mesh.consolidate()  # if you don't have self-contacts
            chunk_size = (
                self.read_write_block_shape_pixels * self.base_voxel_size_funlib
            )
            mesh = mesh.deduplicate_chunk_boundaries(
                chunk_size=chunk_size[::-1],
                offset=self.total_roi.offset[::-1],
            )

        if self.check_mesh_validity:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

            # further cleanup attempts
            vclean, fclean = pymeshfix.clean_from_arrays(
                mesh.vertices,
                mesh.faces,
                remove_smallest_components=self.remove_smallest_components,
            )

            mesh = trimesh.Trimesh(vclean, fclean)

        # simplify and smooth the final mesh
        try:
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

        if not self.do_legacy_neuroglancer:
            _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
        else:
            io_util.write_ngmesh(
                mesh.vertices,
                mesh.faces,
                f"{self.output_directory}/meshes/{mesh_id}",
            )
            with open(f"{self.output_directory}/meshes/{mesh_id}:0", "w") as f:
                f.write(json.dumps({"fragments": [f"./{mesh_id}"]}))
        shutil.rmtree(f"{self.dirname}/{mesh_id}")

    def assemble_meshes(self, dirname):
        os.makedirs(f"{self.output_directory}/meshes/", exist_ok=True)
        self.dirname = dirname
        mesh_ids = os.listdir(dirname)
        b = db.from_sequence(
            mesh_ids, npartitions=min(self.num_workers * 10, len(mesh_ids))
        ).map(self._assemble_mesh)
        with dask_util.start_dask(self.num_workers, "assemble meshes", logger):
            with io_util.Timing_Messager("Assembling meshes", logger):
                b.compute()
        if self.do_legacy_neuroglancer:
            io_util.write_ngmesh_metadata(f"{self.output_directory}/meshes")
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
            analyze = AnalyzeMeshes(
                self.output_directory + "/meshes", self.output_directory + "/metrics"
            )
            analyze.analyze()
