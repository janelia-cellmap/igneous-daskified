from dataclasses import dataclass
import struct
import fast_simplification

# import pymeshlab

from funlib.persistence import Array, open_ds
from funlib.geometry import Roi
import numpy as np
import os
import logging
from funlib.geometry import Roi
from zmesh import Mesher
from zmesh import Mesh as Zmesh
import fastremap
import pandas as pd
from igneous_daskified.util import dask_util, io_util
import dask.bag as db
from cloudvolume.mesh import Mesh as CloudVolumeMesh
import dask.dataframe as dd
from neuroglancer.skeleton import VertexAttributeInfo
import pyvista as pv
import pymeshfix
import shutil
from igneous_daskified.process.downsample_numba import (
    downsample_labels_3d_suppress_zero,
)
import pymeshlab
import trimesh

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Meshify:
    """Skeletonize a segmentation array using kimimaro and dask"""

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        total_roi: Roi = None,
        max_num_voxels=np.inf,  # 50 * 512**3,  # 500 blocks times 512*512*512
        read_write_roi: Roi = None,
        downsample_factor: int | None = None,
        target_reduction: float = 0.99,
        num_workers: int = 10,
    ):

        for file_type in [".n5", ".zarr"]:
            if file_type in input_path:
                path_split = input_path.split(file_type + "/")
                break

        self.segmentation_array = open_ds(path_split[0] + file_type, path_split[1])
        self.output_directory = output_directory

        if total_roi:
            self.total_roi = total_roi
        else:
            self.total_roi = self.segmentation_array.roi
        self.num_workers = num_workers

        if read_write_roi:
            self.read_write_roi = read_write_roi
        else:
            self.read_write_roi: Roi = Roi(
                (0, 0, 0),
                self.segmentation_array.chunk_shape
                * self.segmentation_array.voxel_size,
            )

        self.max_num_blocks = max_num_voxels / np.prod(
            self.read_write_roi.shape / self.segmentation_array.voxel_size
        )
        self.voxel_size = self.segmentation_array.voxel_size

        self.downsample_factor = downsample_factor
        if self.downsample_factor:
            self.voxel_size *= self.downsample_factor
        self.target_reduction = target_reduction

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
        mesher = Mesher(self.voxel_size)  # anisotropy of image
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
        mesher.mesh(segmentation_block, close=False)
        for id in mesher.ids():
            mesh = mesher.get_mesh(id)
            mesh.vertices += block.roi.offset[::-1]
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
            with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
                fp.write(mesh.to_ply())

    def get_chunked_meshes(self, dirname):
        blocks = dask_util.create_blocks(
            self.total_roi,
            self.segmentation_array,
            self.read_write_roi.shape,
            self.voxel_size,
        )

        b = db.from_sequence(blocks, npartitions=self.num_workers * 10).map(
            self._get_chunked_mesh, dirname
        )

        with dask_util.start_dask(self.num_workers, "generate chunked meshes", logger):
            with io_util.Timing_Messager("Generating chunked meshes", logger):
                b.compute()

    @staticmethod
    def simplify_and_smooth_mesh(mesh, target_reduction=0.99, n_smoothing_iter=10):

        def get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_count, aggressiveness, do_simplification
        ):
            if do_simplification:
                simplified_mesh = fast_simplification.simplify_mesh(
                    mesh, target_count=target_count, agg=aggressiveness
                )
            else:
                simplified_mesh = mesh

            simplified_mesh = simplified_mesh.smooth_taubin(n_smoothing_iter)

            # clean mesh, this helps ensure watertightness, but not guaranteed?
            vclean, fclean = pymeshfix.clean_from_arrays(
                simplified_mesh.points,
                simplified_mesh.faces.reshape(-1, 4)[:, 1:],
            )
            return vclean, fclean

        components = mesh.split()

        if len(components) > 0:
            # this will equal zero if not watertight?
            mesh = components[0]
            for m in components[1:]:
                if len(m.faces) > len(mesh.faces):
                    mesh = m

        com = mesh.vertices.mean(axis=0)
        # things seem better if center first otherwise for example taubin smoothing shifts things

        vertices = mesh.vertices - com
        faces = mesh.faces
        # Create a new column filled with 3s since we need to tell pyvista that each face has 3 vertices in this mesh
        new_column = np.full((mesh.faces.shape[0], 1), 3)

        # Concatenate the new column with the original array
        faces = np.concatenate((new_column, mesh.faces), axis=1)
        mesh = pv.PolyData(vertices, faces[:])

        # initial_target_reduction = target_reduction
        min_faces = 100
        aggressiveness = 7
        # simplify mesh
        target_count = max(int(mesh.n_faces * (1 - target_reduction)), min_faces)
        do_simplification = mesh.n_faces > min_faces
        vclean, fclean = get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_count, aggressiveness, do_simplification
        )

        aggressiveness -= 1
        # initially would redo only if fclean==0, but noticed that sometimes it simplifies weird structures to a single face which isnt good
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

        if do_simplification and aggressiveness == -2:
            logger.warning(
                f"Mesh with {mesh.n_faces} faces (min_faces={min_faces}) had to be processed unsimplified."
            )

        if len(fclean) == 0:
            raise Exception(
                f"Mesh with {mesh.n_faces} faces (min_faces={min_faces}) could not be smoothed and cleaned even without simplificaiton."
            )

        vclean += com
        mesh = trimesh.Trimesh(vertices=vclean, faces=fclean)
        mesh.fix_normals()
        return mesh

    def analyze_mesh_df(self, df):
        results_df = []
        for row in df.itertuples():
            try:
                metrics = Meshify.analyze_mesh(
                    f"{self.output_directory}/meshes/{row.id}"
                )
            except Exception as e:
                raise Exception(f"Error analyzing mesh {row.id}: {e}")
            result_df = pd.DataFrame(metrics, index=[0])
            results_df.append(result_df)

        results_df = pd.concat(results_df, ignore_index=True)
        return results_df

    @staticmethod
    def analyze_mesh(mesh_path):
        id = os.path.basename(mesh_path).split(".")[0]
        mesh = trimesh.load_mesh(mesh_path)
        # calculate gneral mesh properties
        metrics = {"id": id}
        metrics["volume"] = mesh.volume
        metrics["surface_area"] = mesh.area
        pic = mesh.principal_inertia_components
        pic_normalized = pic / np.linalg.norm(pic)
        _, ob = trimesh.bounds.oriented_bounds(mesh)
        ob_normalized = ob / np.linalg.norm(ob)
        for axis in range(3):
            metrics[f"pic_{axis}"] = pic[axis]
            metrics[f"pic_normalized_{axis}"] = pic_normalized[axis]
            metrics[f"ob_{axis}"] = ob[axis]
            metrics[f"ob_normalized_{axis}"] = ob_normalized[axis]

        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
        ms.add_mesh(m)
        for idx, metric in enumerate(["mean", "gaussian", "rms", "abs"]):
            ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=idx)
            vsa = ms.current_mesh().vertex_scalar_array()
            if np.isnan(vsa).all():
                raise Exception(f"Mesh {id} has no curvature")
            metrics[f"{metric}_curvature_mean"] = np.nanmean(vsa)
            metrics[f"{metric}_curvature_median"] = np.nanmedian(vsa)
            metrics[f"{metric}_curvature_std"] = np.nanstd(vsa)

        ms.compute_scalar_by_shape_diameter_function_per_vertex()
        vsa = ms.current_mesh().vertex_scalar_array()
        if np.isnan(vsa).all():
            raise Exception(f"Mesh {id} has no thickness")
        metrics["thickness_mean"] = np.nanmean(vsa)
        metrics["thickness_median"] = np.nanmedian(vsa)
        metrics["thickness_std"] = np.nanstd(vsa)

        return metrics

    def analyze_meshes(self, dirname):
        mesh_ids = os.listdir(dirname)

        metrics = ["volume", "surface_area"]
        for axis in range(3):
            metrics.append(f"pic_{axis}")
            metrics.append(f"pic_normalized_{axis}")
            metrics.append(f"ob_{axis}")
            metrics.append(f"ob_normalized_{axis}")

        for metric in [
            "mean_curvature",
            "gaussian_curvature",
            "rms_curvature",
            "abs_curvature",
            "thickness",
        ]:
            metrics.append(f"{metric}_mean")
            metrics.append(f"{metric}_median")
            metrics.append(f"{metric}_std")

        df = pd.DataFrame({"id": mesh_ids})
        # add columns to df
        for metric in metrics:
            df[metric] = 0.0

        ddf = dd.from_pandas(df, npartitions=self.num_workers * 10)

        meta = pd.DataFrame(columns=df.columns)
        ddf_out = ddf.map_partitions(self.analyze_mesh_df, meta=meta)
        with dask_util.start_dask(self.num_workers, "analyze meshes", logger):
            with io_util.Timing_Messager("Analyzing meshes", logger):
                results = ddf_out.compute()
        # write out results to csv
        output_directory = f"{dirname}/metrics"
        os.makedirs(output_directory, exist_ok=True)
        results.to_csv(f"{output_directory}/mesh_metrics.csv", index=False)

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
        mesh_files = os.listdir(f"{self.dirname}/{mesh_id}")
        if len(mesh_files) >= self.max_num_blocks:
            logger.warn(
                f"Mesh {mesh_id} has too many blocks {len(mesh_files)})>{self.max_num_blocks}. Skipping."
            )
            store_skipped_info(self, mesh_id)

            shutil.rmtree(f"{self.dirname}/{mesh_id}")
            return

        for mesh_file in mesh_files:
            with open(f"{self.dirname}/{mesh_id}/{mesh_file}", "rb") as f:
                mesh = Zmesh.from_ply(f.read())
                block_meshes.append(mesh)
        try:
            mesh = Meshify.my_cloudvolume_concatenate(*block_meshes)
        except Exception as e:
            raise Exception(f"{mesh_id} failed, with error: {e}")

        # doing both of the following may be redundant
        mesh = mesh.consolidate()  # if you don't have self-contacts
        mesh = mesh.deduplicate_chunk_boundaries(
            chunk_size=self.read_write_roi.shape, offset=self.total_roi.offset[::-1]
        )
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

        # simplify and smooth the final mesh
        try:
            mesh = Meshify.simplify_and_smooth_mesh(mesh, self.target_reduction)
            if len(mesh.faces) == 0:
                raise Exception(f"Mesh {mesh_id} contains no faces")
        except Exception as e:
            raise Exception(f"{mesh_id} failed, with error: {e}")

        _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
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
        self.analyze_meshes(self.output_directory + "/meshes")
