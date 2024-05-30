from dataclasses import dataclass
import struct
import fast_simplification

# import pymeshlab

import daisy
from funlib.persistence import Array, open_ds
from funlib.geometry import Roi
import numpy as np
import tempfile
import pickle
import os
import json
import logging
import kimimaro
import navis
import skeletor
from cloudvolume import Skeleton as CloudVolumeSkeleton
from neuroglancer.skeleton import Skeleton as NeuroglancerSkeleton
import dask
from dataclasses import dataclass
from funlib.geometry import Roi
from zmesh import Mesher
from zmesh import Mesh as Zmesh
from kimimaro.postprocess import remove_ticks, connect_pieces
import fastremap
import pandas as pd
from igneous_daskified.util import dask_util, io_util, neuroglancer_util
import dask.bag as db
import networkx as nx
from cloudvolume.mesh import Mesh as CloudVolumeMesh
import dask.dataframe as dd
from neuroglancer.skeleton import VertexAttributeInfo
import pyvista as pv
import pymeshfix
import shutil

# prepend ld_library_path with path to pymeshlab shared libraries
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
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
        segmentation_array: Array,
        output_directory: str,
        total_roi: Roi = None,
        read_write_roi: Roi = None,
        log_dir: str = None,
        num_workers: int = 10,
    ):
        self.segmentation_array = segmentation_array
        self.output_directory = output_directory

        if total_roi:
            self.total_roi = total_roi
        else:
            self.total_roi = segmentation_array.roi
        self.num_workers = num_workers

        if read_write_roi:
            self.read_write_roi = read_write_roi
        else:
            self.read_write_roi: Roi = Roi(
                (0, 0, 0),
                self.segmentation_array.chunk_shape
                * self.segmentation_array.voxel_size,
            )

        self.voxel_size = segmentation_array.voxel_size
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            daisy.logging.set_log_basedir(log_dir)

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
            self.segmentation_array.voxel_size,
        )

        b = db.from_sequence(blocks, npartitions=self.num_workers * 10).map(
            self._get_chunked_mesh, dirname
        )

        with dask_util.start_dask(self.num_workers, "generate chunked meshes", logger):
            with io_util.Timing_Messager("Generating chunked meshes", logger):
                b.compute()

    @staticmethod
    def simplify_and_smooth_mesh(mesh, target_reduction=0.99, n_smoothing_iter=10):
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

        # simplify mesh
        min_faces = 100
        if mesh.n_faces > min_faces:
            target_count = max(int(mesh.n_faces * (1 - target_reduction)), min_faces)
            mesh = fast_simplification.simplify_mesh(mesh, target_count=target_count)

        mesh = mesh.smooth_taubin(n_smoothing_iter)

        # clean mesh, this helps ensure watertightness, but not guaranteed?
        vclean, fclean = pymeshfix.clean_from_arrays(
            mesh.points,
            mesh.faces.reshape(-1, 4)[:, 1:],
        )

        vclean += com
        mesh = trimesh.Trimesh(vertices=vclean, faces=fclean)
        mesh.fix_normals()
        return mesh

    def analyze_mesh_df(self, df):
        results_df = []
        for row in df.itertuples():
            try:
                metrics = Meshify.analyze_mesh(f"{self.output_directory}/{row.id}")
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
        output_directory = f"{dirname}metrics"
        os.makedirs(output_directory, exist_ok=True)
        results.to_csv(f"{output_directory}/mesh_metrics.csv", index=False)

    def _assemble_mesh(self, mesh_id):
        block_meshes = []
        for mesh_file in os.listdir(f"{self.dirname}/{mesh_id}"):
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
            mesh = Meshify.simplify_and_smooth_mesh(mesh)
            if len(mesh.faces) == 0:
                raise Exception(f"Mesh {mesh_id} contains no faces")
        except Exception as e:
            raise Exception(f"{mesh_id} failed, with error: {e}")

        _ = mesh.export(f"{self.output_directory}/{mesh_id}.ply")
        shutil.rmtree(f"{self.dirname}/{mesh_id}")

    def assemble_meshes(self, dirname):
        self.dirname = dirname
        mesh_ids = os.listdir(dirname)
        b = db.from_sequence(mesh_ids, npartitions=self.num_workers * 10).map(
            self._assemble_mesh
        )
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
        with tempfile.TemporaryDirectory() as tmpdirname:
            tupdupname = "/nrs/cellmap/ackermand/tmp/ld/20240529_1"
            os.makedirs(tupdupname, exist_ok=True)
            self.get_chunked_meshes(tupdupname)
            self.assemble_meshes(tupdupname)
            self.analyze_meshes(self.output_directory)
