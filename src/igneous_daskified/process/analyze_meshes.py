# import pymeshlab

import numpy as np
import os
import logging
import pandas as pd
from igneous_daskified.util import dask_util, io_util
import dask.dataframe as dd
import pymeshlab
import trimesh

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AnalyzeMeshes:
    """
    Analyze meshes using trimesh and pymeshlab."""

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        num_workers: int = 10,
    ):
        """
        Args:
            input_path (str): Path to the directory containing the meshes.
            output_directory (str): Path to the directory where the metrics will be saved.
            num_workers (int): Number of workers to use for parallel processing.
        """
        self.meshes_dirname = input_path
        self.metrics_dirname = output_directory
        self.num_workers = num_workers

    def analyze_mesh_df(self, df):
        results_df = []
        for row in df.itertuples():
            try:
                metrics = self.analyze_mesh(f"{self.meshes_dirname}/{row.id}")
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
        metrics["volume (nm^3)"] = mesh.volume
        metrics["surface_area (nm^2)"] = mesh.area
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

    def analyze(self):
        mesh_ids = os.listdir(self.meshes_dirname)

        metrics = ["volume (nm^3)", "surface_area (nm^2)"]
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

        ddf = dd.from_pandas(
            df,
            npartitions=dask_util.guesstimate_npartitions(len(df), self.num_workers),
        )

        meta = pd.DataFrame(columns=df.columns)
        ddf_out = ddf.map_partitions(self.analyze_mesh_df, meta=meta)
        with dask_util.start_dask(self.num_workers, "analyze meshes", logger):
            with io_util.Timing_Messager("Analyzing meshes", logger):
                results = ddf_out.compute()

        os.makedirs(self.metrics_dirname, exist_ok=True)
        # sort by id as integer
        results = results.sort_values(by="id", key=lambda x: x.astype(int))
        results.to_csv(f"{self.metrics_dirname}/mesh_metrics.csv", index=False)
