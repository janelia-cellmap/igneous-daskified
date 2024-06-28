# import pymeshlab

from funlib.persistence import Array, open_ds
from funlib.geometry import Roi
import numpy as np
import tempfile
import os
import json
import logging
from cloudvolume import Skeleton as CloudVolumeSkeleton
from neuroglancer.skeleton import Skeleton as NeuroglancerSkeleton
from funlib.geometry import Roi
from kimimaro.postprocess import _remove_ticks
import fastremap
import pandas as pd
from igneous_daskified.util import dask_util, io_util, neuroglancer_util
import dask.bag as db
import networkx as nx
import dask.dataframe as dd
from neuroglancer.skeleton import VertexAttributeInfo
from pybind11_rdp import rdp
import dask
from igneous_daskified.util.skeleton_util import CustomSkeleton

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Skeletonize:
    """Skeletonize a segmentation array using cgal and dask"""

    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        num_workers: int = 10,
        min_branch_length_nm: float = 100,
        simplification_tolerance_nm: float = 50,
    ):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.num_workers = num_workers
        self.min_branch_length_nm = min_branch_length_nm
        self.simplification_tolerance_nm = simplification_tolerance_nm

    @staticmethod
    def read_skeleton_from_custom_file(filename):
        vertices = []
        edges = []
        radii = []
        polylines = []
        with open(filename, "r") as file:
            for line in file:
                data = line.strip().split()
                if data[0] == "v":
                    # vertex
                    vertices.append((float(data[1]), float(data[2]), float(data[3])))
                    radii.append(float(data[4]))
                elif data[0] == "e":
                    # edge
                    edges.append((int(data[1]), int(data[2])))
                elif data[0] == "p":
                    # polyline
                    polyline = np.array([])
                    for v in data[1:]:
                        polyline = np.append(polyline, vertices[int(v)])
                    polylines.append(polyline.reshape(-1, 3))

        return CustomSkeleton(vertices, edges, radii, polylines)

    def _get_skeleton_from_mesh(self, mesh):
        input_file = f"{self.input_directory}/{mesh}"
        os.makedirs(f"{self.output_directory}/cgal", exist_ok=True)
        mesh_id = mesh.split(".")[0]
        output_file = f"{self.output_directory}/cgal/{mesh_id}.txt"
        exit_status = os.system(
            f"/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/cgal_skeletonize_mesh/skeletonize_mesh {input_file} {output_file}"
        )
        if exit_status:
            raise Exception(
                f"Error in skeletonizing {input_file}; exit status {os.WEXITSTATUS(exit_status)}"
            )

    def process_custom_skeleton_df(self, df):
        results_df = []
        for row in df.itertuples():
            try:
                metrics = Skeletonize.process_custom_skeleton(
                    skeleton_path=f"{self.output_directory}/cgal/{row.id}",
                    output_directory=self.output_directory,
                    min_branch_length_nm=self.min_branch_length_nm,
                    simplification_tolerance_nm=self.simplification_tolerance_nm,
                )
            except Exception as e:
                raise Exception(f"Error processing skeleton {row.id}: {e}")
            result_df = pd.DataFrame(metrics, index=[0])
            results_df.append(result_df)

        results_df = pd.concat(results_df, ignore_index=True)
        return results_df

    @staticmethod
    def process_custom_skeleton(
        skeleton_path,
        output_directory,
        min_branch_length_nm=100,
        simplification_tolerance_nm=50,
    ):
        skeleton_id = os.path.basename(skeleton_path).split(".")[0]
        custom_skeleton = Skeletonize.read_skeleton_from_custom_file(skeleton_path)

        custom_skeleton_pruned = custom_skeleton.prune(min_branch_length_nm)

        # get some properties
        metrics = {"id": skeleton_id}
        metrics["lsp (nm)"] = Skeletonize.get_longest_shortest_path_distance(
            custom_skeleton_pruned
        )
        metrics["radius mean (nm)"] = np.mean(custom_skeleton_pruned.radii)
        metrics["radius std (nm)"] = np.std(custom_skeleton_pruned.radii)
        metrics["num branches"] = len(custom_skeleton_pruned.polylines)

        custom_skeleton_pruned_simplified = custom_skeleton_pruned.simplify(
            simplification_tolerance_nm
        )

        custom_skeleton.write_neuroglancer_skeleton(
            f"{output_directory}/skeleton/full/{skeleton_id}"
        )
        custom_skeleton_pruned_simplified.write_neuroglancer_skeleton(
            f"{output_directory}/skeleton/simplified/{skeleton_id}"
        )
        return metrics

    def get_skeletons_from_meshes(self):
        meshes = os.listdir(self.input_directory)
        # meshes = [f"{i}.ply" for i in range(1, 1000)]
        b = db.from_sequence(
            meshes, npartitions=min(self.num_workers * 10, len(meshes))
        ).map(self._get_skeleton_from_mesh)
        with dask_util.start_dask(self.num_workers, "create skeletons", logger):
            with io_util.Timing_Messager("Creating skeletons", logger):
                b.compute()

    def process_custom_skeletons(self):
        self.cgal_output_directory = f"{self.output_directory}/cgal/"
        skeleton_filenames = os.listdir(self.cgal_output_directory)
        metrics = ["lsp (nm)", "radius mean (nm)", "radius std (nm)", "num branches"]
        df = pd.DataFrame({"id": skeleton_filenames})
        # add columns to df
        for metric in metrics:
            df[metric] = -1.0

        ddf = dd.from_pandas(df, npartitions=self.num_workers * 10)

        meta = pd.DataFrame(columns=df.columns)
        ddf_out = ddf.map_partitions(self.process_custom_skeleton_df, meta=meta)
        with dask_util.start_dask(self.num_workers, "process skeletons", logger):
            with io_util.Timing_Messager("Processing skeletons", logger):
                results = ddf_out.compute()

        # write out results to csv
        output_directory = f"{self.output_directory}/metrics"
        os.makedirs(output_directory, exist_ok=True)

        # write out results
        results.sort_values("lsp (nm)", ascending=False, inplace=True)
        results.to_csv(f"{output_directory}/skeleton_metrics.csv", index=False)

        skeleton_ids = [
            skeleton_filename.split(".txt")[0]
            for skeleton_filename in skeleton_filenames
        ]
        # write out neuroglancer metadata
        self.__write_skeleton_metadata(
            f"{self.output_directory}/skeleton/full", skeleton_ids
        )
        self.__write_skeleton_metadata(
            f"{self.output_directory}/skeleton/simplified", skeleton_ids
        )

    # def __write_skeleton(
    #     self,
    #     skeleton_id,
    #     vertices,
    #     edges,
    #     vertex_attributes=None,
    #     grouped_by_cells=False,
    # ):
    #     output_directory = self.output_directory
    #     if grouped_by_cells:
    #         output_directory += "_cells"

    #     with open(f"{output_directory}/{skeleton_id}", "wb") as f:
    #         skel = NeuroglancerSkeleton(vertices, edges, vertex_attributes)
    #         encoded = skel.encode(self.source)
    #         f.write(encoded)

    def __write_skeleton_metadata(
        self, output_directory, skeleton_ids, grouped_by_cells=False
    ):
        if grouped_by_cells:
            output_directory += "_cells"
        os.makedirs(output_directory, exist_ok=True)

        metadata = {
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "segment_properties": "segment_properties",
        }

        # self.source = neuroglancer_util.Source()
        # self.get_chunked_skeletons(tupdupname)
        # self.process_chunked_skeletons(tupdupname)
        # self.source = neuroglancer_util.Source(
        #     vertex_attributes={
        #         "lsp": VertexAttributeInfo(data_type=np.float32, num_components=1)
        #     }
        # )
        # if self.source.vertex_attributes:
        #     vertex_attributes = []
        #     for attribute_name, attribute_info in self.source.vertex_attributes.items():
        #         vertex_attributes.append(
        #             {
        #                 "id": attribute_name,
        #                 "data_type": str(attribute_info.data_type).split("np.")[-1],
        #                 "num_components": attribute_info.num_components,
        #             }
        #         )
        #     metadata["vertex_attributes"] = vertex_attributes

        with open(os.path.join(f"{output_directory}", "info"), "w") as f:
            f.write(json.dumps(metadata))

        os.makedirs(f"{output_directory}/segment_properties", exist_ok=True)
        segment_properties = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(skeleton_id) for skeleton_id in skeleton_ids],
                "properties": [
                    {"id": "label", "type": "label", "values": [""] * len(skeleton_ids)}
                ],
            },
        }
        with open(f"{output_directory}/segment_properties/info", "w") as f:
            f.write(json.dumps(segment_properties))

    def get_all_skeleton_information(self, df):
        results_df = []
        for row in df.itertuples():
            _, _, max_distance = self.get_longest_shortest_path_distance(str(row.id))
            result_df = pd.DataFrame(
                {
                    "id": [row.id],
                    "cell": [row.cell],
                    "com_x_nm": [row.com_x_nm],
                    "com_y_nm": [row.com_y_nm],
                    "com_z_nm": [row.com_z_nm],
                    "lsp": [max_distance],
                }
            )
            results_df.append(result_df)

        results_df = pd.concat(results_df, ignore_index=True)
        return results_df

    @staticmethod
    def get_longest_shortest_path_distance(skeleton):
        # make graph using edges and vertices
        g = nx.Graph()
        g.add_nodes_from(range(len(skeleton.vertices)))
        g.add_edges_from(skeleton.edges)
        # add edge weights to the graph where weights are the distances between vertices
        for edge in skeleton.edges:
            g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                np.array(skeleton.vertices[edge[0]])
                - np.array(skeleton.vertices[edge[1]])
            )
        node_distances = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
        # get the maximum distance between any two nodes
        max_distance = max(
            max(distance_dict.values()) for distance_dict in node_distances.values()
        )

        return max_distance

    # def get_longest_shortest_path_distance(self, skeleton_id):
    #     with open(
    #         f"{self.output_directory}/{skeleton_id}",
    #         "rb",
    #     ) as f:
    #         file_content = f.read()

    #     num_vertices, file_content = unpack_and_remove("I", 1, file_content)
    #     num_edges, file_content = unpack_and_remove("I", 1, file_content)

    #     vertices, file_content = unpack_and_remove("f", 3 * num_vertices, file_content)
    #     vertices = np.reshape(vertices, (num_vertices, 3))

    #     edges, file_content = unpack_and_remove("I", 2 * num_edges, file_content)
    #     edges = np.reshape(edges, (num_edges, 2))

    #     # make graph using edges and vertices
    #     g = nx.Graph()
    #     g.add_nodes_from(range(num_vertices))
    #     g.add_edges_from(edges)
    #     # add edge weights to the graph where weights are the distances between vertices
    #     for edge in edges:
    #         g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
    #             vertices[edge[0]] - vertices[edge[1]]
    #         )
    #     node_distances = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
    #     max_distance = 0
    #     for node, distance_dict in node_distances.items():
    #         max_dist_node = max(distance_dict, key=distance_dict.get)
    #         current_max_distance = distance_dict[max_dist_node]
    #         if current_max_distance > max_distance:
    #             max_distance = current_max_distance
    #             max_distance_node_1 = node
    #             max_distance_node_2 = max_dist_node
    #             # print(node, max_dist_node, current_max_distance)
    #     longest_shortest_path = nx.dijkstra_path(
    #         g, max_distance_node_1, max_distance_node_2, weight="weight"
    #     )

    #     return vertices, edges, max_distance

    def group_skeletons_by_cell(self, cell_id):
        current_cell_mitos = self.mito_ids[
            np.where(self.mito_to_cell_ids == cell_id)[0]
        ]
        # make empty numpy array of vertex positions
        all_vertices = np.array([], dtype=np.float32).reshape(0, 3)
        all_edges = np.array([], dtype=int).reshape(0, 2)
        all_longest_shortest_path_lengths = np.array([], dtype=np.float32)
        for skeleton_id in current_cell_mitos:
            vertices, edges, longest_shortest_path_distance = (
                self.get_longest_shortest_path_distance(skeleton_id)
            )

            all_edges = np.concatenate((all_edges, len(all_vertices) + edges))
            all_vertices = np.concatenate((all_vertices, vertices))
            all_longest_shortest_path_lengths = np.concatenate(
                (
                    all_longest_shortest_path_lengths,
                    [longest_shortest_path_distance] * len(vertices),
                )
            )

            self.__write_skeleton(
                cell_id,
                all_vertices,
                all_edges,
                vertex_attributes={"lsp": all_longest_shortest_path_lengths},
                grouped_by_cells=True,
            )

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

    def group_skeletons_by_cells(self):
        self.assign_mitos_to_cells()
        self.__write_skeleton_metadata(
            self.all_cells,
            grouped_by_cells=True,
        )
        b = db.from_sequence(self.all_cells, npartitions=self.num_workers * 10).map(
            self.group_skeletons_by_cell
        )
        with dask_util.start_dask(self.num_workers, "grouping  skeletons", logger):
            with io_util.Timing_Messager("Grouping skeletons with cells", logger):
                b.compute()

    def assign_skeleton_info_to_com(self):
        def write_skeletons_as_points(results):
            ids = results["id"].to_numpy()
            coords = results[["com_x_nm", "com_y_nm", "com_z_nm"]].to_numpy()
            properties_dict = {"lsp": results["lsp"].to_numpy()}

            cell_grouping = results.groupby(["cell"]).indices
            relationships_dict = {
                cell_id: corresponding_indices
                for cell_id, corresponding_indices in cell_grouping.items()
            }
            output_directory = self.output_directory + "_as_points"
            neuroglancer_util.write_precomputed_annotations(
                output_directory=output_directory,
                annotation_type="point",
                ids=ids,
                coords=coords,
                properties_dict=properties_dict,
                relationships_dict=relationships_dict,
            )

        self.assign_mitos_to_cells()
        df = pd.DataFrame(
            {
                "id": self.mito_ids,
                "cell": self.mito_to_cell_ids,
                "com_x_nm": self.mito_coms[:, 0],
                "com_y_nm": self.mito_coms[:, 1],
                "com_z_nm": self.mito_coms[:, 2],
                "lsp": [-1.0] * len(self.mito_ids),
            }
        )

        ddf = dd.from_pandas(df, npartitions=self.num_workers * 10)

        meta = pd.DataFrame(columns=df.columns)
        ddf_out = ddf.map_partitions(self.get_all_skeleton_information, meta=meta)
        with dask_util.start_dask(self.num_workers, "skeletons by com", logger):
            with io_util.Timing_Messager(
                "Grouping skeletons with cells, by com", logger
            ):
                results = ddf_out.compute()

        write_skeletons_as_points(results)

    def get_skeletons(self):
        """Get the skeletons using daisy and dask save them to disk in a temporary directory"""
        os.makedirs(self.output_directory, exist_ok=True)
        self.get_skeletons_from_meshes()
        self.process_custom_skeletons()
