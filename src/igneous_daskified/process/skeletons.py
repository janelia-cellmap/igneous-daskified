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
from cloudvolume import Mesh
from neuroglancer.skeleton import Skeleton as NeuroglancerSkeleton
import dask
from dataclasses import dataclass
from funlib.geometry import Roi
from zmesh import Mesher, Mesh
from kimimaro.postprocess import remove_ticks, connect_pieces
import fastremap
import pandas as pd
from igneous_daskified.util import dask_util, io_util
import dask.bag as db
import networkx as nx
from neuroglancer.skeleton import VertexAttributeInfo


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def unpack_and_remove(datatype, num_elements, file_content):
    """Read and remove bytes from binary file object

    Args:
        datatype: Type of data
        num_elements: Number of datatype elements to read
        file_content: Binary file object

    Returns:
        output: The data that was unpacked
        file_content: The file contents with the unpacked data removed
    """

    datatype = datatype * num_elements
    output = struct.unpack(datatype, file_content[0 : 4 * num_elements])
    file_content = file_content[4 * num_elements :]
    if num_elements == 1:
        return output[0], file_content
    else:
        return np.array(output), file_content


# encoder for uint64 from https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@dataclass
class Source:
    """Source for the NeuroglancerSkeleton"""
    def __init__(self, vertex_attributes=[]):
        self.vertex_attributes = vertex_attributes


@dataclass
class DaskBlock:
    index: int
    roi: Roi


def create_blocks(
    roi: Roi,
    ds: Array,
    block_size=None,
    padding=None,
):
    with io_util.Timing_Messager("Generating blocks", logger):
        # roi = roi.snap_to_grid(ds.chunk_shape * ds.voxel_size)
        if not block_size:
            block_size = ds.chunk_shape * ds.voxel_size

        num_expected_blocks = int(
            np.prod(
                [np.ceil(roi.shape[i] / block_size[i]) for i in range(len(block_size))]
            )
        )
        # create an empty list with num_expected_blocks elements
        block_rois = [None] * num_expected_blocks
        index = 0
        for z in range(roi.get_begin()[2], roi.get_end()[2], block_size[2]):
            for y in range(roi.get_begin()[1], roi.get_end()[1], block_size[1]):
                for x in range(roi.get_begin()[0], roi.get_end()[0], block_size[0]):
                    block_roi = Roi((x, y, z), block_size).intersect(roi)
                    if padding:
                        block_roi = block_roi.grow(padding, padding)
                    block_rois[index] = DaskBlock(index, block_roi.intersect(roi))
                    index += 1
        if index < len(block_rois):
            block_rois[index:] = []
    return block_rois


class Skeletonize:
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

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            daisy.logging.set_log_basedir(log_dir)

    def zmesh_get_chunked_mesh(self, block, tmpdirname):
        mesher = Mesher((8, 8, 8))  # anisotropy of image
        segmentation_block = self.segmentation_array.to_ndarray(block.roi)
        # initial marching cubes pass
        # close controls whether meshes touching
        # the image boundary are left open or closed
        mesher.mesh(segmentation_block, close=False)
        for id in mesher.ids():
            mesh = mesher.get_mesh(id)
            mesh.vertices += block.roi.offset[::-1]
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
            # Extremely common obj format
            with open(f"{tmpdirname}/{id}/block_{block.index}.obj", "wb") as fp:
                fp.write(mesh.to_obj())

    def get_chunked_mesh(self, block, tmpdirname):
        mesher = Mesher((8, 8, 8))  # anisotropy of image
        segmentation_block = self.segmentation_array.to_ndarray(block.roi)
        # initial marching cubes pass
        # close controls whether meshes touching
        # the image boundary are left open or closed
        mesher.mesh(segmentation_block, close=False)
        for id in mesher.ids():
            mesh = mesher.get_mesh(id)
            mesh.vertices += block.roi.offset[::-1]
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
            # Extremely common obj format
            with open(f"{tmpdirname}/{id}/block_{block.index}.obj", "wb") as fp:
                fp.write(mesh.to_obj())

    def get_chunked_meshes(self, dirname):
        blocks = create_blocks(
            self.total_roi,
            self.segmentation_array,
            self.read_write_roi.shape,
            # self.segmentation_array.voxel_size,
        )

        lazy_results = [None] * len(blocks)
        for block_idx, block in enumerate(blocks):
            lazy_results[block_idx] = dask.delayed(self.get_chunked_mesh)(
                block, dirname
            )

        dask.compute(*lazy_results)

    def get_chunked_skeletons(self, dirname):
        self.dirname = dirname
        blocks = create_blocks(
            self.total_roi,
            self.segmentation_array,
            self.read_write_roi.shape,
            # self.segmentation_array.voxel_size,
        )

        b = db.from_sequence(blocks, npartitions=self.num_workers * 10).map(
            self.get_chunked_skeleton
        )
        with dask_util.start_dask(self.num_workers, "generating  skeletons", logger):
            with io_util.Timing_Messager("Generating chunked skeletons", logger):
                b.compute()
        # lazy_results = []
        # for block in blocks:
        #     lazy_results.append(dask.delayed(self.get_chunked_skeleton)(block, dirname))
        # with io_util.Timing_Messager("Generating chunked skeletons", logger):
        #     with dask_util.start_dask(self.num_workers, "chunked skeletons", logger):
        #         dask.compute(*lazy_results)

    def simplify_and_smooth_mesh(self, mesh, target_reduction=0.9, n_smoothing_iter=20):
        vertices = mesh.vertices
        faces = mesh.faces
        # Create a new column filled with 3s since we need to tell pyvista that each face has 3 vertices in this mesh
        new_column = np.full((original_mesh.faces.shape[0], 1), 3)

        # Concatenate the new column with the original array
        faces = np.concatenate((new_column, original_mesh.faces), axis=1)
        mesh = pv.PolyData(vertices, faces[:])

        output_mesh = fast_simplification.simplify_mesh(
            mesh, target_reduction=target_reduction
        )
        output_mesh.smooth(n_iter=n_smoothing_iter)

        return output_mesh

    def __assemble_mesh(self, dirname, mesh_id):
        block_meshes = []
        for mesh_file in os.listdir(f"{dirname}/{mesh_id}"):
            with open(f"{dirname}/{mesh_id}/{mesh_file}", "rb") as f:
                mesh = Mesh.from_obj(f.read())
                block_meshes.append(mesh)
        mesh = Mesh.concatenate(*block_meshes)

        # doing both of the following may be redundant
        mesh = mesh.consolidate()  # if you don't have self-contacts
        mesh = mesh.deduplicate_chunk_boundaries(
            chunk_size=self.read_write_roi.shape, offset=self.total_roi.offset[::-1]
        )  # if 512,512,512 is your chunk size

        # simplify and smooth the final mesh
        mesh = self.simplify_and_smooth(mesh)
        return mesh

    def __my_kimimaro_postprocess(self, skeleton, tick_threshold):
        def _my_remove_ticks(skeleton, tick_threshold):
            _, unique_counts = fastremap.unique(skeleton.edges, return_counts=True)
            num_terminal_nodes = np.any(unique_counts == 1)
            if num_terminal_nodes:
                skeleton = remove_ticks(skeleton, tick_threshold)
            return skeleton

        label = skeleton.id

        # necessary for removing trivial loops etc
        # remove_loops and remove_ticks assume a
        # clean representation
        skeleton = skeleton.consolidate()
        # The commented out code below was from the original kimimaro postprocess, but based on our processing, we shouldnt have disconnected components and we want to keep loops
        # skeleton = remove_dust(skeleton, dust_threshold)
        # skeleton = remove_loops(skeleton)
        skeleton = connect_pieces(skeleton)
        skeleton = _my_remove_ticks(skeleton, tick_threshold)
        skeleton.id = label
        skeleton = skeleton.consolidate()

        # NOTE: Our current liver-zon-1 mitos have some disconnected components so we have to only keep largest
        # only keep largest component
        comps = skeleton.components()
        if len(skeleton.components()):
            skeleton = comps[0]
            for i in range(1, len(comps)):
                if len(comps[i].vertices) > len(skeleton.vertices):
                    skeleton = comps[i]

        # TODO: put this back
        # assert len(skeleton.components()) == 1

        return skeleton

    def __assemble_skeleton(self, dirname, skeleton_id):
        block_skeletons = []
        for skeleton_file in os.listdir(f"{dirname}/{skeleton_id}"):
            with open(f"{dirname}/{skeleton_id}/{skeleton_file}", "rb") as f:
                block_skeleton = pickle.load(f)
                block_skeletons.append(block_skeleton)
        cloudvolume_skeleton = CloudVolumeSkeleton.simple_merge(block_skeletons)
        cloudvolume_skeleton = self.__my_kimimaro_postprocess(cloudvolume_skeleton, 80)

        return cloudvolume_skeleton

    def __simplify_cloudvolume_skeleton(self, cloudvolume_skeleton):
        # to simplify skeletons: https://github.com/navis-org/skeletor/issues/38#issuecomment-1491714639
        n = navis.TreeNeuron(cloudvolume_skeleton.to_swc(), soma=None)
        ds = navis.downsample_neuron(n, downsampling_factor=20)

        # the above downsampling breaks loops, so we now have to rebuild the loops by finding edges that were broken
        nodes_df = ds.nodes
        endpoints = (
            nodes_df.loc[nodes_df["type"].isin(["root", "end"]), ["node_id"]]
            .to_numpy()
            .flatten()
        )
        cloudvolume_edges = cloudvolume_skeleton.edges.tolist()
        edges = ds.edges
        for i in range(len(endpoints)):
            e1 = endpoints[i]
            for j in range(i + 1, len(endpoints)):
                e2 = endpoints[j]
                if [e1 - 1, e2 - 1] in cloudvolume_edges or [
                    e2 - 1,
                    e1 - 1,
                ] in cloudvolume_edges:
                    print(e1 - 1, e2 - 1)
                    edges = np.concatenate((edges, [[e1, e2]]))
        edges = fastremap.remap(
            edges, dict(zip(nodes_df["node_id"].to_numpy(), nodes_df.index))
        )
        vertices = nodes_df[["x", "y", "z"]].to_numpy()
        radii = nodes_df["radius"].to_numpy().flatten()

        return CloudVolumeSkeleton(edges=edges, vertices=vertices, radii=radii)

    def __write_skeleton(
        self,
        skeleton_id,
        vertices,
        edges,
        vertex_attributes=None,
        grouped_by_cells=False,
    ):
        output_directory = self.output_directory
        if grouped_by_cells:
            output_directory += "_cells"

        with open(f"{output_directory}/{skeleton_id}", "wb") as f:
            skel = NeuroglancerSkeleton(vertices, edges, vertex_attributes)
            encoded = skel.encode(self.source)
            f.write(encoded)

    def __write_skeleton_metadata(self, skeleton_ids, grouped_by_cells=False):
        output_directory = self.output_directory
        if grouped_by_cells:
            output_directory += "_cells"
        os.makedirs(output_directory, exist_ok=True)

        metadata = {
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "segment_properties": "segment_properties",
        }

        if self.source.vertex_attributes:
            vertex_attributes = []
            for attribute_name, attribute_info in self.source.vertex_attributes.items():
                vertex_attributes.append(
                    {
                        "id": attribute_name,
                        "data_type": str(attribute_info.data_type).split("np.")[-1],
                        "num_components": attribute_info.num_components,
                    }
                )
            metadata["vertex_attributes"] = vertex_attributes

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

    # @dask.delayed
    def process_chunked_skeleton(self, skeleton_id):
        try:
            cloudvolume_skeleton = self.__assemble_skeleton(self.dirname, skeleton_id)
        except:
            with open("/nrs/cellmap/ackermand/whydask/fail.txt", "w") as f:
                f.write(
                    json.dumps("/nrs/cellmap/ackermand/whydask/fail.txt {skeleton_id}")
                )
            raise Exception(f"skeleton assembly failed for id {skeleton_id}")
        try:
            simplified_skeletor_skeleton = self.__simplify_cloudvolume_skeleton(
                cloudvolume_skeleton
            )
        except:
            with open("/nrs/cellmap/ackermand/whydask/fail.txt", "w") as f:
                f.write(
                    json.dumps("/nrs/cellmap/ackermand/whydask/fail.txt {skeleton_id}")
                )
            raise Exception(f"skeleton simplification failed for id {skeleton_id}")
        self.__write_skeleton(
            skeleton_id,
            simplified_skeletor_skeleton.vertices,
            simplified_skeletor_skeleton.edges,
        )

    def process_chunked_skeletons(self, dirname):
        self.dirname = dirname
        """Process the chunked skeletons in parallel using dask"""
        # skeleton_ids = os.listdir(dirname)
        # lazy_results = [None] * len(skeleton_ids)
        # for idx, skeleton_id in enumerate(skeleton_ids):
        #     lazy_results[idx] = self.__process_chunked_skeleton(dirname, skeleton_id)

        # with dask_util.start_dask(self.num_workers, "assemble skeletons", logger):
        #     with io_util.Timing_Messager("Generating assemble skeletons", logger):

        #         dask.compute(*lazy_results)

        skeleton_ids = os.listdir(dirname)
        b = db.from_sequence(skeleton_ids, npartitions=self.num_workers * 10).map(
            self.process_chunked_skeleton
        )
        with dask_util.start_dask(self.num_workers, "assemble skeletons", logger):
            with io_util.Timing_Messager("Assembling skeletons", logger):
                b.compute()
                # b.to_textfiles("/nrs/cellmap/ackermand/whydask/*.json.gz")
                # dask.compute(*lazy_results)

        self.__write_skeleton_metadata(skeleton_ids)

    def get_chunked_skeleton(
        self,
        block,  #: DaskBlock,
    ):
        os.makedirs(
            "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts",
            exist_ok=True,
        )
        try:
            # t0 = time.time()
            segmentation_block = self.segmentation_array.to_ndarray(block.roi)
            if np.any(segmentation_block):
                # raise Exception(dask.config.config)
                # swap byte order in place if little endian
                if segmentation_block.dtype.byteorder == ">":
                    segmentation_block = segmentation_block.newbyteorder().byteswap()
                # t1 = time.time()

                skels = kimimaro.skeletonize(
                    segmentation_block,
                    teasar_params={
                        "scale": 4,  # 1.5,
                        "const": 300,  # physical units
                        "pdrf_scale": 100000,
                        "pdrf_exponent": 4,
                        "soma_acceptance_threshold": 100000,  # 3500,  # physical units
                        "soma_detection_threshold": 100000,  # 750,  # physical units
                        "soma_invalidation_const": 300,  # physical units
                        "soma_invalidation_scale": 2,
                        "max_paths": 300,  # default None
                    },
                    # object_ids=[ ... ], # process only the specified labels
                    # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
                    # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
                    dust_threshold=0,  # skip connected components with fewer than this many voxels
                    anisotropy=(8, 8, 8),  # default True
                    fix_branching=True,  # default True
                    fix_borders=True,  # default True
                    fill_holes=False,  # default False
                    fix_avocados=False,  # default False
                    progress=False,  # default False, show progress bar
                    parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
                    parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
                )
                # t2 = time.time()
                for id, skel in skels.items():
                    # skeletons, unlike meshes, come out zyx
                    skel.vertices = np.fliplr(skel.vertices + block.roi.offset)

                    os.makedirs(f"{self.dirname}/{id}", exist_ok=True)
                    with open(
                        f"{self.dirname}/{id}/block_{block.index}.pkl", "wb"
                    ) as fp:
                        pickle.dump(skel, fp)
        except:
            raise Exception(
                f"chunked skeleton failed for block {block},{block.index},{block.roi}"
            )
        # t3 = time.time()
        # write elapsed time to file named with blockid
        # with open(
        #     f"/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts/{block.index}",
        #     "w",
        # ) as f:
        #     f.write(f"{time.time() - t0},{t1-t0},{t2-t1},{t3-t2}")

    def get_all_skeleton_information(self, df):
        results_df = []
        for row in df.itertuples():
            _, _, max_distance = self.get_longest_shortest_path_distance(row.id)
            com_x, com_y, com_z = self.get_com(row.id)
            result_df = pd.DataFrame(
                {
                    "id": [row.id],
                    "COM X (nm)": [com_x],
                    "COM Y (nm)": [com_y],
                    "COM Z (nm)": [com_z],
                    "lsp": [max_distance],
                    "cell": [cell_id],
                }
            )

    def get_longest_shortest_path_distance_df(self, df):
        results_df = []

        for row in df.itertuples():
            _, _, max_distance = self.get_longest_shortest_path_distance(row.id)

            result_df = pd.DataFrame(
                {
                    "id": [row.id],
                    "COM X (nm)": [0.0],
                    "COM Y (nm)": [0.0],
                    "COM Z (nm)": [0.0],
                    "lsp": [max_distance],
                }
            )
            results_df.append(result_df)

        results_df = pd.concat(results_df, ignore_index=True)
        return results_df

    def get_longest_shortest_path_distance(self, skeleton_id):
        with open(
            f"{self.output_directory}/{skeleton_id}",
            "rb",
        ) as f:
            file_content = f.read()

        num_vertices, file_content = unpack_and_remove("I", 1, file_content)
        num_edges, file_content = unpack_and_remove("I", 1, file_content)

        vertices, file_content = unpack_and_remove("f", 3 * num_vertices, file_content)
        vertices = np.reshape(vertices, (num_vertices, 3))

        edges, file_content = unpack_and_remove("I", 2 * num_edges, file_content)
        edges = np.reshape(edges, (num_edges, 2))

        # make graph using edges and vertices
        g = nx.Graph()
        g.add_nodes_from(range(num_vertices))
        g.add_edges_from(edges)
        # add edge weights to the graph where weights are the distances between vertices
        for edge in edges:
            g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                vertices[edge[0]] - vertices[edge[1]]
            )
        node_distances = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
        max_distance = 0
        for node, distance_dict in node_distances.items():
            max_dist_node = max(distance_dict, key=distance_dict.get)
            current_max_distance = distance_dict[max_dist_node]
            if current_max_distance > max_distance:
                max_distance = current_max_distance
                max_distance_node_1 = node
                max_distance_node_2 = max_dist_node
                # print(node, max_dist_node, current_max_distance)
        longest_shortest_path = nx.dijkstra_path(
            g, max_distance_node_1, max_distance_node_2, weight="weight"
        )

        return vertices, edges, max_distance

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

        self.__write_skeleton_metadata(
            all_cells,
            grouped_by_cells=True,
        )

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

    def get_skeletons(self):
        """Get the skeletons using daisy and dask save them to disk in a temporary directory"""
        os.makedirs(self.output_directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tupdupname = "/nrs/cellmap/ackermand/tests/AubreyPresentation20240502/848UpdateTesar/skeletons"  # WithCorrectOverlap"  # "/nrs/cellmap/ackermand/tests/20240430/skeleton5/"  # "/nrs/cellmap/ackermand/tests/AubreyPresentation20240502/20240501/skeletons"  # AubreyPresentation20240502/20240501/skeletons "/nrs/cellmap/ackermand/tests/20240430/skeleton3/"
            # self.source = Source()
            # self.get_chunked_skeletons(tupdupname)
            # self.process_chunked_skeletons(tupdupname)
            self.source = Source(
                vertex_attributes={
                    "lsp": VertexAttributeInfo(data_type=np.float32, num_components=1)
                }
            )
            self.group_skeletons_by_cells()

    def get_meshes(self):
        """Get the meshes using dask save them to disk in a temporary directory"""
        os.makedirs(self.output_directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tupdupname = "/nrs/cellmap/ackermand/tests/tmp/dask/mesh/"
            self.get_chunked_mesh(tupdupname)
