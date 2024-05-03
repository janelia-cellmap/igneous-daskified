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

from igneous_daskified.util import dask_util, io_util
import dask.bag as db

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

    vertex_attributes = []


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
        blocks = create_blocks(
            self.total_roi,
            self.segmentation_array,
            self.read_write_roi.shape,
            self.segmentation_array.voxel_size,
        )
        lazy_results = []
        for block in blocks:
            lazy_results.append(dask.delayed(self.get_chunked_skeleton)(block, dirname))
        with dask_util.start_dask(self.num_workers, "chunked skeletons", logger):
            dask.compute(*lazy_results)

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

    def __assemble_skeleton(self, dirname, skeleton_id):
        block_skeletons = []
        for skeleton_file in os.listdir(f"{dirname}/{skeleton_id}"):
            with open(f"{dirname}/{skeleton_id}/{skeleton_file}", "rb") as f:
                block_skeleton = pickle.load(f)
                block_skeletons.append(block_skeleton)
                # if skeleton_id == "14654803557897":
                #    print(len(block_skeleton.vertices))
        cloudvolume_skeleton = CloudVolumeSkeleton.simple_merge(
            block_skeletons
        ).consolidate()
        cloudvolume_skeleton = kimimaro.postprocess(
            cloudvolume_skeleton,
            dust_threshold=0,
            tick_threshold=80,  # physical units  # physical units
        )
        return cloudvolume_skeleton

    def __simplify_cloudvolume_skeleton(self, cloudvolume_skeleton):
        # to simplify skeletons: https://github.com/navis-org/skeletor/issues/38#issuecomment-1491714639
        n = navis.TreeNeuron(cloudvolume_skeleton.to_swc(), soma=None)
        ds = navis.downsample_neuron(n, downsampling_factor=20)

        # to renumber nodes
        # https://github.com/navis-org/skeletor/issues/26#issuecomment-1086965768
        swc = navis.io.swc_io.make_swc_table(ds)
        # We also need to rename some columns
        swc = swc.rename(
            {
                "PointNo": "node_id",
                "Parent": "parent_id",
                "X": "x",
                "Y": "y",
                "Z": "z",
                "Radius": "radius",
            },
            axis=1,
        ).drop("Label", axis=1)
        # Skeletor excepts node IDs to start with 0, but navis starts at 1 for SWC
        swc["node_id"] -= 1
        swc.loc[swc.parent_id > 0, "parent_id"] -= 1
        # Create the skeletor.Skeleton
        skeletor_skeleton = skeletor.Skeleton(swc)
        return skeletor_skeleton

    def __write_out_skeleton(self, skeleton_id, skeletor_skeleton):
        with open(f"{self.output_directory}/{skeleton_id}", "wb") as f:
            # need to flip to get in xyz
            skel = NeuroglancerSkeleton(
                skeletor_skeleton.vertices, skeletor_skeleton.edges
            )
            encoded = skel.encode(Source())
            f.write(encoded)

    def __write_out_metadata(self, skeleton_ids):
        metadata = {
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "segment_properties": "segment_properties",
        }

        os.makedirs(f"{self.output_directory}/segment_properties", exist_ok=True)
        with open(os.path.join(f"{self.output_directory}", "info"), "w") as f:
            f.write(json.dumps(metadata))

        segment_properties = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": skeleton_ids,
                "properties": [
                    {"id": "label", "type": "label", "values": [""] * len(skeleton_ids)}
                ],
            },
        }
        with open(f"{self.output_directory}/segment_properties/info", "w") as f:
            f.write(json.dumps(segment_properties))

    # @dask.delayed
    def process_chunked_skeleton(self, skeleton_id):
        cloudvolume_skeleton = self.__assemble_skeleton(self.dirname, skeleton_id)
        try:
            simplified_skeletor_skeleton = self.__simplify_cloudvolume_skeleton(
                cloudvolume_skeleton
            )
        except:
            with open("/nrs/cellmap/ackermand/whydask/fail.txt", "w") as f:
                f.write(
                    json.dumps("/nrs/cellmap/ackermand/whydask/fail.txt {skeleton_id}")
                )
            raise Exception(f"skeleton id {skeleton_id} failed")
        self.__write_out_skeleton(skeleton_id, simplified_skeletor_skeleton)

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
        b = db.from_sequence(skeleton_ids, npartitions=100).map(
            self.process_chunked_skeleton
        )
        with dask_util.start_dask(self.num_workers, "assemble skeletons", logger):
            with io_util.Timing_Messager("Generating assemble skeletons", logger):
                b.compute()
                # b.to_textfiles("/nrs/cellmap/ackermand/whydask/*.json.gz")
                # dask.compute(*lazy_results)

        self.__write_out_metadata(skeleton_ids)

    def get_chunked_skeleton(
        self,
        block,  #: DaskBlock,
        tmpdirname,
    ):
        os.makedirs(
            "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts",
            exist_ok=True,
        )

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
                    "scale": 1.5,
                    "const": 300,  # physical units
                    "pdrf_scale": 100000,
                    "pdrf_exponent": 4,
                    "soma_acceptance_threshold": 3500,  # physical units
                    "soma_detection_threshold": 750,  # physical units
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

                os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
                with open(f"{tmpdirname}/{id}/block_{block.index}.pkl", "wb") as fp:
                    pickle.dump(skel, fp)
        # t3 = time.time()
        # write elapsed time to file named with blockid
        # with open(
        #     f"/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts/{block.index}",
        #     "w",
        # ) as f:
        #     f.write(f"{time.time() - t0},{t1-t0},{t2-t1},{t3-t2}")

    def get_skeletons(self):
        """Get the skeletons using daisy and dask save them to disk in a temporary directory"""
        os.makedirs(self.output_directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tupdupname = "/nrs/cellmap/ackermand/tests/AubreyPresentation20240502/20240501/skeletonsWithCorrectOverlap"  # "/nrs/cellmap/ackermand/tests/20240430/skeleton5/"  # "/nrs/cellmap/ackermand/tests/AubreyPresentation20240502/20240501/skeletons"  # AubreyPresentation20240502/20240501/skeletons "/nrs/cellmap/ackermand/tests/20240430/skeleton3/"
            # with io_util.Timing_Messager("Generating chunked skeletons", logger):
            #     self.get_chunked_skeletons(tupdupname)

            self.process_chunked_skeletons(tupdupname)

    def get_meshes(self):
        """Get the meshes using dask save them to disk in a temporary directory"""
        os.makedirs(self.output_directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tupdupname = "/nrs/cellmap/ackermand/tests/tmp/dask/mesh/"
            self.get_chunked_mesh(tupdupname)
