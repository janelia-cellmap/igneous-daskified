from dataclasses import dataclass
import socket

import daisy
from funlib.persistence import Array, open_ds
from funlib.geometry import Roi, Coordinate
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
from dask.distributed import Client, progress
from dataclasses import dataclass
from funlib.geometry import Roi
from yaml.loader import SafeLoader
import yaml
import time

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_chunked_skeleton(
    segmentation_array,
    block,  #: DaskBlock,
    tmpdirname,
):
    os.makedirs(
        "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts",
        exist_ok=True,
    )

    segmentation_block = segmentation_array.to_ndarray(block.roi, fill_value=0)
    # raise Exception(dask.config.config)
    # swap byte order in place if little endian
    if segmentation_block.dtype.byteorder == ">":
        segmentation_block = segmentation_block.newbyteorder().byteswap()

    t0 = time.time()
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
        parallel_chunk_size=10,  # how many skeletons to process before updating progress bar
        in_place=True,
    )
    t1 = time.time()
    for id, skel in skels.items():
        skel.vertices += block.roi.offset
        os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
        with open(f"{tmpdirname}/{id}/block_{block.index}.pkl", "wb") as fp:
            pickle.dump(skel, fp)

    # write elapsed time to file named with blockid
    with open(
        f"/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts/{block.index}",
        "w",
    ) as f:
        f.write(str(t1 - t0))
        f.write(str(time.time() - t1))


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


def create_blocks(roi: Roi, ds: Array, block_size=None):
    roi = roi.snap_to_grid(ds.chunk_shape * ds.voxel_size)
    if not block_size:
        block_size = ds.chunk_shape * ds.voxel_size

    block_rois = []
    index = 0
    for z in range(roi.get_begin()[2], roi.get_end()[2], block_size[2]):
        for y in range(roi.get_begin()[1], roi.get_end()[1], block_size[1]):
            for x in range(roi.get_begin()[0], roi.get_end()[0], block_size[0]):
                block_rois.append(DaskBlock(index, Roi((x, y, z), block_size)))
                index += 1
    print(len(block_rois))
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

    def __get_chunked_skeleton_daisy(
        self,
        block,
        tmpdirname,
    ):
        os.makedirs(
            "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daisy128",
            exist_ok=True,
        )
        t0 = time.time()
        segmentation_block = self.segmentation_array.to_ndarray(
            block.read_roi, fill_value=0
        )

        # # swap byte order in place if little endian
        if segmentation_block.dtype.byteorder == ">":
            segmentation_block = segmentation_block.newbyteorder().byteswap()

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

        _, block_id = block.block_id
        for id, skel in skels.items():
            skel.vertices += block.read_roi.offset
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
            with open(f"{tmpdirname}/{id}/block_{block_id}.pkl", "wb") as fp:
                pickle.dump(skel, fp)

        with open(
            f"/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daisy128/{block_id}",
            "w",
        ) as f:
            f.write(str(time.time() - t0))

    # @dask.delayed
    def __get_chunked_skeleton(
        self,
        block: DaskBlock,
        tmpdirname,
    ):
        os.makedirs(
            "/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts",
            exist_ok=True,
        )
        t0 = time.time()
        segmentation_block = self.segmentation_array.to_ndarray(block.roi, fill_value=0)
        # raise Exception(dask.config.config)
        # swap byte order in place if little endian
        if segmentation_block.dtype.byteorder == ">":
            segmentation_block = segmentation_block.newbyteorder().byteswap()

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
            in_place=True,
        )

        for id, skel in skels.items():
            skel.vertices += block.roi.offset
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)
            with open(f"{tmpdirname}/{id}/block_{block.index}.pkl", "wb") as fp:
                pickle.dump(skel, fp)

        # write elapsed time to file named with blockid
        with open(
            f"/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/timings/daskAttempts/{block.index}",
            "w",
        ) as f:
            f.write(str(time.time() - t0))

    def get_chunked_skeletons(self, dirname):
        blocks = create_blocks(
            self.total_roi, self.segmentation_array, self.read_write_roi.shape
        )
        # config = dask.config.config
        # print(config)
        # print(config["admin"])
        # config["distributed"]["admin"]["tick"]["limit"] = "3h"
        # # config["distributed"]["admin"]["system-monitor"]["gil"]["enabled"] = False
        # dask.config.update(dask.config.config, config)
        # print(config)
        # with open("/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/src/igneous_daskified/process/local-config.yaml") as f:
        #     config = yaml.load(f, Loader=SafeLoader)
        #     dask.config.update(dask.config.config, config)
        # from dask.distributed import LocalCluster

        # with Client(
        #     LocalCluster(n_workers=self.num_workers, threads_per_worker=1)
        # ) as client:  # , memory_limit='32GB')

        #     dashboard_link = client.cluster.dashboard_link
        #     print(
        #         dashboard_link.replace(
        #             "127.0.0.1", socket.gethostbyname_ex(socket.gethostname())[0]
        #         )
        #     )
        lazy_results = []
        for block in blocks:
            lazy_results.append(
                dask.delayed(get_chunked_skeleton)(
                    self.segmentation_array, block, dirname
                )
            )
        # client.rebalance()
        dask.compute(*lazy_results)

    def __assemble_skeleton(self, dirname, skeleton_id):
        block_skeletons = []
        for skeleton_file in os.listdir(f"{dirname}/{skeleton_id}"):
            with open(f"{dirname}/{skeleton_id}/{skeleton_file}", "rb") as f:
                block_skeleton = pickle.load(f)
                block_skeletons.append(block_skeleton)
                if skeleton_id == "14654803557897":
                    print(len(block_skeleton.vertices))
            cloudvolume_skeleton = CloudVolumeSkeleton.simple_merge(block_skeletons)
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
                np.fliplr(skeletor_skeleton.vertices), skeletor_skeleton.edges
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

    @dask.delayed
    def __process_chunked_skeleton(self, dirname, skeleton_id):
        cloudvolume_skeleton = self.__assemble_skeleton(dirname, skeleton_id)
        simplified_skeletor_skeleton = self.__simplify_cloudvolume_skeleton(
            cloudvolume_skeleton
        )
        self.__write_out_skeleton(skeleton_id, simplified_skeletor_skeleton)

    def process_chunked_skeletons(self, dirname):
        """Process the chunked skeletons in parallel using dask"""

        with Client(
            threads_per_worker=1, n_workers=1
        ) as client:  # , memory_limit='32GB')
            client.cluster.scale(self.num_workers)
            dashboard_link = client.cluster.dashboard_link
            print(
                dashboard_link.replace(
                    "127.0.0.1", socket.gethostbyname_ex(socket.gethostname())[0]
                )
            )
            lazy_results = []
            skeleton_ids = os.listdir(dirname)
            for skeleton_id in skeleton_ids:
                lazy_results.append(
                    self.__process_chunked_skeleton(dirname, skeleton_id)
                )
            dask.compute(*lazy_results)

        self.__write_out_metadata(skeleton_ids)

    def get_skeletons(self):
        os.makedirs(self.output_directory, exist_ok=True)
        """Get the skeletons using daisy and dask save them to disk in a temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tupdupname = "/nrs/cellmap/ackermand/tests/tmp/"
            task = daisy.Task(
                total_roi=self.total_roi,
                read_roi=self.read_write_roi,
                write_roi=self.read_write_roi,
                process_function=lambda b: self.__get_chunked_skeleton_daisy(
                    b,
                    tupdupname,  # tmpdirname
                ),
                num_workers=self.num_workers,
                task_id="block_processing",
                fit="shrink",
            )
            # add export of scores
            daisy.run_blockwise([task])
            # self.get_chunked_skeletons(tupdupname)
            # self.process_chunked_skeletons(tupdupname)
