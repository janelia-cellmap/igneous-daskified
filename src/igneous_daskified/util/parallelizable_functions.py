import kimimaro
import os
import time
from funlib.persistence import Array
import pickle


def get_chunked_skeleton(block, segmentation_array: Array, tmpdirname: str):
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
