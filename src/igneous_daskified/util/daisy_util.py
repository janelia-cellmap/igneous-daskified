from funlib.geometry import Roi
from funlib.persistence import open_ds
import click
import logging
import subprocess
import daisy
import os
import kimimaro
import pickle
import time
import yaml


def get_chunked_skeleton_daisy(
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


function_mappings = {
    "get_chunked_skeleton_daisy": get_chunked_skeleton_daisy,
}


def load_dataset(ds_name: str):
    # split on filetype
    filename, dataset_name = ds_name.rsplit(".zarr", 1).rsplit(".n5", 1)
    return open_ds(filename, dataset_name)


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option("-f", "--fnc-name", type=str)
@click.option("-b", "--billing", type=str)
@click.option("-n", "--cores-per-worker", type=int)
def spawn_worker(
    fnc_name,
    cores_per_worker,
    billing,
    local=True,
):
    name = "tmp"

    def run_worker():
        if local:
            subprocess.run(
                [
                    "igneous-daskified",
                    fnc_name,
                    "-n",
                    f"{name}",
                ]
            )
        else:
            subprocess.run(
                [
                    "bsub",
                    "-P",
                    billing,
                    "-J",
                    "igneous_daisified",
                    "-q",
                    "local",
                    "-n",
                    cores_per_worker,
                    "-o",
                    f"prediction_logs/{name}.out",
                    "-e",
                    f"prediction_logs/{name}.err",
                    "python",
                    "scripts/predict_worker.py",
                    "fnc_name",
                    "-n",
                    f"{name}",
                ]
            )

    return run_worker


@cli.command()
@click.option("-c", "--config-file", type=str, required=True)
def submit_daisy(
    config_file,
):

    with open(config_file) as f:
        config = yaml.safe_load(f)

    segmentation_ds = load_dataset(config["segmentation_ds"])
    function_name = config["function_name"]
    num_workers = config["num_workers"]
    cores_per_worker = config["cores_per_worker"]
    billing = config["billing"]
    local = config["local"]

    if "total_read_roi" not in config:
        total_read_roi = Roi(
            config["total_read_roi"]["start"], config["total_roi"]["extent"]
        )
    else:
        total_read_roi = segmentation_ds.roi

    if "read_roi" not in config:
        read_roi = Roi(config["read_roi"]["start"], config["read_roi"]["extent"])
    else:
        read_roi = Roi(
            (0, 0, 0), segmentation_ds.chunk_shape * segmentation_ds.voxel_size
        )
    write_roi = read_roi
    task = daisy.Task(
        "server_task",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=spawn_worker(function_name, cores_per_worker, billing, local),
        check_function=None,
        read_write_conflict=False,
        fit="overhang",
        num_workers=num_workers,
        max_retries=2,
        timeout=None,
    )

    daisy.run_blockwise([task])


@cli.command()
@click.option("-f", "--fnc-name", type=str, required=True)
@click.option("-c", "--config-file", type=str, required=True)
def process_function_blockwise(fnc_name, config_file):
    fnc = function_mappings[fnc_name]
    client = daisy.Client()
    while True:
        with client.acquire_block() as block:
            if block is None:
                break
            fnc(block, config_file)
