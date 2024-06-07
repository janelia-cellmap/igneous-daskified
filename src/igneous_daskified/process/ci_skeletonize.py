import os
from funlib.geometry import Roi, Coordinate
from igneous_daskified.util import dask_util, io_util
from funlib.persistence import open_ds
import logging
import sys
import importlib
import os
from dataclasses import dataclass

############################################
# HACKY WAY TO IMPORT PYMESHLAB CORRECTLY
spec = importlib.util.find_spec("pymeshlab")
pymesh_lib_path = spec.origin.rsplit("/", 1)[0] + "/lib"

if os.getenv("LD_LIBRARY_PATH") == None:
    os.environ["LD_LIBRARY_PATH"] = pymesh_lib_path
    try:
        sys.stdout.flush()
        os.execl(sys.executable, sys.executable, *sys.argv)
    except OSError as e:
        print(e)
elif pymesh_lib_path not in os.getenv("LD_LIBRARY_PATH"):
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [pymesh_lib_path, os.getenv("LD_LIBRARY_PATH")]
    )
    try:
        sys.stdout.flush()
        os.execl(sys.executable, sys.executable, *sys.argv)
    except OSError as e:
        print(e)
############################################

logger = logging.getLogger(__name__)


@dataclass
class RunProperties:
    def __init__(self):
        args = io_util.parser_params()

        # Change execution directory
        self.execution_directory = dask_util.setup_execution_directory(
            args.config_path, logger
        )
        self.logpath = f"{self.execution_directory}/output.log"
        self.run_config = io_util.read_run_config(args.config_path)
        self.run_config["num_workers"] = args.num_workers


def main():
    ds = open_ds(
        "/nrs/cellmap/ackermand/cellmap/withFullPaths/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.zarr",
        "/mito/postprocessed_mito_fixed_filled_volumeFiltered",
    )
    ds = open_ds(
        "/nrs/cellmap/ackermand/cellmap/crop_jrc_mus-liver-zon-1.n5",
        "mito",
    )

    from igneous_daskified.process.skeletons import Skeletonize

    sk = Skeletonize(
        ds,
        output_directory="/nrs/cellmap/ackermand/tmp/skeletons",
        total_roi=Roi(
            (61952, 35072, 120000)[::-1],
            Coordinate(8 * 6000, 8 * 6000, 8 * 6000),
        ),
        read_write_roi=Roi((0, 0, 0), Coordinate(8 * 512, 8 * 512, 8 * 512)),
        num_workers=64,
        log_dir="/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/daisy-logs/",
    )  # , Roi((61952+212*3*8, 35072+212*3*8, 120000+212*3*8)[::-1], Coordinate(8 * 212, 8 * 212, 8 * 212)) )
    sk.get_skeletons()


def mainDaskTest():
    args = io_util.parser_params()
    # Change execution directory
    execution_directory = dask_util.setup_execution_directory(args.config_path, logger)
    logpath = f"{execution_directory}/output.log"

    # Start mesh creation
    with io_util.tee_streams(logpath):
        os.chdir(execution_directory)
        ds = open_ds(
            "/nrs/cellmap/ackermand/cellmap/withFullPaths/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.zarr",
            "/mito/postprocessed_mito_fixed_filled_volumeFiltered",
        )
        # ds = open_ds(
        #     "/nrs/cellmap/ackermand/cellmap/crop_jrc_mus-liver-zon-1.n5",
        #     "mito",
        # )

        # Restart dask to clean up cluster before multires assembly
        with dask_util.start_dask(args.num_workers, "chunked skeletons", logger):
            # Create multiresolution meshes
            with io_util.Timing_Messager("Generating chunked skeletons", logger):
                from .skeletons import Skeletonize

                sk = Skeletonize(
                    ds,
                    output_directory="/nrs/cellmap/ackermand/tmp/skeletons",
                    total_roi=Roi(
                        (61952, 35072, 120000)[::-1],
                        Coordinate(8 * 6000, 8 * 6000, 8 * 6000),
                    ),
                    read_write_roi=Roi(
                        (0, 0, 0), Coordinate(8 * 512, 8 * 512, 8 * 512)
                    ),
                    num_workers=args.num_workers,
                    log_dir="/groups/scicompsoft/home/ackermand/Programming/igneous-daskified/tmp/lsf/daisy-logs/",
                )
                sk.get_chunked_skeletons("/nrs/cellmap/ackermand/tests/tmp/dask/")


def skeletonize():
    args = io_util.parser_params()
    # Change execution directory
    execution_directory = dask_util.setup_execution_directory(args.config_path, logger)
    logpath = f"{execution_directory}/output.log"
    rp = RunProperties()
    # Start mesh creation
    with io_util.tee_streams(logpath):
        os.chdir(execution_directory)
        # ds = open_ds(
        #     "/nrs/cellmap/ackermand/cellmap/crop_jrc_mus-liver-zon-1.n5",
        #     "mito",
        # )
        ds = open_ds(
            "/nrs/cellmap/ackermand/cellmap/withFullPaths/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.zarr",
            "/mito/postprocessed_mito_fixed_filled_volumeFiltered",
        )
        from .skeletons import Skeletonize

        sk = Skeletonize(
            ds,
            output_directory="/nrs/cellmap/ackermand/AubreyPresentation20240502/848UpdateTesar/skeletons",  # WithCorrectOverlap",  # "/nrs/cellmap/ackermand/tmp/20240430/skeletons3",#"/nrs/cellmap/ackermand/AubreyPresentation20240502/20240501/skeletons2"
            # total_roi=Roi(
            #     (61952, 35072, 120000)[::-1],
            #     Coordinate(8 * 6000, 8 * 6000, 8 * 6000),
            # ),
            read_write_roi=Roi((0, 0, 0), Coordinate(8 * 848, 8 * 848, 8 * 848)),
            num_workers=args.num_workers,
        )
        sk.get_skeletons()


def meshify():
    from .meshes import Meshify

    rp = RunProperties()
    # Start mesh creation
    with io_util.tee_streams(rp.logpath):
        os.chdir(rp.execution_directory)
        meshify = Meshify(**rp.run_config)
        meshify.get_meshes()


if __name__ == "__main__":
    main()
