import os
from funlib.geometry import Roi, Coordinate
from igneous_daskified.util import dask_util, io_util
from funlib.persistence import open_ds
import logging

logger = logging.getLogger(__name__)


def main():
    ds = open_ds(
        "/nrs/cellmap/ackermand/cellmap/withFullPaths/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1_postprocessed.zarr",
        "/mito/postprocessed_mito_fixed_filled_volumeFiltered",
    )
    ds = open_ds(
        "/nrs/cellmap/ackermand/cellmap/crop_jrc_mus-liver-zon-1.n5",
        "mito",
    )

    from .skeletons import Skeletonize

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
        ds = open_ds(
            "/nrs/cellmap/ackermand/cellmap/crop_jrc_mus-liver-zon-1.n5",
            "mito",
        )

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
                sk.get_chunked_skeletons("/nrs/cellmap/ackermand/tests/tmp/")


if __name__ == "__main__":
    main()
