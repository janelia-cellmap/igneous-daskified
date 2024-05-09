import struct
import os
import struct
import numpy as np
import json
from time import sleep
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from scipy import spatial


def write_precomputed_annotations(
    output_directory,
    annotation_type,
    ids,
    coords,
    properties_dict,
    relationships_dict=None,
):
    # write_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_directory = f"/nrs/cellmap/ackermand/neuroglancer_annotations/segmentations/mitos_with_property_{annotation_type}"
    os.system(f"rm -rf {output_directory}")
    os.makedirs(f"{output_directory}/spatial0", exist_ok=True)
    os.makedirs(f"{output_directory}/relationships", exist_ok=True)

    if annotation_type == "line":
        coords_to_write = 6
    else:
        coords_to_write = 3

    properties_values = [v for v in properties_dict.values()]
    for v in properties_values:
        assert len(v) == len(coords)

    with open(f"{output_directory}/spatial0/0_0_0", "wb") as outfile:
        total_count = len(coords)
        buf = struct.pack("<Q", total_count)

        flattened = np.column_stack((coords, *properties_values)).flatten()

        buf += struct.pack(
            f"<{(coords_to_write+len(properties_values))*total_count}f", *flattened
        )

        id_buf = struct.pack(f"<{total_count}Q", *ids)

        buf += id_buf
        outfile.write(buf)

    for relationship_id, corresponding_indices in relationships_dict.items():
        with open(
            f"{output_directory}/relationships/{relationship_id}", "wb"
        ) as outfile:
            total_count = len(corresponding_indices)
            buf = struct.pack("<Q", total_count)
            flattened = np.column_stack(
                (
                    coords[corresponding_indices],
                    *(v[corresponding_indices] for v in properties_values),
                )
            ).flatten()
            # print(total_count, flattened)
            buf += struct.pack(f"<{(coords_to_write+1)*total_count}f", *flattened)

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                f"<{total_count}Q", *ids[corresponding_indices]
            )  # so start at 1
            buf += id_buf
            outfile.write(buf)

    max_extents = coords.reshape((-1, 3)).max(axis=0) + 1
    max_extents = [int(max_extent) for max_extent in max_extents]
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"x": [1, "nm"], "y": [1, "nm"], "z": [1, "nm"]},
        "by_id": {"key": "by_id"},
        "lower_bound": [0, 0, 0],
        "upper_bound": max_extents,
        "annotation_type": annotation_type,
        "properties": [
            {"id": key, "type": "float32", "description": key}
            for key in properties_dict.keys()
        ],
        "relationships": [{"id": "cells", "key": "relationships"}],
        "spatial": [
            {
                "chunk_size": max_extents,
                "grid_shape": [1, 1, 1],
                "key": "spatial0",
                "limit": 1,
            }
        ],
    }

    with open(f"{output_directory}/info", "w") as info_file:
        json.dump(info, info_file)

    return output_directory.replace(
        "/nrs/cellmap/ackermand/",
        "precomputed://https://cellmap-vm1.int.janelia.org/nrs/ackermand/",
    )
