"""
Blockwise -> Merge -> Seam Denoise -> Global Simplify

Dependencies:
  pip install numpy trimesh pyfqmr

Typical usage:
  python block_pipeline.py \
    --blocks "blocks/*.ply" \
    --per_block_ratio 0.4 \
    --merge_weld_epsilon 1e-4 \
    --seam_angle_deg 25 \
    --k_ring 2 \
    --taubin_iters 12 \
    --final_ratio 0.3 \
    --out stitched_final.ply \
    --verbose

Notes:
- Per-block simplification preserves open boundaries (preserve_border=True).
- After stitching, we weld + denoise a seam band so the global QEM pass can
  freely collapse across former seams (preserve_border=False).
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

# ---------------------------
# pyfqmr wrapper
# ---------------------------
try:
    from pyfqmr import Simplify
except Exception as e:
    raise RuntimeError("pyfqmr is required. Install with `pip install pyfqmr`.") from e


def fqmr_simplify(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    preserve_border: bool,
    aggressiveness: int = 7,
    verbose: bool = False,
):
    """
    Fast Quadric Mesh Reduction via pyfqmr.
    - target_faces: target face count (int)
    - preserve_border: True to keep open-boundary edges fixed
    """
    simp = Simplify()

    # Handle both old and new pyfqmr API versions
    if hasattr(simp, "setMesh"):
        # Newer API version
        simp.setMesh(verts.astype(np.float64), faces.astype(np.int32))
        simp.simplify_mesh(
            target_count=int(max(4, target_faces)),
            preserve_border=preserve_border,
            aggressiveness=aggressiveness,
            verbose=verbose,
        )
        result = simp.getMesh()
    elif hasattr(simp, "set_mesh"):
        # Older API version
        simp.set_mesh(verts.astype(np.float64), faces.astype(np.int32))
        simp.simplify_mesh(
            target_count=int(max(4, target_faces)),
            preserve_border=preserve_border,
            aggressiveness=aggressiveness,
            verbose=verbose,
        )
        result = simp.get_mesh()
    else:
        raise RuntimeError(
            "Incompatible pyfqmr version. Expected 'setMesh' or 'set_mesh' method."
        )

    # Handle different return formats (some versions return normals too)
    if len(result) == 2:
        v_out, f_out = result
    elif len(result) == 3:
        v_out, f_out, _ = result  # Third element is normals, ignore it
    else:
        raise RuntimeError(f"Unexpected pyfqmr return format: got {len(result)} values")

    # pyfqmr may return float32; normalize dtypes
    return v_out.astype(np.float64), f_out.astype(np.int32)


# ---------------------------
# trimesh helpers
# ---------------------------
def load_any_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"File is not a surface mesh: {path}")
    # Ensure triangular faces
    if not m.is_watertight and m.faces.shape[1] != 3:
        m = m.triangulate()
    return m


def save_mesh(mesh: trimesh.Trimesh, path: str):
    mesh.export(path)


def repair_cleanup(mesh: trimesh.Trimesh, rezero: bool = False):
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    # Only rezero if explicitly requested (default False to preserve spatial positions)
    if rezero and len(mesh.faces) > 0 and mesh.bounds is not None:
        mesh.rezero()
    if len(mesh.faces) > 0:
        mesh.fix_normals()
    return mesh


def weld_vertices(
    mesh: trimesh.Trimesh,
    epsilon: float,
    block_size: np.ndarray = None,
    roi_offset: np.ndarray = None,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    Weld vertices within epsilon distance using an efficient grid-based approach.
    This avoids the memory overhead of KD-tree for large meshes.

    If block_size and roi_offset are provided, only weld vertices near block boundaries
    (within epsilon of a boundary), which is much more efficient for block-based meshes.
    """
    mesh = mesh.copy()
    verts_before = len(mesh.vertices)

    # Use trimesh's built-in merge_vertices for exact duplicates
    mesh.merge_vertices(merge_tex=False, merge_norm=False)

    # For spatial proximity merging within epsilon, use a grid-based approach
    # This is more memory-efficient than KD-tree for large meshes
    if len(mesh.vertices) > 0 and epsilon > 0:
        # If block size provided, only process vertices near block boundaries
        vertices = mesh.vertices

        if block_size is not None and roi_offset is not None:
            # Identify vertices near block boundaries
            # A vertex is near a boundary if its distance to the nearest block boundary <= epsilon
            rel_coords = vertices - roi_offset

            # For each axis, find distance to nearest block boundary
            # Block boundaries are at multiples of block_size
            boundary_dist = np.minimum(
                np.abs(rel_coords % block_size),
                np.abs(block_size - (rel_coords % block_size)),
            )

            # A vertex is near a boundary if min distance across all axes <= epsilon
            near_boundary = np.any(
                boundary_dist <= epsilon * 2, axis=1
            )  # 2x epsilon for safety

            if verbose:
                print(
                    f"    Only processing {near_boundary.sum()} / {len(vertices)} vertices near boundaries"
                )

            # If very few vertices near boundaries, use a simpler approach
            if near_boundary.sum() == 0:
                verts_after = len(mesh.vertices)
                if verbose:
                    print(
                        f"    Welded vertices: {verts_before} -> {verts_after} (removed {verts_before - verts_after})"
                    )
                return repair_cleanup(mesh)

            # Filter to only near-boundary vertices for processing
            boundary_indices = np.where(near_boundary)[0]
            vertices_to_process = vertices[boundary_indices]
        else:
            # Process all vertices
            boundary_indices = np.arange(len(vertices))
            vertices_to_process = vertices

            # Continue with grid-based welding only on vertices_to_process
            # Create a spatial grid with cell size = epsilon
            vmin = vertices_to_process.min(axis=0)

            # Map vertices to grid cells
            grid_coords = np.floor((vertices_to_process - vmin) / epsilon).astype(
                np.int32
            )

            # Create a dictionary mapping grid cells to local indices in vertices_to_process
            from collections import defaultdict

            grid = defaultdict(list)

            for local_idx, coord in enumerate(grid_coords):
                # Use tuple for hashability
                cell = tuple(coord)
                grid[cell].append(local_idx)

            # Union-find data structure for grouping nearby vertices
            # Maps global vertex indices to their root
            vertex_map = np.arange(len(vertices), dtype=np.int32)

            # Check each non-empty cell and its neighbors
            for cell, local_indices in grid.items():
                # Check vertices within the same cell
                for i in range(len(local_indices)):
                    for j in range(i + 1, len(local_indices)):
                        local_i, local_j = local_indices[i], local_indices[j]
                        global_i = boundary_indices[local_i]
                        global_j = boundary_indices[local_j]

                        dist = np.linalg.norm(
                            vertices_to_process[local_i] - vertices_to_process[local_j]
                        )
                        if dist <= epsilon:
                            # Union operation on global indices
                            root_i, root_j = global_i, global_j
                            while vertex_map[root_i] != root_i:
                                root_i = vertex_map[root_i]
                            while vertex_map[root_j] != root_j:
                                root_j = vertex_map[root_j]
                            if root_i < root_j:
                                vertex_map[root_j] = root_i
                            elif root_j < root_i:
                                vertex_map[root_i] = root_j

                # Check neighboring cells (26 neighbors in 3D)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                            neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                            if neighbor_cell in grid:
                                neighbor_local_indices = grid[neighbor_cell]
                                for local_i in local_indices:
                                    for local_j in neighbor_local_indices:
                                        global_i = boundary_indices[local_i]
                                        global_j = boundary_indices[local_j]

                                        dist = np.linalg.norm(
                                            vertices_to_process[local_i]
                                            - vertices_to_process[local_j]
                                        )
                                        if dist <= epsilon:
                                            # Union operation on global indices
                                            root_i, root_j = global_i, global_j
                                            while vertex_map[root_i] != root_i:
                                                root_i = vertex_map[root_i]
                                            while vertex_map[root_j] != root_j:
                                                root_j = vertex_map[root_j]
                                            if root_i < root_j:
                                                vertex_map[root_j] = root_i
                                            elif root_j < root_i:
                                                vertex_map[root_i] = root_j

            # Path compression for union-find
            for i in range(len(vertex_map)):
                root = i
                while vertex_map[root] != root:
                    root = vertex_map[root]
                vertex_map[i] = root

            # Apply the vertex mapping if we found vertices to merge
            unique_roots = np.unique(vertex_map)
            if len(unique_roots) < len(vertices):
                # Create inverse mapping
                inverse_map = np.zeros(len(vertices), dtype=np.int32)
                for new_idx, root in enumerate(unique_roots):
                    inverse_map[vertex_map == root] = new_idx

                # Create new mesh with merged vertices
                new_vertices = vertices[unique_roots]
                new_faces = inverse_map[mesh.faces.ravel()].reshape(-1, 3)
                mesh = trimesh.Trimesh(
                    vertices=new_vertices, faces=new_faces, process=False
                )

    verts_after = len(mesh.vertices)
    if verbose:
        print(
            f"    Welded vertices: {verts_before} -> {verts_after} (removed {verts_before - verts_after})"
        )
    return repair_cleanup(mesh)


def concat_meshes(meshes):
    if len(meshes) == 1:
        return meshes[0].copy()
    m = trimesh.util.concatenate(meshes)
    return repair_cleanup(m)


# ---------------------------
# seam detection & smoothing
# ---------------------------
def vertex_adjacency_list(mesh: trimesh.Trimesh):
    nbrs = [[] for _ in range(len(mesh.vertices))]
    edges = mesh.edges_unique
    for u, v in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)
    return [np.array(n, dtype=np.int32) for n in nbrs]


def expand_k_ring(seed_vertices: np.ndarray, nbrs, k: int):
    ring = set(int(i) for i in seed_vertices)
    frontier = set(ring)
    for _ in range(k):
        new_frontier = set()
        for v in frontier:
            for w in nbrs[v]:
                if w not in ring:
                    ring.add(w)
                    new_frontier.add(w)
        frontier = new_frontier
        if not frontier:
            break
    return np.fromiter(ring, dtype=np.int32)


def detect_seam_vertices(
    mesh: trimesh.Trimesh, angle_degrees: float, verbose: bool = False
) -> np.ndarray:
    # dihedral angles in radians between adjacent faces
    adj = mesh.face_adjacency
    ang = mesh.face_adjacency_angles
    if ang is None or len(ang) == 0:
        if verbose:
            print(
                "    No face adjacency information available - skipping seam detection"
            )
        return np.array([], dtype=np.int32)
    thresh = np.deg2rad(angle_degrees)
    sharp_pairs = adj[ang >= thresh]
    if sharp_pairs.size == 0:
        if verbose:
            print(f"    No seams detected with angle threshold {angle_degrees}°")
        return np.array([], dtype=np.int32)
    shared = trimesh.graph.shared_edges(mesh.faces, sharp_pairs)
    seam_verts = np.unique(shared.reshape(-1))
    if verbose:
        print(
            f"    Detected {len(seam_verts)} seam vertices (angle >= {angle_degrees}°)"
        )
    return seam_verts.astype(np.int32)


def taubin_constrained(
    mesh: trimesh.Trimesh,
    subset_idx: np.ndarray,
    lamb: float = 0.5,
    mu: float = -0.53,
    iterations: int = 10,
    verbose: bool = False,
):
    """
    Non-shrinking Taubin smoothing on a vertex subset.
    Others remain fixed. Uniform (umbrella) weights.
    """
    if len(subset_idx) == 0:
        if verbose:
            print("    No vertices to smooth - skipping Taubin smoothing")
        return

    V = mesh.vertices.view(np.ndarray).copy()
    nbrs = vertex_adjacency_list(mesh)
    subset = np.asarray(subset_idx, dtype=np.int32)

    if verbose:
        print(
            f"    Applying Taubin smoothing (λ={lamb}, μ={mu}) to {len(subset)} vertices for {iterations} iterations"
        )

    def smooth_step(Vcur, step):
        Vnew = Vcur.copy()
        for v in subset:
            n = nbrs[v]
            if n.size == 0:
                continue
            mean_nb = Vcur[n].mean(axis=0)
            Vnew[v] = Vcur[v] + step * (mean_nb - Vcur[v])
        return Vnew

    for _ in range(iterations):
        V = smooth_step(V, lamb)
        V = smooth_step(V, mu)

    mesh.vertices = V
    repair_cleanup(mesh)


# ---------------------------
# pipeline pieces
# ---------------------------
def remove_boundary_vertices(
    mesh: trimesh.Trimesh,
    voxel_size: float,
    block_size: np.ndarray = None,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    Clip mesh at all block boundaries and remove vertices on positive faces.

    This two-stage approach handles marching cubes meshes where triangles can extend
    beyond boundaries even when vertices aren't exactly on them:

    Stage 1: Clip at ALL 6 boundary planes (3 negative + 3 positive faces)
             - Retriangulates faces that cross any boundary plane
             - Creates clean cuts with vertices exactly on boundary planes
             - Ensures adjacent blocks will have matching vertices on shared boundaries

    Stage 2: Remove vertices exactly on the 3 positive faces
             - Prevents duplicate vertices when blocks are merged
             - Keeps vertices on negative faces for sharing with adjacent blocks

    Parameters:
    - mesh: Input trimesh
    - voxel_size: Size of voxels (for tolerance)
    - block_size: array of shape (3,) with block dimensions in world coordinates (required)
    - verbose: Print debug info
    """
    if len(mesh.vertices) == 0:
        return mesh

    if block_size is None:
        if verbose:
            print("    WARNING: block_size required for boundary removal. Skipping.")
        return mesh

    if verbose:
        print(
            f"    Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )
        print(
            f"    Vertex range: [{mesh.vertices.min(axis=0)}] to [{mesh.vertices.max(axis=0)}]"
        )
        print(f"    Block size: {block_size}")

    # Stage 1: Clip at all 6 boundary planes
    # This retriangulates faces that cross boundaries, creating vertices exactly on planes

    tolerance = 0.5 * np.array(voxel_size)

    # For each axis, clip at both min (0) and max (block_size) boundaries
    for axis in range(3):
        # Clip at negative face (min boundary at 0)
        plane_normal = np.zeros(3)
        plane_normal[axis] = 1.0  # Normal points in positive direction
        plane_origin = np.zeros(3)
        plane_origin[axis] = tolerance[axis]  # Slightly inside to account for FP errors

        # slice_mesh_plane keeps geometry on the side the normal points to
        # So this keeps geometry with coordinate >= tolerance (removes stuff below 0)
        mesh = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=plane_normal,
            plane_origin=plane_origin,
            cap=False,  # Don't cap the hole, we want open boundaries
        )
        if mesh is None or len(mesh.faces) == 0:
            if verbose:
                print(
                    f"    WARNING: Mesh empty after clipping axis {axis} negative face"
                )
            return trimesh.Trimesh(
                vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)
            )

        # Clip at positive face (max boundary at block_size)
        plane_normal = np.zeros(3)
        plane_normal[axis] = -1.0  # Normal points in negative direction
        plane_origin = np.zeros(3)
        plane_origin[axis] = block_size[axis] - tolerance[axis]  # Slightly inside

        # This keeps geometry with coordinate <= block_size (removes stuff beyond)
        mesh = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=False
        )
        if mesh is None or len(mesh.faces) == 0:
            if verbose:
                print(
                    f"    WARNING: Mesh empty after clipping axis {axis} positive face"
                )
            return trimesh.Trimesh(
                vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)
            )

    if verbose:
        print(
            f"    After clipping: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

    # Stage 2: Remove vertices that are beyond any of the 6 clipping planes
    # After clipping, vertices should be within [tolerance, block_size - tolerance]
    # Any vertices outside this range are artifacts/precision errors to clean up

    vertices = mesh.vertices

    # Check if vertices are outside the valid range on any axis
    # Below the negative face (< tolerance)
    below_min = vertices < tolerance
    # Beyond the positive face (> block_size - tolerance)
    beyond_max = vertices > (block_size - tolerance)

    # A vertex should be removed if it's outside valid range on ANY axis
    outside_valid_range = np.any(below_min | beyond_max, axis=1)

    if not np.any(outside_valid_range):
        if verbose:
            print("    No vertices outside valid range to remove")
        return repair_cleanup(mesh)

    n_removed = outside_valid_range.sum()
    n_total = len(vertices)

    if verbose:
        print(
            f"    Removing vertices outside valid range: {n_total} -> {n_total - n_removed} vertices ({n_removed} removed)"
        )

    # Create mask for vertices to keep (inside valid range)
    keep_mask = ~outside_valid_range
    keep_indices = np.where(keep_mask)[0]

    # Create mapping from old vertex indices to new ones
    vertex_map = np.full(len(vertices), -1, dtype=np.int32)
    vertex_map[keep_indices] = np.arange(len(keep_indices))

    # Filter vertices
    new_vertices = vertices[keep_indices]

    # Filter faces - keep only faces where all vertices are kept
    face_mask = np.all(vertex_map[mesh.faces] >= 0, axis=1)
    new_faces = vertex_map[mesh.faces[face_mask]]

    if verbose:
        print(f"    Final mesh: {len(new_vertices)} vertices, {len(new_faces)} faces")

    if len(new_faces) == 0:
        if verbose:
            print("    WARNING: All faces removed after vertex filtering!")
        return trimesh.Trimesh(
            vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)
        )

    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    return repair_cleanup(new_mesh)


def simplify_block_preserve_edges(
    mesh: trimesh.Trimesh,
    target_reduction: float,
    voxel_size: float,
    block_size: np.ndarray = None,
    aggressiveness: int = 7,
    verbose: bool = False,
) -> trimesh.Trimesh:
    # Remove positive-face boundary vertices first
    if verbose:
        print(f"  Removing positive-face boundary vertices...")
    mesh = remove_boundary_vertices(mesh, voxel_size, block_size, verbose=verbose)

    F = mesh.faces
    if len(F) == 0:
        if verbose:
            print("  WARNING: No faces left after boundary removal!")
        return mesh

    target_faces = int(max(4, (1 - target_reduction) * F.shape[0]))
    v_out, f_out = fqmr_simplify(
        mesh.vertices,
        F,
        target_faces=target_faces,
        preserve_border=True,  # No borders to preserve since we removed them all
        aggressiveness=aggressiveness,
        verbose=False,  # pyfqmr verbose is too noisy
    )
    m2 = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    return repair_cleanup(m2)


def merge_blocks(
    paths, merge_weld_epsilon: float, verbose: bool = False
) -> trimesh.Trimesh:
    if verbose:
        print(f"  Loading {len(paths)} simplified blocks...")
    meshes = [load_any_mesh(p) for p in paths]

    if verbose:
        total_verts = sum(len(m.vertices) for m in meshes)
        total_faces = sum(len(m.faces) for m in meshes)
        print(f"  Total: {total_verts} vertices, {total_faces} faces")
        print(f"  Concatenating meshes...")

    merged = concat_meshes(meshes)

    if verbose:
        print(
            f"  After concatenation: {len(merged.vertices)} vertices, {len(merged.faces)} faces"
        )
        print(f"  Welding vertices (epsilon={merge_weld_epsilon})...")

    merged = weld_vertices(merged, epsilon=merge_weld_epsilon, verbose=verbose)
    return merged


def denoise_seams_inplace(
    mesh: trimesh.Trimesh,
    seam_angle_deg: float,
    k_ring: int,
    taubin_iters: int,
    lamb: float = 0.5,
    mu: float = -0.53,
    verbose: bool = False,
):
    if verbose:
        print(f"  Detecting seams (angle threshold: {seam_angle_deg}°)...")

    seam_verts = detect_seam_vertices(
        mesh, angle_degrees=seam_angle_deg, verbose=verbose
    )
    if seam_verts.size == 0:
        if verbose:
            print("  No seams detected - skipping seam denoising")
        return

    if verbose:
        print(f"  Expanding seam band ({k_ring}-ring neighborhood)...")

    band = expand_k_ring(seam_verts, vertex_adjacency_list(mesh), k=k_ring)

    if verbose:
        print(
            f"  Seam band expanded to {len(band)} vertices ({100*len(band)/len(mesh.vertices):.1f}% of mesh)"
        )

    taubin_constrained(
        mesh, band, lamb=lamb, mu=mu, iterations=taubin_iters, verbose=verbose
    )


def global_simplify(
    mesh: trimesh.Trimesh,
    final_ratio: float,
    aggressiveness: int = 7,
    verbose: bool = False,
) -> trimesh.Trimesh:
    F = mesh.faces
    target_faces = int(max(4, final_ratio * F.shape[0]))

    if verbose:
        print(
            f"  Target: {target_faces} faces ({final_ratio*100:.1f}% of {F.shape[0]})"
        )

    v_out, f_out = fqmr_simplify(
        mesh.vertices,
        F,
        target_faces=target_faces,
        preserve_border=False,  # <- important: no fixed vertices now
        aggressiveness=aggressiveness,
        verbose=False,
    )
    m2 = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    return repair_cleanup(m2)


# ---------------------------
# I/O orchestration
# ---------------------------
def simplify_blocks_to_temp(
    block_glob: str,
    per_block_ratio: float,
    tmp_dir: Path,
    aggressiveness: int,
    block_size: np.ndarray = None,
    roi_offset: np.ndarray = None,
    halo_size: int = 1,
    verbose: bool = False,
):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_paths = sorted(glob.glob(block_glob))
    if not in_paths:
        raise FileNotFoundError(f"No block files matched: {block_glob}")

    if verbose:
        print(f"Found {len(in_paths)} block files")

    out_paths = []
    total_faces_in = 0
    total_faces_out = 0

    for i, p in enumerate(in_paths):
        m = load_any_mesh(p)
        faces_in = len(m.faces)
        total_faces_in += faces_in

        m_s = simplify_block_preserve_edges(
            m,
            per_block_ratio,
            block_size,
            roi_offset,
            aggressiveness,
            verbose=False,
        )
        faces_out = len(m_s.faces)
        total_faces_out += faces_out

        out_p = tmp_dir / f"simpl_{i:04d}{Path(p).suffix}"
        save_mesh(m_s, str(out_p))
        out_paths.append(str(out_p))

        if verbose:
            reduction = 100 * (1 - faces_out / faces_in) if faces_in > 0 else 0
            print(f"  Block {i+1}/{len(in_paths)}: {Path(p).name}")
            print(
                f"    {faces_in:,} -> {faces_out:,} faces ({reduction:.1f}% reduction)"
            )

    if verbose:
        total_reduction = (
            100 * (1 - total_faces_out / total_faces_in) if total_faces_in > 0 else 0
        )
        print(f"\n  Total per-block simplification:")
        print(
            f"    {total_faces_in:,} -> {total_faces_out:,} faces ({total_reduction:.1f}% reduction)"
        )

    return out_paths


def run_pipeline(
    block_glob: str,
    per_block_ratio: float,
    merge_weld_epsilon: float,
    seam_angle_deg: float,
    k_ring: int,
    taubin_iters: int,
    final_ratio: float,
    aggressiveness_block: int,
    aggressiveness_global: int,
    out_path: str,
    tmp_dir: str,
    block_size: np.ndarray = None,
    roi_offset: np.ndarray = None,
    halo_size: int = 1,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    verbose: bool = False,
    dry_run: bool = False,
):

    # Validation
    if per_block_ratio <= 0 or per_block_ratio > 1:
        raise ValueError(f"per_block_ratio must be in (0,1], got {per_block_ratio}")
    if final_ratio <= 0 or final_ratio > 1:
        raise ValueError(f"final_ratio must be in (0,1], got {final_ratio}")
    if per_block_ratio < final_ratio:
        print(
            f"[WARNING] per_block_ratio ({per_block_ratio}) < final_ratio ({final_ratio})"
        )
        print(
            "          Typically per_block_ratio should be >= final_ratio for better quality"
        )
    if seam_angle_deg < 0 or seam_angle_deg > 180:
        raise ValueError(f"seam_angle_deg must be in [0,180], got {seam_angle_deg}")
    if merge_weld_epsilon <= 0:
        raise ValueError(
            f"merge_weld_epsilon must be positive, got {merge_weld_epsilon}"
        )
    if k_ring < 0:
        raise ValueError(f"k_ring must be non-negative, got {k_ring}")
    if taubin_iters < 0:
        raise ValueError(f"taubin_iters must be non-negative, got {taubin_iters}")

    # Taubin parameter validation
    if not (-1 < taubin_mu < 0):
        print(f"[WARNING] Taubin mu ({taubin_mu}) should typically be in (-1, 0)")
    if not (0 < taubin_lambda < 1):
        print(
            f"[WARNING] Taubin lambda ({taubin_lambda}) should typically be in (0, 1)"
        )

    tmp = Path(tmp_dir)
    start_time = time.time()

    print("=" * 60)
    print("MESH BLOCK PIPELINE")
    print("=" * 60)
    print(f"Input glob:           {block_glob}")
    print(f"Per-block ratio:      {per_block_ratio} ({per_block_ratio*100:.1f}%)")
    if block_size is not None:
        print(f"Block size:           {block_size}")
    if roi_offset is not None:
        print(f"ROI offset:           {roi_offset}")
    if block_size is not None and roi_offset is not None:
        print(f"Halo size:            {halo_size}")
    print(f"Merge weld epsilon:   {merge_weld_epsilon}")
    print(f"Seam angle threshold: {seam_angle_deg}°")
    print(f"K-ring expansion:     {k_ring}")
    print(f"Taubin iterations:    {taubin_iters}")
    print(f"Taubin lambda/mu:     {taubin_lambda}/{taubin_mu}")
    print(f"Final ratio:          {final_ratio} ({final_ratio*100:.1f}%)")
    print(f"Output:               {out_path}")
    print(f"Temp directory:       {tmp_dir}")
    if dry_run:
        print("\n[DRY RUN MODE - will only estimate face counts]")
    print("=" * 60)

    # 1) per-block simplification (borders preserved)
    print("\n[1/4] Per-block simplification (preserving borders)...")
    simpl_paths = simplify_blocks_to_temp(
        block_glob=block_glob,
        per_block_ratio=per_block_ratio,
        tmp_dir=tmp / "blocks_simplified",
        aggressiveness=aggressiveness_block,
        block_size=block_size,
        roi_offset=roi_offset,
        halo_size=halo_size,
        verbose=verbose,
    )

    if dry_run:
        print("\n[DRY RUN] Stopping after per-block simplification estimate")
        print(
            f"[DRY RUN] Temporary simplified blocks saved to: {tmp / 'blocks_simplified'}"
        )
        return

    # 2) merge + weld
    print("\n[2/4] Merging and welding blocks...")
    merged = merge_blocks(simpl_paths, merge_weld_epsilon, verbose=verbose)
    print(
        f"  Merged mesh: {len(merged.vertices):,} vertices, {len(merged.faces):,} faces"
    )

    # 3) seam denoise (weld already done; now Taubin on seam band)
    print("\n[3/4] Denoising seams...")
    denoise_seams_inplace(
        merged,
        seam_angle_deg=seam_angle_deg,
        k_ring=k_ring,
        taubin_iters=taubin_iters,
        lamb=taubin_lambda,
        mu=taubin_mu,
        verbose=verbose,
    )

    # 4) global simplify with no fixed vertices
    print("\n[4/4] Global simplification (no border constraints)...")
    final_mesh = global_simplify(
        merged, final_ratio, aggressiveness=aggressiveness_global, verbose=verbose
    )
    print(
        f"  Final mesh: {len(final_mesh.vertices):,} vertices, {len(final_mesh.faces):,} faces"
    )

    # save
    print(f"\nSaving final mesh to: {out_path}")
    save_mesh(final_mesh, out_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Pipeline completed in {elapsed:.2f}s")
    print(f"[SUCCESS] Output written to: {out_path}")
    print("=" * 60)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Blockwise simplify (fixed edges) -> merge -> denoise seams -> global simplify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python block_pipeline.py --blocks "blocks/*.ply" --out final.ply

  # Custom parameters
  python block_pipeline.py \\
    --blocks "blocks/*.ply" \\
    --per_block_ratio 0.4 \\
    --merge_weld_epsilon 1e-4 \\
    --seam_angle_deg 35 \\
    --k_ring 3 \\
    --taubin_iters 15 \\
    --final_ratio 0.3 \\
    --out stitched_final.ply \\
    --verbose

  # Dry run to check face counts
  python block_pipeline.py --blocks "blocks/*.ply" --out test.ply --dry_run
        """,
    )
    ap.add_argument(
        "--blocks",
        type=str,
        required=True,
        help="Glob for input block meshes, e.g. 'blocks/*.ply'",
    )
    ap.add_argument(
        "--block_size",
        type=float,
        nargs=3,
        default=None,
        help="Block size as 3 values (x y z). Required for halo removal.",
    )
    ap.add_argument(
        "--roi_offset",
        type=float,
        nargs=3,
        default=None,
        help="ROI offset as 3 values (x y z). Required for halo removal.",
    )
    ap.add_argument(
        "--halo_size",
        type=int,
        default=1,
        help="Size of halo in voxels (default: 1)",
    )
    ap.add_argument(
        "--per_block_ratio",
        type=float,
        default=0.5,
        help="Per-block target face ratio (0,1]. e.g. 0.4 keeps 40%% faces (default: 0.5)",
    )
    ap.add_argument(
        "--merge_weld_epsilon",
        type=float,
        default=1e-4,
        help="Vertex welding tolerance for merging stitched blocks (default: 1e-4)",
    )
    ap.add_argument(
        "--seam_angle_deg",
        type=float,
        default=35.0,
        help="Dihedral angle threshold (deg) for seam detection (default: 35.0)",
    )
    ap.add_argument(
        "--k_ring",
        type=int,
        default=2,
        help="Adjacency ring expansion for seam band (default: 2)",
    )
    ap.add_argument(
        "--taubin_iters",
        type=int,
        default=12,
        help="Iterations of constrained Taubin smoothing on seam band (default: 12)",
    )
    ap.add_argument(
        "--taubin_lambda",
        type=float,
        default=0.5,
        help="Taubin smoothing lambda parameter (default: 0.5)",
    )
    ap.add_argument(
        "--taubin_mu",
        type=float,
        default=-0.53,
        help="Taubin smoothing mu parameter (default: -0.53)",
    )
    ap.add_argument(
        "--final_ratio",
        type=float,
        default=0.3,
        help="Final global target face ratio (0,1] (default: 0.3)",
    )
    ap.add_argument(
        "--aggressiveness_block",
        type=int,
        default=7,
        help="pyfqmr aggressiveness for per-block simplify (default: 7)",
    )
    ap.add_argument(
        "--aggressiveness_global",
        type=int,
        default=7,
        help="pyfqmr aggressiveness for global simplify (default: 7)",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the final simplified mesh (e.g., stitched_final.ply)",
    )
    ap.add_argument(
        "--tmp_dir",
        type=str,
        default="_tmp_pipeline",
        help="Directory to store intermediate meshes (default: _tmp_pipeline)",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Estimate face counts without performing full pipeline",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert block_size and roi_offset to numpy arrays if provided
    block_size = np.array(args.block_size) if args.block_size is not None else None
    roi_offset = np.array(args.roi_offset) if args.roi_offset is not None else None

    try:
        run_pipeline(
            block_glob=args.blocks,
            per_block_ratio=args.per_block_ratio,
            merge_weld_epsilon=args.merge_weld_epsilon,
            seam_angle_deg=args.seam_angle_deg,
            k_ring=args.k_ring,
            taubin_iters=args.taubin_iters,
            final_ratio=args.final_ratio,
            aggressiveness_block=args.aggressiveness_block,
            aggressiveness_global=args.aggressiveness_global,
            out_path=args.out,
            tmp_dir=args.tmp_dir,
            block_size=block_size,
            roi_offset=roi_offset,
            halo_size=args.halo_size,
            taubin_lambda=args.taubin_lambda,
            taubin_mu=args.taubin_mu,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
