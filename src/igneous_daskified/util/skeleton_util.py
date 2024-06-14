from dataclasses import dataclass
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

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Source:
    vertex_attributes = []


class CustomSkeleton:
    def __init__(self, vertices=[], edges=[], radii=None, polylines=[]):
        self.vertices = []
        self.edges = []
        self.radii = []
        self.polylines = []

        self.add_vertices(vertices, radii=radii)
        self.add_edges(edges)
        if not polylines:
            g = self.skeleton_to_graph()
            polylines = self.get_polylines_positions_from_graph(g)
        self.add_polylines(polylines)

    def _get_vertex_index(self, vertex):
        if type(vertex) is not tuple:
            vertex = tuple(vertex)
        return self.vertices.index(tuple(vertex))

    def add_vertex(self, vertex, radius=None):
        if vertex not in self.vertices:
            self.vertices.append(vertex)
            if radius:
                self.radii.append(radius)

    def add_vertices(self, vertices, radii):
        if radii:
            for vertex, radius in zip(vertices, radii):
                self.add_vertex(vertex, radius)
        else:
            for vertex in vertices:
                self.add_vertex(vertex)

    def add_edge(self, edge):
        if type(edge[0]) != int:
            # then edges are coordinates, so need to get corresponding radii
            edge_start_id = self._get_vertex_index(edge[0])
            edge_end_id = self._get_vertex_index(edge[1])
            edge = (edge_start_id, edge_end_id)
        self.edges.append(edge)

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)

    def add_polylines(self, polylines):
        for polyline in polylines:
            self.add_polyline(polyline)

    def add_polyline(self, polyline):
        self.polylines.append(polyline)

    def simplify(self, tolerance_nm=200):
        # use polylines
        vertices = []
        radii = []
        edges = []
        simplified_polylines = []
        for polyline in self.polylines:
            simplified_polyline = rdp(polyline, epsilon=tolerance_nm)
            for vertex in simplified_polyline:
                vertices.append(tuple(vertex))
                if self.radii:
                    radii.append(self.radii[self._get_vertex_index(tuple(vertex))])
            simplified_polylines.append(simplified_polyline)

            edges.extend(list(zip(simplified_polyline, simplified_polyline[1:])))

        simplified_skeleton = CustomSkeleton(
            vertices, edges, radii, simplified_polylines
        )
        return simplified_skeleton

    @staticmethod
    def find_branchpoints_and_endpoints(graph):
        branchpoints = []
        endpoints = []

        for node in graph.nodes:
            degree = graph.degree[node]
            if degree <= 1:
                endpoints.append(node)
            elif degree > 2:
                branchpoints.append(node)

        return branchpoints, endpoints

    @staticmethod
    def get_polylines_from_graph(g):
        polylines = []
        edges = list(g.edges)
        g_copy = g.copy()
        branchpoints, _ = CustomSkeleton.find_branchpoints_and_endpoints(g_copy)
        g_copy.remove_nodes_from(branchpoints)
        if len(branchpoints) > 0:
            for component in nx.connected_components(g_copy):
                g_sub = g_copy.subgraph(component)
                _, current_polyline_endpoints = (
                    CustomSkeleton.find_branchpoints_and_endpoints(g_sub)
                )
                if len(current_polyline_endpoints) > 2:
                    raise Exception(
                        "Something went wrong, there should be at most 2 endpoints in a connected component"
                    )

                if len(g_sub.nodes) == 1:
                    # then contains a single node
                    polyline = list(g_sub.nodes)
                else:
                    polyline = nx.dijkstra_path(
                        g_sub,
                        current_polyline_endpoints[0],
                        current_polyline_endpoints[1],
                        weight="weight",
                    )
                # get edges with endpoints in them
                edges_with_endpoints = [
                    tuple(sorted(edge))
                    for edge in edges
                    if (
                        edge[0] in current_polyline_endpoints
                        or edge[1] in current_polyline_endpoints
                    )
                ]

                prepended = False
                appended = False
                # sort edges by smallest node idx first
                polyline_edges = list(zip(polyline[:-1], polyline[1:]))
                polyline_edges = [tuple(sorted(edge)) for edge in polyline_edges]
                for edge in edges_with_endpoints:
                    if (
                        not prepended
                        and edge not in polyline_edges
                        and polyline[0] in edge
                    ):
                        if polyline[0] == edge[0]:
                            polyline.insert(0, edge[1])
                        else:
                            polyline.insert(0, edge[0])
                        polyline_edges.append(edge)
                        prepended = True
                    if (
                        not appended
                        and edge not in polyline_edges
                        and polyline[-1] in edge
                    ):
                        if polyline[-1] == edge[0]:
                            polyline.append(edge[1])
                        else:
                            polyline.append(edge[0])
                        polyline_edges.append(edge)
                        appended = True
                polylines.append(polyline)
        else:
            # then contains separate part(s)
            for component in nx.connected_components(g):
                g_sub = g.subgraph(component)
                path = nx.eulerian_path(g_sub)
                polyline = [node for node, _ in path]
                polyline.append(polyline[0])
                polylines.append(polyline)

        # if two branchpoints separated by a single edge, teh above would ignore them, so add them back in
        all_edges = []
        for polyline in polylines:
            all_edges.extend(list(zip(polyline[:-1], polyline[1:])))
        all_edges = [tuple(sorted(edge)) for edge in all_edges]
        (polylines)
        for edge in g.edges:
            if edge not in all_edges:
                polylines.append([edge[0], edge[1]])
        return polylines

    @staticmethod
    def remove_smallest_qualifying_branch(g, min_tick_length_nm=200):
        # get endpoints and branchpoints from g
        branchpoints, endpoints = CustomSkeleton.find_branchpoints_and_endpoints(g)
        current_min_tick_length_nm = np.inf
        current_min_tick_path = None

        for endpoint in endpoints:
            for branchpoint in branchpoints:
                path = nx.dijkstra_path(g, endpoint, branchpoint, weight="weight")
                path_length_nm = nx.shortest_path_length(
                    g, endpoint, branchpoint, weight="weight"
                )
                if (
                    path_length_nm < min_tick_length_nm
                    and path_length_nm < current_min_tick_length_nm
                    and len(path) < g.number_of_nodes()
                ):
                    current_min_tick_length_nm = path_length_nm
                    current_min_tick_path = path
        if current_min_tick_path:
            g.remove_edges_from(
                list(zip(current_min_tick_path[:-1], current_min_tick_path[1:]))
            )
            g.remove_nodes_from(list(nx.isolates(g)))
        return current_min_tick_path, g

    def skeleton_to_graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(len(self.vertices)))
        # add radii as properties to the nodes
        if self.radii:
            for idx in range(len(self.vertices)):
                g.nodes[idx]["radius"] = self.radii[idx]
        g.add_edges_from(self.edges)
        # add edge weights to the graph where weights are the distances between vertices
        for edge in self.edges:
            g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                np.array(self.vertices[edge[0]]) - np.array(self.vertices[edge[1]])
            )

        return g

    def get_polylines_positions_from_graph(self, g):
        polylines = []
        for polyline_by_vertex_id in CustomSkeleton.get_polylines_from_graph(g):
            polylines.append(
                np.array(
                    [
                        np.array(self.vertices[vertex_id])
                        for vertex_id in polyline_by_vertex_id
                    ]
                )
            )
        return polylines

    def graph_to_skeleton(self, g):
        vertices = [self.vertices[idx] for idx in g.nodes]
        radii = [self.radii[idx] for idx in g.nodes]
        edges = fastremap.remap(
            np.array(g.edges), dict(zip(list(g.nodes), list(range(len(g.nodes)))))
        )
        edges = edges.tolist()
        polylines = self.get_polylines_positions_from_graph(g)
        return CustomSkeleton(vertices, edges, radii, polylines)

    def prune(self, min_branch_length_nm=200):

        g = self.skeleton_to_graph()
        current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
            g, min_branch_length_nm
        )
        while current_min_tick_path:
            current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
                g, min_branch_length_nm
            )

        return self.graph_to_skeleton(g)

    def write_neuroglancer_skeleton(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            skel = NeuroglancerSkeleton(
                self.vertices, self.edges, vertex_attributes=None
            )
            encoded = skel.encode(Source())
            f.write(encoded)
