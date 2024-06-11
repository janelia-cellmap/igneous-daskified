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


class CustomSkeleton:
    def __init__(self, vertices=[], edges=[], radii=[], polylines=[]):
        self.vertices = []
        self.edges = []
        self.radii = []
        self.polylines = []

        self.add_vertices(vertices, radii=radii)
        self.add_edges(edges)
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
            if degree == 1:
                endpoints.append(node)
            elif degree > 2:
                branchpoints.append(node)

        return branchpoints, endpoints

    @staticmethod
    def get_polylines_from_graph(g):
        branchpoints, endpoints = CustomSkeleton.find_branchpoints_and_endpoints(g)
        polylines = []
        polyline_endpoints = endpoints + branchpoints
        for idx, polyline_endpoint_1 in enumerate(polyline_endpoints[:-1]):
            # if path between node and branchpoint does not contain another branchpoint, then it is a polyline
            for polyline_endpoint_2 in polyline_endpoints[idx + 1 :]:
                path = nx.dijkstra_path(
                    g, polyline_endpoint_1, polyline_endpoint_2, weight="weight"
                )
                if len([node for node in path if node in polyline_endpoints]) == 2:
                    polylines.append(path)
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
        for idx in range(len(self.vertices)):
            g.nodes[idx]["radius"] = self.radii[idx]
        g.add_edges_from(self.edges)
        # add edge weights to the graph where weights are the distances between vertices
        for edge in self.edges:
            g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                np.array(self.vertices[edge[0]]) - np.array(self.vertices[edge[1]])
            )

        return g

    def graph_to_skeleton(self, g):
        vertices = [self.vertices[idx] for idx in g.nodes]
        radii = [self.radii[idx] for idx in g.nodes]
        edges = fastremap.remap(
            np.array(g.edges), dict(zip(list(g.nodes), list(range(len(g.nodes)))))
        )
        edges = edges.tolist()
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
        return CustomSkeleton(vertices, edges, radii, polylines)

    def prune(self, min_tick_length_nm=200):

        g = self.skeleton_to_graph()
        current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
            g, min_tick_length_nm
        )
        while current_min_tick_path:
            current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
                g, min_tick_length_nm
            )

        return self.graph_to_skeleton(g)
