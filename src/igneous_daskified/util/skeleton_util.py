# %%
from dataclasses import dataclass
import struct
import numpy as np
import os
import logging
from neuroglancer.skeleton import Skeleton as NeuroglancerSkeleton
import fastremap
import networkx as nx
from pybind11_rdp import rdp

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
        if type(vertex) is not tuple:
            vertex = tuple(vertex)

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
        self.vertices = self.vertices

    def add_edge(self, edge):
        if not isinstance(edge[0], (int, np.integer)):
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
        # print("graph.nodes", len(graph.nodes))
        for node in graph.nodes:
            degree = graph.degree[node]
            if degree <= 1:
                endpoints.append(node)
            elif degree > 2:
                branchpoints.append(node)

        return branchpoints, endpoints

    @staticmethod
    def get_polyline_from_subgraph(subgraph, all_graph_edges):
        if len(subgraph.nodes) == 1:
            # then contains a single node
            node = list(subgraph.nodes)[0]
            path = [(node, node)]
        else:
            path = list(nx.eulerian_path(subgraph))
        start_node = path[0][0]
        end_node = path[-1][-1]
        prepended = False
        appended = False
        output_path = path.copy()
        for edge in all_graph_edges:
            if edge not in path and edge[::-1] not in path:
                if start_node in edge and not prepended:
                    output_path.insert(0, edge if edge[1] == start_node else edge[::-1])
                    prepended = True
                elif end_node in edge and not appended:
                    output_path.append(edge if edge[0] == end_node else edge[::-1])
                    appended = True

        if len(output_path) == 1 and output_path[0][0] == output_path[0][1]:
            # then single node
            polyline = [output_path[0][0]]
        else:
            # remove edge in output_path if the start and endpoints are the same, which we added above if there is only one endpoint
            for edge in output_path:
                if edge[0] == edge[1]:
                    output_path.remove(edge)

            polyline = [node for node, _ in output_path]
            polyline.append(output_path[-1][-1])

        return polyline

    @staticmethod
    def get_polylines_from_graph(g):
        polylines = []
        edges = list(g.edges)
        g_copy = g.copy()
        branchpoints, _ = CustomSkeleton.find_branchpoints_and_endpoints(g_copy)
        g_copy.remove_nodes_from(branchpoints)
        # if len(branchpoints) > 0:
        from tqdm import tqdm

        for component in tqdm(nx.connected_components(g_copy)):
            g_sub = g_copy.subgraph(component)
            polyline = CustomSkeleton.get_polyline_from_subgraph(g_sub, edges)
            polylines.append(polyline)
        # if two branchpoints separated by a single edge, teh above would ignore them, so add them back in
        all_edges = []
        for polyline in polylines:
            all_edges.extend(list(zip(polyline[:-1], polyline[1:])))
        all_edges = [tuple(sorted(edge)) for edge in all_edges]
        for edge in edges:
            if edge not in all_edges:
                polylines.append([edge[0], edge[1]])
        return polylines

    @staticmethod
    def remove_smallest_qualifying_branch(g, min_branch_length_nm=200):
        # get endpoints and branchpoints from g
        branchpoints, _ = CustomSkeleton.find_branchpoints_and_endpoints(g)
        current_min_branch_length_nm = np.inf
        current_min_branch_path = None
        # for endpoint in endpoints:
        #     for branchpoint in branchpoints:
        #         path_length_nm = nx.shortest_path_length(
        #             g, endpoint, branchpoint, weight="weight"
        #         )
        #         if (
        #             path_length_nm < min_tick_length_nm
        #             and path_length_nm < current_min_tick_length_nm
        #         ):
        #             path = nx.dijkstra_path(g, endpoint, branchpoint, weight="weight")
        #             if len(path) < g.number_of_nodes():
        #                 current_min_tick_length_nm = path_length_nm
        #                 current_min_tick_path = path

        polylines_by_vertex_id = CustomSkeleton.get_polylines_from_graph(g)

        for polyline_by_vertex_id in polylines_by_vertex_id:
            if (polyline_by_vertex_id[0] in branchpoints) ^ (
                polyline_by_vertex_id[-1] in branchpoints
            ):
                polyline_length_nm = 0
                for v1, v2 in zip(
                    polyline_by_vertex_id[:-1], polyline_by_vertex_id[1:]
                ):
                    polyline_length_nm += np.linalg.norm(
                        np.array(g.nodes[v1]["position_nm"])
                        - np.array(g.nodes[v2]["position_nm"])
                    )
                if (
                    polyline_length_nm < min_branch_length_nm
                    and polyline_length_nm < current_min_branch_length_nm
                ):
                    if len(set(polyline_by_vertex_id)) < g.number_of_nodes():
                        current_min_branch_length_nm = polyline_length_nm
                        current_min_branch_path = polyline_by_vertex_id

        if current_min_branch_path:
            g.remove_edges_from(
                list(zip(current_min_branch_path[:-1], current_min_branch_path[1:]))
            )
            g.remove_nodes_from(list(nx.isolates(g)))
        return current_min_branch_path, g

    def skeleton_to_graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(len(self.vertices)))
        for idx in range(len(self.vertices)):
            g.nodes[idx]["position_nm"] = self.vertices[idx]
            # add radii as properties to the nodes
            if self.radii:
                g.nodes[idx]["radius"] = self.radii[idx]

        g.add_edges_from(self.edges)
        # add edge weights to the graph where weights are the distances between vertices
        for edge in self.edges:
            g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                np.array(self.vertices[edge[0]]) - np.array(self.vertices[edge[1]])
            )

        return g

    @staticmethod
    def get_polylines_positions_from_graph(g):
        polylines_by_vertex_id = CustomSkeleton.get_polylines_from_graph(g)
        polylines = []
        for polyline_by_vertex_id in polylines_by_vertex_id:
            polylines.append(
                np.array(
                    [
                        np.array(g.nodes[vertex_id]["position_nm"])
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
        polylines = CustomSkeleton.get_polylines_positions_from_graph(g)
        return CustomSkeleton(vertices, edges, radii, polylines)

    def prune(self, min_branch_length_nm=200):
        if len(self.vertices) == 1:
            return self
        g = self.skeleton_to_graph()
        current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
            g, min_branch_length_nm
        )
        while current_min_tick_path:
            current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
                g, min_branch_length_nm
            )

        return self.graph_to_skeleton(g)

    @staticmethod
    def lineseg_dists(p, a, b):
        # https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
        # Handle case where p is a single point, i.e. 1d array.
        p = np.atleast_2d(p)

        # TODO for you: consider implementing @Eskapp's suggestions
        if np.all(a == b):
            return np.linalg.norm(p - a, axis=1)

        # normalized tangent vector
        d = np.divide(b - a, np.linalg.norm(b - a))

        # signed parallel distance components
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros_like(s)])

        # perpendicular distance component, as before
        # note that for the 3D case these will be vectors
        c = np.linalg.norm(np.cross(p - a, d), axis=1)

        # use hypot for Pythagoras to improve accuracy
        return np.hypot(h, c)

    def write_neuroglancer_skeleton(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            skel = NeuroglancerSkeleton(
                self.vertices, self.edges, vertex_attributes=None
            )
            encoded = skel.encode(Source())
            f.write(encoded)

    @staticmethod
    def read_neuroglancer_skeleton(path):
        source_info = Source()
        with open(path, "rb") as f:
            data = f.read()

        offset = 0

        # 1) Read number of vertices (n_vertices) and number of edges (n_edges).
        n_vertices, n_edges = struct.unpack_from("<II", data, offset)
        offset += 8

        # 2) Decode vertex_positions.
        num_vp_values = n_vertices * 3
        vertex_positions = np.frombuffer(
            data, dtype="<f4", count=num_vp_values, offset=offset
        )
        offset += vertex_positions.nbytes
        # Reshape to (n_vertices, vp_dim).
        vertex_positions = vertex_positions.reshape((n_vertices, 3))

        # 3) Decode edges.
        num_edge_values = n_edges * 2
        edges = np.frombuffer(data, dtype="<u4", count=num_edge_values, offset=offset)
        offset += edges.nbytes
        # Reshape to (n_edges, edges_dim).
        edges = edges.reshape((n_edges, 2))

        # 4) Decode vertex_attributes (if any).
        decoded_attributes = {}
        if source_info.vertex_attributes:
            for attr_name, (
                attr_dtype,
                num_components,
            ) in source_info.vertex_attributes.items():
                # We expect shape = (n_vertices, num_components).
                expected_size = n_vertices * num_components
                attribute = np.frombuffer(
                    data,
                    dtype=attr_dtype.newbyteorder("<"),
                    count=expected_size,
                    offset=offset,
                )
                offset += attribute.nbytes
                attribute = attribute.reshape((n_vertices, num_components))
                decoded_attributes[attr_name] = attribute
        # 5) Package results.
        return vertex_positions, edges
