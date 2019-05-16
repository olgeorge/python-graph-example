import numpy as np
import re
from typing import List, Tuple, Union, Callable
from sys import stdin

INPUT_SPLIT_REGEX = re.compile("\s*,\s*")
NO_SUCH_ROUTE = "NO SUCH ROUTE"


class CityGraph(object):
    node_indexes = None
    num_nodes = None
    graph_weights = None
    shortest_weights = None

    @staticmethod
    def from_str(string_repr: str):
        """
        Build a graph from the given string representation

        :param string_repr: String representation in the format "XYN, ZXM, XZK, ..." where X, Y, Z are single-character
        edge IDs, N, M, K are edge weights (integer, single or multi-digit)
        :return: CityGraph object
        """
        try:
            edges = [(e[0], e[1], int(e[2:])) for e in INPUT_SPLIT_REGEX.split(string_repr)]
        except Exception as e:
            raise ValueError("Cannot parse graph representation {}".format(string_repr))
        return CityGraph(edges)

    @staticmethod
    def __build_node_indexes(edges: List[Tuple[str, str, int]]) -> Tuple[dict, int]:
        """ Build a map from node ID to its index """
        node_indexes = {}
        node_ind = 0
        for source, target, _ in edges:
            if node_indexes.get(source) is None:
                node_indexes[source] = node_ind
                node_ind += 1
            if node_indexes.get(target) is None:
                node_indexes[target] = node_ind
                node_ind += 1

        return node_indexes, len(node_indexes)

    @staticmethod
    def __build_graph_weights(edges: List[Tuple[str, str, int]], node_indexes: dict, num_nodes: int) -> np.ndarray:
        """ Build graph matrix representation """
        graph_weights = np.ones((num_nodes, num_nodes), dtype=np.int) * np.inf
        for source, target, weight in edges:
            graph_weights[node_indexes[source]][node_indexes[target]] = weight
        return graph_weights

    @staticmethod
    def __build_shortest_weights(graph_weights: np.ndarray, num_nodes: int) -> np.ndarray:
        """ Creates shortest paths all-to-all matrix using Floyd-Warshall DP"""
        shortest_weights = np.copy(graph_weights)
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    weight_via_k = shortest_weights[i][k] + shortest_weights[k][j]
                    if shortest_weights[i][j] > weight_via_k:
                        shortest_weights[i][j] = weight_via_k
        return shortest_weights

    def __init__(self, edges: List[Tuple[str, str, int]]):
        self.node_indexes, self.num_nodes = self.__build_node_indexes(edges)
        self.graph_weights = self.__build_graph_weights(edges, self.node_indexes, self.num_nodes)
        self.shortest_weights = self.__build_shortest_weights(self.graph_weights, self.num_nodes)

    def get_route_distance(self, route_str: str) -> Union[int, str]:
        """
        For the given path return its weight in the graph or "NO SUCH ROUTE" if path doesn't exist

        :param route_str: string representing path in the format "X-Y-Z"
        :return: route distance
        """
        nodes = route_str.split('-')
        if len(nodes) < 2:
            return NO_SUCH_ROUTE
        weight = self.__get_path_weight(nodes, 0)
        return int(weight) if weight is not None else NO_SUCH_ROUTE

    def __get_path_weight(self, nodes: List[str], current_weight: float) -> Union[float, None]:
        if len(nodes) < 2:
            return current_weight
        weight = self.graph_weights[self.node_indexes[nodes[0]]][self.node_indexes[nodes[1]]]
        if weight == np.inf:
            return None
        return self.__get_path_weight(nodes[1:], current_weight + weight)

    def get_trips_number_max_stops(self, trips_str: str, max_stops: int) -> int:
        """
        For given stating node X and ending node Y return the number of trips between X and Y with max_stops maximum
        stops. Starting node doesn't count as a stop.

        :param trips_str: string representing start and end nodes in the format "X-Y"
        :param max_stops: number of max stops including the last node but excluding the first
        :return: number of paths
        """

        def get_edge_score(edge_from_ind, edge_to_ind):
            # Score is number of steps and each edge score is just one
            return 1

        nodes = trips_str.split('-')
        return self.__get_num_paths(self.node_indexes[nodes[0]], self.node_indexes[nodes[1]],
                                    stops_made=False,
                                    # We need less or equal to max_stops, and steps are int, so increase by one
                                    remaining_score=max_stops + 1,
                                    get_edge_score=get_edge_score)

    def get_trips_number_exact_stops(self, trips_str: str, exact_stops: int) -> int:
        """
        For given stating node X and ending node Y return the number of trips between X and Y with exactly exact_stops
        stops. Starting node doesn't count as a stop.

        :param trips_str: string representing start and end nodes in the format "X-Y"
        :param exact_stops: number of stops including the last node but excluding the first
        :return: number of paths
        """

        def get_edge_score(edge_from_ind, edge_to_ind):
            # Score is number of steps and each edge score is just one
            return 1

        def should_count_path(remaining_score):
            # Only count path when the exact number of steps is reached. Since it is initially increased by one,
            # count the path when there is only one step remaining (which won't be executed)
            return remaining_score == 1

        nodes = trips_str.split('-')
        return self.__get_num_paths(self.node_indexes[nodes[0]], self.node_indexes[nodes[1]],
                                    stops_made=False,
                                    # We need less or equal to max_stops, and steps are int, so increase by one
                                    remaining_score=exact_stops + 1,
                                    get_edge_score=get_edge_score,
                                    should_count_path=should_count_path)

    def get_trips_number_max_distance(self, trips_str: str, max_distance: float) -> int:
        """
        For given stating node X and ending node Y return the number of trips between X and Y with maximum path distance
        strictly less than max_distance.

        :param trips_str: string representing start and end nodes in the format "X-Y"
        :param max_distance: (exclusive) maximum distance of the returning path
        :return: number of paths
        """

        def get_edge_score(edge_from_ind, edge_to_ind):
            # Score is path weight and each edge score is its weight
            return self.graph_weights[edge_from_ind][edge_to_ind]

        nodes = trips_str.split('-')
        return self.__get_num_paths(self.node_indexes[nodes[0]], self.node_indexes[nodes[1]],
                                    stops_made=False,
                                    remaining_score=max_distance,
                                    get_edge_score=get_edge_score)

    def __get_num_paths(self, from_node_ind: int, to_node_ind: int,
                        stops_made: bool,
                        remaining_score: float,
                        get_edge_score: Callable,
                        should_count_path: Callable=None) -> int:
        """
        For a given stating node index X and ending node index Y return the number of paths in the graph
        where score of the path is strictly less than remaining_score. Score of each edge each is determined by
        get_edge_score, additional termination condition is supplied via should_count_path. Different paths may have
        common edges.

        :param from_node_ind: Starting path node index in self.graph_weights
        :param to_node_ind: Ending path node index in self.graph_weights
        :param stops_made: Recursive argument, always pass False
        :param remaining_score: The method will return paths with scores strictly less than remaining_score
        :param get_edge_score: method accepting edge_from_ind, edge_to_ind and returning score for that node
        With each step get_edge_score determines how much the remaining_score decreases
        For example, to count path weights divided by half, use:
        lambda edge_from_ind, edge_to_ind: self.graph_weights[edge_from_ind][edge_to_ind]
        :param should_count_path: method accepting remaining_score and returning True if the path should be counted
        towards to total amount given the remaining_score or not
        :return: number of paths
        """
        if remaining_score <= 0:
            return 0
        outgoing_node_inds = np.where(self.graph_weights[from_node_ind] != np.inf)[0]
        if not len(outgoing_node_inds):
            return 0
        num_paths = 1 if (from_node_ind == to_node_ind and stops_made and
                          (not should_count_path or should_count_path(remaining_score))) else 0
        for e in outgoing_node_inds:
            num_paths += self.__get_num_paths(e, to_node_ind, True, remaining_score -
                                              get_edge_score(from_node_ind, e), get_edge_score, should_count_path)
        return num_paths

    def get_shortest_trip_distance(self, trips_str: str) -> int:
        from_node, to_node = trips_str.split('-')
        shortest_path = self.shortest_weights[self.node_indexes[from_node]][self.node_indexes[to_node]]
        return int(shortest_path) if shortest_path != np.inf else None


def main():
    default_input = 'AB5, BC4, CD8, DC8, DE6, AD5, CE2, EB3, AE7'
    print("Input graph [default `{}`]:".format(default_input))
    line = stdin.readline().replace("\n", "")
    graph = CityGraph.from_str(line or default_input)
    print("Output #1:", graph.get_route_distance('A-B-C'))
    print("Output #2:", graph.get_route_distance('A-D'))
    print("Output #3:", graph.get_route_distance('A-D-C'))
    print("Output #4:", graph.get_route_distance('A-E-B-C-D'))
    print("Output #5:", graph.get_route_distance('A-E-D'))
    print("Output #6:", graph.get_trips_number_max_stops('C-C', 3))
    print("Output #7:", graph.get_trips_number_exact_stops('A-C', 4))
    print("Output #8:", graph.get_shortest_trip_distance('A-C'))
    print("Output #9:", graph.get_shortest_trip_distance('B-B'))
    print("Output #10:", graph.get_trips_number_max_distance('C-C', 30))


if __name__ == '__main__':
    main()
