from unittest import TestCase, main
from graph.citygraph import CityGraph
import numpy as np


class TestBuildGraph(TestCase):

    def test_build_graph(self):
        with self.assertRaises(ValueError):
            CityGraph.from_str('A5').split(2)
        with self.assertRaises(ValueError):
            CityGraph.from_str('AB5; BC1').split(2)
        with self.assertRaises(ValueError):
            CityGraph.from_str('NOT_A_GRAPH').split(2)
        graph = CityGraph.from_str('AB1,BC7,CA12')
        self.assertTrue(np.allclose(np.array([
            [np.inf, 1, np.inf],
            [np.inf, np.inf, 7],
            [12, np.inf, np.inf],
        ]), graph.graph_weights))
        self.assertTrue(np.allclose(np.array([
            [20, 1, 8],
            [19, 20, 7],
            [12, 13, 20],
        ]), graph.shortest_weights))


class TestOriginalGraph(TestCase):

    graph = None

    def setUp(self):
        self.graph = CityGraph.from_str('AB5, BC4, CD8, DC8, DE6, AD5, CE2, EB3, AE7')

    def test_route_distances(self):
        self.assertEqual(9, self.graph.get_route_distance('A-B-C'))
        self.assertEqual(5, self.graph.get_route_distance('A-D'))
        self.assertEqual(13, self.graph.get_route_distance('A-D-C'))
        self.assertEqual(22, self.graph.get_route_distance('A-E-B-C-D'))
        self.assertEqual(42, self.graph.get_route_distance('E-B-C-D-E-B-C-D-E'))
        self.assertEqual("NO SUCH ROUTE", self.graph.get_route_distance('A-E-D'))
        self.assertEqual("NO SUCH ROUTE", self.graph.get_route_distance('A'))
        self.assertEqual("NO SUCH ROUTE", self.graph.get_route_distance('A-A'))
        self.assertEqual("NO SUCH ROUTE", self.graph.get_route_distance('A-A-A'))

    def test_num_routes(self):
        self.assertEqual(2, self.graph.get_trips_number_max_stops('C-C', 3))
        self.assertEqual(1, self.graph.get_trips_number_max_stops('A-D', 1))
        self.assertEqual(2, self.graph.get_trips_number_max_stops('A-C', 2))

        self.assertEqual(3, self.graph.get_trips_number_exact_stops('A-C', 4))
        self.assertEqual(1, self.graph.get_trips_number_exact_stops('C-E', 1))

        self.assertEqual(7, self.graph.get_trips_number_max_distance('C-C', 30))
        self.assertEqual(2, self.graph.get_trips_number_max_distance('C-C', 17))

    def test_trip_distances(self):
        self.assertEqual(9, self.graph.get_shortest_trip_distance('A-C'))
        self.assertEqual(9, self.graph.get_shortest_trip_distance('B-B'))
        self.assertEqual(5, self.graph.get_shortest_trip_distance('C-B'))
        self.assertEqual(2, self.graph.get_shortest_trip_distance('C-E'))


class TestDisconnectedGraph(TestCase):

    graph = None

    def setUp(self):
        self.graph = CityGraph.from_str('AB2 ,BC10, CB1 ,  DE11')

    def test_route_distances(self):
        self.assertEqual(12, self.graph.get_route_distance('A-B-C'))
        self.assertEqual(11, self.graph.get_route_distance('D-E'))
        self.assertEqual("NO SUCH ROUTE", self.graph.get_route_distance('A-D'))
        self.assertEqual("NO SUCH ROUTE", self.graph.get_route_distance('A-D-C'))
        self.assertEqual("NO SUCH ROUTE", self.graph.get_route_distance('A-E-B-C-D'))

    def test_num_routes(self):
        self.assertEqual(1, self.graph.get_trips_number_max_stops('C-C', 3))
        self.assertEqual(0, self.graph.get_trips_number_max_stops('A-D', 1))
        self.assertEqual(2, self.graph.get_trips_number_max_stops('A-C', 4))

        self.assertEqual(1, self.graph.get_trips_number_exact_stops('C-C', 4))
        self.assertEqual(0, self.graph.get_trips_number_exact_stops('C-E', 4))

        self.assertEqual(2, self.graph.get_trips_number_max_distance('A-B', 14))
        self.assertEqual(0, self.graph.get_trips_number_max_distance('C-D', 100))

    def test_trip_distances(self):
        self.assertEqual(2, self.graph.get_shortest_trip_distance('A-B'))
        self.assertEqual(12, self.graph.get_shortest_trip_distance('A-C'))
        self.assertEqual(11, self.graph.get_shortest_trip_distance('C-C'))
        self.assertEqual(11, self.graph.get_shortest_trip_distance('D-E'))
        self.assertEqual(None, self.graph.get_shortest_trip_distance('D-B'))
        self.assertEqual(None, self.graph.get_shortest_trip_distance('A-E'))


if __name__ == '__main__':
    main()
