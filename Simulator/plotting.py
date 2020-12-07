from itertools import permutations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Colormap
from re import findall

EPSILON = 1E-12
WEIGHT_SCALING = 2  # improve visibility
COLOR_RANGE = 256
COLOR_MIDPOINT = COLOR_RANGE / 2 - 1


class CoefficientGraph:

    def __init__(self, coefficients: np.array, concentrations: np.array, edge_threshold: float = 0.0,
                 color_map="RdYlGn"):
        """
        Initialize a graph
        :param coefficients: the real-valued coefficients to make a graph of
        :param edge_threshold: the percentile that edges must be above to be shown
        :param color_map: the name of a color map (https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
        """
        # public
        self.coefficients = coefficients
        self.edge_threshold = edge_threshold
        self.concentrations = concentrations
        self.color_map = color_map

        # private
        self._validate_input()
        self._G = nx.DiGraph()
        self._weights = list()
        self._colors = list()
        self._cm = plt.get_cmap(self.color_map)
        self._make_graph()

    def _validate_input(self):
        assert isinstance(self.coefficients, np.ndarray)  # np
        assert np.issubdtype(self.coefficients.dtype, np.float64)  # floats
        assert len(self.coefficients) == len(self.coefficients[0])  # square

        assert isinstance(self.concentrations, np.ndarray)  # np
        assert np.issubdtype(self.concentrations.dtype, np.float64)  # floats
        assert len(self.coefficients) == len(self.concentrations)

        try:
            assert isinstance(plt.get_cmap(self.color_map), Colormap)
        except ValueError:
            raise AssertionError("invalid colormap provided")

        assert isinstance(self.edge_threshold, float)
        assert 0 <= self.edge_threshold < 1

    def _make_graph(self):
        # scale the absolute coefficients to be between 0-1
        negative = self.coefficients < 0
        coefficients = np.abs(self.coefficients)
        small, large = coefficients.min(), coefficients.max()
        normalized = (coefficients - small) / (large - small + EPSILON)

        nodes = len(self.coefficients)
        for u, v in permutations(range(nodes), 2):
            weight = normalized[u, v]
            if weight >= self.edge_threshold:
                u_label = self._make_label(u + 1, self.concentrations[u])
                v_label = self._make_label(v + 1, self.concentrations[v])

                self._G.add_edge(u_label, v_label)
                self._weights.append(weight * WEIGHT_SCALING)

                # re-scale to be between 0 and 255, with 0 centered on 127
                color = COLOR_MIDPOINT + weight * COLOR_MIDPOINT * (-1 if negative[u, v] else 1)
                self._colors.append(self._cm(int(color)))

    @staticmethod
    def _make_label(index, concentration):
        # relabel nodes must have proper regex for getting the index out of this string!
        return f"g{index}:\n[{round(concentration[0], 3)}]"

    def _relabel_nodes(self, concentrations: np.array):
        assert len(concentrations) == len(self.concentrations)
        mapping = dict()
        for node in self._G.nodes:
            # regex must parse the _make_label output for node index
            index = int(findall("(?<=g)[0-9]+(?=:)", node)[0])
            mapping[node] = self._make_label(index, concentrations[index - 1])

        nx.relabel_nodes(self._G, mapping, copy=False)

    def plot(self, new_concentrations: np.array = None):
        if new_concentrations is not None:
            self._relabel_nodes(new_concentrations)

        pos = nx.spring_layout(self._G)
        nx.draw_networkx_nodes(self._G, pos, node_size=50, node_color="grey", node_shape="s")
        nx.draw_networkx_labels(self._G, pos, font_size=5, font_family="sans-serif", font_color="white")
        nx.draw_networkx_edges(self._G, pos, edge_color=self._colors, width=self._weights,
                               arrowstyle="->",
                               arrowsize=20)

        plt.title("Network Coefficients (Shown: {} percentile)".format(self.edge_threshold * 100))
        plt.show()


if __name__ == "__main__":
    num_genes = 10
    coeff = np.random.normal(0, 1, (num_genes, num_genes))
    conc = np.random.random((num_genes, 1))
    g = CoefficientGraph(coeff, conc, edge_threshold=0.0)
    g.plot()
    conc = np.random.random((num_genes, 1))
    g.plot(conc)
