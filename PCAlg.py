import numpy as np
from .utils import check_dataset, get_features, trackcalls
import networkx as nx
from itertools import combinations, permutations
from collections import defaultdict


class pcalg():
    """
    Peter Sprites and Clark Glymour algorithm

    input :
    dataset =  N*M numpy array where N is the sample
            size and M the feature size

    feature_names = dictionary where key = column
                    position and value = column name.
                    if no feature_names provided,
                    key=value=column position

    """

    def __init__(self, dataset, feature_names=None):
        check_dataset(dataset)
        self.dataset = dataset
        self.features = get_features(dataset,
                                     feature_names)
        self.G = nx.Graph()

    def _instantiate_fully_connected_graph(self):
        self.G.add_nodes_from(self.features.keys())
        for x, y in combinations(self.features.keys(), 2):
            self.G.add_edge(x, y)

    @trackcalls
    def identify_skeleton(self, indep_test, alpha=0.05
                          stable=False):
        """
        STEP 1 of PC algorighm
        estimate skeleton graph from the data.
        input :
        indep_test = independence function

        alpha = significance level for independence test

        stable = use the SGS variant of the PC algorithm
                default = false

        """
        self._istantiate_fully_connected_graph()
        self.d_separators = {}
        d = 0
        cont = True

        while cont:
            print("Level order: {}".format(d))
            cont = False
            # in the stable version,
            # only update neighbors at each level
            x_neighbors = list(self.G.neighbors(x))
            for x, y in permutations(self.features.keys(), 2):
                if not stable:
                    # in the original version, update neighbors within
                    # each level
                    x_neighbors = list(self.G.neighbors(x))
                if y not in x_neighbors:
                    continue
                x_neighbors.remove(y)
                if len(x_neighbors) >= d:
                    cont = True
                    for z in combinations(x_neighbors, d):
                        if indep_test(self.dataset, x, y,
                                      z, alpha):
                            self.G.remove_edge(x, y)
                            self.d_separators[(x, y)] = z
                            self.d_separators[(y, x)] = z
                            break
            d += 1

    def orient_graph(self, indep_test, alpha):
        """
        STEP 2 of the PC algorithm: edge orientation
        """
        self.G = self.G.to_directed()

        # STEP 1: IDENTIFYING UNSHIELDED COLLIDERS
        # for each X and Y, only connected through
        # a third variable (e.g. Z in X--Z--Y), test idenpendence
        # between X and Y conditioned upon Z.
        # If conditionally dependent, Z is an unshielded collider.
        # Orient edges to point into Z (X->Z<-Y)
        # is the conditional Independence test needed???
        for x, y in combinations(self.features.keys(), 2):
            x_successors = self.G.successors(x)
            if y in x_successors:
                continue
            y_successors = self.G.successors(y)
            if x in y_successors:
                continue
            intersect = set(x_successors).intersection(set(y_successors))
            for z in intersect:
                pass

        # STEP 2: PREVENT SPURIOUS UNSHIELDED COLLIDERS
        # for each X Z Y such that
        # X->Z--Y
        # and where X and Y are not directly connected,
        # orient the ZY edge to point into Y:
        # X->Z->Y
        # if  X->Z<-Y were true, Z would have been picked up
        # as unshielded collider in STEP 1

        #  STEP 3: PREVENT CYCLES
        # If there is a pair of variables, X and Y connected 
        # both by an undirected edge and by a directed path,
        # starting at X, through one or more other variables to Y,
        # orient the undirected edge as X->Y

    def render_graph(self):
        render = nx.draw_networkx(G=self.G, labels=self.features)
        return render

    def save_class(self):
        return
