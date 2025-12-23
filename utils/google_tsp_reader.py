import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle
from geopy.distance import geodesic
import math

class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class GoogleTSPReader(object):
    """Iterator that reads TSP dataset files and yields mini-batches.
    
    Format expected as in Vinyals et al., 2015: https://arxiv.org/abs/1506.03134, http://goo.gl/NDcOIG
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        self.filedata = shuffle(open(filepath, "r").readlines())  # Always shuffle upon reading data
        self.max_iter = (len(self.filedata) // batch_size)

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def process_batch(self, lines):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_tour_nodes = []
        batch_tour_len = []

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list
            
            # Compute signal on nodes
            nodes = np.ones(self.num_nodes)  # All 1s for TSP...
            
            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * self.num_nodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
            
            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
            else:
                W = np.zeros((self.num_nodes, self.num_nodes))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(self.num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = np.zeros(self.num_nodes)
            edges_target = np.zeros((self.num_nodes, self.num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            nodes_target[j] = len(tour_nodes) - 1
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1
            tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(W)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_tour_nodes.append(tour_nodes)
            batch_tour_len.append(tour_len)
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.edges_target = np.stack(batch_edges_target, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
        batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch



import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle

class VariableTSPReader:
    """
    Reader for variable-size TSP datasets in JSON format.
    Creates batches grouped by number of nodes (N).
    """

    def __init__(self, filepath, batch_size, knn_ratio=None, metric="EUC_2D", distance_fn=None, iteration_mode="curriculum"):
        self.batch_size = batch_size
        self.knn_ratio = knn_ratio
        self.metric = metric
        self.distance_fn = distance_fn
        self.iteration_mode = iteration_mode
        # Load JSON dataset
        with open(filepath, "r") as f:
            data = json.load(f)

        # Group instances by number of nodes
        self.by_size = {}
        for inst in data:
            N = len(inst["coords"])
            self.by_size.setdefault(N, []).append(inst)

        # Shuffle per size and compute batches per N
        self.sizes = sorted(self.by_size.keys())
        self.num_batches = {}
        for N in self.sizes:
            self.by_size[N] = shuffle(self.by_size[N])
            self.num_batches[N] = len(self.by_size[N]) // self.batch_size
            
        self.max_iter = sum(self.num_batches.values())

    # ----------------------------------------------------------
    # ----------------------  ITERADOR  ------------------------
    # ----------------------------------------------------------



    def _iter_fully_random(self):
        import random

        all_batches = []

        for N in self.sizes:
            inst_list = self.by_size[N]
            B = self.batch_size
            nb = self.num_batches[N]

            for batch_idx in range(nb):
                batch_slice = inst_list[batch_idx * B : (batch_idx + 1) * B]
                all_batches.append((N, batch_slice))

        random.shuffle(all_batches)

        for N, batch_slice in all_batches:
            yield self.process_batch(batch_slice, N)


    def _iter_curriculum(self):
        # assume self.sizes já é sorted
        for N in self.sizes:
            inst_list = self.by_size[N]
            B = self.batch_size
            nb = self.num_batches[N]

            for batch_idx in range(nb):
                batch_slice = inst_list[batch_idx * B : (batch_idx + 1) * B]
                yield self.process_batch(batch_slice, N)


    def __iter__(self):
        if self.iteration_mode == "random":
            return self._iter_fully_random()
        elif self.iteration_mode == "curriculum":
            return self._iter_curriculum()
        else:
            raise ValueError(
                f"Unknown iteration_mode '{self.iteration_mode}'. "
                "Use 'random' or 'curriculum'."
            )

    # ----------------------------------------------------------
    # ----------- MÉTODOS AUXILIARES (AGORA DA CLASSE) ---------
    # ----------------------------------------------------------

    def dist_att(self, a, b):
        rij = np.sqrt(((a[0] - b[0])**2 + (a[1] - b[1])**2) / 10.0)
        return int(rij + 0.5)



    def geo_tsplib(self, coords: np.ndarray) -> np.ndarray:
        """
        TSPLIB GEO distance (vectorized).

        coords: (N, 2) array in TSPLIB DDD.MM format
                [lat, lon] or [x, y] conforme dataset TSPLIB
        returns: (N, N) distance matrix (float64), TSPLIB-compliant
        """
        RRR = 6378.388
        PI = np.pi

        # --- TSPLIB degree-minute → radians ---
        deg = np.trunc(coords)
        minutes = coords - deg
        rad = PI * (deg + (5.0 * minutes) / 3.0) / 180.0

        lat = rad[:, 0]
        lon = rad[:, 1]

        # --- Broadcasting total (N x N) ---
        lat_i = lat[:, None]
        lat_j = lat[None, :]
        lon_i = lon[:, None]
        lon_j = lon[None, :]

        q1 = np.cos(lon_i - lon_j)
        q2 = np.cos(lat_i - lat_j)
        q3 = np.cos(lat_i + lat_j)

        # TSPLIB great-circle formula
        arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)

        # estabilidade numérica
        np.clip(arg, -1.0, 1.0, out=arg)

        return np.floor(RRR * np.arccos(arg) + 1.0) / 1000.0

    # distância modularizada
    def compute_distance_matrix(self, coords):
        N = len(coords)

        # custom function
        if self.distance_fn is not None:
            W = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    W[i, j] = self.distance_fn(coords[i], coords[j])
            return W

        # predefined metrics
        if self.metric == "EUC_2D":
            return squareform(pdist(coords, metric="euclidean"))
        elif self.metric == "MAN_2D":
            return squareform(pdist(coords, metric="cityblock"))
        elif self.metric == "ATT":
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    W[i, j] = float(self.dist_att(coords[i], coords[j]))

            # Escalar ATT para reduzir magnitude
            W = W / 1000.0
            return W
        elif self.metric == "GEO":
            return self.geo_tsplib(coords)
        raise ValueError(f"Unknown metric '{self.metric}'")

    # ----------------------------------------------------------
    # ----------------------  KNN GRAPH  ------------------------
    # ----------------------------------------------------------
    def build_knn_graph(self, W_val, num_neighbors):
        N = W_val.shape[0]

        if num_neighbors == -1:
            W = np.ones((N, N))
            np.fill_diagonal(W, 2)
            return W

        W = np.zeros((N, N))
        knns = np.argpartition(W_val, kth=num_neighbors, axis=-1)[:, :num_neighbors+1]
        for i in range(N):
            W[i, knns[i]] = 1
        np.fill_diagonal(W, 2)
        return W

    # ----------------------------------------------------------
    # --------------------  TOUR TARGETS  ----------------------
    # ----------------------------------------------------------
    def build_tour_targets(self, tour, W_val):
        N = len(W_val)
        edges_target = np.zeros((N, N))
        nodes_target = np.zeros(N)
        tour_len = 0

        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i+1]
            edges_target[a, b] = edges_target[b, a] = 1
            nodes_target[a] = i
            tour_len += W_val[a, b]

        # close cycle
        last, first = tour[-1], tour[0]
        edges_target[last, first] = edges_target[first, last] = 1
        nodes_target[last] = len(tour) - 1
        tour_len += W_val[last, first]

        return edges_target, nodes_target, tour_len

    # ----------------------------------------------------------
    # -------------------- PROCESSAMENTO -----------------------
    # ----------------------------------------------------------
    def process_batch(self, instances, num_nodes):
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []
        batch_nodes = []
        batch_nodes_target = []
        batch_nodes_coord = []
        batch_tour_nodes = []
        batch_tour_len = []

        for inst in instances:
            coords = np.array(inst["coords"])
            
            tour = inst["tour"]

            # 1. distance matrix
            W_val = self.compute_distance_matrix(coords)

            # 2. KNN graph

            k = max(1, int(num_nodes * self.knn_ratio))

            W = self.build_knn_graph(W_val, k)

            # 3. TSP targets
            edges_target, nodes_target, tour_len = self.build_tour_targets(tour, W_val)

            # 4. Node features
            nodes = np.ones(num_nodes)

            # 5. Stack
            batch_edges.append(W)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(coords)
            batch_tour_nodes.append(tour)
            batch_tour_len.append(tour_len)

        return DotDict(
            edges=np.stack(batch_edges),
            edges_values=np.stack(batch_edges_values),
            edges_target=np.stack(batch_edges_target),
            nodes=np.stack(batch_nodes),
            nodes_target=np.stack(batch_nodes_target),
            nodes_coord=np.stack(batch_nodes_coord),
            tour_nodes=np.stack(batch_tour_nodes),
            tour_len=np.stack(batch_tour_len),
        )
