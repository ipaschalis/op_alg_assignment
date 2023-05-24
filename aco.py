import numpy as np
from tqdm import tqdm


def eucl_dist(p1, p2):
    summation = 0
    for dim in np.arange(len(p1)):
        summation += (p1[dim] - p2[dim]) ** 2

    return np.sqrt(summation)


def path_length(array):
    length = 0
    for i in np.arange(array.shape[0] - 1):
        length += eucl_dist(array[i], array[i + 1])

    length += eucl_dist(array[-1], array[0])

    return length


def distance_matrix(points_d):
    points_size = points_d.shape[0]
    dist_mat = np.zeros((points_size, points_size))

    for i in range(points_size):
        for j in range(points_size):
            dist_mat[i, j] = eucl_dist(points_d[i], points_d[j])

    return dist_mat


def inverse_distance_matrix(points_d):
    points_size = points_d.shape[0]
    inv_dist_mat = np.zeros((points_size, points_size))

    dist_mat = distance_matrix(points_d)

    for i in range(points_size):
        for j in range(points_size):
            if i == j:
                inv_dist_mat[i, j] = 0.0
            else:
                inv_dist_mat[i, j] = 1.0 / dist_mat[i, j]

    return inv_dist_mat


class Ant:
    def __init__(self, n_locations):
        self.n_locations = n_locations
        self.position = np.random.choice(n_locations)

        self.places_visited = [self.position]
        self.places_left = list(np.arange(self.n_locations))
        self.places_left.remove(self.position)

        self.phero_graph = np.full((n_locations, n_locations), 0.0)
        self.travel_probas = np.zeros(n_locations - 1)

        self.tour_cost = 0.0

    def ant_trip(self, g_phero_graph, dist_mat, inv_dist_mat, alpha=1, beta=1,
                 Q=1):

        for _ in np.arange(len(self.places_left)):

            # determine the probabilities for next move
            # ------------------------------------------

            # compute numerator and denominator for proba-calculating fractions
            allowed_weights = []
            for loc in self.places_left:
                # vector with numerators for each allowable move
                allowed_weights.append(
                    (g_phero_graph[self.position, loc]) * (
                                inv_dist_mat[self.position, loc]))

            # this is the denominator of the proba-calculating fraction
            allowed_weights_sum = np.sum(allowed_weights)

            # probabilities for next move
            travel_probas = allowed_weights / allowed_weights_sum

            # stochastically pick next destination
            # ------------------------------------
            next_destination = np.random.choice(self.places_left,
                                                p=travel_probas)

            # update info
            # ------------

            # add distance into total distance travelled
            self.tour_cost += dist_mat[self.position, next_destination]

            # then change position and update travel-log variables
            self.position = next_destination
            self.places_visited.append(next_destination)
            self.places_left.remove(next_destination)

        # after it has finished, update self.phero_graph
        for i, j in zip(self.places_visited[:-1], self.places_visited[1:]):
            self.phero_graph[i, j] = Q / self.tour_cost

    def ant_flush(self):
        self.places_visited = [self.position]
        self.places_left = list(np.arange(self.n_locations))
        self.places_left.remove(self.position)

        self.phero_graph = np.full((self.n_locations, self.n_locations), 0.0)
        self.travel_probas = np.zeros(self.n_locations - 1)

        self.tour_cost = 0.0


def update_pheromones(g_phero_graph, ants, evapo_coef=0.05):
    dim = g_phero_graph.shape[0]

    for i in range(dim):
        for j in range(dim):
            g_phero_graph[i, j] = ((1 - evapo_coef) * g_phero_graph[i, j]
                                   + np.sum([ant.phero_graph[i, j]
                                             for ant in ants]))
            g_phero_graph[i, j] = max(g_phero_graph[i, j], 1e-08)

    return g_phero_graph


def aco(points, alpha, beta, evapo_coef, colony_size, num_iter):
    # compute (once) the distance matrices
    dist_mat = distance_matrix(points)
    inv_dist_mat = inverse_distance_matrix(points)

    n_locations = points.shape[0]
    ants = [Ant(n_locations) for _ in range(colony_size)]

    # determine initial pheromone value
    phero_init = (inv_dist_mat.mean()) ** (beta / alpha)
    g_phero_graph = np.full((n_locations, n_locations), phero_init)

    # determine scaling coefficient "Q"
    [ant.ant_trip(g_phero_graph, dist_mat, inv_dist_mat, 1) for ant in ants]
    best_ant = np.argmin([ant.tour_cost for ant in ants])
    Q = (ants[best_ant].tour_cost) * phero_init / (0.1 * colony_size)
    print(Q)

    best_path_length = ants[best_ant].tour_cost
    best_path = ants[best_ant].places_visited.copy()

    monitor_cost = []

    for _ in tqdm(np.arange(num_iter)):
        [ant.ant_trip(g_phero_graph, dist_mat, inv_dist_mat, Q) for ant in
         ants]
        g_phero_graph = update_pheromones(g_phero_graph, ants,
                                          evapo_coef).copy()

        iteration_winner = np.argmin([ant.tour_cost for ant in ants])
        best_path_iteration = ants[iteration_winner].places_visited

        # update global best if better
        if best_path_length > ants[iteration_winner].tour_cost:
            best_ant = iteration_winner
            best_path_length = ants[iteration_winner].tour_cost
            best_path = best_path_iteration.copy()

        monitor_cost.append(best_path_length)

        [ant.ant_flush() for ant in ants]

    return best_path, monitor_cost
