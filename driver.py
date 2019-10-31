import matplotlib.pyplot as plt
import networkx as nx
import random
import operator


class Map:
    def __init__(self, cities, distance_matrix):
        self.cities = cities
        self.distances = distance_matrix
        self.pheromones = []
        self.local_pheromones = []
        for i in range(0, len(cities)):
            self.pheromones.append([1] * len(self.cities))
            self.local_pheromones.append([0] * len(self.cities))

    def update_pheromone_local(self, fro, to, val):
        self.local_pheromones[fro][to] += val

    def update_pheromones_global(self):
        for i in range(0, len(self.cities)):
            for j in range(0, len(self.cities)):
                self.pheromones[i][j] = (1 - evaporation_factor) * self.pheromones[i][j] + self.local_pheromones[i][j]
        self.reset_locals()

    def get_pheromone(self, fro, to):
        return self.pheromones[fro][to]

    def get_distance(self, fro, to):
        return self.distances[fro][to]

    def reset_locals(self):
        self.local_pheromones.clear()
        for i in range(0, len(self.cities)):
            self.local_pheromones.append([0] * len(self.cities))


class Ant:
    def __init__(self, current, unvisited):
        self.current = current
        self.unvisited = unvisited

    def travel_next(self):
        prob = [0]*len(self.unvisited)
        fro = ord(self.current) - 65
        sum = 0
        for i in range(0, len(prob)):
            to = ord(self.unvisited[i])-65
            prob[i] = ((country.get_pheromone(fro, to) ** alpha) * (country.get_distance(fro, to) ** beta),
                       self.unvisited[i])
            sum += prob[i][0]
        for i in range(0, len(prob)):
            prob[i] = (prob[i][0] / sum, prob[i][1])
        prob.sort(key=operator.itemgetter(0), reverse=True)
        probability = random.random()
        dest = -1
        for i in range(0, len(prob)):
            if probability < prob[i][0]:
                dest = i
                self.current = prob[i][1]
                break
            else:
                probability -= prob[i][0]
        self.unvisited.remove(self.current)
        if fro != dest:
            country.update_pheromone_local(fro, dest, country.get_pheromone(fro, dest) + 1 / country.get_distance(fro,
                                                                                                                  dest))

    def reset_ant(self):
        self.unvisited.clear()
        self.unvisited += country.cities
        self.unvisited.remove(self.current)
        print(self.unvisited)
        print(self.current)


def draw_graph(graph, i):
    G = nx.Graph()
    G.add_edges_from(graph[0])
    pos = nx.spring_layout(G)
    plt.figure(i)
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in G.nodes()})

    nx.draw_networkx_edge_labels(G, pos, edge_labels=graph[1], font_color='red')
    plt.savefig(save_at + 'pheromones'+str(i)+'.png', dpi=300)
    return G


def init():
    # making nodes
    for i in range(0, node_count):
        nodes.append(chr(i + 65))

    # deciding distances
    for i in range(0, node_count):
        temp = []
        for j in range(0, i):
            temp.append(distances[j][i])
        for j in range(i, node_count):
            if i == j:
                temp.append(0)
            else:
                temp.append(random.randrange(min_distance_limit, max_distance_limit))
        distances.append(temp)


node_count = 5
min_distance_limit = 1
max_distance_limit = 30
iterations = 50
evaporation_factor = 0.1
alpha = 1
beta = 1
save_at = '/Users/Esteev/Desktop/Results/'
nodes = []
edges = []
distances = []
edgeLabels = {}

init()
country = Map(nodes, distances)
ants = []
copy_nodes = nodes
random.shuffle(copy_nodes)
print(copy_nodes)
print()
for i in range(0, len(copy_nodes)):
    this_ant = Ant(copy_nodes[i], list(set(copy_nodes) - set(copy_nodes[i])))
    ants.append(this_ant)

print(country.pheromones)

cur = 0
while cur < iterations:
    for i in range(len(nodes) - 1):
        for ant in ants:
            ant.travel_next()
    for ant in ants:
        ant.reset_ant()
    print("HADOOPA")
    country.update_pheromones_global()
    print(country.pheromones)
    cur += 1
print(distances)

# adding edges and edge labels to distance graph
for i in range(0, node_count):
    for j in range(0, node_count):
        if i != j:
            edges.append([nodes[i], nodes[j]])
            edgeLabels[(nodes[i], nodes[j])] = distances[i][j]
distance_graph = [edges, edgeLabels]
dg = draw_graph(distance_graph, 0)

# adding edges and edge labels to pheromone graph
for i in range(0, node_count):
    for j in range(0, node_count):
        if i != j:
            edgeLabels[(nodes[i], nodes[j])] = country.pheromones[i][j]
pheromone_graph = [edges, edgeLabels]
pg = draw_graph(pheromone_graph, 1)

'''

edgeLabels.clear()
# adding edges and edge labels to pheromone graph
for i in range(0, node_count):
    for j in range(0, node_count):
        if i != j:
            edgeLabels[(nodes[i], nodes[j])] = 1
pheromone_graph = [edges, edgeLabels]
pg = draw_graph(pheromone_graph, 2)
'''
plt.show()
