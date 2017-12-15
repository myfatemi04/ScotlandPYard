import networkx as nx
from ScotlandPYard.config.gameconfig import maps


def get_map_graph(map_name):
    if map_name not in maps:
        return None

    map = maps[map_name]
    G = nx.MultiGraph()
    for connection in map["connections"]:
        keys = G.add_edges_from(map["connections"][connection], ticket=connection)

    return G


