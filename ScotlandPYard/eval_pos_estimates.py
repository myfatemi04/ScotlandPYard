import numpy as np
import matplotlib.pyplot as plt
from .spyengine.maputils import get_map_graph
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import pygraphviz
import io
import cv2

# graph = get_map_graph('map3')
# # creates a pygraphviz graph
# A: pygraphviz.AGraph = to_agraph(graph)
# pos = nx.spring_layout(graph, scale=500, center=(0, 0), iterations=100)
# for nodeid in graph.nodes():
#     A.get_node(nodeid).attr['pos'] = f"{pos[nodeid][0]},{pos[nodeid][1]}!"
# A.layout('neato', '-n')
# A.get_node('1').attr['label'] = 'Mr. X'
# for (u, v, tick) in graph.edges(data='ticket'):
#     A.get_edge(u, v).attr['color'] = {"Taxi": "red", "Bus": "blue", "Underground": "green"}[tick]

# bio = io.BytesIO()

# # draw to output
# A.draw(bio, format='png')

# cv2.imshow("Image", cv2.imdecode(np.frombuffer(bio.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)) #[::3, ::3])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def simulate(positionUpdateMatrix):
    # Starts Mr X at a random initial position
    # Shape os positionUpdateMatrix: (n_ticket_types, new_position, old_position)
    # Then, we can matrix multiply by an (old_position,) vector to get a matrix of new position
    # probabilities by ticket type.
    N = positionUpdateMatrix.shape[1]
    N_ticket_types = positionUpdateMatrix.shape[0]
    
    pctiles = []
    pctiles_rand = []
    probs = []
    probs_rand = []
    prob_ratios = []

    for i in range(1000):
        pos = np.random.randint(0, N)
        superpos = np.zeros(N)
        superpos[pos] = 1

        for step in range(30):
            # if (step + 1) % 5 == 0:
            #     superpos[:] = 0
            #     superpos[pos] = 1

            # set next position. this reveals the ticket type used
            true_available_ticket_types = positionUpdateMatrix[:, :, pos].sum(axis=1) > 0
            chosen_ticket_type = np.random.choice(np.arange(N_ticket_types), p=true_available_ticket_types / true_available_ticket_types.sum())
            transition_probs = positionUpdateMatrix[chosen_ticket_type, :, pos]
            pos = np.random.choice(N, p=transition_probs)

            superpos = positionUpdateMatrix[chosen_ticket_type] @ superpos
            superpos = superpos / superpos.sum()

            n_possible_positions = (superpos > 0).sum()
            sorted_prob_indexes = np.argsort(superpos)[::-1]
            sorted_probs = superpos[sorted_prob_indexes]
            prob_position = sorted_prob_indexes.tolist().index(pos)
            random_prob_percentile = sorted_probs[sorted_probs > (1 / n_possible_positions)].sum()
            true_pos_prob = superpos[pos]
            true_pos_prob_percentile = sorted_probs[:prob_position].sum()

            # print(f"{true_pos_prob:.4f} {1/n_possible_positions:.4f} {true_pos_prob_percentile:.4f} {random_prob_percentile:.4f}")

            pctiles.append(true_pos_prob_percentile)
            pctiles_rand.append(random_prob_percentile)
            probs.append(true_pos_prob) #*n_possible_positions)
            probs_rand.append(1/n_possible_positions)
            prob_ratios.append(true_pos_prob * n_possible_positions)

    print("Mean percentile:", np.mean(pctiles))
    print("Mean percentile (for random choice):", np.mean(pctiles_rand))
    print("Mean likelihood assigned to true position:", np.mean(probs))
    print("Mean likelihood assigned to random position:", np.mean(probs_rand))
    print("Mean ratio of likelihood assigned to true position to number of possible positions:", np.mean(prob_ratios))

    plt.title("Probability of true position vs. random position")
    plt.hist(pctiles, bins=100, alpha=0.5, label="True position percentile")
    plt.hist(pctiles_rand, bins=100, alpha=0.5, label="Random position percentile")
    plt.legend()
    plt.show()

    plt.title("Likelihood assigned to true position")
    plt.hist(probs, bins=100, alpha=0.5, label="True position probability")
    plt.legend()
    plt.show()

    plt.title("True position likelihood / Random position likelihood")
    plt.hist(prob_ratios, bins=100, alpha=0.5, label="True position likelihood / Random position likelihood")
    plt.legend()
    plt.show()

# def results(mrXLikelihoodVector, truePos):
#     n_possible_positions = (mrXLikelihoodVector > 0).sum()
#     sorted_prob_indexes = np.argsort(mrXLikelihoodVector)[::-1]
#     sorted_probs = mrXLikelihoodVector[sorted_prob_indexes]
#     prob_position = sorted_prob_indexes.tolist().index(truePos)
#     random_prob_percentile = sorted_probs[sorted_probs >= (1 / n_possible_positions)].sum()
#     true_pos_prob = mrXLikelihoodVector[truePos]
simulate(np.load("positionUpdateMatrix.npy", allow_pickle=True))