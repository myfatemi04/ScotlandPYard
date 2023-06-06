import numpy as np
import matplotlib.pyplot as plt

def simulate(positionUpdateMatrix):
    # Starts Mr X at a random initial position
    N = positionUpdateMatrix.shape[1]
    print(positionUpdateMatrix.shape)
    
    pctiles = []
    pctiles_rand = []

    for i in range(100):
        pos = np.random.randint(0, N)
        superpos = np.zeros(N)

        superpos[pos] = 1

        for step in range(30):
            available_ticket_types = positionUpdateMatrix[:, :, pos].sum(axis=1) > 0

            transition_probs = positionUpdateMatrix[available_ticket_types].mean(axis=0)[:, pos]

            superpos = positionUpdateMatrix[available_ticket_types].mean(axis=0).T @ superpos
            superpos = superpos / superpos.sum()

            n_possible_positions = (superpos > 0).sum()
            sorted_prob_indexes = np.argsort(superpos)[::-1]
            sorted_probs = superpos[sorted_prob_indexes]
            prob_position = sorted_prob_indexes.tolist().index(pos)
            random_prob_percentile = sorted_probs[sorted_probs >= (1 / n_possible_positions)].sum()
            true_pos_prob = superpos[pos]
            true_pos_prob_percentile = sorted_probs[:prob_position].sum()

            # set next position
            pos = np.random.choice(N, p=transition_probs)

            # print(f"{true_pos_prob:.4f} {1/n_possible_positions:.4f} {true_pos_prob_percentile:.4f} {random_prob_percentile:.4f}")

            pctiles.append(true_pos_prob_percentile)
            pctiles_rand.append(random_prob_percentile)

    print(np.mean(pctiles))
    print(np.mean(pctiles_rand))

    plt.hist(pctiles, bins=100, alpha=0.5, label="True position percentile")
    plt.hist(pctiles_rand, bins=100, alpha=0.5, label="Random position percentile")
    plt.show()

# def results(mrXLikelihoodVector, truePos):
#     n_possible_positions = (mrXLikelihoodVector > 0).sum()
#     sorted_prob_indexes = np.argsort(mrXLikelihoodVector)[::-1]
#     sorted_probs = mrXLikelihoodVector[sorted_prob_indexes]
#     prob_position = sorted_prob_indexes.tolist().index(truePos)
#     random_prob_percentile = sorted_probs[sorted_probs >= (1 / n_possible_positions)].sum()
#     true_pos_prob = mrXLikelihoodVector[truePos]
simulate(np.load("positionUpdateMatrix.npy", allow_pickle=True))