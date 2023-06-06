import numpy as np
from numpy.random import choice
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .abstractdetective import AbstractDetective
from .abstractmrx import AbstractMrX
from .StupidAIDetective import StupidAIDetective
from .StupidAIMrX import StupidAIMrX

# from .humandetective import HumanDetective


class IllegalMoveException(Exception):
    pass

TICKET_TYPES = ['Bus', 'Taxi', 'Underground', '2x', 'BlackTicket']


class GameEngine(QObject):
    game_state_changed = pyqtSignal()
    game_over_signal = pyqtSignal(str)

    def __init__(self, spymap, num_detectives=4, maxMoves=30, revealedstates=[]):
        super(GameEngine, self).__init__()
        self.spymap = spymap
        self.graph = spymap.graph
        self.num_detectives = num_detectives
        self.maxMoves = maxMoves
        self.revealedstates = revealedstates
        self.mrXLikelihoodVector = np.zeros(len(self.graph.nodes()))
        # can be sorted by ticket type later on
        """
        "Bus": 3,
        "Taxi": 4,
        "Underground": 3,
        "2x": 2,
        "BlackTicket": num_players
        """
        self.positionUpdateMatrix = np.zeros((3, len(self.graph.nodes()), len(self.graph.nodes())))
        for (u, v, t) in spymap.graph.edges.data('ticket'):
            u = int(u.nodeid) - 1
            v = int(v.nodeid) - 1
            set_frame = TICKET_TYPES.index(t)
            self.positionUpdateMatrix[set_frame, u, v] = 1
            self.positionUpdateMatrix[set_frame, v, u] = 1
        # normalize columns of the matrix
        # add an epsilon to avoid division by zero
        self.positionUpdateMatrix = self.positionUpdateMatrix / (np.sum(self.positionUpdateMatrix, axis=1, keepdims=True) + 1e-10)
        # self.players = [HumanDetective(self) for i in range(num_detectives)]
        self.players = [StupidAIDetective(self) for i in range(num_detectives)]
        self.turn = 0
        self.game_over = False
        self.mrxMoves = []
        self.mrxLastKnownLocation = None
        taken_locations = set()
        for detective in self.players:
            chosen = choice(list(set(self.graph.nodes()).difference(taken_locations)))
            taken_locations.add(chosen)
            detective.set_location(chosen)

        self.mrx = StupidAIMrX(self, num_players=num_detectives)
        self.mrx.set_location(choice(list(set(self.graph.nodes()).difference(taken_locations))))
        self.players.append(self.mrx)
        self.game_state_changed.connect(self.check_game_state)

    def get_game_state(self):
        state = {
            "players_state": [player.get_info() for player in self.players],
            "turn": self.turn,
            "mrxmoves": self.mrxMoves
        }
        return state

    def check_game_state(self):
        '''
        Checks if the game is over or not
        '''
        # for p in self.players[:-1]:
        #     if p.location == self.mrx.location:
        #         self.game_over = True
                # msg = "{} has caught Mr.X\n\tGame over!".format(p.name)

        if len(self.mrxMoves) == self.maxMoves:
            self.game_over = True
            msg = "Mr.X has evaded justice!\n\tGame over!"

        if self.game_over:
            self.game_over_signal.emit(msg)
            print(msg)
            exit()

    def get_valid_nodes(self, player_name, ticket):
        player = None
        for p in self.players:
            if p.name == player_name:
                player = p
                break
        if player is None: return []

        valid_nodes = []
        for u, v, tick in self.graph.edges(nbunch=player.location, data='ticket'):
            if tick == ticket and player.tickets[ticket] > 0:
                valid_nodes.append(v)

        # print("Player now at: {}".format(player.location.nodeid))
        # print("Available moves by {}: ".format(ticket))
        # print([n.nodeid for n in valid_nodes])

        return valid_nodes

    def sendNextMove(self, node=None, ticket="Taxi"):
        if node is not None:
            player = self.players[self.turn]
            if node not in self.get_valid_nodes(player.name, ticket):
                raise IllegalMoveException("This move is not allowed.")

            player.tickets[ticket] -= 1
            if isinstance(player, AbstractDetective):
                self.mrx.tickets[ticket] += 1

            if isinstance(player, AbstractMrX):
                # print(self.revealedstates, len(self.mrxMoves))
                if len(self.mrxMoves) in self.revealedstates:
                    # (79) 79 <class 'str'>
                    # print(node, node.nodeid, type(node.nodeid))
                    self.mrxLastKnownLocation = node.nodeid
                    self.mrxMoves.append([self.mrxLastKnownLocation, ticket])
                    self.mrXLikelihoodVector[:] = 0
                    self.mrXLikelihoodVector[int(node.nodeid) - 1] = 1
                    # print(self.mrxMoves, self.mrXLikelihoodVector)
                else:
                    self.mrxMoves.append([None, ticket])
                    # Update Mr. X likelihood vector
                    ticket_frames = [TICKET_TYPES.index(ticket)]
                    if ticket == 'BlackTicket':
                        print("Black ticket used")
                        ticket_frames = [0, 1, 2]
                    elif ticket == '2x':
                        print("2X ticket used")
                        ticket_frames = [0, 1, 2]
                    self.mrXLikelihoodVector = self.positionUpdateMatrix[ticket_frames].mean(axis=0) @ self.mrXLikelihoodVector
                    # print(self.mrXLikelihoodVector.sum())
                    self.mrXLikelihoodVector = self.mrXLikelihoodVector / self.mrXLikelihoodVector.sum()

                if (len(self.mrxMoves) + 1) % 1 == 0:
                    # print("Most likely positions for Mr. X:", np.argsort(self.mrXLikelihoodVector)[-5:] + 1)
                    # print("Mr. X true position:", node.nodeid)
                    n_possible_positions = (self.mrXLikelihoodVector > 0).sum()
                    sorted_prob_indexes = np.argsort(self.mrXLikelihoodVector)[::-1]
                    sorted_probs = self.mrXLikelihoodVector[sorted_prob_indexes]
                    prob_position = sorted_prob_indexes.tolist().index(int(node.nodeid) - 1)
                    random_prob_percentile = sorted_probs[sorted_probs >= (1 / n_possible_positions)].sum()
                    print("We last knew Mr. X's position", len(self.mrxMoves), "moves ago")
                    print("p(true position):", self.mrXLikelihoodVector[int(node.nodeid) - 1])
                    print("p(rand position):", 1/n_possible_positions)
                    print("rank of true position:", prob_position + 1, "out of", n_possible_positions, "possible positions")
                    print("percentile:", self.mrXLikelihoodVector[sorted_prob_indexes[:prob_position]].sum())
                    print("percentile [rand]:", random_prob_percentile)
                    # print("num possible moves from true position:", (self.positionUpdateMatrix[:, int(node.nodeid) - 1] > 0).sum())
                    # print("p_max:", np.max(self.mrXLikelihoodVector))
                    # input()

            player.set_location(node)

        self.turn = (self.turn + 1) % len(self.players)
        self.game_state_changed.emit()

        # prompt next player to play if its AI.
        if self.players[self.turn].is_ai and not self.game_over:
            # print("is AI")
            # Wait for 1 second to simulate thinking and give people time to see
            QTimer.singleShot(50, lambda: self.players[self.turn].play_next())

    def start_game(self):
        self.players[self.turn].play_next()
