# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from math import inf

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        scoreAchieved = successorGameState.getScore() - currentGameState.getScore();
        scaredScore = 0;

        if len(newFood.asList()) == 0:  # if by moving new state we eat all the food , so it is a very good state
            foodScore = 1
        else: # the further closest food is, the worst that new state would be
            foodScore = 1.0 / min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])

        for g in newScaredTimes: # sum of scared time for all the ghosts is a good factor for eval function
            scaredScore += g

        # if the distance between pacman and nearest ghost is less than 2 in successor state,
        # it is a very dangerous state and its evaluation value has to be minimized
        successorGhostPositions = successorGameState.getGhostPositions()
        closestGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in successorGhostPositions])
        ghostScore = 0
        if closestGhost < 2:
            ghostScore = -inf

        return scoreAchieved + foodScore + ghostScore + scaredScore


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth, index):
            depth -= 1
            if depth < 0:  # it means we reached defined depth in minimax tree
                return [self.evaluationFunction(state), None]
            currentNodeVal = -inf
            legalMoves = state.getLegalActions(index)
            if len(legalMoves) == 0:  # we have lost or won the game
                return [self.evaluationFunction(state), None]

            for action in legalMoves:
                newState = state.generateSuccessor(index, action)
                successorValue = min_value(newState, depth, index + 1)[0]
                if successorValue > currentNodeVal:
                    currentNodeVal = successorValue
                    currentNodeAction = action
            return [currentNodeVal, currentNodeAction]

        def min_value(state, depth, index):  # index of current agent
            currentNodeVal = inf
            if index < state.getNumAgents() - 1:
                valueFunction = min_value
                nextIndex = index + 1
            else:
                valueFunction = max_value
                nextIndex = 0
            legalMoves = state.getLegalActions(index)
            if len(legalMoves) == 0:  # we have lost or won the game
                return [self.evaluationFunction(state), None]
            for action in legalMoves:
                successor = state.generateSuccessor(index, action)
                successorValue = valueFunction(successor, depth, nextIndex)[0]
                if successorValue < currentNodeVal:
                    currentNodeVal = successorValue
                    currentNodeAction = action
            return [currentNodeVal, currentNodeAction]

        return max_value(gameState, self.depth, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth, index, alpha, beta):
            depth -= 1
            if depth < 0:  # it means we reached defined depth in minimax tree
                return [self.evaluationFunction(state), None]
            currentNodeVal = -inf
            legalMoves = state.getLegalActions(index)
            if len(legalMoves) == 0:  # we have lost or won the game
                return [self.evaluationFunction(state), None]

            for action in legalMoves:
                newState = state.generateSuccessor(index, action)
                successorValue = min_value(newState, depth, index + 1, alpha, beta)[0]
                if successorValue > currentNodeVal:
                    currentNodeVal = successorValue
                    currentNodeAction = action
                if currentNodeVal > beta:  # pruning condition - alpha > v
                    return [currentNodeVal, currentNodeAction]
                alpha = max(alpha, currentNodeVal)
            return [currentNodeVal, currentNodeAction]

        def min_value(state, depth, index, alpha, beta):  # index of current agent
            currentNodeVal = inf
            if index < state.getNumAgents() - 1:
                valueFunction = min_value
                nextIndex = index + 1
            else:
                valueFunction = max_value
                nextIndex = 0
            legalMoves = state.getLegalActions(index)
            if len(legalMoves) == 0:  # we have lost or won the game
                return [self.evaluationFunction(state), None]
            for action in legalMoves:
                successor = state.generateSuccessor(index, action)
                successorValue = valueFunction(successor, depth, nextIndex, alpha, beta)[0]
                if successorValue < currentNodeVal:
                    currentNodeVal = successorValue
                    currentNodeAction = action
                if currentNodeVal < alpha:
                    return [currentNodeVal, currentNodeAction]
                beta = min(beta, currentNodeVal)
            return [currentNodeVal, currentNodeAction]
        return max_value(gameState, self.depth, 0, -inf, inf)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth, index):
            depth -= 1
            if depth < 0:  # it means we reached defined depth in minimax tree
                return [self.evaluationFunction(state), None]
            currentNodeVal = -inf
            legalMoves = state.getLegalActions(index)
            if len(legalMoves) == 0:  # we have lost or won the game
                return [self.evaluationFunction(state), None]

            for action in legalMoves:
                newState = state.generateSuccessor(index, action)
                successorValue = expected_value(newState, depth, index + 1)[0]
                if successorValue > currentNodeVal:
                    currentNodeVal = successorValue
                    currentNodeAction = action
            return [currentNodeVal, currentNodeAction]

        def expected_value(state, depth, index):  # index of current agent
            if index < state.getNumAgents() - 1:
                valueFunction = expected_value
                nextIndex = index + 1
            else:
                valueFunction = max_value
                nextIndex = 0
            legalMoves = state.getLegalActions(index)
            if len(legalMoves) == 0:  # we have lost or won the game
                return [self.evaluationFunction(state), None]
            Sum = 0
            for action in legalMoves:
                successor = state.generateSuccessor(index, action)
                score = valueFunction(successor, depth, nextIndex)[0]
                Sum += score

            return [Sum / len(legalMoves), None]  # the value of this node should be expected value of all of its
            # successors

        return max_value(gameState, self.depth, 0)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    scaredScore = 0
    foodDistances = 0  # sum of Manhattan distances to all of the foods
    currentStateScore = currentGameState.getScore()
    minGhostDistances = inf
    ghostPositions = currentGameState.getGhostPositions()

    for f in foodPositions:
        foodDistances += util.manhattanDistance(f, pacmanPosition)

    for time in scaredTimes:
        scaredScore += time

    for ghostPos in ghostPositions:
        if util.manhattanDistance(pacmanPosition, ghostPos) < minGhostDistances:
            minGhostDistances = util.manhattanDistance(pacmanPosition, ghostPos)

    if len(foodPositions) == 0:  # to prevent division by zero we add this line
        return 1 + scaredScore + currentStateScore
    return 1 / foodDistances + scaredScore + currentStateScore + minGhostDistances

# Abbreviation
better = betterEvaluationFunction
