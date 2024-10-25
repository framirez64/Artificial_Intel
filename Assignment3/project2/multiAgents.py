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


import random

import util
from game import Agent, Directions
from util import manhattanDistance


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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
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
        # Score measure
        score = successorGameState.getScore()
        # Ghost info
        ghostpos = successorGameState.getGhostPosition(1)
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        ghostDistances = [util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
        # Food info
        foods = newFood.asList()
        capsulesLeft = currentGameState.getCapsules()
        shortestFoodDistance = min([util.manhattanDistance(newPos, food) for food in foods]) if foods else 0
        # Incentivize chasing ghosts
        for i, ghostDistance in enumerate(ghostDistances):
            if newScaredTimes[i] > 0:  
                score += 200 / (ghostDistance + 1)  
            elif ghostDistance < 2:  
                score -= 100  
        score += 100 / (shortestFoodDistance + 1) 
        # Check if any food has been eaten
        if len(foods) < len(currentGameState.getFood().asList()):
            score += 100  #
        capsulesLeft = successorGameState.getCapsules()
        if newPos in capsulesLeft:
            score += 200
        # Stopping is bad
        if action == Directions.STOP:
            score -= 1000
        
        return score


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
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
        bestAction, _ = self.minimax(gameState,0,0)
        return bestAction
    
    def minimax(self,gameState,depth,agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (None, self.evaluationFunction(gameState))
        if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    _, value = self.minimax(nextState, depth, 1)  # Move to the next agent (first ghost)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return (bestAction, bestValue)
        else:
            # Ghost's turn (agentIndex > 0)
            bestValue = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex + 1 < gameState.getNumAgents():
                    # Move to the next ghost
                    _, value = self.minimax(nextState, depth, agentIndex + 1)
                else:
                    # Move to Pacman's turn, increase the depth
                    _, value = self.minimax(nextState, depth + 1, 0)
                if value < bestValue:
                    bestValue = value
            return (None, bestValue)  # Ghosts do not return an action
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        bestAction, _ = self.alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))
        return bestAction
    
    def alphaBeta(self, gameState, depth, agentIndex, alpha, beta):
            # Base case: check if game is over or max depth is reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (None, self.evaluationFunction(gameState))

            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return (None, self.evaluationFunction(gameState))

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                bestValue = float('-inf')
                bestAction = None

                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    _, value = self.alphaBeta(nextState, depth, 1, alpha, beta)  # Move to the next agent (ghost)

                    if value > bestValue:
                        bestValue = value
                        bestAction = action

                    alpha = max(alpha, bestValue)
                    if beta < alpha:
                        break
                return (bestAction,bestValue)
            

            else:  # Ghost's turn (minimizing player)
                bestValue = float('inf')

                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIndex, action)

                    # If it's the last ghost, switch back to Pacman and increase the depth
                    if agentIndex + 1 == gameState.getNumAgents():
                        _, value = self.alphaBeta(nextState, depth + 1, 0, alpha, beta)  # Next agent is Pacman
                    else:
                        _, value = self.alphaBeta(nextState, depth, agentIndex + 1, alpha, beta)

                    bestValue = min(bestValue, value)
                    beta = min(beta, bestValue)

                    # Pruning: only prune when beta < alpha, not on equality
                    if beta < alpha:  
                        break

                return (None, bestValue)


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
        bestAction, _ = self.expectimax(gameState,0,0)
        return bestAction
    def expectimax(self, gameState, depth, agentIndex):
            # Base case: check if game is over or max depth is reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (None, self.evaluationFunction(gameState))

            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return (None, self.evaluationFunction(gameState))

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                bestValue = float('-inf')
                bestAction = None

                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    _, value = self.expectimax(nextState, depth, 1)  # Move to the next agent (ghost)

                    if value > bestValue:
                        bestValue = value
                        bestAction = action

                return (bestAction, bestValue)
            else:  # Ghost's turn (expectation player)
                expectedValue = 0.0
                probability = 1.0 / len(legalActions)  # Assuming equal probability for each action

                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIndex, action)

                    # If it's the last ghost, switch back to Pacman and increase the depth
                    if agentIndex + 1 == gameState.getNumAgents():
                        _, value = self.expectimax(nextState, depth + 1, 0)  # Next agent is Pacman
                    else:
                        _, value = self.expectimax(nextState, depth, agentIndex + 1)

                    expectedValue += value * probability  # Accumulate expected value

                return (None, expectedValue)  # Ghosts do not return an action
def betterEvaluationFunction(currentGameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Score measure
        score = successorGameState.getScore()
        # Ghost info
        ghostpos = successorGameState.getGhostPosition(1)
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        ghostDistances = [util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
        # Food info
        foods = newFood.asList()
        capsulesLeft = currentGameState.getCapsules()
        shortestFoodDistance = min([util.manhattanDistance(newPos, food) for food in foods]) if foods else 0
        # Incentivize chasing ghosts
        for i, ghostDistance in enumerate(ghostDistances):
            if newScaredTimes[i] > 0:  
                score += 200 / (ghostDistance + 1)  
            elif ghostDistance < 2:  
                score -= 100  
        score += 100 / (shortestFoodDistance + 1) 
        # Check if any food has been eaten
        if len(foods) < len(currentGameState.getFood().asList()):
            score += 100  #
        capsulesLeft = successorGameState.getCapsules()
        if newPos in capsulesLeft:
            score += 200
        # Stopping is bad
        if action == Directions.STOP:
            score -= 1000
        
        return score



# Abbreviation
better = betterEvaluationFunction
