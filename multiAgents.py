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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        score = 0
        currentFoodList = currentGameState.getFood().asList()
        newFoodList = newFood.asList()
        newGhostPositions = [newGhostState.getPosition() for newGhostState in newGhostStates]

        # If a ghost and pacman are in same state (ghost eats pacman), returning negative score to avoid this action
        if newPos in newGhostPositions:
            return -100

        if (newFoodList):
            # If there is less food left, increase score
            if (len(newFoodList) < len(currentFoodList)):
                score += 50
            # Score is more if the distance to closest food is less
            # It motivates pacman to move in direction closer to food 
            foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodList]
            closestFoodDistance = min(foodDistances)
            score = score - closestFoodDistance / 10
        else:
            score = 100
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        # Format of result = [score, action]
        result = self.get_value(gameState, 0, 0)
        # Return the action from result
        return result[1]

    def get_value(self, gameState, index, depth):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return self.evaluationFunction(gameState), ""
        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(gameState, index, depth)
        # Min-agent: Ghost has index > 0
        else:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""
        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth
            current_value = self.get_value(successor, successor_index, successor_depth)[0]
            if current_value > max_value:
                max_value = current_value
                max_action = action
        return max_value, max_action

    def min_value(self, gameState, index, depth):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index)
        min_value = float("inf")
        min_action = ""
        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth
            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            current_value = self.get_value(successor, successor_index, successor_depth)[0]
            if current_value < min_value:
                min_value = current_value
                min_action = action
        return min_value, min_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Format of result = [score, action]
        result = self.get_value(gameState, 0, 0, float("-inf"), float("inf"))
        # Return the action from result
        return result[1]
    
    
    def get_value(self, gameState, index, depth, alpha, beta):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return self.evaluationFunction(gameState), ""
        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(gameState, index, depth, alpha, beta)
        # Min-agent: Ghost has index > 0
        else:
            return self.min_value(gameState, index, depth, alpha, beta)

    def max_value(self, gameState, index, depth, alpha, beta):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""
        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth
            current_value = self.get_value(successor, successor_index, successor_depth, alpha, beta)[0]
            if current_value > max_value:
                max_value = current_value
                max_action = action
            alpha = max(alpha, max_value)
            if alpha > beta:
                return max_value, max_action
        return max_value, max_action

    def min_value(self, gameState, index, depth, alpha, beta):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index)
        min_value = float("inf")
        min_action = ""
        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth
            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            current_value = self.get_value(successor, successor_index, successor_depth, alpha, beta)[0]
            if current_value < min_value:
                min_value = current_value
                min_action = action
            beta = min(beta, min_value)
            if alpha > beta:
                return min_value, min_action
        return min_value, min_action

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
        currentDepth = 0
        pacmanAgentIndex = 0
        successorStates = getSuccessorStates(gameState, pacmanAgentIndex)
        # Calculate scores for the pacman's next states
        nextScores = [self.expectimax(successorState, currentDepth, pacmanAgentIndex + 1) for successorState in successorStates]
        # Get max score from the list of scores
        maxNextScore = max(nextScores)
        # Find the index of the max score
        nextIndex = [index for index in range(len(nextScores)) if nextScores[index] == maxNextScore]
        # Return the first action that has the maximum score
        return gameState.getLegalActions(pacmanAgentIndex)[nextIndex[0]]

    def expectimax(self, gameState, currDepth, agentIndex):
        # If current depth equals maximum depth or the game ends, calculate the terminal value
        # Else calculate expectimax value
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        else:
            successorStates = getSuccessorStates(gameState, agentIndex)
            if agentIndex == 0:
                # Agent index 0 is for pacman, so maximize the expectimax values
                return max(self.expectimax(successorState, currDepth, agentIndex + 1) for successorState in successorStates)
            else:
                # Other agents are ghosts, so take average of expectimax values
                agentIndex += 1
                # If agent index is equal to the total number of agents, 
                # cycle back the agent index to 0 and increase depth 
                if agentIndex == gameState.getNumAgents():
                    currDepth += 1
                    agentIndex = 0
                sumOfValues = sum(self.expectimax(successorState, currDepth, agentIndex) for successorState in successorStates)
                return sumOfValues / len(successorStates)
                
def getSuccessorStates(gameState, agentIndex):
    """
    Get successor states for all pacman's legal actions
    """
    legalActions = gameState.getLegalActions(agentIndex)
    return [gameState.generateSuccessor(agentIndex, action) for action in legalActions]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    1. Check if the pacman position is same as ghost state or not. If yes, punish pacman.
    2. Check if there are any foods remaining, if not reward pacman
    3. If there are foods remaining, reward is more if the number of foods remaning is less and
       the pacman is closer to the closest food
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    score = 0
    newFoodList = newFood.asList()
    newGhostPositions = [newGhostState.getPosition() for newGhostState in newGhostStates]
    # Avoid bombing ghost
    if newPos in newGhostPositions:   
        return -1000
    # Check if foods are remaining
    if (newFoodList):
        # If there is less food left, increase score
        score = score - len(newFoodList)
        # Score is more if the distance to closest food is less
        # It motivates pacman to move in direction closer to food 
        foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodList]
        closestFoodDistance = min(foodDistances)
        score = score - closestFoodDistance
    else:
        score = 1000
    return score + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
