# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import collections

import mdp
import util
from learningAgents import ValueEstimationAgent

# Changes #1
class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        times = self.iterations
        while times:
            currValue = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                maxValue = -99999
                for action in self.mdp.getPossibleActions(state):
                    actionValue = 0
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        actionValue += prob * (self.mdp.getReward(state, action, nextState) +
                                               self.discount * self.values[nextState])
                    maxValue = max(maxValue, actionValue)
                currValue[state] = maxValue
            self.values = currValue
            times -= 1
    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        actionValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            actionValue += prob * (self.mdp.getReward(state, action, nextState) +  self.discount * self.values[nextState])
        return actionValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return
        bestAction = self.mdp.getPossibleActions(state)[0]
        maxValue = -99999
        for action in self.mdp.getPossibleActions(state):
            Q_value = self.computeQValueFromValues(state, action)
            if Q_value > maxValue:
                maxValue = Q_value
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

# Changes #2
class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num_states = len(states)
        for i in range(self.iterations):
          state = states[i % num_states]
          if not self.mdp.isTerminal(state):
            values = []
            for action in self.mdp.getPossibleActions(state):
              q_value = self.computeQValueFromValues(state, action)
              values.append(q_value)
            self.values[state] = max(values)
# Changes #3 - fix the time complexity
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        predecessors = {}
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            for action in self.mdp.getPossibleActions(state):
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                if nextState in predecessors:
                  predecessors[nextState].add(state)
                else:
                  predecessors[nextState] = {state}

        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            values = []
            for action in self.mdp.getPossibleActions(state):
              q_value = self.computeQValueFromValues(state, action)
              values.append(q_value)
            diff = abs(max(values) - self.values[state])
            pq.update(state, - diff)

        for i in range(self.iterations):
          if pq.isEmpty():
            break
          temp_state = pq.pop()
          if not self.mdp.isTerminal(temp_state):
            values = []
            for action in self.mdp.getPossibleActions(temp_state):
              q_value = self.computeQValueFromValues(temp_state, action)
              values.append(q_value)
            self.values[temp_state] = max(values)

          for p in predecessors[temp_state]:
            if not self.mdp.isTerminal(p):
              values = []
              for action in self.mdp.getPossibleActions(p):
                q_value = self.computeQValueFromValues(p, action)
                values.append(q_value)
              diff = abs(max(values) - self.values[p])
              if diff > self.theta:
                pq.update(p, -diff)