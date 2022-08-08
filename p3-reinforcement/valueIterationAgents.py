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


import mdp, util
import math
from learningAgents import ValueEstimationAgent
import collections


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
        for i in range(self.iterations):
            temp_values = util.Counter()
            for state in self.mdp.getStates():
                max = math.inf * -1
                temp_values[state] = 0.0
                for action in self.mdp.getPossibleActions(state):
                    tempValue = 0.0
                    temp = self.mdp.getTransitionStatesAndProbs(state, action)
                    for (sPrime, t) in temp:
                        # print((sPrime, t))
                        reward = self.mdp.getReward(state, action, sPrime)
                        sPrimeValue = self.values[sPrime]
                        tempValue += t * (reward + self.discount * sPrimeValue)  # using bellman equation
                    if tempValue > max:
                        max = tempValue

                    temp_values[state] = max
            self.values = temp_values  # update values after ith iteration

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
        tempValue = 0
        temp = self.mdp.getTransitionStatesAndProbs(state, action)
        for (sPrime, t) in temp:
            reward = self.mdp.getReward(state, action, sPrime)
            sPrimeValue = self.getValue(sPrime)
            tempValue = tempValue + t * (reward + self.discount * sPrimeValue)  # using bellman equation

        return tempValue

        # util.raiseNotDefined()

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
            self.values[state] = 0
            return None
        optimalPolicy = None
        maxQval = math.inf * -1
        for action in self.mdp.getPossibleActions(state):
            qVal = self.computeQValueFromValues(state, action)
            if qVal >= maxQval:
                optimalPolicy = action
                maxQval = max(maxQval, qVal)
        return optimalPolicy

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


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
        numberOfSatets = len(self.mdp.getStates())
        for i in range(self.iterations):
            state = self.mdp.getStates()[i % numberOfSatets]
            if self.mdp.isTerminal(state):
                continue
            max = math.inf * -1
            for action in self.mdp.getPossibleActions(state):
                tempValue = 0.0
                temp = self.mdp.getTransitionStatesAndProbs(state, action)
                for (sPrime, t) in temp:
                    # print((sPrime, t))
                    reward = self.mdp.getReward(state, action, sPrime)
                    sPrimeValue = self.values[sPrime]
                    tempValue += t * (reward + self.discount * sPrimeValue)  # using bellman equation
                if tempValue > max:
                    max = tempValue
            self.values[state] = max  # update value of i%numberOfState th state in ith iteration


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
        prioriteies = util.PriorityQueue()
        predecessors = {}
        for state in self.mdp.getStates():  # initializing predecessors of s in predecessors dictionary as a set
            if not self.mdp.isTerminal(state):
                predecessors[state] = set()

        for state in self.mdp.getStates():  # put predecessors of s in predecessors dictionary
            if self.mdp.isTerminal(state):
                continue
            maxQVal = math.inf * -1
            for action in self.mdp.getPossibleActions(state):
                for (sPrime, t) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if not self.mdp.isTerminal(sPrime):
                            predecessors[sPrime].add(state)  # add predecessors to set
                qVal = self.computeQValueFromValues(state, action)
                if qVal > maxQVal:
                    maxQVal = qVal
            diff = abs(self.values[state] - maxQVal)
            prioriteies.push(state, -1 * diff)  # add predecessors to queue based on priority

        for i in range(self.iterations):
            if prioriteies.isEmpty():
                return
            state = prioriteies.pop()  # pop s from queue

            if not self.mdp.isTerminal(state):  # update value of state if its not terminal using bellman equation
                optimalPolicy = self.computeActionFromValues(state)
                maxQVal = self.computeQValueFromValues(state, optimalPolicy)
                self.values[state] = maxQVal

                for p in predecessors[state]:
                    if not self.mdp.isTerminal(p):
                        maxQVal = math.inf * -1
                        for action in self.mdp.getPossibleActions(p):
                            qVal = self.computeQValueFromValues(p, action)
                            if qVal > maxQVal:
                                maxQVal = qVal
                        diff = abs(self.values[p] - maxQVal)
                        if diff > self.theta:
                            prioriteies.update(p, -1 * diff)  # update method puts this element in queue only
                            # if it has higher priority in comparison to its previous priority( if it exists in queue)
