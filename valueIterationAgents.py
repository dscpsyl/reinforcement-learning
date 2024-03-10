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

from learningAgents import ValueEstimationAgent
import collections
import numpy as np

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.

        Functions you should fill in:
        - computeQValueFromValues (Question 1)
        - computeActionFromValues (Question 1)
        - runValueIteration (Question 1)

        
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        # Since it is generally not possible to retrieve all the states at once for a large MDP model,
        # we will iterate through the states and update the value of each state in each iteration.
        _explore = util.Queue()
        startState = self.mdp.getStartState()
        
        _calculated = []
        _actionvals = []
        
        # Run the value update function for the specified amount of iterations
        for _ in range(self.iterations):
            _explore.push(startState)
            _calculated.clear()
            # We are calculating it based on the batch verson of value iteration
            _prevValues = self.values.copy()
            
            # For each iteration, we go until we have explored all the states
            while not _explore.isEmpty():
                currentState = _explore.pop()
                
                # Get the possible actions of the current state
                actions = self.mdp.getPossibleActions(currentState)
                
                # If the state has no actions and it is not a terminal state, then we set it to 0
                if len(actions) == 0 and not self.mdp.isTerminal(currentState):
                    self.values[currentState] = 0
                    continue
                
                # For each action, calculate the value of the action based on the Bellman equation
                _actionvals.clear()
                for a in actions:
                    _actionvals.append(self._iterValueCalc(currentState, a, _prevValues))
                
                
                # Update the value of the state to the maximum value of the actions for this iteration
                self.values[currentState] = max(_actionvals)
                
                # Add the state to the calculated list
                _calculated.append(currentState)
                
                # Add the next states to the queue
                for a in actions:
                    _nextStates = self.mdp.getTransitionStatesAndProbs(currentState, a)
                    for _s in _nextStates:
                        # Make sure this next state is not already calculated or in the queue or that is is a terminal state
                        if _s[0] not in _calculated and _s[0] not in _explore.list and not self.mdp.isTerminal(_s[0]):
                            _explore.push(_s[0])

            # Return the action that is most appropriate for the state
            self.computeActionFromValues(startState)
                   
    def _iterValueCalc(self, state, action, vals):
        """Helper function for runValueIteration
        that calculates the value of a single action
        uitilizing vector operations

        Args:
            state: The current state that we are processing
            action: The action that we are processing
            vals: The previous values of the state
        
        Return:
            (float): The updated value based on the action and state
        """
        
        # Define the parts of the value calculation
        _statesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        _discount = np.array([self.discount for _ in range(len(_statesProbs))])
        _vPrimes = np.array([vals[_statesProbs[i][0]] for i in range(len(_statesProbs))])
        _rewards = np.array([self.mdp.getReward(state, action, _statesProbs[i][0]) for i in range(len(_statesProbs))])
        _tProbs = np.array([i[1] for i in _statesProbs])
        
        
        # Calculate the terms
        _discountedValues = np.multiply(_discount, _vPrimes) # \gamma * V_{k}(s')
        _updateState = np.add(_rewards, _discountedValues) # R(s, a, s') + \gamma * V_{k}(s')
        _valueProbs = np.multiply(_tProbs, _updateState) # T(s, a, s') * (R(s, a, s') + \gamma * V_{k}(s'))
        result = np.sum(_valueProbs) # \sum_{s'} T(s, a, s') * (R(s, a, s') + \gamma * V_{k}(s'))
        
        if np.isscalar(result):
            return result
        else:
            raise Exception("ValueIterationAgent::_iterValueCalc - result is not a scalar")

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
        
        # Get all possible resulting states and their probabilities
        _nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        _qVector = np.empty(len(_nextStates))
        for i, _s in enumerate(_nextStates):
            _qVector[i] = _s[1] * (self.mdp.getReward(state, action, _s[0]) + self.discount * self.getValue(_s[0]))
        
        # return the sum of the individual state Q-values
        result = np.sum(_qVector)
        if np.isscalar(result):
            return result
        else:
            raise Exception("ValueIterationAgent::computeQValueFromValues - result is not a scalar")

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        # First get the possible actions of the state
        _actions = self.mdp.getPossibleActions(state)
        
        # If there are no actions, return None
        if len(_actions) == 0:
            return None
        
        _qVals = np.array([self.computeQValueFromValues(state, a) for a in _actions])
        return _actions[np.argmax(_qVals)]

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
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
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
        util.raiseNotDefined()

    

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        util.raiseNotDefined()

