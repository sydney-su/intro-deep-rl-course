# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
        
    def dfs():
        curr_state = stack.pop()    
        if curr_state[0] in seen:
            return False
        
        seen.add(curr_state[0])     # mark state as seen
        if problem.isGoalState(curr_state[0]):      # if we have reached the goal state, insert the action and return True
            actions.insert(0, curr_state[1])
            return True
        
        children = problem.getSuccessors(curr_state[0])
        for child in children:      # for each successor, push onto stack and run dfs
            if child[0] not in seen:
                stack.push(child)
                if dfs():           # if a successor reaches the goal state, insert the action and return True
                    actions.insert(0, curr_state[1])
                    return True

        return False    # no successors reach a goal state

    start_state = problem.getStartState()
    actions = []    # the sequence of actions to return
    seen = set()    # to keep track of states we have already visited
    seen.add(start_state)
    stack = util.Stack()    # FILO used for dfs
    
    if problem.isGoalState(start_state):    # if we are already in the goal state, no actions needed
        return actions
    
    for child in problem.getSuccessors(start_state):    # for each successor, push onto the stack and run dfs
        stack.push(child)
        if dfs():
            return actions
    
    return actions


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    seen = set()    # to keep track of states we have already visited
    seen.add(start_state)
    queue = util.Queue()    # FIFO used for bfs, layer by layer
    
    if problem.isGoalState(start_state):    # if we are already in the goal state, no actions needed
        return []
    for child in problem.getSuccessors(start_state):    # for each successor, push onto the queue with the sequence of actions
        seen.add(child[0])
        queue.push(child + ([child[1]],))
    
    while not queue.isEmpty():
        for _ in range(len(queue.list)):    # visit all states in a layer at a time
            curr_state = queue.pop()
            if problem.isGoalState(curr_state[0]):      # if we have reached the goal state, return the sequence of actions taken
                return curr_state[3]
        
            children = problem.getSuccessors(curr_state[0])
            seq = curr_state[3]
            for child in children:      # for each successor, push onto the queue with the sequence of actions if not seen
                if child[0] not in seen:
                    next_seq = seq + [child[1]]
                    queue.push(child + (next_seq,))
                    seen.add(child[0])

    return []
    
def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # use priority queue and pop off least costly
    start_state = problem.getStartState()
    seen = {}
    seen[start_state] = 0      # to keep track of states we have already visited as well as the current cost to visit that state
    queue = util.PriorityQueue()    # priority queue (heap) used to pop lowest cost state

    if problem.isGoalState(start_state):    # if we are already in the goal state, no actions needed
        return []
    
    for child in problem.getSuccessors(start_state):    # for each successor, push onto the pq with the sequence of actions and cost
        seen[child[0]] = child[2]
        queue.push(child + ([child[1]],), seen[child[0]])
    
    while not queue.isEmpty():
        curr_state = queue.pop()
        
        if problem.isGoalState(curr_state[0]):       # if we have reached the goal state, return the sequence of actions taken
            return curr_state[3]
    
        children = problem.getSuccessors(curr_state[0])
        seq = curr_state[3]
        cost = seen[curr_state[0]]
        # need to store costs associated with states
        for child in children:      # for each successor, push onto the queue with the sequence of actions if not seen or encounter lower cost
            next_cost = cost + child[2]
            if child[0] not in seen or next_cost < seen[child[0]]:
                seen[child[0]] = next_cost
                next_seq = seq + [child[1]]
                queue.push(child + (next_seq,), seen[child[0]])

    return []


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    seen = {}       # key = (x, y), value = g where g = cost from start
    seen[start_state] = 0   # to keep track of states we have already visited as well as the current g cost to visit that state
    queue = util.PriorityQueue()    # priority queue (heap) used to pop lowest cost state (f cost)

    if problem.isGoalState(start_state):    # if we are already in the goal state, no actions needed
        return []
    
    for child in problem.getSuccessors(start_state):    # for each successor, push onto the pq with the sequence of actions and f cost
        seen[child[0]] = child[2]
        queue.push(child + ([child[1]],), child[2] + heuristic(child[0], problem))
    
    while not queue.isEmpty():
        curr_state = queue.pop()

        if problem.isGoalState(curr_state[0]):  # if we have reached the goal state, return the sequence of actions taken
            return curr_state[3]

        children = problem.getSuccessors(curr_state[0])
        seq = curr_state[3]
        cost = seen[curr_state[0]]
        
        for child in children:   # for each successor, push onto the queue with the sequence of actions if not seen or encounter lower cost
            next_g = cost + child[2]
            next_h = heuristic(child[0], problem)
            next_f = next_g + next_h
            if child[0] not in seen or seen[child[0]] > next_g:
                seen[child[0]] = next_g
                next_seq = seq + [child[1]]
                queue.push(child + (next_seq,), next_f)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
