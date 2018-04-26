# coding:utf-8
import math
import random

#how to introduce map?
PLAYERNUM=3
class GameState:
    players=range(PLAYERNUM)
    #players =['RED','BLUE','GREEN']
    def __init__(self, ss):
        # At the root pretend the player just moved is player n-1
        # player 0 has the first move
        self.lastplayer = self.players[-1]
        self.cities = 0 #changable enviroments,include stage,time
        self.operators = ss
        self.reward= 0.0 #or score
        self.score = 0.0
        self.rollout_layer=10

    def GetActions(self):
        """ Get all possible actions from this state with map
        """
        actions=range(48)
        validactions=[i for i in actions if i%2==0] #remove zero actions
        #also return policy probabilite
        return validactions,validactions

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        state = GameState(0)
        state.lastplayer = self.lastplayer
        state.cities = self.cities
        state.operators = self.operators
        state.reward= self.reward
        state.score =self.score
        state.rollout_layer = self.rollout_layer
        return state

    def DoAction(self, act):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        if act==[] or act==None: #nothing to do, just change player
            self.lastplayer = (self.lastplayer + 1) % PLAYERNUM
            return

        assert act >= 0 and act <= 20 and act == int(act)
        self.cities = 0
        self.operators -= act #update state
        self.lastplayer = (self.lastplayer+1)%PLAYERNUM
        self.reward  = act
        self.score += self.reward
        self.rollout_layer -= 1


    def Terminal(self):
        #time is over or layer is deep or blood is 0
        if self.operators>0 or self.rollout_layer>0:
            return False
        else:
            return True

    def GetResult(self, lastplayer):
        """ Get the game result from the viewpoint of playerjm.
        """
        #if score1>score2:
        if self.lastplayer == lastplayer:
            return 1.0 # playerjm took the last chip and has won
        else:
            return 0.0 # playerjm's opponent took the last chip and has won

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, action = None,parent = None, state = None):
        self.lastaction=action
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.actions,self.probability = state.GetActions() # future child nodes
        self.lastplayer = state.lastplayer
        self.expandnum=0  #min(len(self.actions)*5,10)
        self.expand_layer = 10
        self.greedprob=0.95

    def FullyExpand(self):
        if self.expandnum < min(len(self.actions)*5,10):
            return False
        else:
            return True

    def Choice(self):
        p=random.random()
        action=int(len(self.actions)*p)
        return self.actions[action]

    def AddChild(self, act, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        node = Node(action = act,parent = self, state = s)
        #self.actions.remove(act)
        self.childNodes.append(node)
        return node

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def SelectBestChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + math.sqrt(2*math.log(self.visits)/c.visits))[-1]
        return s


    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass

def UCT(rootstate, itermax):
    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()
        state.rollout_layer=10
        while not state.Terminal():
            if not node.FullyExpand():
                act=node.Choice()
                state.DoAction(act)
                node=node.AddChild(act,state)
            else:
                node=SelectBestChild()

        # Rollout
        state.rollout_layer = 10
        while not state.Terminal(): # while state is non-terminal
            state.DoAction(1) #rule-based action

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.lastplayer)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    node=sorted(rootnode.childNodes, key=lambda c: c.visits)[-1]
    node.parent=None
    return node

def UCTPlayGame():
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    #initial state
    state = GameState(15)
    action = UCT(rootstate=state, itermax=1000)


if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    UCTPlayGame()

            
                          
            

