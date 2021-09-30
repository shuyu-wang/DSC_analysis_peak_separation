""" utils
"""
import matplotlib.pyplot as plt
import numpy as np
import random    
from mpl_toolkits.mplot3d import Axes3D

def str_key(*args):
    '''Connect the parameters with "_" as the keys of the dictionary. Note that 
    the parameters themselves may be tuple or list types.
    For example, a form similar to ((a,b,c),d). 
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value

def get_dict(target_dict, *args):
    return target_dict.get(str_key(*args),0)


def greedy_pi(A, s, Q, a):
    '''According to greedy selection, calculate the probability of behavior a 
    being greedily selected in behavior space A in state s.
    Consider a situation where multiple actions have the same value 
    '''
    #print("in greedy_pi: s={},a={}".format(s,a))
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        #print("get q from dict Q:{}".format(q))
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            #print("in greedy_pi: {} == {}".format(q,max_q))
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0: return 0.0
    return 1.0/n if a in a_max_q else 0.0

def greedy_policy(A, s, Q):
    """In a given state, select a behavior a from the behavior space A such that Q(s,a) = max(Q(s,)).
     Consider the situation where multiple behaviors have the same value. 
    """
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)
        
def epsilon_greedy_pi(A, s, Q, a, epsilon = 0.1):
    m = len(A)
    greedy_p = greedy_pi(A, s, Q, a)
    #print("greedy prob:{}".format(greedy_p))
    if greedy_p == 0:
        return epsilon / m
    n = int(1.0/greedy_p)
    return (1 - epsilon) * greedy_p + epsilon/m
'''
def epsilon_greedy_policy(self, dealer, epsilon = None):
    player_rmse = self.get_rmse()
    if player_rmse < 2:
        return self.A[1] # 停止加峰
    if player_rmse > 2:
        return self.A[0]
    else:
        A, Q = self.A, self.Q
        s = self.get_state_name(dealer)
        if epsilon is None:
             epsilon = 1.0/(1 + 4 * math.log10(1+player.total_learning_times))
        return epsilon_greedy_policy(A, s, Q, epsilon)
'''
def epsilon_greedy_policy(A, s, Q, epsilon, show_random_num = False):
    pis = []
    m = len(A)
    for i in range(m):
        pis.append(epsilon_greedy_pi(A, s, Q, A[i], epsilon))
    rand_value = random.random() 
    #print(rand_value)
    for i in range(m):
        if show_random_num:
            print("random number:{:.2f}, Probability to be subtracted{}".format(rand_value, pis[i]))
        rand_value -= pis[i]
        if rand_value < 0:
            return A[i]
      
def draw_value(value_dict, is_q_dict = False, A = None):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(1, 3, 1) 
    #y = np.arange(0.01, 2, 0.1) 
    y = np.arange(1, 10, 1) 
    ax.set_xlabel('Player\'s Showing peak numbers')
    ax.set_ylabel('Dealer\'s Current RMSE')
    ax.set_zlabel('State Value')
    ax.view_init(ax.elev, -120) 
 
    X, Y = np.meshgrid(x, y)
    row, col = X.shape
    Z = np.zeros((row,col))
    if is_q_dict:
        n = len(A)
    for i in range(row):
        for j in range(col):
            state_name = str(X[i,j])+"_"+str(Y[i,j])
            if not is_q_dict:
                Z[i,j] = get_dict(value_dict, state_name)
            else:
                assert(A is not None)
                for a in A:
                    new_state_name = state_name + "_" + str(a) 
                    q = get_dict(value_dict, new_state_name)
                    if q >= Z[i,j]:
                        Z[i,j] = q
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.cm.coolwarm)
    plt.show()
    
def draw_policy(policy, A, Q, epsilon):
    def value_of(a):
        if a == A[0]:
            return 0
        else:
            return 1
    rows, cols = 8, 10
    Z = np.zeros((rows, cols))
    dealer_first_card = np.arange(1, 2) 
    player_nums = np.arange(2, 10)
    for i in range(2, 10): 
        for j in range(1, 3):  
            s = j, i
            s = str_key(s)
            a = policy(A, s, Q, epsilon)
            Z[i-2,j-1] = value_of(a) 
            #print(s, a)
    
    plt.imshow(Z, cmap=plt.cm.cool, interpolation=None, origin="lower", extent=[0, 7.5, 2, 10.5])

    
    
    
    