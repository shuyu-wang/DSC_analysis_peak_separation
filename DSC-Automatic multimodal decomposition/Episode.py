"""
   Monte Carlo -Peak signal automatic multi-peak decomposition
The state sequence and state value function of the peak signal decomposition are drawn. 
The rate_rmse method is used to determine the behavior.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import math 
import matplotlib

from queue import Queue 
from random import shuffle 
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from utils import str_key, set_dict, get_dict


def Gaussian(x, amp, cen, wid):
    return amp*np.exp(-((x-cen)/wid)**2)
    #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def Mul_Gaussian(x, *pars):
    params = np.array(pars)
    MulGauss = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp, cen, wid = params[i:i+3]
        MulGauss = MulGauss + Gaussian(x, amp, cen, wid)
    
    return MulGauss

# Parameters of multiple Gaussian distribution functions
def Mul_Param(n, a, m, s):
    guess = np.zeros(n)
    for i in range(0, n, 3):
        guess[i] = a
        guess[i+1] = m
        guess[i+2] = s
    
    return guess

# Root mean square error
def RMSE(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    
    Rmse = np.sqrt(sum(squaredError) / len(squaredError))
    
    return Rmse



########## Build a peak decomposition environment #########

# Build a decomposer and provide three parameters, namely the behavior space (A) 
# and whether to display specific information on the terminal (display)
class Decomer():
    ''' decomposer '''
    def __init__(self, name = "", A = None, display = False):
        self.name = name
        self.x = x # abscissa of peak signal
        self.y = y # coordinates of the peak signal
        self.nums = [] # peak decomposition number
        self.rmse = [] # evaluate the fitting effect, RMSE
        self.display = display # Wwether to display decomposed text information
        self.policy = None # policy
        self.learning_method = None # learning method
        self.A = A # action space

    def __str__(self):
        return self.name
    
    def total_nums(self):
        total_nums = 0
        nums = self.nums
        if nums is None:
            return 0
        for num in nums:
            total_nums += num
        return total_nums
    
    # RMSE information after fitting according to the number of peaks 
    def get_rmse(self):
        total_nums = 0  # number of peak parameters
        total_nums = self.total_nums()
        Nums = total_nums * 3
        
        # dsc data
        #AmpMin = (max(y)-min(y)) 
        AmpMin = 1
        WidMin = 1
        Cen = (min(x)+max(x))/2
        #amp = 1 + abs(max(y))/(abs(min(y))+0.1)
        guess = Mul_Param(n = Nums, a = AmpMin, m = Cen, s = WidMin)
        #guess = Mul_Param(n = Nums, a = amp, m = Cen, s = WidMin)
        popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=300000)
      
        """# mass spectra
        AmpMin = 500
        WidMin = 1
        Cen = (min(x)+max(x))/2
        guess = Mul_Param(n = Nums, a = AmpMin, m = Cen, s = WidMin)
        popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=300000)
        """
        GaussFit = Mul_Gaussian(x,*popt)
        rmse = RMSE(y, GaussFit)
        self.rmse = rmse
        #self.nums = total_nums
        
        return self.rmse
    
    # The number of peaks the player gets
    def receive(self, nums = []): 
        nums = list(nums)
        for num in nums:
            self.nums.append(num)
    
    # Clear the peak fitting data in the player's hand
    def discharge_nums(self):  
        # clear the number of peaks
        self.nums.clear()
    
    # Player at this time, Peak decomposition information
    def nums_info(self): 
        # Display the number of peak decomposition and RMSE specific information
        self.rmse = self.rmse
        self._info("{}{} current peak information: number:{}, RMSE:{}\n".format(self.role, self, self.nums, self.rmse))

    def rmse_info(self): 
        # Display the number of peak decomposition and RMSE specific information
        self._info("{}{}current peak information: RMSE:{}\n".format(self.role, self, self.rmse))
        
    def _info(self, msg):
        if self.display:
            print(msg, end = "")


# Player 1 and Player 2 are both a decomposer. We can inherit from 
# the Decomer class Dealer and Player to represent Player 1 and Player 2, respectively.
class Dealer(Decomer):
    '''Player 1
     Inherited the Decormer class, which means that it has all the attributes and methods of the decomposer. 
     Player 1’s strategy is fixed: as long as the RMSE of player 1’s peak fitting reaches or is less than 1, 
     stop increasing the number of peaks
    '''
    def __init__(self, name = "", A = None, display = False):
        super(Dealer,self).__init__(name, A, display)
        self.role = "player 1" # role
        self.policy = self.dealer_policy # Player 1's policy

    def total_num_value(self): 
        if self.nums is None or len(self.nums) == 0:
            return 0
        return self.total_nums(self.nums)
        #return self.total_nums(self.nums[0])
    
    def dealer_policy(self, Dealer = None): # details of player's policy
        action = ""
        dealer_rmse = self.get_rmse() # get RMSE
        if dealer_rmse < 1:
        #if dealer_rmse < 10:
            action = self.A[1] # stop increasing the number of peaks
        else:
            action = self.A[0] 
            
        return action        


class Player(Decomer):
    ''' Player 2
     The player’s most primitive strategy is formulated, as long as the RMSE is greater than 1, 
     the number of peaks will continue to increase. Get current status information and prepare for strategy evaluation.
    '''
    def __init__(self, name = "", A = None, display = False):
        super(Player, self).__init__(name, A, display)
        self.policy = self.naive_policy
        self.role = "player 2"
    
    def get_state(self, dealer):
        #dealer_first_num_value = dealer.first_num_value()
        dealer_total_num_value = self.total_nums() 
        player_rmse = self.get_rmse()
        player_nums = self.total_nums()
        return dealer_total_num_value, player_nums
    
    def get_state_name(self, dealer):
        return str_key(self.get_state(dealer))
    
    def naive_policy(self, dealer=None):
        player_rmse = self.get_rmse()
        if player_rmse > 1:
        #if player_rmse > 10:
            action = self.A[0] # continue to increase the number of peaks
        else:
            action = self.A[1]
        return action    
    
# Peak signal fitting data, organizing game matches, judging wins and losses, etc.
class Arena():
    # Responsible for game management
    def __init__(self, x = None, y = None, display = None, A = None):
        self.x = x 
        self.y = y 
        self.nums = [1, 2] *50
        self.num_q = Queue(maxsize = 100)
        self.nums_in_pool = [] # used cards
        self.display = display
        self.episodes = [] # generate a list of game information
        self.load_nums(self.nums)
        self.A = A           

    def load_nums(self, nums):
        shuffle(nums) 
        for num in nums: 
            self.num_q.put(num)
        nums.clear()
        
        return nums 
            
    def reward_of(self, dealer, player):
        '''Judge the player reward value, with the peak number and RMSE information 
        of player 1 and player 2
        '''
        dealer_rmse = dealer.get_rmse()
        player_rmse = player.get_rmse()
        sub_rmse = abs(dealer_rmse - player_rmse)
        rate_rmse = sub_rmse/dealer_rmse
        
        if player_rmse < 0.0001 or dealer_rmse <0.0001: 
            reward = -1
        else:
            #if sub_rmse/dealer < 0.01 and player_rmse < 0.05:
            if rate_rmse < 0.5:
                reward = 1
            else:
                reward = -1
        
        return reward, player_rmse, dealer_rmse, rate_rmse

    # Decomposition function realization            
    def serve_num_to(self, player, n = 1):
        # Assign the number of peaks to player 1 or player 2      
        nums = [] # information on the number of peaks to be allocated, brand
        for _ in range(n):
            nums.append(self.num_q.get())
        self._info("outgiving {} peak ({}) for {}{};".format(n, nums, player.role, player))
        player.receive(nums) # the number of peaks that a player accepts
        player.get_rmse() # a player performs peak decomposition to get RMSE
        player.nums_info()

    
    def _info(self, message):
        if self.display:
            print(message, end="")    
            
    # After the end of a round, the number of decomposed peaks is recovered
    def recycle_nums(self, *players):
        # Reset peak signal number data. 
        if len(players) == 0:
            return
        for player in players:
            for num in player.nums:
                self.nums_in_pool.append(num)
            player.discharge_nums() # player has no data on the number of split peaks
    
    
    # Have player 1 and player 2 play a game
    def play_game(self, dealer, player):
        # Play a game splitter to generate a state sequence and final reward
        self._info("\n========= Start a new round =========\n")
        self.serve_num_to(dealer, n=1) # number of peaks for player 1
        self.serve_num_to(player, n=2) # number of peaks for player 2
        episode = [] # record a match
        if player.policy is None:
            self._info("Player 2 needs a strategy")
            return
        if dealer.policy is None:
            self._info("Player 1 needs a strategy")
            return
        
        # Player 2 plays a peak decompose game
        while True:
            action = player.policy(dealer)
            # Player 2’s strategy produces an action
            self._info("{}{} select: {};".format(player.role, player, action))
            episode.append((player.get_state_name(dealer), action)) # record a (s, a)
            if action == self.A[0]: # continue to increase the number of peaks
                self.serve_num_to(player) # Pass it to player 2 and decompose the number signal by one peak
            else: # stop add the number of peaks
                break
        
        ''' Description of the stop condition below
        '''
        # After player 2 stops peaking, the RMSE in the player’s hands is calculated. 
        # If the player overfits, the dealer does not need to continue.
        reward, player_rmse, dealer_rmse, rate_rmse = self.reward_of(dealer, player)
        
        if player_rmse < 0.001:
            self._info("Player 2 overfitting {} lose，Score:{}\n".format(player_rmse, reward))
            self.recycle_nums(player, dealer)
            self.episodes.append((episode, reward)) 
            self._info("========= End of the game==========\n")
            
            return episode, reward
        
        # The RMSE change rate between player 2 and player 1 does not meet the requirements
        #self._info("\n")
        while True:
            action = dealer.policy() 
            self._info("{}{} select: {};".format(dealer.role, dealer, action))
            if action == self.A[0]: 
                self.serve_num_to(dealer)
            else:
                break
        
        self._info("\nBoth parties stopped increasing the number of peaks;\n")
        reward, player_rmse, dealer_rmse, rate_rmse = self.reward_of(dealer, player)
        player.nums_info()
        dealer.nums_info()
        playerNums = player.total_nums()
        dealerNums = dealer.total_nums()
        selectNums = min(dealerNums, playerNums)
        selectRmse = max(player_rmse, dealer_rmse)
        
        
        if 0 < rate_rmse < 0.5:
            if selectNums == playerNums: 
                reward = 1
                self._info("Player 2 win!\n")
                self._info("The number of peaks of the decomposer should be selected：{}, RMSE is:{}\n".
                           format(selectNums, selectRmse))
            else:
                reward = -1
                self._info("Player 1 win!\n")
                self._info("The number of peaks of the decomposer should be selected：{}, RMSE is:{}\n".
                           format(selectNums, selectRmse))
        elif rate_rmse == 0:
            reward = 1
            self._info("Both sides draw!\n")
            self._info("The number of peaks of the decomposer should be selected：{}, RMSE is:{}\n".
                       
                       format(selectNums, selectRmse))
        else:
            #self.play_game(dealer, player)  
            reward = 0
            self._info("Both lose!\n")
            
        self._info("========= End of the game ==========\n")
        self.recycle_nums(player, dealer) 
        self.episodes.append((episode, reward)) 
        
        return episode, reward   
     
         
    # Generate multiple games at once
    def play_games(self, dealer, player, num = 2, show_statistic = True):
        ''' Play multiple games at once'''
        results = [0, 0, 0] 
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode ,reward)
        
 
        if show_statistic:
            print("Played a {} games，Player2 wins {} rounds， draw {} rounds， lost {} rounds， win rate {:.2f}, loss rate:{:.2f}"\
                  .format(num, results[2],results[1],results[0],results[2]/num, results[0]/num))
                
        return


    def _info(self, message):
        if self.display:
            print(message, end = "")
            



# Plot the value function
def draw_value(value_dict, is_q_dict = False, A = None):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(1, 10, 1) 
    #y = np.arange(0.01, 2, 0.1) 
    y = np.arange(1, 10, 1) 
    ax.set_xlabel('Player\'s Showing peak numbers')
    ax.set_ylabel('Dealer\'s Current card')
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
    #ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1)
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.cm.coolwarm)
    plt.show()

           
###   Generate game data   ### 
######   dsc data   ######
#file = r'../combined_dsc1.csv' # BSA protein
#file = r'../calfitter_dsc2.csv' # LinB protein
#file = r'../lysozyme_dsc1.csv' # Lysozyme protein
#file = r'../dvd_dsc1.csv' # dvd protein
#file = r'../dvd_dsc2.csv' # dvd protein, subtract buffer
file = r'../dvd_dsc2_substractBase.csv' # dvd protein, subtract buffer, substract baseline
#file = r'../mab_dsc1.csv' # mab protein
#file = r'../mab_dsc1.csv' # mab protein substract baseline

df = pd.read_csv(file, header = None)
df = df.T
#print(df)

Q = df.iloc[0,1:]
Q = Q.astype(float)
Q = np.array(Q,dtype=np.float32)
#print(Q)

I = df.iloc[1:, 1:]
I = I.astype(float)
I =np.array(I,dtype=np.float32)

# Preparing the data
i = 0
x = Q
y = I[i]


A=["continue","stop"]
display = False
# Create a player 1 and a dealer 2. Player 2 uses the original strategy, 
# and player 1 uses its fixed strategy
player = Player(A = A, display = display)
dealer = Dealer(A = A, display = display)
arena = Arena(x = x, y = y, A = A, display = display)
'''
# Generate num the complete game
#arena.play_game(dealer, player)
arena.play_games(dealer, player, num = 2)
'''

# Observe the information of several games
display = True
player.display, dealer.display, arena.display = display, display, display
arena.play_games(dealer, player, num = 20) 
print(arena.episodes) 


###  Strategy evaluation  ###
# The value of the statistical state, the attenuation factor is 1, 
# the instant reward of the intermediate state is 0, and the incremental Monte Carlo evaluation
def policy_evaluate(episodes, V, Ns):
    for episode, r in episodes:
        for s, a in episode:
            ns = get_dict(Ns, s)
            v = get_dict(V, s)
            set_dict(Ns, ns+1, s)
            set_dict(V, v+(r-v)/(ns+1), s)

V = {} 
Ns = {} 
policy_evaluate(arena.episodes, V, Ns) 
 
draw_value(V,  A = A) # Draw state value                   









