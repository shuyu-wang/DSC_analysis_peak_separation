"""
episode and state value function of net peak signal deconvolution 
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simps
import pandas as pd
from tqdm import tqdm
import math 
import matplotlib
import random

from queue import Queue #
from random import shuffle # Random distribution use
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

# Parameters of multiple Gaussian distribution functions
def Mul_ParamTest(n, a, m, s):
    #guess = np.zeros(n)
    guess = np.zeros(n*3)

    for i in range(0, n):
        j = 3 * i 
        guess[j] = a[i]
        guess[j+1] = m[i]
        guess[j+2] = s[i]
      
    return guess

#def wid(x, y)
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
        
        minX, maxX = min(self.x), max(self.x)
        maxY = max(self.y)
        Nums = total_nums
        Cen = np.zeros(Nums)
        WidMin = np.ones(Nums) *5
            
        AmpMin = np.ones(Nums)
        Cen = np.zeros(Nums)
        count = 0
        for i in range(0, Nums):
            AmpMin[i] = random.uniform(0, maxY) 
            #Cen[i] = random.uniform(minX, maxX)
            #Cen[i] = random.uniform(55, 90)  # dvd1
            Cen[i] = random.uniform(65, 85)  # mab1
            WidMin[i] = random.uniform(0, 5)
            
        guess = Mul_ParamTest(n = Nums, a = AmpMin, m = Cen, s = WidMin)
        popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=600000)  
        GaussFit = Mul_Gaussian(x,*popt)
        rmse = RMSE(y, GaussFit)
        self.rmse = rmse
       
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

    def first_num_value(self): 
        if self.nums is None or len(self.nums) == 0:
            return 0
        nums = []
        nums = self.nums
        dealer_total_num_value = self.total_nums() #
        return dealer_total_num_value
    
    def dealer_policy(self, Dealer = None): # details of player's policy
        action = ""
        dealer_rmse = self.get_rmse() # get RMSE
        if dealer_rmse < 0.01: # dvd
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
        '''complete status sequence/ episode '''
        dealer_total_num_value = dealer.first_num_value()
        player_rmse = self.get_rmse()
        player_nums = self.total_nums()
        return dealer_total_num_value, player_nums
    
    def get_state_name(self, dealer):
        return str_key(self.get_state(dealer))
    
    def naive_policy(self, dealer=None):
        player_rmse = self.get_rmse()
        if player_rmse > 0.01:
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
        # Number of peak signals
        shuffle(nums) #Randomly assigned
        for num in nums: # The deque data structure can only be added one by one
            self.num_q.put(num)
        nums.clear() # data clear
        
        return nums 
            
    def reward_of(self, dealer, player):
        '''Judge the player reward value, with the peak number and RMSE information 
        of player 1 and player 2
        '''
        dealer_rmse = dealer.get_rmse()
        player_rmse = player.get_rmse()
        sub_rmse = abs(dealer_rmse - player_rmse)
        rate_rmse = sub_rmse/dealer_rmse
        
        if player_rmse < 0.0001 or dealer_rmse <0.0001: # Two players are overfitting, the reward is -1
            reward = -1
        else:
            #if sub_rmse/dealer < 0.01 and player_rmse < 0.05:
            if rate_rmse < 0.5:
                # The RMSE change rate of Player 1 and Player 2 is less than 0.01, 
                #and rmse<0.01. Reward is 1 
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
        # Reset peak signal data
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
        
        if player_rmse < 0.0001:
            self._info("Player 2 overfitting {} lose，Score:{}\n".format(player_rmse, reward))
            self.recycle_nums(player, dealer)
            #When predicting, you need to form an episode list and then concentrate on learning
            self.episodes.append((episode, reward)) 
            self._info("========= End of the game==========\n")
            
            return episode, reward
        
        # The RMSE change rate between player 2 and player 1 does not meet the requirements
        #self._info("\n")
        while True:
            action = dealer.policy() # The player1 gets an action from its strategy
            self._info("{}{} select: {};".format(dealer.role, dealer, action))
            episode.append((player.get_state_name(dealer), action)) # record a (s, a)
            if action == self.A[0]: # player1 continue to increase numbers of sub-peaks
                self.serve_num_to(dealer)
            else:
                break
        
        # Both sides stop increasing the number of sub-peaks
        self._info("\nBoth parties stopped increasing the number of peaks;\n")
        reward, player_rmse, dealer_rmse, rate_rmse = self.reward_of(dealer, player)
        player.nums_info()
        dealer.nums_info()
        playerNums = player.total_nums()
        dealerNums = dealer.total_nums()
        selectNums = min(dealerNums, playerNums)
        selectRmse = max(player_rmse, dealer_rmse)
        
        
        # Judge whether the player wins or loses
        if 0 < rate_rmse < 0.5:
            if selectNums == playerNums: 
                reward = 1
                self._info("Player 2 win!\n")
                self._info("The number of peaks of the decomposer should be selected：{}, RMSE is:{}\n".
                           format(selectNums, selectRmse))
            else:
                reward = 0
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
            reward = -1
            self._info("Both lose!\n")
            
        self._info("========= End of the game ==========\n")
        self.recycle_nums(player, dealer) # Clear the peak fitting information in the player's hands
        # Add the complete game just generated to the state sequence list, not needed for Monte Carlo control
        self.episodes.append((episode, reward)) 
        
        return episode, reward   
     
         
    # Generate multiple games at once
    def play_games(self, dealer, player, num = 2, show_statistic = True):
        ''' Play multiple games at once'''
        results = [0, 0, 0] # Player 2 loses, draws, and wins
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode ,reward)
        if show_statistic:
            print("Played a {} games，Player2 wins {} rounds， draw {} rounds， lost {} rounds， win rate {:.2f}, loss rate:{:.2f}"\
                  .format(num, results[2],results[1],results[0],results[2]/num,(
                      results[2]+results[1])/num))         
        return


    def _info(self, message):
        if self.display:
            print(message, end = "")
            


# Plot the value function
def draw_value(value_dict, is_q_dict = False, A = None):
    # define of figure
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(1, 5, 1) # Player 1's peak number interval
    #y = np.arange(0.01, 2, 0.1) #Player 2's RMSE value
    y = np.arange(1, 10, 1) # Player 2's peak number interval
    ax.set_xlabel('Player2\'s sub-peak numbers')
    ax.set_ylabel('Player1\'s sub-peak numbers')
    ax.set_zlabel('State Value')
    ax.view_init(ax.elev, -120) 
    
    X, Y = np.meshgrid(x, y)
    # Retrieve the Z axis height from the V dictionary
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

           
###   Generate game data  ### 
#file = r'../combined_dsc1.csv' # BSA protein
#file = r'../calfitter_dsc2.csv' # LinB protein
#file = r'../lysozyme_dsc1.csv' # Lysozyme protein
    
########## Multi peak dsc data   ########## 
#file = r'../dvd_dsc2.csv' # dvd protein, subtract buffer
#file = r'../dvd_dsc2_substractBase.csv' # dvd protein, subtract buffer, substract baseline
file = r'../mab696_dsc2.csv' # mab protein. substract baseline


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
# Create a scene
arena = Arena(x = x, y = y, A = A, display = display)
display = True
player.display, dealer.display, arena.display = display, display, display
arena.play_games(dealer, player, num = 10) 
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

V = {} # State value dictionary
Ns = {} # Number of times the state has been visited
policy_evaluate(arena.episodes, V, Ns) # Learn V value
draw_value(V,  A = A) # Draw state value                 
         

# Parameters of multiple Gaussian distribution functions
def Mul_ParamTest(n, a, m, s):
    guess = np.zeros(n*3)
    for i in range(0, n):
        j = 3 * i 
        guess[j] = a[i]
        guess[j+1] = m[i]
        guess[j+2] = s[i]
      
    return guess

# plot the respective decomposition and rmse of the two players 
def draw_peakDecon(episodes, state, x, y):
    minX, maxX = min(x), max(x)
    maxY = max(y)
    PeakDecNum1 = [] 
    PeakDecNum2 = [] # The number of peak decompositions, especially for player2
    lenEpisodes = len(episodes)  # episode information
    for episode, r in episodes: 
        for s, a in episode: 
            state.append(s)  
        nums_player1 = int(episode[-1][0][0])
        nums_player2 = int(episode[-1][0][2])
        PeakDecNum1.append(nums_player1)
        PeakDecNum2.append(nums_player2) 
         
        if r == 1 or r==0: # reward=1, the information that player 2 won
            Nums = nums_player2
            Cen = np.zeros(Nums)
            WidMin = np.ones(Nums) *5
            AmpMin = np.ones(nums_player2)
            Cen = np.zeros(nums_player2)
            count = 0
            for i in range(0, nums_player2):
                AmpMin[i] = random.uniform(0, maxY) 
                #Cen[i] = random.uniform(60, 95) # dvd
                #Cen[i] = random.uniform(55, 90)  # dvd
                Cen[i] = random.uniform(65, 85) #mab1
                WidMin[i] = random.uniform(0, 5)
            guess = Mul_ParamTest(n = Nums, a = AmpMin, m = Cen, s = WidMin)
            popt, pcov = curve_fit(Mul_Gaussian, x, y, guess, maxfev=600000)
                
            GaussFit = Mul_Gaussian(x,*popt) # Gaussian fitting result
                
            plt.figure(figsize=(8,6), dpi=600)
            j = 0
            for i in range(0, nums_player2): 
                j =3*i
                print("Model Variables", i, ":")
                print("  amp", i, ":", popt[j])
                print("  cen", i, ":", popt[j+1])
                print("  wid", i, ":", popt[j+2])

                # mAb1 DSC data nomarlization: 1mg/mL *238.89
                print(" Enthalpy change of Gaussian fitting-> " + str(i), simps(Gaussian(x, popt[j],popt[j+1], popt[j+2])*238.89, x) )
                #print(" Enthalpy change of Gaussian fitting-> " + str(i), simps(Gaussian(x, popt[j],popt[j+1], popt[j+2])*119.45, x) )
                #print(" Enthalpy change of Gaussian fitting-> " + str(i), simps(Gaussian(x, popt[j],popt[j+1], popt[j+2])*47.78, x) )
                
                # DVD1 DSC data nomarlization:1 mg/mL to convert Kcal/mol/K *3360, KJ/mol/K *3360/4.18=803
                #print(" Enthalpy change of Gaussian fitting-> " + str(i), simps(Gaussian(x, popt[j],popt[j+1], popt[j+2])*803/2, x) )
                #print(" Enthalpy change of Gaussian fitting-> " + str(i), simps(Gaussian(x, popt[j],popt[j+1], popt[j+2])*160, x) )
                #print(" Enthalpy change of Gaussian fitting-> " + str(i), simps(Gaussian(x, popt[j],popt[j+1], popt[j+2])*80, x) ) 
                print("#########################################")
                if(popt[j] > 0): #
                    # mAb1 DSC data nomarlization: 1mg/mL *238.89
                    plt.plot(x, Gaussian(x, popt[j], popt[j+1], popt[j+2])*238.89, "darkgreen", label= 'Peak-'+str(i))
                    #plt.plot(x, Gaussian(x, popt[j], popt[j+1], popt[j+2])*119.45, "darkgreen", label= 'Peak-'+str(i))
                    #plt.plot(x, Gaussian(x, popt[j], popt[j+1], popt[j+2])*47.78, "darkgreen", label= 'Peak-'+str(i))
                    
                    # dvd1 dsc data   1mg/mL -> Kcal/mol/K *3360. KJ/mol/K *3360/4.18=803
                    #plt.plot(x, Gaussian(x, popt[j], popt[j+1], popt[j+2])*803/2, "darkgreen", label= 'Peak-'+str(i))
                    #plt.plot(x, Gaussian(x, popt[j], popt[j+1], popt[j+2])*160, "darkgreen", label= 'Peak-'+str(i))
                    #plt.plot(x, Gaussian(x, popt[j], popt[j+1], popt[j+2])*80, "darkgreen", label= 'Peak-'+str(i))
                    
            # mab1 dsc data 
            plt.plot(x, y*238.89, "darkviolet", label='Net Peak Signal')
            #plt.plot(x, y*119.45, "darkviolet", label='Net Peak Signal')
            #plt.plot(x, y*47.78, "darkviolet", label='Net Peak Signal')
            # dvd1 dsc data nomarlization
            #plt.plot(x, y*803/2, "darkviolet", label='Net Peak Signal')
            #plt.plot(x, y*160, "darkviolet", label='Net Peak Signal')
            #plt.plot(x, y*80, "darkviolet", label='Net Peak Signal')
            
            plt.legend(loc='best', fontsize=16)
            plt.xlabel("Temperature (℃)", size=22, labelpad=10)
            plt.ylabel("Heat capacity (KJ/mol/K)", size=22, labelpad=10)
            plt.tick_params(labelsize=16)
            plt.show()
            
            plt.figure(figsize=(8,6), dpi=600)
            plt.plot(x, GaussFit, 'C3-', label='Gaussian Fitting')
            plt.plot(x, y, "C2.", label='Net Signal')
            plt.legend(loc='best', fontsize=16)
            plt.show()
            
    plt.figure(figsize=(8,6), dpi=600)       
    nums = len( PeakDecNum2)
    xGames = np.linspace(start = 1, stop = nums, num =nums, dtype = int)
    #xGames = np.arange(start = 1, stop = nums, step = 1)
    bar_width = 0.2
    index_Player1 = np.arange(len(xGames))
    index_Player2 = index_Player1 +bar_width
    '''
    plt.bar(index_Player1, PeakDecNum1, width=bar_width, color='b', label='player1')
    plt.bar(index_Player2, PeakDecNum2, width=bar_width, color='c', label='player2')
    '''
    xGames2 = xGames + bar_width
    # #The number of separated sub-peaks of the overlapping peak of player1
    plt.bar(xGames, PeakDecNum2, width=bar_width, color='g', label='player2')
    plt.legend(loc='best', fontsize=16) 
    plt.ylabel('Number of sub-peaks', size=20, labelpad=10)  
    plt.xlabel('Number of games', size = 20, labelpad = 10)
    plt.tick_params(labelsize=16)
    plt.show()
 
state = []               
draw_peakDecon(arena.episodes, state, x, y) 


##########   Calculate delta Cp
# x: temperature, y: heat capacity Cp, Tm: peak temperature
def calculDelatCp(x, y, Tm, amp, cen, wid):
    for i in x:
        if(i == Tm):
            x_i = i
    #deltaHmA = simps(Gaussian(x, popt[j],popt[j+1], popt[j+2])*160, x)  
    deltaHmA = simps(Gaussian(x_i, amp, cen, wid)*160, x_i)  
    deltaHA = simps(Gaussian(x_i, amp, cen, wid)*160, x_i)  
    Cp = (deltaHA - deltaHmA)/(x_i-x_i)
    return Cp

"""
Monte Carlo control of peak decomposition
"""
from utils import epsilon_greedy_policy
from mpl_toolkits.axes_grid1 import make_axes_locatable

class MC_Player(Player):
    '''Players with Monte Carlo control capabilities
    '''
    def __init__(self, name = "", A = None, display = False):
        super(MC_Player, self).__init__(name, A, display)
        self.Q = {}   
        self.Nsa = {} 
        self.total_learning_times = 0
        self.policy = self.epsilon_greedy_policy # 
        self.learning_method = self.learn_Q 
    
    def learn_Q(self, episode, r): # Learn the Q value from the state sequence
        '''Learn from an Episode
        '''
        for s, a in episode:
            nsa = get_dict(self.Nsa, s, a)
            set_dict(self.Nsa, nsa+1, s, a)
            q = get_dict(self.Q, s,a)
            set_dict(self.Q, q+(r-q)/(nsa+1), s, a)
        self.total_learning_times += 1
    
    def reset_memory(self):
        '''Forget previous learning experience
        '''
        self.Q.clear()
        self.Nsa.clear()
        self.total_learning_times = 0

    
    def epsilon_greedy_policy(self, dealer, epsilon = None):
        '''The greedy strategy here is with epsilon parameters
        '''
        #player_points, _ = self.get_points()
        #reward, player_rmse, dealer_rmse, rate_rmse = self.reward_of() 
        player_rmse = self.get_rmse()
        #if player_rmse < 0.01:
        if player_rmse < 0.02:
            return self.A[1] # Stop peaking
        #if player_rmse > 0.01:
        if player_rmse > 0.02:
            return self.A[0]
        else:
            A, Q = self.A, self.Q
            s = self.get_state_name(dealer)
            if epsilon is None:
                epsilon = 1.0/(1 + 4 * math.log10(1+player.total_learning_times))
            return epsilon_greedy_policy(A, s, Q, epsilon)
    
    def greedy_policy(self, dealer, epsilon = None):
        player_rmse = self.get_rmse()
        #if player_rmse < 0.01:
        if player_rmse < 0.02:
            return self.A[1] # stop peaking
        #if player_rmse > 0.01:
        if player_rmse > 0.02:
            return self.A[0]
        else:
            A, Q = self.A, self.Q
            s = self.get_state_name(dealer)
            if epsilon is None:
                epsilon = 1.0/(1 + 4 * math.log10(1+player.total_learning_times))
            return epsilon_greedy_policy(A, s, Q, epsilon)


#A=["Continue to add the number of sub-peaks","Stop adding the number of sub peaks"]
A=["continue", "stop"]
display = False
player = MC_Player(A = A, display = display)
dealer = Dealer(A = A, display = display)
# Create a scene
arena = Arena(x = x, y = y, A = A, display = display)
arena.play_games(dealer = dealer, player = player,num = 10, show_statistic = True)

def draw_policy(policy, A, Q, epsilon):
    def value_of(a):
        if a == A[0]:
            return 0
        else:
            return 1
    x_range = np.arange(1, 10) 
    y_range = np.arange(1, 5) 
    X, Y = np.meshgrid(x_range, y_range)
    row, col = X.shape
    row, col = 10, 3
    Z = np.zeros((row,col))
    for i in range(1, 10): 
        for j in range(1, 3): 
            s = j, i
            s = str_key(s)
            a = policy(A, s, Q, epsilon)
            Z[i-1,j-1] = value_of(a) 
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121)
    surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[0, 10, 0, 3])
    plt.xticks(x_range)
    plt.yticks(y_range)
    plt.gca().invert_yaxis()
    ax.set_xlabel('Player2\'s sub-peak numbers')
    ax.set_ylabel('Player1\'s sub-peak numbers')
    ax.grid(color='w', linestyle='-', linewidth=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(surf, ticks=[0,1], cax=cax)
    #cbar.ax.set_yticklabels(['0 (continue)','1 (stop)'])
    cbar.ax.set_yticklabels(['0 (continue)','1 (stop)'])
    plt.show()
            
draw_value(player.Q, is_q_dict=True, A = player.A) # Draw state behavior versus value
draw_policy(epsilon_greedy_policy, player.A, player.Q, epsilon = 1e-10) #Drawing strategy


