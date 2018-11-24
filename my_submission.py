'''

2017 IFN680 Assignment

Instructions: 
    - You should implement the class PatternPosePopulation

'''

import numpy as np
import matplotlib.pyplot as plt

import pattern_utils
import population_search



# ------------------------------------------------------------------------------

class PatternPosePopulation(population_search.Population):
    '''
    
    '''

    def __init__(self, W, pat):
        '''
        Constructor. Simply pass the initial population to the parent
        class constructor.
        @param
          W : initial population
        '''
        self.pat = pat
        super().__init__(W)

    def evaluate(self):
        '''
        Evaluate the cost of each individual.
        Store the result in self.C
        That is, self.C[i] is the cost of the ith individual.
        Keep track of the best individual seen so far in 
            self.best_w 
            self.best_cost 
        @return 
           best cost of this generation            
        '''
        height, width = self.distance_image.shape[:2]
        
        #clip those coords out of img
        
        np.clip(self.W[:, 0], 0, width - 1, self.W[:, 0])
        np.clip(self.W[:, 1], 0, height - 1, self.W[:, 1])

        # Call function in pattern_util.evaluate,find each cost of W[i,:]

        for i in range(0, len(self.W)):

            self.C[i], _ = (self.pat).evaluate(self.distance_image, self.W[i])

        i_min = self.C.argmin()
        cost_min = self.C[i_min]

        if cost_min < self.best_cost:
            self.best_w = self.W[i_min].copy()
            self.best_cost = cost_min

        return cost_min

    def mutate(self):
        '''
        Mutate each individual.
        The x and y coords should be mutated by adding with equal probability 
        -1, 0 or +1. That is, with probability 1/3 x is unchanged, with probability
        1/3 it is decremented by 1 and with the same probability it is 
        incremented by 1.
    
        The mutation for the angle and scale coefficient is the same as for the x and y coords.
        @post:
          self.W has been mutated.
        '''

        assert self.W.shape == (self.n, 4)
        
        # add rangom -1,0,1 to mutate the coords
        coords = np.random.choice([-1, 0, 1], 2 * self.n, replace=True, p=[1 / 3, 1 / 3, 1 / 3]).reshape(-1, 2)
        
        # convert degree into radians
        angle = np.random.choice([-1, 0, 1], self.n, replace=True, p=[1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)
        
        # add rangom -1,0,1 to mutate the scale
        scale = np.random.choice([-1, 0, 1], self.n, replace=True, p=[1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)

        mutations = np.concatenate([coords, (angle *  np.pi/180), scale], axis=1)

        self.W = self.W + mutations

    def set_distance_image(self, distance_image):

        self.distance_image = distance_image


# ------------------------------------------------------------------------------

def initial_population(region, scale=10, pop_size=20):
    '''
    
    '''
    # initial population: exploit info from region
    rmx, rMx, rmy, rMy = region
    W = np.concatenate((
        np.random.uniform(low=rmx, high=rMx, size=(pop_size, 1)),
        np.random.uniform(low=rmy, high=rMy, size=(pop_size, 1)),
        np.random.uniform(low=-np.pi, high=np.pi, size=(pop_size, 1)),
        np.ones((pop_size, 1)) * scale
        # np.random.uniform(low=scale*0.9, high= scale*1.1, size=(pop_size,1))
    ), axis=1)
    return W


# ------------------------------------------------------------------------------
def test_particle_filter_search():
    '''
    Run the particle filter search on test image 1 or image 2of the pattern_utils module
    
    '''

    if True:
        # use image 1
        imf, imd, pat_list, pose_list = pattern_utils.make_test_image_1(True)
        ipat = 2  # index of the pattern to target
    else:
        # use image 2
        imf, imd, pat_list, pose_list = pattern_utils.make_test_image_2(True)
        ipat = 0  # index of the pattern to target

    # Narrow the initial search region
    pat = pat_list[ipat]  # (100,30, np.pi/3,40),
    #    print(pat)
    xs, ys = pose_list[ipat][:2]
    region = (xs - 20, xs + 20, ys - 20, ys + 20)
    scale = pose_list[ipat][3]

    pop_size = 300
    W = initial_population(region, scale, pop_size)

    pop = PatternPosePopulation(W, pat)
    pop.set_distance_image(imd)

    pop.temperature = 5

    Lw, Lc = pop.particle_filter_search(10, log=True)

    plt.plot(Lc)
    plt.title('Cost vs generation index')
    plt.show()

    print(pop.best_w)
    print(pop.best_cost)

    pattern_utils.display_solution(pat_list,
                                   pose_list,
                                   pat,
                                   pop.best_w)

    pattern_utils.replay_search(pat_list,
                                pose_list,
                                pat,
                                Lw)
    return pop.best_w, pop.best_cost

def test(times):
    
        n = times
        sum_cost = 0
        cost_list=[]
        best_w_list=[]
        
        for i in range(0,n):
            
            best_w, best_c = test_particle_filter_search()
            sum_cost = sum_cost + best_c
            cost_list.append(best_c)
            best_w_list.append(best_w)
       
        x=np.arange(1, n+1, 1)  
        y=np.array(cost_list)
            
        plt.plot(x,y,"or")
        plt.show()
            
        average_cost = sum_cost / n
        
        
        print("the sum_cost is ",sum_cost)
        
        print("the everage of best cost of {} times is {}".format(n,average_cost))
        print("the best cost list is {}".format(cost_list))
        print("the best weighted list is {}".format(best_w_list))
            


# ------------------------------------------------------------------------------

if __name__ == '__main__':

    test(100)
    


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        

#        
#    def test_2():
#        '''
#        Run the particle filter search on test image 2 of the pattern_utils module
#        
#        '''
#        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(False)
#        pat = pat_list[0]
#        
#        #region = (100,150,40,60)
#        xs, ys = pose_list[0][:2]
#        region = (xs-20, xs+20, ys-20, ys+20)
#        
#        W = initial_population_2(region, scale = 30, pop_size=40)
#        
#        pop = PatternPosePopulation(W, pat)
#        pop.set_distance_image(imd)
#        
#        pop.temperature = 5
#        
#        Lw, Lc = pop.particle_filter_search(40,log=True)
#        
#        plt.plot(Lc)
#        plt.title('Cost vs generation index')
#        plt.show()
#        
#        print(pop.best_w)
#        print(pop.best_cost)
#        
#        
#        
#        pattern_utils.display_solution(pat_list, 
#                          pose_list, 
#                          pat,
#                          pop.best_w)
#                          
#        pattern_utils.replay_search(pat_list, 
#                          pose_list, 
#                          pat,
#                          Lw)
#    
#    #------------------------------------------------------------------------------        
#
