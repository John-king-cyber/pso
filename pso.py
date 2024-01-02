import numpy as np 
import matplotlib.pyplot as plt

def cost_func(x):
    return np.sum(np.square(x))

class Swarm :
    """
    ******Parameters of PSO****
    
    obj_funct  = objective function
    low_bound  = lowwer bound
    upp_bound  = upper bound
    nPar       = number of particles
    nIt        = number of iterations
    nVar       = number of variables in the search space
    c1         = social efficient
    c2         = cognitive efficient
    w          = inertia weight 
    damp_coef  = damping coefficient    
    
    """
    # initializing parameters of swarm
    def __init__(self, obj_func = cost_func, low_bound: int = -100, upp_bound: int = 100,
                 nPar:int = 100, nIt: int =2000, c1:float = 2.0,
                 c2:float = 2.0, w:float = 0.9, damp_coef:float = 0.1) -> None:
        self.nPar            = nPar  
        self.low_bound       = low_bound 
        self.upp_bound       = upp_bound 
        self.nIt             = nIt
        self.nVar            = 10
        self.vMin            = - 0.2 * (upp_bound - low_bound)
        self.vMax            = 0.2 * (upp_bound - low_bound)
        self.c1              = c1
        self.c2              = c2
        self.w               = w
        self.damp_coef       = damp_coef
        self.obj_func        = obj_func
        self.position        = np.random.uniform(self.low_bound, self.upp_bound, (self.nPar,self.nVar))
        self.velocity        = np.random.uniform(self.vMin, self.vMax, (self.nPar, self.nVar))
        self.pbest           = np.copy(self.position)
        self.cost            = np.zeros(self.nPar)
        self.cost[:]         = self.obj_func(self.position[:])
        self.pbest_cost      = np.copy(self.cost)
        self.gbest           = np.min(self.cost)
        self.gbest_pos       = self.position[np.argmin(self.cost)]
        self.gbest_cost      = self.pbest_cost[np.argmin(self.gbest)]
        self.best_cost_iter  = np.zeros(self.nIt)  #best cost after iteration
    
    #updating particle velocity and position
    def update_position_velocity(self,dis:bool = True) -> None:
            for it in range(self.nIt):
                for i in range(self.nPar):
                   self.velocity[i] = (self.w*self.velocity[i] + self.c1*np.random.rand(self.nVar)* (self.pbest[i] - self.position[i])
                                       + self.c2*np.random.rand(self.nVar)*(self.gbest - self.position[i])) 
                   self.velocity[i] = self.limitV(self.velocity[i])
                   
                   self.position[i] = self.position[i] + self.velocity[i]
                   self.position[i] = self.limitX(self.position[i])
                   self.cost[i] = self.obj_func(self.position[i])
                   if self.cost[i] < self.pbest_cost[i]:
                       self.pbest[i] = self.position[i]
                       self.pbest_cost[i] = self.cost[i]
                       if self.pbest_cost[i] < self.gbest_cost:
                           self.gbest = self.pbest[i]
                           self.gbest_cost = self.pbest_cost[i]
                self.best_cost_iter[it] = self.gbest_cost
                self.w *= self.damp_coef
                if dis:
                    print(f'Iteration {it}:   gbest cost : {self.gbest_cost} ')
    #constraining position to search space   
    def limitX(self, x:list ) -> list:
        for i in range(len(x)):
            if x[i] > self.upp_bound :
                x[i] = self.upp_bound
            if x[i] < self.low_bound:
                x[i] = self.low_bound
        return x
    #constraining velocity
    def limitV(self, v: list ) -> list:
        for i in range(len(v)):
            if v[i] > self.vMax :
                v[i] = self.vMax
            if v[i] < self.vMin:
                v[i] = self.vMin
        return v
    #displaying outpout
    def display(self) -> None :
        plt.semilogy(self.best_cost_iter)
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('cost')
        plt.show()
        
        
pso= Swarm(nIt=200)
pso.update_position_velocity(dis=True)
pso.display()



