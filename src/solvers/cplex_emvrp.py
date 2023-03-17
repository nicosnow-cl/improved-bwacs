class CPLEX_EMVRP:
    def __init__(self, depot, clients, loc_x, loc_y, demands, tare, vehicle_capacity, k, metric):
        from docplex.mp.model import Model
        import numpy as np
        from itertools import permutations
        import matplotlib.pyplot as plt
        
        self.depot = depot
        self.clients = clients
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.clients_ids = {}
        self.ordered_clients = []
        self.q = demands
        self.demands_array = []
        self.coords_matrix = []
        self.V = [self.depot] + self.clients
        self.A = []
        self.d = {}
        self.Q0 = tare
        self.Q = vehicle_capacity
        self.m = k
        self.metric = metric
        self.mdl = Model('EMVRP')
        self.x = None
        self.y = None
        self.np = np
        self.permutations = permutations
        self.plt = plt
     
    def setInitialParameters(self):
        self.A = list(self.permutations(self.V, 2))
        
        if self.metric == 'manhattan':
            ord_ = 1
        else:
            ord_ = 2
            
        self.d = {(i, j): self.np.linalg.norm(self.np.array((self.loc_x[j], self.loc_y[j])) - self.np.array((self.loc_x[i], self.loc_y[i])),
                                              ord = ord_) for i, j in self.A}
    
    def defineConstraints(self):
        ## Restricciones 5 y 6 // todos los vehiculos son usados
        self.mdl.add_constraints(self.mdl.sum(self.x[depot, i] for i in self.clients) == self.m for depot in [self.depot])
        self.mdl.add_constraints(self.mdl.sum(self.x[i, depot] for i in self.clients) == self.m for depot in [self.depot])

        ## Restricciones 7 y 8 // restricciones de grado para cada nodo
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j] for i in self.V if i != j) == 1 for j in self.clients)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j] for j in self.V if i != j ) == 1 for i in self.clients)
        
        ## Restriccion 9 // conservación clasica de la ecuacion de flujo que prohibe cualquier subviaje ilegal
        self.mdl.add_constraints((self.mdl.sum(self.y[i, j] for j in self.V if i != j) - self.mdl.sum(self.y[j, i] for j in self.V if i != j)) == self.q[i] for i in self.clients)
        
        ## Restriccion 10 // inicializa el recorrido en el primer arco de cada ruta
        self.mdl.add_constraints(self.y[self.depot, i] == (self.Q0 * self.x[self.depot, i]) for i in self.clients)
        
        ## Restricciones 11 y 12
        self.mdl.add_constraints(self.y[i, j] <= ((self.Q + self.Q0 - self.q[j]) * self.x[i, j]) for i, j in self.A)
        self.mdl.add_constraints(self.y[i, j] >= ((self.Q0 + self.q[i]) * self.x[i, j]) for i, j in self.A)
    
    def printSolution(self, arcs):
        self.plt.rc('font', size = 16)         # controls default text sizes
        self.plt.rc('axes', titlesize = 9)     # fontsize of the axes title
        self.plt.rc('axes', labelsize = 12)    # fontsize of the x and y labels
        self.plt.rc('xtick', labelsize = 9)    # fontsize of the tick labels
        self.plt.rc('ytick', labelsize = 9)    # fontsize of the tick labels
        self.plt.rc('legend', fontsize = 9)    # legend fontsize
        self.plt.rc('figure', titlesize = 16)  # fontsize of the figure title
        self.plt.figure(figsize=(22, 12))

        for pos, i in enumerate(self.clients):
            self.plt.plot(self.loc_x[i], self.loc_y[i], c = 'blue', marker = 'o', markersize = 12)
            self.plt.annotate('$q_{%d}=%d$'%(i, self.q[i]), (self.loc_x[i] + 6, self.loc_y[i] - 10))
                
        for x, y in arcs:
            self.plt.plot([self.loc_x[x], self.loc_x[y]], [self.loc_y[x], self.loc_y[y]], c = 'green', alpha = 0.5)
        
        self.plt.plot(self.loc_x[self.depot], self.loc_y[self.depot], c = 'r', marker = 's', markersize = 12)
        self.plt.annotate('DEPOT', (self.loc_x[self.depot] + 5, self.loc_y[self.depot] + 15))
        self.plt.show()
    
    def solve(self):
        self.setInitialParameters()
        
        self.x = self.mdl.binary_var_dict(self.A, name = 'x')
        self.y = self.mdl.continuous_var_dict(self.A, name = 'y')    
        
        self.mdl.minimize(self.mdl.sum(self.d[i, j] * self.y[i, j] for i, j in self.A))
        
        self.defineConstraints()
        
        solution = self.mdl.solve(log_output = True)
        
        arcos_activos = [k for k in self.A if self.x[k].solution_value > 0.9]
        pesos_activos = {(k): self.y[k].solution_value for k in self.A if self.y[k].solution_value > 0}

        print('Arcos: ' + str(arcos_activos))
        print('\nPesos: ' + str(pesos_activos))

        solution_quality = 0
        for i, j in arcos_activos:
            solution_quality += (self.d[(i, j)] * pesos_activos[(i, j)])

        print('\nCalidad de la solución: ' + str(solution_quality))
        
        self.printSolution(arcos_activos)