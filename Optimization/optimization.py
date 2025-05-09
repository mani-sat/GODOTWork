import cvxpy as cp
import numpy as np
import pickle


class DL_optimizer_gw:
    def __init__(self, M, buffersize):
        self.M = M
        self.buffersize = buffersize
    
    def setup(self, Rsc:float, Rdl:float, Rgw:float):
        self.Rsc = Rsc
        self.Rdl = Rdl
        self.Rgw = Rgw
    
    def load_data(self, folder_path):
        with open(f"{folder_path}/som.pickle", 'rb') as f:
            full_som = pickle.load(f)
        f.close()
        with open(f"{folder_path}/los.pickle", "rb") as f:
            full_los= pickle.load(f)
        f.close()
        with open(f"{folder_path}/gwlos.pickle", "rb") as f:
            full_GW_los=pickle.load(f)
        f.close()
        with open(f"{folder_path}/rs.pickle", "rb") as f:
            Rsc_init = pickle.load(f)
        f.close()
        with open(f"{folder_path}/rd.pickle", "rb") as f:
            Rdl_init = pickle.load(f)
        f.close()
        with open(f"{folder_path}/gwr.pickle", "rb") as f:
            gw_rate_init=pickle.load(f)
        f.close()
        return full_som, full_los, full_GW_los, Rsc_init, Rdl_init, gw_rate_init

    def split_data(self, full_los):
        x=[0]+[1 if full_los[i-1]==1 and full_los[i]==0 else 0 for i in range(1,len(full_los))]
        burst_end=np.where(np.array(x) == 1)
        indexes=[0]
        for i in range(len(burst_end[0])-1):
            if burst_end[0][i+1]-burst_end[0][i]>600:
                indexes.append(burst_end[0][i])
        indexes.append(burst_end[0][-1])
        return indexes

    def get_data(self, full_sos, full_los, full_GW_los, indexes:list):
        """
        indexes: length 2 list of start index and end index
        """
        T_sos=full_sos[indexes[0]:indexes[1]]
        T_los=full_los[indexes[0]:indexes[1]]
        T_GW_los=full_GW_los[indexes[0]:indexes[1]]
        return T_sos, T_los, T_GW_los

    def optimize(self, som, los, GW_los, buffer):
        M = self.M
        buffer_size = self.buffersize
        sun_light = som
        los_sc = los
        GW_los = GW_los
        N=len(som)
        Rsc = self.Rsc
        Rdl = self.Rdl
        Rgw = self.Rgw

        # Decision variables
        T_sc = cp.Variable(N, boolean=True)
        T_dl = cp.Variable(N, boolean=True)
        T_idle = cp.Variable(N, boolean=True)
        T_gw = cp.Variable(N, boolean=True)

        # Constraints list
        constraints = []

        # Constraint 1: can only collect data when there is sunligt
        constraints += [T_sc <= sun_light]
        constraints += [T_gw <= GW_los]
        constraints += [T_dl <= los_sc]

        # Constraint 2: Only one variable avaiable at each time step
        constraints += [(T_sc + T_dl + T_idle + T_gw == 1)]

        # Constraint 3: We cannot transmit what we have not scienced
        for t in range(1, N):
            constraints += [
                cp.multiply(cp.sum(T_dl[:t]), Rdl) + cp.multiply(cp.sum(T_gw[:t]), Rgw) <= cp.multiply(cp.sum(T_sc[:t]), Rsc) + buffer
            ]
        
        # Constraint 4: Buffer size must not be exceeded
        for t in range(1,N):
            constraints += [
                cp.multiply(cp.sum(T_sc[:t]), Rsc) + buffer - cp.multiply(cp.sum(T_dl[:t]), Rdl) - cp.multiply(cp.sum(T_gw), Rgw) <= buffer_size
            ]

        # Ground station cappin
        rising_edges = cp.Variable(N, boolean=True)
        falling_edges = cp.Variable(N, boolean=True)
        GS = cp.Variable(N, boolean=True)

        # All initial falling edges must be zero
        constraints += [falling_edges[:M] == False]

        for t in range(N - M):
            constraints += [falling_edges[t+M] == rising_edges[t]]

            # There can not be a negative amount of falling edges
            constraints += [(cp.sum(rising_edges[:t]) - cp.sum(falling_edges[:t])) >= 0]
            #There can not be more than one GS window at once
            constraints += [(cp.sum(rising_edges[:t]) - cp.sum(falling_edges[:t])) <= 1]


        constraints += [rising_edges[N-M:N] == False]
        


        for t in range(N):
            constraints += [GS[t] == cp.sum(rising_edges[:t]) - cp.sum(falling_edges[:t])]
        constraints += [T_dl <= GS]
        constraints += [GS >=0 ]


        # Objective: minimize total downlink usage
        #objective = cp.Minimize(cp.sum(GS) - sc_total)
        objective = cp.Maximize(cp.multiply(cp.sum(T_dl), Rdl)-0.25*cp.sum(GS)+0.5*cp.multiply(cp.sum(T_gw), Rgw))
        
        # Solve
        problem = cp.Problem(objective, constraints)
        print(cp.installed_solvers())

        # Ensure the CBC solver is installed
        problem.solve(solver=cp.CBC, verbose=True)  # Using the CBC solver as an alternative
        # problem.solve(solver=cp.GUROBI, verbose=True)

        # Output
        # print("Status:", problem.status)
        # print("Total cost (timeslots slots used):", problem.value)
        # print("Total cost (downlink slots used):", sum(T_dl.value))
        # print("Total downlined:", sum(T_dl.value * np.array(Rdl)))
        # print("Total scienced:", sum(T_sc.value * np.array(Rsc)))
        # for t in range(N):
        #     print(f"t={t}: T_sc={T_sc.value[t]:.0f}, T_dl={T_dl.value[t]:.0f}, T_idle={T_idle.value[t]:.0f}, GS={GS.value[t]}")

        return T_sc.value, T_dl.value, T_gw.value, T_idle.value, GS.value
    
    def analyze_optimization(self, R_dl_tot, T_dl, T_sc, T_gw):
        pass #return buffer size and give output whether buffer is exceeded

if __name__=="__main__":
    M=60
    buffersize=1e7
    optimizer=DL_optimizer_gw(M, buffersize)
    #load data and get rates
    full_som, full_los, full_GW_los, Rsc, Rdl, Rgw=optimizer.load_data("Optimization/tmp_9_5")
    indexes=optimizer.split_data(full_los)
    #setup rates
    optimizer.setup(Rsc, Rdl, Rgw)

    day=0
    T_sos, T_los, T_GW_los=optimizer.get_data(full_som, full_los, full_GW_los, [indexes[day], indexes[day+1]])
    T_sc, T_dl, T_gw, T_idle, GS=optimizer.optimize(T_sos, T_los, T_GW_los, 0)
    save_folder="./Optimization/Optim_results"
    with open(f'./{save_folder}/T_sc.pickle', 'wb') as f:
        pickle.dump(T_sc, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/T_dl.pickle', 'wb') as f:
        pickle.dump(T_dl, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/T_gw.pickle', 'wb') as f:
        pickle.dump(T_gw, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/T_idle.pickle', 'wb') as f:
        pickle.dump(T_idle, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/GS.pickle', 'wb') as f:
        pickle.dump(GS, f, protocol=pickle.HIGHEST_PROTOCOL)