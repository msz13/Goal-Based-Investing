import numpy as np


class Goals:

    def __init__(self, goals: dict) -> None:
        self.__k_dict = self.__convert_goals_to_k(goals)

    def get_k(self):
        return self.__k_dict
    
    def get_investment_period(self):
        return np.fromiter(self.__k_dict.keys(), np.int32).max() + 1
    
    def get_k_array(self, t):
        return self.__k_dict.get(t)
    
    def get_costs_for_time(self, t):
        return self.get_k_array(t)[:,0] if (self.__k_dict.get(t) is not None) else None 
    
    def get_highest_costs(self):
        costs = [0] * self.get_investment_period()
        goals_t = self.__k_dict.keys()

        for t in goals_t:
            costs[t] = self.get_k_array(t)[:,0].max()

        return costs                  
    
    def __convert_goals_to_k(self, goals):
        result = {}

        for goal in goals:
            if result.get(goal['time']) is None:
                result[goal['time']] = np.array([[goal['cost'], goal['utility']]]) 
            else:
                result[goal['time']] = np.append(result[goal['time']], np.array([[goal['cost'], goal['utility']]]),axis=0)

        return result

