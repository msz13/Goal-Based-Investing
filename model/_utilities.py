import numpy as np


class Goals:

    def __init__(self, goals: dict) -> None:
        self.__k_dict = self.__convert_goals_to_k(goals)

    def get_k(self):
        return self.__k_dict
    
    def get_investment_period(self):
        return np.fromiter(self.__k_dict.keys(), np.int32).max() + 2
    
    def get_highest_cost_for_time(self, t):
        return self.__k_dict.get(t)[:,0].max() if (self.__k_dict.get(t) is not None) else 0 
    
    def get_k_array(self, t):
        return self.__k_dict.get(t)
        
    
    def __convert_goals_to_k(self, goals):
        result = {}

        for goal in goals:
            result[goal['time']] = np.array([[goal['cost'], goal['utility']]]) 

        return result

