import numpy as np
from dataclasses import dataclass 


goals = {
        5: {
        'cost': 10800,
        'max_outflow': 1
        }
    }

'''
transakcje 
| rodzaj        | ilosc shares | wartość | prowizja |
|wpłata         | 0 | 10000 | 0 |
|zakup          | 50 | 9900| 5 |
| sprzedarz     | 50 | 12000 | 6 |
|wyplata - cel1 | 0 | 12000| 0 |
'''

'''
INDEX-T 0 - T
wpłaty
kupno/sprzedarz
wypłata - cel1
wypłata - cel2
wypłata cel3
'''


    


class PortfoliosSimulator:

    def __init__(self) -> None:
        pass

    def set_params(self, assets_prices, assets_weights, inflows, outflows):
        self.__prices = assets_prices
        self.__assets_weights = assets_weights        
        self.__inflows = inflows
        self.__planed_outflows = outflows

    def get_porfolio_final_value(self):
        return np.around(np.sum(self.__shares.sum(0) * self.__prices[:,-1],axis=1),2)
    
    def get_outflows(self):
        return self.__outflows
                     
    
    def run(self):
        
        self.__shares = np.zeros((len(self.__inflows)+1,self.__prices.shape[0],self.__prices.shape[2]))
        
        for t in range (len(self.__inflows)):
            transaction = transactions(self.__inflows[t],self.__assets_weights,self.__prices[:,t])
            self.__shares = transaction.delta_shares

        #self.__assets_shares = shares.sum(0)
