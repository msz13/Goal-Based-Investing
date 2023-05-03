import numpy as np
import numpy.testing as npt
import pytest


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



def transactions(inflow, assets_weights, prices):
    return (inflow*assets_weights) / prices


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
            transaction = transactions(self.__inflows[t]-self.__planed_outflows[t] ,self.__assets_weights,self.__prices[:,t])
            

        #self.__assets_shares = shares.sum(0)

        

@pytest.fixture
def prices():
    return np.array([[[30.00, 50.00],       
                        [32.7, 51.5],
                        [25.18, 56.14],
                        [23.67, 56.70],
                        [30.53, 57.83],
                        [34.2, 59.57]],
                        [[30.0,	50.0],
                         [28.2,	52.00],
                         [24.82, 47.84],
                         [24.07, 51.67],
                         [25.76, 54.25],
                         [31.42, 55.88]]])

      


test_data= [(np.array([10000,1000,1000,1000,1000]),[16306.50,15439.19]),
       (np.array([10000,0,0,0,0]), [11605.60, 10754.40])]

@pytest.mark.parametrize('inflows,expected',test_data, ids=['inflows', 'no inflows'])
def test_should_get_porfolios_last_value_(prices, inflows, expected):  
        
        
    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=np.array([0.6,0.4]),        
        inflows = inflows,
        outflows = np.zeros(5)
        )


    portfolios_simulator.run()
    
    wealth = portfolios_simulator.get_porfolio_final_value()

    npt.assert_array_equal(wealth, expected)


def test_should_get_porfolios_outflows_(prices):  
        
    inflows = np.array([10000,1000,1000,1000,1000])

    planned_outflows = np.array([0,0,0,0,16000])

    expected_final_value = np.array([306, 0])
    expectet_outflows = np.array([15000, 15000])

    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=np.array([0.6,0.4]),       
        inflows = inflows,
        outflows = planned_outflows
        )


    portfolios_simulator.run()
    
    wealth = portfolios_simulator.get_porfolio_final_value()
    outflows = portfolios_simulator.get_outflows()    

    #npt.assert_array_equal(wealth, expected_final_value)
    npt.assert_array_equal(outflows, expectet_outflows)



''' outflows = {
    1: 10000,
    2: 100000,
    3: 50000
}

outflows = np.array([10000,100000,50000])
 '''