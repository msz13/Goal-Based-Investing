import tensorflow as tf

@tf.function()
def calculate_outflows(goal, portfolio_value):
  

  if (goal is None):
    return 0
      
  goal_target = goal[0] 
  goal_max_allocation = goal[1]

  goal_allocation = portfolio_value * goal_max_allocation
  return tf.where(goal_target <= goal_allocation, goal_target, goal_allocation)

@tf.function()
def transaction(inflow, shares_owned, assets_weights, prices, goal):
  current_assets_value = tf.multiply(shares_owned, prices)
  current_value = tf.reduce_sum(current_assets_value,axis=1) + inflow
  outflows = calculate_outflows(goal, current_value)
  expected_value = current_value - tf.cast(outflows, tf.float32)
  delta_value = tf.reshape(expected_value,(expected_value.shape[0],1)) * assets_weights - current_assets_value
  delta_shares =  delta_value / prices

  return delta_shares, outflows

@tf.function()
def softmax(x):
  z = tf.math.exp(x)
  return z / tf.reduce_sum(z)

class PortfoliosSimulator:
    
    def __init__(self) -> None:
        pass

    def set_params(self, assets_prices, assets_weights, inflows, goals):
        self.__prices = tf.constant(assets_prices)
        self.__assets_weights = assets_weights        
        self.__inflows = inflows
        self.__goals = goals
        self.T = tf.reduce_max(list(self.__goals.keys())).numpy()+1
        self.__shares = tf.zeros((tf.shape(self.__prices).numpy()[0],tf.shape(self.__prices).numpy()[2]),dtype=tf.dtypes.float32)

    def get_porfolio_final_value(self):
        return tf.reduce(self.__shares * self.__prices[:,-1],1)
    
    def get_outflows(self):
        return self.__outflows
    
    def get_shares(self):
        return self.__shares
    
    ''' def get_goals_achevement_probabilities(self):
        goals_targets = list(map(lambda x: x[0],  list(self.__goals.values())))
        return calculate_prob_of_goal_achivement(goals_targets, self.get_outflows()) '''
                     
    @tf.function()
    def __call__(self):          
       
        self.__outflows = []


        for t in range (self.T):
            delta_shares, t_outflows= transaction(self.__inflows[t], self.__shares, self.__assets_weights[t],self.__prices[:,t], self.__goals.get(t))
            self.__shares += delta_shares
            if (self.__goals.get(t) is not None):
                self.__outflows.append(t_outflows) 
        
        return self.__outflows
          
def goal_probs(outflows, goal_targets):
    number_of_scenarios = outflows[0].shape[0]
    targets = goal_targets * number_of_scenarios
    sum_goals_outflows = tf.reduce_sum(outflows, axis=1) 
    return sum_goals_outflows / targets