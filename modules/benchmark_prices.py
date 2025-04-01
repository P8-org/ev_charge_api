class Benchmark:
    def __init__(self, schedule, prices, battery_capacity, max_charging_rate):
        self.schedule = schedule
        self.prices = prices
        self.battery_capacity = battery_capacity
        self.max_charging_rate = max_charging_rate
    
        
    def optimized_schedule_price(self):
        optimized_prices = [a * b for a,b in zip(self.schedule,self.prices)]
        optimized_price = 0
        for i in optimized_prices:
            optimized_price += i
        return optimized_price
        

    def greedy_schedule_price(self):
        charge_list = []
        current_charge = 0

        for hour in range(len(self.schedule)):
            if current_charge < self.battery_capacity:
                charge = min(self.max_charging_rate, self.battery_capacity - current_charge)
                current_charge += charge
            else:
                charge = 0
            charge_list.append(charge)
        #get total price of naive schedule
        return sum([a*b for a,b in zip(charge_list,self.prices)])
    
    def compare(self):
        print(f"total price of optimized schedule: {self.optimized_schedule_price()}")
        print(f"total price of greedy schedule: {self.greedy_schedule_price()}")
        
        


        
           

