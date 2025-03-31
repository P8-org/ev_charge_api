pub struct EV {
    pub battery_capacity: f64,
    pub max_charging_rate: f64,
    pub battery_level: f64,
    pub initial_battery_level: f64,
}

impl EV {
    pub fn new(battery_capacity: f64, max_charging_rate: f64, battery_level: f64) -> Self {
        EV {
            battery_capacity,
            max_charging_rate,
            battery_level,
            initial_battery_level: battery_level,
        }
    }

    pub fn is_full(&self) -> bool {
        self.battery_level >= self.battery_capacity
    }

    pub fn step(&mut self, action: usize, charger_max_charging_rate: f64) {
        self.battery_level = (self.battery_level + self.max_charging_rate.min(charger_max_charging_rate) * action as f64)
            .min(self.battery_capacity);
    }


    pub fn reset(&mut self) {
        self.battery_level = self.initial_battery_level;
    }

    pub fn hours_to_charge(&mut self, max_charging_rate: f64) -> f64 {
        self.reset();
        let mut total_hours = 0.0;
        let time_step = 1.0;

        while self.battery_level < self.battery_capacity {
            let rate = self.max_charging_rate.min(max_charging_rate);
            let energy_added = rate * time_step;
            self.battery_level += energy_added;
            if self.battery_level > self.battery_capacity {
                self.battery_level = self.battery_capacity;
            }
            total_hours += time_step;
        }
        self.reset();
        total_hours
    }
}