use crate::ev::EV;
use crate::types::State;

pub struct Charger {
    pub max_charging_rate: f64,
    pub connected_ev: Option<usize>,
}

impl Charger {
    pub fn new(max_charging_rate: f64) -> Self {
        Charger {
            max_charging_rate,
            connected_ev: None,
        }
    }
}

pub struct Env {
    pub evs: Vec<EV>,
    pub charger: Charger,
    pub prices: Vec<f64>,
    pub num_hours: usize,
    current_hour: usize,
    battery_cap_cache: Option<f64>,
}

impl Env {
    pub fn new(evs: Vec<EV>, charger: Charger, prices: Vec<f64>, num_hours: usize) -> Self {
        Env {
            evs,
            charger,
            prices,
            num_hours: num_hours,
            current_hour: 0,
            battery_cap_cache: None
        }
    }    


    /// returns `(reward, kwh)`
    pub fn step(&mut self, action: Option<usize>) -> (f64, f64) {
        let mut total_reward = 0.0;
        self.charger.connected_ev = action;
        let mut rate = 0.0;
    
        if let Some(ev_idx) = self.charger.connected_ev {
            let ev = &mut self.evs[ev_idx];
            rate = ev.max_charging_rate.min(self.charger.max_charging_rate);
            ev.step(1, self.charger.max_charging_rate);
            let price = self.prices[self.current_hour];
            let mut reward = 0.0;
            reward += -price * rate - 0.01;
            total_reward += reward;
        }

        if self.current_hour == self.num_hours - 1 && self.get_total_battery_level() < self.get_total_battery_capacity() {
            total_reward += -1000.0 - 2000.0 * (1.0 - self.get_total_battery_level() / self.get_total_battery_capacity());
            //total_reward += -2000.0;
        }


        self.current_hour += 1;
        (total_reward, rate)
    }

    pub fn reset(&mut self) {
        self.current_hour = 0;
        self.evs.iter_mut().for_each(|ev| ev.reset());
    }

    pub fn get_hash_key(&self) -> State {
        let evs = self.evs.iter().map(|ev| ev.battery_level as usize).collect();
        (self.current_hour, evs)
    }
    
    pub fn get_total_battery_level(&self) -> f64 {
        self.evs.iter().map(|ev| ev.battery_level).sum()
    }

    pub fn get_total_battery_capacity(&mut self) -> f64 {
        return *self.battery_cap_cache.get_or_insert(self.evs.iter().map(|ev| ev.battery_capacity).sum())
    }

    pub fn hours_to_charge(&mut self) -> f64 {
        self.evs.iter_mut().map(|ev| ev.hours_to_charge(self.charger.max_charging_rate)).sum()
    }
}

