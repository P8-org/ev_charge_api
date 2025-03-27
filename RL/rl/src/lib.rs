use std::f64::NEG_INFINITY;
use env::Charger;
use env::Env;
use ev::EV;
use pyo3::prelude::*;
use rand::seq::IndexedRandom;
use rand::Rng;
use rand::rng;
use fxhash::FxBuildHasher;
use types::InnerMap;
use types::QTable;

pub mod types;
pub mod env;
pub mod ev;

#[pymodule]
fn rl_scheduling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_schedule, m)?)
}

#[pyfunction]
fn get_schedule(num_hours: usize, alpha: f64, epsilon: f64, episodes: usize, bat_lvl: f64, bat_cap: f64, max_charging_rate: f64, prices: Vec<f64>, print: bool) -> PyResult<Vec<bool>> {
    let ev = EV::new(bat_cap, max_charging_rate, bat_lvl);
    let mut env = Env::new(vec![ev], Charger::new(max_charging_rate), prices, num_hours);

    let mut q_table = QTable::with_hasher(FxBuildHasher::default());

    train(&mut q_table, episodes, &mut env, alpha, 1.0, epsilon, print);

    let schedule = run(&mut env, &q_table, print);

    return Ok(schedule.iter().map(|o| o.is_some()).collect());
}


pub fn train(
    q_table: &mut QTable,
    episodes: usize,
    env: &mut Env,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    print_progress: bool
) {
    let mut rng = rng();

    let mut actions = vec![None];
    for i in 0..env.evs.len() {
        actions.push(Some(i));
    }


    
    let mut empty_hashmap = InnerMap::with_hasher(FxBuildHasher::new());
    actions.iter().for_each(|&a| {
        empty_hashmap.insert(a, 0.0);
    });

    q_table.insert(env.get_hash_key(), empty_hashmap.clone());
    let mut best_reward = NEG_INFINITY;
    let mut best_pct = 0.0;

    for ep in 0..episodes {
        env.reset();
        let mut total_reward = 0.0;
        let mut key = env.get_hash_key();

        for _ in 0..env.num_hours {
            let action = if rng.random_range(0.0..=1.0) < epsilon {
                //rng.random_range(0..2)
                actions.choose(&mut rng).unwrap().clone()
            } else {
                let best_action = q_table.get(&key).unwrap().iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0.clone();
                best_action
            };
            let (reward, _) = env.step(action);
            let next_key = env.get_hash_key();

            if !q_table.contains_key(&next_key){
                q_table.insert(next_key.clone(), empty_hashmap.clone());
            }

            // update q value
            let max_next_q = q_table.get(&next_key).unwrap().iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().1.clone();
            let current_q = q_table.get_mut(&key).unwrap().get_mut(&action).unwrap();
            *current_q = (1.0 - alpha) * *current_q + alpha * (reward + gamma * max_next_q);

            total_reward += reward;
            key = next_key;
        }
        
        if total_reward > best_reward {
            best_reward = total_reward;
            best_pct = env.get_total_battery_level() / env.get_total_battery_capacity() * 100.0;
        }
        if print_progress && ep % 10000 == 0 {
            println!("E{:8}   best R: {:.2}, lvl: {:.0}%, a: {:.3}, e: {:.3}", ep, best_reward, best_pct, alpha, epsilon);
        }
    }
}


pub fn run(env: &mut Env, q_table: &QTable, print: bool) -> Vec<Option<usize>> {
    env.reset();
    let mut charging_schedule = vec![];
    let mut total_reward = 0.0;
    for hour in 0..env.num_hours {
        let key = env.get_hash_key();
        let action = q_table.get(&key).unwrap_or(&InnerMap::with_hasher(FxBuildHasher::new())).iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap_or((&None, &0.0)).0.clone();
        charging_schedule.push(action);
        let (reward, kwh) = env.step(action);
        total_reward += reward;
        if print {
            let cap = env.get_total_battery_capacity();
            println!(
                "T{:2}, price: {:.2}, charged: {:.2} {:?}, lvl = {:.0}/{:.0}",
                hour + 1,
                env.prices[hour],
                kwh.abs(),
                action,
                env.get_total_battery_level(),
                cap
            );
        }
    }
    if print {
        println!("Total cost: {:.2}", total_reward);
    }
    charging_schedule
}



pub fn generate_prices(time_slots: usize, min_price: f64, max_price: f64) -> Vec<f64> {
    let mut rng = rng();
    (0..time_slots).map(|_| rng.random_range(min_price..max_price)).collect()
}

pub fn fixed_prices(num_hours: usize) -> Vec<f64> {
    [
        330, 129, 675, 835, 1000, 1019, 917, 905, 820, 500, 300, 150, 721, 722, 671, 754, 1075, 1144, 999, 737, 629, 580, 566, 576, 625, 719, 771, 1051,
        1233, 1170, 918, 863, 789, 730,
    ]
    //[0.30, 0.25, 0.15, 0.20, 0.10, 0.35, 0.40, 0.18, 0.12, 0.22, 0.55,0.12,0.68,1.00,0.11,0.92]
    .iter()
    .take(num_hours)
    .map(|&x| x as f64 / 1000.0)
    .collect()
}