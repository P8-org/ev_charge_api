use fxhash::FxBuildHasher;
use rl::{env::{Charger, Env}, ev::EV, fixed_prices, generate_prices, run, train, types::QTable};
use std::time::Instant;


fn main() {
    let num_hours = 16;
    let random_prices = false;

    let price_per_hour = if random_prices {
        generate_prices(num_hours, 0.1, 2.0)
    } else {
        fixed_prices(num_hours)
    };


    let alpha = 0.2; // learning rate
    let gamma = 1.0; // discount factor
    let epsilon = 0.1; // exploration rate
    let episodes = 100_000;

    let evs = vec![
        EV::new(1000.0, 300.0, 0.0),
        //EV::new(63.0, 300.0, 0.0),
    ];

    let mut env = Env::new(evs, Charger::new(300.0), price_per_hour, num_hours);

    let start = Instant::now();

    let mut q_table: QTable = QTable::with_hasher(FxBuildHasher::default());

    train(&mut q_table, episodes, &mut env, alpha, gamma, epsilon, false);

    let end = Instant::now();
    println!("Training time: {:?}", end - start);

    run(&mut env, &q_table, true);
    //println!("{:?}", charging_schedule);

    // println!("Finished battery level: {}%", (ev.battery_level / ev.battery_capacity * 100.0) as usize);
    sequential_charging(&mut env);
    println!("Hours to charge: {}", env.hours_to_charge());
    println!("QTable size: {}", q_table.len() + q_table.values().map(|v| v.len()).sum::<usize>());

    // println!("Total cost: {:.2}", cost(&charging_schedule, &mut ev, &price_per_hour));
    // println!("Optimal(not really) cost: {:.2}", simple_optimal_cost(&mut ev, &price_per_hour));
    // println!("Normal charging cost: {:.2}", simple_cost(&mut ev, &price_per_hour));

    // // calculate average savings in percent over 1000 runs. also generate a random price list for each run to simulate different days and train
    // let mut total_savings = 0.0;
    // let runs = 100;
    // for _ in 0..runs {
    //     let price_per_hour = generate_prices(num_hours, 0.1, 2.0);
    //     let mut ev = EV::new(600.0, 400.0, 0.0);
    //     let mut q_table: HashMap<(usize, Vec<usize>, Option<usize>), f64> = HashMap::new();
    //     train(&mut q_table, episodes, &mut env, alpha, gamma, epsilon);
    //     let charging_schedule = run(&mut ev, num_hours, &q_table, &price_per_hour, false);
    //     let cost = cost(&charging_schedule, &mut ev, &price_per_hour);
    //     let optimal_cost = simple_optimal_cost(&mut ev, &price_per_hour);
    //     total_savings += (optimal_cost - cost) / optimal_cost;
    // }
    // println!("Average savings: {:.2}%", total_savings / runs as f64 * 100.0);
    
}



fn sequential_charging(env: &mut Env) -> Vec<Option<usize>> {
    env.reset();
    let mut charging_schedule = vec![];
    let total_hours_needed = env.hours_to_charge().ceil() as usize;

    // Create a vector of (price, hour) tuples and sort by price
    let mut price_hour_pairs= env.prices.clone();
    price_hour_pairs.sort_by(|a, b| a.partial_cmp(&b).unwrap());

    let mut total_reward = 0.0;
    let max_price = price_hour_pairs.iter().take(total_hours_needed).last().unwrap().clone();
    let mut ev_idx = 0;
    for price in env.prices.clone() {
        if ev_idx > env.evs.len() - 1 {
            charging_schedule.push(None);
            continue;
        }
        if env.evs[ev_idx].is_full() {
            ev_idx += 1;
        }
        if ev_idx > env.evs.len() - 1 {
            charging_schedule.push(None);
            continue;
        }
        if price > max_price {
            total_reward += env.step(None).0;
            charging_schedule.push(None);
        }
        else {
            total_reward += env.step(Some(ev_idx)).0;
            charging_schedule.push(Some(ev_idx));
        }
    }

    println!("Total cost for sequential charging: {:.2}, lvl: {:.0}/{:.0}", total_reward, env.get_total_battery_level(), env.get_total_battery_capacity());
    //println!("{:?}", charging_schedule);
    charging_schedule
}