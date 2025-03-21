use fxhash::FxHashMap;

pub type State = (usize, Vec<usize>); // (hour, battery levels)
pub type Action = Option<usize>;
pub type InnerMap = FxHashMap<Action, f64>;
pub type QTable = FxHashMap<State, InnerMap>;