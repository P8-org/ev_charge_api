HOW TO RUN:
- create virtual env: https://fastapi.tiangolo.com/virtual-environments/#create-a-virtual-environment
- activate virtual env: https://fastapi.tiangolo.com/virtual-environments/#activate-the-virtual-environment
- install libs: `pip install -r requirements.txt`
- install rust RL lib:
    - install rust compiler: https://www.rust-lang.org/tools/install
    - run `maturin develop -m RL/rl/Cargo.toml --release`
- to run server: `fastapi dev app.py`