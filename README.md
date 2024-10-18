## TD learning implementation

This repo implements Q-learning (currently done), SARSA and Expected SARSA on the cliffwalking reinforcement learning problem. 

-------------------

To run the repo/notebook, download [uv](https://docs.astral.sh/uv/) and run the following command:

```bash
uv venv
source .venv/bin/activate # .venv\Scripts\activate on Win
uv sync
```

This will install the following packages:

* Numpy
* gymnasium
* tqdm