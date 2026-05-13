# File System Outline

```text
final/
│
├── README.md
├── config.py
│
├── data_prep.py          # load + clean SBB tables
├── delay_model.py        # train/load/predict delay probabilities
├── algorithm.py          # robust route planner
├── vis.py                # map + widgets + route display
├── validation.py         # tests, backtesting, metrics
├── demo.ipynb            # launchable final "app"
│
├── artifacts/            # saved .parquets/data/tables to reduce re-run time of scripts
│   └── ...
│
└── tests/                # for validation
    ├── test_data_prep.py
    ├── test_delay_model.py
    └── test_algorithm.py
```

# Code Flow

```text
config.py
   ↓
data_prep.py
   ↓
delay_model.py
   ↓
algorithm.py
   ↓
vis.py
   ↓
demo.ipynb
```