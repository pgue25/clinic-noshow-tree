# Clinic no-shows: should the clinic overbook this slot?

Decision-tree classifier that predicts appointment no-shows and values an overbooking rule against the clinic's current "hold every slot" policy.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analysis.py
```

`analysis.py` reproduces every number on the slides (seed 42, stratified 70/30 split). Outputs go to `output/`:

- `results.txt`: full printed results (data, tree rules, confusion matrices, decisions for D and E, dollar values)
- `results.csv`: baseline vs v1 metrics and value
- `tree_v1.png`, `confusion_matrices.png`, `value_per_appt.png`: charts

Data: `data/clinic_appointments.csv` (400 appointments, 8 columns).
