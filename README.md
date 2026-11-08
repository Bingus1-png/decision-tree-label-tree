# Decision Trees and Label Noise

Two Python experiments on how incorrect labels affect decision-tree performance and model selection. Both use synthetic two-moon data.

| Experiment | Question | Design and outputs |
| --- | --- | --- |
| [1. Training label noise](experiment_1_training_noise/) | How does limiting tree depth affect test accuracy when training labels are flipped? | 20 repetitions, 320 model fits, 2 CSV files, and 8 figures |
| [2. Validation label noise](experiment_2_validation_noise/) | How do flipped validation labels change the selected depth and its accuracy on an untouched test set? | 30 repetitions, 480 candidate model fits, 480 selections, 3 CSV files, and 5 figures |

Run both experiments from the repository root with Python 3.12 or newer. The script writes the results and plots to the two folders above.

```bash
python -m pip install numpy pandas matplotlib scikit-learn
python experiment.py
```
