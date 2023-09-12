# mb-downscaling

Temporal downscaling of glacier mass balance from seasonal observations.

## Installation & usage

Download the Python module [`mb_downscaling.py`](/mb_downscaling.py) to your local working directory. You can then import and use it in your Python script (requirements: `python >= 3.5`, `numpy >= 1`). For example:

```py
import mb_downscaling

mb_downscaling.downscale_seasonal_balances(winter_balance=2.5, summer_balance=-3.5)
```

See the Jupyter Notebook ([`demo.ipynb`](/demo.ipynb)) for a more complete example.
