# mb-downscaling

Temporal downscaling of glacier mass balance from seasonal observations.

## Installation & usage

Download the Python module [`mass_balance_interpolations.py`](/mass_balance_interpolations.py) to your local working directory. You can then import and use it in your Python script. For example:

```py
import mass_balance_interpolations as mbi

mbi.sine_interpolation_from_mean_balances(annual_balance=-1, balance_amplitude=3)
```

See the Jupyter Notebook ([`demo.ipynb`](/demo.ipynb)) for a more complete example.
