#In part 1 of the project, you are supposed to do a project on dynamic 
# modelling of the thermodynamic process of an energy storage plant and simulate
#the operation of the plant for a relevant period of operation time.


#First, we shall model the thermodynamic process of an energy storage plant
#We shall look at the charging, storage, and discharging of the storage plant

#What are we interested in returning for the plant?
#We would like to define a charging function that takes the flow of water, 
#the temperature of the water, and duration as inputs and returns:
#the stored heat of the tank after the charging.

from typing import Optional, Union
import pandas as pd

def _load_dni_input(dni_input: Union[str, float, int, pd.Series]) -> Union[float, pd.Series]:
	"""Return DNI as float or pandas Series depending on input type."""
	if isinstance(dni_input, (float, int)):
		return float(dni_input)
	if isinstance(dni_input, pd.Series):
		return dni_input.astype(float)
	if isinstance(dni_input, str):
		df = pd.read_csv(dni_input, header=0)
		try:
			df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
			df = df.set_index(df.columns[0])
		except Exception:
			# leave as-is if parsing fails
			pass
		# prefer column named like the example
		if "dni_wm2" in df.columns:
			return df["dni_wm2"].astype(float)
		# otherwise take the first numeric column
		for col in df.columns:
			try:
				s = pd.to_numeric(df[col], errors="coerce")
				if s.notna().any():
					s.index = df.index
					return s
			except Exception:
				continue
		raise ValueError("Could not find a numeric DNI column in CSV")
	raise TypeError("dni_input must be a path, float, int, or pandas Series")