import numpy as np

def _Integrate(j,dl):
	'''
	Integrate j assuming constant dl
	
	Inputs
	======
	j : float
		Array of function values to be integrated
	dl : float
		Single floating point value to define the distance between each 
		point in j.
		
	Returns
	=======
	j integrated using the trapezoid rule.
	
	'''
	
	return 0.5*dl*np.sum(j[1:] + j[:-1])
