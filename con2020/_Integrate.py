import numpy as np

def _Integrate(j,dl):
	'''
	Integrate j assuming constant dl
	
	'''
	
	return 0.5*dl*np.sum(j[1:] + j[:-1])
