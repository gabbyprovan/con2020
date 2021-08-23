import numpy as np


class _Switcher(object):
	def __init__(self,check1,check2):
		'''
		This object will take a couple of checks (1 and 2) and 
		calculate a bunch of stuff based on those checks for each zcase.
		
		Inputs
		======
		check1 : float
			abs(abs(z) - d)
		check2 : bool
			abs(z) <= 1.1*d
		
		
		'''
		
		#store the checks
		self.check1 = check1
		self.check2 = check2
		
		#populate each of the different index arrays we want
		self.checks = np.zeros(6,dtype='object')
		ch2eq0 = self.check2 == 0 #used for case 2,4,6
		ch2eq1 = self.check2 == 1 #used for case 1,3,5
		ch1ge07 = self.check1 >= 0.7 #used for 1 and 2
		ch1lt07 = self.check1 < 0.7 #used for 3 and 4
		ch1ge01 = self.check1 >= 0.1 #used for 3 and 4
		ch1lt01 = self.check1 < 0.1 #used for 5 and 6
		self.checks[0] = np.where(ch2eq1 & ch1ge07)[0]
		self.checks[1] = np.where(ch2eq0 & ch1ge07)[0]
		self.checks[2] = np.where(ch2eq1 & ch1ge01 & ch1lt07)[0]
		self.checks[3] = np.where(ch2eq0 & ch1ge01 & ch1lt07)[0]
		self.checks[4] = np.where(ch2eq1 & ch1lt01)[0]
		self.checks[5] = np.where(ch2eq0 & ch1lt01)[0]
		
		#lambda max b rho and z
		self.lambda_max_brho = [4,4,40,40,100,100]
		self.lambda_max_bz = [100,20,100,20,100,20]

	def FetchCase(self,zcase):
		'''
		Fetch the appropriate set of indices and lambdas based on the 
		current zcase (from 1 to 7)
		
		Inputs
		======
		zcase : int
			Which z-case to use.
			
		Returns
		=======
		inds : int
			Array of indices where check1 and check2 match the input
			zcase.
		lambda_max_brho : float
			Lambda limit over which to integrate the Bessel functions
			for the Brho integral.
		lambda_max_bz : float
			Lambda limit over which to integrate the Bessel functions
			for the Bz integral.
		
		'''
		
		#convert to integer
		z = np.int32(zcase) - 1
		
		#return stuff
		return self.checks[z],self.lambda_max_brho[z],self.lambda_max_bz[z]
