import numpy as np

class _Switcher(object):

	def __init__(self,check1,check2):
		'''
		This object does something. What exactly, I'm not sure...
		
		Inputs
		======
		check1 :
		
	 
		check1 : 

			
		'''	
		self.check1=check1
		self.check2=check2
	def indirect(self,i):
		'''
		This method will return one of the other methods of this class
		to be used as a function based on the index provided.
		
		Inputs
		======
		i : int
			Integer corresponding to the function to be returned, where
			the function is called "number_i", i.e. if i == 3, this will
			return the pointer to member function "_Switcher.number_3()"
		
		Returns
		=======
		method : callable
		
		'''
		method_name='number_'+str(i)
		method=getattr(self,method_name,lambda :'Invalid')		
		return method()
	def number_1(self):
		if np.size(self.check1) == 1:
			check3=np.where((self.check2 ==1) & (self.check1 > 0.7))[0]
		else:
			check3=np.where((self.check2 ==1) & (self.check1 > 0.7))[1]
		lambda_max_brho =   4.0
		lambda_max_bz   = 100.0
		returns=(check3,lambda_max_brho,lambda_max_bz)
		return(returns)
	def number_2(self):
		if np.size(self.check1) == 1:
			check3=np.where((self.check2 ==0) & (self.check1 > 0.7))[0]
		else:		
			check3=np.where((self.check2 ==0) & (self.check1 > 0.7))[1]
		lambda_max_brho =   4.0
		lambda_max_bz   = 20.0
		returns=(check3,lambda_max_brho,lambda_max_bz)
		return(returns)
	def number_3(self):
		if np.size(self.check1) == 1:
			check3=np.where((self.check2 == 1) & (self.check1 < 1) & (self.check1 < 0.7))[0]
		else:
			check3=np.where((self.check2 == 1) & (self.check1 < 1) & (self.check1 < 0.7))[1]
		lambda_max_brho =  40.0
		lambda_max_bz   = 100.0
		returns=[check3,lambda_max_brho,lambda_max_bz]
		return(returns)
	def number_4(self):
		if np.size(self.check1) == 1:
			check3=np.where((self.check2 ==0) & (self.check1 < 1) & (self.check1 < 0.7))[0]
		else:
			check3=np.where((self.check2 ==0) & (self.check1 < 1) & (self.check1 < 0.7))[1]
		lambda_max_brho =  40.0
		lambda_max_bz   = 20.0
		returns=[check3,lambda_max_brho,lambda_max_bz]
		return(returns)
	def number_5(self):
		if np.size(self.check1) == 1:
			check3=np.where((self.check2 ==1) & ( self.check1 < 0.1))[0]
		else:
			check3=np.where((self.check2 ==1) & ( self.check1 < 0.1))[1]
		lambda_max_brho =  100.0
		lambda_max_bz   = 100.0
		returns=[check3,lambda_max_brho,lambda_max_bz]
		return(returns)
	def number_6(self):
		if np.size(self.check1) == 1:
			check3=np.where((self.check2 ==0) & (self.check1 < 0.1))[0]
		else:
			check3=np.where((self.check2 ==0) & (self.check1 < 0.1))[1]		
		lambda_max_brho =  100.0
		lambda_max_bz   = 20.0
		returns=[check3,lambda_max_brho,lambda_max_bz]
		return(returns)
