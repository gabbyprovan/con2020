import numpy as np

def _SmallRhoApproxEdwards(rho,zmd,zpd,mui2,a2):
	'''
	Small rho approximations calculated using equations 9a and 9b of 
	Edwards et al 2001.
	
	Inputs
	======
	rho : float
		rho current sheet coordinate (Rj).
	zmd : float
		z - d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	zpd : float
		z + d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	mui2 : float
		Current sheet current density (nT).
	a2 : float
		Inner/Outer edge of current disk squared (Rj^2).
	
	Returns
	=======
	Brho : float
		rho-component of the magnetic field.
	Bz : float
		z-component of the magnetic field.	
	
	'''
	zmd2 = zmd*zmd
	zpd2 = zpd*zpd
	f1 = np.sqrt(zmd2 + a2)
	f2 = np.sqrt(zpd2 + a2)
	f1cubed = f1*f1*f1
	f2cubed = f2*f2*f2	

	#calculate the terms which make equations 9a and 9b
	rhoov2 = rho/2.0
	rho2ov4 = rhoov2*rhoov2
	rho3ov16 = rho2ov4*rhoov2/2.0
	
	#these bits are used to form 9a
	f3a = f1*f1
	f4a = f2*f2
	f3 = (a2 - 2*zmd2)/(f3a*f3a*f1)
	f4 = (a2 - 2*zpd2)/(f4a*f4a*f2)
	
	terma0 = rhoov2*(1/f1 - 1/f2)
	terma1 = rho3ov16*(f3 - f4)
	
	Brho = mui2*(terma0 + terma1)
	
	#now equation 9b
	termb0 = np.log((zpd + f2)/(zmd + f1))
	termb1 = rho2ov4*(zpd/f2cubed - zmd/f1cubed)
	Bz = mui2*(termb0 + termb1)
	
	return Brho,Bz
	
def _SmallRhoApprox(rho,z,zmd,zpd,mui2,a2,D):
	'''
	Small rho approximations calculated using equations A1 and A2 of 
	Connerney et al 1981.
	
	Inputs
	======
	rho : float
		rho current sheet coordinate (Rj).
	z : float
		z current sheet coordinate (Rj).
	zmd : float
		z - d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	zpd : float
		z + d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	mui2 : float
		Current sheet current density (nT).
	a2 : float
		Inner/Outer edge of current disk squared (Rj^2).
	D : float
		Current sheet half-thickness (Rj).
			
	Returns
	=======
	Brho : float
		rho-component of the magnetic field.
	Bz : float
		z-component of the magnetic field.	
	
	'''
	zmd2 = zmd*zmd
	zpd2 = zpd*zpd
	f1 = np.sqrt(zmd2 + a2)
	f2 = np.sqrt(zpd2 + a2)
	f1cubed = f1*f1*f1
	f2cubed = f2*f2*f2	

	Brho = mui2*(rho/2.0)*(1/f1 - 1/f2)
	Bz = mui2*(2*D*(1/np.sqrt(z*z + a2)) - ((rho*rho)/4)*((zmd/f1cubed) - (zpd/f2cubed)))

	return Brho,Bz
	
def _LargeRhoApproxEdwards(rho,z,zmd,zpd,mui2,a2,D):
	'''
	Small rho approximations calculated using equations 13a and 13b of 
	Edwards et al 2001.
	
	Inputs
	======
	rho : float
		rho current sheet coordinate (Rj).
	z : float
		z current sheet coordinate (Rj).
	zmd : float
		z - d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	zpd : float
		z + d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	mui2 : float
		Current sheet current density (nT).
	a2 : float
		Inner/Outer edge of current disk squared (Rj^2).
	D : float
		Current sheet half-thickness (Rj).
			
	Returns
	=======
	Brho : float
		rho-component of the magnetic field.
	Bz : float
		z-component of the magnetic field.	
	
	'''	
	#some common variables
	zmd2 = zmd*zmd
	zpd2 = zpd*zpd
	rho2 = rho*rho
	f1 = np.sqrt(zmd2 + rho2)
	f2 = np.sqrt(zpd2 + rho2)
	f1cubed = f1*f1*f1
	f2cubed = f2*f2*f2	
	
	#equation 13a
	terma0 = (1/rho)*(f1 - f2)
	terma1 = (rho*a2/4)*(1/f2cubed - 1/f1cubed)
	terma2 = (2.0/rho)*z.clip(max=D,min=-D)
	Brho = mui2*(terma0 + terma1 + terma2)
	
	#equation 13b
	termb0 = np.log((zpd + f2)/(zmd + f1))
	termb1 = (a2/4)*(zpd/f2cubed - zmd/f1cubed)
	Bz = mui2*(termb0 + termb1)
	
	return Brho,Bz
	
def _LargeRhoApprox(rho,z,zmd,zpd,mui2,a2,D):
	'''
	Small rho approximations calculated using equations A7 and A8 of 
	Connerney et al 1981.
	
	Inputs
	======
	rho : float
		rho current sheet coordinate (Rj).
	z : float
		z current sheet coordinate (Rj).
	zmd : float
		z - d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	zpd : float
		z + d, where z is the z current sheet coordinate and d is the
		current sheet half-thickness (Rj).
	mui2 : float
		Current sheet current density (nT).
	a2 : float
		Inner/Outer edge of current disk squared (Rj^2).
	D : float
		Current sheet half-thickness (Rj).
	
	Returns
	=======
	Brho : float
		rho-component of the magnetic field.
	Bz : float
		z-component of the magnetic field.	
	
	'''	
	#some common variables
	zmd2 = zmd*zmd
	zpd2 = zpd*zpd
	rho2 = rho*rho
	f1 = np.sqrt(zmd2 + rho2)
	f2 = np.sqrt(zpd2 + rho2)
	f1cubed = f1*f1*f1
	f2cubed = f2*f2*f2	
	
	#Brho
	termr0 = (1.0/rho)*(f1 -f2 + 2*z.clip(max=D,min=-D))
	termr1 = (a2*rho/4.0)*(1/f1cubed - 1/f2cubed)
	Brho = mui2*(termr0 - termr1)
	
	#Bz
	termz0 = 2*D/np.sqrt(z*z + rho*rho)
	termz1 = (a2/4.0)*(zmd/f1cubed - zpd/f2cubed)
	Bz = mui2*(termz0 - termz1)
	

	
	return Brho,Bz
	

def _AnalyticOriginal(rho,z,D,a,mui2):
	
	
	
	#these values appear to be used for all parts of the process
	#so let's calculate them all
	zpd = z + D
	zmd = z - D
	a2 = a*a
	
	#switch between vector and scalar
	if np.size(rho) == 1:
		if rho >= a:
			Brho,Bz = _LargeRhoApprox(rho,z,zmd,zpd,mui2,a2,D)
		else:
			Brho,Bz = _SmallRhoApprox(rho,z,zmd,zpd,mui2,a2,D)
	else:

		#use rho and a to decide whether to use large or small approx
		lrg = np.where(rho >= a)[0]
		sml = np.where(rho < a)[0]

		#create output arrays
		Brho = np.zeros(rho.size,dtype='float64')
		Bz = np.zeros(rho.size,dtype='float64')
		
		#fill them
		Brho[lrg],Bz[lrg] = _LargeRhoApprox(rho[lrg],z[lrg],zmd[lrg],zpd[lrg],mui2,a2,D)
		Brho[sml],Bz[sml] = _SmallRhoApprox(rho[sml],z[sml],zmd[sml],zpd[sml],mui2,a2,D)		


	return Brho,Bz
	
	
def _AnalyticEdwards(rho,z,D,a,mui2):
	'''
	This function will calculate the model using the Edwards et al., 
	2001 equations. 
	
	https://www.sciencedirect.com/science/article/abs/pii/S0032063300001641
	
	Inputs
	======
	rho : float
		This should be a numpy.ndarray of the rho coordinate.
	z : float
		This should also be a numpy.ndarray of the z coordinate.
	D : float
		Constant half-thickness of the current sheet in Rj.
	a : float
		Inner edge of the current sheet in Rj.
	mui2 : float
		mu_0 * I_0/2 - current sheet current density in nT.
		
	Returns
	=======
	Brho : float
		array of B in rho direction
	Bz : float
		array of B in z direction
	
	'''
	#these values appear to be used for all parts of the process
	#so let's calculate them all
	zpd = z + D
	zmd = z - D
	a2 = a*a
	
	#choose scalar or vectorized version of the code
	if np.size(rho) == 1:
		if rho >= a:
			Brho,Bz = _LargeRhoApproxEdwards(rho,z,zmd,zpd,mui2,a2,D)
		else:
			Brho,Bz = _SmallRhoApproxEdwards(rho,zmd,zpd,mui2,a2)
			
	else:

		#use rho and a to decide whether to use large or small approx
		lrg = np.where(rho >= a)[0]
		sml = np.where(rho < a)[0]
		
		#create output arrays
		Brho = np.zeros(rho.size,dtype='float64')
		Bz = np.zeros(rho.size,dtype='float64')
		
		#fill them
		Brho[lrg],Bz[lrg] = _LargeRhoApproxEdwards(rho[lrg],z[lrg],zmd[lrg],zpd[lrg],mui2,a2,D)
		Brho[sml],Bz[sml] = _SmallRhoApproxEdwards(rho[sml],zmd[sml],zpd[sml],mui2,a2)
	
	return Brho,Bz
	
def _Analytic(rho,z,D,a,mui2,Edwards=True):
	'''
	Calculate the analytical version of the model.
	
	Inputs
	======
	rho : float
		rho coordinate (Rj).
	z : float
		z coordinate (Rj).
	D : float
		Constant half-thickness of the current sheet in Rj.
	a : float
		Inner edge of the current sheet in Rj.
	mui2 : float
		mu_0 * I_0/2 - current sheet current density in nT.
	Edwards : bool
		If True, use Edwards et al 2001 equations, otherwise use the
		Connerney et al 1981 equations.
		
	Returns
	=======
	Brho : float
		array of B in rho direction
	Bz : float
		array of B in z direction
	
	'''
	
	if Edwards:
		return _AnalyticEdwards(rho,z,D,a,mui2)
	else:
		return _AnalyticOriginal(rho,z,D,a,mui2)


def _Finite(rho,z,D,a,mui2,Edwards=True):
	'''
	Calculate the analytical version of the model using the outer edge
	of the current sheet - this provides a field to subtract from that
	calculated using the inner edge which will take into account the 
	finite size of the current sheet.
	
	Inputs
	======
	rho : float
		rho coordinate (Rj).
	z : float
		z coordinate (Rj).
	D : float
		Constant half-thickness of the current sheet in Rj.
	a : float
		Outer edge of the current sheet in Rj.
	mui2 : float
		mu_0 * I_0/2 - current sheet current density in nT.
	Edwards : bool
		If True, use Edwards et al 2001 equations, otherwise use the
		Connerney et al 1981 equations.
		
	Returns
	=======
	Brho : float
		array of B in rho direction
	Bz : float
		array of B in z direction
	
	'''
	zpd = z + D
	zmd = z - D
	a2 = a*a
		
	if Edwards:
		#the following function will split it into large and small rho 
		#approximations as appropriate
		return _AnalyticEdwards(rho,z,D,a,mui2)
	else:
		return _SmallRhoApprox(rho,z,zmd,zpd,mui2,a2,D)
		
def _FiniteEdwards(rho,z,D,a,mui2):
	'''
	Calculate the Edwards et al version of the model using the outer edge
	of the current sheet - this provides a field to subtract from that
	calculated using the inner edge which will take into account the 
	finite size of the current sheet.
	
	Inputs
	======
	rho : float
		rho coordinate (Rj).
	z : float
		z coordinate (Rj).
	D : float
		Constant half-thickness of the current sheet in Rj.
	a : float
		Outer edge of the current sheet in Rj.
	mui2 : float
		mu_0 * I_0/2 - current sheet current density in nT.
	Edwards : bool
		If True, use Edwards et al 2001 equations, otherwise use the
		Connerney et al 1981 equations.
		
	Returns
	=======
	Brho : float
		array of B in rho direction
	Bz : float
		array of B in z direction
	
	'''
	return _AnalyticEdwards(rho,z,D,a,mui2)

def _FiniteOriginal(rho,z,D,a,mui2):
	'''
	Calculate the Connerney et al version of the model using the outer edge
	of the current sheet - this provides a field to subtract from that
	calculated using the inner edge which will take into account the 
	finite size of the current sheet.
	
	Inputs
	======
	rho : float
		rho coordinate (Rj).
	z : float
		z coordinate (Rj).
	D : float
		Constant half-thickness of the current sheet in Rj.
	a : float
		Outer edge of the current sheet in Rj.
	mui2 : float
		mu_0 * I_0/2 - current sheet current density in nT.
	Edwards : bool
		If True, use Edwards et al 2001 equations, otherwise use the
		Connerney et al 1981 equations.
		
	Returns
	=======
	Brho : float
		array of B in rho direction
	Bz : float
		array of B in z direction
	
	'''

	zpd = z + D
	zmd = z - D
	a2 = a*a	
	return _SmallRhoApprox(rho,z,zmd,zpd,mui2,a2,D)
