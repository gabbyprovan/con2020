import numpy as np

def _SmallRhoApproxEdwards(rho,zmd,zpd,mui2,a2):

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
	print('Original')
	#these values appear to be used for all parts of the process
	#so let's calculate them all
	zpd = z + D
	zmd = z - D
	a2 = a*a
	

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
	print('Edwards')
	#these values appear to be used for all parts of the process
	#so let's calculate them all
	zpd = z + D
	zmd = z - D
	a2 = a*a
	

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
	
	
	if Edwards:
		return _AnalyticEdwards(rho,z,D,a,mui2)
	else:
		return _AnalyticOriginal(rho,z,D,a,mui2)


def _Finite(rho,z,D,a,mui2,Edwards=True):

	zpd = z + D
	zmd = z - D
	a2 = a*a
		
	if Edwards:
		return _SmallRhoApproxEdwards(rho,zmd,zpd,mui2,a2)
	else:
		return _SmallRhoApprox(rho,z,zmd,zpd,mui2,a2,D)
