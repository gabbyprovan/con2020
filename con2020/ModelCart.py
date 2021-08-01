import numpy as np
from .OldModel import OldModel

def ModelCart(x,y,z,mu_i=139.6,i_rho=16.7,r0=7.8,r1=51.4,d=3.6,xt=9.3,
			xp=-24.2,equation_type='hybrid',no_error_check=False,
			Cartesian=True):
	'''
	This is function calls Model() but accepts cartesian (system III) 
	coordinates as opposed to spherical polar (i.e. IAU_JUPITER in SPICE).
	
	Code to calculate the perturbation magnetic field produced by the 
	Connerney (CAN) current sheet, which is represented by a finite disk 
	of current.	This disk has variable parameters including the current 
	density mu0i0, inner edge R0, outer edge R1, thickness D. The disk 
	is centered on the magnetic equator (shifted in longitude and tilted 
	according to the dipole field parameters of an internal field model 
	like VIP4 or JRM09). This 2020 version includes a radial current per 
	Connerney et al. (2020), 
	https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JA028138

	Inputs
	======
	x : float
		x in Rj (System III)
	y : float
		y in Rj (System III)
	z : float
		z in Rj (System III)
	mu_i : float
		mu0i0/2 term (current sheet current density), in nT
	i_rho : float
		azimuthal current term from Connerney et al., 2020
	r0 : float
		Inner edge of current disk in Rj
	r1 : float
		Outer edge of current disk in Rj
	d : float
		Current sheet half thickness in Rj
	xt : float
		Dipole tilt in degrees
	xp : float
		Dipole longitude (right handed) in degrees
	equation_type: str
		Define method for calculating the current sheet field, may be 
		one of the following: 'hybrid'|'analytic'|'integral'
		See notes below for more information.
	no_error_check : bool
		Do not do extra checks that inputs are valid.		
	Cartesian : bool
		If True, magnetic field is returned in Cartesian coordinates,
		otherwise the returned values are in spherical polar coordinates

	Returns
	========
	Magnetic field in SIII coordinates (right handed)
	bx : float
		Bx, in nT (or Radial field if Cartesian == False)
	by : float
		By, in nT (or Meridional field if Cartesian == False)
	bz : float
		Bz, in nT (or Azimuthal field if Cartesian == False)

	This code takes a hybrid approach to calculating the current sheet 
	field, using the integral equations in some regions and the analytic 
	equations in others. Following Connerney et al. 1981, figure A1, and 
	Edwards et al. (2001), figure 2, the choice of integral vs. analytic 
	equations is most important near rho = r0 and z = 0.
	
	By default, this code uses the analytic equations everywhere except 
	|Z| < D*1.5 and |Rho-R0| < 2.

	Analytic Equations
	==================
	For the analytic equations, we use the equations provided in 
	Connerney et al., 1981 
	(https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JA086iA10p08370)
	   
	Other analytic approximations to the CAN sheet equations are 
	provided by Edwards et al. 2001: 
	https://www.sciencedirect.com/science/article/abs/pii/S0032063300001641
	
	
	Integral Equations
	==================
	For the integral equations we use the Bessel functions from 
	Connerney et al. 1981, eqs. 14, 15, 17, 18.
	
	We do not integrate lambda from zero to infinity, but vary the 
	integration limit depending on the value of the Bessel functions.
	
	Other Notes
	===========
	
	Keyword equation_type can be set to 'integral' or 'analytic' if the 
	user wants to force using the integral or analytic equations ,by 
	Marissa Vogt, March 2021.
	
	RJ Wilson did some speedups and re-formatting of lines, also March 2021
	'''
	#convert to spherical polar coords
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arccos(z/r)
	phi = (np.arctan2(y,x) + (2*np.pi)) % (2*np.pi)

	#call Model()
	return OldModel(r,theta,phi,mu_i=mu_i,i_rho=i_rho,r0=r0,r1=r1,d=d,xt=xt,
				xp=xp,equation_type=equation_type,
				no_error_check=no_error_check,Cartesian=Cartesian)
	
	
