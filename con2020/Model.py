import numpy as np
from scipy.special import jv,j0,j1
from ._Switcher import _Switcher
from ._Analytic import _AnalyticEdwards,_AnalyticOriginal,_FiniteEdwards,_FiniteOriginal
from ._Integrate import _Integrate


class Model(object):
	def __init__(self,**kwargs):
		'''
		Code to calculate the perturbation magnetic field produced by the 
		Connerney (CAN) current sheet, which is represented by a finite disk 
		of current.	This disk has variable parameters including the current 
		density mu0i0, inner edge R0, outer edge R1, thickness D. The disk 
		is centered on the magnetic equator (shifted in longitude and tilted 
		according to the dipole field parameters of an internal field model 
		like VIP4 or JRM09). This 2020 version includes a radial current per 
		Connerney et al. (2020), 
		https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JA028138

		Keyword Arguments (shorthand keywords in brackets)
		=================
		mu_i_div2__current_density_nT (mu_i): float
			mu0i0/2 term (current sheet current density), in nT
		i_rho__azimuthal_current_density_nT (i_rho) : float
			azimuthal current term from Connerney et al., 2020
		r0__inner_rj (r0) : float
			Inner edge of current disk in Rj
		r1__outer_rj (r1) : float
			Outer edge of current disk in Rj
		d__cs_half_thickness_rj (d) : float
			Current sheet half thickness in Rj
		xt__cs_tilt_degs (xt) : float
			Current sheet tilt in degrees
		xp__cs_rhs_azimuthal_angle_of_tilt_degs (xp) : float
			Current sheet tilt longitude (right handed) in degrees
		equation_type: str
			Define method for calculating the current sheet field, may be 
			one of the following: 'hybrid'|'analytic'|'integral'
			See notes below for more information.
		error_check : bool
			If True (default) then inputs will be checked for potential errors.		
		CartesianIn : bool
			If True (default) the inputs to the model will be expected to be 
			in Cartesian right-handed System III coordinates. If False, then
			the inputs should be in spherical polar coordinates.
		CartesianOut : bool
			If True (default) the output magnetic field will be in Cartesian
			right-handed System III coordinates. Otherwise, the magnetic 
			field components produced will be radial, meridional and 
			azimuthal.

		Returns
		========
		model : object
			This is an instance of the con2020.Model object. To obtain the
			magnetic field, call the Field() member function, e.g.:
			
			model = con2020.Model()
			B = model.Field(x,y,z)

		This code takes a hybrid approach to calculating the current sheet 
		field, using the integral equations in some regions and the analytic 
		equations in others. Following Connerney et al. 1981, figure A1, and 
		Edwards et al. (2001), figure 2, the choice of integral vs. analytic 
		equations is most important near rho = r0 and z = 0.
		
		By default, this code uses the analytic equations everywhere except 
		|Z| < D*1.5 and |Rho-R0| < 2.

		Analytic Equations
		==================
		For the analytic equations, we use the equations  
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
		
		#list the default arguments here
		defargs = {	'mu_i'			: 139.6,
					'i_rho' 		: 16.7,
					'r0'			: 7.8,
					'r1'			: 51.4,
					'd'				: 3.6,
					'xt'			: 9.3,
					'xp'			: -24.2,
					'equation_type'	: 'hybrid',
					'error_check'	: True,
					'CartesianIn'	: True,
					'CartesianOut'	: True,
					'Edwards'		: True }
					
		#list the long names
		longnames = {	'mu_i'	: 'mu_i_div2__current_density_nT',
						'r0'	: 'r0__inner_rj',
						'r1'	: 'r1__outer_rj',
						'd'		: 'd__cs_half_thickness_rj',
						'xt'	: 'xt__cs_tilt_degs',
						'xp'	: 'xp__cs_rhs_azimuthal_angle_of_tilt_degs',
						'i_rho'	: 'i_rho__azimuthal_current_density_nT'		  }
						
		#check input kwargs
		#for those which exist (either in long or short name form) add
		#them to this object using the short name as the object tag
		#Otherwise use the default value
		
		#the input keys
		ikeys = list(kwargs.keys())
		
		#default keys
		dkeys = list(defargs.keys())
		
		#short and long name keys
		skeys = list(longnames.keys())
		lkeys = [longnames[k] for k in skeys]
			
		#loop through each one		
		for k in dkeys:
			if k in ikeys:
				#short name found in kwargs - add to this object
				setattr(self,k,kwargs[k])
			elif longnames.get(k,'') in ikeys:
				#long name found - add to object
				setattr(self,k,kwargs[longnames[k]])
			else:
				#key not found, use default
				setattr(self,k,defargs[k])
				
		
		#check for additional keys and issue a warning
		for k in ikeys:
			if not ((k in skeys) or (k in lkeys) or (k in dkeys)):
				print("Keyword argument {:s} unrecognized, ignoring.".format(k))
		
		#now do the checks
		self.equation_type = self.equation_type.lower()
		if not self.equation_type in ['analytic','hybrid','integral']:
			raise SystemExit("ERROR: 'equation_type' has unrecognized string - it should be 'analytic'|'hybrid'|'integral'")	
		
		ckeys = ['mu_i','i_rho','r0','r1','d','xt']
		for k in ckeys:
			x = getattr(self,k)
			if (x <= 0) or (np.isfinite(x) == False):
				raise SystemExit("'{:s}' should be greater than 0 and finite".format(k))	

		if (np.isfinite(self.xp) == False):
			raise SystemExit("'xp' should be finite")	
			
		#set the analytic function to use
		if self.Edwards:
			self._AnalyticFunc = _AnalyticEdwards
		else:
			self._AnalyticFunc = _AnalyticOriginal
			
		#set the analytic function to use for the outer bit of the current sheet
		if self.Edwards:
			self._Finite = _FiniteEdwards
		else:
			self._Finite = _FiniteOriginal		
			
		#set the integral functions (scalar and vector)
		if self.equation_type == 'analytic':
			self._ModelFunc = self._Analytic
		elif self.equation_type == 'integral':
			self._ModelFunc = self._Integral
		else:
			self._ModelFunc = self._Hybrid
				
		#some constants
		self.Deg2Rad = np.pi/180.0
		self.dipole_shift = self.xp*self.Deg2Rad # xp is longitude of the current sheet
		self.theta_cs = self.xt*self.Deg2Rad # current sheet tilt
		self.cosxp = np.cos(self.dipole_shift)
		self.sinxp = np.sin(self.dipole_shift)
		self.cosxt = np.cos(self.theta_cs)
		self.sinxt = np.sin(self.theta_cs)

		#this stuff is for integration
		self.dlambda_brho    = 1e-4  #% default step size for Brho function
		self.dlambda_bz      = 5e-5  #% default step size for Bz function
		
		#each of the following variables will be indexed by zcase (starting at 0)
		self.lambda_max_brho = [4,4,40,40,100,100]
		self.lambda_max_bz = [100,20,100,20,100,20]
		
		self.lambda_int_brho = []
		self.lambda_int_bz = []
		
		self.beselj_rho_r0_0 = []
		self.beselj_z_r0_0 = []

		for i in range(0,6):
			#save the lambda arrays
			self.lambda_int_brho.append(np.arange(self.dlambda_brho,self.dlambda_brho*(self.lambda_max_brho[i]/self.dlambda_brho),self.dlambda_brho))
			self.lambda_int_bz.append(np.arange(self.dlambda_bz,self.dlambda_bz*(self.lambda_max_bz[i]/self.dlambda_bz),self.dlambda_bz))
			
			#save the Bessel functions
			self.beselj_rho_r0_0.append(j0(self.lambda_int_brho[i]*self.r0))
			self.beselj_z_r0_0.append(j0(self.lambda_int_bz[i]*self.r0))
	
	def _ConvInputCart(self,x0,y0,z0):
		'''
		Converts input coordinates from Cartesian right-handed System 
		III to current sheet coordinates.
		
		Inputs
		======
		x0 : float
			System III x-coordinate (Rj).
		y0 : float
			System III y-coordinate (Rj).
		z0 : float
			System III z-coordinate (Rj).
			
		Returns
		=======
		x1 : float
			x current sheet coordinate
		y1 : float
			y current sheet coordinate
		z1 : float
			z current sheet coordinate
		cost : float
			cos(theta) - where theta is the colatitude
		sint : float
			sin(theta)
		cosp : float
			cos(phi) - where phi is east longitude
		sinp : float	
			sin(phi)
		'''

		rho2 = x0*x0 + y0*y0
		rho0 = np.sqrt(rho2)
		r = np.sqrt(rho2 + z0**2)

		cost = z0/r
		sint = rho0/r
		sinp = y0/rho0
		cosp = x0/rho0

		#rotate x and y to align with the current sheet longitude
		x = rho0*(cosp*self.cosxp + sinp*self.sinxp)
		y1 = rho0*(sinp*self.cosxp - cosp*self.sinxp)

		#rotate about y axis to align with current sheet
		x1 = x*self.cosxt + z0*self.sinxt
		z1 = z0*self.cosxt - x*self.sinxt		
			
		return x1,y1,z1,cost,sint,cosp,sinp
		
	def _ConvOutputCart(self,x1,y1,Brho1,Bphi1,Bz1):
		'''
		Convert the output magnetic field from cylindrical current 
		sheet coordinates to Cartesian right-handed System III
		coordinates.
		
		Inputs
		======
		x1 : float
			x-position in current sheet coords (Rj).
		y1 : float
			y-position in current sheet coords (Rj).
		Brho1 : float	
			Rho component of magnetic field (nT).
		Bphi1 : float
			Phi (azimuthal) component of the magnetic field (nT).
		Bz1 : float
			z component of the magnetic field (nT).
			
		Returns
		=======
		Bx0 : float
			x-component of magnetic field in right-handed System III
			coordinates (nT).
		By0 : float
			y-component of magnetic field in right-handed System III
			coordinates (nT).
		Bz0 : float
			z-component of magnetic field in right-handed System III
			coordinates (nT).
			
		
		'''
		rho = np.sqrt(x1*x1 + y1*y1)
		cosphi1 = x1/rho
		sinphi1 = y1/rho
		
		Bx1 = Brho1*cosphi1 - Bphi1*sinphi1
		By1 = Brho1*sinphi1 + Bphi1*cosphi1 		

		Bx = Bx1*self.cosxt - Bz1*self.sinxt
		Bz0 = Bx1*self.sinxt + Bz1*self.cosxt		

		Bx0 = Bx*self.cosxp - By1*self.sinxp
		By0 = By1*self.cosxp + Bx*self.sinxp	
	
		return Bx0,By0,Bz0
		
	
	def _ConvInputPol(self,r,theta,phi):
		'''
		Converts input coordinates from spherical polar right-handed 
		System III to Cartesian current sheet coordinates.
		
		Inputs
		======
		r : float
			System III radial distance (Rj).
		theta : float
			System III colatitude (rad).
		phi : float
			System III east longitude (rad).
			
		Returns
		=======
		x1 : float
			x current sheet coordinate
		y1 : float
			y current sheet coordinate
		z1 : float
			z current sheet coordinate
		cost : float
			cos(theta) - where theta is the colatitude
		sint : float
			sin(theta)
		cosp : float
			cos(phi) - where phi is east longitude
		sinp : float	
			sin(phi)
		'''		
	
		sint = np.sin(theta)
		cost = np.cos(theta)
		sinp = np.sin(phi)
		cosp = np.cos(phi)

		#surprisingly this is slightly (~20%) quicker than 
		#x = r*sint*np.cos(phi - self.dipole_shift) etc.
		x = r*sint*(cosp*self.cosxp + sinp*self.sinxp)
		y1 = r*sint*(sinp*self.cosxp - cosp*self.sinxp)
		z = r*cost
		
		x1 = x*self.cosxt + z*self.sinxt
		z1 = z*self.cosxt - x*self.sinxt	
		
		
		return x1,y1,z1,cost,sint,cosp,sinp


	def _ConvOutputPol(self,cost,sint,cosp,sinp,x1,y1,Brho1,Bphi1,Bz1):
		'''
		Convert the output magnetic field from cylindrical current 
		sheet coordinates to spherical polar right-handed System III
		coordinates.
		
		Inputs
		======
		cost : float
			cos(theta) - where theta is the colatitude
		sint : float
			sin(theta)
		cosp : float
			cos(phi) - where phi is east longitude
		sinp : float	
			sin(phi)
		x1 : float
			x-position in current sheet coords (Rj).
		y1 : float
			y-position in current sheet coords (Rj).
		Brho1 : float	
			Rho component of magnetic field (nT).
		Bphi1 : float
			Phi (azimuthal) component of the magnetic field (nT).
		Bz1 : float
			z component of the magnetic field (nT).
			
		Returns
		=======
		Br : float
			Radial component of magnetic field in right-handed System 
			III coordinates (nT).
		Bt : float
			Meridional component of magnetic field in right-handed 
			System III coordinates (nT).
		Bp : float
			Azimuthal component of magnetic field in right-handed System 
			III coordinates (nT).
			
		
		'''		
		rho = np.sqrt(x1*x1 + y1*y1)
		cosphi1 = x1/rho
		sinphi1 = y1/rho
		
		Bx1 = Brho1*cosphi1 - Bphi1*sinphi1
		By1 = Brho1*sinphi1 + Bphi1*cosphi1 		
		
		costheta_cs = np.cos(self.theta_cs)
		sintheta_cs = np.sin(self.theta_cs)
		Bx = Bx1*costheta_cs - Bz1*sintheta_cs
		Bz = Bx1*sintheta_cs + Bz1*costheta_cs		
	
		cos_xp = np.cos(self.dipole_shift)
		sin_xp = np.sin(self.dipole_shift)
		Bx2 = Bx*cos_xp - By1*sin_xp
		By2 = By1*cos_xp + Bx*sin_xp	

		Br =  Bx2*sint*cosp+By2*sint*sinp+Bz*cost#
		Bt =  Bx2*cost*cosp+By2*cost*sinp-Bz*sint#
		Bp = -Bx2*     sinp+By2*     cosp#
	
		return Br,Bt,Bp

		
	def _CheckInputCart(self,x,y,z):
		'''
		Check the Cartesian inputs - if the checks fail then the
		function raises an error.

		
		Inputs
		======
		x0 : float
			System III x-coordinate (Rj).
		y0 : float
			System III y-coordinate (Rj).
		z0 : float
			System III z-coordinate (Rj).
			
		'''
		if (np.size(x) != np.size(y)) or (np.size(x) != np.size(z)):
			raise SystemExit ('ERROR: Input coordinate arrays must all be of the same length. Returning...')
		
		#calculate r
		r = np.sqrt(x*x + y*y + z*z)

		if np.min(r) <= 0 or np.max(r) >= 200:
			raise SystemExit ('ERROR: Radial distance r must be in units of Rj and >0 but <200 only, and not outside that range (did you use km instead?). Returning...')


	def _CheckInputPol(self,r,theta,phi):
		'''
		Check the spherical polar inputs - if the checks fail then the
		function raises an error.

		
		Inputs
		======
		r : float
			System III radial distance (Rj).
		theta : float
			System III colatitude (rad).
		phi : float
			System III east longitude (rad).
			
		'''
		if np.min(r) <= 0 or np.max(r) >= 200:
			raise SystemExit ('ERROR: Radial distance r must be in units of Rj and >0 but <200 only, and not outside that range (did you use km instead?). Returning...')

		if np.min(theta) < 0 or np.max(theta) > np.pi:
			raise SystemExit ('ERROR: CoLat must be in radians of 0 to pi only, and not outside that range (did you use degrees instead?). Returning...')

		if np.min(phi)  < 0 or np.max(phi) > 2*np.pi:
			raise SystemExit ('ERROR: Long must be in radians of 0 to 2pi only, and not outside that range (did you use degrees instead?). Returning...')	
			
		if (np.size(r) != np.size(phi)) or (np.size(r) != np.size(theta)):
			raise SystemExit ('ERROR: Input coordinate arrays must all be of the same length. Returning...')

	def _Bphi(self,rho,abs_z,z):
		'''
		New to CAN2020 (not included in CAN1981): radial current 
		produces an azimuthal field, so Bphi is nonzero

		Inputs
		======
		rho : float
			distance in the x-z plane of the current sheet in Rj.
		abs_z : float
			absolute value of the z-coordinate
		z : float
			signed version of the z-coordinate
			
		Returns
		=======
		Bphi : float
			Azimuthal component of the magnetic field.

		'''
		Bphi = 2.7975*self.i_rho/rho
		
		if np.size(rho) == 1:
			if abs_z < self.d:
				Bphi *= (abs_z/self.d)
			if z > 0:
				Bphi = -Bphi
		else:
			ind = np.where(abs_z < self.d)[0]
			if ind.size > 0:
				Bphi[ind] *= (abs_z[ind]/self.d)
			ind = np.where(z > 0)[0]
			if ind.size > 0:
				Bphi[ind] = -Bphi[ind]
		
		return Bphi
		
	def _Analytic(self,x,y,z):
		'''		
		Calculate the magnetic field associated with the current sheet
		using analytical equations either from Connerney et al 1981 or
		the divergence-free equations from Edwards et al 2001 (defualt).
		
		The equations used are predefined on creation of the object
		using the "Edwards" keyword (default=True).
		
		If Edwards == True:
			Using equations 9a and 9b for the small rho approximation
			and 13a and 13b for the large rho approximation of Edwards
			et al.
		
		If Edwards == False:
			Use equations A1-A4 of Connerney et al 1981 for the large 
			rho approximation and A7 and A8 for the small rho 
			approximation.
			
		Inputs
		======
		x : float
			x coordinate
		y : float
			y coordinate
		z : float
			z coordinate
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bphi : float
			phi-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''
		
		#a couple of other bits needed
		rho_sq = x*x + y*y
		rho = np.sqrt(rho_sq)
		abs_z = np.abs(z)
		
		#calculate the analytic solution first for Brho and Bz
		Brho,Bz = self._AnalyticFunc(rho,z,self.d,self.r0,self.mu_i)
		
		#calculate Bphi
		Bphi = self._Bphi(rho,abs_z,z)
		
		#subtract outer edge contribution
		Brho_fin,Bz_fin = self._Finite(rho,z,self.d,self.r1,self.mu_i)
		#Bphi_fin = -self.i_rho*Brho_fin/self.mu_i
		Brho -= Brho_fin
		#Bphi -= Bphi_fin
		Bz -= Bz_fin
		
		return Brho,Bphi,Bz
		
		
	def _IntegralScalar(self,x,y,z):
		'''
		Integrates the model equations for an single set of input 
		coordinates.
		
		Inputs
		======
		x : float
			x coordinate
		y : float
			y coordinate
		z : float
			z coordinate
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''				
		#a couple of other bits needed
		rho_sq = x*x + y*y
		rho = np.sqrt(rho_sq)
		abs_z = np.abs(z)
		
		#check which "zcase" we need for this vector
		check1 = np.abs(abs_z - self.d)		
		check2 = abs_z <= self.d*1.1
		
		if check1 >= 0.7:
			#case 1 or 2
			zc = 1
		elif (check1 < 0.7) and (check1 >= 0.1):
			#case 3 or 4
			zc = 3
		else:
			#case 5 or 6
			zc = 5
		#this bit does two things - it both takes into account the
		#check2 thing and it makes zc an index in range 0 to 5 as 
		#opposed to the zcase 1 to 6, so zi = zcase -1
		zc -= np.int(check2)
		
		#do the integration
		beselj_rho_rho1_1 = j1(self.lambda_int_brho[zc]*rho)
		beselj_z_rho1_0   = j0(self.lambda_int_bz[zc]*rho)
		if (abs_z > self.d): #% Connerney et al. 1981 eqs. 14 and 15
			brho_int_funct = beselj_rho_rho1_1*self.beselj_rho_r0_0[zc] \
							*np.sinh(self.d*self.lambda_int_brho[zc]) \
							*np.exp(-abs_z*self.lambda_int_brho[zc]) \
							/self.lambda_int_brho[zc]
			bz_int_funct   = beselj_z_rho1_0 *self.beselj_z_r0_0[zc] \
							*np.sinh(self.d*self.lambda_int_bz[zc]) \
							*np.exp(-abs_z*self.lambda_int_bz[zc]) \
							/self.lambda_int_bz[zc]  
			Brho = self.mu_i*2.0*_Integrate(brho_int_funct,self.dlambda_brho)
			if z < 0:
				Brho = -Brho
		else:
			brho_int_funct = beselj_rho_rho1_1*self.beselj_rho_r0_0[zc] \
							*(np.sinh(z*self.lambda_int_brho[zc]) \
							*np.exp(-self.d*self.lambda_int_brho[zc])) \
							/self.lambda_int_brho[zc]
			bz_int_funct   = beselj_z_rho1_0  *self.beselj_z_r0_0[zc] \
							*(1.0 -np.cosh(z*self.lambda_int_bz[zc]) \
							*np.exp(-self.d*self.lambda_int_bz[zc])) \
							/self.lambda_int_bz[zc]
			Brho = self.mu_i*2.0*_Integrate(brho_int_funct,self.dlambda_brho)#
		Bz = self.mu_i*2.0*_Integrate(bz_int_funct,self.dlambda_bz)
		
		return Brho,Bz


	def _IntegralVector(self,x,y,z):
		'''
		Integrates the model equations for an array of input coordinates.
		
		Inputs
		======
		x : float
			x coordinate
		y : float
			y coordinate
		z : float
			z coordinate
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''		
				
		#a couple of other bits needed
		rho_sq = x*x + y*y
		rho = np.sqrt(rho_sq)
		abs_z = np.abs(z)
		
		#check which "zcase" we need for this vector
		check1 = np.abs(abs_z - self.d)		
		check2 = abs_z <= self.d*1.1
		s = _Switcher(check1,check2)

		#create the output arrays for this function
		Brho = np.zeros(np.size(x),dtype='float64')
		Bz = np.zeros(np.size(x),dtype='float64')

		for zcase in range(1,7):
			
			ind_case,lambda_max_brho,lambda_max_bz = s.FetchCase(zcase)
			n_ind_case=len(ind_case)
			zc = zcase - 1
			
			if n_ind_case > 0:

				for zi in range(0,n_ind_case):
					ind_for_integral = ind_case[zi] #;% sub-indices of sub-indices!

					beselj_rho_rho1_1 = j1(self.lambda_int_brho[zc]*rho[ind_for_integral])
					beselj_z_rho1_0   = j0(self.lambda_int_bz[zc]*rho[ind_for_integral] )
					if (abs_z[ind_for_integral] > self.d): #% Connerney et al. 1981 eqs. 14 and 15
						brho_int_funct = beselj_rho_rho1_1*self.beselj_rho_r0_0[zc] \
										*np.sinh(self.d*self.lambda_int_brho[zc]) \
										*np.exp(-abs_z[ind_for_integral]*self.lambda_int_brho[zc]) \
										/self.lambda_int_brho[zc]
						bz_int_funct   = beselj_z_rho1_0*self.beselj_z_r0_0[zc] \
										*np.sinh(self.d*self.lambda_int_bz[zc]) \
										*np.exp(-abs_z[ind_for_integral]*self.lambda_int_bz[zc]) \
										/self.lambda_int_bz[zc]
						Brho[ind_for_integral] = self.mu_i*2.0*_Integrate(brho_int_funct,self.dlambda_brho)
						if z[ind_for_integral] < 0:
							Brho[ind_for_integral] = -Brho[ind_for_integral]
					else:
						brho_int_funct = beselj_rho_rho1_1*self.beselj_rho_r0_0[zc] \
										*(np.sinh(z[ind_for_integral]*self.lambda_int_brho[zc]) \
										*np.exp(-self.d*self.lambda_int_brho[zc])) \
										/self.lambda_int_brho[zc]
						bz_int_funct   = beselj_z_rho1_0*self.beselj_z_r0_0[zc] \
										*(1.0 -np.cosh(z[ind_for_integral]*self.lambda_int_bz[zc]) \
										*np.exp(-self.d*self.lambda_int_bz[zc])) \
										/self.lambda_int_bz[zc]  
						Brho[ind_for_integral] = self.mu_i*2.0*_Integrate(brho_int_funct,self.dlambda_brho)
					Bz[ind_for_integral]   = self.mu_i*2.0*_Integrate(bz_int_funct,self.dlambda_bz)
		
		return Brho,Bz			
	
			
		
	def _Integral(self,x,y,z):
		'''		
		Calculate the magnetic field associated with the current sheet
		by integrating equations 14, 15, 17 and 18 of Connerney et al
		1981.
		
		Inputs
		======
		x : float
			x coordinate
		y : float
			y coordinate
		z : float
			z coordinate
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bphi : float
			phi-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''		
		rho_sq = x*x + y*y
		rho = np.sqrt(rho_sq)
		abs_z = np.abs(z)
		
		if np.size(x) == 1:
			#scalar version of the code
			Brho,Bz = self._IntegralScalar(x,y,z)
		else:
			#vectorized version
			Brho,Bz = self._IntegralVector(x,y,z)
		
		#calculate Bphi
		Bphi = self._Bphi(rho,abs_z,z)
		
		#subtract outer edge contribution
		Brho_fin,Bz_fin = self._Finite(rho,z,self.d,self.r1,self.mu_i)
		#Bphi_fin = -self.i_rho*Brho_fin/self.mu_i
		Brho -= Brho_fin
		#Bphi -= Bphi_fin
		Bz -= Bz_fin
		
		return Brho,Bphi,Bz		
		
		
	def _Hybrid(self,x,y,z):
		'''		
		Calculate the magnetic field associated with the current sheet
		by using a combination of analytical equations and numerical
		integration.
		
		Inputs
		======
		x : float
			x coordinate
		y : float
			y coordinate
		z : float
			z coordinate
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bphi : float
			phi-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''		
		#a couple of other bits needed
		rho_sq = x*x + y*y
		rho = np.sqrt(rho_sq)
		abs_z = np.abs(z)

		if np.size(x) == 1:
			#do the scalar version
			
			#check if we need to integrate numerically, or use analytical equations
			if (abs_z <= self.d*1.5) and (np.abs(rho - self.r0) <= 2.0):
				#use integration
				Brho,Bz = self._IntegralScalar(x,y,z)
			else:
				#analytical
				Brho,Bz = self._AnalyticFunc(rho,z,self.d,self.r0,self.mu_i)

		else:
			#this would be the vectorized version
			n = np.size(x)
			Brho = np.zeros(n,dtype='float64')
			Bz = np.zeros(n,dtype='float64')

			doint = (abs_z <= self.d*1.5) & (np.abs(rho-self.r0) <= 2)
			Iint = np.where(doint)[0]
			Iana = np.where(doint == False)[0]
			
			if Iint.size > 0:
				Brho[Iint],Bz[Iint] = self._IntegralVector(x[Iint],y[Iint],z[Iint])
			
			if Iana.size > 0:
				Brho[Iana],Bz[Iana] = self._AnalyticFunc(rho[Iana],z[Iana],self.d,self.r0,self.mu_i)


		#calculate Bphi
		Bphi = self._Bphi(rho,abs_z,z)
		
		#subtract outer edge contribution
		Brho_fin,Bz_fin = self._Finite(rho,z,self.d,self.r1,self.mu_i)
		#Bphi_fin = -self.i_rho*Brho_fin/self.mu_i
		Brho -= Brho_fin
		#Bphi -= Bphi_fin
		Bz -= Bz_fin
		
		return Brho,Bphi,Bz		
				
		
				
	def Field(self,*args):
		'''
		Return the magnetic field vector(s) for a given input position
		in right-handed System III coordinates.
		
		Input Arguments
		===============
		The function should be called using three arguments, either
		scalars or vectors (not a mixture):
		
		args[0] : float
			First input coordinate(s) - x or r (in Rj).
		args[1] : float
			Second input coordinate(s) - y (in Rj) or theta (in rad).
		args[2] : float
			Third input coordinate(s) - z (in Rj) or phi (in rad).
			
		Whether or not the input coordinates are treated as Cartesian or
		spherical polar depends upon how the model was initialized with
		the "CartesianIn" keyword.
		
		e.g.:
		# for Cartesian input coordinates:
		B = Model.Field(x,y,z)
		
		#or spherical polar coordinates:
		B = Model.Field(r,theta,phi)
		
		Returns
		=======
		B : float
			(n,3) shaped array containing the magnetic field vectors in
			either Cartesian SIII coordinates or spherical polar ones,
			depending upon how the model was initialized, where "n" is
			the number of elements contained in the input arguments.
		'''
		
		#the inputs should be 3 scalars or arrays
		if self.CartesianIn:
			try:
				x0,y0,z0 = args
			except:
				raise SystemError("Input arguments should be x,y,z or r,theta,phi")
			#check inputs
			if self.error_check:
				self._CheckInputCart(x0,y0,z0)
				
			#rotate to current sheet coords	
			x,y,z,costheta,sintheta,cosphi,sinphi = self._ConvInputCart(x0,y0,z0)
		else:
			try:
				r0,t0,p0 = args
			except:
				raise SystemError("Input arguments should be x,y,z or r,theta,phi")				

			#check inputs
			if self.error_check:
				self._CheckInputPol(r0,t0,p0)
			
			#rotate to current sheet coords	
			x,y,z,costheta,sintheta,cosphi,sinphi = self._ConvInputPol(r0,t0,p0)	
			
		#create the output arrays
		n = np.size(x)
		Bout = np.zeros((n,3),dtype='float64')
		
		Brho = np.zeros(n,dtype='float64')
		Bphi = np.zeros(n,dtype='float64')
		Bz = np.zeros(n,dtype='float64')
		

		
		
		#call the model function
		Brho,Bphi,Bz = self._ModelFunc(x,y,z)
		
		
		#return to SIII coordinates
		if self.CartesianOut:
			B0,B1,B2 = self._ConvOutputCart(x,y,Brho,Bphi,Bz)
		else:
			B0,B1,B2 = self._ConvOutputPol(costheta,sintheta,cosphi,sinphi,
											x,y,Brho,Bphi,Bz)
		
		#turn into a nx3 array
		Bout[:,0] = B0
		Bout[:,1] = B1
		Bout[:,2] = B2

		return Bout
