import numpy as np
from scipy.special import jv
from ._Switcher import _Switcher

def Model(r,theta,phi,mu_i=139.6,i_rho=16.7,r0=7.8,r1=51.4,d=3.6,xt=9.3,
			xp=-24.2,equation_type='hybrid',no_error_check=False,
			Cartesian=False):
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
	Inputs
	======
	r : float
		Radial distance, in Rj (System III)
	theta : float
		Colatitude, in radians (System III)
	phi : float
		longitude, right handed, in radians (System III)
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
	br : float
		Radial field, in nT (or Bx if Cartesian == True)
	bt : float
		Meridional field, in nT (or By if Cartesian == True)
	bp : float
		Azimuthal field, in nT (or Bz if Cartesian == True)
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

	if not no_error_check:
		if not equation_type in ['analytic','hybrid','integral']:
			raise SystemExit ('ERROR: case statement has unrecognized string - was your equation_type lower case?')	

		if np.min(r) < 0 or np.max(r) > 200:
			raise SystemExit ('ERROR: Radial distance r must be in units of Rj and >0 but <200 only, and not outside that range (did you use km instead?). Returning...')

		if np.min(theta) < 0 or np.max(theta) > np.pi:
			raise SystemExit ('ERROR: CoLat must be in radians of 0 to pi only, and not outside that range (did you use degrees instead?). Returning...')

		if np.min(phi)  < 0 or np.max(phi) > 2*np.pi:
			raise SystemExit ('ERROR: Long must be in radians of 0 to 2pi only, and not outside that range (did you use degrees instead?). Returning...')	

	#convert inputs to numpy arrays if scalars are provided
	if not hasattr(r,'__iter__'):
		r = np.array([r])
		theta = np.array([theta])
		phi = np.array([phi])
		
	#convert lists/tuples to numpy arrays
	conv = not (isinstance(r,np.ndarray) and isinstance(theta,np.ndarray) and isinstance(phi,np.ndarray))
	if conv:
		r = np.array(r)
		theta = np.array(theta)
		phi = np.array(phi)
		
#% Convert to cartesian coordinates and rotate into magnetic longitude
#% (x,y,z) are the shifted (phi) coordinates
#  dipole_shift = double(xp_value)*!dpi/180. #xp_value is longitude of the dipole
#  x = r*sin(theta)*cos(phi-dipole_shift)
#  y = r*sin(theta)*sin(phi-dipole_shift)
#  z = r*cos(theta)
#% RJW way
	Deg2Rad = np.pi/180.0
	sin_theta = np.sin(theta)#
	cos_theta = np.cos(theta)#
	sin_phi   = np.sin(phi)#
	cos_phi   = np.cos(phi)#

	dipole_shift = xp*Deg2Rad # % xp_value is longitude of the dipole
	x = r*sin_theta*np.cos(phi-dipole_shift)
	y = r*sin_theta*np.sin(phi-dipole_shift)
	z = r*cos_theta#]

#% Rotate by the amount of the dipole tilt
#% (x1,y1,z1) are the tilted (theta) and shifted (phi) coordinates
#  theta_cs = double(xt_value)*!dpi/180. #dipole tilt is xt_value
#  x1 = x*cos(theta_cs) + z*sin(theta_cs)
#  y1 = y
#  z1 = z*cos(theta_cs) - x*sin(theta_cs)
#  rho1 = sqrt(x1^2.d + y1^2.d) #cylindrical radial distance

#% RJW way
	theta_cs = xt*Deg2Rad # % dipole tilt is xt_value
	x1 = x*np.cos(theta_cs) + z*np.sin(theta_cs)#
	y1 = y# RJW - NOT NEEDED REALLY - BUT USED IN ATAN LATER
	z1 = z*np.cos(theta_cs) - x*np.sin(theta_cs)#
	rho1_sq = x1*x1 + y1*y1
	rho1 = (rho1_sq)**0.5 # %cylindrical radial distance
	mui_2 = mu_i


#% ===========
#% Decide whether to use integral equations or analytic equations
#% ===========
#  if ((abs(z1) le d_value*1.5d and abs(rho1-r0_value) le 2.d) or eq_type eq 'integral') then do_integral = 1 else do_integral = 0
#  if eq_type eq 'analytic' then do_integral = 0
#% RJW way - mainly for clarity

	

	abs_z1 = np.abs(z1)#
	N = z1.size

	do_integral=np.zeros(N,dtype='bool')
	if equation_type == 'analytic':
		do_integral[0:N] = False
	elif equation_type == 'integral':
		do_integral[0:N] = True
	elif equation_type == 'hybrid':
		sel_hybrid = np.where((abs_z1 <= d*1.5) & (np.abs(rho1-r0) <= 2))[0]
		do_integral[sel_hybrid]=1


	ind_analytic = np.where(do_integral == 0)[0]
	ind_integral = np.where(do_integral == 1)[0]

	n_ind_analytic = ind_analytic.size
	n_ind_integral = ind_integral.size
	
	brho1= np.zeros(N)
	bz1= np.zeros(N)			


	'''	
 ===========
Integral equations - Brho and Bz eqs. vary depending on region with respect to the disk
lambda_max_brho = 4.#default integration limit for Brho function
lambda_max_bz = 100.#default integration limit for Bz function
lambda = 1e-4 #default step size for both
dlambda_brho = 1e-4#default step size for Brho function
dlambda_bz = e-5#default step size for Bz function
if abs(abs(z1)-d_value) lt .7 then lambda_max_brho = 40.d #Z > D
if abs(abs(z1)-d_value) lt .1 then lambda_max_brho = 100.d #Z very close to D
if abs(z1) gt d_value*1.1 then lambda_max_bz = 20.d #Small Z
lambda_int_brho = dindgen(lambda_max_brho/dlambda_brho)*dlambda_brho
lambda_int_bz = dindgen(lambda_max_bz/dlambda_bz)*dlambda_bz
sign_z = z1/abs(z1)
if abs(z1) gt d_value then begin #Connerney et al. 1981 eqs. 14 and 15
brho_int_funct = beselj(lambda_int_brho*rho1,1)*beselj(lambda_int_brho*r0_value,0)*sinh(d_value*lambda_int_brho)*exp((-1.d)*abs(z1)*lambda_int_brho)/lambda_int_brho
brho1 = sign_z*mui_2*2.d*int_tabulated(lambda_int_brho(where(abs(brho_int_funct) lt 1e99)),brho_int_funct(where(abs(brho_int_funct) lt 1e99)))
bz_int_funct = beselj(lambda_int_bz*rho1,0)*beselj(lambda_int_bz*r0_value,0)*sinh(d_value*lambda_int_bz)*exp((-1.d)*abs(z1)*lambda_int_bz)/lambda_int_bz
bz1 = mui_2*2.d*int_tabulated(lambda_int_bz(where(abs(bz_int_funct) lt 1e99)),bz_int_funct(where(abs(bz_int_funct)lt 1e99)))
endif else begin #Connerney et al. 1981 eqs. 17 and 18
brho_int_funct = beselj(lambda_int_brho*rho1,1)*beselj(lambda_int_brho*r0_value,0)*sinh(z1*lambda_int_brho)*exp((-1.d)*d_value*lambda_int_brho)/lambda_int_brho
brho1 = mui_2*2.d*int_tabulated(lambda_int_brho(where(abs(brho_int_funct) lt 1e99)),brho_int_funct(where(abs(brho_int_funct) lt 1e99)))
bz_int_funct = beselj(lambda_int_bz*rho1,0)*beselj(lambda_int_bz*r0_value,0)*(1.d - cosh(z1*lambda_int_bz)*exp((-1.d)*d_value*lambda_int_bz))/lambda_int_bz
bz1 = mui_2*2.d*int_tabulated(lambda_int_bz(where(abs(bz_int_funct) lt 1e99)),bz_int_funct(where(abs(bz_int_funct)lt 1e99)))
endelse
    #% RJW way
	
			'''	

	if (n_ind_integral != 0):

		#% lambda_max_brho = 4.0d  #% default integration limit for Brho function
		#% lambda_max_bz   = 100d  #% default integration limit for Bz function
		dlambda         = 1e-4  #% default step size for both
		dlambda_brho    = 1e-4  #% default step size for Brho function
		dlambda_bz      = 5e-5  #% default step size for Bz function


		#%      if abs(abs_z1[ind_for_integral] - d_value) lt 0.7d then BEGIN
		#%        if abs(abs_z1[ind_for_integral] - d_value) lt 0.1d then BEGIN
		#%          lambda_max_brho = 100.0d #% Z very close to D
		#%        ENDIF ELSE BEGIN
		#%          lambda_max_brho = 40.0d #% Z > D
		#%        ENDELSE
		#%      ENDIF ELSE BEGIN
		#%        lambda_max_brho = 4.0d  #% default integration limit for Brho function
		#%      ENDELSE
		#%      #%if (abs_z1 gt d_value*1.1d) then lambda_max_bz = 20.d else lambda_max_bz   = 100d#% Small Z or default in else
		#%      if (abs_z1[ind_for_integral] le d_value*1.1d) then lambda_max_bz = 100d else lambda_max_bz   = 20d #% Small Z or default in else


		check1=np.abs(abs_z1[ind_integral]-d)
		ncheck1=(check1.size)
		check2=np.zeros(ncheck1)
		inside_d = np.where(abs_z1[ind_integral] < d*1.1)[0]
		flag = not np.any(inside_d)
		if flag == False:
			check2[inside_d]=1

		zcase=1

		for zcase in range(1,7):
			s = _Switcher(check1,check2)
			ind_case,lambda_max_brho,lambda_max_bz = s.indirect(zcase)
			n_ind_case=len(ind_case)

			if n_ind_case >= 0:
				lambda_int_brho = np.arange(dlambda_brho,dlambda_brho*(lambda_max_brho/dlambda_brho - 1),dlambda_brho ) 
				lambda_int_bz = np.arange(dlambda_bz,dlambda_bz*(lambda_max_bz/dlambda_bz - 1),dlambda_bz) 
				beselj_rho_r0_0   = jv(0,lambda_int_brho*r0)# % Only 6 sets of values
				beselj_z_r0_0     = jv(0,lambda_int_bz*r0)# % Only 6 sets of values

				for zi in range(0,n_ind_case):
					ind_for_integral = ind_integral[ind_case[zi]] #;% sub-indices of sub-indices!
 
					beselj_rho_rho1_1 = jv(1, lambda_int_brho*rho1[ind_for_integral])
					beselj_z_rho1_0   = jv(0,lambda_int_bz *rho1[ind_for_integral] )
					if (abs_z1[ind_for_integral] > d): #% Connerney et al. 1981 eqs. 14 and 15
						brho_int_funct = beselj_rho_rho1_1*beselj_rho_r0_0 *np.sinh(d*lambda_int_brho) *np.exp(-abs_z1[ind_for_integral]*lambda_int_brho)/lambda_int_brho#
						bz_int_funct   = beselj_z_rho1_0 *beselj_z_r0_0  *np.sinh(d*lambda_int_bz  ) *np.exp(-abs_z1[ind_for_integral]*lambda_int_bz  )/lambda_int_bz  #
						brho1[ind_for_integral] = mui_2*2.0*np.trapz(brho_int_funct,lambda_int_brho)#
						if z1[ind_for_integral] < 0:
							brho1[ind_for_integral] = -brho1[ind_for_integral]
					else:
						brho_int_funct = beselj_rho_rho1_1*beselj_rho_r0_0*(np.sinh(z1[ind_for_integral]*lambda_int_brho)*np.exp(-d*lambda_int_brho))/lambda_int_brho#
						bz_int_funct   = beselj_z_rho1_0  *beselj_z_r0_0  *(1.0 -np.cosh(z1[ind_for_integral]*lambda_int_bz  )*np.exp(-d*lambda_int_bz  ))/lambda_int_bz  #
						brho1[ind_for_integral] = mui_2*2.0*np.trapz(brho_int_funct,lambda_int_brho)#
					bz1[ind_for_integral]   = mui_2*2.0*np.trapz(bz_int_funct,lambda_int_bz)
					
	
			

	'''
===========
 Analytic equations
===========
Connerney et al. 1981's equations for the field produced by a semi-infinite disk of thickness D, inner edge R0, outer edge R1 -
Here we use the Edwards+ (2001) updated Connerney approximations for small rho (9a and 9b) and large rho (13a and 13b)
RJW way
Doing these 3 equations on the whole array to save getting confused by indices,  Will swap to just required indices later
	'''
	
	if (n_ind_analytic != 0):
		z1pd = z1+d;
		z1md = z1-d
		r0_value_sq = r0_value*r0_value;

		ind_LT = np.where((rho1 < r0) & (do_integral == 0))[0]
		ind_GE = np.where((rho1 > r0) & (do_integral == 0))[0]

		n_ind_LT = ind_LT.size
		n_ind_GE = ind_GE.size



		if (n_ind_LT != 0): 
			f1 = np.sqrt(z1md[ind_LT]*z1md[ind_LT] +r0**2)
			f2 =np.sqrt(z1pd[ind_LT]*z1pd[ind_LT] +r0**2)
			f1_cubed = f1**3
			f2_cubed = f2**3
			f3_a = (r0_value_sq +    z1md[ind_LT]*z1md[ind_LT])
			f3   = (r0_value_sq - 2*z1md[ind_LT]*z1md[ind_LT])/(f3_a*f3_a*np.sqrt(f3_a))
			f4_a = (r0_value_sq +    z1pd[ind_LT]*z1pd[ind_LT])
			f4   = (r0_value_sq - 2*z1pd[ind_LT]*z1pd[ind_LT])/(f4_a*f4_a*np.sqrt(f4_a))

			brho1[ind_LT] = mui_2*((rho1[ind_LT]/2)*((1/f1)-(1/f2))  + (rho1[ind_LT]*rho1[ind_LT]*rho1[ind_LT]/16)*(f3-f4)) #; Edwards et al eq 9a
			bz1[  ind_LT] = mui_2*(alog((z1pd[ind_LT]+f2)/(z1md[ind_LT]+f1)) + (rho1[ind_LT]*rho1[ind_LT] / 4)*(((z1pd[ind_LT])/f2_cubed) - ((z1md[ind_LT])/f1_cubed)))   #; Edwards et al eq 9b

		if (n_ind_GE != 0):
			f1 = np.sqrt(z1md[ind_GE]*z1md[ind_GE] +rho1_sq[ind_GE])
			f2 = np.sqrt(z1pd[ind_GE]*z1pd[ind_GE] +rho1_sq[ind_GE])
			f1_cubed = f1**3
			f2_cubed = f2**3
			bz1[ind_GE] = mui_2*(np.log((z1pd[ind_GE]+f2)/(z1md[ind_GE]+f1)) + (r0_value_sq/4)*((z1pd[ind_GE]/f2_cubed) - (z1md[ind_GE]/f1_cubed)))  #;Edwards et al eq 13b

			ind2_LT = np.where(abs_z1[ind_GE] > d)[0]
			ind2_GE = np.where(abs_z1[ind_GE] < d)[0]

			n_ind2_LT = ind2_LT.size
			n_ind2_GE = ind2_GE.size
			
			
			if (n_ind2_LT != 0):
				ind3 = ind_GE[ind2_LT];
				brho1[ind3] = mui_2*((1/rho1[ind3])*(f1[ind2_LT]-f2[ind2_LT]+2*d_value*z1[ind3]/abs_z1[ind3]) - (r0_value_sq*rho1[ind3]/4)*((1/f1_cubed[ind2_LT])-(1/f2_cubed[ind2_LT]))) #;Edwards et al eq 13a / Connerney et al eq A7
  
			if (n_ind2_GE != 0):
				ind3 = ind_GE[ind2_GE];
				brho1[ind3] = mui_2*((1/rho1[ind3])*(f1[ind2_GE]-f2[ind2_GE]+2*z1[ind3]) - (r0_value_sq*rho1[ind3]/4)*((1/f1_cubed[ind2_GE])-(1/f2_cubed[ind2_GE])))  #;Edwards et al eq 13a / Connerney et al eq A9
       
	'''
 =======
 New to CAN2020 (not included in CAN1981): radial current produces an azimuthal field, so Bphi is nonzero
 =======
 bphi1 = 2.7975d*i_rho_value/rho1
 if abs(z1) lt d_value then bphi1 = bphi1*abs(z1)/d_value
 if z1 gt 0.d then bphi1 = (-1.d)*bphi1
#% RJW way
	'''
 
	bphi1 = 2.7975*i_rho/rho1

	ind = np.where(abs_z1 < d)[0]
	sized = ind.size
	if sized != 0:
		bphi1[ind] =  bphi1[ind] * abs_z1[ind] / d

	ind = np.where(z1 > 0.0)[0]
	sized = ind.size
	if sized != 0:
		bphi1[ind] =  -bphi1[ind] 
	
	'''
=====================
Account for finite nature of current sheet by subtracting the field values (using small rho approximation for a semi-infinite sheet)
with a (inner edge) = r1 (= 51.4 Rj by default in CAN2020) following Edwards et al.
Note that the Connerney et al. 1981 paper (and equations in the Dessler book) mentions a different approximation, which is simply
 subtracting 0.1*mu0i0/2 from Bz - and leaves Br unchanged (see Connerney et al. 1981 Figure 4).
a1 = r1_value #outer edge
f1 = sqrt((z1-d_value)^2.d +a1^2.d)
f2 = sqrt((z1+d_value)^2.d +a1^2.d)
brho_finite = mui_2*(rho1/2.d)*((1.d/f1)-(1.d/f2))
bz_finite = mui_2*(2.d*d_value*(z1^2. +a1^2.d)^(-0.5d) - ((rho1^2.d)/4.d)*(((z1-d_value)/(f1^3.d)) - ((z1+d_value)/f2^3.d)))
brho1 = brho1 - brho_finite
bphi_finite = (-1.d)*(brho_finite/mui_2)*i_rho_value
bz1 = bz1 - bz_finite
RJW way
a1 = r1_value #% outer edge
a1_sq = a1*a1# % outer edge squared
	'''
	a1_sq = r1_value * r1_value
	z1md = z1-d_value
	z1pd = z1+d_value
	z1md_sq = z1md*z1md
	z1pd_sq = z1pd*z1pd
	f1 = np.sqrt(z1md_sq +a1_sq)
	f2 =np.sqrt(z1pd_sq +a1_sq)
	f3_a = (a1_sq +    z1md_sq)
	f3   = (a1_sq - 2*z1md_sq)/(f3_a*f3_a*np.sqrt(f3_a))
	f4_a = (a1_sq +    z1pd_sq)
	f4   = (a1_sq - 2*z1pd_sq)/(f4_a*f4_a*np.sqrt(f4_a))
	brho_finite = mui_2*((rho1/2)*((1/f1)-(1/f2)) + (rho1*rho1*rho1/16)*(f3-f4)) #; Edwards et al eq 9a
	bz_finite   = mui_2*(np.log((z1pd+f2)/(z1md+f1))   + (rho1*rho1/4)*(((z1pd)/(f2*f2*f2)) - ((z1md)/(f1*f1*f1)))) #; Edwards et al eq 9b
	brho1       = brho1 - brho_finite
	bphi_finite = (-1)*i_rho_value * brho_finite/mui_2
	bz1   = bz1 - bz_finite



	'''
brho1, bphi1, and bz1 here are the ultimately calculated brho and bz values from the CAN model
the remaining calculations just rotate the field back into SIII
Calculate 'magnetic longitude' and convert the field into cartesian coordinates
phi1     = atan(y1,x1)#
cos_phi1 = cos(phi1)#
sin_phi1 = sin(phi1)#
The above three commands is really a cos(atan(y1,x1)) and a sin(atan(y1,x1)).  This simplifies
	'''

	cos_phi1 = x1/rho1 #% cos(atan(y1,x1)) see https://www.wolframalpha.com/input/?i=cos%28atan2%28y%2Cx%29%29
	sin_phi1 = y1/rho1 #% sin(atan(y1,x1)) see https://www.wolframalpha.com/input/?i=sin%28atan2%28y%2Cx%29%29
	
	bx1 = brho1*cos_phi1 - bphi1*sin_phi1 # % lines above were used for CAN1981, but since Bphi != 0 in CAN 2020 this is updated here
	by1 = brho1*sin_phi1 + bphi1*cos_phi1 #

	'''
Rotate back by dipole tilt amount, into coordinate system that is aligned with Jupiter's spin axis
bx = bx1*cos(theta_cs) - bz1*sin(theta_cs)
by = by1
bz = bx1*sin(theta_cs) + bz1*cos(theta_cs)
RJW way
	'''

	cos_theta_cs = np.cos(theta_cs)#
	sin_theta_cs =  np.sin(theta_cs)#
	bx = bx1*cos_theta_cs - bz1*sin_theta_cs#
	bz = bx1*sin_theta_cs + bz1*cos_theta_cs#


#% Finally, shift back to SIII longitude

	cos_xp = np.cos(dipole_shift) #% cos(xp_value*Deg2Rad)#
	sin_xp = np.sin(dipole_shift) #% sin(xp_value*Deg2Rad)#
	bx2 = bx*cos_xp - by1*sin_xp# % used sin(-a) = asin(a) & cos(-a) = cos(a)
	by2 = by1*cos_xp + bx*sin_xp#

	'''
return, [[bx2],[by2],[bz]]#% Exit here if we want output in Cartesian, not Spherical
Convert to spherical coordinates
bx = bx2#
by = by2#
br =  bx*sin_theta*cos_phi+by*sin_theta*sin_phi+bz*cos_theta#
bt =  bx*cos_theta*cos_phi+by*cos_theta*sin_phi-bz*sin_theta#
bp = -bx*          sin_phi+by*          cos_phi#
	'''

	if Cartesian:
		return bx2,by2,bz

	br =  bx2*sin_theta*cos_phi+by2*sin_theta*sin_phi+bz*cos_theta#
	bt =  bx2*cos_theta*cos_phi+by2*cos_theta*sin_phi-bz*sin_theta#
	bp = -bx2*          sin_phi+by2*          cos_phi#

	return br,bt,bp
