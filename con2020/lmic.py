import numpy as np
import matplotlib.pyplot as plt

def PlotLMIC():
	
	
	#calculate the angular velocity
	theta = np.linspace(0.0,20.0,1000)*np.pi/180.0
	wO = AngularVelocityRatio(theta)
	
	Ri = 67350000.0
	Rj = 71492000.0
	ri = Ri/Rj
	
	IhP = PedersenCurrent(theta)/1e6
	Bp = BphiIonosphere(theta)
	Bpi = Bphi(Ri/Rj,theta)
	Bpr = Bphi(1.5,theta)

	thetad = theta*180.0/np.pi
	
	fig = plt
	fig.figure(figsize=(8,11))
	ax0 = fig.subplot2grid((3,1),(0,0))
	ax0.plot(thetad,wO)
	ax0.set_ylabel('$\omega/\Omega_J$')
	ax0.set_xlabel('$\\theta$ ($^\circ$)')
	
	ax1 = fig.subplot2grid((3,1),(1,0))
	ax1.plot(thetad,IhP)
	ax1.set_ylabel('$I_{hP}$')
	ax1.set_xlabel('$\\theta$ ($^\circ$)')
	
	
	ax2 = fig.subplot2grid((3,1),(2,0))
	ax2.plot(thetad,Bp,label='ionosphere ({:4.2f} $R_J$)'.format(ri))
	ax2.plot(thetad,Bpi,label='ionosphere ({:4.2f} $R_J$)'.format(ri))
	ax2.plot(thetad,Bpr,label='1.5 $R_J$')
	ax2.legend()
	ax2.set_ylabel('$B_{\\phi}$')
	ax2.set_xlabel('$\\theta$ ($^\circ$)')
		
	
def BphiIonosphere(	thetai,g=417659.3836476442,
					wO_open=0.1,wO_om=0.35,
					thetamm=16.1,dthetamm=0.5,
					thetaoc=10.716,dthetaoc=0.125):
	'''
	Calculate the Bphi component at the ionosphere.

	Inputs
	======
	thetai : float
		Colatitude of the ionospheric footprint of the
		current field line.
	g : float
		Dipole coefficient, nT.
	wO_om : float
		Angular velocity of plasma on open field lines (~0.091)
	wO_om : float
		Angular velocity w of plasma divided by angular velocity of 
		Jupiter in the outer magnetosphere (0.25 for low and 0.5 for
		high velocity models)
	thetamm : float
		central colatitude of the middle magnetosphere (deg)
	dthetamm : float
		colatitudinal width of the middle magnetosphere (deg)
	thetaoc : float
		colatitude of the open-closed field line boundary (deg)
	dthetaoc : float
		colatitudinal width of the OCB (deg)

	Returns
	=======
	Bphi : float
		Ionospheric Bphi, nT.

	'''


	Ri = 67350000.0
	lat = np.pi/2.0 - thetai
	IhP = PedersenCurrent(thetai,g,wO_open,wO_om,thetamm,
							dthetamm,thetaoc,dthetaoc)
	mu0 = 4*np.pi*1e-7
	rho = Ri*np.sin(thetai)
	
	sgn = -np.sign(lat)
	Bphi = (sgn*mu0/(2*np.pi*rho))*IhP
	
	return Bphi*1e9


def ScalarPotSmallRho(	rho,z,a,
						mui2=139.6,D=3.6):
	'''
	Edwards et al. equation 8

	Inputs
	======
	rho : float
		cylindrical rho coordinate in Rj
	z : float
		z-coordinate in Rj
	a : float
		Inner edge of the current sheet, Rj.
	mui2 : float
		mu_0 I_0/2 parameter (nT).
	D : float
		current sheet half thickness

	Returns
	=======
	A : float
		Scalar potential.
	'''

	a2 = a*a
	zpd = z + D
	zmd = z - D
	zpd2 = zpd*zpd
	zmd2 = zmd*zmd
	zpda = np.sqrt(zpd2 + a2)
	zmda = np.sqrt(zmd2 + a2)
	zpda3 = zpda*zpda*zpda
	zmda3 = zmda*zmda*zmda

	term0 = (rho/2)*np.log((zpd + zpda)/(zmd + zmda))
	term1 = (rho**3/16.0)*(zpd/zpda3 - zmd/zmda3)

	A = mui2*(term0 + term1)
	return A

def ScalarPotLargeRho(	rho,z,a,
						mui2=139.6,D=3.6,
						DeltaZ=0.1):
	'''
	Large rho approximation, Edwards et al equation 12
	with the final term modified to be smooth across z=+-D

	Inputs
	======
	rho : float
		cylindrical rho coordinate in Rj
	z : float
		z-coordinate in Rj
	a : float
		Inner edge of the current sheet, Rj.
	mui2 : float
		mu_0 I_0/2 parameter (nT).
	D : float
		current sheet half thickness
	DeltaZ : float
		Scale length over which the large rho approx is
		smoothed across +/-D boundary.

	Returns
	=======
	A : float
		Scalar potential.

	'''

	a2 = a*a
	rho2 = rho*rho
	zpd = z + D
	zmd = z - D
	zpd2 = zpd*zpd
	zmd2 = zmd*zmd
	zpdr = np.sqrt(zpd2 + rho2)
	zmdr = np.sqrt(zmd2 + rho2)

	term0 = (1/(2*rho))*(zpd*zpdr - zmd*zmdr)
	term1 = (rho/2)*np.log((zpd + zpdr)/(zmd + zmdr))
	term2 = (a2/(4*rho))*(zmd/zmdr - zpd/zpdr)

	#this bit isn't the same as the last term in equation 12
	zpddz = zpd/DeltaZ
	zmddz = zmd/DeltaZ
	tanpd = np.tanh(zpddz)
	tanmd = np.tanh(zmddz)
	term3 = (-1/rho)*(D*z*(tanpd + tanmd) + 0.5*(D**2 + z**2)*(tanpd - tanmd))

	A = mui2*(term0 + term1 + term2 + term3)
	return A

def ScalarPot(	rho,z,a,
				mui2=139.6,D=3.6,
				DeltaRho=1.0,DeltaZ=0.1):
	'''
	Stan's smoothed combination of the two approx functions

	Inputs
	======
	rho : float
		cylindrical rho coordinate in Rj
	z : float
		z-coordinate in Rj
	a : float
		Inner edge of the current sheet, Rj.
	mui2 : float
		mu_0 I_0/2 parameter (nT).
	D : float
		current sheet half thickness
	DeltaRho : float
		Scale length over which to smooth the transition
		from small to large rho approximations
	DeltaZ : float
		Scale length over which the large rho approx is
		smoothed across +/-D boundary.

	Returns
	=======
	A : float
		Scalar potential.

	'''

	As = ScalarPotSmallRho(rho,z,a,mui2,D)
	Al = ScalarPotLargeRho(rho,z,a,mui2,D,DeltaZ)

	radr = (rho - a)/DeltaRho
	A = 0.5*As*(1 - np.tanh(radr)) + 0.5*Al*(1 + np.tanh(radr))
	return A

def FluxFunc(	rho,z,
				r0=7.8,r1=51.4,
				mui2=139.6,D=3.6,
				DeltaRho=1.0,DeltaZ=0.1):
	'''
	Calculate the CAN contribution to the flux function

	Inputs
	======
	rho : float
		cylindrical rho coordinate in Rj
	z : float
		z-coordinate in Rj
	r0 : float
		Inner edge of the current sheet, Rj.
	r1 : float
		Outer edge of the current sheet, Rj.
	mui2 : float
		mu_0 I_0/2 parameter (nT).
	D : float
		current sheet half thickness
	DeltaRho : float
		Scale length over which to smooth the transition
		from small to large rho approximations
	DeltaZ : float
		Scale length over which the large rho approx is
		smoothed across +/-D boundary.

	Returns
	=======
	F : float
		CAN flux

	'''
	A0 = ScalarPot(rho,z,r0,mui2,D,DeltaRho,DeltaZ)
	A1 = ScalarPot(rho,z,r1,mui2,D,DeltaRho,DeltaZ)

	F = rho*(A0 - A1)
	return F



def OldDzFunc(z):

	D = 3.6
	inD = np.abs(z) < D
	winD = np.where(inD)
	woutD = np.where(inD == False)

	out = np.zeros(z.size,dtype='float64')
	out[woutD] = np.abs(z[woutD])*D
	out[winD] = 0.5*(D**2 + z[winD]**2)

	return out*2

def NewDzFunc(z):

	D = 3.6
	deltaz = 0.01
	zpddz = (z + D)/deltaz
	zmddz = (z - D)/deltaz
	tanpd = np.tanh(zpddz)
	tanmd = np.tanh(zmddz)

	term0 = D*z*(tanpd + tanmd)
	term1 = 0.5*(D**2 + z**2)*(tanpd - tanmd)

	out = term0 + term1
	return out

def CompareDzFunc():

	z = np.linspace(-7,7,1001)

	dz0 = OldDzFunc(z)
	dz1 = NewDzFunc(z)
	plt.figure()
	plt.plot(z,dz0,label='Edwards Term')
	plt.plot(z,dz1,label='Smoothed Term')
	plt.legend()


def ThetaIonosphere(r,theta,g=417659.3836476442,
					r0=7.8,r1=51.4,
					mui2=139.6,D=3.6,
					DeltaRho=1.0,DeltaZ=0.1):
	'''
	Calculate theta_i using a dipole field and the 
	Edwards et al flux functions

	Uses equation 17 from Cowley et al 2008

	Inputs
	======
	r : float
		Radial coordinate in Rj
	theta : float
		colatitude (rad)
	g : float
		Dipole coefficient, nT.
	r0 : float
		Inner edge of the current sheet, Rj.
	r1 : float
		Outer edge of the current sheet, Rj.
	mui2 : float
		mu_0 I_0/2 parameter (nT).
	D : float
		current sheet half thickness
	DeltaRho : float
		Scale length over which to smooth the transition
		from small to large rho approximations
	DeltaZ : float
		Scale length over which the large rho approx is
		smoothed across +/-D boundary.

	Returns
	=======
	thetai : float
		Colatitude of the ionospheric footprint of the
		current field line.

	'''
	Ri = 67350000.0
	Rj = 71492000.0


	#CAN flux
	rho = r*np.sin(theta)
	z = r*np.cos(theta)
	Fcan = FluxFunc(rho,z,r0,r1,mui2,D,DeltaRho,DeltaZ)

	#flux due to dipole at current position
	Fdip = g*np.sin(theta)**2/r

	#theta at ionosphere

	thetai = np.arcsin(np.sqrt((Ri/Rj)*(Fcan + Fdip)/g))
	return thetai


	
def BphiLMIC(	r,theta,
				g=417659.3836476442,
				r0=7.8,r1=51.4,
				mui2=139.6,D=3.6,
				DeltaRho=1.0,DeltaZ=0.1,
				wO_open=0.1,wO_om=0.35,
				thetamm=16.1,dthetamm=0.5,
				thetaoc=10.716,dthetaoc=0.125):
	'''
	
	Calculate the Bphi componentusing the L-MIC model

	Inputs
	======
	r : float
		Radial coordinate in Rj
	theta : float
		colatitude (rad)
	g : float
		Dipole coefficient, nT.
	r0 : float
		Inner edge of the current sheet, Rj.
	r1 : float
		Outer edge of the current sheet, Rj.
	mui2 : float
		mu_0 I_0/2 parameter (nT).
	D : float
		current sheet half thickness
	DeltaRho : float
		Scale length over which to smooth the transition
		from small to large rho approximations
	DeltaZ : float
		Scale length over which the large rho approx is
		smoothed across +/-D boundary.
	wO_om : float
		Angular velocity of plasma on open field lines (~0.091)
	wO_om : float
		Angular velocity w of plasma divided by angular velocity of 
		Jupiter in the outer magnetosphere (0.25 for low and 0.5 for
		high velocity models)
	thetamm : float
		central colatitude of the middle magnetosphere (deg)
	dthetamm : float
		colatitudinal width of the middle magnetosphere (deg)
	thetaoc : float
		colatitude of the open-closed field line boundary (deg)
	dthetaoc : float
		colatitudinal width of the OCB (deg)

	Returns
	=======
	Bphi : float
		Bphi, nT.

	'''
	#calculate corresponding ionospheric colatitude and r
	lat = np.pi/2.0 - theta
	thetai = ThetaIonosphere(r,theta,g,r0,r1,mui2,D,DeltaRho,DeltaZ)
	
	IhP = PedersenCurrent(thetai,g,wO_open,wO_om,thetamm,
							dthetamm,thetaoc,dthetaoc)
	mu0 = 4*np.pi*1e-7
	Rj = 71492000.0
	rho = r*np.sin(theta)*Rj
	
	sgn = -np.sign(lat)
	Bphi = (sgn*mu0/(2*np.pi*rho))*IhP
	
	return Bphi*1e9


def PedersenCurrent(thetai,g=417659.3836476442,
					wO_open=0.1,wO_om=0.35,
					thetamm=16.1,dthetamm=0.5,
					thetaoc=10.716,dthetaoc=0.125):
	'''
	Calculate the Pedersen current which maps to a
	given ionospheric latitude using equation 6 of Cowley et
	al., 2008.

	Inputs
	======
	thetai : float
		colatitud of the ionospheric field line footprint.
	g : float
		dipole coefficient, nT.
	wO_om : float
		Angular velocity of plasma on open field lines (~0.091)
	wO_om : float
		Angular velocity w of plasma divided by angular velocity of 
		Jupiter in the outer magnetosphere (0.25 for low and 0.5 for
		high velocity models)
	thetamm : float
		central colatitude of the middle magnetosphere (deg)
	dthetamm : float
		colatitudinal width of the middle magnetosphere (deg)
	thetaoc : float
		colatitude of the open-closed field line boundary (deg)
	dthetaoc : float
		colatitudinal width of the OCB (deg)
	
	Returns
	=======
	IhP : float
		Ionspheric Pedersen current (A).

	'''

	#height integrated conductivity
	SigmaP = 0.25
	
	#ionospheric radius (m)
	Ri = 67350000.0
	
	#angular velocity ratio
	wO = AngularVelocityRatio(thetai,wO_open,wO_om,
								thetamm,dthetamm,
								thetaoc,dthetaoc)

	#domega
	OmegaJ = 1.758e-4
	domega = OmegaJ*(1 - wO)
	
	#f(theta)
	ft = f_theta(thetai)
	
	#absolute radial field
	Rje = 71492000.0
	Bdr = 2*g*np.cos(thetai)*(Rje/Ri)**3
	Bdr = np.abs(Bdr)*1e-9
	
	#now the current
	rhoi = Ri*np.sin(thetai)
	IhP = 2*np.pi*SigmaP*rhoi**2*domega*Bdr*ft
	
	return IhP


def f_theta(thetai):
	'''
	Equation 5 of Cowley et al 2008

	Inputs
	======
	thetai : float
		Colatitude (radians) of the ionospheric footprint of
		the current flux tube.

	Returns
	=======
	f_theta	: float
		1 + 0.25*tan^2 thetai
	
	'''
	return 1.0 + 0.25*np.tanh(thetai)**2
	

def AngularVelocityRatio(	thetai,
							wO_open=0.1,wO_om=0.35,
							thetamm=16.1,dthetamm=0.5,
							thetaoc=10.716,dthetaoc=0.125):
	'''
	Angular velocity mapped to the ionospheric colatitude
	
	Inputs
	======
	theta : float
		colatitud of the ionospheric field line footprint.
	wO_om : float
		Angular velocity of plasma on open field lines (~0.091)
	wO_om : float
		Angular velocity w of plasma divided by angular velocity of 
		Jupiter in the outer magnetosphere (0.25 for low and 0.5 for
		high velocity models)
	thetamm : float
		central colatitude of the middle magnetosphere (deg)
	dthetamm : float
		colatitudinal width of the middle magnetosphere (deg)
	thetaoc : float
		colatitude of the open-closed field line boundary (deg)
	dthetaoc : float
		colatitudinal width of the OCB (deg)
	
	'''
	
	#isabell et al 1984
	#open field line angular velocity
	#wO_open = 0.091
	
	#convert boundary angles to radians
	d2r = np.pi/180.0
	tmm = thetamm*d2r
	dtmm = dthetamm*d2r
	toc = thetaoc*d2r
	dtoc = dthetaoc*d2r
	
	#calculate angular velocity using equation 19 of Cowley et al 2005
	wO = wO_open + 0.5*(wO_om - wO_open)*(1 + np.tanh((thetai - toc)/dtoc)) \
		+ 0.5*(1 - wO_om)*(1 + np.tanh((thetai-tmm)/dtmm))

	return wO
