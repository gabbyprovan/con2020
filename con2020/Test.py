import numpy as np
from scipy.io.idl import readsav
import matplotlib.pyplot as plt
import os
from .Model import Model
import time

def _RCrossings(doy,r,r0,r1):
	'''
	Return the day numbers where r0 and r1 were crossed.
	
	Inputs
	======
	doy : float
		Day number array.
	r : float
		Array of R.
	r0 : float
		Inside edge of the current.
	r0 : float 
		Outside edge of the current.
		
	Returns
	=======
	dc : float
		Day numbers of the r0/r1 crossings.
	
	'''
	
	rc0 = ((r[1:] >= r0) & (r[:-1] < r0)) | ((r[:-1] >= r0) & (r[1:] < r0))
	rc1 = ((r[1:] >= r1) & (r[:-1] < r1)) | ((r[:-1] >= r1) & (r[1:] < r1))
	rc0 = np.where(rc0)[0]
	rc1 = np.where(rc1)[0]
	rc = np.append(rc0,rc1)
	rc.sort()
	
	dc = 0.5*(doy[rc] + doy[rc+1])
	
	return dc
	
	
	
	
def Test():
	'''
	Run a quick test to see if the model works.
	
	'''
	# this is the path of this source file
	ModulePath = os.path.dirname(__file__)+'/'

	#name and path of the test data file
	fname = ModulePath + '__data/testdata.sav'

	#read the data
	print('Reading Data')
	data = readsav(fname).test

	#get the time
	year = data.time_year[0]
	dayno = data.time_ddate[0]
	
	#limit the dayno
	use = np.where((dayno >= 290) & (dayno <= 315))[0]
	year = year[use]
	dayno = dayno[use]
	
	#and the model inputs (positions)
	r = data.r[0][use]
	theta = data.SYS3_COLAT_RADS[0][use]
	phi = data.SYS3_ELONG_RADS[0][use]
	
	#model fields to test against
	con20_analytic= data.CS_FIELD_ANALYTIC[0]
	con20_hybrid=  data.CS_FIELD_HYBRID[0]
	con20_integral= data.CS_FIELD_INTEGRAL[0]

	#call the model code
	print('Calling Model')
	M = Model(CartesianIn=False,CartesianOut=False,mu_i=149.9)
	B = M.Field(r,theta,phi)
	
	#get the r0/r1 crossings
	dc = _RCrossings(dayno,r,M.r0,M.r1)

	#create a plot window
	plt.figure(figsize=(11,8))
	
	#create the subplots
	ax0 = plt.subplot2grid((4,1),(0,0))
	ax1 = plt.subplot2grid((4,1),(1,0))
	ax2 = plt.subplot2grid((4,1),(2,0))
	ax3 = plt.subplot2grid((4,1),(3,0))
	
	#plot each component
	ax0.plot(dayno,B[:,0],color='k',label=r'$B_{r}$ (nT)')
	ax1.plot(dayno,B[:,1],color='k',label=r'$B_{\theta}$ (nT)')
	ax2.plot(dayno,B[:,2],color='k',label=r'$B_{\phi}$')
	ax3.plot(dayno,r,color='k',label=r'$r$')
	
	#fix y limits
	y0 = ax0.get_ylim()
	y1 = ax1.get_ylim()
	y2 = ax2.get_ylim()
	y3 = ax3.get_ylim()
	ax0.set_ylim(y0)
	ax1.set_ylim(y1)
	ax2.set_ylim(y2)
	ax3.set_ylim(y3)
	
	#and x limits
	ax0.set_xlim([290,315])
	ax1.set_xlim([290,315])
	ax2.set_xlim([290,315])
	ax3.set_xlim([290,315])
	
	#plot r0/r1 crossings
	ax0.vlines(dc,y0[0],y0[1],color='k',linestyle='--')
	ax1.vlines(dc,y1[0],y1[1],color='k',linestyle='--')
	ax2.vlines(dc,y2[0],y2[1],color='k',linestyle='--')
	ax3.vlines(dc,y3[0],y3[1],color='k',linestyle='--')
	
	#y labels plot labels
	ax0.set_ylabel(r'$B_r$ / nT')
	ax1.set_ylabel(r'$B_{\theta}$ / nT')
	ax2.set_ylabel(r'$B_{\phi}$ / nT')
	ax3.set_ylabel(r'$r$ / R$_J$')

	#title
	ax0.set_title(r'PJ16, con2020: $\mu_0I_{MD}/2$=' + '{:5.1f}, $R_0$={:3.1f} R$_J$, $R_1$={:4.1f} R$_J$'.format(M.i_rho,M.r0,M.r1))
	
	#x labels
	ax0.set_xticks([])
	ax1.set_xticks([])
	ax2.set_xticks([])
	ax3.set_xlabel('DOY (2018)')
	
	plt.subplots_adjust(hspace=0.0)

def TestCompareAnalytic():
	'''
	Compare the two analytic models.
	
	'''
	# this is the path of this source file
	ModulePath = os.path.dirname(__file__)+'/'

	#name and path of the test data file
	fname = ModulePath + '__data/testdata.sav'

	#read the data
	print('Reading Data')
	data = readsav(fname).test

	#get the time
	year = data.time_year[0]
	dayno = data.time_ddate[0]
	
	#limit the dayno
	use = np.where((dayno >= 290) & (dayno <= 315))[0]
	year = year[use]
	dayno = dayno[use]
	
	#and the model inputs (positions)
	r = data.r[0][use]
	theta = data.SYS3_COLAT_RADS[0][use]
	phi = data.SYS3_ELONG_RADS[0][use]
	
	#call the model code
	print('Calling Model')
	Me = Model(Edwards=True,CartesianIn=False,CartesianOut=False,equation_type='analytic')
	Mo = Model(Edwards=False,CartesianIn=False,CartesianOut=False,equation_type='analytic')
	Be = Me.Field(r,theta,phi)
	Bo = Mo.Field(r,theta,phi)
	
	#get the r0/r1 crossings
	dc = _RCrossings(dayno,r,Me.r0,Me.r1)


	#create a plot window
	plt.figure(figsize=(11,8))
	
	#create the subplots
	ax0 = plt.subplot2grid((4,1),(0,0))
	ax1 = plt.subplot2grid((4,1),(1,0))
	ax2 = plt.subplot2grid((4,1),(2,0))
	ax3 = plt.subplot2grid((4,1),(3,0))
	
	#plot each component
	ax0.plot(dayno,Be[:,0],color='k',label=r'$B_{r}$ (nT) (Edwards)')
	ax1.plot(dayno,Be[:,1],color='k',label=r'$B_{\theta}$ (nT) (Edwards)')
	ax2.plot(dayno,Be[:,2],color='k',label=r'$B_{\phi}$ (nT) (Edwards)')
	ax0.plot(dayno,Bo[:,0],color='r',label=r'$B_{r}$ (nT) (Connerney)')
	ax1.plot(dayno,Bo[:,1],color='r',label=r'$B_{\theta}$ (nT) (Connerney)')
	ax2.plot(dayno,Bo[:,2],color='r',label=r'$B_{\phi}$ (nT) (Connerney)')
	ax3.plot(dayno,r,color='k',label=r'$r$')
	
	#fix y limits
	y0 = ax0.get_ylim()
	y1 = ax1.get_ylim()
	y2 = ax2.get_ylim()
	y3 = ax3.get_ylim()
	ax0.set_ylim(y0)
	ax1.set_ylim(y1)
	ax2.set_ylim(y2)
	ax3.set_ylim(y3)
	
	#and x limits
	ax0.set_xlim([290,315])
	ax1.set_xlim([290,315])
	ax2.set_xlim([290,315])
	ax3.set_xlim([290,315])
	
	#plot r0/r1 crossings
	ax0.vlines(dc,y0[0],y0[1],color='k',linestyle='--')
	ax1.vlines(dc,y1[0],y1[1],color='k',linestyle='--')
	ax2.vlines(dc,y2[0],y2[1],color='k',linestyle='--')
	ax3.vlines(dc,y3[0],y3[1],color='k',linestyle='--')
	
	#y labels plot labels
	ax0.set_ylabel(r'$B_r$ / nT')
	ax1.set_ylabel(r'$B_{\theta}$ / nT')
	ax2.set_ylabel(r'$B_{\phi}$ / nT')
	ax3.set_ylabel(r'$r$ / R$_J$')

	#x labels
	ax0.set_xticks([])
	ax1.set_xticks([])
	ax2.set_xticks([])
	ax3.set_xlabel('DOY (2018)')
	
	#legends
	ax0.legend()
	ax1.legend()
	ax2.legend()
	
	plt.subplots_adjust(hspace=0.0)

def TestTimingIntVsAn(n=1000):
	'''
	Compare the timing of the integral Vs analytic equation types using
	an array of positions.
	
	Prior to modifications: 
		ti = 2.4s
		ta = 0.0003s
		
	Moved initialization of _Switcher object outside of loop:
		no change

	NewSwitcher:
		no change
		
	Switch to j0 and j1:
		Much faster
		ti : 0.6s
	
	'''
	
	# this is the path of this source file
	ModulePath = os.path.dirname(__file__)+'/'

	#name and path of the test data file
	fname = ModulePath + '__data/testdata.sav'

	#read the data
	print('Reading Data')
	data = readsav(fname).test

	#create the new model object
	MI = Model(equation_type='integral',CartesianIn=False,CartesianOut=False)
	MA = Model(equation_type='analytic',CartesianIn=False,CartesianOut=False)
	
	#and the model inputs (positions)
	r = data.r[0][:n]
	theta = data.SYS3_COLAT_RADS[0][:n]
	phi = data.SYS3_ELONG_RADS[0][:n]
	
	print('Testing {:d} model vectors'.format(n))

	#call the model code
	print('Calling Integral Model')
	ti0 = time.time()
	B = MI.Field(r,theta,phi)
	ti1 = time.time()
	print('Completed in {:f}s'.format(ti1-ti0))
	
	print('Calling Analytic Model')
	ta0 = time.time()
	B = MA.Field(r,theta,phi)
	ta1 = time.time()
	print('Completed in {:f}s'.format(ta1-ta0))
	
def TestTimingIntVsAnSingle(n=1000):
	'''
	Compare the time taken to call model one position at a time.

	Prior to modifications: 
		ti = 4.5s
		ta = 0.002s
	
	Moved initialization of _Switcher object outside of loop:
		no change
		
	NewSwitcher:
		no change

	Switch to j0 and j1:
		Much faster
		ti : 0.8s
	
	'''
	
	# this is the path of this source file
	ModulePath = os.path.dirname(__file__)+'/'

	#name and path of the test data file
	fname = ModulePath + '__data/testdata.sav'

	#read the data
	print('Reading Data')
	data = readsav(fname).test
	
	#and the model inputs (positions)
	r = data.r[0][:n]
	theta = data.SYS3_COLAT_RADS[0][:n]
	phi = data.SYS3_ELONG_RADS[0][:n]
	
	#create the new model object
	MI = Model(equation_type='integral',CartesianIn=False,CartesianOut=False)
	MA = Model(equation_type='analytic',CartesianIn=False,CartesianOut=False)
	
	print('Testing {:d} model vectors'.format(n))

	#call the model code
	print('Calling Integral Model')
	ti0 = time.time()
	for i in range(0,n):
		B = MI.Field(r[i],theta[i],phi[i])
	ti1 = time.time()
	print('Completed in {:f}s'.format(ti1-ti0))
	
	print('Calling Analytic Model')
	ta0 = time.time()
	for i in range(0,n):
		B = MA.Field(r[i],theta[i],phi[i])
	ta1 = time.time()
	print('Completed in {:f}s'.format(ta1-ta0))
	
def _ConvertTime(year,dayno):
	'''
	This will convert year and day number to date (yyyymmdd) and ut in
	hours since the start of the day. Also will provide a continuous 
	time axis.
	
	'''
	try:
		import DateTimeTools as TT
	except:
		print('Install DateTimeTools package to use this function:')
		print('pip3 install DateTimeTools --user')
		
	#get the dayno as an integer
	dn = np.int32(np.floor(dayno))
	
	#the date
	Date = TT.DayNotoDate(year,dn)
	
	#time
	ut = (dayno % 1.0)*24.0
	
	#continuous time (for plotting)
	utc = TT.ContUT(Date,ut)
	
	return Date,ut,utc
	
	
def Dump(n = 5):
	import PyFileIO as pf
	
	# this is the path of this source file
	ModulePath = os.path.dirname(__file__)+'/'

	#name and path of the test data file
	fname = ModulePath + '__data/testdata.sav'

	#read the data
	print('Reading Data')
	data = readsav(fname).test

	#get the time
	year = data.time_year[0]
	dayno = data.time_ddate[0]
	
	#convert to another time format
	Date,ut,utc = _ConvertTime(year,dayno)
	
	#and the model inputs (positions)
	
	r = data.r[0][:n]
	theta = data.SYS3_COLAT_RADS[0][:n]
	phi = data.SYS3_ELONG_RADS[0][:n]
	
	#model fields to test against
	con20_analytic= data.CS_FIELD_ANALYTIC[0][:n]
	con20_hybrid=  data.CS_FIELD_HYBRID[0][:n]
	con20_integral= data.CS_FIELD_INTEGRAL[0][:n]

	#call the model code
	print('Calling Model')
	Me = Model(Edwards=True,CartesianIn=False,CartesianOut=False)
	B = Model(r,theta,phi)
	
	
	#output
	dtype = [	('Date','int32'),
				('ut','float32'),
				('r','float32'),
				('t','float32'),
				('p','float32'),
				('Br','float32'),
				('Bt','float32'),
				('Bp','float32'),]
	out = np.recarray(n,dtype=dtype)
	
	out.Date = Date[:n]
	out.ut = ut[:n]
	out.r = r
	out.t = theta
	out.p = phi
	out.Br = B[:,0]
	out.Bt = B[:,1]
	out.Bp = B[:,2]
	
	#dump it to a file in the home directory
	pf.WriteASCIIData('con2020dump.dat',out)


def TestBessel(rho,z):
	from scipy.special import j0,j1
	
	abs_z = np.abs(z)
	
	m = Model()


	#check which "zcase" we need for this vector
	check1 = np.abs(abs_z - m.d)		
	check2 = abs_z <= m.d*1.1
		
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
	beselj_rho_rho1_1 = j1(m.lambda_int_brho[zc]*rho)
	beselj_z_rho1_0   = j0(m.lambda_int_bz[zc]*rho)
	if (abs_z > m.d): #% Connerney et al. 1981 eqs. 14 and 15
		brho_int_funct = beselj_rho_rho1_1*m.beselj_rho_r0_0[zc] \
						*np.sinh(m.d*m.lambda_int_brho[zc]) \
						*np.exp(-abs_z*m.lambda_int_brho[zc]) \
						/m.lambda_int_brho[zc]
		bz_int_funct   = beselj_z_rho1_0 *m.beselj_z_r0_0[zc] \
						*np.sinh(m.d*m.lambda_int_bz[zc]) \
						*np.exp(-abs_z*m.lambda_int_bz[zc]) \
						/m.lambda_int_bz[zc]  
		#Brho = m.mu_i*2.0*_Integrate(brho_int_funct,m.dlambda_brho)
		#if z < 0:
		#	Brho = -Brho
	else:
		brho_int_funct = beselj_rho_rho1_1*m.beselj_rho_r0_0[zc] \
						*(np.sinh(z*m.lambda_int_brho[zc]) \
						*np.exp(-m.d*m.lambda_int_brho[zc])) \
						/m.lambda_int_brho[zc]
		bz_int_funct   = beselj_z_rho1_0  *m.beselj_z_r0_0[zc] \
						*(1.0 -np.cosh(z*m.lambda_int_bz[zc]) \
						*np.exp(-m.d*m.lambda_int_bz[zc])) \
						/m.lambda_int_bz[zc]
		#Brho = m.mu_i*2.0*_Integrate(brho_int_funct,m.dlambda_brho)#
	#Bz = m.mu_i*2.0*_Integrate(bz_int_funct,m.dlambda_bz)	
	
	
	plt.figure(figsize=(11,8))
	ax = plt.subplot2grid((1,1),(0,0))
	
	ax.plot(m.lambda_int_brho[zc],brho_int_funct,label=r'$\rho$')
	ax.plot(m.lambda_int_bz[zc],bz_int_funct,label=r'$z$')
	ax.legend()
