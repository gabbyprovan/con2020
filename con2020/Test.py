import numpy as np
from scipy.io.idl import readsav
import matplotlib.pyplot as plt
import os
import DateTimeTools as TT
from .Model import Model
from .OldModel import OldModel
import time

def _ConvertTime(year,dayno):
	'''
	This will convert year and day number to date (yyyymmdd) and ut in
	hours since the start of the day. Also will provide a continuous 
	time axis.
	
	'''
	#get the dayno as an integer
	dn = np.int32(np.floor(dayno))
	
	#the date
	Date = TT.DayNotoDate(year,dn)
	
	#time
	ut = (dayno % 1.0)*24.0
	
	#continuous time (for plotting)
	utc = TT.ContUT(Date,ut)
	
	return Date,ut,utc

def _PlotComponent(t,x,xa,xh,xi,maps=[1,1,0,0],Comp='',nox=False):
	'''
	Plot a single component in a panel
	
	'''
	
	#create the subplot
	ax = plt.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
	
	ax.plot(t,x,color='black',label=Comp+' (This Model)',zorder=1.0)
	ax.plot(t,xa,linestyle=':',color='red',label=Comp+' (Con2020 Analytical)',zorder=1.0)
	ax.plot(t,xi,linestyle=':',color='darkorange',label=Comp+' (Con2020 Integral)',zorder=1.0)
	ax.plot(t,xh,linestyle=':',color='darkgoldenrod',label=Comp+' (Con2020 Hybrid)',zorder=1.0)
	
	ax.legend()
	ax.set_ylabel(Comp)
	
	if nox:
		ax.set_xticks([])
	else:
		TT.DTPlotLabel(ax)
		ax.set_xlabel('UT')
	
	return ax

def Test(Edwards=True):
	'''
	Run a quick test to see if the model works - this file might either
	be removed from __init__ or removed completely at some point.
	
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
	
	#convert to another time format
	Date,ut,utc = _ConvertTime(year,dayno)
	
	#and the model inputs (positions)
	r = data.r[0]
	theta = data.SYS3_COLAT_RADS[0]
	phi = data.SYS3_ELONG_RADS[0]
	
	#model fields to test against
	con20_analytic= data.CS_FIELD_ANALYTIC[0]
	con20_hybrid=  data.CS_FIELD_HYBRID[0]
	con20_integral= data.CS_FIELD_INTEGRAL[0]

	#call the model code
	print('Calling Model')
	Br,Bt,Bp = OldModel(r,theta,phi,Edwards=Edwards)
	print('Calling new code')
	M = Model(Edwards=Edwards,CartesianIn=False,CartesianOut=False)
	B = M.Field(r,theta,phi)
	#create a plot
	plt.figure(figsize=(11,8))
	
	ax0 = _PlotComponent(utc,Br,con20_analytic[0],con20_hybrid[0],
			con20_integral[0],maps=[1,3,0,0],Comp=r'$B_{r}$ (nT)',nox=True)
	ax1 = _PlotComponent(utc,Bt,con20_analytic[1],con20_hybrid[1],
			con20_integral[1],maps=[1,3,0,1],Comp=r'$B_{\theta}$ (nT)',nox=True)
	ax2 = _PlotComponent(utc,Bp,con20_analytic[2],con20_hybrid[2],
			con20_integral[2],maps=[1,3,0,2],Comp=r'$B_{\phi}$ (nT)',nox=False)
	ax0.plot(utc,B[:,0],color='magenta',label=r'$B_{r}$ (nT) New Code')
	ax1.plot(utc,B[:,1],color='magenta',label=r'$B_{\theta}$ (nT) New Code')
	ax2.plot(utc,B[:,2],color='magenta',label=r'$B_{\phi}$ (nT) New Code')
	
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
	
	#convert to another time format
	Date,ut,utc = _ConvertTime(year,dayno)
	
	#and the model inputs (positions)
	r = data.r[0]
	theta = data.SYS3_COLAT_RADS[0]
	phi = data.SYS3_ELONG_RADS[0]

	#convert coordinates to get rho
	xt = 9.3
	xp = -24.2
	Deg2Rad = np.pi/180.0
	sin_theta = np.sin(theta)#
	cos_theta = np.cos(theta)#
	sin_phi   = np.sin(phi)#
	cos_phi   = np.cos(phi)#

	dipole_shift = xp*Deg2Rad # % xp_value is longitude of the dipole
	x = r*sin_theta*np.cos(phi-dipole_shift)
	y = r*sin_theta*np.sin(phi-dipole_shift)
	z = r*cos_theta#]

	theta_cs = xt*Deg2Rad # % dipole tilt is xt_value
	x1 = x*np.cos(theta_cs) + z*np.sin(theta_cs)#
	y1 = y# RJW - NOT NEEDED REALLY - BUT USED IN ATAN LATER
	z1 = z*np.cos(theta_cs) - x*np.sin(theta_cs)#
	rho1_sq = x1*x1 + y1*y1
	rho = np.sqrt(rho1_sq)
		
	#find where we cross each boundary
	r0 = 7.8
	r1 = 51.4

	rg0 = rho > r0
	rl0 = rho < r0
	u0 = np.where((rg0[1:] & rl0[:-1]) | (rg0[:-1] & rl0[1:]))[0]
	  
	rg1 = rho > r1
	rl1 = rho < r1
	u1 = np.where((rg1[1:] & rl1[:-1]) | (rg1[:-1] & rl1[1:]))[0]
	  
	utcr0 = 0.5*(utc[u0] + utc[u0+1])
	utcr1 = 0.5*(utc[u1] + utc[u1+1])
	
	#model fields to test against
	con20_analytic= data.CS_FIELD_ANALYTIC[0]
	con20_hybrid=  data.CS_FIELD_HYBRID[0]
	con20_integral= data.CS_FIELD_INTEGRAL[0]

	#call the model code
	print('Calling Model')
	Bro,Bto,Bpo = Model(r,theta,phi,Edwards=False,equation_type='analytic')
	Bre,Bte,Bpe = Model(r,theta,phi,Edwards=True,equation_type='analytic')
	Bri,Bti,Bpi = Model(r,theta,phi,Edwards=False,equation_type='hybrid')

	#create a plot
	plt.figure(figsize=(11,8))
	ax0 = plt.subplot2grid((3,1),(0,0))
	ax0.plot(utc,Bro,color='black',label='Connerney',zorder=1)
	ax0.plot(utc,Bre,color='red',linestyle='--',label=r'Edwards',zorder=3)
	ax0.plot(utc,Bri,color='lime',linestyle='--',label=r'Hybrid',zorder=2)
	
	ax1 = plt.subplot2grid((3,1),(1,0))
	ax1.plot(utc,Bto,color='black',label='Connerney',zorder=1)
	ax1.plot(utc,Bte,color='red',linestyle='--',label='Edwards',zorder=3)
	ax1.plot(utc,Bti,color='lime',linestyle='--',label=r'Hybrid',zorder=2)
	
	ax2 = plt.subplot2grid((3,1),(2,0))
	ax2.plot(utc,Bpo,color='black',label='Connerney',zorder=1)
	ax2.plot(utc,Bpe,color='red',linestyle='--',label='Edwards',zorder=3)
	ax2.plot(utc,Bpi,color='lime',linestyle='--',label=r'Hybrid',zorder=2)
	
	#add lines indicating crossing r0 and r1
	y0 = ax0.get_ylim()
	ax0.set_ylim(y0)
	y1 = ax1.get_ylim()
	ax1.set_ylim(y1)
	y2 = ax2.get_ylim()
	ax2.set_ylim(y2)

	ax0.vlines(utcr0,y0[0],y0[1],color='magenta',linestyle=':',label='$r_0$',zorder=4)
	ax0.vlines(utcr1,y0[0],y0[1],color='magenta',linestyle='--',label='$r_1$',zorder=4)

	ax1.vlines(utcr0,y1[0],y1[1],color='magenta',linestyle=':',label='$r_0$',zorder=4)
	ax1.vlines(utcr1,y1[0],y1[1],color='magenta',linestyle='--',label='$r_1$',zorder=4)

	ax2.vlines(utcr0,y2[0],y2[1],color='magenta',linestyle=':',label='$r_0$',zorder=4)
	ax2.vlines(utcr1,y2[0],y2[1],color='magenta',linestyle='--',label='$r_1$',zorder=4)

	ax0.set_ylabel(r'$B_{r}$ (nT)')
	ax1.set_ylabel(r'$B_{\theta}$ (nT)')
	ax2.set_ylabel(r'$B_{\phi}$ (nT)')

	
	ax0.set_xticks([])
	ax1.set_xticks([])
	TT.DTPlotLabel(ax2)
	ax2.set_xlabel('UT')
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
	print('Calling Old Integral Model')
	ti0 = time.time()
	Br,Bt,Bp = OldModel(r,theta,phi,equation_type='integral')
	ti1 = time.time()
	print('Completed in {:f}s'.format(ti1-ti0))
	
	print('Calling Old Analytic Model')
	ta0 = time.time()
	Br,Bt,Bp = OldModel(r,theta,phi,equation_type='analytic')
	ta1 = time.time()
	print('Completed in {:f}s'.format(ta1-ta0))
	
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
	print('Calling Old Integral Model')
	ti0 = time.time()
	for i in range(0,n):
		Br,Bt,Bp = OldModel(r[i],theta[i],phi[i],equation_type='integral')
	ti1 = time.time()
	print('Completed in {:f}s'.format(ti1-ti0))
	
	print('Calling Old Analytic Model')
	ta0 = time.time()
	for i in range(0,n):
		Br,Bt,Bp = OldModel(r[i],theta[i],phi[i],equation_type='analytic')
	ta1 = time.time()
	print('Completed in {:f}s'.format(ta1-ta0))
	
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
	Br,Bt,Bp = Model(r,theta,phi,Edwards=True,equation_type='hybrid')
	
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
	out.Br = Br
	out.Bt = Bt
	out.Bp = Bp
	
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
