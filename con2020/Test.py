import numpy as np
from scipy.io.idl import readsav
import matplotlib.pyplot as plt
import os
import DateTimeTools as TT
from .Model import Model
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

def Test():
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
	Br,Bt,Bp = Model(r,theta,phi)

	#create a plot
	plt.figure(figsize=(11,8))
	
	ax0 = _PlotComponent(utc,Br,con20_analytic[0],con20_hybrid[0],
			con20_integral[0],maps=[1,3,0,0],Comp=r'$B_{r}$',nox=True)
	ax1 = _PlotComponent(utc,Bt,con20_analytic[1],con20_hybrid[1],
			con20_integral[1],maps=[1,3,0,1],Comp=r'$B_{\theta}$',nox=True)
	ax2 = _PlotComponent(utc,Bp,con20_analytic[2],con20_hybrid[2],
			con20_integral[2],maps=[1,3,0,2],Comp=r'$B_{\phi}$',nox=False)

	
	plt.subplots_adjust(hspace=0.0)



def TestTimingIntVsAn(n=10):
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
	
	#and the model inputs (positions)
	r = data.r[0][:n]
	theta = data.SYS3_COLAT_RADS[0][:n]
	phi = data.SYS3_ELONG_RADS[0][:n]
	print(r.size)
	
	print('Testing {:d} model vectors'.format(n))
	#call the model code
	print('Calling Integral Model')
	ti0 = time.time()
	Br,Bt,Bp = Model(r,theta,phi,equation_type='integral')
	ti1 = time.time()
	print('Completed in {:f}s'.format(ti1-ti0))
	
	print('Calling Analytic Model')
	ta0 = time.time()
	Br,Bt,Bp = Model(r,theta,phi,equation_type='analytic')
	ta1 = time.time()
	print('Completed in {:f}s'.format(ta1-ta0))
	
def TestTimingIntVsAnSingle(n=10):
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
	
	print('Testing {:d} model vectors'.format(n))
	#call the model code
	print('Calling Integral Model')
	ti0 = time.time()
	for i in range(0,n):
		Br,Bt,Bp = Model(r[i],theta[i],phi[i],equation_type='integral')
	ti1 = time.time()
	print('Completed in {:f}s'.format(ti1-ti0))
	
	print('Calling Analytic Model')
	ta0 = time.time()
	for i in range(0,n):
		Br,Bt,Bp = Model(r[i],theta[i],phi[i],equation_type='analytic')
	ta1 = time.time()
	print('Completed in {:f}s'.format(ta1-ta0))
	
