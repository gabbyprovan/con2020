import numpy as np
from scipy.io.idl import readsav
import matplotlib.pyplot as plt
import os
import DateTimeTools as TT
from .Model import Model

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
	ax.plot(t,xa,linestyle=':',color='red',label=Comp+' (JRM09 Analytical)',zorder=1.0)
	ax.plot(t,xi,linestyle=':',color='darkorange',label=Comp+' (JRM09 Integral)',zorder=1.0)
	ax.plot(t,xh,linestyle=':',color='darkgoldenrod',label=Comp+' (JRM09 Hybrid)',zorder=1.0)
	
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

	inds = np.arange(5000) + 30000
	#get the time
	year = data.time_year[0][inds]
	dayno = data.time_ddate[0][inds]
	
	#convert to another time format
	Date,ut,utc = _ConvertTime(year,dayno)
	
	#and the model inputs (positions)
	r = data.r[0][inds]
	theta = data.SYS3_COLAT_RADS[0][inds]
	phi = data.SYS3_ELONG_RADS[0][inds]
	
	#model fields to test against
	jrm09_analytic= data.CS_FIELD_ANALYTIC[0][:,inds]
	jrm09_hybrid=  data.CS_FIELD_HYBRID[0][:,inds]
	jrm09_integral= data.CS_FIELD_INTEGRAL[0][:,inds]

	#call the model code
	print('Calling Model')
	Br,Bt,Bp = Model(r,theta,phi)

	#create a plot
	plt.figure(figsize=(11,8))
	
	ax0 = _PlotComponent(utc,Br,jrm09_analytic[0],jrm09_hybrid[0],
			jrm09_integral[0],maps=[1,3,0,0],Comp=r'$B_{r}$',nox=True)
	ax1 = _PlotComponent(utc,Bt,jrm09_analytic[1],jrm09_hybrid[1],
			jrm09_integral[1],maps=[1,3,0,1],Comp=r'$B_{\theta}$',nox=True)
	ax2 = _PlotComponent(utc,Bp,jrm09_analytic[2],jrm09_hybrid[2],
			jrm09_integral[2],maps=[1,3,0,2],Comp=r'$B_{\phi}$',nox=False)

	
	plt.subplots_adjust(hspace=0.0)





def TestSingle():
	'''
	Run a quick test to see if the model works - this file might either
	be removed from __init__ or removed completely at some point.
	
	'''
	# this is the path of this source file
	ModulePath = os.path.dirname(__file__)+'/'

	#name and path of the test data file
	fname = ModulePath + '__data/peri-16_pc_r60s_examples_all_update1.sav'

	#read the data
	print('Reading Data')
	data = readsav(fname).test

	inds = 30237
	#get the time
	year = data.time_year[0][inds]
	dayno = data.time_ddate[0][inds]
	
	#convert to another time format
	Date,ut,utc = _ConvertTime(year,dayno)
	
	#and the model inputs (positions)
	r = data.r[0][inds]
	theta = data.SYS3_COLAT_RADS[0][inds]
	phi = data.SYS3_ELONG_RADS[0][inds]
	print(r,theta,phi)
	#model fields to test against
	jrm09_analytic= data.CS_FIELD_ANALYTIC[0][:,inds]
	jrm09_hybrid=  data.CS_FIELD_HYBRID[0][:,inds]
	jrm09_integral= data.CS_FIELD_INTEGRAL[0][:,inds]

	#call the model code
	print('Calling Model')
	Br,Bt,Bp = Model(r,theta,phi)
	
