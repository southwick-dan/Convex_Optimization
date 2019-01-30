#!/usr/bin/env python		
import sys
from optparse import OptionParser
import matplotlib.pyplot as pl
import numpy as np


def process_args():
    parser = OptionParser(usage='%prog [options] <name> <path>')
    parser.add_option("-c", help="clean", action="store_true", default=False)
    parser.add_option("-f", help="file",default='')		# type = "string" as default
    parser.add_option("--fmt", help="fig format",type = "string",default="png")
    parser.add_option("--clean", help="clean", action="store_true", default=False)
    parser.add_option("--log", help="plot y in log axis", action="store_true", default=False)
    (opts, args) = parser.parse_args()
    for n, m in opts.__dict__.iteritems():# Verify that that all the arguments have been provided
        if m == None:
            print n,m
            print >>sys.stderr, "missing argument: -%c" %n
            sys.exit(1)
    return opts,args

def main(argv=None):
    opts,args=process_args()
    if opts.c:
            clean()
    plot_figure(opts)
    print >>sys.stderr, "    --------- done --------  "     # done  

def plot_figure(opts):

    pl.rc('font',family='Arial')

    ffs = (opts.f).split('/')
    fs = ffs[-1].split('_')

    method = fs[1]
    dn = fs[0]
    if method == "newton":
        method = "Newton"
    epsilon = float(fs[-1][0:-4])

    fig1=pl.figure(1,figsize=(6,5))
    pl.clf()
    pl.subplots_adjust(left=0.15,hspace=0.05,wspace=0.35,right=0.96,top=0.98,bottom=0.13)
            
    ax1=pl.subplot(111)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.0)
    ax1.yaxis.set_major_locator(pl.MaxNLocator(5))
    ax1.tick_params(axis='both',labelsize=16)
    ax1.tick_params(width=2.0)

    pl.xlabel('Iteration',fontsize=20)
    pl.ylabel('Function value',fontsize=20)

    data = np.loadtxt(opts.f)
    iteration = np.arange(len(data))
    f_value = data
    if opts.log:
        pl.semilogy(iteration,f_value,'r-',linewidth=1.0)
    else:
        pl.plot(iteration,f_value,'r-',linewidth=1.0)

    lefty = 0.55
    if dn == "gisette":
        lefty = 0.62
    elif dn == "spamData":
        lefty = 0.55
    ax1.annotate('Data: %s.mat\n$\epsilon$: %f\nMethod: %s'%(dn,epsilon,method),xy=(lefty,0.8),xycoords='axes fraction',fontsize=16,color='blue')
    #ax1.annotate('Data: %s.mat\nMethod: %s'%(dn,method),xy=(lefty,0.8),xycoords='axes fraction',fontsize=16,color='blue')

    fig1.savefig('./%s_%s_epsilon_%s.png'%(dn,method,epsilon))




if __name__ == "__main__":                      # invoke main function
	sys.exit(main())
