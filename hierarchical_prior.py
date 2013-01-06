"""binomial likelihood with hierarchical prior 
example taken from: Kruscke JK, 'Doing Bayesian Data Analysis'
Chapter 9.2, p. 169 ff
"""

import pymc
import numpy as np
from pymc.Matplot import plot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def set_flips(data, A_mu, B_mu, A_kappa, B_kappa):

    if not (A_mu or B_mu or A_kappa or B_kappa):
        return None

    mu = pymc.Beta('mu', alpha=A_mu, beta=B_mu)
    kappa = pymc.Gamma('kappa', alpha=A_kappa, beta=B_kappa)

    @pymc.deterministic
    def alpha(mu=mu, kappa=kappa):
        return mu * kappa

    @pymc.deterministic
    def beta(mu=mu, kappa=kappa):
        return (1 - mu) * kappa


    coin_count = 0
    flips = []
    thetas = []

    for N, z in data:

        # generate coinflips for MCMC
    
        coin_count += 1

        print "CURRENT FLIP", coin_count

        flip_data = [1 for i in range(0,z)]
        flip_data += [0 for i in range(z+1,N+1)]
        flip_data = np.array(flip_data)
    
        print "FLIP DATA", flip_data
    
        theta = pymc.Beta('theta%s' % coin_count, alpha=alpha, beta=beta)
        thetas.append(theta)

        flip = pymc.Bernoulli('flip%s' % coin_count, p=theta, value=flip_data, observed=True)
        flips.append(flip)

    return thetas, flips, mu, kappa

def gen_plot(model, plot_no, file_name='posterior', format='pdf'):
    plot(model, format='pdf')
    fig = Figure(figsize=(plot_no*4,plot_no*4))
    canvas = FigureCanvas(fig)
    i = 1
    for t in range(1, plot_no + 1):
        for p in ['mu', 'kappa']:
            ax = fig.add_subplot(plot_no, 3, i)
            ax.set_xlim([0,1])
            ax.set_xlabel('theta%s' % t, fontsize=12)
            ax.set_ylabel(p, fontsize=12)
            ax.grid(True,linestyle='-',color='0.75')
            if i == 1:
                ax.scatter(model.trace('mu')[:], model.trace('kappa')[:], s=20, color='tomato')
            elif i == 2:
                ax.hist(model.trace('mu')[:])
            elif i == 3:
                ax.hist(model.trace('mu')[:])
            elif i%3 == 1:
                pass
                #ax.hist(model.trace('theta%s' %t)[:])
            else:
                print "ADDING SCATTER PLOT FOR THETA", t, p
                #ax.scatter(model.trace('theta%s' % t)[:], model.trace(p)[:], s=20,color='tomato')
            i += 1

    canvas.print_figure('%s.%s' % (file_name, format))
    

def main():
    # set parameters for hierarchical prior
    A_mu = 2
    B_mu = 2
    A_kappa = 1
    B_kappa = 0.1
    # flip data format (N, z), where N=sample number, z=number of heads (i.e. 1)
    #data = [(10,1), (10,5), (10,9)]
    data = [(5,1), (5,1), (5,1), (5,1), (5,5)]

    # generate model
    model=pymc.MCMC(set_flips(data, 
                              A_mu, 
                              B_mu, 
                              A_kappa, 
                              B_kappa))

    if model:
        # run MCMC and plot/print stats
        model.sample(iter=10000, burn=9000, thin=2)
        gen_plot(model, plot_no=len(data), format='pdf')
        print "NODE\t95%HPD\tMEAN\tSD"
        for node, stats in model.stats().items():
            print "%s\t%s\t%.3f\t%.3f" % (node, "-".join(["%.3f" %x for x in stats['95% HPD interval']]), stats['mean'], stats['standard deviation'])
    
if __name__ == '__main__':
    main()
