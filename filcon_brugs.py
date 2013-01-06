"""Hierarchical prior with binomial adapted from:
Kruschke, J. K. (2010). Doing Bayesian data analysis:
A Tutorial with R and BUGS. Academic Press / Elsevier.
p 178 ff."""

import pymc
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import time

def filcon_brugs(data, a_kappa, b_kappa, a_mu, b_mu, N):
    model_data = []
    for condition in range(len(data)):
        mu = pymc.Beta('mu%s' % condition, alpha=a_mu, beta=b_mu)
        kappa = pymc.Gamma('kappa%s' % condition, alpha=a_kappa, beta=b_kappa)

        model_data.extend([mu, kappa])

        @pymc.deterministic
        def alpha(mu=mu, kappa=kappa):
            return mu * kappa

        @pymc.deterministic
        def beta(mu=mu, kappa=kappa):
            return (1 - mu) * kappa

        
        for  subject in range(len(data[condition])):
            print "SUBJECT", condition, subject
            theta = pymc.Beta('theta%s_%s' % (condition, subject), alpha=alpha, beta=beta)
            trial = pymc.Binomial('trial%s_%s' % (condition, subject), n=N, p=theta, value=data[condition], observed=True)    

            model_data.extend([theta, trial]) 

    return model_data

def gen_plot(model, node, pairings, file_name="filcon_brugs", format='pdf'):
    plot_no = len(pairings)
    print "PLOT NO", plot_no
    fig = Figure(figsize=(plot_no*4,plot_no*4))
    canvas = FigureCanvas(fig)
    i = 1
    for p1, p2 in pairings:
        print "ADDING PLOT", i
        diff = [model.trace('%s%s' % (node, p1))[:][j] - model.trace('%s%s' % (node, p2))[:][j] for j in range(len(model.trace('%s%s' % (node, p1))[:]))]
        print "DIFF %s%s - %s%s MEAN %.3f SD %.3f" % (node, p1, node, p2, np.mean(diff), np.std(diff))
        ax = fig.add_subplot(plot_no, 1, i)
        ax.set_xlabel('mu%s - mu%s' % (p1, p2), fontsize=12)
        ax.hist(diff)
        i += 1
    canvas.print_figure('%s.%s' % (file_name, format))

def main():
    data = [[45,63,58,64,58,63,51,60,59,47,
             63,61,60,51,59,45,61,59,60,58,
             63,56,63,64,64,60,64,62,49,64,
             64,58,64,52,64,64,64,62,64,61],
            [59,59,55,62,51,58,55,54,59,57,
             58,60,54,42,59,57,59,53,53,42,
             59,57,29,36,51,64,60,54,54,38,
             61,60,61,60,62,55,38,43,58,60],
            [44,44,32,56,43,36,38,48,32,40,
             40,34,45,42,41,32,48,36,29,37,
             53,55,50,47,46,44,50,56,58,42,
             58,54,57,54,51,49,52,51,49,51],
            [46,46,42,49,46,56,42,53,55,51,
             55,49,53,55,40,46,56,47,54,54,
             42,34,35,41,48,46,39,55,30,49,
             27,51,41,36,45,41,53,32,43,33]]

    mean_gamma = 10.
    sd_gamma = 10.
    a_kappa = mean_gamma**2/sd_gamma**2
    b_kappa = mean_gamma/sd_gamma**2
    a_mu = 1
    b_mu = 1
    N = 64

    model = pymc.MCMC(filcon_brugs(data, a_kappa, b_kappa, a_mu, b_mu, N))
    start = time.time()
    print "START SAMPLING"
    model.sample(iter=10000,burn=8000,thin=2)
    print "FINISHED SAMPLING AFTER %s MIN" % (time.time()-start/60)
    print "NODE\t95%HPD\tMEAN\tSD"
    for node, stats in model.stats().items():
        print "%s\t%s\t%.3f\t%.3f" % (node, "-".join(["%.3f" %x for x in stats['95% HPD interval']]), stats['mean'], stats['standard deviation'])
    #gen_plot(model, 'mu', [(0,1), (2,3), ((0,1), (2,3))])
    gen_plot(model, 'mu', [(0,1), (2,3)])

    return model


if __name__ == '__main__':
    main()
