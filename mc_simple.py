import pymc
import hierarchical_prior
from pymc.Matplot import plot

def main():

    Amu = 2
    Bmu = 2
    Skappa = 1
    Rkappa = 0.1
    data = [(12, 6), (7,2)]

    model=pymc.MCMC(hierarchical_prior.set_flips(data, 
                                                 Amu, 
                                                 Bmu, 
                                                 Skappa, 
                                                 Rkappa))



    print "PRIOR KAPPA/MU", model.kappa.value, model.mu.value
    model.sample(iter=10000, burn=9000, thin=2)
    print "POST KAPPA/MU", model.kappa.value, model.mu.value
    
    print "MODEL", model.variables

    plot(model, format='pdf')
    
if __name__ == '__main__':

    main()
