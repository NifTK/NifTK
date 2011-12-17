import scipy.stats as stat


def mean(X):
    return sum(X)/ float(len(X))
 

def bootstrap(sample, samplesize = None, nsamples = 1000, statfunc = mean):
    """
    Arguments:
       sample     - input sample of values
       nsamples   - number of samples to generate
       samplesize - sample size of each generated sample
       statfunc   - statistical function to apply to each generated sample.
 
    Performs resampling from sample with replacement, gathers
    statistic in a list computed by statfunc on the each generated sample.
    """
    if samplesize is None:                                                                   
        samplesize=len(sample)
    print "input sample = ",  sample
    n = len(sample)
    X = []
    for i in range(nsamples):
        print "i = ",  i, 
        resample = [sample[j] for j in stat.randint.rvs(0, n-1, size=samplesize)] 
        x = statfunc(resample)
        X.append(x)
    return X