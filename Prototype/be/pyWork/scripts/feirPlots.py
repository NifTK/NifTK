#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import configurationPlotter as cfgPlt
from pylab import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show, xlim, ylim
import matplotlib.pyplot as plt

# FEIR folders...
baseDirNetwork = 'X:/NiftyRegValidationWithTannerData/outFEIR/'
baseDirLocal   = 'C:/data/regValidationWithTannerData/outFEIR/'

# fast masked...
cfgLstFastM = [ 'cfg011',
                'cfg012',
                'cfg013',
                'cfg014',
                'cfg015',
                'cfg016',
                'cfg001',
                'cfg002',
                'cfg003',
                'cfg004',
                'cfg005',
                'cfg006',
                'cfg007',
                'cfg008',
                'cfg009',
                'cfg010' ]
                
# fast NOT masked
cfgLstFastNOM = [ 'cfg017',
                'cfg018',
                'cfg019',
                'cfg020',
                'cfg021',
                'cfg022',
                'cfg023',
                'cfg024',
                'cfg025',
                'cfg026',
                'cfg027',
                'cfg028',
                'cfg029',
                'cfg030',
                'cfg031',
                'cfg032' ]
                
# standard masked
cfgLstStandardM = [ 'cfg033',
                    'cfg034',
                    'cfg035',
                    'cfg036',
                    'cfg037',
                    'cfg038',
                    'cfg039',
                    'cfg040',
                    'cfg041',
                    'cfg042',
                    'cfg043',
                    'cfg044',
                    'cfg045',
                    'cfg046',
                    'cfg047',
                    'cfg048' ]

cfgLstHM    = [ 'cfg049',
                'cfg050',
                'cfg051',
                'cfg052',
                'cfg053',
                'cfg054',
                'cfg055',
                'cfg056',
                'cfg057',
                'cfg058',
                'cfg059',
                'cfg060',
                'cfg061',
                'cfg062',
                'cfg063',
                'cfg064' ]

cfgLstPlanstrNFM = [ 'cfg065',
                     'cfg066',
                     'cfg067',
                     'cfg068',
                     'cfg069',
                     'cfg070',
                     'cfg071',
                     'cfg072',
                     'cfg073',
                     'cfg074',
                     'cfg075',
                     'cfg076',
                     'cfg077',
                     'cfg078',
                     'cfg079',
                     'cfg080' ]

notRegisteredTREs = [ 1.52 , 3.00, 1.94, 2.22 ]

p_f = cfgPlt.configurationPlotter( baseDirNetwork, cfgLstFastM,     2, afterInitialisationVals=notRegisteredTREs )
p_s = cfgPlt.configurationPlotter( baseDirNetwork, cfgLstStandardM, 2 )
p_h = cfgPlt.configurationPlotter( baseDirLocal,   cfgLstHM,        2 )

p_fNoMask = cfgPlt.configurationPlotter( baseDirNetwork, cfgLstFastNOM, 2 )

p_fPlanstrN = cfgPlt.configurationPlotter( baseDirLocal, cfgLstPlanstrNFM, 2, afterInitialisationVals=notRegisteredTREs )