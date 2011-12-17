#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import configurationPlotter as cfgPlt
from pylab import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show, xlim, ylim
import matplotlib.pyplot as plt

# Aladin folders...
baseDir = 'c:/data/regValidationWithTannerData/outAladin/'

# Rigid, percent inlier varied (similarity)
cfgLst1 = ['cfg159',
           'cfg160',
           'cfg161',
           'cfg162',
           'cfg163',
           'cfg164',
           'cfg165',
           'cfg166',
           'cfg167',
           'cfg168',
           'cfg169',
           'cfg170',
           'cfg171',
           'cfg172',
           'cfg173',
           'cfg174',
           'cfg175',
           'cfg176',
           'cfg177',
           'cfg178' ]

# Affine, percent inlier varied (similarity)
cfgLst2 = ['cfg179',
           'cfg180',
           'cfg181',
           'cfg182',
           'cfg183',
           'cfg184',
           'cfg185',
           'cfg186',
           'cfg187',
           'cfg188',
           'cfg189',
           'cfg190',
           'cfg191',
           'cfg192',
           'cfg193',
           'cfg194',
           'cfg195',
           'cfg196',
           'cfg197',
           'cfg198' ]

# Rigid, percent block varied (variance)
cfgLst3 = ['cfg199',
           'cfg200',
           'cfg201',
           'cfg202',
           'cfg203',
           'cfg204',
           'cfg205',
           'cfg206',
           'cfg207',
           'cfg208',
           'cfg209',
           'cfg210',
           'cfg211',
           'cfg212',
           'cfg213',
           'cfg214',
           'cfg215',
           'cfg216',
           'cfg217',
           'cfg218' ]

# Affine, percent block varied (variance)
cfgLst4 = ['cfg219',
           'cfg220',
           'cfg221',
           'cfg222',
           'cfg223',
           'cfg224',
           'cfg225',
           'cfg226',
           'cfg227',
           'cfg228',
           'cfg229',
           'cfg230',
           'cfg231',
           'cfg232',
           'cfg233',
           'cfg234',
           'cfg235',
           'cfg236',
           'cfg237',
           'cfg238' ]

# Wrong results...
# Rigid, percent inlier varied (similarity)
cfgLst5 = ['cfg001',
           'cfg002',
           'cfg003',
           'cfg004',
           'cfg005',
           'cfg006',
           'cfg007',
           'cfg008',
           'cfg009',
           'cfg010',
           'cfg012',# 10 equal to 11
           'cfg013',
           'cfg014',
           'cfg015',
           'cfg016' ]

# Rigid, percent inlier varied (similarity)
cfgLst6 = ['cfg027',
           'cfg028',
           'cfg029',
           'cfg030',
           'cfg031',
           'cfg032',
           'cfg033',
           'cfg034',
           'cfg035',
           'cfg036' ]

#
# Experiments with number of iterations...
#
cfgLst_itR = [ 'cfg239', 'cfg240', 'cfg241', 'cfg242' ]
cfgLst_itA = [ 'cfg243', 'cfg244', 'cfg245', 'cfg246' ]



#
# Reg_f3d cfg folders...
#
baseDirF3D = 'C:/data/regValidationWithTannerData/outF3d/'

cfgLst_s60 = ['cfg133',
              'cfg134',
              'cfg135',
              'cfg136',
              'cfg137',
              'cfg138',
              'cfg139',
              'cfg140', #
              'cfg141', # 0.8
              'cfg142', # 0.9
              'cfg343', # 0.91
              'cfg265', # 0.92
              'cfg344', # 0.93
              'cfg266', # 0.94
              'cfg143', # 0.95
              'cfg267', # 0.96
              'cfg345', # 0.97
              'cfg268', # 0.98
              'cfg346', # 0.99
              'cfg347' ]# 0.995

cfgLst_s40 = ['cfg144', #0
              'cfg145',
              'cfg146',
              'cfg147',
              'cfg148',
              'cfg149',
              'cfg150',
              'cfg151',
              'cfg152',
              'cfg153',  # 0.9
              'cfg348',  # 0.91
              'cfg269',  # 0.92
              'cfg349',  # 0.93
              'cfg270',  # 0.94
              'cfg154',  # 0.95
              'cfg271',  # 0.96
              'cfg350',  # 0.97
              'cfg272',  # 0.98
              'cfg351',  # 0.99
              'cfg352' ] # 0.995

cfgLst_s20 = ['cfg155',
              'cfg156',
              'cfg157',
              'cfg158',
              'cfg159',
              'cfg160',
              'cfg161',
              'cfg162',
              'cfg163',
              'cfg164',  # 0.9
              'cfg353',  # 0.91
              'cfg273',  # 0.92
              'cfg354',  # 0.93
              'cfg274',  # 0.94
              'cfg165',  # 0.95
              'cfg275',  # 0.96
              'cfg355',  # 0.97
              'cfg276',  # 0.98
              'cfg356',  # 0.99
              'cfg357' ] # 0.995

cfgLst_s10 = ['cfg166',
              'cfg167',
              'cfg168',
              'cfg169',
              'cfg170',
              'cfg171',
              'cfg172',
              'cfg173',
              'cfg174',
              'cfg175',  # 0.9
              'cfg358',  # 0.91 
              'cfg277',  # 0.92
              'cfg359',  # 0.93  
              'cfg278',  # 0.94
              'cfg176',  # 0.95
              'cfg279',  # 0.96
              'cfg360',  # 0.97
              'cfg280',  # 0.98
              'cfg361',  # 0.99
              'cfg362' ] # 0.995

cfgLst_m10 = ['cfg177',
              'cfg178',
              'cfg179',
              'cfg180',
              'cfg181',
              'cfg182',
              'cfg183',
              'cfg184',
              'cfg185',
              'cfg186',  # 0.9
              'cfg363',  # 0.91
              'cfg281',  # 0.92 
              'cfg364',  # 0.93
              'cfg282',  # 0.94
              'cfg187',  # 0.95
              'cfg283',  # 0.96
              'cfg365',  # 0.97
              'cfg284',  # 0.98
              'cfg366',  # 0.99
              'cfg367' ] # 0.995  

cfgLst_m05 = ['cfg188',
              'cfg189',
              'cfg190',
              'cfg191',
              'cfg192',
              'cfg193',
              'cfg194',
              'cfg195',
              'cfg196',
              'cfg197',  # 0.9
              'cfg368',  # 0.91
              'cfg285',  # 0.92
              'cfg369',  # 0.93
              'cfg286',  # 0.94
              'cfg198',  # 0.95
              'cfg287',  # 0.96
              'cfg370',  # 0.97
              'cfg288',  # 0.98
              'cfg371',  # 0.99
              'cfg372' ] # 0.995
              

# RREG initialisations
cfgLst_s60R = ['cfg199',
               'cfg200',
               'cfg201',
               'cfg202',
               'cfg203',
               'cfg204',
               'cfg205',
               'cfg206',
               'cfg207',
               'cfg208',  # 0.9
               'cfg313',  # 0.91
               'cfg289',  # 0.92
               'cfg314',  # 0.93
               'cfg290',  # 0.94
               'cfg209',  # 0.95
               'cfg291',  # 0.96
               'cfg315',  # 0.97
               'cfg292',  # 0.98
               'cfg316',  # 0.99
               'cfg317' ] # 0.995

cfgLst_s40R = ['cfg210',
               'cfg211',
               'cfg212',
               'cfg213',
               'cfg214',
               'cfg215',
               'cfg216',
               'cfg217',
               'cfg218',
               'cfg219',  # 0.9
               'cfg318',  # 0.91
               'cfg293',  # 0.92
               'cfg319',  # 0.93
               'cfg294',  # 0.94
               'cfg220',  # 0.95
               'cfg295',  # 0.96
               'cfg320',  # 0.97
               'cfg296',  # 0.98
               'cfg321',  # 0.99
               'cfg322']  # 0.995

cfgLst_s20R = ['cfg221',
               'cfg222',
               'cfg223',
               'cfg224',
               'cfg225',
               'cfg226',
               'cfg227',
               'cfg228',
               'cfg229',
               'cfg230',  # 0.90
               'cfg323',  # 0.91
               'cfg297',  # 0.92
               'cfg324',  # 0.93
               'cfg298',  # 0.94
               'cfg231',  # 0.95
               'cfg299',  # 0.96
               'cfg325',  # 0.97
               'cfg300',  # 0.98
               'cfg326',  # 0.99
               'cfg327' ] # 0.995

cfgLst_s10R = ['cfg232',
               'cfg233',
               'cfg234',
               'cfg235',
               'cfg236',
               'cfg237',
               'cfg238',
               'cfg239',
               'cfg240',
               'cfg241',  # 0.9
               'cfg328',  # 0.91
               'cfg301',  # 0.92
               'cfg329',  # 0.93
               'cfg302',  # 0.94
               'cfg242',  # 0.95
               'cfg303',  # 0.96
               'cfg330',  # 0.97
               'cfg304',  # 0.98
               'cfg331',  # 0.99
               'cfg332' ] # 0.995
               

cfgLst_m10R = ['cfg243',
               'cfg244',
               'cfg245',
               'cfg246',
               'cfg247',
               'cfg248',
               'cfg249',
               'cfg250',
               'cfg251',
               'cfg252',  # 0.9
               'cfg333',  # 0.91
               'cfg305',  # 0.92
               'cfg334',  # 0.93
               'cfg306',  # 0.94
               'cfg253',  # 0.95
               'cfg307',  # 0.96
               'cfg335',  # 0.97
               'cfg308',  # 0.98
               'cfg336',  # 0.99
               'cfg337' ] # 0.995               

cfgLst_m05R = ['cfg254',
               'cfg255',
               'cfg256',
               'cfg257',
               'cfg258',
               'cfg259',
               'cfg260',
               'cfg261',
               'cfg262',
               'cfg263',  # 0.9
               'cfg338',  # 0.91
               'cfg309',  # 0.92
               'cfg339',  # 0.93
               'cfg310',  # 0.94
               'cfg264',  # 0.95
               'cfg311',  # 0.96
               'cfg340',  # 0.97
               'cfg312',  # 0.98
               'cfg341',  # 0.99
               'cfg342' ] # 0.995


# Initial values from aladin
aladinTREs = [ 0.84507397847833, # Bresat mean 
               1.83484476839333, # Breast percentile
               0.759312332675,   # Lesion mean
               0.911904378635 ]  # Lesion percentile

# Initial values from rreg
rregTREs = [ 0.76939799637000, # Bresat mean 
             1.73393667509500, # Breast percentile
             0.50661508875167,   # Lesion mean
             0.67222505121333 ]  # Lesion percentile

aregTREs = [ 0.554295884,
             1.377623316,
             0.438465927,
             0.591725209 ]

# Aladin plotters
p1_RI    = cfgPlt.configurationPlotter( baseDir, cfgLst1, 6 )
p2_AI    = cfgPlt.configurationPlotter( baseDir, cfgLst2, 6 )
p3_RV    = cfgPlt.configurationPlotter( baseDir, cfgLst3, 5 )
p4_AV    = cfgPlt.configurationPlotter( baseDir, cfgLst4, 5 )

# - masking
networkBaseDir = 'X:/NiftyRegValidationWithTannerData/outAladin'
p5_RI_noMask = cfgPlt.configurationPlotter( networkBaseDir, cfgLst5, 6 )
p6_AI_noMask = cfgPlt.configurationPlotter( networkBaseDir, cfgLst6, 6 )

# -iterations
p7_itR = cfgPlt.configurationPlotter( baseDir, cfgLst_itR, 3 )
p8_itA = cfgPlt.configurationPlotter( baseDir, cfgLst_itA, 3 )

# f3d plotters
p_s60 = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s60, 3, afterInitialisationVals = aladinTREs )
p_s40 = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s40, 3, afterInitialisationVals = aladinTREs )
p_s20 = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s20, 3, afterInitialisationVals = aladinTREs )
p_s10 = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s10, 3, afterInitialisationVals = aladinTREs )
p_m10 = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_m10, 3, afterInitialisationVals = aladinTREs )
p_m05 = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_m05, 3, afterInitialisationVals = aladinTREs )

p_s60R = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s60R, 3, afterInitialisationVals = rregTREs )
p_s40R = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s40R, 3, afterInitialisationVals = rregTREs )
p_s20R = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s20R, 3, afterInitialisationVals = rregTREs )
p_s10R = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s10R, 3, afterInitialisationVals = rregTREs )
p_m10R = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_m10R, 3, afterInitialisationVals = rregTREs )
p_m05R = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_m05R, 3, afterInitialisationVals = rregTREs )


# Additional plot with wrong masking
cfgLst_s40R_NoMask = ['cfg078',
                      'cfg079',
                      'cfg080',
                      'cfg081',
                      'cfg082',
                      'cfg083',
                      'cfg084',
                      'cfg085',
                      'cfg086',
                      'cfg087',
                      'cfg088' ]

p_s40R_noMask = cfgPlt.configurationPlotter( baseDirF3D, cfgLst_s40R_NoMask, 3, afterInitialisationVals = rregTREs )

