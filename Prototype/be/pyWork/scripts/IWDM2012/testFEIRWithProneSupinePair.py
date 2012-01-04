#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import feirRegistrationTask as feirTask
import aladinRegistrationTask as aladinTask
import f3dRegistrationTask as f3dTask
import commandExecution as cmdEx
import nibabel as nib

# directories
meshDir            = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/'
regDirFEIR         = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/regFEIR/'
regDirAladin       = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/regAladin/'
regDirF3D          = 'W:/philipsBreastProneSupine/Meshes/meshMaterials3/regF3D/'

# xml model and generated output
xmlModel           = meshDir + 'model.xml'
strSimulatedSupine = meshDir + 'out.nii'

# original images
strProneImg        = 'W:/philipsBreastProneSupine/proneCrop2Pad.nii'
strSupineImg       = 'W:/philipsBreastProneSupine/rigidAlignment/supine1kTransformCrop2Pad_zeroOrig.nii'

# run the simulation and resampling at the same time
simCommand = 'ucltkDeformImageFromNiftySimulation'
simParams   = ' -i '    + strProneImg
simParams  += ' -x '    + xmlModel 
simParams  += ' -o '    + strSimulatedSupine
simParams  += ' -mval 0 ' 
simParams  += ' -interpolate bspl '

# run the simulation
print('Starting niftySim-Ulation')
cmdEx.runCommand( simCommand, simParams )


f3dReg = f3dTask.f3dRegistrationTask( strSimulatedSupine, strSupineImg, strSimulatedSupine, regDirF3D, 'NA', 
                                      bendingEnergy=0.001, logOfJacobian=0.025, finalGridSpacing=10, numberOfLevels=5, maxIterations=300, gpu=True)

f3dReg.run()
f3dReg.constructNiiDeformationFile()


dispImg = nib.load( f3dReg.dispFieldITK )


###############################
#print('Starting aladin-registration')
#alRegTask = aladinTask.aladinRegistrationTask( strSimulatedSupine, strSupineImg, strSimulatedSupine, regDirAladin, 'aladin', 
#                                               rigOnly=True, levels=4 )
#
#alRegTask.run()


############################################
# run the registration...
#print('Starting FEIR-registration')
#regTask = feirTask.feirRegistrationTask( strSimulatedSupine, strSupineImg, regDirFEIR, 
#                                         'mu-8-n', mode='fast', mask=True, mu=2**(-4)*0.0025, lm=0.)
#regTask.run()
#regTask.constructNiiDeformationFile()
#print('Resampling the image...')
#regTask.resampleSourceImage()
#print('Done.')

