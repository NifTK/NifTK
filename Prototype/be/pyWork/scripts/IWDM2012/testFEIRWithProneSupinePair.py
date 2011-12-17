#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import feirRegistrationTask as feirTask

strProneImg  = 'W:/philipsBreastProneSupine/proneCrop2Pad.nii'
strSupineImg = 'W:/philipsBreastProneSupine/supine1kTransformCrop2Pad_ChestOnly.nii'
regDir       = 'W:/philipsBreastProneSupine/feirTest/'

strProneImg  = 'W:/philipsBreastProneSupine/proneCrop2Pad.nii'
strSupineImg = 'W:/philipsBreastProneSupine/supine1kTransformCrop2Pad_ChestOnly.nii'
regDir       = 'W:/philipsBreastProneSupine/feirTest/'

regTask = feirTask.feirRegistrationTask(strSupineImg, strProneImg, regDir, 'noMask', mode='fast', mask=True, planStr='trn' )
regTask.run()
regTask.constructNiiDeformationFile()
regTask.resampleSourceImage()

