#! /usr/bin/env python 
# -*- coding: utf-8 -*-



import meshFromSegmentation as mfs


meshDir          = 'W:/philipsBreastProneSupine/Meshes/meshMaterials6Supine/'
meshlabScript    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_smooth6.mlx'
breastTissueMask = 'W:/philipsBreastProneSupine/SegmentationSupine/segmOutChestPectMuscFatGland_voi.nii'

meshGenerator = mfs.meshFromSegmentation( breastTissueMask, meshDir )
meshGenerator.generateBreastMesh( meshlabScript )


