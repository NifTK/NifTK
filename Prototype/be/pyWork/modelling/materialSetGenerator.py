#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import vtk
import vtk.util.numpy_support as VN

import pointWithinTetrahedron as inTet


class materialSetGenerator:
    ''' Aim is to get the numbers of the elements for the labels in the label image
    '''

    
    def __init__( self, nodeArray, elementArray, labelImageName, skinMaskImage, vtkVolMesh,
                  fatLabelVal=1, glandLabelVal=2, muscleLabelVal=3, minNumOfNodesForSurf=3, chestWallMaskImage=None ) :
        ''' @param skinMaskImage: Image with the skin mask, is tested for unequality to  
                                  zero at the element mid points as a second criterion to 
                                  surface contact only. 
        '''
        
        if isinstance( vtkVolMesh, str ) :
            #
            # Well, if a string is given, then it is helpful to read this file. 
            #
            ugr = vtk.vtkUnstructuredGridReader()
            ugr.SetFileName( vtkVolMesh )
            ugr.Update()
            
            vtkVolMesh = ugr.GetOutput()

        
        # Get the volume mesh
        surfaceExtractor = vtk.vtkDataSetSurfaceFilter()
        surfaceExtractor.SetInput( vtkVolMesh )
        surfaceExtractor.Update()
        self.surfNodes = VN.vtk_to_numpy( surfaceExtractor.GetOutput().GetPoints().GetData() )
        #self.surfEls   = VN.vtk_to_numpy( surfaceExtractor.GetOutput().GetPolys().GetData()  )
        #self.surfEls   = self.surfEls.reshape( self.surfEls.shape[0]/4, 4 )[:,1:]
        
        if labelImageName != None :
            self.labelImg    = nib.load( labelImageName )
        if skinMaskImage != None :
            self.skinMaskImg = nib.load( skinMaskImage  )
        if chestWallMaskImage != None :
            self.chestMaskImg = nib.load( chestWallMaskImage )
        
        # Transformation from real world coordinate to image index...
        rot90Z               = np.array(([-1,0,0,0],[0,-1,0,0], [0,0,1,0], [0,0,0,1]))           # for itk written images
        self.labelXToIMat    = np.linalg.inv( np.dot( self.labelImg.get_affine(),    rot90Z ) )
        self.skinMaskXToIMat = np.linalg.inv( np.dot( self.skinMaskImg.get_affine(), rot90Z ) )
        
        self.nodes    = nodeArray
        self.elements = elementArray
        
        # sometimes the first entry gives the number of elements
        if self.elements.shape[1] == 5 : 
            self.elements = self.elements[:,1:]

        
        # the nodes are assigned here
        self.muscleElements = []
        self.skinElements   = []
        self.skinNodes      = []
        self.fatElemetns    = []
        self.glandElements  = []
        
        self.muscleElementMidPoints = []
        self.skinElementMidPoints   = []
        self.fatElementMidPoints    = []
        self.glandElementMidPoints  = []
        
        self.labelMuscle = muscleLabelVal
        self.labelFat    = fatLabelVal
        self.labelGland  = glandLabelVal
        
        self.numTetNodesForSurf = minNumOfNodesForSurf
        self._assignMaterials()
        
        pass
    
    
    
    
    def _assignMaterials( self ) :
        
        #
        # Here is the plan
        #
        # 1) Iterate through the elements
        #  a) Check if element is skin surface (three points are on surface, )
        #    - if so, assign to skin element list and go to next element
        #    - otherwise find the correct element...
        #  a) Get bounding box (BB) of element 
        #  b) Iterate through BB and check for each pixel if inside the current element
        #    - if inside tetrahedron: remember the element internally
        #  c) assign most common   
        #
        
        print('Starting to assign materials...')
    
        nNdsSkinFound    = 0
        labelImgData = self.labelImg.get_data()
        skinImgData  = self.skinMaskImg.get_data() 
        
        for i in range( self.elements.shape[0] ) :
            
            #
            # Print progress
            #
            if ( (i+1) % (self.elements.shape[0] / 10 ) ) == 0 :
                print('... %3i percent done' % ( 100*i/self.elements.shape[0] + 1) )
            
            #
            # Get the coordinates
            #
            (A,B,C,D) = self.elements[i]
            
            cdsA = self.nodes[A]
            cdsB = self.nodes[B]
            cdsC = self.nodes[C]
            cdsD = self.nodes[D]
            
            cdsAR = np.tile(cdsA,(self.surfNodes.shape[0],1))
            cdsBR = np.tile(cdsB,(self.surfNodes.shape[0],1))
            cdsCR = np.tile(cdsC,(self.surfNodes.shape[0],1))
            cdsDR = np.tile(cdsD,(self.surfNodes.shape[0],1))
            
            skinCDS = []
            
            # All three coordinates must be 
            found = []
            if np.max( np.min(cdsAR==self.surfNodes,1) ):
                found.append( 0 ) 
                skinCDS.append(cdsA)

            if np.max( np.min(cdsBR==self.surfNodes,1) ):
                found.append( 1 )
                skinCDS.append(cdsB)
                
            if np.max( np.min(cdsCR==self.surfNodes,1) ):
                found.append( 2 )
                skinCDS.append(cdsC)
            
            if np.max( np.min(cdsDR==self.surfNodes,1) ):
                found.append( 3 )
                skinCDS.append(cdsD)
            
            if len( found ) >= self.numTetNodesForSurf:
                
                # Check if the coordinate is labelled in the skin image
                #c = [cdsA, cdsB, cdsC, cdsD ]
                #c = c[ min( found ) ]
                # average over those coordinates which were found to be on the surfce 
                skinCDS = np.array( skinCDS )
                skinCDS = np.mean( skinCDS, 0 )
                                
                #idxS = np.array( (np.around( np.dot( self.skinMaskXToIMat, np.hstack( ( c, 1 ) ) ) ) ), dtype=np.int ) 
                idxS = np.array( (np.around( np.dot( self.skinMaskXToIMat, np.hstack( ( skinCDS, 1 ) ) ) ) ), dtype=np.int )
                
                if skinImgData[ idxS[0], idxS[1], idxS[2] ] != 0 :
                    nNdsSkinFound = nNdsSkinFound + 1 
                    self.skinElements.append( i )
                    self.skinElementMidPoints.append( (cdsA + cdsB + cdsC + cdsD) / 4. )
                    
                    if found.count( 0 ) == 1 : self.skinNodes.append ( A )
                    if found.count( 1 ) == 1 : self.skinNodes.append ( B )
                    if found.count( 2 ) == 1 : self.skinNodes.append ( C )
                    if found.count( 3 ) == 1 : self.skinNodes.append ( D )
                    
                    continue
            
            # First attempt: Take a single sample in the label image
            cdsTetCentre = (cdsA + cdsB + cdsC + cdsD) / 4.
            idxCentre    = np.array( (np.around( np.dot( self.labelXToIMat, np.hstack( ( cdsTetCentre, 1 ) ) ) ) ), dtype=np.int ) 
            
            curLabel = labelImgData[idxCentre[0], idxCentre[1], idxCentre[2]]
            
            if curLabel == self.labelFat :
                self.fatElemetns.append( i )
                self.fatElementMidPoints.append( cdsTetCentre )
                continue
                
            if curLabel == self.labelGland :
                self.glandElements.append( i )
                self.glandElementMidPoints.append( cdsTetCentre )
                continue
                
            if curLabel == self.labelMuscle :
                self.muscleElements.append( i )
                self.muscleElementMidPoints.append( cdsTetCentre )
                continue
            
            # default case if the mask is not accurate...
            self.fatElemetns.append( i )
            self.fatElementMidPoints.append( cdsTetCentre )
             
        
        #
        # Convert the arrays to numpy and print some information 
        #
        self.skinElements   = np.array( self.skinElements ) 
        self.fatElemetns    = np.array( self.fatElemetns  ) 
        self.glandElements  = np.array( self.glandElements) 
        self.muscleElements = np.array( self.muscleElements  ) 
        
        self.skinElementMidPoints   = np.array( self.skinElementMidPoints )
        self.fatElementMidPoints    = np.array( self.fatElementMidPoints )
        self.glandElementMidPoints  = np.array( self.glandElementMidPoints )
        self.muscleElementMidPoints = np.array( self.muscleElementMidPoints )
        
        self.skinNodes = np.array( self.skinNodes )
        self.skinNodes = np.unique( self.skinNodes )
        
        print( 'Found the following number of elements: ')
        print( ' - Skin:    %8i' % nNdsSkinFound )       
        print( ' - Fat:     %8i' % self.fatElemetns.shape[0] )       
        print( ' - Muscle:  %8i' % self.muscleElements.shape[0] )       
        print( ' - Gland:   %8i' % self.glandElements.shape[0] )       
        
        