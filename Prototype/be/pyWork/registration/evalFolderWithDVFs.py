#! /usr/bin/env python 
# -*- coding: utf-8 -*-

"""
@author:  Bjoern Eiben

@summary: This script evaluates deformations vector fields in a given folder. These fields are assumed to originate from the 
          registration of Christine Tanner's simulations.
          
          The deformation vector fields are supposed to be named according to the following convention:
          [Source]_to_[Target]_[id]_Transform.nii
"""


import externalRegistrationTask     as extRegTask
import evaluationTask               as evTask
import fileCorrespondence           as fc
import registrationTaskListExecuter as executer
import evaluationListAnalyser       as evalListAnalyser
from os import path, makedirs

import sys, argparse, os




def main() :
    
    if len( sys.argv ) != 4:
        print( 'Usage: \n --> evalFolderWithDVFs.py dirToEvaluate dirWithReferenceDeforms dirWithMasks'  )
        return
    
    evalDir      = sys.argv[1]
    refDeformDir = sys.argv[2]
    maskDir      = sys.argv[3]

    if ( not os.path.exists( evalDir ) ) or ( not os.path.exists( refDeformDir )) or ( not os.path.exists( maskDir ) ):
        print( 'Error: One or more paths given do not exist...' )
        return
    
    # get the deformation vector fields from the DVF folder
    possibleDeformationVectorFields = fc.getFiles( evalDir, 'nii' )
    deformationVectorFields         = []
    
    
    # optionally check if there is an expected identifying part in the dvf file name...
    
    transformID = 'Transform'
    
    for pdvf in possibleDeformationVectorFields : 
        if pdvf.count( transformID ) == 1 :
            deformationVectorFields.append( pdvf )

    print 'Deformation Vector Fields:'

    
    # create a dummy registration task for each DVF and append to list
    regTaskList  = []
    
    for dvf in deformationVectorFields : 
        print dvf
        regTaskList.append( extRegTask.externalRegistrationTask( os.path.join( evalDir, dvf ) ) ) 
    
    
    evalFileName = evalDir + '/eval.txt'
    evalTaskList = []
    
    for regTask in regTaskList :
        # referenceDeformDirIn, maskDirIn, registrationTaskIn, evalFileIn = 'eval.txt'
        evalTaskList.append( evTask.evaluationTask( refDeformDir, maskDir, regTask, evalFileName ) )
    
    evalExecuter = executer.registrationTaskListExecuter( evalTaskList, 1 )
    evalExecuter.execute()
    
    analyser = evalListAnalyser.evaluationListAnalyser( evalTaskList )
    analyser.printInfo()
    analyser.writeSummaryIntoFile()
    
    print( 'Done.' )
    



if __name__ == "__main__" :
    main()
    
    
    
    
    