#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

import subprocess, shlex, os, platform



def mean( numberList ):
    floatNums = [float(x) for x in numberList]
    return sum(floatNums) / len(numberList)



class evaluationListAnalyser :
    ''' This class implements an interface to access the data which was created by the evaluation 
        task list. 
        
        In the end it could be used to feed these information into a database.  
    
    '''
    def __init__( self, evaluationTaskListIn ) :
        
        self.evalTaskList = evaluationTaskListIn
        self.evalFileBreast = ''
        self.evalFileLesion = ''
        
        self._checkEvalTaskList()

        # TRE values from the file...
        self.breastMinInitialTREs   = []
        self.breastMeanInitialTREs  = []
        self.breastMaxInitialTREs   = []
        self.breastStdInitialTREs   = []
        self.breastPercInitialTREs  = []
        
        self.breastMinTREs   = []
        self.breastMeanTREs  = []
        self.breastMaxTREs   = []
        self.breastStdTREs   = []
        self.breastPercTREs  = []
        self.breastNthPercs  = []
        
        self.lesionMinInitialTREs   = []
        self.lesionMeanInitialTREs  = []
        self.lesionMaxInitialTREs   = []
        self.lesionStdInitialTREs   = []
        self.lesionPercInitialTREs  = []
        
        self.lesionMinTREs   = []
        self.lesionMeanTREs  = []
        self.lesionMaxTREs   = []
        self.lesionStdTREs   = []
        self.lesionPercTREs  = []
        self.lesionNthPercs  = []
        
        self.regDurations    = []
        self.meanRegDuration = 0
        self._calcMeanRegDuration()
        
        self._readFromFile( True  )
        self._readFromFile( False )


        # calculate the TRE quantities, which Christine used in her evaluation
        # mean error
        self.breastMeanOfMeanTREs = mean( self.breastMeanTREs )
        self.lesionMeanOfMeanTREs = mean( self.lesionMeanTREs )
        
        # mean initial error
        self.breastMeanOfMeanInitialTREs = mean( self.breastMeanInitialTREs )
        self.lesionMeanOfMeanInitialTREs = mean( self.lesionMeanInitialTREs )
        
        # mean error percentile
        self.breastMeanOfPercentileTREs = mean( self.breastPercTREs )
        self.lesionMeanOfPercentileTREs = mean( self.lesionPercTREs )        
        
        # mean initial percentile
        self.breastMeanOfPercentileInitialTREs = mean( self.breastPercInitialTREs )
        self.lesionMeanOfPercentileInitialTREs = mean( self.lesionPercInitialTREs )
        
        
        
        
    def printInfo( self ):
        print( 'Evaluation files: '                       )
        print( ' - breast: ' + self.evalFileBreast        )
        print( ' - lesion: ' + self.evalFileLesion + '\n' )
        
        print( '       | meanTRE          | meanTRE(initial) | improvement (initialTRE - TRE)' )
        print( 'breast | %16.14f | %16.14f | %16.14f ' 
               % (self.breastMeanOfMeanTREs, self.breastMeanOfMeanInitialTREs, (self.breastMeanOfMeanInitialTREs - self.breastMeanOfMeanTREs) ) )
        print( 'lesion | %16.14f | %16.14f | %16.14f ' 
               % (self.lesionMeanOfMeanTREs, self.lesionMeanOfMeanInitialTREs, (self.lesionMeanOfMeanInitialTREs - self.lesionMeanOfMeanTREs) ) )
        print( '\n' )
        print( 'Used %.1f-th percentile' % self.breastNthPercs[0] )
        print( '       | meanPerc         | meanPerc(ini)    | improvement (initial - error)' )
        print( 'breast | %16.14f | %16.14f | %16.14f ' 
               % (self.breastMeanOfPercentileTREs, self.breastMeanOfPercentileInitialTREs, (self.breastMeanOfPercentileInitialTREs - self.breastMeanOfPercentileTREs ) ) )
        print( 'lesion | %16.14f | %16.14f | %16.14f ' 
               % (self.lesionMeanOfPercentileTREs, self.lesionMeanOfPercentileInitialTREs, (self.lesionMeanOfPercentileInitialTREs - self.lesionMeanOfPercentileTREs ) ) )
        print( '\n\nMean registration duration: %.3fs' % self.meanRegDuration )
        
        
        
    def _checkEvalTaskList( self ) :
        ''' Purpose check that the registration tasks have all the same output file. 
        '''

        refEvalFileBreast = self.evalTaskList[0].evalFileBreast
        refEvalFileLesion = self.evalTaskList[0].evalFileLesion 
        
         
        for task in self.evalTaskList :
            if (refEvalFileBreast != task.evalFileBreast) or (refEvalFileLesion != task.evalFileLesion) :
                print( 'ERROR: Different evaluation files in the task list!' )
                return
            
        self.evalFileBreast = refEvalFileBreast
        self.evalFileLesion = refEvalFileLesion
            
            
                 
                 
    def _calcMeanRegDuration( self ) :
        ''' Iterate through the evaluation tasks, get the associated registration tasks and their 
            registraion duration.
        '''
        
        for evTask in self.evalTaskList :
            self.regDurations.append( evTask.registrationTask.regDuration )
            
        self.meanRegDuration = mean( self.regDurations )
            
            
                 
    def _readFromFile( self, breast = True ) :
        
        if breast :
            f = file( self.evalFileBreast, 'r' )
        else :
            f = file( self.evalFileLesion, 'r' )
            
        lines = f.readlines()
        
        for line in lines :
            if line.startswith('ini') or line.startswith( '---' ) :
                continue
            
            cols = line.split()
            
            if len( cols ) < 13 :
                print( 'Error: Not enough columns in file!' )
                return 
            
            if breast :
                self.breastMinInitialTREs.append ( float( cols[0] ) )   
                self.breastMeanInitialTREs.append( float( cols[1] ) )
                self.breastMaxInitialTREs.append ( float( cols[2] ) )
                self.breastStdInitialTREs.append ( float( cols[3] ) )
                self.breastPercInitialTREs.append( float( cols[4] ) )
                
                self.breastMinTREs.append ( float( cols[ 5] ) )
                self.breastMeanTREs.append( float( cols[ 6] ) )
                self.breastMaxTREs.append ( float( cols[ 7] ) )
                self.breastStdTREs.append ( float( cols[ 8] ) )
                self.breastPercTREs.append( float( cols[ 9] ) )
                self.breastNthPercs.append( float( cols[10] ) )
            
            else :
                self.lesionMinInitialTREs.append ( float( cols[0] ) )  
                self.lesionMeanInitialTREs.append( float( cols[1] ) )
                self.lesionMaxInitialTREs.append ( float( cols[2] ) )
                self.lesionStdInitialTREs.append ( float( cols[3] ) )
                self.lesionPercInitialTREs.append( float( cols[4] ) )
                
                self.lesionMinTREs.append ( float( cols[ 5] ) )
                self.lesionMeanTREs.append( float( cols[ 6] ) )
                self.lesionMaxTREs.append ( float( cols[ 7] ) )
                self.lesionStdTREs.append ( float( cols[ 8] ) )
                self.lesionPercTREs.append( float( cols[ 9] ) )
                self.lesionNthPercs.append( float( cols[10] ) )          
            
        f.close()
        
            
    def writeSummaryIntoFile( self ) :
        ''' Write the results into the evaluation files (breast and lesion results separately)
        '''
        
        # Breast file
        breastFile = file( self.evalFileBreast, 'a+' )
        
        breastFile.write( '\n'                )
        breastFile.write( 'Breast Summary\n==============\n'       )
        
        breastFile.write( 'Number of entries:\n  %3d \n\n' % len( self.breastMeanTREs ) )
        
        breastFile.write( 'Initial\n-------\n\n' )
        breastFile.write( 'Mean of mean TREs:       \n  %16.14f \n'     %( self.breastMeanOfMeanInitialTREs       ) )
        breastFile.write( 'Mean of percentile TREs: \n  %16.14f \n\n'   %( self.breastMeanOfPercentileInitialTREs ) )
        
        breastFile.write( 'Result\n------\n\n' )
        breastFile.write( 'Mean of mean TREs:       \n  %16.14f \n'     %( self.breastMeanOfMeanTREs       ) )
        breastFile.write( 'Mean of percentile TREs: \n  %16.14f \n\n'   %( self.breastMeanOfPercentileTREs ) )
        
        breastFile.write( 'Improvement\n-----------\n\n' )
        breastFile.write( 'Difference in mean of mean TREs:\n  %16.14f \n'          %( self.breastMeanOfMeanInitialTREs       - self.breastMeanOfMeanTREs       ) )
        breastFile.write( 'Difference in mean of percentile TREs: \n  %16.14f \n\n' %( self.breastMeanOfPercentileInitialTREs - self.breastMeanOfPercentileTREs ) )
        
        if self.evalTaskList[0].registrationTask.regCommand != 'extern' :
            breastFile.write( '\n\nMean registration duration: %.3fs' % self.meanRegDuration )
        
        breastFile.flush()
        breastFile.close()

        # Lesion file
        lesionFile = file( self.evalFileLesion, 'a+' )
        lesionFile.write( '\n'                       )
        lesionFile.write( 'Lesion Summary\n==============\n'       )
        
        lesionFile.write( 'Number of entries:\n  %3d \n\n' % len( self.lesionMeanTREs ) )
        
        lesionFile.write( 'Initial\n=======\n\n'       )
        lesionFile.write( 'Mean of mean TREs:       \n  %16.14f \n'     %( self.lesionMeanOfMeanInitialTREs       ) )
        lesionFile.write( 'Mean of percentile TREs: \n  %16.14f \n\n'   %( self.lesionMeanOfPercentileInitialTREs ) )
        
        lesionFile.write( 'Result\n======\n\n'       )
        lesionFile.write( 'Mean of mean TREs:       \n  %16.14f \n'     %( self.lesionMeanOfMeanTREs       ) )
        lesionFile.write( 'Mean of percentile TREs: \n  %16.14f \n\n'   %( self.lesionMeanOfPercentileTREs ) )
        
        lesionFile.write( 'Improvement\n===========\n\n' )
        lesionFile.write( 'Difference in mean of mean TREs:       \n  %16.14f \n'   %( self.lesionMeanOfMeanInitialTREs       - self.lesionMeanOfMeanTREs       ) )
        lesionFile.write( 'Difference in mean of percentile TREs: \n  %16.14f \n\n' %( self.lesionMeanOfPercentileInitialTREs - self.lesionMeanOfPercentileTREs ) )
        
        if self.evalTaskList[0].registrationTask.regCommand != 'extern' :
            lesionFile.write( '\n\nMean registration duration: %.3fs' % self.meanRegDuration )
        
        
        lesionFile.flush()
        lesionFile.close()
        
        
        
        
        
        
        
        
        
        