#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''

from glob import glob
import re, os


def getFiles( dirIn, extIn = 'nii' ) : 
    ''' Returns the nifti files in the given directory
        by default the 
    '''  
    ext = '*.' + extIn
    niiFileList = []
    
    for niiFile in glob( os.path.join( dirIn, ext ) ) :
        fileName = os.path.split(niiFile)[1]
        niiFileList.append( fileName )
        
    return niiFileList



    
def matchListWithPattern(list, pattern) : 
    selectedFiles = []
     
    for item in list :
        m = pattern.match( item ) 

        if  m :
            selectedFiles.append( item )

    return selectedFiles


    
def getPrecontrast( fileList ) :
    
    preconPattern = re.compile( '(S\d{1,2}i1)' )
    return matchListWithPattern( fileList, preconPattern )



def getPostcontrast( fileList ) :
    
    postconPattern = re.compile( '(S\d{1,2}i3\.)' )
    return matchListWithPattern( fileList, postconPattern )
    
    
def getDeformationFields( fileList ) :
    
    deformationPattern = re.compile( '\S*uxyz' )
    return matchListWithPattern( fileList, deformationPattern )


def limitToTrainData( fileList ) :

    testDataPat   = re.compile( '(S[2-5])|(S1i)' )
    return matchListWithPattern( fileList, testDataPat )
    


def limitToTestData( fileList ) :

    testDataPat   = re.compile( '(S[6-9])|(S10)' )
    return matchListWithPattern( fileList, testDataPat )
    

def matchTargetAndSource( targetFileName, sourceList, fileExt = 'nii' ) :
    patSubID = re.compile( 'S\d{1,2}' ) 
    
    subID  = patSubID.match( targetFileName ).group()
    subID += 'i1.' + fileExt
    i = sourceList.index( subID )
	
    print( "Matched: " )
    print( "  - Target: " + targetFileName )
    print( "  - Source: " + sourceList[i] )		
    return sourceList[i]
    
    
def matchTargetAndDeformationField( targetFileName, deformFileList ) :
    
    # strip everything from the 
    # special care needs to be taken due to gipl.Z files...
    target     = os.path.split( targetFileName )[1] 
    targetBase = os.path.splitext( target      )[0]
    targetBase = os.path.splitext( targetBase  )[0]
    
    if targetBase.endswith( 'full' ) :
        targetBase = targetBase.split('full')[0]  
    
    # Use the base of the target name as the pattern...
    pattern = re.compile('(' + targetBase + ')')
    matchedList = matchListWithPattern( deformFileList, pattern )
    if len(matchedList) != 1 :
        print( 'Warning: Ambiguity or no match')
        return None
    
    return matchedList[0]
    
    
def matchTargetAndTargetMask( targetFileName, maskList ) :
    
    target     = os.path.split( targetFileName )[1] 
    targetBase = os.path.splitext( target      )[0]
    targetBase = os.path.splitext( targetBase  )[0]
    
    # take care, that "full" targets are also handled...
    targetBase = targetBase.split('full')[0]
    
    # Use the base of the target name as the pattern...
    pattern = re.compile('(' + targetBase + 'mask)')
    matchedList = matchListWithPattern( maskList, pattern )
    
    if len(matchedList) != 1 :
        print( 'Warning: Ambiguity or no match')
        return None
    
    return matchedList[0]

def matchSourceAndSoruceMask( sourceFileName, maskList ) :
    source     = os.path.split( sourceFileName )[1] 
    #sourceBase = os.path.splitext( source     )[0]
    #sourceBase = os.path.splitext( sourceBase )[0]
    
    # Use the base of the target name as the pattern...
    #pattern = re.compile('(' + sourceBase + 'mask)')
    patternExtractPatNum = re.compile( '(S\d{1,2}i)' )
    patNum               = patternExtractPatNum.match( source ).group()[0:-1]
    patternPatNum        = re.compile( '(' + patNum + 'mask)' )
    
    matchedList = matchListWithPattern( maskList, patternPatNum )
    
    if len(matchedList) != 1 :
        print( 'Warning: Ambiguity or no match')
        return None
    
    return matchedList[0]
    pass
    

def getAnyMasks( fileList ) :
    maskPattern = re.compile('(\S*mask)')
    return matchListWithPattern( fileList, maskPattern )


def getAnyBreastMasks( fileList ) :
    breastMaskPattern = re.compile( '(\S*mask_breast)' )
    return matchListWithPattern( fileList, breastMaskPattern )


def getAnyLesiontMasks( fileList ) :
    breastMaskPattern = re.compile( '(\S*mask_lesion)' )
    return matchListWithPattern( fileList, breastMaskPattern )


def getSourceBreastMasks( fileList ) :
    sourceBreastMaskPattern = re.compile( '(S\d{1,2}mask_breast)' )
    return matchListWithPattern( fileList, sourceBreastMaskPattern )

def getSourceLesionMasks( fileList ) :
    sourceBreastMaskPattern = re.compile( '(S\d{1,2}mask_lesion)' )
    return matchListWithPattern( fileList, sourceBreastMaskPattern )


    
def getFullDeformed( fileList ) :
    
    deformPattern = re.compile( '(S\d{1,2}i3)((AP)|(OP)|(TP)|(PP)|(PR)|(PT))(\d{1,2})(full)(\.)' )
    return matchListWithPattern( fileList, deformPattern )


def getDeformed( fileList ):

    deformPattern = re.compile( '(S\d{1,2}i3)((AP)|(OP)|(TP)|(PP)|(PR)|(PT))(\d{1,2})(\.)' )
    return matchListWithPattern( fileList, deformPattern )


    
if __name__ == '__main__' : 

    import sys
    if len(sys.argv) != 2 : 
        print( 'Usage: %s INPUTDIR' % sys.argv[0] )
        sys.exit()
    
    
    
    dirIn      = sys.argv[1]
    niiFiles   = getFiles( dirIn )
    
    
    
    lTargets  = limitToTrainData( getDeformed ( niiFiles ) ) 
    lSources  = getPrecontrast( niiFiles )
    
    print( 'Given input directory:  %s ' % dirIn)
    print( 'num of target elements: %d ' % len( lTargets ) )
    print( 'num of source elements: %d ' % len( lSources ) )    
    
    for target in lTargets :
        matchTargetAndSource( target, lSources )
