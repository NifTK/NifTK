#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import os


class xmlModelParameterSweeper :
    
    def __init__( self, referenceModelFileName, parameterStringToChange, parameterArray, idArray, outDir ):
        
        
        xmlF = open( referenceModelFileName )
        d = xmlF.read()
        
        parToChangeOpen  ='<'  + parameterStringToChange + '>'
        parToChangeClose ='</' + parameterStringToChange + '>'
        
        pBegin = d.find( parToChangeOpen  )    
        pEnd   = d.find( parToChangeClose )   
        
        outFileNames = [] 
        
        for i in range( len( parameterArray ) ) : 
            p    = parameterArray[i]
            id   = idArray[i]
            dNew = d[ 0 : pBegin + len( parToChangeOpen ) ] + str( p ) +  d[ pEnd : len( d ) ] 
            
            outFileNames.append( outDir + os.path.splitext( os.path.split( referenceModelFileName )[1] )[0] + id + '.xml' )
            out = open(outFileNames[-1], 'w')
            out.write( dNew )
            out.close()


