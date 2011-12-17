#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import polarisToolRecords as ptr

class polarisRecordingReader :
    
    def __init__( self, csvRecordingFilePath ) :
        
        self.csvRecordingFilePath = csvRecordingFilePath 
        self.header               = []
        self.rawList              = []
        
        # start reading the list straight away
        f = file( csvRecordingFilePath, 'r' )
        lines = f.readlines()
        
        for line in lines :
            
            # Extract the header information
            if line.startswith( 'Tools' ) :
                self.header = line.split( ',' )
                continue
            
            # save the rawfiles
            self.rawList.append( line.split(',') )
        
        # Get the number of tools at the beginning of the recording
        self.numTools = int(self.rawList[0][0])
        self.tools = []
    
        for i in range( self.numTools ) :
            tmpPortList  = []
            tmpFrameList = []
            tmpFaceList  = []
            tmpStateList = []
            tmpRxList    = []
            tmpRyList    = []
            tmpRzList    = []
            tmpTxList    = []
            tmpTyList    = []
            tmpTzList    = []
            tmpErrorList = []
            
            
            for j in range( len( self.rawList ) ) :
                tmpPortList.append ( int  ( self.rawList[ j ][ 11*i +  1 ] ) )
                tmpFrameList.append( int  ( self.rawList[ j ][ 11*i +  2 ] ) )
                tmpFaceList.append ( int  ( self.rawList[ j ][ 11*i +  3 ] ) )
                tmpStateList.append( str  ( self.rawList[ j ][ 11*i +  4 ] ) )
                tmpRxList.append   ( float( self.rawList[ j ][ 11*i +  5 ] ) )
                tmpRyList.append   ( float( self.rawList[ j ][ 11*i +  6 ] ) )
                tmpRzList.append   ( float( self.rawList[ j ][ 11*i +  7 ] ) )
                tmpTxList.append   ( float( self.rawList[ j ][ 11*i +  8 ] ) )
                tmpTyList.append   ( float( self.rawList[ j ][ 11*i +  9 ] ) )
                tmpTzList.append   ( float( self.rawList[ j ][ 11*i + 10 ] ) )
                tmpErrorList.append( float( self.rawList[ j ][ 11*i + 11 ] ) )
                
            self.tools.append( ptr.polarisToolRecords( self.header[ 11*i +  1 ], tmpPortList, tmpFrameList, tmpFaceList, tmpStateList, tmpRxList, tmpRyList, tmpRzList, tmpTxList, tmpTyList, tmpTzList, tmpErrorList ) )


        
        
        
