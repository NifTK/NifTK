#! /usr/bin/env python

import bz2, os, re

def bZipFile( inFileName, outFileName = '', mode = 'b' ) :

    inFile      = file( inFileName, 'r' + mode )
    datIn       = inFile.read()
    
    if len( outFileName ) == 0 :
        outFileName = inFileName + '.bz2'
    elif not outFileName.endswith( '.bz2' ) :
        outFileName = outFileName + '.bz2'
        
    bzOutFile   = bz2.BZ2File( outFileName, 'w' + mode )
    
    bzOutFile.write( datIn )
    bzOutFile.close()
    
    return outFileName
    
    
def bUnzipFile( inFileName, outFileName = '' ) :

    bz2File = bz2.BZ2File( inFileName )
    mod     = bz2File.mode
    uncompressedData = bz2File.read()
    
    pat    = re.compile('(r)')
    outMod = re.sub( pat, 'w', mod )
    
    if len (outFileName) == 0 : 
        (outFileName, ext) = os.path.splitext( inFileName )
    
    fileOut = file( outFileName, outMod )
    fileOut.write( uncompressedData )
    fileOut.close()
    
    return outFileName
    
    
    
    
if __name__ == '__main__' : 
    
    import sys
    
    if len( sys.argv ) != 2 : 
        print( 'Usage: %s INPUTDIR' % sys.argv[0] )
        sys.exit()
    
    fileIn  = sys.argv[ 1 ]
    
    
    