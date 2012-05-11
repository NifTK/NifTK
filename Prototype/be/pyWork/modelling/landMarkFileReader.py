#! /usr/bin/env python 
# -*- coding: utf-8 -*-


from xml.dom.minidom import parseString
import numpy as np

class landMarkFileReader:
    
    def __init__( self, targetPointFile, sourcePointFile ):
        
        self.targetPointFile = targetPointFile
        self.sourcePointFile = sourcePointFile
        
        self.targetPoints = self._readPointFile( self.targetPointFile )
        self.sourcePoints = self._readPointFile( self.sourcePointFile )
        
        print( 'Found target points:' )
        print self.targetPoints
        print( 'Found target source:' )
        print self.sourcePoints
        
        
    def _readPointFile( self, fileName ):
        
        f    = open(fileName, 'r')
        data = f.read()
        f.close()
        
        dom = parseString( data )
        
        xEls = dom.getElementsByTagName('x')
        yEls = dom.getElementsByTagName('y')
        zEls = dom.getElementsByTagName('z')
        
        # Save the points
        points = np.zeros( (xEls.length,3) )
        
        
        for i in range( xEls.length ):
            
            points[i,0] = float( xEls[i].toxml().replace('<x>','').replace('</x>','') )
            points[i,1] = float( yEls[i].toxml().replace('<y>','').replace('</y>','') )
            points[i,2] = float( zEls[i].toxml().replace('<z>','').replace('</z>','') )
        
        return points
            


if __name__ == '__main__':
    
    targetLMFile = 'W:/Scil/Landmarks/Supine1_5.mps'
    sourceLMFile = 'W:/Scil/Landmarks/Prone1_5.mps'
    lmfr = landMarkFileReader( targetLMFile, sourceLMFile )
    
    # Visualise some reuslts...
    from mayaviPlottingWrap import plotVectorsAtPoints
    from mayaviPlottingWrap import plotArrayAs3DPoints
    #plotVectorsAtPoints(lmfr.targetPoints - lmfr.sourcePoints, lmfr.sourcePoints )
    
    plotArrayAs3DPoints(lmfr.targetPoints, (1.0,0,0))
    plotArrayAs3DPoints(lmfr.sourcePoints, (0.0,1.0,0))
    
    
    
    
    
