#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import numpy as np
import vtkMeshFileReader as vmr
import matplotlib.pyplot as plt



class meshStatistics:
    ''' Class which builds mesh statistics 
     
    '''
    
    def __init__( self, nodes, elements ):
        
        self.nodes    = nodes
        self.elements = elements
        
        self._calcStatistics()
        pass
    
    
    
    
    def _calcStatistics( self ):
        ''' Calculate the statistics for each element and  
        '''
        
        xPts = self.nodes[:,0]
        yPts = self.nodes[:,1]
        zPts = self.nodes[:,2]
        
        # big Matrix with coordinate positions. 
        M = np.array( ( xPts[self.elements[:,0]], yPts[self.elements[:,0]], zPts[self.elements[:,0]], 
                        xPts[self.elements[:,1]], yPts[self.elements[:,1]], zPts[self.elements[:,1]],
                        xPts[self.elements[:,2]], yPts[self.elements[:,2]], zPts[self.elements[:,2]],
                        xPts[self.elements[:,3]], yPts[self.elements[:,3]], zPts[self.elements[:,3]] ) ).T
        
        a = M[:,0:3] - M[:,3: 6]
        b = M[:,0:3] - M[:,6: 9]
        c = M[:,0:3] - M[:,9:12]
        d = M[:,3:6] - M[:,6: 9]
        e = M[:,3:6] - M[:,9:12]
        f = M[:,6:9] - M[:,9:12]
        
        #
        # Calculate lengths
        #
        self.Lengths = np.array( np.sqrt( ( np.sum(a * a, 1 ), 
                                            np.sum(b * b, 1 ),
                                            np.sum(c * c, 1 ),
                                            np.sum(d * d, 1 ),
                                            np.sum(e * e, 1 ),
                                            np.sum(f * f, 1 ) ) ) ).T
                        
        plt.hist( self.Lengths[:], bins=120 )
        
        #
        # Calculate volumes:
        # 
        #      | a.(b x c) |
        # V = ---------------
        #            6
         
        self.Vols = np.abs( a[:,0] *(  b[:,1] * c[:,2] - b[:,2] * c[:,1] ) + 
                            a[:,1] *(  b[:,2] * c[:,0] - b[:,0] * c[:,2] ) + 
                            a[:,2] *(  b[:,0] * c[:,1] - b[:,1] * c[:,0] ) ) / 6.0  
        
        print( 'Done.' )
        
        

    
    
if __name__ == '__main__' :

    vtkMeshName = 'W:/philipsBreastProneSupine/referenceState/00/referenceState/surfMeshImpro.1.vtk'
    mesh = vmr.vtkMeshFileReader( vtkMeshName )
    
    stat = meshStatistics(mesh.points, mesh.cells[:,1:] )




