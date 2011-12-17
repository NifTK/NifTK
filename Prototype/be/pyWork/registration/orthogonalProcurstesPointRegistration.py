import numpy as np
import time


class orthogonalProcrustesPointRegistration :
    
    def __init__( self, pointsSource, pointsTarget ):
        
        # these point sets are NOT in homogeneous coordinates  
        self.sourcePoints = pointsSource[0:3,:]
        self.targetPoints = pointsTarget[0:3,:]
        
        # Checking the data might need become important
        self.numPoints = np.shape( self.sourcePoints )[ 1 ]
        
        
        
    
    def register( self ) :
        tic = time.clock()
        # First "de-mean" the points
        self.sourcePointsMean = np.sum( self.sourcePoints, 1 ) / float( np.shape( self.sourcePoints )[1] )
        self.targetPointsMean = np.sum( self.targetPoints, 1 ) / float( np.shape( self.targetPoints )[1] )
                
        # Derive the rigid rotation from covariance matrix
        sourceDeMeaned = self.sourcePoints - np.tile(self.sourcePointsMean.reshape(3,1), (1, self.numPoints) )
        targetDeMeaned = self.targetPoints - np.tile(self.targetPointsMean.reshape(3,1), (1, self.numPoints) )
        
        # build the covariance matrix H from de-meaned data
        H = np.zeros( (3,3) )
        
        for i in range( self.numPoints ) :
            H = H + np.dot( sourceDeMeaned[:,i].reshape(3,1), targetDeMeaned[:,i].reshape(3,1).T)
        
        U, Sig, V      = np.linalg.svd( H )
        self.rotMat    = np.dot( V.T, U.T )  
        self.transVect = self.targetPointsMean - np.dot( self.rotMat, self.sourcePointsMean ) 

        # Construct the transformation in homogeneous coordinates
        self.homRigTransfromMat = np.eye( 4 ) 
        self.homRigTransfromMat[0:3, 0:3] = self.rotMat
        self.homRigTransfromMat[0:3,   3] = self.transVect
        
        toc = time.clock()
        
        print 'Registration took: '  + str( toc - tic      ) + 's'
        print 'Rotation matrix: '    + str( self.rotMat    )
        print 'Translation vector: ' + str( self.transVect )











