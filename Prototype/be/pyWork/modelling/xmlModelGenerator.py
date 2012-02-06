#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author:  Bjoern Eiben
@summary: Class to generate an xml file for niftySim FEM simulations. Currently only 3D will be supported - niftySim is 3D only anyway... 
'''


from xml.dom.minidom import Document
import numpy as np
import vtk

class xmlModelGenrator :

    def __init__ ( self, nodes, elements, elementType = 'T4ANP' ):
        ''' @param nodes: Expected to be a 2D numpy array of type 'float'. 
            @param elements: Expected to be a 2D numpy array of type 'int'.
        '''
        self.nodes                  = nodes
        self.elements               = elements
        self.fixConstraintNodes     = [None, None, None]
        self.materialSets           = []
        self.contactSurfaces        = []
        self.contactSurfaceVTKFiles = []
        self.contactCylinders       = []
        self.uniformDispConstraints = []
        self.difformDispConstraints = []
        self.elementType            = elementType
        
        # Track which and how many elements were set.
        self._systemParametersSet   = False
        self._gravityConstraintSet  = False
        self._outputSet             = False

        self.allNodesArray    = np.array( range( self.nodes.shape[0] ) )
        self.allElemenstArray = np.array( range( self.elements.shape[0]  ) )


    
    def setSystemParameters( self, timeStep, totalTime, dampingCoefficient, hgKappa, density ):
        ''' @summary: simply collect the parameters needed
        '''
        
        self.timeStep           = timeStep
        self.totalTime          = totalTime
        self.dampingCoefficient = dampingCoefficient
        self.hgKappa            = hgKappa
        self.density            = density
        
        self._systemParametersSet = True
        
        
        
        
    def setGravityConstraint( self, direction, magnitude, nodes, loadMode = 'STEP' ):
        ''' @summary: Specify magnityde and direction of gravity for the given nodes
        '''
        self.gravityDirection = direction
        self.gravityMagnitude = magnitude
        self.gravityNodes     = nodes
        self.gravityLoadMode  = loadMode
        
        self._gravityConstraintSet = True
        
        
        
        
    def setFixConstraint( self, nodes, dofNum ):
        ''' @summary: Define the nodes which are fixed in the final model. Only node numbers which 
                      are not already fixed will be added.
            @param dofNum: 0, 1 or 2 depending on the DOF which shall be fixed.
        '''
        if self.fixConstraintNodes[dofNum] == None :
            self.fixConstraintNodes[dofNum] = nodes
        else :
            self.fixConstraintNodes[dofNum] = np.hstack( ( self.fixConstraintNodes[dofNum], nodes ) )
            self.fixConstraintNodes[dofNum] = np.unique( self.fixConstraintNodes[dofNum] )
        
    
    
    def setUniformDispConstraint( self, dofNum, loadShape, nodes, magnitude ):
        ''' @summary: Define those nodes that should be displaced into direction of the specified dof number.
                      The same displacement will be applied to all nodes
        '''
        self.uniformDispConstraints.append( [ dofNum, loadShape, nodes, magnitude ] )
    
    
    
    
    def setDifformDispConstraint( self, loadShape, nodes, magnitudeArray ):
        ''' @summary: Define those nodes that should be displaced into direction of the specified dof number.
                      The same displacement will be applied to all nodes
                      @param magnitude: numpy array with three columns wich represent the displacemnt into the differnt 
                                        dof-directions. This has to have the same length as the numver of nodes specified.
        '''
        self.difformDispConstraints.append( [ loadShape, nodes, magnitudeArray ] )
        
        
        
        
    def setMaterialElementSet( self, materialType, materialName, materialElasticParams, materialElements ):
        ''' @summary: Specify a material set and the nodes accordingly. 
        '''
        self.materialSets.append([materialType, materialName, materialElasticParams, materialElements ])
        pass
        
        
    
    
    def setOutput( self, freq, variableName ):
        ''' @summary: Specify which variable should be  saved with which frequency.
        '''
        
        self.outputFreq   = freq
        self.outputVarName = variableName
        
        self._outputSet = True
        
        
        
        
    def setContactSurface(self, contactNodes, contactElements, slvNodes, contactMeshType = 'T3' ):
        
        self.contactSurfaces.append( [contactNodes, contactElements, slvNodes, contactMeshType] )
    
    
    
    def setContactSurfaceVTKFile(self, vtkMeshFileName, strType, numNodes ) :
        ''' Use with care, as the vtk file might not be scaled correctly...
        '''
        self.contactSurfaceVTKFiles.append( [vtkMeshFileName, strType, numNodes] )
        
    
    
    
    def setContactCylinder(self, arrayOrigin, arrayAxis, scalarRadius, scalarLength, arrayOrigDisp, scalarRadChange, slvNodes ):
        
        self.contactCylinders.append( [arrayOrigin, arrayAxis, scalarRadius, scalarLength, arrayOrigDisp, scalarRadChange, slvNodes ] )
        
    
    
    def writeXML( self, xmlFileName ):
        
        print( 'Preparing xml document...' )
        
        self.xmlFileName = xmlFileName
        
        # Create the minidom document
        doc = Document()

        # Generate the main model node
        model = doc.createElement( 'Model' )
        doc.appendChild( model )
        

        # first handle the nodes (float coordinates)
        nds = doc.createElement('Nodes')
        nds.setAttribute('DOF', '%i' % self.nodes.shape[1])
        nds.setAttribute('NumNodes', '%i' % self.nodes.shape[0] )

        ndsCoordinates = doc.createTextNode( writeArrayToStr( self.nodes, True ) )
        
        nds.appendChild( ndsCoordinates )
        model.appendChild( nds )

        
        # Write the elements
        els = doc.createElement( 'Elements' )
        els.setAttribute( 'NumEls', '%i' % self.elements.shape[0] )
        els.setAttribute( 'Type', self.elementType )
        
        elsComb = doc.createTextNode( writeArrayToStr( self.elements, False ) )
        els.appendChild( elsComb )
        model.appendChild( els )


        # Write material sets
        for i in range( len( self.materialSets ) ) :
            
            # Write the element set
            elSet = doc.createElement('ElementSet')
            elSet.setAttribute('Size', '%i' % self.materialSets[i][3].shape[0] )
            
            mat = doc.createElement( 'Material' )
            mat.setAttribute( 'Type', self.materialSets[i][0] )
            mat.setAttribute( 'Name', self.materialSets[i][1] )
            
            elaParams = doc.createElement( 'ElasticParams' )
            elaParams.setAttribute( 'NumParams', '%i' % len( self.materialSets[i][2] ) )
            elaParamEntries = doc.createTextNode('%i %i' % ( self.materialSets[i][2][0],self.materialSets[i][2][1] ) )
            elaParams.appendChild( elaParamEntries )
            
            mat.appendChild(elaParams)
            
            elSetEntries = doc.createTextNode( writeArrayToStr(self.materialSets[i][3], False ) )
            elSet.appendChild( mat )
            elSet.appendChild( elSetEntries )
            model.appendChild( elSet )


        # Write fix constraints
        for i in range( len( self.fixConstraintNodes ) ):
            
            fixNodes = self.fixConstraintNodes[i]
            
            if fixNodes == None :
                continue
            
            # Write constraint
            constr = doc.createElement( 'Constraint' )
            constr.setAttribute( 'DOF', '%i' % i )
            constr.setAttribute( 'NumNodes', '%i' % fixNodes.shape[0] ) 
            constr.setAttribute( 'Type', 'Fix' )
            
            ndsC        = doc.createElement( 'Nodes' )
            ndsCEntires = doc.createTextNode( writeArrayToStr( fixNodes, False, '      ' ) )
            
            ndsC.appendChild( ndsCEntires )
            constr.appendChild( ndsC )
            model.appendChild( constr )
        
        
        # Write uniform displacement constraints
        for i in range( len( self.uniformDispConstraints ) ) :
            # dof shape nodes
            dispConstr = doc.createElement( 'Constraint' )
            dispConstr.setAttribute( 'Type',      'Disp'                                     )
            dispConstr.setAttribute( 'DOF',       '%i' % self.uniformDispConstraints[i][0]          )
            dispConstr.setAttribute( 'LoadShape', '%s' % self.uniformDispConstraints[i][1]          )
            dispConstr.setAttribute( 'NumNodes',  '%i' % self.uniformDispConstraints[i][2].shape[0] )
            
            # handle the nodes (integer numbers)
            nds = doc.createElement( 'Nodes' )
            ndsIndices = doc.createTextNode( writeArrayToStr( self.uniformDispConstraints[i][2], False, '      '  ) )
            nds.appendChild( ndsIndices )
            
            dispMag =doc.createElement( 'Magnitudes' )
            dispMag.setAttribute( 'Type', 'UNIFORM' )
            dispMag.appendChild( doc.createTextNode('%f' % self.uniformDispConstraints[i][3] ))
            
            
            dispConstr.appendChild( nds     )
            dispConstr.appendChild( dispMag )
            model.appendChild( dispConstr )
            
        
        # Write difform displacement constraints
        for i in range( len( self.difformDispConstraints ) ) :
            # the three dimensional array needs to be split up into its coplumns
            # but it is unlikely, that displacements are always only into one direction
            # dof shape nodes
            
            # array order: [ loadShape, nodes, magnitudeArray ]
            
            for dim in range(3): 
                dispConstr = doc.createElement( 'Constraint' )
                dispConstr.setAttribute( 'Type',      'Disp'                                     )
                dispConstr.setAttribute( 'DOF',       '%i' % dim          )
                dispConstr.setAttribute( 'LoadShape', '%s' % self.difformDispConstraints[i][0]          )
                dispConstr.setAttribute( 'NumNodes',  '%i' % self.difformDispConstraints[i][1].shape[0] )
                
                # handle the nodes (integer numbers)
                nds = doc.createElement( 'Nodes' )
                ndsIndices = doc.createTextNode( writeArrayToStr( self.difformDispConstraints[i][1], False, '      '  ) )
                nds.appendChild( ndsIndices )
                
                curDispMags = self.difformDispConstraints[i][2]
                
                dispMag =doc.createElement( 'Magnitudes' )
                dispMag.setAttribute( 'Type', 'DIFFORM' )
                dispMag.appendChild( doc.createTextNode( writeArrayToStr( curDispMags[:,dim], True, '      '  ) ) )
                
                
                dispConstr.appendChild( nds     )
                dispConstr.appendChild( dispMag )
                model.appendChild( dispConstr )
                

        # Write contact surface constraint
        for i in range( len( self.contactSurfaces ) ) :
            contactConstr = doc.createElement( 'ContactSurface' )
            
            contactNodes = doc.createElement( 'Nodes' )
            contactNodes.setAttribute( 'DOF', '%i' % self.contactSurfaces[i][0].shape[1] )
            contactNodes.setAttribute( 'NumNodes', '%i' % self.contactSurfaces[i][0].shape[0] )
            
            contactCoordinates = doc.createTextNode( writeArrayToStr( self.contactSurfaces[i][0], True, '      ' ) )
            contactNodes.appendChild( contactCoordinates )
            
            
            cntEls = doc.createElement( 'Elements' )
            cntEls.setAttribute( 'NumEls', '%i' % self.contactSurfaces[i][1].shape[0] )
            cntEls.setAttribute( 'Type', '%s' % self.contactSurfaces[i][3] )
            contactElementNums = doc.createTextNode( writeArrayToStr( self.contactSurfaces[i][1], False, '      ' ) )
            cntEls.appendChild( contactElementNums )
            
            
            contactSLVNodes = doc.createElement( 'SlvNodes' )
            contactSLVNodes.setAttribute( 'NumNodes', '%i' % self.contactSurfaces[i][2].shape[0] )
            slvNodeNums= doc.createTextNode( writeArrayToStr(self.contactSurfaces[i][2], False, '      ' ) )
            
            contactSLVNodes.appendChild( slvNodeNums )
            
            contactConstr.appendChild( contactNodes )
            contactConstr.appendChild( cntEls )
            
            contactConstr.appendChild( contactSLVNodes )
            
            model.appendChild( contactConstr )
        
        
        # Write contact VTK files    
        for i in range( len( self.contactSurfaceVTKFiles ) ) :  
            contactConstr = doc.createElement( 'ContactSurface' )
            
            vtkSurfNode = doc.createElement( 'VTKSurface' )
            vtkSurfNode.setAttribute( 'Type', self.contactSurfaceVTKFiles[i][1] )
            
            fileNameNode = doc.createTextNode( self.contactSurfaceVTKFiles[i][0] )
            vtkSurfNode.appendChild( fileNameNode )
            
            contactConstr.appendChild( vtkSurfNode )
            
            contactSLVNodes = doc.createElement( 'SlvNodes' )
            contactSLVNodes.setAttribute( 'NumNodes', '%i' % self.contactSurfaceVTKFiles[i][2] )
            slvNodeNums= doc.createTextNode( '0' ) 
            
            contactSLVNodes.appendChild( slvNodeNums )
            
            contactConstr.appendChild( contactSLVNodes )
            
            model.appendChild( contactConstr )
            

        # Write contact cylinder
        for i in range( len( self.contactCylinders ) ) :
            contactCylinder = doc.createElement( 'ContactCylinder' )
            
            cylOrigin = doc.createElement( 'Origin' )
            cylOrigin.appendChild( doc.createTextNode( writeArrayToStr( self.contactCylinders[i][0], True ) ) )
            
            cylAxis = doc.createElement( 'Axis' )
            cylAxis.appendChild( doc.createTextNode( writeArrayToStr( self.contactCylinders[i][1], True ) ) )
            
            cylRadius = doc.createElement( 'Radius' )
            cylRadius.appendChild( doc.createTextNode( '%e' % self.contactCylinders[i][2] ) )
            
            cylLength = doc.createElement( 'Length' )
            cylLength.appendChild( doc.createTextNode( '%e' % self.contactCylinders[i][3] ) )
            
            cylOrigDisp = doc.createElement( 'OrigDisp' )
            cylOrigDisp.appendChild( doc.createTextNode( writeArrayToStr( self.contactCylinders[i][4], True ) ) )
            
            cylRadChange = doc.createElement( 'RadChange' )
            cylRadChange.appendChild( doc.createTextNode( '%e' % self.contactCylinders[i][5] ) )
            
            
            cylSLVNodes = doc.createElement( 'SlvNodes' )
            cylSLVNodes.setAttribute( 'NumNodes', '%i' % self.contactCylinders[i][6].shape[0] )
            cylSLVNodes.appendChild( doc.createTextNode( writeArrayToStr(self.contactCylinders[i][6], False, '      ' ) ) )
            
            # assemble this node
            contactCylinder.appendChild( cylOrigin )
            contactCylinder.appendChild( cylAxis )
            contactCylinder.appendChild( cylRadius )
            contactCylinder.appendChild( cylLength )
            contactCylinder.appendChild( cylOrigDisp )
            contactCylinder.appendChild( cylRadChange )
            contactCylinder.appendChild( cylSLVNodes )
            
            model.appendChild( contactCylinder )


        # Now the gravity constraint
        if self._gravityConstraintSet :
            constr = doc.createElement( 'Constraint' )
            constr.setAttribute( 'LoadShape', self.gravityLoadMode )
            constr.setAttribute( 'NumNodes', '%i' % self.gravityNodes.shape[0] ) 
            constr.setAttribute( 'Type', 'Gravity' )
            
            ndsCg = doc.createElement( 'Nodes' )
            ndsCgEntires = doc.createTextNode( writeArrayToStr(self.gravityNodes, False, '      ' ) )
            
            accMag = doc.createElement( 'AccelerationMagnitude' )
            accMag.appendChild(doc.createTextNode('%f' % self.gravityMagnitude ))
            
            
            accDir = doc.createElement( 'AccelerationDirection' )
            accDir.appendChild( doc.createTextNode( '%f %f %f' % ( self.gravityDirection[0], self.gravityDirection[1], self.gravityDirection[2] ) ) )
            
            ndsCg.appendChild( ndsCgEntires )
            constr.appendChild( ndsCg )
            
            constr.appendChild( accMag )
            constr.appendChild( accDir )
            
            model.appendChild( constr )
        

        # System parameters
        if self._systemParametersSet :
            
            sysPars = doc.createElement( 'SystemParams' )
            timeStep =  doc.createElement( 'TimeStep' )
            timeStep.appendChild( doc.createTextNode('%e' % self.timeStep ) )
            
            totalTime = doc.createElement( 'TotalTime' )
            totalTime.appendChild(doc.createTextNode('%f' % self.totalTime ) )
            
            dampingCoeff = doc.createElement( 'DampingCoeff' )
            dampingCoeff.appendChild( doc.createTextNode('%f' % self.dampingCoefficient ) )
            
            hgKappa = doc.createElement( 'HGKappa' )
            hgKappa.appendChild( doc.createTextNode('%f' % self.hgKappa ) )
            
            density = doc.createElement( 'Density' )
            density.appendChild( doc.createTextNode('%d' % self.density ) )
            
            
            sysPars.appendChild( timeStep     )
            sysPars.appendChild( totalTime    )
            sysPars.appendChild( dampingCoeff )
            sysPars.appendChild( hgKappa      )
            sysPars.appendChild( density      )
            
            model.appendChild( sysPars )

        # Output
        if self._outputSet :
            outPut = doc.createElement( 'Output' )
            outPut.setAttribute( 'Freq', '%i' % self.outputFreq )
            var = doc.createElement( 'Variable' )
            var.appendChild( doc.createTextNode( self.outputVarName ) )
            outPut.appendChild( var )
            model.appendChild( outPut )
             

        # Done... now generate the model
        print('Writing xml document to ' + xmlFileName )
        #print doc.toprettyxml( indent="  " )
        
        xmlFile = file( xmlFileName, 'w' )
        doc.writexml( xmlFile, indent='', addindent='  ', newl='\n', encoding='utf-8' )
        xmlFile.close()




def writeArrayToStr( array, floatingPoint=True, indent = '    ' ) :
    string = ''
    
    if array.ndim == 2 :

        iRange = range( array.shape[ 0 ] )
        jRange = range( array.shape[ 1 ] )
        
        if floatingPoint :
            for i in iRange : 
                for j in jRange :
                    string = string + str('%.14f '  % (array[i,j]) )
                
                string = string+str( '\n' + indent )
        
        else :
            for i in iRange : 
                for j in jRange :
                    string = string + str('%i '  % int(array[i,j]) )
            
                string = string+str( '\n' + indent )

    
    if array.ndim == 1 :
        iRange = range( array.shape[ 0 ] ) 
        if floatingPoint :
            for i in iRange : 
                string = string + str('%.14f '  % (array[i]) )
                string = string+str( '\n' + indent )
        
        else :
            for i in iRange : 
                string = string + str('%i '  % int(array[i]) )
                string = string+str( '\n' + indent )
        
        
    

    return string
