#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from xml.dom.minidom import parseString
import numpy as np
import xml2obj



class xmlModelReader :
    ''' Provide a quick way to read in an xmlModel for niftySim 
        Currently only files with one main model (no sub-models) are supported
        Only models with T4 elements are read correctly
    '''
    
    def __init__( self, xmlFileName ):
        
        self.xmlFileName = xmlFileName
        self._convertXMLFileIntoObject()
        self._extractNodesAndElements()
        self._extractMaterialParams()
        self._extractConstraints()
        self._extractSystemParameters()
        self._extractOutput()
        self._extractShellElements()
       
       
       
    def _convertXMLFileIntoObject( self ):
        
        f    = open( self.xmlFileName, 'r' )
        data = f.read()
        f.close()
        
        self.modelObject = xml2obj.xml2obj( data )
        
        pass
            
    
    
    
    def _extractNodesAndElements( self ):
                
        nds = self.modelObject.Nodes.data.splitlines()
        n=[]
        
        for l in nds:
            try:
                a, b, c = l.split()
                a = float(a)
                b = float(b)
                c = float(c)
                n.append([a,b,c])

            except:
                continue   
            
        self.nodes = np.array(n)

        
        #
        # Parse for the elements
        #
        nds = self.modelObject.Elements.data.splitlines()
        
        n=[]

        if len( nds[0].split() ) == 4:
            print( '4-node tetrahedral elements' )
            for l in nds:
                try:
                    a, b, c, d = l.split()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    
                    n.append([a,b,c,d])

                except:
                    continue   
        
        if len( nds[0].split() ) == 8:
            print( '8-node hexahedral elements' )
        
            for l in nds:
                try:
                    a, b, c, d, e, f, g, h = l.split()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    e = int(e)
                    f = int(f)
                    g = int(g)
                    h = int(h)
                    
                    n.append( [a,b,c,d,e,f,g,h] )
    
                except:
                    continue   
            
        self.elements = np.array(n)
        
        print('Done')
    
    
    
        
    def _extractMaterialParams( self ):
        
        self.materials = []
        
        
        for i in range( len(self.modelObject.ElementSet ) ):
            
            self.materials.append({})
            
            self.materials[-1]['Type']= self.modelObject.ElementSet[i].Material.Type 
            
            p =  self.modelObject.ElementSet[i].Material.ElasticParams.data.split()
            
            for j in range(len(p)):
                p[j] = float(p[j])
            
            self.materials[-1][ 'ElasticParams' ] = np.array(p) 
            
            # Visco Params 
            if self.modelObject.ElementSet[i].Material.ViscoParams != None :
                p =  self.modelObject.ElementSet[i].Material.ViscoParams.data.split()
                for j in range(len(p)):
                    p[j] = float(p[j])
                    
                self.materials[-1]['ViscoParams']      = np.array(p) 
                self.materials[-1]['NumIsoTerms'] = int( self.modelObject.ElementSet[i].Material.ViscoParams.NumIsoTerms )
                self.materials[-1]['NumVolTerms'] = int( self.modelObject.ElementSet[i].Material.ViscoParams.NumVolTerms )

            # Elements
            p=self.modelObject.ElementSet[i].data.split()
            
            for j in range(len(p)):
                try:
                    p[j] = int(p[j])
                except:
                    continue
            
            
                
                
            self.materials[-1]['Elements'] = np.array(p) 
        
        pass
        
        

        
    def _extractConstraints( self ):
        
        numConstraints = len( self.modelObject.Constraint )
        
        self.fixConstraints     = []
        self.gravityConstraints = []
        
        for i in range( numConstraints ):
            
            #
            # Fix Constraint
            #
            if self.modelObject.Constraint[i].Type == 'Fix':
                
                self.fixConstraints.append({})
                self.fixConstraints[-1]['DOF']   = int( self.modelObject.Constraint[i].DOF )
                
                p = self.modelObject.Constraint[i].Nodes.split()
                for j in range(len(p)):
                    try:
                        p[j] = int(p[j])
                    except:
                        continue

                self.fixConstraints[-1]['Nodes'] = np.array( p ) 
                
            
            #
            # Gravity Constraint
            #
            if self.modelObject.Constraint[i].Type == 'Gravity':
                
                self.gravityConstraints.append({})
                self.gravityConstraints[-1]['AccelerationMagnitude'] = float( self.modelObject.Constraint[i].AccelerationMagnitude )
                
                p = self.modelObject.Constraint[i].AccelerationDirection.split()
                
                for j in range(len(p)):
                    try:
                        p[j] = float(p[j])
                    except:
                        continue
                
                self.gravityConstraints[-1]['AccelerationDirection'] = np.array( p )
                
                p = self.modelObject.Constraint[i].Nodes.split()
                
                for j in range(len(p)):
                    try:
                        p[j] = int(p[j])
                    except:
                        continue
                
                self.gravityConstraints[-1]['Nodes'] = np.array( p )
                

            
        
    
    def _extractSystemParameters(self):
        self.systemParams = {}
        
        self.systemParams[ 'TimeStep'  ]    = float( self.modelObject.SystemParams.TimeStep )
        self.systemParams[ 'TotalTime' ]    = float( self.modelObject.SystemParams.TotalTime )
        self.systemParams[ 'DampingCoeff' ] = float( self.modelObject.SystemParams.DampingCoeff )
        self.systemParams['HGKappa']        = float( self.modelObject.SystemParams.HGKappa )
        self.systemParams['Density']        = float( self.modelObject.SystemParams.Density )
        pass
    
    
    
    
    def _extractShellElements(self):
        
        self.shellElements = {}
        self.shellElementSet = []
        
        if self.modelObject.ShellElements == None:
            return
        
        if self.modelObject.ShellElementSet == None:
            return
        
        #
        # Geometry of elements
        #
        self.shellElements['Type'] = str( self.modelObject.ShellElements.Type )
        

        nds = self.modelObject.ShellElements.data.splitlines()
        
        n=[]

        if len( nds[0].split() ) == 3:
            for l in nds:
                try:
                    a, b, c = l.split()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    
                    n.append([a,b,c])
                
                except:
                    continue   
        
            
        self.shellElements['Elements'] = np.array(n)
        
        
        #
        # Material definition of element sets
        #
        
        for i in range( len( self.modelObject.ShellElementSet ) ):
            self.shellElementSet.append({})
            
            self.shellElementSet[-1]['Size'] = int( self.modelObject.ShellElementSet[i].Size )
            
            
            p = self.modelObject.ShellElementSet[i].data.split()
                
            for j in range(len(p)):
                try:
                    p[j] = int(p[j])
                except:
                    continue
            
            self.shellElementSet[-1]['Elements']          = np.array( p )
            self.shellElementSet[-1]['MaterialType']      = str( self.modelObject.ShellElementSet[i].Material.Type )
            self.shellElementSet[-1]['MaterialDensity']   = float( self.modelObject.ShellElementSet[i].Material.Density )
            self.shellElementSet[-1]['MaterialThickness'] = float( self.modelObject.ShellElementSet[i].Material.Thickness )
            
            p = self.modelObject.ShellElementSet[i].Material.data.split()
            for j in range(len(p)):
                try:
                    p[j] = float(p[j])
                except:
                    continue
                
            self.shellElementSet[-1]['MaterialParams'] = np.array( p )

        pass
    
    
    
    
    def _extractOutput( self ):
        
        if self.modelObject.Output == None:
            return
        
        self.output = {}
        
        self.output['Freq'] = int( self.modelObject.Output.Freq )
        self.output['Variables'] = []
        
        for v in self.modelObject.Output.Variable:
            self.output['Variables'].append( str( v ) ) 
        
    
    
    
if __name__ == '__main__':
    #xReader = xmlModelReader( 'W:/philipsBreastProneSupine/referenceState/00s/modelFat_prone1G_phi00.xml' )
    xReader = xmlModelReader( 'D:/development/niftysim/trunk/nifty_sim/models/cubehex10.xml' )
    
    
    