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
            
            p=self.modelObject.ElementSet[i].data.split()
            
            for j in range(len(p)):
                try:
                    p[j] = int(p[j])
                except:
                    continue
            
            self.materials[-1]['Elements'] = np.array(p) 
        
        pass
        
    
    
    
if __name__ == '__main__':
    xReader = xmlModelReader( 'W:/philipsBreastProneSupine/referenceState/00s/modelFat_prone1G_phi00.xml' )
    
    
    