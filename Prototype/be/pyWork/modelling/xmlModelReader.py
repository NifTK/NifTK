#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from xml.dom.minidom import parseString
import numpy as np


class xmlModelReader :
    ''' Provide a quick way to read in an xmlModel for niftySim 
        Currently only files with one main model (no sub-models) are supported
        Only models with T4 elements are read correctly
    '''
    
    def __init__( self, xmlFileName ):
        
        self.xmlFileName = xmlFileName
        self._extractNodesAndElements()
       
        
    
    def _extractNodesAndElements( self ):
        
        f    = open( self.xmlFileName, 'r' )
        data = f.read()
        f.close()
        
        self.dom = parseString( data )       
        
        #
        # Parse the nodes:
        #
        el       = self.dom.getElementsByTagName('Nodes')[0]
        nds      = el.toxml().splitlines()
        
        n=[]
        r=range(len(nds))
        for l in r[1:-1]:
            try:
                a, b, c = nds[l].split()
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
        el       = self.dom.getElementsByTagName('Elements')[0]
        nds      = el.toxml().splitlines()
        
        n=[]
        r=range(len(nds))
        for l in r[1:-1]:
            try:
                a, b, c, d = nds[l].split()
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                
                n.append([a,b,c,d])

            except:
                continue   
            
        self.elements = np.array(n)
        
        print('Done')
        
    
    
    
    
if __name__ == '__main__':
    xReader = xmlModelReader( 'W:/philipsBreastProneSupine/referenceState/00AB/modelFat_prone1G_phi00.xml' )
    