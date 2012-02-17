#! /usr/bin/env python 
# -*- coding: utf-8 -*-

from xml.dom.minidom import parse
import numpy as np
import vtk

class xmlModelReader :
    ''' Provide a quick way to read in an xmlModel for niftySim 
    '''
    
    def __init__( self, xmlFileName ):
       
       
        self.xmlDoc = parse( xmlFileName )
        pass
    
    
    
    
    def _extractNodesAndElements( self ):
        
        self.nodse = self.xmlDoc.getElementsByTagName('Nodes')
                
        pass