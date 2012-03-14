

def numpyArrayToStr( array, floatingPoint=True, indent = '    ' ) :
    ''' @summary: Convert a numpy array into a string. Only one and two
                  dimensional arrays are supported. 1D arrays will be printed
                  like a column vector. 
        @param array: Input array, expected to be a numpy one.
        @param floatingPoint: Set to true, if array should be printed as a
        @param indent: Indentation used from second line on.  
    '''
    
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


