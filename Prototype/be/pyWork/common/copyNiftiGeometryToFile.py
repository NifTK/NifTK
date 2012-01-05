#! /usr/bin/env python 
# -*- coding: utf-8 -*-

''' This Script reads the header entries of one nifti file and copies these into another
    Brutal but effective.
'''


import sys 
import nibabel as nib



def main( hdrSourceNiftiFile = None, hdrTargetNiftiFile = None, getSysArgs = True ) :
    
    if getSysArgs :
        if len( sys.argv ) != 3:
            print( 'Usage: \n --> copyNiftiHeader.py source-nifti-file target-nifti-file'  )
            return
        
        headerSourceFile = sys.argv[1]
        headerTargetFile = sys.argv[2]
    
    else:
        headerSourceFile = hdrSourceNiftiFile
        headerTargetFile = hdrTargetNiftiFile
        

    target = nib.load( headerTargetFile )
    source = nib.load( headerSourceFile )
    
    hdrT = target.get_header()
    hdrS = source.get_header()
    
    hdrT.set_qform( hdrS.get_qform() )
    hdrT.set_sform( hdrS.get_sform() )
    
    newImg = nib.Nifti1Image( target.get_data(), source.get_affine(), header = hdrT )
    newImg.update_header()
    newImg.to_filename( headerTargetFile )
    
    return

#    # try to find the executable
#    if fe.findExecutable( niftiTool ) == None :
#        print( 'Cannot find the essential executable: '  + niftiTool )
#        return
#    
#    # get the nim information of the first (header-source) file (hdr will be adapted accordingly...)
#    params      = '-disp_nim -infiles ' + headerSourceFile 
#    cmd         = shlex.split ( niftiTool + ' ' + params ) 
#    strProgOut  = subprocess.check_output( cmd )
#    strNIMLines = strProgOut.splitlines()
#    
#    #TODO: Some more sophisticated checking of file characteristics would be necessary
#    
#    parameterStart = -1
#    nimFields      = []
#    
#    for i in range( len( strNIMLines ) ) :
#        if strNIMLines[i].strip().startswith( '-' ) :
#            print('Found --- in line %i' % i )
#            parameterStart = i+1
#            continue
#        
#        if parameterStart != -1 :
#            nimFields.append(strNIMLines[i].split()) 
#    
#    # now construct the modification command:
#    modCmd = '-infiles ' + headerTargetFile + ' -overwrite -mod_nim '
#    
#    for f in nimFields :
#        
#        # skip a few fields
#        if not ( (f[0] == 'ndim') or 
#                 (f[0] == 'dim') or 
#                 (f[0] == 'dx') or 
#                 (f[0] == 'dy') or 
#                 (f[0] == 'dz') or 
#                 (f[0] == 'pixdim') or 
#                 (f[0] == 'quatern_b') or 
#                 (f[0] == 'quatern_c') or 
#                 (f[0] == 'quatern_d') or 
#                 (f[0] == 'qoffset_x') or 
#                 (f[0] == 'qoffset_y') or 
#                 (f[0] == 'qoffset_z') or 
#                 (f[0] == 'qfac') or 
#                 (f[0] == 'qto_xyz') or 
#                 (f[0] == 'qto_ijk') or 
#                 (f[0] == 'sto_xyz') or 
#                 (f[0] == 'sto_ijk') or 
#                 (f[0] == 'xyz_units') ) : 
#            continue   
#        
#        vals = ''
#        for v in range( 3, len(f) ):
#            vals += f[v] + ' '
#        vals = vals.strip()
#        
#        if len(f) > 4:
#            vals = '\'' + vals + '\'' 
#        
#        modCmd += ' -mod_field ' + f[0] + ' ' + vals 
#    
#    print( strProgOut )
#    print( modCmd     )
#    
#    cmd = shlex.split ( niftiTool + ' ' + modCmd ) 
#    subprocess.Popen( cmd ).wait()




if __name__ == "__main__" :
    main( getSysArgs = True )
    