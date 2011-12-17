#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import re, os, sys


def findExecutable(executable, path=None):
    '''Try to find 'executable' in the directories listed in 'path' (a
    string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH']).  Returns the complete filename or None if not
    found
    '''
    
    if path is None:
        path = os.environ[ 'PATH' ]
        
    paths = path.split( os.pathsep )
    extlist = [ '' ]
    
    if os.name == 'os2':
        ( base, ext ) = os.path.splitext( executable )
        # executable files on OS/2 can have an arbitrary extension, but
        # .exe is automatically appended if no dot is present in the name
        
        if not ext:
            executable = executable + ".exe"
    
    elif sys.platform == 'win32':
        pathext = os.environ[ 'PATHEXT' ].lower().split(os.pathsep)
        ( base, ext ) = os.path.splitext( executable )
        
        if ext.lower() not in pathext:
            extlist = pathext
    
    for ext in extlist:
        execname = executable + ext
        
        if os.path.isfile(execname):
            return execname
        else:
            for p in paths:
                f = os.path.join(p, execname)
                if os.path.isfile(f):
                    return f
    else:
        return None

        
if __name__ == '__main__' : 
    
    import sys
    
    if len( sys.argv ) != 2 : 
        print( 'Usage: %s executable' % sys.argv[0] )
        sys.exit()
        
    print( findExecutable( sys.argv[1] ) )
    
    
    
