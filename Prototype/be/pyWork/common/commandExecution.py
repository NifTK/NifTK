#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import findExecutable
import shlex, time, subprocess, os


def runCommand( cmdIn, paramsIn, logFileName=None, workDir=None, onlyPrintCommand=False ) :
    
    if workDir != None :
        curDir = os.getcwd()
        os.chdir( workDir )
        
    # Is the executable available?
    if findExecutable.findExecutable( cmdIn ) == None:
        print('Cannot find %s in path. Sorry.' % cmdIn )
        return
    
    print('Running:\n -> %s %s' % ( cmdIn, paramsIn ) )
    cmd = shlex.split ( cmdIn + ' ' + paramsIn ) 

    tic = time.clock()
    
    if not onlyPrintCommand :
        if logFileName == None :
            ret=subprocess.Popen( cmd ).wait()
            
        else:
            logFile = file( logFileName, 'a+' )
            ret = subprocess.Popen( cmd, stdout = logFile, stderr = logFile ).wait()
            logFile.close()
    
        print('Return code: ' + str(ret) )
        
    if workDir != None :
        os.chdir( curDir )
    
    toc = time.clock()
    
    print('Done. This took %.2fs' %( toc-tic ) )
    return ret
