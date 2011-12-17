#! /usr/bin/env python 
# -*- coding: utf-8 -*-

'''
@author: Bjoern Eiben
'''


from threading import Thread
import time


class registrationThread( Thread ):
    def __init__ ( self, taskListIn, threadID = 0 ) :
        
        Thread.__init__(self)
        self.taskList = taskListIn
        self.threadID       = threadID
        
    def run(self):
        print( 'Registration thread %d started' % self.threadID )
        print( ' Length of task list %d' % len(self.taskList) )
        
        i = 0
        
        for task in self.taskList :
            print('Thread ' + str(self.threadID) + ' running task ' + str(i) )
            task.run() 
            i = i+1
        
        
        

class registrationTaskListExecuter :
    ''' This class takes care that the task list is executed in a parallel fashion
        The tasks need to provide a method which is called 'run'. Each task within 
        the list will be run as task.run()
        
    '''
    
    def __init__( self, taskListIn, numThreadsIn = 1 ) :
        
        self.taskList        = taskListIn
        self.numThreads      = numThreadsIn
        self.regThreads      = []
        self.threadTaskLists = []
        
         
         
    def execute( self ) :
        
        # divide the task List into several ones
        for thrNum in range( self.numThreads ) :
            self.threadTaskLists.append([])
        
        i = 0
        
        for task in self.taskList :
            thrNum = i % self.numThreads
            self.threadTaskLists[thrNum].append( task )
            i = i+1
                
        # Start the threads

        
        for thrNum in range( self.numThreads ) :
            thread = registrationThread( self.threadTaskLists[ thrNum ], thrNum )
            self.regThreads.append( thread )
            thread.start()
            time.sleep(5)
        
        for thrNum in range( self.numThreads ) :
            self.regThreads[ thrNum ].join()
            print('Thread ' + str( thrNum ) + ' ...Done...' )
            
