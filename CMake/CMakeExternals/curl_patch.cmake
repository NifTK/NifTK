#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

## MASSIVE HACK: Adding a patch to libcurl as a workaround for cmake bug 0011240
## http://public.kitware.com/Bug/view.php?id=11240  
  
  set( _SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set( _TARGET "${_SOURCE_DIR}/CMakeLists.txt" )
    
  if( NOT (EXISTS "${_TARGET}") )
	message( FATAL_ERROR "Cannot find ${_TARGET}" )
  endif()
  
  ## HAVE to force STRINGS mode otherwise cmake's read command is removing all the semicolons
  file(STRINGS "${_TARGET}" _DATA NEWLINE_CONSUME)
  list( APPEND _DATA "#=== Patch - append BEGIN ===\n" )
  
  # Error detected with generator Visual Studio 10 Win64
  # > LINK : warning LNK4068: /MACHINE not specified; defaulting to X86
  # > fatal error LNK1112: module machine type 'x64' conflicts with target machine type 'X86'
  #
  # There is no default value for static libraries and cmake isn't setting it either.
  # We fix this by adding the flag manually.
	
  message("\n ********* Specifying machine type: /machine:x64 ********* \n" )
  list( APPEND _DATA "set_target_properties(libcurl PROPERTIES STATIC_LIBRARY_FLAGS \"/MACHINE:x64\")\n" )
  list( APPEND _DATA "#=== Patch - append END ===\n" )
  file( WRITE "${_TARGET}" ${_DATA} )
   
## END OF MASSIVE HACK: Adding a patch to libcurl as a workaround for cmake bug 0011240 