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


# - this module looks for Matlab
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:   required libraries: libmex, etc
#  MATLAB_MEX_LIBRARY: path to libmex.lib
#  MATLAB_MX_LIBRARY:  path to libmx.lib
#  MATLAB_ENG_LIBRARY: path to libeng.lib


set(MATLAB_FOUND 0)
if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
    set(MATLAB_ROOT "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/lib/win32/microsoft/msvc60")
  else(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
      # Assume people are generally using 7.1,
      # if using 7.0 need to link to: ../extern/lib/win32/microsoft/msvc70
      set(MATLAB_ROOT "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/lib/win32/microsoft/msvc71")
    else(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
      if(${CMAKE_GENERATOR} MATCHES "Borland")
        # Same here, there are also: bcc50 and bcc51 directories
        set(MATLAB_ROOT "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/lib/win32/microsoft/bcc54")
      else(${CMAKE_GENERATOR} MATCHES "Borland")
        message(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
      endif(${CMAKE_GENERATOR} MATCHES "Borland")
    endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
  endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
  find_library(MATLAB_MEX_LIBRARY
    libmex
    ${MATLAB_ROOT}
    )
  find_library(MATLAB_MX_LIBRARY
    libmx
    ${MATLAB_ROOT}
    )
  find_library(MATLAB_ENG_LIBRARY
    libeng
    ${MATLAB_ROOT}
    )

  find_path(MATLAB_INCLUDE_DIR
    "mex.h"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/include"
    )
else( WIN32 )

  if(APPLE)
 
    set(MATLAB_ROOT
      /Applications/MATLAB_R2009a.app/bin/maci
      )      

    find_path(MATLAB_INCLUDE_DIR
      "mex.h"
      "/Applications/MATLAB_R2009a.app/extern/include/"
      )
      
  else(APPLE)
  
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
      # Regular x86
      set(MATLAB_ROOT
        /usr/local/lib/matlab-7.2.0.294/bin/glnx86/
        /share/apps/matlab/bin/glnxa64
        /var/lib/matlab64/bin/glnxa64/
        /usr/local/matlab-7sp1/bin/glnx86/
        /opt/matlab-7sp1/bin/glnx86/
        $ENV{HOME}/matlab-7sp1/bin/glnx86/
        $ENV{HOME}/redhat-matlab/bin/glnx86/
        $ENV{HOME}/MATLAB/bin/glnxa64/
        )
    else(CMAKE_SIZEOF_VOID_P EQUAL 4)
      # AMD64:
      set(MATLAB_ROOT
        /usr/local/lib/matlab-7.2.0.294/bin/glnx86/
        /share/apps/matlab/bin/glnxa64
        /var/lib/matlab64/bin/glnxa64/
        /usr/local/matlab-7sp1/bin/glnxa64/
        /opt/matlab-7sp1/bin/glnxa64/
        $ENV{HOME}/matlab7_64/bin/glnxa64/
        $ENV{HOME}/matlab-7sp1/bin/glnxa64/
        $ENV{HOME}/redhat-matlab/bin/glnxa64/
        $ENV{HOME}/MATLAB/bin/glnxa64/
        )
    endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

    find_path(MATLAB_INCLUDE_DIR
      "mex.h"
      "/usr/local/lib/matlab-7.2.0.294/extern/include/"
      "/share/apps/matlab/extern/include"
      "/var/lib/matlab64/extern/include/"
      "/usr/local/matlab-7sp1/extern/include/"
      "/opt/matlab-7sp1/extern/include/"
      "$ENV{HOME}/matlab-7sp1/extern/include/"
      "$ENV{HOME}/redhat-matlab/extern/include/"
      "$ENV{HOME}/MATLAB/extern/include/"
      )
    
  endif(APPLE)
  
  find_library(MATLAB_MEX_LIBRARY
    mex
    ${MATLAB_ROOT}
    )
  find_library(MATLAB_MX_LIBRARY
    mx
    ${MATLAB_ROOT}
    )
  find_library(MATLAB_ENG_LIBRARY
    eng
    ${MATLAB_ROOT}
    )

endif(WIN32)

# This is common to UNIX and Win32:
set(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_ENG_LIBRARY}
)

if(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  set(MATLAB_FOUND 1)
endif(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

mark_as_advanced(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
)

