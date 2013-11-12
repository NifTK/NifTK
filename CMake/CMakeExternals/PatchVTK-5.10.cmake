# Copyright (c) 2003-2012 German Cancer Research Center,
# Division of Medical and Biological Informatics
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the
# following conditions are met:
# 
#  * Redistributions of source code must retain the above
#    copyright notice, this list of conditions and the
#    following disclaimer.
# 
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 
#  * Neither the name of the German Cancer Research Center,
#    nor the names of its contributors may be used to endorse
#    or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Called by VTK.cmake (ExternalProject_Add) as a patch for VTK 5.10
if(APPLE)
  # patch for VTK 5.10 to work on Mac OS X

  # Updates vtkQtBarChart.cxx which fails to build on Mac OS X. The Mac OS X compiler can not resolve a isnan call of vtk.
  # Calling std::isnan solves this problem. But std::isnan is part of c++ 11 which most likely will not be recognized
  # by windows and linux compilers so this patch is needed only for Mac OS X systems
  # read whole file vtkQtBarChart.cxx
  file(STRINGS GUISupport/Qt/Chart/vtkQtBarChart.cxx sourceCode NEWLINE_CONSUME)

  # substitute dependency to gdcmMSFF by dependencies for more libraries
    string(REGEX REPLACE "[(]isnan" "(std::isnan" sourceCode ${sourceCode})

  # set variable CONTENTS, which is substituted in TEMPLATE_FILE
  set(CONTENTS ${sourceCode})
  configure_file(${TEMPLATE_FILE} GUISupport/Qt/Chart/vtkQtBarChart.cxx @ONLY)

endif()

if (WIN32)
  # patch for VTK 5.10, bug http://paraview.org/Bug/view.php?id=14122
  # fix is included in VTK 6, so we can remove this part as soon as VTK 6 is our default

  # complete patched file is in ${WIN32_OPENGL_RW_FILE}

  file(STRINGS ${WIN32_OPENGL_RW_FILE} sourceCode NEWLINE_CONSUME)
  set(CONTENTS ${sourceCode})
  configure_file(${TEMPLATE_FILE} Rendering/vtkWin32OpenGLRenderWindow.cxx @ONLY)
endif()
