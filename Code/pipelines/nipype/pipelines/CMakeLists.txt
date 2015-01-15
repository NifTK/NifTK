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

file(GLOB __bash_files      "*.sh")
file(GLOB __matlab_files    "*.m")
file(GLOB __python_programs "perform_*.py")

install(FILES    ${__matlab_files}    DESTINATION bin)
install(PROGRAMS ${__bash_files}      DESTINATION bin)
install(PROGRAMS ${__python_programs} DESTINATION python)
