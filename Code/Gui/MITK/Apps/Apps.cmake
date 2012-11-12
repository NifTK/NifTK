#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8038 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# This file is included in the top-level CMakeLists.txt file to allow early dependency checking

option(NIFTK_Apps/NiftyView "Build NiftyView - Research application for all users." ON)
option(NIFTK_Apps/NiftyMIDAS "Build NiftyMIDAS - Dementia Research Centre application for clinical trials in Dementia." ON)
option(NIFTK_Apps/NiftyIGI "Build NiftyIGI - Research application for general image guided interventions" ON)

# This variable is fed to ctkFunctionSetupPlugins() macro in the
# top-level CMakeLists.txt file. This allows to automatically
# enable required plug-in runtime dependencies for applications using
# the CTK DGraph executable and the ctkMacroValidateBuildOptions macro.
# For this to work, directories containing executables must contain
# a CMakeLists.txt file containing a "project(...)" command and a
# target_libraries.cmake file setting a list named "target_libraries"
# with required plug-in target names.

set(NIFTK_APPS
  NiftyView^^NIFTK_Apps/NiftyView
  NiftyMIDAS^^NIFTK_Apps/NiftyMIDAS
  NiftyIGI^^NIFTK_Apps/NiftyIGI
)

