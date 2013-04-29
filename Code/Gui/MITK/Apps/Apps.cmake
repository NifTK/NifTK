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

# This file is included in the top-level CMakeLists.txt file to allow early dependency checking

option(NIFTK_Apps/NiftyView "Build NiftyView - Research application for all users." OFF)
option(NIFTK_Apps/NiftyMIDAS "Build NiftyMIDAS - Dementia Research Centre application for clinical trials in Dementia." OFF)
option(NIFTK_Apps/NiftyIGI "Build NiftyIGI - Research application for general image guided interventions" OFF)

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

