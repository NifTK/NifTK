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

option(NIFTK_USE_COTIRE "Use Cotire for compilation speed-up. Experimental." OFF)
if(NIFTK_USE_COTIRE)
  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake;${CMAKE_MODULE_PATH}")
  include(cotire)
endif()

###########################################################################
# Our version number. Edit this to generate a new version.
# However, be warned, you need to set CMAKE_INSTALL_PREFIX manually.
# Also, Trac 1592, version number must change if CTK changes significantly.
###########################################################################
set(NIFTK_VERSION_MAJOR 15 CACHE STRING "Year version number" FORCE )
set(NIFTK_VERSION_MINOR 01 CACHE STRING "Month version number" FORCE )
set(NIFTK_VERSION_PATCH 0 CACHE STRING "Patch version number" FORCE )
mark_as_advanced(NIFTK_VERSION_MAJOR)
mark_as_advanced(NIFTK_VERSION_MINOR)
mark_as_advanced(NIFTK_VERSION_PATCH)

######################################################################
# Set the minimum CMake version.
######################################################################
if(WIN32)
  cmake_minimum_required(VERSION 2.8.11.2)
else()
  cmake_minimum_required(VERSION 2.8.9)
endif()

######################################################################
# Setting supported build types. Should ONLY be Debug or Release.
######################################################################

if(DEFINED CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Valid options are Debug/Release")
else()
   set(CMAKE_BUILD_TYPE Release CACHE STRING "Setting build type to Release (default). Valid options: Debug/Release.")
endif()

if (NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Release"))
  message(FATAL_ERROR "Build type ${CMAKE_BUILD_TYPE} is not supported.")
endif()

##################################################################################
# Set some CMake Policies.
# See http://cmake.org/cmake/help/cmake-2-8-docs.html#section_Policies for details
##################################################################################

set(project_policies
  CMP0001 # NEW: CMAKE_BACKWARDS_COMPATIBILITY should no longer be used.
  CMP0002 # NEW: Logical target names must be globally unique.
  CMP0003 # NEW: Libraries linked via full path no longer produce linker search paths.
  CMP0004 # NEW: Libraries linked may NOT have leading or trailing whitespace.
  CMP0005 # NEW: Preprocessor definition values are now escaped automatically.
  CMP0006 # NEW: Installing MACOSX_BUNDLE targets requires a BUNDLE DESTINATION.
  CMP0007 # NEW: List command no longer ignores empty elements.
  CMP0008 # NEW: Libraries linked by full-path must have a valid library file name.
  CMP0009 # NEW: FILE GLOB_RECURSE calls should not follow symlinks by default.
  CMP0010 # NEW: Bad variable reference syntax is an error.
  CMP0011 # NEW: Included scripts do automatic cmake_policy PUSH and POP.
  CMP0012 # NEW: if() recognizes numbers and boolean constants.
  CMP0013 # NEW: Duplicate binary directories are not allowed.
  CMP0014 # NEW: Input directories must have CMakeLists.txt
  )
foreach(policy ${project_policies})
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

######################################################################
# We have a super-build option. (Terminology comes from MITK/CTK).
######################################################################

option(BUILD_SUPERBUILD "Build NIFTK and the projects it depends on via SuperBuild.cmake." ON)

if(BUILD_SUPERBUILD)
  project(NIFTK-SUPERBUILD)
else(BUILD_SUPERBUILD)
  project(NIFTK)
endif(BUILD_SUPERBUILD)

######################################################################
# Configure GUI options from the choice of applications.
# The user cannot be expected to know the component parts of apps.
######################################################################

include("${CMAKE_CURRENT_SOURCE_DIR}/Code/Gui/MITK/Apps/Apps.cmake")
set(BUILD_GUI OFF)
set(BUILD_IGI OFF)
set(BUILD_MIDAS OFF)

# Please leave this off, as we can then JUST build command line apps, without
# filling the CMakeCache.txt with GUI related variables which will only confuse
# when someone subsequently tries to switch all the GUIs off.
option(NIFTK_BUILD_ALL_APPS "Build all NifTK Graphical User Interfaces" OFF)

# Check each GUI application.
foreach(NIFTK_APP ${NIFTK_APPS})

  # extract option_name
  string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
  set(target_info_list ${target_info})
  list(GET target_info_list 1 option_name)
  list(GET target_info_list 0 app_name)

  if(${option_name} OR NIFTK_BUILD_ALL_APPS)

    set(BUILD_GUI ON)

    if (${app_name} STREQUAL "NiftyIGI")
      set(BUILD_IGI ON)
    endif()

    if (${app_name} STREQUAL "NiftyView")
      #set(BUILD_MIDAS ON)
      #set(BUILD_IGI ON)
    endif()

    if (${app_name} STREQUAL "NiftyMIDAS")
      set(BUILD_MIDAS ON)
    endif()

  endif()

endforeach()

######################################################################
# Options. These are set up front, so are available when configuring
# the SuperBuild, and hence they must also be passed to the normal
# build. So, look in CMake/Superbuild.cmake to see where they are
# passed to the main build of NifTK when doing the Superbuild.
######################################################################
option(BUILD_COMMAND_LINE_PROGRAMS "Build command line applications in NifTK/Code/Applications." ON)
option(BUILD_COMMAND_LINE_SCRIPTS "Build command line scripts in NifTK/Code/Scripts." ON)
option(BUILD_CAMINO "Build camino." OFF)
option(BUILD_NIFTYREG "Build NiftyReg." ON)
option(BUILD_NIFTYREC "Build NiftyRec." OFF)
option(BUILD_NIFTYSIM "Build NiftySim." ON)
option(BUILD_NIFTYSEG "Build NiftySeg." ON)
option(BUILD_PCL "Build PCL - Experimental." OFF)
option(BUILD_RTK "Build RTK - Experimental." OFF)
option(BUILD_ITKFFTW "Build ITK's FFTW - Not Supported." OFF)
option(BUILD_PROTOTYPE "Build prototype code in NifTK/Prototype." OFF)
option(BUILD_SHARED_LIBS "Build NifTK with shared libraries." ON)
option(BUILD_TESTING "Build Unit tests." OFF)
option(BUILD_SLS_TESTING "Build Smart Liver Surgery additional tests." OFF)
option(BUILD_UTILITIES "Build NifTK/Utilities." OFF)
option(NIFTK_CHECK_COVERAGE "Enable/Disable code coverage checking." OFF)
# the console window is normally shown only for debug builds. but can be explicitly enabled.
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(NIFTK_SHOW_CONSOLE_WINDOW_VAL ON)
else()
  set(NIFTK_SHOW_CONSOLE_WINDOW_VAL OFF)
endif()
option(NIFTK_SHOW_CONSOLE_WINDOW "Use this to enable or disable the console window when starting GUI Applications on Windows" ${NIFTK_SHOW_CONSOLE_WINDOW_VAL})
option(NIFTK_BUILD_ALL_PLUGINS "Build all NIFTK plugins" OFF)
option(NIFTK_GENERATE_DOXYGEN_HELP "Use this to generate Doxygen help in GUI" OFF)
option(NIFTK_VERBOSE_COMPILER_WARNINGS "Add in all the warning flags that MITK does" OFF)
option(NIFTK_USE_FFTW "Use FFTW. Warning FFTW is GPL, so for binary external releases we can't use this. This is used for Kelvin's Fluid registration implementation." OFF)
option(NIFTK_USE_CUDA "Use CUDA. Experimental. " OFF)
option(NIFTK_DELAYLOAD_CUDA "(Windows only) If enabled the CUDA DLLs will be a soft delay-load dependency and NifTK can run without them." OFF)
option(NIFTK_NIFTYLINK_DEV "Build NiftyLink development branch" OFF)
option(NIFTK_RESOLVE_COMMAND_LINE_DEPENDENCIES "For make install, for command line apps, we compute all library dependencies and install them" ON)

######################################################################
# Meshing module-related checks and options
######################################################################
include (CheckIncludeFiles)

# superbuild for cgal is only supported on non-windows.
# so for windows, this check here does not apply (and in fact prevents us
# from passing in an existing cgal installation).
if(NOT MSVC)
  CHECK_INCLUDE_FILES(mpfr.h NIFTK_MESHING_CAN_BUILD)
  if (NIFTK_MESHING_CAN_BUILD)
    option(BUILD_MESHING "Build meshing?" OFF)
  endif()
endif()

######################################################################
# Variables that get compiled into C++ code.
######################################################################
set(NIFTK_PLATFORM "NifTK" CACHE STRING "Full name of platform." FORCE )
set(NIFTK_LICENSE_SHORT_STRING "Not licensed for use outside of UCL." CACHE STRING "Short description of license" FORCE )
set(NIFTK_COPYRIGHT "Copyright (C) 2008-2014 University College London (UCL). All rights reserved." CACHE STRING "Copyright string." FORCE )
set(NIFTK_ORIGIN_URL "http://cmic.cs.ucl.ac.uk/" CACHE STRING "URL of originating institution." FORCE )
set(NIFTK_ORIGIN_SHORT_TEXT "CMIC" CACHE STRING "Short name of originating institution." FORCE)
set(NIFTK_ORIGIN_LONG_TEXT "Centre For Medical Image Computing" CACHE STRING "Full name of originating institution." FORCE)
set(NIFTK_USER_CONTACT "https://www.mailinglists.ucl.ac.uk/mailman/listinfo/niftk-users" CACHE STRING "Contact address for users." FORCE )
if(WIN32)
  set(NIFTK_BASE_NAME "NifTK" CACHE STRING "Base name for installation folder. Windows convention is upper case letters" FORCE )
else(WIN32)
  set(NIFTK_BASE_NAME "niftk" CACHE STRING "Base name for installation folder. Unix convention is lower case letters." FORCE )
endif(WIN32)
set(NIFTK_VERSION_STRING "${NIFTK_VERSION_MAJOR}.${NIFTK_VERSION_MINOR}.${NIFTK_VERSION_PATCH}" CACHE STRING "String to describe fully named version" FORCE)
set(NIFTK_DEPLOY_NAME "${NIFTK_BASE_NAME}-${NIFTK_VERSION_STRING}" CACHE STRING "String to describe deployed name" FORCE)
set(NIFTK_ARCHTYPE_DATA "git" CACHE STRING "Archive type NifTKData (git or tar)" FORCE)
set(NIFTK_VERSION_DATA "cefeb2364e" CACHE STRING "Version of NifTKData" FORCE)
set(NIFTK_VERSION_Boost "1_56_0" CACHE STRING "Version of Boost" FORCE)
set(NIFTK_VERSION_GDCM "2.4.1" CACHE STRING "Version of GDCM" FORCE)
set(NIFTK_VERSION_DCMTK "3.6.1_20121102" CACHE STRING "Version of DCMTK" FORCE)
set(NIFTK_VERSION_ITK "4.5.1-3e550bf8" CACHE STRING "Version of ITK" FORCE)
set(NIFTK_VERSION_RTK "196aec7d3b" CACHE STRING "Version of RTK" FORCE)
set(NIFTK_VERSION_VTK "6.1.0+74f4888" CACHE STRING "Version of VTK" FORCE)
# Note: If the CTK version changes, then you either clear the plugin cache
# or change the deploy path by changing the patch level.
set(NIFTK_VERSION_CTK "9331130fe3" CACHE STRING "Version of CTK" FORCE)
set(NIFTK_VERSION_OPENCV "2.4.8.2" CACHE STRING "Version of OpenCV" FORCE)
set(NIFTK_VERSION_MITK "9afda08263" CACHE STRING "Version of MITK" FORCE)
set(NIFTK_VERSION_NIFTYLINK "d1680fd98a" CACHE STRING "Version of NiftyLink" FORCE)
set(NIFTK_VERSION_NIFTYREG "7ac1f44ab7" CACHE STRING "Version of NiftyReg" FORCE)
set(NIFTK_VERSION_NIFTYSEG "606b45d145" CACHE STRING "Version of NiftySeg" FORCE)
set(NIFTK_VERSION_NIFTYREC "14" CACHE STRING "Version of NiftyRec" FORCE)
set(NIFTK_VERSION_NIFTYSIM "29132ae860" CACHE STRING "Version of NiftySim" FORCE)
set(NIFTK_VERSION_CGAL "4.4-patched" CACHE STRING "Version of CGAL" FORCE)
set(NIFTK_VERSION_ARUCO "1.2.4" CACHE STRING "Version of aruco" FORCE)
set(NIFTK_VERSION_SEM "11ea15b81e" CACHE STRING "Version of Slicer Execution Module package" FORCE)
set(NIFTK_VERSION_CAMINO "4612bee5fa" CACHE STRING "Version of Camino" FORCE)
set(NIFTK_VERSION_EIGEN "3.2.2" CACHE STRING "Version of EIGEN" FORCE)
set(NIFTK_VERSION_APRILTAGS "3c6af59723" CACHE STRING "Version of AprilTags" FORCE)
set(NIFTK_VERSION_FLANN "1.8.4.1" CACHE STRING "Version of FLANN" FORCE)
set(NIFTK_VERSION_PCL "c2203fa60a" CACHE STRING "Version of PCL" FORCE)
set(NIFTK_VERSION_qRestAPI "5f3a03b15d" CACHE STRING "Version of qRestAPI" FORCE) 
set(NIFTK_LOCATION_DATA_GIT "git@cmiclab.cs.ucl.ac.uk:CMIC/NifTKData" CACHE STRING "Version of NifTKData in git" FORCE)
set(NIFTK_LOCATION_DATA_TAR "http://cmic.cs.ucl.ac.uk/platform/dependencies/NifTKData-${NIFTK_VERSION_DATA_TAR}.tar.gz" CACHE STRING "Location of NifTKData" FORCE)
set(NIFTK_LOCATION_Boost "http://cmic.cs.ucl.ac.uk/platform/dependencies/boost_${NIFTK_VERSION_Boost}.tar.gz" CACHE STRING "Location of Boost" FORCE)
set(NIFTK_LOCATION_GDCM "http://cmic.cs.ucl.ac.uk/platform/dependencies/gdcm-${NIFTK_VERSION_GDCM}.tar.gz" CACHE STRING "Location of GDCM" FORCE)
set(NIFTK_LOCATION_DCMTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/dcmtk-${NIFTK_VERSION_DCMTK}.tar.gz" CACHE STRING "Location of DCMTK" FORCE)
set(NIFTK_LOCATION_ITK "http://cmic.cs.ucl.ac.uk/platform/dependencies/InsightToolkit-${NIFTK_VERSION_ITK}.tar.gz" CACHE STRING "Location of ITK" FORCE)
set(NIFTK_LOCATION_RTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/NifTK-RTK-${NIFTK_VERSION_RTK}.tar.gz" CACHE STRING "Location of RTK" FORCE)
set(NIFTK_LOCATION_VTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/VTK-${NIFTK_VERSION_VTK}.tar.gz" CACHE STRING "Location of VTK" FORCE)
set(NIFTK_LOCATION_CTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/commontk-CTK-${NIFTK_VERSION_CTK}.tar.gz" CACHE STRING "Location of CTK" FORCE)
set(NIFTK_LOCATION_OPENCV "http://cmic.cs.ucl.ac.uk/platform/dependencies/OpenCV-${NIFTK_VERSION_OPENCV}.tar.gz" CACHE STRING "Location of OpenCV" FORCE)
set(NIFTK_LOCATION_MITK "http://cmic.cs.ucl.ac.uk/platform/dependencies/NifTK-MITK-${NIFTK_VERSION_MITK}.tar.gz" CACHE STRING "Location of MITK" FORCE)
set(NIFTK_LOCATION_NIFTYLINK_TARBALL "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftyLink-${NIFTK_VERSION_NIFTYLINK}.tar.gz" CACHE STRING "Location of NiftyLink tarball" FORCE)
set(NIFTK_LOCATION_NIFTYLINK_REPOSITORY "git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyLink" CACHE STRING "Location of NiftyLink repository" FORCE)
set(NIFTK_LOCATION_PROTOTYPE "https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NiftyPrototype" CACHE STRING  "Location of prototype code (internal access only)")
set(NIFTK_LOCATION_NIFTYREG "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftyReg-${NIFTK_VERSION_NIFTYREG}.tar.gz" CACHE STRING  "Location of NiftyReg" FORCE)
set(NIFTK_LOCATION_NIFTYSEG "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftySeg-${NIFTK_VERSION_NIFTYSEG}.tar.gz" CACHE STRING  "Location of NiftySeg" FORCE)
set(NIFTK_LOCATION_NIFTYREC "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftyRec-${NIFTK_VERSION_NIFTYREC}.tar.gz" CACHE STRING  "Location of NiftyRec" FORCE)
set(NIFTK_LOCATION_NIFTYSIM "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftySim-${NIFTK_VERSION_NIFTYSIM}.tar.gz" CACHE STRING  "Location of NiftySim" FORCE)
set(NIFTK_LOCATION_CGAL "http://cmic.cs.ucl.ac.uk/platform/dependencies/CGAL-${NIFTK_VERSION_CGAL}.tar.gz" CACHE STRING  "Location of CGAL" FORCE)
set(NIFTK_LOCATION_SEM "http://cmic.cs.ucl.ac.uk/platform/dependencies/Slicer-SlicerExecutionModel-${NIFTK_VERSION_SEM}.tar.gz" CACHE STRING  "Location of Slicer Execution Module package" FORCE)
set(NIFTK_LOCATION_ARUCO "http://cmic.cs.ucl.ac.uk/platform/dependencies/aruco-${NIFTK_VERSION_ARUCO}.tar.gz" CACHE STRING  "Location of Aruco package" FORCE)
set(NIFTK_LOCATION_CAMINO "http://cmic.cs.ucl.ac.uk/platform/dependencies/camino-${NIFTK_VERSION_CAMINO}.tar.gz" CACHE STRING "Location of Camino package" FORCE)
set(NIFTK_LOCATION_EIGEN "http://cmic.cs.ucl.ac.uk/platform/dependencies/Eigen-${NIFTK_VERSION_EIGEN}-headers-only.tar.gz" CACHE STRING "Location of Eigen" FORCE)
set(NIFTK_LOCATION_APRILTAGS "http://cmic.cs.ucl.ac.uk/platform/dependencies/NifTK-apriltags-${NIFTK_VERSION_APRILTAGS}.tar.gz" CACHE STRING "Location of AprilTags" FORCE)
set(NIFTK_LOCATION_FLANN "http://cmic.cs.ucl.ac.uk/platform/dependencies/flann-${NIFTK_VERSION_FLANN}-src.tar.gz" CACHE STRING "Location of FLANN" FORCE)
set(NIFTK_LOCATION_PCL "http://cmic.cs.ucl.ac.uk/platform/dependencies/PointCloudLibrary-pcl-${NIFTK_VERSION_PCL}.tar.gz" CACHE STRING "Location of PCL" FORCE)
set(NIFTK_LOCATION_qRestAPI "http://cmic.cs.ucl.ac.uk/platform/dependencies/commontk-qRestAPI-${NIFTK_VERSION_qRestAPI}.tar.gz" CACHE STRING "Location of qRestAPI" FORCE)

######################################################################
# Hide these variables from the user, unless they are 'advanced' :-)
######################################################################
mark_as_advanced(NIFTK_PLATFORM)
mark_as_advanced(NIFTK_LICENSE_SHORT_STRING)
mark_as_advanced(NIFTK_COPYRIGHT)
mark_as_advanced(NIFTK_ORIGIN_URL)
mark_as_advanced(NIFTK_ORIGIN_SHORT_TEXT)
mark_as_advanced(NIFTK_ORIGIN_LONG_TEXT)
mark_as_advanced(NIFTK_HELP_URL_TEXT)
mark_as_advanced(NIFTK_USER_CONTACT)
mark_as_advanced(NIFTK_BASE_NAME)
mark_as_advanced(NIFTK_VERSION_STRING)
mark_as_advanced(NIFTK_DEPLOY_NAME)
mark_as_advanced(NIFTK_SHOW_CONSOLE_WINDOW)
mark_as_advanced(NIFTK_BUILD_ALL_PLUGINS)
mark_as_advanced(NIFTK_CHECK_COVERAGE)
mark_as_advanced(NIFTK_VERBOSE_COMPILER_WARNINGS)
mark_as_advanced(NIFTK_GENERATE_DOXYGEN_HELP)
mark_as_advanced(NIFTK_ARCHTYPE_DATA)
mark_as_advanced(NIFTK_VERSION_DATA)
mark_as_advanced(NIFTK_VERSION_Boost)
mark_as_advanced(NIFTK_VERSION_GDCM)
mark_as_advanced(NIFTK_VERSION_DCMTK)
mark_as_advanced(NIFTK_VERSION_ITK)
mark_as_advanced(NIFTK_VERSION_VTK)
mark_as_advanced(NIFTK_VERSION_CTK)
mark_as_advanced(NIFTK_VERSION_OPENCV)
mark_as_advanced(NIFTK_VERSION_MITK)
mark_as_advanced(NIFTK_VERSION_NIFTYLINK)
mark_as_advanced(NIFTK_VERSION_NIFTYREG)
mark_as_advanced(NIFTK_VERSION_NIFTYSEG)
mark_as_advanced(NIFTK_VERSION_NIFTYREC)
mark_as_advanced(NIFTK_VERSION_NIFTYSIM)
mark_as_advanced(NIFTK_VERSION_CGAL)
mark_as_advanced(NIFTK_VERSION_SEM)
mark_as_advanced(NIFTK_VERSION_ARUCO)
mark_as_advanced(NIFTK_VERSION_qRestAPI)
mark_as_advanced(NIFTK_LOCATION_DATA_GIT)
mark_as_advanced(NIFTK_LOCATION_DATA_SVN)
mark_as_advanced(NIFTK_LOCATION_DATA_TAR)
mark_as_advanced(NIFTK_LOCATION_Boost)
mark_as_advanced(NIFTK_LOCATION_GDCM)
mark_as_advanced(NIFTK_LOCATION_DCMTK)
mark_as_advanced(NIFTK_LOCATION_ITK)
mark_as_advanced(NIFTK_LOCATION_VTK)
mark_as_advanced(NIFTK_LOCATION_CTK)
mark_as_advanced(NIFTK_LOCATION_OPENCV)
mark_as_advanced(NIFTK_LOCATION_MITK)
mark_as_advanced(NIFTK_LOCATION_NIFTYLINK_TARBALL)
mark_as_advanced(NIFTK_LOCATION_NIFTYLINK_REPOSITORY)
mark_as_advanced(NIFTK_LOCATION_PROTOTYPE)
mark_as_advanced(NIFTK_LOCATION_NIFTYREG)
mark_as_advanced(NIFTK_LOCATION_NIFTYSEG)
mark_as_advanced(NIFTK_LOCATION_NIFTYREC)
mark_as_advanced(NIFTK_LOCATION_NIFTYSIM)
mark_as_advanced(NIFTK_LOCATION_CGAL)
mark_as_advanced(NIFTK_LOCATION_SEM)
mark_as_advanced(NIFTK_LOCATION_ARUCO)
mark_as_advanced(NIFTK_LOCATION_qRestAPI)
mark_as_advanced(NIFTK_USE_FFTW)
mark_as_advanced(NIFTK_USE_CUDA)
mark_as_advanced(NIFTK_DELAYLOAD_CUDA)
mark_as_advanced(BUILD_SLS_TESTING)
mark_as_advanced(NIFTK_NIFTYLINK_DEV)
mark_as_advanced(BUILD_SHARED_LIBS)
mark_as_advanced(NIFTK_RESOLVE_COMMAND_LINE_DEPENDENCIES)
mark_as_advanced(BUILD_CAMINO)
mark_as_advanced(NIFTK_LOCATION_CAMINO)
mark_as_advanced(NIFTK_VERSION_CAMINO)
mark_as_advanced(NIFTK_LOCATION_EIGEN)
mark_as_advanced(NIFTK_VERSION_EIGEN)
mark_as_advanced(NIFTK_LOCATION_APRILTAGS)
mark_as_advanced(NIFTK_VERSION_APRILTAGS)
mark_as_advanced(NIFTK_VERSION_FLANN)
mark_as_advanced(NIFTK_LOCATION_FLANN)
mark_as_advanced(BUILD_PCL)
mark_as_advanced(NIFTK_VERSION_PCL)
mark_as_advanced(NIFTK_LOCATION_PCL)
mark_as_advanced(BUILD_RTK)
mark_as_advanced(NIFTK_VERSION_RTK)
mark_as_advanced(NIFTK_LOCATION_RTK)
mark_as_advanced(BUILD_ITKFFTW)

# PCL's cmake bits are extremely chatty and fill the cmake window/log with noise.
if(BUILD_PCL)
  set(PCL_FIND_QUIETLY ON)
endif()

######################################################################
# Make sure Git is available.
######################################################################
find_package(Git REQUIRED)
if (WIN32)
  set(GITCOMMAND ${GIT_EXECUTABLE})
endif()

######################################################################
# Setup the path to load CMake macros, and extra CMake files.
######################################################################
set(CMAKE_MODULE_PATH
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/CMake
    ${CMAKE_SOURCE_DIR}/CMake/CMakeExternals
    ${CMAKE_SOURCE_DIR}/CMake/cuda
    ${CMAKE_MODULE_PATH}
   )

######################################################################
# Add in any functions/macros.
######################################################################
include(mitkMacroEmptyExternalProject)
include(mitkFunctionGetVersion)
include(mitkFunctionGetGccVersion)
include(niftkMacroGetGitDateTime)
include(niftkMacroGetGitBranch)
include(niftkMacroBuildCLI)
include(niftkMacroGenerateCLIScript)
include(niftkMacroInstallCLIScript)
include(niftkMacroCreateCommandLineApplication)
include(niftkMacroCreateCommandLineScript)
include(niftkMacroCreateGuiApplication)
include(niftkMacroFixupCommandLineApps)
include(niftkMacroForcePluginCacheValue)
include(CMakePackageConfigHelpers)

######################################################################
# Test for some required system information. This came from old
# ITK CMake scripts, so I (Matt) am unsure if we still need it.
######################################################################
include(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityC.cmake)
include(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityCXX.cmake)

######################################################################
# Setting build name based on local system details
######################################################################

mitkFunctionGetVersion(${CMAKE_SOURCE_DIR} NIFTK)
niftkMacroGetGitDateTime(${CMAKE_SOURCE_DIR} NIFTK)
niftkMacroGetGitBranch(${CMAKE_SOURCE_DIR} NIFTK)

if(CMAKE_GENERATOR MATCHES Make AND NOT (CMAKE_GENERATOR MATCHES NMake) )

  find_program(UNAME NAMES uname)
  mark_as_advanced(UNAME)

  macro(getuname name flag)
    exec_program("${UNAME}" ARGS "${flag}" OUTPUT_VARIABLE "${name}")
  endmacro(getuname)

  getuname(osname -s)
  getuname(cpu    -m)

  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CompilerName "gcc")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CompilerName "clang")
  else()
    set(CompilerName "${CMAKE_CXX_COMPILER_ID}")
  endif()

  set(CTBN "${osname}-${cpu}-${CompilerName}-${CMAKE_CXX_COMPILER_VERSION}-${NIFTK_BRANCH_NAME}-${NIFTK_REVISION_SHORTID}")

else()

  set(CTBN "${CMAKE_GENERATOR}-${NIFTK_BRANCH_NAME}-${NIFTK_REVISION_SHORTID}")

endif()

# append a short release/debug tag, so we know which one is which on the dashboard.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CTBN "${CTBN}-Dbg")
else()
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CTBN "${CTBN}-Rel")
else()
  # this should not happen. further above we check for Debug and Release and fail if it's neither.
  set(CTBN "${CTBN}-${CMAKE_BUILD_TYPE}")
endif()
endif()

string(REPLACE "\n" "" CTBN ${CTBN})
set(BUILDNAME ${CTBN} CACHE STRING "${CTBN}" FORCE)
set(CTEST_BUILD_NAME ${CTBN} CACHE STRING "${CTBN}" FORCE)
mark_as_advanced(BUILDNAME)
mark_as_advanced(CTEST_BUILD_NAME)

message("NifTK branch=${NIFTK_BRANCH_NAME}")
message("NifTK version=${NIFTK_REVISION_SHORTID}")
message("NifTK date=${NIFTK_DATE_TIME}")
message("NifTK build=${CTEST_BUILD_NAME}")

# Putting this here, so all subfolders have access to it.
if(CMAKE_COMPILER_IS_GNUCXX)
  mitkFunctionGetGccVersion(${CMAKE_CXX_COMPILER} GCC_VERSION)
  message("NifTK gcc=${GCC_VERSION}")
endif()

######################################################################
# Configure Dart testing support.  This should be done before any
# message(FATAL_ERROR ...) commands are invoked.
######################################################################
include(${CMAKE_ROOT}/Modules/Dart.cmake)
mark_as_advanced(TCL_TCLSH DART_ROOT)
enable_testing()
if(BUILD_TESTING)
  configure_file(${CMAKE_SOURCE_DIR}/Doc/Images/NifTKLogo.gif
                 ${CMAKE_BINARY_DIR}/Testing/HTML/TestingResults/Icons/Logo.gif
                 COPYONLY IMMEDIATE)
  set(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  mark_as_advanced(BUILDNAME)

  # Setup file for setting custom ctest vars
  configure_file(CMake/CTestCustom.cmake.in ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)
endif(BUILD_TESTING)

######################################################################
# Platform checks:
# Check availability of some standard API functions
######################################################################
include(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
include(${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake)
CHECK_FUNCTION_EXISTS(mkstemps HAVE_MKSTEMPS)
CHECK_INCLUDE_FILES("sys/socket.h" HAVE_SYS_SOCKET_H)
if(HAVE_SYS_SOCKET_H)
  add_definitions("-DHAVE_SYS_SOCKET_H")
endif()

###########################################################################
# Set these compiler flags early, so it can be applied to all dependencies.
###########################################################################

set(NIFTK_ADDITIONAL_C_FLAGS "" CACHE STRING "Additional C Flags for NifTK only")
mark_as_advanced(NIFTK_ADDITIONAL_C_FLAGS)
set(NIFTK_ADDITIONAL_CXX_FLAGS "" CACHE STRING "Additional CXX Flags for NifTK only")
mark_as_advanced(NIFTK_ADDITIONAL_CXX_FLAGS)

set(MITK_ADDITIONAL_C_FLAGS "" CACHE STRING "Additional C Flags for MITK only")
mark_as_advanced(MITK_ADDITIONAL_C_FLAGS)
set(MITK_ADDITIONAL_CXX_FLAGS "" CACHE STRING "Additional CXX Flags for MITK only")
mark_as_advanced(MITK_ADDITIONAL_CXX_FLAGS)
set(CTK_ADDITIONAL_C_FLAGS "" CACHE STRING "Additional C Flags for CTK only")
mark_as_advanced(CTK_ADDITIONAL_C_FLAGS)
set(CTK_ADDITIONAL_CXX_FLAGS "" CACHE STRING "Additional CXX Flags for CTK only")
mark_as_advanced(CTK_ADDITIONAL_CXX_FLAGS)
set(OPENCV_ADDITIONAL_C_FLAGS "" CACHE STRING "Additional C Flags for OpenCV only")
mark_as_advanced(OPENCV_ADDITIONAL_C_FLAGS)
set(OPENCV_ADDITIONAL_CXX_FLAGS "" CACHE STRING "Additional CXX Flags for OpenCV only")
mark_as_advanced(OPENCV_ADDITIONAL_CXX_FLAGS)
set(OPENCV_WITH_CUDA OFF CACHE BOOL "Build OpenCV with CUDA support (takes a long time)")
mark_as_advanced(OPENCV_WITH_CUDA)
set(OPENCV_WITH_FFMPEG OFF CACHE BOOL "Build OpenCV with FFMPEG support")
mark_as_advanced(OPENCV_WITH_FFMPEG)
set(OPENCV_WITH_NONFREE OFF CACHE BOOL "Build OpenCV with nonfree library")
mark_as_advanced(OPENCV_WITH_NONFREE)

set(NIFTYSIM_USE_CUDA OFF CACHE BOOL "Build NiftySim with CUDA support")
mark_as_advanced(NIFTYSIM_USE_CUDA)


########################################################################
# Set NIFTK_INSTALL_PREFIX, and NIFTK_LINK_PREFIX, used in scripts.
# Note: DONT try changing CMAKE_INSTALL_PREFIX. Think of it as reserved.
########################################################################

get_filename_component(NIFTK_LINK_PREFIX ${CMAKE_INSTALL_PREFIX} PATH)
set(NIFTK_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

######################################################################
# Sanity Checks before doing Superbuild.
######################################################################
if(BUILD_GUI)
  if(NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "If you wan't to build a GUI, you must use dynamic linking, turn BUILD_SHARED to ON.")
  endif(NOT BUILD_SHARED_LIBS)
endif(BUILD_GUI)

if(BUILD_GUI)
  set(NIFTK_GENERATE_DOXYGEN_HELP ON)
endif(BUILD_GUI)

if(BUILD_ITKFFTW)
  message(WARNING "You have chosen to compile FFTW. This means the binary code, and all executables linked against it will be covered by GPL. This is not supported by NifTK.")
endif(BUILD_ITKFFTW)

########################################################################
# Simplest if we list ALL our include directories here.
# We need to reference the CMAKE_BINARY_DIR, as we generate stuff there.
########################################################################
set(NIFTK_INCLUDE_DIRS_BUILD_TREE
  ${CMAKE_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/Config
  ${CMAKE_BINARY_DIR}/Code
  ${CMAKE_SOURCE_DIR}/Code/Libs/Common
  ${CMAKE_SOURCE_DIR}/Code/Libs/Common/Exceptions
  ${CMAKE_SOURCE_DIR}/Code/Libs/MIDAS
  ${CMAKE_SOURCE_DIR}/Code/Libs/VTK
  ${CMAKE_SOURCE_DIR}/Code/Libs/VTK/Common
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Common
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/BasicFilters
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/IO
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Numerics
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Pipelines
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/BoundaryShiftIntegral
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/BreastCancerImaging
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/CorticalThickness
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Metrics
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Methods
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Transforms
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Optimizers
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Commands
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Construction
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Constraints
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Filters
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Methods
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Optimizers
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Projection
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Metrics
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Commands
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/xRay
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Transforms
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Segmentation
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Segmentation/MIDASMorphologicalEditor
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Segmentation/MIDASIrregularVolumeEditor
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITK/VesselToolbox
  ${CMAKE_SOURCE_DIR}/Code/Libs/ITKVTK/IO
  ${CMAKE_SOURCE_DIR}/Testing/Code/
  ${CMAKE_SOURCE_DIR}/Testing/Code/Libs/ITK/RegistrationToolbox
)
if (BUILD_MESHING)
  set(NIFTK_INCLUDE_DIRS_BUILD_TREE
    ${NIFTK_INCLUDE_DIRS_BUILD_TREE}
    ${CMAKE_SOURCE_DIR}/Code/Libs/Common/Meshing
  )
endif (BUILD_MESHING)

######################################################################
# Find Qt. We need to check for Qt first, before we go ahead and do
# the Superbuild, because VTK needs to know if we have Qt.
######################################################################

set(DESIRED_QT_VERSION 4 CACHE STRING "Pick a version of Qt to use: 4 or 5")

# We ALWAYS need Qt. This is because MITK always needs it.
if(BUILD_IGI)
  find_package(Qt4 COMPONENTS QtCore QtGui QtXml QtNetwork REQUIRED)
else(BUILD_IGI)
  find_package(Qt4 COMPONENTS QtCore QtGui QtXml REQUIRED)
endif(BUILD_IGI)

if(QT_FOUND)
  set(QT_USE_QTXML 1)
  set(QT_USE_QTXMLPATTERNS 1)
  set(QT_USE_PHONON 0)
  include(${QT_USE_FILE})
  if(BUILD_SUPERBUILD)
    set(VTK_QT_ARGS
      -DVTK_QT_VERSION:STRING=@DESIRED_QT_VERSION@
      -DVTK_USE_GUISUPPORT:BOOL=ON
      -DVTK_USE_QVTK_QTOPENGL:BOOL=OFF
      -DVTK_USE_QT:BOOL=ON
      -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
    )
  endif(BUILD_SUPERBUILD)
else()
 message(FATAL "Qt is mandatory, but QT_FOUND is false.")
endif()
 
######################################################################
# Find CUDA: Required by the Nifty packages
######################################################################

if(NIFTK_USE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${CUDA_TOOLKIT_INCLUDE})
    add_definitions(-D_USE_CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
    message("Found CUDA")
  else(CUDA_FOUND)
    message("Didn't find CUDA")
  endif(CUDA_FOUND)
endif(NIFTK_USE_CUDA)

######################################################################
# Find NVidia API: Required for NVidia SDI pipeline in IGI GUI
######################################################################
find_package(NVidiaAPI)
if(NVAPI_FOUND)
  message("Found NVidia API")
  if (CUDA_FOUND AND NIFTK_USE_CUDA)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NVAPI_INCLUDE_DIR})
    add_definitions(-D_USE_NVAPI)
  else()
    message("NVidia API disabled: need CUDA. Try NIFTK_USE_CUDA.")
    set(NVAPI_FOUND 0)
  endif()
else()
  message("Did not find NVidia API")
endif()

######################################################################
# Now, if required, do the SuperBuild
# If we are doing SuperBuild
#   We configure up to this point (see the return() statement)
#   and then we call SuperBuild.cmake, which builds all the
#   dependencies as CMake ExternalProjects, and then also builds
#   NifTK as an ExternalProject. However instead of downloading
#   a tar file, you set the SOURCE_DIR to be THIS project, and force
#   the BUILD_SUPERBUILD flag to be off (to avoid infinite loop).
#
# If we are NOT doing superbuild, then the next statement has no
# effect, and the build goes on the same as before, as in version
# NifTK 2.2.0 and earlier.
######################################################################

if(BUILD_SUPERBUILD)
  include("CMake/SuperBuild.cmake")
  return()
endif(BUILD_SUPERBUILD)

######################################################################
# End of SuperBuild. Print out where the source and binary folders
# are, just to make it really explicit... well, explicit to the user
# that bothers to read these messages! :-)
######################################################################

message("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message("CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

######################################################################
# Add additional entries to the CMAKE_MODULE_PATH, now all externals
# are already built.
######################################################################
set(CMAKE_MODULE_PATH
    ${CMAKE_BINARY_DIR}/../aruco-install/lib/cmake
    ${CMAKE_MODULE_PATH}
   )

######################################################################
# Check each GUI application to set up MACOSX_BUNDLE_NAMES
# as libraries are copied into each bundle.
######################################################################

if(APPLE)

  set(MACOSX_BUNDLE_NAMES)
  include("${CMAKE_CURRENT_SOURCE_DIR}/Code/Gui/MITK/Apps/Apps.cmake")

  foreach(NIFTK_APP ${NIFTK_APPS})

    # extract option_name
    string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
    set(target_info_list ${target_info})
    list(GET target_info_list 1 option_name)
    list(GET target_info_list 0 app_name)

    if(${option_name} OR NIFTK_BUILD_ALL_APPS)
      set(MACOSX_BUNDLE_NAMES ${MACOSX_BUNDLE_NAMES} ${app_name})
    endif()

  endforeach()

endif()

######################################################################
# Configure CMake files before we need to use them in the FIND_XXX.
#
# So, the variables NIFTK_BOOSTINSTALL etc.
# should be passed in on cmake command line, or set up using ccmake.
######################################################################

configure_file(${CMAKE_SOURCE_DIR}/CMake/FindFFTW.cmake ${CMAKE_BINARY_DIR}/FindFFTW.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftyReg.cmake ${CMAKE_BINARY_DIR}/FindNiftyReg.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftyRec.cmake ${CMAKE_BINARY_DIR}/FindNiftyRec.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftySim.cmake ${CMAKE_BINARY_DIR}/FindNiftySim.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftySeg.cmake ${CMAKE_BINARY_DIR}/FindNiftySeg.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindEigen.cmake ${CMAKE_BINARY_DIR}/FindEigen.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/Findapriltags.cmake ${CMAKE_BINARY_DIR}/Findapriltags.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/mitkFunctionGetGccVersion.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/mitkFunctionGetGccVersion.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/mitkFunctionGetVersion.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/mitkFunctionGetVersion.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/mitkMacroEmptyExternalProject.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/mitkMacroEmptyExternalProject.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroBuildCLI.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroBuildCLI.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroCreateCommandLineApplication.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroCreateCommandLineApplication.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroCreateCommandLineScript.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroCreateCommandLineScript.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroCreateGuiApplication.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroCreateGuiApplication.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroGenerateCLIScript.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroGenerateCLIScript.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroGetGitBranch.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroGetGitBranch.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroGetGitDateTime.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroGetGitDateTime.cmake COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/niftkMacroInstallCLIScript.cmake ${CMAKE_BINARY_DIR}/CMakeMacros/niftkMacroInstallCLIScript.cmake COPYONLY)
file(GLOB package_depends_files ${CMAKE_SOURCE_DIR}/CMake/PackageDepends/*.cmake)
foreach(file_to_copy ${package_depends_files})
  file(COPY ${file_to_copy} DESTINATION ${CMAKE_BINARY_DIR}/CMakePackageDepends)
endforeach()


if (BUILD_MESHING)
  configure_file(${CMAKE_SOURCE_DIR}/CMake/FindCGAL.cmake
    ${CMAKE_BINARY_DIR}/FindCGAL.cmake @ONLY)
endif (BUILD_MESHING)

if(BUILD_IGI)
  add_definitions(-DBUILD_IGI)
endif()

######################################################################
# Find Mandatory External packages
######################################################################

######################################################################
# Find Boost.
######################################################################
set(Boost_NO_SYSTEM_PATHS ON)
# off is the default, but our optional pcl dependency will try to use static libs if we dont define it explicitly.
set(Boost_USE_STATIC_LIBS OFF)

find_package( Boost 1.54 COMPONENTS filesystem system date_time regex thread REQUIRED)
if(Boost_FOUND)
  message("Found Boost")
  set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  if(WIN32)
    if(NIFTK_WITHIN_SUPERBUILD)
      add_definitions(-DBoost_LIB_DIAGNOSTIC_DEFINITIONS)  # To get debug messages
      add_definitions(-DBOOST_ALL_NO_LIB)      # To stop auto-linking, which seems to be adding "lib" as library prefix in .obj files.
    endif(NIFTK_WITHIN_SUPERBUILD)
  endif(WIN32)
endif(Boost_FOUND)

######################################################################
# Find OpenMP and make sure Microsoft DirectComposition Library (dcomp.dll) is deployed
######################################################################

if (WIN32)
  find_package(OpenMP)

  if(OPENMP_FOUND)
    message("Found OpenMP")

    # the debug versions of the runtime library are considered "non redistributable".
    # we never hand out debug-build installers, so no problem having these packaged along correctly for internal testing.
    set(CMAKE_MSVC_ARCH_DEBUG_PREFIX "")
    set(CMAKE_MSVC_LIBNAME_FRAGMENT "")
    set(CMAKE_MSVC_OPENMP_DLLNAMESUFFIX "")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_MSVC_ARCH_DEBUG_PREFIX "Debug_NonRedist/")
      set(CMAKE_MSVC_LIBNAME_FRAGMENT "Debug")
      set(CMAKE_MSVC_OPENMP_DLLNAMESUFFIX "d")
    endif()

    if(CMAKE_CL_64)
      if(MSVC_VERSION GREATER 1599)
        # VS 10 and later:
        set(CMAKE_MSVC_ARCH "${CMAKE_MSVC_ARCH_DEBUG_PREFIX}x64")
      else()
        # VS 9 and earlier:
        set(CMAKE_MSVC_ARCH "${CMAKE_MSVC_ARCH_DEBUG_PREFIX}amd64")
      endif()
    else()
      set(CMAKE_MSVC_ARCH "${CMAKE_MSVC_ARCH_DEBUG_PREFIX}x86")
    endif()

    if (MSVC90)
      set (_omp_dll "${MSVC90_REDIST_DIR}/${CMAKE_MSVC_ARCH}/Microsoft.VC90.${CMAKE_MSVC_LIBNAME_FRAGMENT}OPENMP/vcomp90${CMAKE_MSVC_OPENMP_DLLNAMESUFFIX}.dll")
    elseif (MSVC10)
      # for some random reason, MSVC10_REDIST_DIR can be empty sometimes. as in empty-string, not
      # as in undefined-value.
      if (NOT "${MSVC10_REDIST_DIR}" STREQUAL "")
        set (_omp_dll "${MSVC10_REDIST_DIR}/${CMAKE_MSVC_ARCH}/Microsoft.VC100.${CMAKE_MSVC_LIBNAME_FRAGMENT}OPENMP/vcomp100${CMAKE_MSVC_OPENMP_DLLNAMESUFFIX}.dll")
      endif()
    elseif (MSVC11)
      # MSVC11_REDIST_DIR being empty, like above for 2010, has so far never happened for 2011.
      # so this is only precautionary.
      if (NOT "${MSVC11_REDIST_DIR}" STREQUAL "")
        set (_omp_dll "${MSVC11_REDIST_DIR}/${CMAKE_MSVC_ARCH}/Microsoft.VC110.${CMAKE_MSVC_LIBNAME_FRAGMENT}OPENMP/vcomp110${CMAKE_MSVC_OPENMP_DLLNAMESUFFIX}.dll")
      endif()
    elseif (MSVC12)
      if (NOT "${MSVC12_REDIST_DIR}" STREQUAL "")
        set (_omp_dll "${MSVC12_REDIST_DIR}/${CMAKE_MSVC_ARCH}/Microsoft.VC120.${CMAKE_MSVC_LIBNAME_FRAGMENT}OPENMP/vcomp120${CMAKE_MSVC_OPENMP_DLLNAMESUFFIX}.dll")
      endif()
    endif ()

    if ((NOT ("${_omp_dll}" STREQUAL "")) AND (EXISTS "${_omp_dll}"))
      message("Found OpenMP dll at: ${_omp_dll}")
      set (CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS "${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS};${_omp_dll}")
    else()
      message(WARNING "Did not find OpenMP DLLs (even though the compiler supports it)! The built installer might be missing dependencies.")
    endif()

  endif()
endif()

######################################################################
# Find VTK.
######################################################################
find_package(VTK REQUIRED)
if(VTK_FOUND)
  message("Found VTK")
  include(${VTK_USE_FILE})
endif(VTK_FOUND)

######################################################################
# Find GDCM. ITK is built with GDCM, so GDCM is effectively mandatory.
######################################################################
find_package(GDCM PATHS ${GDCM_DIR} REQUIRED)
if(GDCM_FOUND)
  message("Found GDCM")
  list(APPEND NIFTK_INCLUDE_DIRS_BUILD_TREE ${GDCM_INCLUDE_DIRS})
  link_directories(${GDCM_LIBRARY_DIRS})
endif()

# Find OpenCV.
######################################################################
if(BUILD_IGI)
  find_package(OpenCV REQUIRED)
  if(OpenCV_FOUND)
    message("Found OpenCV")
  endif(OpenCV_FOUND)

  # Here we do a find, just so we can read the license file
  find_package(NiftyLink REQUIRED)
  if(NiftyLink_FOUND)
    include(${NiftyLink_USE_FILE})
  endif()

endif(BUILD_IGI)


######################################################################
# Find MITK. Even though MITK can be built without Qt, CTK can't.
######################################################################
set(MITK_USE_EXT 1)
set(MITK_USE_Boost 1)
set(MITK_USE_DCMTK 1)
set(MITK_USE_BLUEBERRY ${QT_FOUND})
set(MITK_USE_QT ${QT_FOUND})
set(MITK_DESIRED_QT_VERSION @DESIRED_QT_VERSION@)
#set(MITK_DESIRED_QT_VERSION 4)
set(MITK_USE_OpenCV ${BUILD_IGI})

if(NIFTK_GENERATE_DOXYGEN_HELP)
  set(BLUEBERRY_USE_QT_HELP ON)
endif(NIFTK_GENERATE_DOXYGEN_HELP)

find_package(MITK REQUIRED)
if(MITK_FOUND)
  message("Found MITK")
  set(CMAKE_MODULE_PATH
    ${MITK_SOURCE_DIR}/CMake
    ${CMAKE_MODULE_PATH}
  )
  link_directories(${MITK_LINK_DIRECTORIES})
endif(MITK_FOUND)

######################################################################
# Find Optional External packages
######################################################################

######################################################################
# Find CGAL.
######################################################################

if (BUILD_MESHING)
  # on windows, you have to build cgal yourself because it has dependencies
  # on other libraries that cannot be built with the superbuild.
  # so in case, do not overwrite the carefully user-configured CGAL_DIR with
  # our superbuild path.
  if (NOT CGAL_DIR)
    set(CGAL_DIR "${PROJECT_BINARY_DIR}/../CGAL-install/lib/CGAL" CACHE PATH "")
  endif()
  find_package(CGAL QUIET COMPONENTS Core ImageIO)
  link_directories("${CGAL_DIR}/..")
endif (BUILD_MESHING)

######################################################################
# Find FFTW.
######################################################################
if(NIFTK_USE_FFTW)
  find_package(FFTW)
  if(FFTW_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${FFTW_INCLUDE_DIR})
    message("Found FFTW")
  else(FFTW_FOUND)
    message("Didn't find FFTW, so Fluid based registration will not be built.")
  endif(FFTW_FOUND)
endif(NIFTK_USE_FFTW)

######################################################################
# Find NIFTYREG - (version will vary according to whether CUDA found)
######################################################################
if(BUILD_NIFTYREG)
  find_package(NiftyReg)
  if(NIFTYREG_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NIFTYREG_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYREG)
    set(NIFTYREG_NVCC_FLAGS "${NIFTYREG_NVCC_FLAGS};-DUSE_NIFTYREG")
    message("Found NiftyReg")
  else(NIFTYREG_FOUND)
    message("Didn't find NiftyReg")
  endif(NIFTYREG_FOUND)
endif(BUILD_NIFTYREG)

if(NOT NIFTYREG_FOUND)
  message("No NiftyReg found, so turning off plugin")
  set("NIFTK_Plugins/uk.ac.ucl.cmic.niftyreg" OFF CACHE BOOL "NiftyReg plugin" FORCE)
endif()

######################################################################
# Find NIFTYREC - (version will vary according to whether CUDA found)
# NB: Is dependent on NIFTYREG and at time of writing only has GPU version
######################################################################

if(BUILD_NIFTYREC)
  find_package(NiftyRec)
  if(NIFTYREC_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NIFTYREC_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYREC)
    set(NIFTYREC_NVCC_FLAGS "${NIFTYREC_NVCC_FLAGS};-DUSE_NIFTYREC")
    message("Found NiftyRec")
  else(NIFTYREC_FOUND)
    message("Didn't find NiftyRec")
  endif(NIFTYREC_FOUND)
endif(BUILD_NIFTYREC)

######################################################################
# Find NIFTYSIM
######################################################################

if(BUILD_NIFTYSIM)
  find_package(NiftySim)
  if(NIFTYSIM_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NIFTYSIM_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYSIM)
    set(NIFTYSIM_NVCC_FLAGS "${NIFTYSIM_NVCC_FLAGS};-DUSE_NIFTYSIM")
    message("Found NiftySim")
  else(NIFTYSIM_FOUND)
    message("Didn't find NiftySim")
  endif(NIFTYSIM_FOUND)
endif(BUILD_NIFTYSIM)

######################################################################
# Find NiftySeg
######################################################################

if(BUILD_NIFTYSEG)
  find_package(NiftySeg)
  if(NIFTYSEG_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NIFTYSEG_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYSEG)
    set(NIFTYSEG_NVCC_FLAGS "${NIFTYSEG_NVCC_FLAGS};-DUSE_NIFTYSEG")
    message("Found NiftySeg")
  else(NIFTYSEG_FOUND)
    message("Didn't find NiftySeg")
  endif(NIFTYSEG_FOUND)
endif(BUILD_NIFTYSEG)

if(NOT NIFTYSEG_FOUND)
  message("No NiftySeg found, so turning off breast segmentation plugin")
  set("NIFTK_Plugins/uk.ac.ucl.cmic.breastsegmentation" OFF CACHE BOOL "Breast segmentation plugin" FORCE)
endif()

######################################################################
# Find RTK
######################################################################

if(BUILD_RTK)
  find_package(RTK)
  if(RTK_FOUND)
    message("Found RTK")
    include(${RTK_USE_FILE})
  endif(RTK_FOUND)
endif(BUILD_RTK)

######################################################################
# Find SlicerExecutionModel
######################################################################
if(BUILD_COMMAND_LINE_PROGRAMS)
  find_package(SlicerExecutionModel REQUIRED)
  if(SlicerExecutionModel_FOUND)
    message("Found SlicerExecutionModel")
    include(${SlicerExecutionModel_USE_FILE})
  else()
    message("Didn't find SlicerExecutionModel")
  endif()
endif()

######################################################################
# Configure prototype section, to get correct include path.
######################################################################
if (BUILD_PROTOTYPE)
  set(PROTOTYPE_DIR ${CMAKE_SOURCE_DIR}/Prototype)

  if (NOT EXISTS ${PROTOTYPE_DIR})
    find_package(Subversion)
    if (Subversion_FOUND)
      message("Found Subversion")
      message("Retrieving NiftyPrototype...")
      execute_process(
        COMMAND "${Subversion_SVN_EXECUTABLE}" checkout "${NIFTK_LOCATION_PROTOTYPE}" Prototype
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
    else()
      message("Subversion not found. The Prototype project will not be built.")
    endif()
  endif()

  if (EXISTS ${PROTOTYPE_DIR} AND IS_DIRECTORY ${PROTOTYPE_DIR})
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE
        ${NIFTK_INCLUDE_DIRS_BUILD_TREE}
        ${CMAKE_SOURCE_DIR}/Prototype/CUDA

        # Note that for these Insight Journal extensions, we dont compile the subpackage,
        # but by adding them to the include path, you can use any templated classes easily.
        # You may however have to recompile your ITK using ITK_USE_REVIEW=ON and ITK_USE_OPTIMIZED_REGISTRATION_METHODS=ON
        # depending on which classes you use.

        ${PROTOTYPE_DIR}/InsightJournal/Filters
        ${PROTOTYPE_DIR}/InsightJournal/IJ_181_ITKbinaryThinningImageFilter3D/Source
        ${PROTOTYPE_DIR}/InsightJournal/IJ_120_ITKSkeleton
        ${PROTOTYPE_DIR}/InsightJournal/IJ_644_ITKLogDomainDemonsRegistration/LogDomainDemonsRegistration-0.0.4-Source/Code
        ${PROTOTYPE_DIR}/InsightJournal/IJ_687_ITKSphericalDemons/QuadEdgeMeshFieldSmoothingFilters/Source
        ${PROTOTYPE_DIR}/InsightJournal/IJ_687_ITKSphericalDemons/QuadEdgeMeshRigidRegistration/Source
    )
  endif()
endif()

######################################################################
# Output directories, for when compiling, not installing.
######################################################################

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(SLICER_EXECUTABLE_OUTPUTPATH ${EXECUTABLE_OUTPUT_PATH}/cli-modules)
set(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})              # Used to control, where to put unit test binaries.
set(BASELINE ${NIFTK_DATA_DIR}/Baseline)
set(INPUT_DATA ${NIFTK_DATA_DIR}/Input)
make_directory(${CMAKE_BINARY_DIR}/Testing/Temporary)
set(TEMP ${CMAKE_BINARY_DIR}/Testing/Temporary)

foreach(type LIBRARY RUNTIME ARCHIVE)
  set(output_dir ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_${type}_OUTPUT_DIRECTORY ${output_dir} CACHE INTERNAL "Single output directory for building all libraries.")
  mark_as_advanced(CMAKE_${type}_OUTPUT_DIRECTORY)
endforeach()

configure_file(${CMAKE_SOURCE_DIR}/Utilities/GenerateCommandLineDoxygen.sh.in
  ${EXECUTABLE_OUTPUT_PATH}/GenerateCommandLineDoxygen @ONLY)

######################################################################
# Compilation specific stuff, like flags etc.
######################################################################

if(NIFTK_CHECK_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(NIFTK_COVERAGE_FLAGS "-g -fprofile-arcs -ftest-coverage  -O0 -DNDEBUG" )
    set(NIFTK_COVERAGE_C_FLAGS ${NIFTK_COVERAGE_FLAGS} CACHE STRING "C flags for coverage checking")
    set(NIFTK_COVERAGE_CXX_FLAGS ${NIFTK_COVERAGE_FLAGS} CACHE STRING "C++ flags for coverage checking")
  endif()
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /W2")
  set(CMAKE_CXX_WARNING_LEVEL 2)
  if(NIFTK_WITHIN_SUPERBUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_LIB_PREFIX=\"\"")
  endif(NIFTK_WITHIN_SUPERBUILD)
endif(WIN32)

# On Visual Studio 8 MS deprecated C. This removes all 1.276E1265 security warnings
if(WIN32)
  if(NOT BORLAND)
    if(NOT CYGWIN)
      if(NOT MINGW)
        if(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
          add_definitions(
            -D_CRT_FAR_MAPPINGS_NO_DEPRECATE
            -D_CRT_IS_WCTYPE_NO_DEPRECATE
            -D_CRT_MANAGED_FP_NO_DEPRECATE
            -D_CRT_NONSTDC_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE_GLOBALS
            -D_CRT_SETERRORMODE_BEEP_SLEEP_NO_DEPRECATE
            -D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
            -D_CRT_VCCLRIT_NO_DEPRECATE
            -D_SCL_SECURE_NO_DEPRECATE
            )
        endif(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
      endif(NOT MINGW)
    endif(NOT CYGWIN)
  endif(NOT BORLAND)
endif(WIN32)

# This should now always be true as MITK is required.
if(MITK_FOUND)
  set(${PROJECT_NAME}_MODULES_PACKAGE_DEPENDS_DIR "${PROJECT_SOURCE_DIR}/CMake/PackageDepends")
  list(INSERT MODULES_PACKAGE_DEPENDS_DIRS 0 ${${PROJECT_NAME}_MODULES_PACKAGE_DEPENDS_DIR})
  include(mitkCompilerSettings)
endif()

if(WIN32 AND NOT BUILD_SHARED_LIBS)
  add_definitions(-DNIFTK_STATIC)
endif()

#######################################################################
# Set the main install locations.
# These are relative to CMAKE_INSTALL_PREFIX which we MUST NOT touch.
#######################################################################
set(NIFTK_INSTALL_BASE_DIR ".")
set(NIFTK_INSTALL_BIN_DIR "bin")
set(NIFTK_INSTALL_INCLUDE_DIR "include/NifTK")
set(NIFTK_INSTALL_MATLAB_DIR "matlab")
set(NIFTK_INSTALL_DOC_DIR "doc")
set(NIFTK_INSTALL_LICENSES_DIR "licenses")
set(NIFTK_INSTALL_LIB_DIR "bin")

#######################################################################
# Set the main include path, just before generating NifTKConfigure.
#######################################################################
include_directories(
  ${NIFTK_INCLUDE_DIRS_BUILD_TREE}
  )

######################################################################
# Configure files that need variables substituting. Note that
# we 'Configure' them, which copies them to the CMAKE_BINARY_DIR
# while substituting variables, but it is the 'INSTALL' directives
# that place them in the installation directory.
######################################################################

configure_file(${CMAKE_SOURCE_DIR}/INSTALLATION.txt ${CMAKE_BINARY_DIR}/INSTALLATION.txt @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/LICENSE.txt ${CMAKE_BINARY_DIR}/LICENSE.txt @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/README.txt ${CMAKE_BINARY_DIR}/README.txt @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CONTRIBUTORS.txt ${CMAKE_BINARY_DIR}/CONTRIBUTORS.txt @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/Config/SetupDependencies.sh.in ${CMAKE_BINARY_DIR}/SetupDependencies.sh @ONLY )
configure_file(${CMAKE_SOURCE_DIR}/Config/SetupDependencies.csh.in ${CMAKE_BINARY_DIR}/SetupDependencies.csh @ONLY )
configure_file(${CMAKE_SOURCE_DIR}/Config/SetupNifTK.sh.in ${CMAKE_BINARY_DIR}/SetupNifTK.sh @ONLY )
configure_file(${CMAKE_SOURCE_DIR}/Config/SetupNifTK.csh.in ${CMAKE_BINARY_DIR}/SetupNifTK.csh @ONLY )
configure_file(${CMAKE_SOURCE_DIR}/Doc/Doxygen/niftkdoxygen.pl.in ${CMAKE_BINARY_DIR}/niftkdoxygen.pl)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Doxygen/doxygen.config.in ${CMAKE_BINARY_DIR}/doxygen.config)
configure_file(${CMAKE_SOURCE_DIR}/Doc/License.dox.in ${CMAKE_BINARY_DIR}/Doxygen/License.dox)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/Aruco.txt ${CMAKE_BINARY_DIR}/LICENSE_Aruco.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/AprilTags.txt ${CMAKE_BINARY_DIR}/LICENSE_AprilTags.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/Boost.txt ${CMAKE_BINARY_DIR}/LICENSE_Boost.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/CTK.txt ${CMAKE_BINARY_DIR}/LICENSE_CTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/DCMTK.txt ${CMAKE_BINARY_DIR}/LICENSE_DCMTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/GDCM.txt ${CMAKE_BINARY_DIR}/LICENSE_GDCM.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/ITK.txt ${CMAKE_BINARY_DIR}/LICENSE_ITK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/RTK.txt ${CMAKE_BINARY_DIR}/LICENSE_RTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/VTK.txt ${CMAKE_BINARY_DIR}/LICENSE_VTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/MITK.txt ${CMAKE_BINARY_DIR}/LICENSE_MITK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyReg.txt ${CMAKE_BINARY_DIR}/LICENSE_NiftyReg.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySeg.txt ${CMAKE_BINARY_DIR}/LICENSE_NiftySeg.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySim.txt ${CMAKE_BINARY_DIR}/LICENSE_NiftySim.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/OpenCV.txt ${CMAKE_BINARY_DIR}/LICENSE_OpenCV.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/Qt.txt ${CMAKE_BINARY_DIR}/LICENSE_Qt.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/EIGEN.txt ${CMAKE_BINARY_DIR}/LICENSE_EIGEN.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/FLANN.txt ${CMAKE_BINARY_DIR}/LICENSE_FLANN.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/PCL.txt ${CMAKE_BINARY_DIR}/LICENSE_PCL.txt)
if(BUILD_IGI)
  configure_file(${NiftyLink_LICENSE_FILE} ${CMAKE_BINARY_DIR}/ExternalLicenses/NiftyLink.txt @ONLY)
  configure_file(${NiftyLink_OIGT_LICENSE_FILE} ${CMAKE_BINARY_DIR}/ExternalLicenses/OpenIGTLink.txt @ONLY)
  configure_file(${NiftyLink_QSLOG_LICENSE_FILE} ${CMAKE_BINARY_DIR}/ExternalLicenses/QsLog.txt @ONLY)
endif()

#############################################################################################
# These represent the libraries that the apps, libraries and unit tests actually link against
#############################################################################################

set(NIFTK_VTK_LIBS_BUT_WITHOUT_QT vtkIOCore vtkIOLegacy vtkCommonCore vtkRenderingCore vtkRenderingOpenGL vtkFiltersSources vtkFiltersGeometry vtkInteractionStyle vtkRenderingFreeType vtkRenderingFreeTypeOpenGL vtkIOImage vtkIOPLY)
set(NIFTK_VTK_LIBS_WITH_QT vtkViewsQt vtkRenderingQt vtkGUISupportQt vtkGUISupportQtSQL vtkGUISupportQtWebkit)

######################################################################
# Install commands for things like README, licenses etc.
######################################################################

install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Boost.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/ITK.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/RTK.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/VTK.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Qt.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/MITK.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/CTK.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyReg.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyRec.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySeg.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySim.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/DCMTK.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/OpenCV.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/SlicerExecutionModel.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Aruco.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/AprilTags.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/EIGEN.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/FLANN.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/PCL.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
if(BUILD_IGI)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/ExternalLicenses/NiftyLink.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/ExternalLicenses/OpenIGTLink.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/ExternalLicenses/QsLog.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
endif()
install(PROGRAMS ${CMAKE_BINARY_DIR}/LICENSE.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
install(PROGRAMS ${CMAKE_BINARY_DIR}/README.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
install(PROGRAMS ${CMAKE_BINARY_DIR}/INSTALLATION.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
install(PROGRAMS ${CMAKE_BINARY_DIR}/CONTRIBUTORS.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
if(NOT WIN32 AND NOT APPLE)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/SetupDependencies.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/SetupDependencies.csh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/SetupNifTK.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/SetupNifTK.csh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
endif()

##########################################################################
# Install commands for the NiftyReg, NiftySeg, NiftyRec, NiftySim packages
##########################################################################

# NiftyReg
if(BUILD_NIFTYREG AND NIFTYREG_FOUND)
  file(GLOB NIFTY_REG_LIB_FILES
       ${CMAKE_BINARY_DIR}/../NiftyReg-install/lib/*.so
       ${CMAKE_BINARY_DIR}/../NiftyReg-install/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_REG_LIB_FILES})

  file(GLOB NIFTY_REG_BIN_FILES ${CMAKE_BINARY_DIR}/../NiftyReg-install/bin/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_REG_BIN_FILES})

  foreach(prog reg_aladin reg_f3d reg_jacobian reg_resample)
    set(FULL_APP_NAME ${prog})
    NIFTK_GENERATE_CLI_SCRIPT(NAME ${prog})
  endforeach()
endif(BUILD_NIFTYREG AND NIFTYREG_FOUND)

# NiftySeg
if(BUILD_NIFTYSEG)
  file(GLOB NIFTY_SEG_LIB_FILES
       ${CMAKE_BINARY_DIR}/../NiftySeg-install/lib/*.so
       ${CMAKE_BINARY_DIR}/../NiftySeg-install/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_SEG_LIB_FILES})

  file(GLOB NIFTY_SEG_BIN_FILES ${CMAKE_BINARY_DIR}/../NiftySeg-install/bin/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_SEG_BIN_FILES})

  foreach(prog seg_EM seg_LabFusion)
    set(FULL_APP_NAME ${prog})
    NIFTK_GENERATE_CLI_SCRIPT(NAME ${prog})
  endforeach()

  file(GLOB NIFTY_SEG_PRIORS_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/priors/*)
  install(PROGRAMS ${NIFTY_SEG_PRIORS_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/priors COMPONENT applications)
endif(BUILD_NIFTYSEG)

# NiftyRec
if(BUILD_NIFTYREC AND NIFTYREC_FOUND)
  file(GLOB NIFTY_REC_LIB_FILES
       ${CMAKE_BINARY_DIR}/../NiftyRec-install/lib/*.so
       ${CMAKE_BINARY_DIR}/../NiftyRec-install/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_REC_LIB_FILES})

  file(GLOB NIFTY_REC_INCL_FILES ${CMAKE_BINARY_DIR}/../NiftyRec-install/include/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_REC_INCL_FILES})
endif(BUILD_NIFTYREC AND NIFTYREC_FOUND)

# NiftySim
if(BUILD_NIFTYSIM AND NIFTYSIM_FOUND)
  file(GLOB NIFTY_SIM_LIB_FILES
       ${CMAKE_BINARY_DIR}/../NiftySim-install/lib/*.so
       ${CMAKE_BINARY_DIR}/../NiftySim-install/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_SIM_LIB_FILES})

  file(GLOB NIFTY_SIM_BIN_FILES ${CMAKE_BINARY_DIR}/../NiftySim-install/bin/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_SIM_BIN_FILES})
endif(BUILD_NIFTYSIM AND NIFTYSIM_FOUND)

# Camino
if(BUILD_CAMINO AND NOT WIN32 AND NOT APPLE )
  file(GLOB CAMINO_BIN_FILES
       ${CMAKE_BINARY_DIR}/../camino-src/bin/*
      )
  file(GLOB CAMINO_MAN_FILES
       ${CMAKE_BINARY_DIR}/../camino-src/man/man1/*
      )
  install(FILES ${CAMINO_MAN_FILES} DESTINATION man/man1/ PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
  install(FILES ${CAMINO_BIN_FILES} DESTINATION bin PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

######################################################################
# Decide what subdirectories we are building, and go and build them.
######################################################################

add_subdirectory(Doc)
add_subdirectory(Code)

if(BUILD_UTILITIES)
  add_subdirectory(Utilities)
endif(BUILD_UTILITIES)

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif(BUILD_TESTING)

if(BUILD_PROTOTYPE)
  add_subdirectory(Prototype)
endif(BUILD_PROTOTYPE)

######################################################################
# Packaging code.
######################################################################

# 1. Setup defaults, common for all generators.
include(CPackSetup)

# 2. Set variables that may be platform (Windows/Linux/Mac) or Generator (TGZ,DEB,NSIS) specific.
#    When CPack runs, it just uses all the information in the generated files cmake_install.cmake.
#    So, CPack does not read all your configuration information in CMakeLists.txt, and CPack
#    does not re-run any cmake process.  So, it is cmake that reads all the CMakeLists. files
#    and generates all the cmake_install.cmake.  So this command will use cmake to generate
#    and additional file, that we can politely ask cpack to include, in addition to all the
#    cmake_install.cmake files.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Configuring the Start Menu and Desktop icon strings for NSIS

if (WIN32)
  set(ICONS_SETUP_STRING "")
  set(ICONS_REMOVE_STRING "")
  set(DESKTOP_ICONS_SETUP_STRING "")
  set(DESKTOP_ICONS_REMOVE_STRING "")
  set(DESKTOP_ICONS_REMOVE_STRING "")

  foreach(NIFTK_APP ${NIFTK_APPS})
    # extract option_name
    string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
    set(target_info_list ${target_info})
    list(GET target_info_list 1 option_name)
    list(GET target_info_list 0 app_name)

    message("Current App Name: ${app_name}")

    if(EXISTS "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/${app_name}.exe")
      set(ICONS_SETUP_STRING "${ICONS_SETUP_STRING}CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\${app_name}.lnk' '$INSTDIR\\\\bin\\\\${app_name}.exe'\n")
      set(ICONS_REMOVE_STRING "${ICONS_REMOVE_STRING}Delete '$SMPROGRAMS\\\\$MUI_TEMP\\\\${app_name}.lnk'\n")

      set(DESKTOP_ICONS_SETUP_STRING "${DESKTOP_ICONS_SETUP_STRING}CreateShortCut '$DESKTOP\\\\${app_name}.lnk' '$INSTDIR\\\\bin\\\\${app_name}.exe'\n")
      set(DESKTOP_ICONS_REMOVE_STRING "${DESKTOP_ICONS_REMOVE_STRING}Delete '$DESKTOP\\\\${app_name}.lnk'\n")
    endif()
  endforeach()
endif()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

configure_file(${CMAKE_SOURCE_DIR}/CMake/CPackOptions.cmake.in
               ${CMAKE_BINARY_DIR}/NIFTKCPackOptions.cmake)

# 3. Set a variable with the name of this file.		
set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_BINARY_DIR}/NIFTKCPackOptions.cmake")

# 4. Include this optional file.
include(NIFTKCPackOptions)

# 5. Include CPack module once all variables are set.
include(CPack)

# 6. Trac #1796 - Still need to call MITK rules to get Mac Bundles etc.
include(mitkInstallRules)

######################################################################
# If we are under Windows, create two batch files which correctly
# set up the environment for the application and for Visual Studio.
# These are only used to start VS when developing. Not used in
# the final installation package.
######################################################################
if(WIN32)
  if(MITK_FOUND)

    include(mitkFunctionCreateWindowsBatchScript)

    # if we didnt build pcl then substitute some benign empty path instead of potential garbage.
    if(BUILD_PCL)
      set(PCL_BIN_PATH_FOR_BATCH_FILE "${PCL_DIR}/../bin")
    else()
      set(PCL_BIN_PATH_FOR_BATCH_FILE "")
    endif()

    # on windows, the cgal-provided installer has some required 3rd party dlls.
    if(BUILD_MESHING AND CGAL_FOUND)
      # both gmp and mpfr sit in the same directory, so we only need one here.
      get_filename_component(CGALGMP_BIN_PATH_FOR_BATCH_FILE ${GMP_LIBRARIES} PATH)
      # and of course, dont forget cgal's own dlls.
      set(CGALGMP_BIN_PATH_FOR_BATCH_FILE "${CGALGMP_BIN_PATH_FOR_BATCH_FILE};${CGAL_INSTALL_PREFIX}/bin")
    else()
      set(CGALGMP_BIN_PATH_FOR_BATCH_FILE "")
    endif()

    # some other (top secret) fem code needs niftysim at runtime.
    if(NIFTYSIM_FOUND)
      set(NIFTYSIM_BIN_PATH_FOR_BATCH_FILE "${NIFTYSIM_DIR}/bin")
    else()
      set(NIFTYSIM_BIN_PATH_FOR_BATCH_FILE "")
    endif()

    set(VS_SOLUTION_FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.sln")
    foreach(VS_BUILD_TYPE debug release)
      mitkFunctionCreateWindowsBatchScript("${CMAKE_SOURCE_DIR}/CMake/StartVS.bat.in"
        ${PROJECT_BINARY_DIR}/StartVS_${VS_BUILD_TYPE}.bat
        ${VS_BUILD_TYPE})

      message( "CreateWindowsBatchScript: Creating ${PROJECT_BINARY_DIR}/StartVS_${VS_BUILD_TYPE}.bat" )
    endforeach()

    foreach(NIFTK_APP ${NIFTK_APPS})
      # extract option_name
      string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
      set(target_info_list ${target_info})
      list(GET target_info_list 1 option_name)
      list(GET target_info_list 0 app_name)
      message("Current App Name: ${app_name}")

      foreach(VS_BUILD_TYPE debug release)
        message("CreateWindowsBatchScript: Creating ${PROJECT_BINARY_DIR}/bin/Start_${app_name}_${VS_BUILD_TYPE}.bat" )
        mitkFunctionCreateWindowsBatchScript("${CMAKE_SOURCE_DIR}/CMake/StartApp.bat.in"
          ${PROJECT_BINARY_DIR}/bin/Start${app_name}_${VS_BUILD_TYPE}.bat
          ${VS_BUILD_TYPE})
      endforeach()
    endforeach()

  else(MITK_FOUND)
    message( "CreateWindowsBatchScript: MITK not found" )
  endif(MITK_FOUND)
else(WIN32)
  message( "CreateWindowsBatchScript: WIN32 not found" )
endif(WIN32)

######################################################################
# Start: "NifTK Package Config"
######################################################################
set(NIFTK_EXPORTS_FILE "${NIFTK_BINARY_DIR}/NifTKExports.cmake")
file(REMOVE ${NIFTK_EXPORTS_FILE})

set(targets_to_export)
get_property(module_targets GLOBAL PROPERTY MITK_MODULE_TARGETS)
if(module_targets)
  list(APPEND targets_to_export ${module_targets})
endif()

if(MITK_USE_BLUEBERRY)
  get_property(plugin_targets GLOBAL PROPERTY NIFTK_EXPORTED_PLUGINS)
  if(plugin_targets)
    list(APPEND targets_to_export ${plugin_targets})
  endif()
endif()

export(TARGETS ${targets_to_export} APPEND FILE ${NIFTK_EXPORTS_FILE})

set(NIFTK_EXPORTED_TARGET_PROPERTIES )
foreach(target_to_export ${targets_to_export})
  get_target_property(autoload_targets ${target_to_export} MITK_AUTOLOAD_TARGETS)
  if(autoload_targets)
    set(MITK_EXPORTED_TARGET_PROPERTIES "${MITK_EXPORTED_TARGET_PROPERTIES} set_target_properties(${target_to_export} PROPERTIES MITK_AUTOLOAD_TARGETS \"${autoload_targets}\")")
  endif()
  get_target_property(autoload_dir ${target_to_export} MITK_AUTOLOAD_DIRECTORY)
  if(autoload_dir)
    set(MITK_EXPORTED_TARGET_PROPERTIES "${MITK_EXPORTED_TARGET_PROPERTIES} set_target_properties(${target_to_export} PROPERTIES MITK_AUTOLOAD_DIRECTORY \"${autoload_dir}\")")
  endif()
endforeach()

get_property(MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS_CONFIG GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS)

set(VISIBILITY_AVAILABLE 0)
set(visibility_test_flag "")
mitkFunctionCheckCompilerFlags("-fvisibility=hidden" visibility_test_flag)
if(visibility_test_flag)
  # The compiler understands -fvisiblity=hidden (probably gcc >= 4 or Clang)
  set(VISIBILITY_AVAILABLE 1)
endif()

# This generates a C++ header file with values that need compiling into code.
configure_file(${CMAKE_SOURCE_DIR}/Config/NifTKConfigure.h.in ${CMAKE_BINARY_DIR}/NifTKConfigure.h)

# This generates a C++ header file for GUI apps only.
configure_file(${CMAKE_SOURCE_DIR}/Code/Gui/MITK/Apps/NifTKApplication.h ${CMAKE_BINARY_DIR}/NifTKApplication.h)

# The next two generate CMake code to enable external projects to correctly use NifTK.
configure_file(${CMAKE_SOURCE_DIR}/Config/NifTKConfig.cmake.in ${CMAKE_BINARY_DIR}/NifTKConfig.cmake @ONLY)
write_basic_config_version_file(${CMAKE_CURRENT_BINARY_DIR}/NifTKConfigVersion.cmake VERSION "${NIFTK_VERSION_MAJOR}.${NIFTK_VERSION_MINOR}.${NIFTK_VERSION_PATCH}" COMPATIBILITY AnyNewerVersion)

######################################################################
# End: "NifTK Package Config"
######################################################################
