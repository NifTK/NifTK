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
#  Last Changed      : $LastChangedDate: 2011-12-17 14:35:07 +0000 (Sat, 17 Dec 2011) $ 
#  Revision          : $Revision: 8065 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

######################################################################
# Our version number. Edit this to generate a new version.
# However, be warned, you need to set CMAKE_INSTALL_PREFIX manually.
# Also, Trac 1592, version number MUST change if CTK changes.
######################################################################
SET(NIFTK_VERSION_MAJOR 12 CACHE STRING "Year version number" FORCE )
SET(NIFTK_VERSION_MINOR 07-dev CACHE STRING "Month version number" FORCE )
MARK_AS_ADVANCED(NIFTK_VERSION_MAJOR)
MARK_AS_ADVANCED(NIFTK_VERSION_MINOR)

######################################################################
# Set the minimum CMake version.
######################################################################
IF(APPLE)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
ELSE(APPLE)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.8.4)
ENDIF(APPLE)

# If CMake is the version we're expecting, don't show the
# CMAKE_BACKWARDS_COMPATIBILITY option. If the version is higher than the
# minimal version required, then show the backward compatibility option.
IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.4$")
  MARK_AS_ADVANCED(FORCE CMAKE_BACKWARDS_COMPATIBILITY)
ELSE("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.4$")
  MARK_AS_ADVANCED(CLEAR CMAKE_BACKWARDS_COMPATIBILITY)
ENDIF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" MATCHES "^2\\.8\\.4$")

######################################################################
# Setting supported build types. Should ONLY be Debug or Release.
######################################################################

IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Valid options are Debug/Release")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Setting build type to Release (default). Valid options: Debug/Release.")
ENDIF()

IF (NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Release")) 
  MESSAGE(FATAL_ERROR "Build type ${CMAKE_BUILD_TYPE} is not supported.")
ENDIF()

##################################################################################
# Set some CMake Policies.
# See http://cmake.org/cmake/help/cmake-2-8-docs.html#section_Policies for details
##################################################################################

SET(project_policies
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
FOREACH(policy ${project_policies})
  IF(POLICY ${policy})
    CMAKE_POLICY(SET ${policy} NEW)
  ENDIF()
ENDFOREACH()

######################################################################
# We have a super-build option. (Terminology comes from MITK/CTK).
######################################################################

OPTION(BUILD_SUPERBUILD "Build NIFTK and the projects it depends on via SuperBuild.cmake." ON)

If(BUILD_SUPERBUILD)
  PROJECT(NIFTK-SUPERBUILD)
ELSE(BUILD_SUPERBUILD)
  PROJECT(NIFTK)
ENDIF(BUILD_SUPERBUILD)

######################################################################
# Options. These are set up front, so are available when configuring
# the SuperBuild, and hence they must also be passed to the normal
# build. So, look in CMake/Superbuild.cmake to see where they are
# passed to the main build of NifTK when doing the Superbuild.
######################################################################
OPTION(BUILD_COMMAND_LINE_PROGRAMS "Build command line applications in NifTK/Code/Applications." ON)
OPTION(BUILD_COMMAND_LINE_SCRIPTS "Build command line scripts in NifTK/Code/Scripts." ON)
OPTION(BUILD_GUI "Build GUI in NifTK/Code/Gui." OFF)
OPTION(BUILD_IGI "Build CMIC IGI platform." OFF)
OPTION(BUILD_MESHING "Build meshing?" OFF)
OPTION(BUILD_NIFTYREG "Build NiftyReg." OFF)
OPTION(BUILD_NIFTYREC "Build NiftyRec." OFF)
OPTION(BUILD_NIFTYSIM "Build NiftySim." OFF)
OPTION(BUILD_NIFTYSEG "Build NiftySeg." OFF)
OPTION(BUILD_PROTOTYPE "Build prototype code in NifTK/Prototype." OFF)
OPTION(BUILD_PROTOTYPE_BE "Build prototype code in NifTK/Prototype/be." OFF)
OPTION(BUILD_PROTOTYPE_GY "Build prototype code in NifTK/Prototype/gy." OFF)
OPTION(BUILD_PROTOTYPE_IM "Build prototype code in NifTK/Prototype/im." OFF)
OPTION(BUILD_PROTOTYPE_JHH "Build prototype code in NifTK/Prototype/jhh." OFF)
OPTION(BUILD_PROTOTYPE_KKL "Build prototype code in NifTK/Prototype/kkl." OFF)
OPTION(BUILD_PROTOTYPE_MJC "Build prototype code in NifTK/Prototype/mjc." OFF)
OPTION(BUILD_PROTOTYPE_TM "Build prototype code in NifTK/Prototype/tm." OFF)
OPTION(BUILD_SHARED_LIBS "Build NifTK with shared libraries." ON)
OPTION(BUILD_TESTING "Build Unit tests in NifTK/Testing/Code." OFF)
OPTION(NIFTK_CHECK_COVERAGE "Enable/Disable code coverage checking." OFF)
IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  SET(NIFTK_SHOW_CONSOLE_WINDOW ON)
ELSE()
  SET(NIFTK_SHOW_CONSOLE_WINDOW OFF)
ENDIF()
OPTION(NIFTK_SHOW_CONSOLE_WINDOW "Use this to enable or disable the console window when starting GUI Applications on Windows" ${NIFTK_SHOW_CONSOLE_WINDOW})
OPTION(NIFTK_GENERATE_DOXYGEN_HELP "Use this to generate Doxygen help in GUI" OFF)
OPTION(NIFTK_VERBOSE_COMPILER_WARNINGS "Add in all the warning flags that MITK does" OFF)
OPTION(NIFTK_USE_FFTW "Use FFTW. Warning FFTW is GPL, so for binary external releases we can't use this. This is used for Kelvin's Fluid registration implementation." OFF)
OPTION(NIFTK_USE_CUDA "Use CUDA. Experimental. " OFF)
OPTION(NIFTK_USE_GIT_PROTOCOL "If behind a firewall set to OFF to use http: and ON to use git:" OFF)
OPTION(NIFTK_USE_OPENCV "Turn on OpenCV in MITK Build" OFF)

######################################################################
# Variables that get compiled into C++ code. 
######################################################################
SET(NIFTK_PLATFORM "NifTK" CACHE STRING "Full name of platform." FORCE )
SET(NIFTK_COPYRIGHT "Copyright (C) 2008-2012 University College London" CACHE STRING "Copyright string." FORCE )
SET(NIFTK_ORIGIN_URL "http://cmic.cs.ucl.ac.uk/" CACHE STRING "URL of originating institution." FORCE )
SET(NIFTK_ORIGIN_SHORT_TEXT "CMIC" CACHE STRING "Short name of originating institution." FORCE)
SET(NIFTK_ORIGIN_LONG_TEXT "Centre For Medical Image Computing" CACHE STRING "Full name of originating institution." FORCE)
SET(NIFTK_WIKI_URL "https://cmicdev.cs.ucl.ac.uk/trac/wiki/MidasPlatform" CACHE STRING "Wiki URL" FORCE )
SET(NIFTK_WIKI_TEXT "wiki" CACHE STRING "Wiki text" FORCE )
SET(NIFTK_DASHBOARD_URL "https://cmicdev.cs.ucl.ac.uk/cdash/index.php?project=NifTK" CACHE STRING "Dashboard URL" FORCE)
SET(NIFTK_DASHBOARD_TEXT "dashboard" CACHE STRING "Dashboard text" FORCE)
SET(NIFTK_USER_CONTACT "https://www.mailinglists.ucl.ac.uk/mailman/listinfo/niftk-users" CACHE STRING "Contact address for users." FORCE )
SET(NIFTK_DEVELOPER_CONTACT "https://www.mailinglists.ucl.ac.uk/mailman/listinfo/niftk-developers" CACHE STRING "Contact address for developers." FORCE )
IF(WIN32)
  SET(NIFTK_BASE_NAME "NifTK" CACHE STRING "Base name for installation folder. Windows convention is upper case letters" FORCE )
ELSE(WIN32)
  SET(NIFTK_BASE_NAME "niftk" CACHE STRING "Base name for installation folder. Unix convention is lower case letters." FORCE )
ENDIF(WIN32)
SET(NIFTK_VERSION_STRING "${NIFTK_VERSION_MAJOR}.${NIFTK_VERSION_MINOR}" CACHE STRING "String to describe fully named version" FORCE)
SET(NIFTK_DEPLOY_NAME "${NIFTK_BASE_NAME}-${NIFTK_VERSION_STRING}" CACHE STRING "String to describe deployed name" FORCE)
SET(NIFTK_NIGHTLY_DOCS "https://cmicdev.cs.ucl.ac.uk/NifTK/html/index.html" CACHE STRING "Nightly generated documentation" )
SET(NIFTK_BUILD_INSTRUCTIONS "http://cmic.cs.ucl.ac.uk/platform/niftk/html/BuildInstructions.html" CACHE STRING "Build instructions")
SET(NIFTK_INSTALL_INSTRUCTIONS "http://cmic.cs.ucl.ac.uk/platform/niftk/html/BuildInstructions.html" CACHE STRING "Installation instructions")
SET(NIFTK_VERSION_BOOST "1.46.1" CACHE STRING "Version of Boost")
SET(NIFTK_VERSION_GDCM "2.0.18" CACHE STRING "Version of GDCM")
SET(NIFTK_VERSION_DCMTK "3.6.1_20120222" CACHE STRING "Version of DCMTK")
SET(NIFTK_VERSION_ITK "3.20.1" CACHE STRING "Version of ITK")
SET(NIFTK_VERSION_VTK "5.8.0" CACHE STRING "Version of VTK")
SET(NIFTK_VERSION_NIFTYREG "309" CACHE STRING "Version of NiftyReg")
SET(NIFTK_VERSION_NIFTYSEG "88" CACHE STRING "Version of NiftySeg")
SET(NIFTK_VERSION_NIFTYREC "14" CACHE STRING "Version of NiftyRec")
SET(NIFTK_VERSION_NIFTYSIM "127" CACHE STRING "Version of NiftySim")
SET(NIFTK_LOCATION_BOOST "http://cmic.cs.ucl.ac.uk/platform/dependencies/boost-${NIFTK_VERSION_BOOST}.tar.gz" CACHE STRING "Location of Boost")
SET(NIFTK_LOCATION_GDCM "http://cmic.cs.ucl.ac.uk/platform/dependencies/gdcm-${NIFTK_VERSION_GDCM}.tar.gz" CACHE STRING "Location of GDCM")
SET(NIFTK_LOCATION_DCMTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/dcmtk-${NIFTK_VERSION_DCMTK}.tar.gz" CACHE STRING "Location of DCMTK")
SET(NIFTK_LOCATION_ITK "http://cmic.cs.ucl.ac.uk/platform/dependencies/InsightToolkit-${NIFTK_VERSION_ITK}.tar.gz" CACHE STRING "Location of ITK")
SET(NIFTK_LOCATION_VTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/vtk-${NIFTK_VERSION_VTK}.tar.gz" CACHE STRING "Location of VTK")
SET(NIFTK_LOCATION_CTK "//github.com/commontk/CTK.git" CACHE STRING "Location of CTK - see also NIFTK_USE_GIT_PROTOCOL.")
SET(NIFTK_LOCATION_MITK "//github.com/MattClarkson/MITK.git" CACHE STRING "Location of CTK - see also NIFTK_USE_GIT_PROTOCOL.")
SET(NIFTK_LOCATION_NIFTYLINK "//cmicdev.cs.ucl.ac.uk/NiftyLink" CACHE STRING "Location of NiftyLink - see also NIFTK_USE_GIT_PROTOCOL.")
SET(NIFTK_LOCATION_DATA "https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NifTKData" CACHE STRING  "Location of NifTK unit test data.")
SET(NIFTK_LOCATION_NIFTYREG "https://niftyreg.svn.sourceforge.net/svnroot/niftyreg/trunk/nifty_reg/" CACHE STRING  "Location of NiftyReg.")
SET(NIFTK_LOCATION_NIFTYSEG "https://niftyseg.svn.sourceforge.net/svnroot/niftyseg" CACHE STRING  "Location of NiftySEG.")
SET(NIFTK_LOCATION_NIFTYREC "https://niftyrec.svn.sourceforge.net/svnroot/niftyrec/" CACHE STRING  "Location of NiftyRec.")
SET(NIFTK_LOCATION_NIFTYSIM "https://niftysim.svn.sourceforge.net/svnroot/niftysim/trunk/nifty_sim/" CACHE STRING  "Location of NiftyRec.")

######################################################################
# Hide these variables from the user, unless they are 'advanced' :-)
######################################################################
MARK_AS_ADVANCED(NIFTK_PLATFORM)
MARK_AS_ADVANCED(NIFTK_COPYRIGHT)
MARK_AS_ADVANCED(NIFTK_ORIGIN_URL)
MARK_AS_ADVANCED(NIFTK_ORIGIN_SHORT_TEXT)
MARK_AS_ADVANCED(NIFTK_ORIGIN_LONG_TEXT)
MARK_AS_ADVANCED(NIFTK_HELP_URL_TEXT)
MARK_AS_ADVANCED(NIFTK_WIKI_URL)
MARK_AS_ADVANCED(NIFTK_WIKI_TEXT)
MARK_AS_ADVANCED(NIFTK_DASHBOARD_URL)
MARK_AS_ADVANCED(NIFTK_DASHBOARD_TEXT)
MARK_AS_ADVANCED(NIFTK_USER_CONTACT)
MARK_AS_ADVANCED(NIFTK_DEVELOPER_CONTACT)
MARK_AS_ADVANCED(NIFTK_BASE_NAME)
MARK_AS_ADVANCED(NIFTK_VERSION_STRING)
MARK_AS_ADVANCED(NIFTK_DEPLOY_NAME)
MARK_AS_ADVANCED(NIFTK_SHOW_CONSOLE_WINDOW)
MARK_AS_ADVANCED(NIFTK_USE_GIT_PROTOCOL)
MARK_AS_ADVANCED(NIFTK_CHECK_COVERAGE)
MARK_AS_ADVANCED(NIFTK_VERBOSE_COMPILER_WARNINGS)
MARK_AS_ADVANCED(NIFTK_GENERATE_DOXYGEN_HELP)
MARK_AS_ADVANCED(NIFTK_NIGHTLY_DOCS)
MARK_AS_ADVANCED(NIFTK_BUILD_INSTRUCTIONS)
MARK_AS_ADVANCED(NIFTK_INSTALL_INSTRUCTIONS)
MARK_AS_ADVANCED(NIFTK_VERSION_BOOST)
MARK_AS_ADVANCED(NIFTK_VERSION_GDCM)
MARK_AS_ADVANCED(NIFTK_VERSION_DCMTK)
MARK_AS_ADVANCED(NIFTK_VERSION_ITK)
MARK_AS_ADVANCED(NIFTK_VERSION_VTK)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYREG)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYSEG)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYREC)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYSIM)
MARK_AS_ADVANCED(NIFTK_LOCATION_BOOST)
MARK_AS_ADVANCED(NIFTK_LOCATION_GDCM)
MARK_AS_ADVANCED(NIFTK_LOCATION_DCMTK)
MARK_AS_ADVANCED(NIFTK_LOCATION_ITK)
MARK_AS_ADVANCED(NIFTK_LOCATION_VTK)
MARK_AS_ADVANCED(NIFTK_LOCATION_CTK)
MARK_AS_ADVANCED(NIFTK_LOCATION_MITK)
MARK_AS_ADVANCED(NIFTK_LOCATION_DATA)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYREG)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYSEG)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYREC)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYSIM)
MARK_AS_ADVANCED(NIFTK_USE_FFTW)
MARK_AS_ADVANCED(NIFTK_USE_CUDA)
MARK_AS_ADVANCED(NIFTK_USE_GIT_PROTOCOL)
MARK_AS_ADVANCED(NIFTK_USE_OPENCV)

######################################################################
# Setup the path to load CMake macros, and extra CMake files.
######################################################################
SET(CMAKE_MODULE_PATH 
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

######################################################################
# Test for some required system information. This came from old
# ITK CMake scripts, so I (Matt) am unsure if we still need it.
######################################################################
INCLUDE(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityC.cmake)
INCLUDE(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityCXX.cmake)

######################################################################
# Configure Dart testing support.  This should be done before any
# MESSAGE(FATAL_ERROR ...) commands are invoked.
######################################################################
INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)
MARK_AS_ADVANCED(TCL_TCLSH DART_ROOT)
ENABLE_TESTING()
IF(BUILD_TESTING)
  CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Doc/Images/NifTKLogo.gif
                 ${CMAKE_BINARY_DIR}/Testing/HTML/TestingResults/Icons/Logo.gif
                 COPYONLY IMMEDIATE)
  SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  MARK_AS_ADVANCED(BUILDNAME)
  
  # Setup file for setting custom ctest vars
  CONFIGURE_FILE(CMake/CTestCustom.cmake.in ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)

ENDIF(BUILD_TESTING)

######################################################################
# Platform checks:
# Check availability of some standard API functions
######################################################################
INCLUDE(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
CHECK_FUNCTION_EXISTS(mkstemps HAVE_MKSTEMPS) 

###########################################################################
# Set these compiler flags early, so it can be applied to all dependencies.
###########################################################################

SET(NIFTK_ADDITIONAL_C_FLAGS "" CACHE STRING "Additional C Flags")
MARK_AS_ADVANCED(NIFTK_ADDITIONAL_C_FLAGS)
SET(NIFTK_ADDITIONAL_CXX_FLAGS "" CACHE STRING "Additional CXX Flags")
MARK_AS_ADVANCED(NIFTK_ADDITIONAL_CXX_FLAGS)

######################################################################
# Set GIT protocol, as dependencies need it
######################################################################
FIND_PACKAGE(Git REQUIRED)
SET(GIT_PROTOCOL "git")
IF(NOT NIFTK_USE_GIT_PROTOCOL)
  SET(GIT_PROTOCOL "http")
ENDIF()

########################################################################
# Set NIFTK_INSTALL_PREFIX, and NIFTK_LINK_PREFIX, used in scripts.
# Note: DONT try changing CMAKE_INSTALL_PREFIX. Think of it as reserved.
########################################################################

GET_FILENAME_COMPONENT(NIFTK_LINK_PREFIX ${CMAKE_INSTALL_PREFIX} PATH)
SET(NIFTK_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

######################################################################
# Sanity Checks before doing Superbuild.
######################################################################
IF(BUILD_GUI)
  IF(NOT BUILD_SHARED_LIBS)
    MESSAGE(FATAL_ERROR "If you wan't to build the GUI, you must use dynamic linking, turn BUILD_SHARED to ON.")
  ENDIF(NOT BUILD_SHARED_LIBS)
ENDIF(BUILD_GUI)

IF(BUILD_GUI)
  SET(NIFTK_GENERATE_DOXYGEN_HELP ON)
ENDIF(BUILD_GUI)

IF(BUILD_IGI AND NOT BUILD_GUI)
  MESSAGE(FATAL_ERROR "You have selected BUILD_IGI=ON, but to do so, you must enabled BUILD_GUI=ON")
ENDIF()

######################################################################
# Find Qt. We need to check for Qt first, before we go ahead and do
# the Superbuild, because VTK needs to know if we have Qt.
######################################################################

IF(BUILD_GUI)
  IF(BUILD_IGI)
    FIND_PACKAGE(Qt4 COMPONENTS QtCore QtGui QtXml QtNetwork REQUIRED)
  ELSE(BUILD_IGI)
    FIND_PACKAGE(Qt4 COMPONENTS QtCore QtGui QtXml REQUIRED)
  ENDIF(BUILD_IGI)
  
  IF(QT_FOUND)
  
    MESSAGE("Found Qt")
    
    SET(QT_USE_QTXML 1)
    SET(QT_USE_QTXMLPATTERNS 1)
    SET(QT_USE_PHONON 0)

    INCLUDE(${QT_USE_FILE})
  
    # This is to get NiftyView Help working, which launches the Qt Assistant, and the QtHelp framework needs libQtCLucene.
    IF(APPLE)
      LIST(APPEND ALL_LIBRARIES QtCLucene )
    ENDIF(APPLE)
    
    IF(BUILD_SUPERBUILD)
      SET(VTK_QT_ARGS
        -DDESIRED_QT_VERSION:STRING=4
        -DVTK_USE_GUISUPPORT:BOOL=ON
        -DVTK_USE_QVTK_QTOPENGL:BOOL=ON
        -DVTK_USE_QVTK:BOOL=ON
        -DVTK_USE_QT:BOOL=ON
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
       )
     ENDIF(BUILD_SUPERBUILD)
     
  ENDIF(QT_FOUND)
ENDIF(BUILD_GUI)
  
######################################################################
# Find CUDA: Required by the Nifty packages
######################################################################

IF(NIFTK_USE_CUDA)
  FIND_PACKAGE(CUDA)
  IF(CUDA_FOUND)
    INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE})
    CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})
    ADD_DEFINITIONS(-D_USE_CUDA)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_USE_CUDA")
    MESSAGE("Found CUDA")
  ELSE(CUDA_FOUND)
    MESSAGE("Didn't find CUDA")
  ENDIF(CUDA_FOUND)
ENDIF(NIFTK_USE_CUDA)

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

IF(BUILD_SUPERBUILD)
  INCLUDE("CMake/SuperBuild.cmake")
  RETURN()
ENDIF(BUILD_SUPERBUILD)

######################################################################
# End of SuperBuild. Print out where the source and binary folders
# are, just to make it really explicit... well explicit to the user
# that bothers to read these messages! :-)
######################################################################

MESSAGE("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
MESSAGE("CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

######################################################################
# Configure CMake files before we need to use them in the FIND_XXX.
#
# So, the variables NIFTK_BOOSTINSTALL etc.
# should be passed in on cmake command line, or set up using ccmake.
######################################################################

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/FindFFTW.cmake 
  ${CMAKE_BINARY_DIR}/FindFFTW.cmake @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/FindBoost.cmake 
  ${CMAKE_BINARY_DIR}/FindBoost.cmake @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/FindNiftyReg.cmake 
  ${CMAKE_BINARY_DIR}/FindNiftyReg.cmake @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/FindNiftyRec.cmake 
  ${CMAKE_BINARY_DIR}/FindNiftyRec.cmake @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/FindNiftySim.cmake 
  ${CMAKE_BINARY_DIR}/FindNiftySim.cmake @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/FindNiftySeg.cmake 
  ${CMAKE_BINARY_DIR}/FindNiftySeg.cmake @ONLY)

IF (BUILD_MESHING)
  CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/FindCGAL.cmake 
    ${CMAKE_BINARY_DIR}/FindCGAL.cmake @ONLY)
  IF (NOT WIN32) 
    MESSAGE("cgal => -L${CGAL_DIR}/..")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CGAL_DIR}/..")
  ENDIF (NOT WIN32) 
ENDIF (BUILD_MESHING)

######################################################################
# Find Mandatory External packages
######################################################################

######################################################################
# Find Boost.
######################################################################
#FIND_PACKAGE( Boost 1.35.0 COMPONENTS filesystem system date_time REQUIRED)
FIND_PACKAGE( Boost COMPONENTS filesystem system date_time REQUIRED)
IF(Boost_FOUND)
  MESSAGE("Found Boost")
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
  LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
  IF(WIN32)
    IF(NIFTK_WITHIN_SUPERBUILD)
      ADD_DEFINITIONS(-DBOOST_LIB_DIAGNOSTIC)  # To get debug messages
      ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB)      # To stop auto-linking, which seems to be adding "lib" as library prefix in .obj files.
    ENDIF(NIFTK_WITHIN_SUPERBUILD)
  ENDIF(WIN32)  
ENDIF(Boost_FOUND)


######################################################################
# Find ITK.
######################################################################
FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
  MESSAGE("Found ITK")
  INCLUDE(${ITK_USE_FILE})
ENDIF(ITK_FOUND)


######################################################################
# Find VTK.
######################################################################
FIND_PACKAGE(VTK REQUIRED)
IF(VTK_FOUND)
  MESSAGE("Found VTK")
  INCLUDE(${VTK_USE_FILE})
ENDIF(VTK_FOUND)


######################################################################
# Find MITK. Even though MITK can be built without Qt, CTK can't.
######################################################################
SET(MITK_USE_EXT 1)
SET(MITK_USE_Boost 1)
SET(MITK_USE_BLUEBERRY ${QT_FOUND})
SET(MITK_USE_QT ${QT_FOUND})

IF(NIFTK_GENERATE_DOXYGEN_HELP)
  SET(BLUEBERRY_USE_QT_HELP ON) 
ENDIF(NIFTK_GENERATE_DOXYGEN_HELP)
  
FIND_PACKAGE(MITK REQUIRED)
IF(MITK_FOUND)
  MESSAGE("Found MITK")
  SET(CMAKE_MODULE_PATH 
    ${MITK_SOURCE_DIR}/CMake
    ${CMAKE_MODULE_PATH}
  )
  LINK_DIRECTORIES(${MITK_LINK_DIRECTORIES})    
ENDIF(MITK_FOUND)

######################################################################
# Find Optional External packages
######################################################################

######################################################################
# Find CGAL.
######################################################################

IF (BUILD_MESHING)
  SET(CGAL_DIR "${PROJECT_BINARY_DIR}/../CMakeExternals/Install/CGAL/lib/CGAL")
  FIND_PACKAGE(CGAL)
  IF (NOT WIN32) 
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CGAL_DIR}/..")
  ENDIF (NOT WIN32) 
ENDIF (BUILD_MESHING)

######################################################################
# Find NiftyLink
######################################################################
IF(BUILD_IGI)
  FIND_PACKAGE(NiftyLink REQUIRED)
  IF(NiftyLink_FOUND)
    MESSAGE("Found NiftyLink")    
    INCLUDE(${NiftyLink_USE_FILE})
  ENDIF(NiftyLink_FOUND)
ENDIF(BUILD_IGI)

IF(NOT NiftyLink_FOUND)
  MESSAGE("No NiftyLink found, so turning off Surgical Guidance Plugin, and setting BUILD_IGI to false.")
  SET("NIFTK_Plugins/uk.ac.ucl.cmic.surgicalguidance" OFF CACHE BOOL "Surgical Guidance plugin" FORCE)
  SET("BUILD_IGI" OFF CACHE BOOL "Build CMIC IGI platform." FORCE)
ENDIF()

######################################################################
# Find FFTW.
######################################################################
IF(NIFTK_USE_FFTW)
  FIND_PACKAGE(FFTW)
  IF(FFTW_FOUND)
    INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
    MESSAGE("Found FFTW")
  ELSE(FFTW_FOUND)
    MESSAGE("Didn't find FFTW, so Fluid based registration will not be built.")
  ENDIF(FFTW_FOUND)
ENDIF(NIFTK_USE_FFTW)


######################################################################
# Find NIFTYREG - (version will vary according to whether CUDA found)
######################################################################
IF(BUILD_NIFTYREG)
  FIND_PACKAGE(NiftyReg)
  IF(NIFTYREG_FOUND)
    INCLUDE_DIRECTORIES(${NIFTYREG_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_NIFTYREG)
    SET(NIFTYREG_NVCC_FLAGS "${NIFTYREG_NVCC_FLAGS};-DUSE_NIFTYREG")
    MESSAGE("Found NiftyReg")
  ELSE(NIFTYREG_FOUND)
    MESSAGE("Didn't find NiftyReg")
  ENDIF(NIFTYREG_FOUND)
ENDIF(BUILD_NIFTYREG)

IF(NOT NIFTYREG_FOUND)
  MESSAGE("No NiftyReg found, so turning off plugin")
  SET("NIFTK_Plugins/uk.ac.ucl.cmic.niftyreg" OFF CACHE BOOL "NiftyReg plugin" FORCE)
ENDIF()


######################################################################
# Find NIFTYREC - (version will vary according to whether CUDA found)
# NB: Is dependent on NIFTYREG and at time of writing only has GPU version
######################################################################

IF(BUILD_NIFTYREC)
  FIND_PACKAGE(NiftyRec)
  IF(NIFTYREC_FOUND)
    INCLUDE_DIRECTORIES(${NIFTYREC_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_NIFTYREC)
    SET(NIFTYREC_NVCC_FLAGS "${NIFTYREC_NVCC_FLAGS};-DUSE_NIFTYREC")
    MESSAGE("Found NiftyRec")
  ELSE(NIFTYREC_FOUND)
    MESSAGE("Didn't find NiftyRec")
  ENDIF(NIFTYREC_FOUND)
ENDIF(BUILD_NIFTYREC)


######################################################################
# Find NIFTYSIM
######################################################################

IF(BUILD_NIFTYSIM)
  FIND_PACKAGE(NiftySim)
  IF(NIFTYSIM_FOUND)
    INCLUDE_DIRECTORIES(${NIFTYSIM_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_NIFTYSIM)
    SET(NIFTYSIM_NVCC_FLAGS "${NIFTYSIM_NVCC_FLAGS};-DUSE_NIFTYSIM")
    MESSAGE("Found NiftySim")
  ELSE(NIFTYSIM_FOUND)
    MESSAGE("Didn't find NiftySim")
  ENDIF(NIFTYSIM_FOUND)
ENDIF(BUILD_NIFTYSIM)


######################################################################
# Find NIFTYSEG
######################################################################

IF(BUILD_NIFTYSEG)
  FIND_PACKAGE(NiftySeg)
  IF(NIFTYSEG_FOUND)
    INCLUDE_DIRECTORIES(${NIFTYSEG_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_NIFTYSEG)
    SET(NIFTYSEG_NVCC_FLAGS "${NIFTYSEG_NVCC_FLAGS};-DUSE_NIFTYSEG")
    MESSAGE("Found NiftySeg")
  ELSE(NIFTYSEG_FOUND)
    MESSAGE("Didn't find NiftySeg")
  ENDIF(NIFTYSEG_FOUND)
ENDIF(BUILD_NIFTYSEG)

IF(NOT NIFTYSEG_FOUND)
  MESSAGE("No NiftySeg found, so turning off breast segmentation plugin")
  SET("NIFTK_Plugins/uk.ac.ucl.cmic.breastsegmentation" OFF CACHE BOOL "Breast segmentation plugin" FORCE)
ENDIF()


########################################################################
# Simplest if we list ALL our include directories here.
# We need to reference the CMAKE_BINARY_DIR, as we generate stuff there.
########################################################################
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/Config)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/Code)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/Code/Libs/Qt)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/Common)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/Common/File)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/Common/Exceptions)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/Common/Image)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/MIDAS)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/VTK)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/VTK/Common)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Common)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/BasicFilters)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/IO)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Pipelines)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/BoundaryShiftIntegral)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/CorticalThickness)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Metrics)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Methods)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Transforms)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Optimizers)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Commands)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Construction)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Constraints)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/RegistrationToolbox/Filters)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Methods)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Optimizers)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Projection)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Metrics)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Commands)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/xRay)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/2D3DToolbox/Transforms)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Segmentation)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Segmentation/MIDASMorphologicalEditor)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Segmentation/MIDASIrregularVolumeEditor)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITKVTK/IO)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/Qt)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Testing/Code/Libs/ITK/RegistrationToolbox)

IF (BUILD_MESHING) 
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/Common/Meshing)
ENDIF (BUILD_MESHING) 

######################################################################
# Output directories, for when compiling, not installing.
######################################################################

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})              # Used to control, where to put unit test binaries.

FOREACH(type LIBRARY RUNTIME ARCHIVE)
  SET(output_dir ${CMAKE_BINARY_DIR}/bin)
  SET(CMAKE_${type}_OUTPUT_DIRECTORY ${output_dir} CACHE INTERNAL "Single output directory for building all libraries.")
  MARK_AS_ADVANCED(CMAKE_${type}_OUTPUT_DIRECTORY)
ENDFOREACH()

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Utilities/GenerateCommandLineDoxygen.sh.in 
  ${EXECUTABLE_OUTPUT_PATH}/GenerateCommandLineDoxygen @ONLY)

######################################################################
# Compilation specific stuff, like flags etc.
######################################################################

IF(NIFTK_CHECK_COVERAGE)
  IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(NIFTK_COVERAGE_FLAGS "-g -fprofile-arcs -ftest-coverage  -O0 -DNDEBUG" )
    SET(NIFTK_COVERAGE_C_FLAGS ${NIFTK_COVERAGE_FLAGS} CACHE STRING "C flags for coverage checking")
    SET(NIFTK_COVERAGE_CXX_FLAGS ${NIFTK_COVERAGE_FLAGS} CACHE STRING "C++ flags for coverage checking")
  ENDIF()
ENDIF()

IF(WIN32)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /W2")
  SET(CMAKE_CXX_WARNING_LEVEL 2)
  IF(NIFTK_WITHIN_SUPERBUILD)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_LIB_PREFIX=\"\"")
  ENDIF(NIFTK_WITHIN_SUPERBUILD)
ENDIF(WIN32)

# On Visual Studio 8 MS deprecated C. This removes all 1.276E1265 security warnings
IF(WIN32)
  IF(NOT BORLAND)
    IF(NOT CYGWIN)
      IF(NOT MINGW)
        IF(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
          ADD_DEFINITIONS(
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
        ENDIF(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
      ENDIF(NOT MINGW)
    ENDIF(NOT CYGWIN)
  ENDIF(NOT BORLAND)
ENDIF(WIN32)

# This should now always be true as MITK is required.
IF(MITK_FOUND)
  INCLUDE(mitkCompilerSettings)
ENDIF(MITK_FOUND)

#######################################################################
# Set the main install locations. 
# These are relative to CMAKE_INSTALL_PREFIX which we MUST NOT touch.
#######################################################################
SET(NIFTK_INSTALL_BASE_DIR ".")
SET(NIFTK_INSTALL_BIN_DIR "bin")
SET(NIFTK_INSTALL_INCLUDE_DIR "include/NifTK")
SET(NIFTK_INSTALL_MATLAB_DIR "matlab")
SET(NIFTK_INSTALL_DOC_DIR "doc")
SET(NIFTK_INSTALL_LIB_DIR "bin")

######################################################################
# Configure files that need variables substituting. Note that
# we 'Configure' them, which copies them to the CMAKE_BINARY_DIR
# while substituting variables, but it is the 'INSTALL' directives
# that place them in the installation directory.
######################################################################

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/INSTALLATION.txt ${CMAKE_BINARY_DIR}/INSTALLATION.txt @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/LICENSE.txt ${CMAKE_BINARY_DIR}/LICENSE.txt @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/LICENSE.txt ${CMAKE_BINARY_DIR}/Doxygen/LICENSE.dox @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/README.txt ${CMAKE_BINARY_DIR}/README.txt @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CONTRIBUTORS.txt ${CMAKE_BINARY_DIR}/CONTRIBUTORS.txt @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Config/SetupDependencies.sh.in ${CMAKE_BINARY_DIR}/SetupDependencies.sh @ONLY )
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Config/SetupDependencies.csh.in ${CMAKE_BINARY_DIR}/SetupDependencies.csh @ONLY )
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Config/SetupNifTK.sh.in ${CMAKE_BINARY_DIR}/SetupNifTK.sh @ONLY )
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Config/SetupNifTK.csh.in ${CMAKE_BINARY_DIR}/SetupNifTK.csh @ONLY )
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Config/NifTKConfigure.h.in ${CMAKE_BINARY_DIR}/NifTKConfigure.h)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Doc/Doxygen/niftkdoxygen.pl.in ${CMAKE_BINARY_DIR}/niftkdoxygen.pl)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Doc/Doxygen/doxygen.config.in ${CMAKE_BINARY_DIR}/doxygen.config)
IF(NOT APPLE)
  FILE(COPY ${CMAKE_SOURCE_DIR}/Doc/Images/Blank.png DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
ENDIF(NOT APPLE)

#############################################################################################
# These represent the libraries that the apps, libraries and unit tests actually link against
#############################################################################################

SET(NIFTK_ITK_LIB ITKCommon ITKIO ITKBasicFilters ITKAlgorithms itkzlib)
SET(NIFTK_VTK_LIBS_BUT_WITHOUT_QT vtkIO vtkCommon vtkGraphics vtkRendering )
SET(NIFTK_VTK_LIBS_WITH_QT vtkRendering QVTK)

######################################################################
# Install commands for things like README, licenses etc.
######################################################################

INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Images/Blank.png DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Boost.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/ITK.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/VTK.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Qt.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/MITK.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/CTK_LICENSE.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyReg.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyRec.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySeg.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySim.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR}/licenses COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupDependencies.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupDependencies.csh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupNifTK.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupNifTK.csh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/LICENSE.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/README.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/INSTALLATION.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/CONTRIBUTORS.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)

##########################################################################
# Install commands for the NiftyReg, NiftySeg, NiftyRec, NiftySim packages
##########################################################################

# NiftyReg
IF(BUILD_NIFTYREG AND NIFTYREG_FOUND)
  FILE(GLOB NIFTY_REG_LIB_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREG/lib/*.so)
  INSTALL(PROGRAMS ${NIFTY_REG_LIB_FILES} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT applications)

  FILE(GLOB NIFTY_REG_BIN_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREG/bin/*)
  INSTALL(PROGRAMS ${NIFTY_REG_BIN_FILES} DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

  FILE(GLOB NIFTY_REG_INCL_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREG/include/*)
  INSTALL(PROGRAMS ${NIFTY_REG_INCL_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/include COMPONENT applications)
ENDIF(BUILD_NIFTYREG AND NIFTYREG_FOUND)

# NiftySeg
IF(BUILD_NIFTYSEG)
  FILE(GLOB NIFTY_SEG_LIB_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/lib/*.so)
  INSTALL(PROGRAMS ${NIFTY_SEG_LIB_FILES} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT applications)

  FILE(GLOB NIFTY_SEG_BIN_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/bin/*)
  INSTALL(PROGRAMS ${NIFTY_SEG_BIN_FILES} DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

  FILE(GLOB NIFTY_SEG_INCL_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/include/*)
  INSTALL(PROGRAMS ${NIFTY_SEG_INCL_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/include COMPONENT applications)

  FILE(GLOB NIFTY_SEG_PRIORS_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/priors/*)
  INSTALL(PROGRAMS ${NIFTY_SEG_PRIORS_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/priors COMPONENT applications)
ENDIF(BUILD_NIFTYSEG)

# NiftyRec
IF(BUILD_NIFTYREC AND NIFTYREC_FOUND)
  FILE(GLOB NIFTY_REC_LIB_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREC/lib/*)
  INSTALL(PROGRAMS ${NIFTY_REC_LIB_FILES} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT applications)

  FILE(GLOB NIFTY_REC_INCL_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREC/include/*)
  INSTALL(PROGRAMS ${NIFTY_REC_INCL_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/include COMPONENT applications)
ENDIF(BUILD_NIFTYREC AND NIFTYREC_FOUND)

# NiftySim
IF(BUILD_NIFTYSIM AND NIFTYSIM_FOUND)
  FILE(GLOB NIFTY_SIM_LIB_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSIM/lib/*)
  INSTALL(PROGRAMS ${NIFTY_SIM_LIB_FILES} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT applications)

  FILE(GLOB NIFTY_SIM_BIN_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSIM/bin/*)
  INSTALL(PROGRAMS ${NIFTY_SIM_BIN_FILES} DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

  FILE(GLOB NIFTY_SIM_INCL_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSIM/include/*)
  INSTALL(PROGRAMS ${NIFTY_SIM_INCL_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/include COMPONENT applications)
ENDIF(BUILD_NIFTYSIM AND NIFTYSIM_FOUND)

#########################################################################
# If we are building with shared libs, then the command line apps need
# all the libraries copying into the right place. At the moment we
# need all the Boost, ITK, VTK, GDCM. MITK and CTK are only used in the
# GUI. If we wanted to use MITK in the command line apps, then we should 
# do something a bit clever to copy over the MITK stuff. 
#########################################################################

IF(BUILD_SHARED_LIBS)

  IF(WIN32)

    ##############Debug Build##############
    IF(CMAKE_BUILD_TYPE MATCHES "Debug")

      SET(TMP_VTK_LIBRARY_DIR "${VTK_LIBRARY_DIRS}/Debug")
      SET(TMP_ITK_LIBRARY_DIR "${ITK_LIBRARY_DIRS}/Debug")

    ENDIF(CMAKE_BUILD_TYPE MATCHES "Debug")
    ##############Debug Build##############
 
    ##############Release Build##############
    IF(CMAKE_BUILD_TYPE MATCHES "Release")

      SET(TMP_VTK_LIBRARY_DIR "${VTK_LIBRARY_DIRS}/Release")
      SET(TMP_ITK_LIBRARY_DIR "${ITK_LIBRARY_DIRS}/Release")

    ENDIF(CMAKE_BUILD_TYPE MATCHES "Release")
    ##############Release Build##############

    # This is for vtk, to copy over the DLLs
    FOREACH(vtkdir ${TMP_VTK_LIBRARY_DIR})
      FILE(GLOB vtkdlls ${vtkdir}/*.dll)
      FOREACH(vtkdll ${vtkdlls})
        INSTALL(FILES ${vtkdll} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
      ENDFOREACH(vtkdll)
    ENDFOREACH(vtkdir ${TMP_VTK_LIBRARY_DIR})
	  
    # This is for itk, to copy over the DLLs
    FOREACH(itkdir ${TMP_ITK_LIBRARY_DIR})
      FILE(GLOB itkdlls ${itkdir}/*.dll)
      FOREACH(itkdll ${itkdlls})
        INSTALL(FILES ${itkdll} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
      ENDFOREACH(itkdll)
    ENDFOREACH(itkdir ${TMP_ITK_LIBRARY_DIR})
        
  ENDIF(WIN32)

  FOREACH(vtkdir ${VTK_LIBRARY_DIRS})
    IF(UNIX)
      IF(APPLE)
        FILE(GLOB vtklibs ${vtkdir}/*.dylib)
      ELSE(APPLE)
        FILE(GLOB vtklibs ${vtkdir}/*.so*)
      ENDIF(APPLE)
    ELSE(UNIX)
      FILE(GLOB vtklibs ${vtkdir}/*.lib)
    ENDIF(UNIX)
    FOREACH(vtklib ${vtklibs})
      INSTALL(FILES ${vtklib} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
    ENDFOREACH(vtklib)
  ENDFOREACH(vtkdir)

  IF(WIN32)
	  SET(TMP_VTK_BIN_DIR "${VTK_LIBRARY_DIRS}/../../bin")
	  FOREACH(vtkdir ${TMP_VTK_BIN_DIR})
      FILE(GLOB vtkdlls ${vtkdir}/*.dll)
	    FOREACH(vtkdll ${vtkdlls})
	      INSTALL(FILES ${vtkdll} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
	    ENDFOREACH(vtkdll)
	  ENDFOREACH(vtkdir ${TMP_VTK_BIN_DIR})
  ENDIF(WIN32)
    
  FOREACH(itkdir ${ITK_LIBRARY_DIRS})
    IF(UNIX)
      IF(APPLE)
        FILE(GLOB itklibs ${itkdir}/*.dylib)
      ELSE(APPLE)
        FILE(GLOB itklibs ${itkdir}/*.so*)
      ENDIF(APPLE)
    ELSE(UNIX)
      FILE(GLOB itklibs ${itkdir}/*.dll)
    ENDIF(UNIX)
    FOREACH(itklib ${itklibs})
      INSTALL(FILES ${itklib} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
    ENDFOREACH(itklib)
  ENDFOREACH(itkdir)

  IF(WIN32)
    SET(TMP_ITK_BIN_DIR "${ITK_LIBRARY_DIRS}/../../bin")
    FOREACH(itkdir ${TMP_ITK_BIN_DIR})
      FILE(GLOB itkdlls ${itkdir}/*.dll)
      FOREACH(itkdll ${itkdlls})
        INSTALL(FILES ${itkdll} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
      ENDFOREACH(itkdll)
    ENDFOREACH(itkdir ${TMP_ITK_BIN_DIR})
  ENDIF(WIN32)

  FOREACH(gdcmdir ${GDCM_LIBRARY_DIRS})
    IF(UNIX)
      IF(APPLE)
        FILE(GLOB gdcmlibs ${gdcmdir}/*.dylib)
      ELSE(APPLE)
        FILE(GLOB gdcmlibs ${gdcmdir}/*.so*)
      ENDIF(APPLE)
    ELSE(UNIX)
      FILE(GLOB gdcmlibs ${gdcmdir}/*.lib)
    ENDIF(UNIX)
    FOREACH(gdcmlib ${gdcmlibs})
      INSTALL(FILES ${gdcmlib} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
    ENDFOREACH(gdcmlib)
  ENDFOREACH(gdcmdir)

  IF(WIN32)
    FIND_PACKAGE(GDCM)
    IF(GDCM_FOUND)
      IF(CMAKE_BUILD_TYPE MATCHES "Release")
        SET(TMP_GDCM_BIN_DIR "${GDCM_LIBRARY_DIRS}/Release")
      ELSE(CMAKE_BUILD_TYPE MATCHES "Release")
        SET(TMP_GDCM_BIN_DIR "${GDCM_LIBRARY_DIRS}/Debug")
      ENDIF(CMAKE_BUILD_TYPE MATCHES "Release")
      FOREACH(gdcmdir ${TMP_GDCM_BIN_DIR})
        FILE(GLOB gdcmdlls ${gdcmdir}/*.dll)
        FOREACH(gdcmdll ${gdcmdlls})
          INSTALL(FILES ${gdcmdll} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
        ENDFOREACH(gdcmdll)
      ENDFOREACH(gdcmdir ${TMP_GDCM_BIN_DIR})
    ELSE(GDCM_FOUND)
      MESSAGE("GDCM not found. So, NiftyView will not run.")
    ENDIF(GDCM_FOUND)
  ENDIF(WIN32)


  IF(NIFTK_USE_FFTW)
    GET_FILENAME_COMPONENT(fftwlibbase ${FFTW_LIBRARIES} NAME_WE)
    GET_FILENAME_COMPONENT(fftwlibdir ${FFTW_LIBRARIES} PATH)
    FILE(GLOB fftwlibs ${fftwlibdir}/${fftwlibbase}*)
    FOREACH(fftwlib ${fftwlibs})
      INSTALL(FILES ${fftwlib} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
    ENDFOREACH(fftwlib) 
  ENDIF(NIFTK_USE_FFTW)
  
  FOREACH(boostlib ${Boost_LIBRARIES})
    GET_FILENAME_COMPONENT(boostlibbase ${boostlib} NAME_WE)
    GET_FILENAME_COMPONENT(boostlibdir ${boostlib} PATH)
    FILE(GLOB boostlibraries ${boostlibdir}/${boostlibbase}*)
    FOREACH(boostlibrary ${boostlibraries})
      INSTALL(FILES ${boostlibrary} DESTINATION  ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
    ENDFOREACH(boostlibrary)
  ENDFOREACH(boostlib)

  ############################################################################################
  # Trac 1497: Getting NiftyLink properly working within NiftyView, fixing installers.
  #            The whole of this section, copying libraries around needs sorting out.
  ############################################################################################
  IF(NiftyLink_FOUND)
    SET(_subdir)
    IF(WIN32)
      SET(_subdir ${CMAKE_BUILD_TYPE})
    ENDIF()
    FOREACH(niftylinklibrary ${NiftyLink_LIBRARIES})
      FOREACH(niftklinkdir ${NiftyLink_LIBRARY_DIRS})
        FILE(GLOB niftylinklibs ${niftklinkdir}/${_subdir}/*${niftylinklibrary}*)
        FOREACH(niftylinklib ${niftylinklibs})
          INSTALL(FILES ${niftylinklib} DESTINATION ${NIFTK_INSTALL_LIB_DIR} COMPONENT libraries)
        ENDFOREACH()
      ENDFOREACH()
    ENDFOREACH()
  ENDIF()
  
ELSE(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTK_STATIC)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

######################################################################
# Decide what subdirectories we are building, and go and build them.
######################################################################

SUBDIRS(Doc)
SUBDIRS(Code)

IF(BUILD_TESTING) 
  SUBDIRS(Testing)
ENDIF(BUILD_TESTING)   

IF(EXISTS "${CMAKE_SOURCE_DIR}/Prototype" AND IS_DIRECTORY "${CMAKE_SOURCE_DIR}/Prototype" AND BUILD_PROTOTYPE) 
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/CUDA)
  
  # Note that for these Insight Journal extensions, we dont compile the subpackage,
  # but by adding them to the include path, you can use any templated classes easily.
  # You may however have to recompile your ITK using ITK_USE_REVIEW=ON and ITK_USE_OPTIMIZED_REGISTRATION_METHODS=ON
  # depending on which classes you use.

  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/InsightJournal/Filters)
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/InsightJournal/IJ_181_ITKbinaryThinningImageFilter3D/Source)
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/InsightJournal/IJ_120_ITKSkeleton)
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/InsightJournal/IJ_644_ITKLogDomainDemonsRegistration/LogDomainDemonsRegistration-0.0.4-Source/Code)
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/InsightJournal/IJ_687_ITKSphericalDemons/QuadEdgeMeshFieldSmoothingFilters/Source)
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/InsightJournal/IJ_687_ITKSphericalDemons/QuadEdgeMeshRigidRegistration/Source)

  SUBDIRS(Prototype)
ENDIF()   

######################################################################
# Packaging code.
######################################################################

# 1. Setup defaults, common for all generators.
INCLUDE(CPackSetup)

# 2. Set variables that may be platform (Windows/Linux/Mac) or Generator (TGZ,DEB,NSIS) specific.
#    When CPack runs, it just uses all the information in the generated files cmake_install.cmake.
#    So, CPack does not read all your configuration information in CMakeLists.txt, and CPack
#    does not re-run any cmake process.  So, it is cmake that reads all the CMakeLists. files
#    and generates all the cmake_install.cmake.  So this command will use cmake to generate
#    and additional file, that we can politely ask cpack to include, in addition to all the
#    cmake_install.cmake files.

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/CPackOptions.cmake.in
               ${CMAKE_BINARY_DIR}/NIFTKCPackOptions.cmake)

# 3. Set a variable with the name of this file.		   
SET(CPACK_PROJECT_CONFIG_FILE "${CMAKE_BINARY_DIR}/NIFTKCPackOptions.cmake")

# 4. Include this optional file.
INCLUDE(NIFTKCPackOptions)

# 5. Include CPack module once all variables are set. i.e. this must be last.
INCLUDE(CPack)
  
# 6. Trac #1264, fixing installer to include Qt Assistant for help files.

SET(MY_PROJECT_NAME NifTK)
SET(MY_APP_NAME NiftyView)
SET(MACOSX_BUNDLE_NAMES)
IF(APPLE)
  LIST(APPEND MACOSX_BUNDLE_NAMES ${MY_APP_NAME})
ENDIF(APPLE)
INCLUDE(mitkInstallRules)

######################################################################
# If we are under Windows, create two batch files which correctly
# set up the environment for the application and for Visual Studio
######################################################################
IF(WIN32)
  IF(MITK_FOUND)
  
    include(mitkFunctionCreateWindowsBatchScript)

    set(VS_SOLUTION_FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.sln")
    foreach(VS_BUILD_TYPE debug release)
      mitkFunctionCreateWindowsBatchScript("${CMAKE_SOURCE_DIR}/CMake/StartVS.bat.in"
        ${PROJECT_BINARY_DIR}/StartVS_${VS_BUILD_TYPE}.bat
        ${VS_BUILD_TYPE})
    endforeach()
    
  ENDIF(MITK_FOUND)
ENDIF(WIN32)


