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

###########################################################################
# Our version number. Edit this to generate a new version.
# However, be warned, you need to set CMAKE_INSTALL_PREFIX manually.
# Also, Trac 1592, version number must change if CTK changes significantly.
###########################################################################
SET(NIFTK_VERSION_MAJOR 13 CACHE STRING "Year version number" FORCE )
SET(NIFTK_VERSION_MINOR 03 CACHE STRING "Month version number" FORCE )
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

IF(BUILD_SUPERBUILD)
  PROJECT(NIFTK-SUPERBUILD)
ELSE(BUILD_SUPERBUILD)
  PROJECT(NIFTK)
ENDIF(BUILD_SUPERBUILD)

######################################################################
# Configure GUI options from the choice of applications.
# The user cannot be expected to know the component parts of apps.
######################################################################

INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/Code/Gui/MITK/Apps/Apps.cmake")
SET(BUILD_GUI OFF)
SET(BUILD_IGI OFF)

# Please leave this off, as we can then JUST build command line apps, without
# filling the CMakeCache.txt with GUI related variables which will only confuse
# when someone subsequently tries to switch all the GUIs off.
OPTION(NIFTK_BUILD_ALL_APPS "Build all NifTK Graphical User Interfaces" OFF) 

# Check each GUI application.
FOREACH(NIFTK_APP ${NIFTK_APPS})
  
  # extract option_name
  STRING(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
  SET(target_info_list ${target_info})
  LIST(GET target_info_list 1 option_name)
  LIST(GET target_info_list 0 app_name)

  IF(${option_name} OR NIFTK_BUILD_ALL_APPS)

    SET(BUILD_GUI ON)
    
    IF (${app_name} STREQUAL "NiftyIGI")
      SET(BUILD_IGI ON)
    ENDIF()
  
  ENDIF()
      
ENDFOREACH()

######################################################################
# Options. These are set up front, so are available when configuring
# the SuperBuild, and hence they must also be passed to the normal
# build. So, look in CMake/Superbuild.cmake to see where they are
# passed to the main build of NifTK when doing the Superbuild.
######################################################################
OPTION(BUILD_COMMAND_LINE_PROGRAMS "Build command line applications in NifTK/Code/Applications." ON)
OPTION(BUILD_COMMAND_LINE_SCRIPTS "Build command line scripts in NifTK/Code/Scripts." ON)
OPTION(BUILD_NIFTYREG "Build NiftyReg." ON)
OPTION(BUILD_NIFTYREC "Build NiftyRec." OFF)
OPTION(BUILD_NIFTYSIM "Build NiftySim." OFF)
OPTION(BUILD_NIFTYSEG "Build NiftySeg." ON)
OPTION(BUILD_PROTOTYPE "Build prototype code in NifTK/Prototype." OFF)
OPTION(BUILD_SHARED_LIBS "Build NifTK with shared libraries." ON)
OPTION(BUILD_TESTING "Build Unit tests in NifTK/Testing/Code." OFF)
OPTION(NIFTK_CHECK_COVERAGE "Enable/Disable code coverage checking." OFF)
IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  SET(NIFTK_SHOW_CONSOLE_WINDOW_VAL ON)
ELSE()
  SET(NIFTK_SHOW_CONSOLE_WINDOW_VAL OFF)
ENDIF()
OPTION(NIFTK_SHOW_CONSOLE_WINDOW "Use this to enable or disable the console window when starting GUI Applications on Windows" ${NIFTK_SHOW_CONSOLE_WINDOW_VAL})
OPTION(NIFTK_BUILD_ALL_PLUGINS "Build all NIFTK plugins" OFF)
OPTION(NIFTK_GENERATE_DOXYGEN_HELP "Use this to generate Doxygen help in GUI" OFF)
OPTION(NIFTK_VERBOSE_COMPILER_WARNINGS "Add in all the warning flags that MITK does" OFF)
OPTION(NIFTK_USE_FFTW "Use FFTW. Warning FFTW is GPL, so for binary external releases we can't use this. This is used for Kelvin's Fluid registration implementation." OFF)
OPTION(NIFTK_USE_CUDA "Use CUDA. Experimental. " OFF)
OPTION(NIFTK_NIFTYLINK_DEV "Build NiftyLink development branch" OFF)

######################################################################
# Meshing module-related checks and options
######################################################################
INCLUDE (CheckIncludeFiles)

CHECK_INCLUDE_FILES(mpfr.h NIFTK_MESHING_CAN_BUILD)
IF (NIFTK_MESHING_CAN_BUILD) 
  OPTION(BUILD_MESHING "Build meshing?" OFF)
ELSE (NIFTK_MESHING_CAN_BUILD) 
  MESSAGE("BUILD_MESHING option disabled")
ENDIF (NIFTK_MESHING_CAN_BUILD) 

######################################################################
# Variables that get compiled into C++ code. 
######################################################################
SET(NIFTK_PLATFORM "NifTK" CACHE STRING "Full name of platform." FORCE )
SET(NIFTK_LICENSE_SHORT_STRING "Not licensed for use outside of UCL." CACHE STRING "Short description of license" FORCE )
SET(NIFTK_COPYRIGHT "Copyright (C) 2008-2012 University College London (UCL). All rights reserved." CACHE STRING "Copyright string." FORCE )
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
SET(NIFTK_BUILD_INSTRUCTIONS "http://cmic.cs.ucl.ac.uk/platform/niftk/current/html/BuildInstructions.html" CACHE STRING "Build instructions")
SET(NIFTK_INSTALL_INSTRUCTIONS "http://cmic.cs.ucl.ac.uk/platform/niftk/current/html/InstallationInstructions.html" CACHE STRING "Installation instructions")
SET(NIFTK_ARCHTYPE_DATA "tar" CACHE STRING "Archive type NifTKData (git, svn or tar)" FORCE)
SET(NIFTK_VERSION_DATA_GIT "d9fe07385c" CACHE STRING "Version of NifTKData in git" FORCE)
SET(NIFTK_VERSION_DATA_SVN "9561" CACHE STRING "Version of NifTKData in svn" FORCE)
SET(NIFTK_VERSION_DATA_TAR "d9fe07385c" CACHE STRING "Version of NifTKData in tarball" FORCE)
SET(NIFTK_VERSION_BOOST "1.46.1" CACHE STRING "Version of Boost" FORCE)
SET(NIFTK_VERSION_GDCM "2.0.18" CACHE STRING "Version of GDCM" FORCE)
SET(NIFTK_VERSION_DCMTK "3.6.1_20120222" CACHE STRING "Version of DCMTK" FORCE)
SET(NIFTK_VERSION_ITK "3.20.1.1" CACHE STRING "Version of ITK" FORCE)
SET(NIFTK_VERSION_VTK "5.8.0" CACHE STRING "Version of VTK" FORCE)
# Note: If the CTK version changes, then you either clear the plugin cache
# or change the deploy path by changing the patch level.
SET(NIFTK_VERSION_CTK "dd4386a4b0" CACHE STRING "Version of CTK" FORCE)
SET(NIFTK_VERSION_MITK "62e2e5ba98" CACHE STRING "Version of MITK" FORCE)
SET(NIFTK_VERSION_curl "7.28.git" CACHE STRING "Version of curl" FORCE)
SET(NIFTK_VERSION_NIFTYLINK "f87aa369bc" CACHE STRING "Version of NiftyLink" FORCE)
SET(NIFTK_VERSION_NIFTYREG "360" CACHE STRING "Version of NiftyReg" FORCE)
SET(NIFTK_VERSION_NIFTYSEG "122" CACHE STRING "Version of NiftySeg" FORCE)
SET(NIFTK_VERSION_NIFTYREC "14" CACHE STRING "Version of NiftyRec" FORCE)
SET(NIFTK_VERSION_NIFTYSIM "127" CACHE STRING "Version of NiftySim" FORCE)
SET(NIFTK_VERSION_CGAL "3.8" CACHE STRING "Version of CGAL" FORCE)
SET(NIFTK_VERSION_SEM "7365853e2b" CACHE STRING "Version of Slicer Execution Module package" FORCE)
SET(NIFTK_LOCATION_DATA_GIT "https://cmicdev.cs.ucl.ac.uk/git/NifTKData" CACHE STRING "Version of NifTKData in git" FORCE)
SET(NIFTK_LOCATION_DATA_SVN "https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NifTKData" CACHE STRING "Version of NifTKData in svn" FORCE)
SET(NIFTK_LOCATION_DATA_TAR "http://cmic.cs.ucl.ac.uk/platform/dependencies/NifTKData-${NIFTK_VERSION_DATA_TAR}.tar.gz" CACHE STRING "Location of NifTKData" FORCE) 
SET(NIFTK_LOCATION_BOOST "http://cmic.cs.ucl.ac.uk/platform/dependencies/boost-${NIFTK_VERSION_BOOST}.tar.gz" CACHE STRING "Location of Boost" FORCE)
SET(NIFTK_LOCATION_GDCM "http://cmic.cs.ucl.ac.uk/platform/dependencies/gdcm-${NIFTK_VERSION_GDCM}.tar.gz" CACHE STRING "Location of GDCM" FORCE)
SET(NIFTK_LOCATION_DCMTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/dcmtk-${NIFTK_VERSION_DCMTK}.tar.gz" CACHE STRING "Location of DCMTK" FORCE)
SET(NIFTK_LOCATION_ITK "http://cmic.cs.ucl.ac.uk/platform/dependencies/InsightToolkit-${NIFTK_VERSION_ITK}.tar.gz" CACHE STRING "Location of ITK" FORCE)
SET(NIFTK_LOCATION_VTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/vtk-${NIFTK_VERSION_VTK}.tar.gz" CACHE STRING "Location of VTK" FORCE)
SET(NIFTK_LOCATION_CTK "http://cmic.cs.ucl.ac.uk/platform/dependencies/NifTK-CTK-${NIFTK_VERSION_CTK}.tar.gz" CACHE STRING "Location of CTK" FORCE)
SET(NIFTK_LOCATION_MITK "http://cmic.cs.ucl.ac.uk/platform/dependencies/NifTK-MITK-${NIFTK_VERSION_MITK}.tar.gz" CACHE STRING "Location of MITK" FORCE)
SET(NIFTK_LOCATION_curl "http://cmic.cs.ucl.ac.uk/platform/dependencies/curl-${NIFTK_VERSION_curl}.zip" CACHE STRING "Location of curl" FORCE)
SET(NIFTK_LOCATION_NIFTYLINK_TARBALL "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftyLink-${NIFTK_VERSION_NIFTYLINK}.tar.gz" CACHE STRING "Location of NiftyLink tarball" FORCE)
SET(NIFTK_LOCATION_NIFTYLINK_REPOSITORY "https://cmicdev.cs.ucl.ac.uk/git/NiftyLink" CACHE STRING "Location of NiftyLink repository")
SET(NIFTK_LOCATION_PROTOTYPE "https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NiftyPrototype" CACHE STRING  "Location of prototype code (internal access only)")
SET(NIFTK_LOCATION_NIFTYREG "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftyReg-${NIFTK_VERSION_NIFTYREG}.tar.gz" CACHE STRING  "Location of NiftyReg" FORCE)
SET(NIFTK_LOCATION_NIFTYSEG "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftySeg-${NIFTK_VERSION_NIFTYSEG}.tar.gz" CACHE STRING  "Location of NiftySeg" FORCE)
SET(NIFTK_LOCATION_NIFTYREC "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftyRec-${NIFTK_VERSION_NIFTYREC}.tar.gz" CACHE STRING  "Location of NiftyRec" FORCE)
SET(NIFTK_LOCATION_NIFTYSIM "http://cmic.cs.ucl.ac.uk/platform/dependencies/NiftySim-${NIFTK_VERSION_NIFTYSIM}.tar.gz" CACHE STRING  "Location of NiftyRec" FORCE)
SET(NIFTK_LOCATION_CGAL "http://cmic.cs.ucl.ac.uk/platform/dependencies/CGAL-${NIFTK_VERSION_CGAL}.tar.gz" CACHE STRING  "Location of CGAL" FORCE)
SET(NIFTK_LOCATION_SEM "http://cmic.cs.ucl.ac.uk/platform/dependencies/Slicer-SlicerExecutionModel-${NIFTK_VERSION_SEM}.tar.gz" CACHE STRING  "Location of Slicer Execution Module package" FORCE)

######################################################################
# Hide these variables from the user, unless they are 'advanced' :-)
######################################################################
MARK_AS_ADVANCED(NIFTK_PLATFORM)
MARK_AS_ADVANCED(NIFTK_LICENSE_SHORT_STRING)
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
MARK_AS_ADVANCED(NIFTK_BUILD_ALL_PLUGINS)
MARK_AS_ADVANCED(NIFTK_CHECK_COVERAGE)
MARK_AS_ADVANCED(NIFTK_VERBOSE_COMPILER_WARNINGS)
MARK_AS_ADVANCED(NIFTK_GENERATE_DOXYGEN_HELP)
MARK_AS_ADVANCED(NIFTK_NIGHTLY_DOCS)
MARK_AS_ADVANCED(NIFTK_BUILD_INSTRUCTIONS)
MARK_AS_ADVANCED(NIFTK_INSTALL_INSTRUCTIONS)
MARK_AS_ADVANCED(NIFTK_ARCHTYPE_DATA)
MARK_AS_ADVANCED(NIFTK_VERSION_DATA_GIT)
MARK_AS_ADVANCED(NIFTK_VERSION_DATA_SVN)
MARK_AS_ADVANCED(NIFTK_VERSION_DATA_TAR)
MARK_AS_ADVANCED(NIFTK_VERSION_BOOST)
MARK_AS_ADVANCED(NIFTK_VERSION_GDCM)
MARK_AS_ADVANCED(NIFTK_VERSION_DCMTK)
MARK_AS_ADVANCED(NIFTK_VERSION_ITK)
MARK_AS_ADVANCED(NIFTK_VERSION_VTK)
MARK_AS_ADVANCED(NIFTK_VERSION_CTK)
MARK_AS_ADVANCED(NIFTK_VERSION_MITK)
MARK_AS_ADVANCED(NIFTK_VERSION_curl)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYLINK)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYREG)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYSEG)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYREC)
MARK_AS_ADVANCED(NIFTK_VERSION_NIFTYSIM)
MARK_AS_ADVANCED(NIFTK_VERSION_CGAL)
MARK_AS_ADVANCED(NIFTK_VERSION_SEM)
MARK_AS_ADVANCED(NIFTK_LOCATION_DATA_GIT)
MARK_AS_ADVANCED(NIFTK_LOCATION_DATA_SVN)
MARK_AS_ADVANCED(NIFTK_LOCATION_DATA_TAR)
MARK_AS_ADVANCED(NIFTK_LOCATION_BOOST)
MARK_AS_ADVANCED(NIFTK_LOCATION_GDCM)
MARK_AS_ADVANCED(NIFTK_LOCATION_DCMTK)
MARK_AS_ADVANCED(NIFTK_LOCATION_ITK)
MARK_AS_ADVANCED(NIFTK_LOCATION_VTK)
MARK_AS_ADVANCED(NIFTK_LOCATION_CTK)
MARK_AS_ADVANCED(NIFTK_LOCATION_MITK)
MARK_AS_ADVANCED(NIFTK_LOCATION_curl)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYLINK_TARBALL)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYLINK_REPOSITORY)
MARK_AS_ADVANCED(NIFTK_LOCATION_PROTOTYPE)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYREG)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYSEG)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYREC)
MARK_AS_ADVANCED(NIFTK_LOCATION_NIFTYSIM)
MARK_AS_ADVANCED(NIFTK_LOCATION_CGAL)
MARK_AS_ADVANCED(NIFTK_LOCATION_SEM)
MARK_AS_ADVANCED(NIFTK_USE_FFTW)
MARK_AS_ADVANCED(NIFTK_USE_CUDA)

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
include(niftkMacroCreateApplication)
include(niftkMacroBuildCLI)
include(niftkMacroInstallCLI)
include(niftkMacroConfigureAndBuildCLI)

######################################################################
# Test for some required system information. This came from old
# ITK CMake scripts, so I (Matt) am unsure if we still need it.
######################################################################
INCLUDE(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityC.cmake)
INCLUDE(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityCXX.cmake)

######################################################################
# Set GIT protocol, as dependencies need it. (Obselete???)
######################################################################
FIND_PACKAGE(Git REQUIRED)
IF (WIN32)
  SET(GITCOMMAND ${GIT_EXECUTABLE})
ENDIF()

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
INCLUDE(${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake)
CHECK_FUNCTION_EXISTS(mkstemps HAVE_MKSTEMPS) 
CHECK_INCLUDE_FILES("sys/socket.h" HAVE_SYS_SOCKET_H)
IF(HAVE_SYS_SOCKET_H)
  ADD_DEFINITIONS("-DHAVE_SYS_SOCKET_H")
ENDIF()

###########################################################################
# Set these compiler flags early, so it can be applied to all dependencies.
###########################################################################

SET(NIFTK_ADDITIONAL_C_FLAGS "" CACHE STRING "Additional C Flags")
MARK_AS_ADVANCED(NIFTK_ADDITIONAL_C_FLAGS)
SET(NIFTK_ADDITIONAL_CXX_FLAGS "" CACHE STRING "Additional CXX Flags")
MARK_AS_ADVANCED(NIFTK_ADDITIONAL_CXX_FLAGS)

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
    MESSAGE(FATAL_ERROR "If you wan't to build a GUI, you must use dynamic linking, turn BUILD_SHARED to ON.")
  ENDIF(NOT BUILD_SHARED_LIBS)
ENDIF(BUILD_GUI)

IF(BUILD_GUI)
  SET(NIFTK_GENERATE_DOXYGEN_HELP ON)
ENDIF(BUILD_GUI)

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
  
    SET(QT_USE_QTXML 1)
    SET(QT_USE_QTXMLPATTERNS 1)
    SET(QT_USE_PHONON 0)

    INCLUDE(${QT_USE_FILE})
  
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
# Find NVidia API: Required for NVidia SDI pipeline in IGI GUI
######################################################################
FIND_PACKAGE(NVidiaAPI)
IF(NVAPI_FOUND)
  MESSAGE("Found NVidia API")
  INCLUDE_DIRECTORIES(${NVAPI_INCLUDE_DIR})
  ADD_DEFINITIONS(-D_USE_NVAPI)
ELSE()
  MESSAGE("Did not find NVidia API")
ENDIF()

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
# are, just to make it really explicit... well, explicit to the user
# that bothers to read these messages! :-)
######################################################################

MESSAGE("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
MESSAGE("CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

######################################################################
# Check each GUI application to set up MACOSX_BUNDLE_NAMES
# as libraries are copied into each bundle.
######################################################################

IF(APPLE)

  SET(MACOSX_BUNDLE_NAMES)
  INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/Code/Gui/MITK/Apps/Apps.cmake")

  FOREACH(NIFTK_APP ${NIFTK_APPS})
    
    # extract option_name
    STRING(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
    SET(target_info_list ${target_info})
    LIST(GET target_info_list 1 option_name)
    LIST(GET target_info_list 0 app_name)

    IF(${option_name} OR NIFTK_BUILD_ALL_APPS)
      SET(MACOSX_BUNDLE_NAMES ${MACOSX_BUNDLE_NAMES} ${app_name})
    ENDIF()
          
  ENDFOREACH()
  
ENDIF()

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
IF (BUILD_IGI)
  FIND_PACKAGE(NiftyLink)
  IF (NiftyLink_FOUND)
    MESSAGE("Found NiftyLink")
    INCLUDE(${NiftyLink_USE_FILE})
    ADD_DEFINITIONS(-DBUILD_IGI)
    MESSAGE("BUILD_IGI=ON, so turning surgical guidance plugin ON.")
    SET("NIFTK_Plugins/uk.ac.ucl.cmic.surgicalguidance" ON CACHE BOOL "Surgical Guidance plugin" FORCE)
  ELSE (NOT NiftyLink_FOUND)
    MESSAGE("BUILD_IGI=ON, but no NiftyLink found, so setting BUILD_IGI to OFF.")
    SET("BUILD_IGI" OFF CACHE BOOL "Build CMIC IGI platform." FORCE)
  ENDIF(NiftyLink_FOUND)
ELSE (BUILD_IGI)
  MESSAGE("BUILD_IGI=OFF, so turning surgical guidance plugin OFF.")
  SET("NIFTK_Plugins/uk.ac.ucl.cmic.surgicalguidance" OFF CACHE BOOL "Surgical Guidance plugin" FORCE)
ENDIF (BUILD_IGI)

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


######################################################################
# Find SlicerExecutionModel
######################################################################
IF(BUILD_COMMAND_LINE_PROGRAMS)
  FIND_PACKAGE(SlicerExecutionModel REQUIRED)
  IF(SlicerExecutionModel_FOUND)
    MESSAGE("Found SlicerExecutionModel")
    INCLUDE(${SlicerExecutionModel_USE_FILE})
  ELSE()
    MESSAGE("Didn't find SlicerExecutionModel")
  ENDIF()
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
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/ITK/Numerics)
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
SET(SLICER_EXECUTABLE_OUTPUTPATH ${EXECUTABLE_OUTPUT_PATH}/cli-modules)
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
ENDIF()

IF(WIN32 AND NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DNIFTK_STATIC)
ENDIF()

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

#############################################################################################
# These represent the libraries that the apps, libraries and unit tests actually link against
#############################################################################################

SET(NIFTK_ITK_LIB ITKCommon ITKIO ITKBasicFilters ITKAlgorithms itkzlib)
SET(NIFTK_VTK_LIBS_BUT_WITHOUT_QT vtkIO vtkCommon vtkGraphics vtkRendering )
SET(NIFTK_VTK_LIBS_WITH_QT vtkRendering QVTK)

######################################################################
# Install commands for things like README, licenses etc.
######################################################################

INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Boost.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/ITK.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/VTK.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Qt.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/MITK.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/CTK_LICENSE.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/CTK_NOTICE.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyReg.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyRec.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySeg.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySim.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyLink.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/SlicerExecutionModel.txt DESTINATION ${NIFTK_INSTALL_DOC_DIR} COMPONENT documentation)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/LICENSE.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/README.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/INSTALLATION.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/CONTRIBUTORS.txt DESTINATION ${NIFTK_INSTALL_BASE_DIR} COMPONENT applications)
IF(NOT WIN32 AND NOT APPLE)
  INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupDependencies.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupDependencies.csh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupNifTK.sh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
  INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/SetupNifTK.csh DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
ENDIF()

##########################################################################
# Install commands for the NiftyReg, NiftySeg, NiftyRec, NiftySim packages
##########################################################################

# NiftyReg
IF(BUILD_NIFTYREG AND NIFTYREG_FOUND) 
  FILE(GLOB NIFTY_REG_LIB_FILES 
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREG/lib/*.so
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREG/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_REG_LIB_FILES})

  FILE(GLOB NIFTY_REG_BIN_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREG/bin/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_REG_BIN_FILES})
ENDIF(BUILD_NIFTYREG AND NIFTYREG_FOUND)

# NiftySeg
IF(BUILD_NIFTYSEG) 
  FILE(GLOB NIFTY_SEG_LIB_FILES 
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/lib/*.so
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_SEG_LIB_FILES})

  FILE(GLOB NIFTY_SEG_BIN_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/bin/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_SEG_BIN_FILES})

  FILE(GLOB NIFTY_SEG_PRIORS_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSEG/priors/*)
  INSTALL(PROGRAMS ${NIFTY_SEG_PRIORS_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/priors COMPONENT applications)
ENDIF(BUILD_NIFTYSEG)

# NiftyRec
IF(BUILD_NIFTYREC AND NIFTYREC_FOUND) 
  FILE(GLOB NIFTY_REC_LIB_FILES 
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREC/lib/*.so
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREC/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_REC_LIB_FILES})

  FILE(GLOB NIFTY_REC_INCL_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYREC/include/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_REC_INCL_FILES})
ENDIF(BUILD_NIFTYREC AND NIFTYREC_FOUND)

# NiftySim
IF(BUILD_NIFTYSIM AND NIFTYSIM_FOUND) 
  FILE(GLOB NIFTY_SIM_LIB_FILES 
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSIM/lib/*.so
       ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSIM/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${NIFTY_SIM_LIB_FILES})

  FILE(GLOB NIFTY_SIM_BIN_FILES ${CMAKE_BINARY_DIR}/../CMakeExternals/Install/NIFTYSIM/bin/*)
  MITK_INSTALL(PROGRAMS ${NIFTY_SIM_BIN_FILES})
ENDIF(BUILD_NIFTYSIM AND NIFTYSIM_FOUND)

######################################################################
# Decide what subdirectories we are building, and go and build them.
######################################################################

SUBDIRS(Doc)
SUBDIRS(Code)

IF(BUILD_TESTING) 
  SUBDIRS(Testing)
ENDIF(BUILD_TESTING)   

IF (BUILD_PROTOTYPE)
  SET(PROTOTYPE_DIR ${CMAKE_SOURCE_DIR}/Prototype)

  IF (NOT EXISTS ${PROTOTYPE_DIR})
    FIND_PACKAGE(Subversion)
    IF (Subversion_FOUND)
      MESSAGE("Found Subversion")
      MESSAGE("Retrieving NiftyPrototype...")
      EXECUTE_PROCESS(
        COMMAND "${Subversion_SVN_EXECUTABLE}" checkout "${NIFTK_LOCATION_PROTOTYPE}" Prototype
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
    ELSE()
      MESSAGE("Subversion not found. The Prototype project will not be built.")
    ENDIF()
  ENDIF()

  IF (EXISTS ${PROTOTYPE_DIR} AND IS_DIRECTORY ${PROTOTYPE_DIR})
    INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Prototype/CUDA)

    # Note that for these Insight Journal extensions, we dont compile the subpackage,
    # but by adding them to the include path, you can use any templated classes easily.
    # You may however have to recompile your ITK using ITK_USE_REVIEW=ON and ITK_USE_OPTIMIZED_REGISTRATION_METHODS=ON
    # depending on which classes you use.

    INCLUDE_DIRECTORIES(${PROTOTYPE_DIR}/InsightJournal/Filters)
    INCLUDE_DIRECTORIES(${PROTOTYPE_DIR}/InsightJournal/IJ_181_ITKbinaryThinningImageFilter3D/Source)
    INCLUDE_DIRECTORIES(${PROTOTYPE_DIR}/InsightJournal/IJ_120_ITKSkeleton)
    INCLUDE_DIRECTORIES(${PROTOTYPE_DIR}/InsightJournal/IJ_644_ITKLogDomainDemonsRegistration/LogDomainDemonsRegistration-0.0.4-Source/Code)
    INCLUDE_DIRECTORIES(${PROTOTYPE_DIR}/InsightJournal/IJ_687_ITKSphericalDemons/QuadEdgeMeshFieldSmoothingFilters/Source)
    INCLUDE_DIRECTORIES(${PROTOTYPE_DIR}/InsightJournal/IJ_687_ITKSphericalDemons/QuadEdgeMeshRigidRegistration/Source)

    ADD_SUBDIRECTORY(Prototype)
  ENDIF()
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Configuring the Start Menu and Desktop icon strings for NSIS

if (WIN32)
  SET(ICONS_SETUP_STRING "")
  SET(ICONS_REMOVE_STRING "")
  SET(DESKTOP_ICONS_SETUP_STRING "")
  SET(DESKTOP_ICONS_REMOVE_STRING "")
  SET(DESKTOP_ICONS_REMOVE_STRING "")
 
  FOREACH(NIFTK_APP ${NIFTK_APPS})  
    # extract option_name
    STRING(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
    SET(target_info_list ${target_info})
    LIST(GET target_info_list 1 option_name)
    LIST(GET target_info_list 0 app_name)

	message("Current App Name: ${app_name}")
	
	if(EXISTS "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/${app_name}.exe")
	  SET(ICONS_SETUP_STRING "${ICONS_SETUP_STRING}CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\${app_name}.lnk' '$INSTDIR\\\\bin\\\\${app_name}.exe'\n")
      SET(ICONS_REMOVE_STRING "${ICONS_REMOVE_STRING}Delete '$SMPROGRAMS\\\\$MUI_TEMP\\\\${app_name}.lnk'\n")
	
      SET(DESKTOP_ICONS_SETUP_STRING "${DESKTOP_ICONS_SETUP_STRING}CreateShortCut '$DESKTOP\\\\${app_name}.lnk' '$INSTDIR\\\\bin\\\\${app_name}.exe'\n") 
      SET(DESKTOP_ICONS_REMOVE_STRING "${DESKTOP_ICONS_REMOVE_STRING}Delete '$DESKTOP\\\\${app_name}.lnk'\n")
	endif()
  ENDFOREACH()
endif()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/CPackOptions.cmake.in
               ${CMAKE_BINARY_DIR}/NIFTKCPackOptions.cmake)

# 3. Set a variable with the name of this file.		   
SET(CPACK_PROJECT_CONFIG_FILE "${CMAKE_BINARY_DIR}/NIFTKCPackOptions.cmake")

# 4. Include this optional file.
INCLUDE(NIFTKCPackOptions)

# 5. Include CPack module once all variables are set. i.e. this must be last.
INCLUDE(CPack)
  
# 6. Trac #1796 - Still need to call MITK rules to get Mac Bundles etc.
INCLUDE(mitkInstallRules)

######################################################################
# If we are under Windows, create two batch files which correctly
# set up the environment for the application and for Visual Studio.
# These are only used to start VS when developing. Not used in
# the final installation package.
######################################################################
IF(WIN32)
  IF(MITK_FOUND)
  
    include(mitkFunctionCreateWindowsBatchScript)

    set(VS_SOLUTION_FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.sln")
    foreach(VS_BUILD_TYPE debug release)
      mitkFunctionCreateWindowsBatchScript("${CMAKE_SOURCE_DIR}/CMake/StartVS.bat.in"
        ${PROJECT_BINARY_DIR}/StartVS_${VS_BUILD_TYPE}.bat
        ${VS_BUILD_TYPE})

      MESSAGE( "CreateWindowsBatchScript: Creating ${PROJECT_BINARY_DIR}/StartVS_${VS_BUILD_TYPE}.bat" )

    endforeach()
    
  ELSE(MITK_FOUND)
    MESSAGE( "CreateWindowsBatchScript: MITK not found" )
  ENDIF(MITK_FOUND)
ELSE(WIN32)
  MESSAGE( "CreateWindowsBatchScript: WIN32 not found" )
ENDIF(WIN32)
