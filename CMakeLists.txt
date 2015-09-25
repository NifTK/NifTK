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

######################################################################
# Set the minimum CMake version.
######################################################################
set(NIFTK_CMAKE_MINIMUM_REQUIRED_VERSION 3.2)
cmake_minimum_required(VERSION ${NIFTK_CMAKE_MINIMUM_REQUIRED_VERSION})

##################################################################################
# Set some CMake Policies.
# See http://cmake.org/cmake/help/cmake-2-8-docs.html#section_Policies for details
##################################################################################

set(project_policies )
foreach(policy ${project_policies})
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

###########################################################################
# Precompiled headers
###########################################################################

option(NIFTK_USE_COTIRE "Use Cotire for compilation speed-up." OFF)
if(NIFTK_USE_COTIRE)
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})
  include(cotire)
  mark_as_advanced(COTIRE_ADDITIONAL_PREFIX_HEADER_IGNORE_EXTENSIONS)
  mark_as_advanced(COTIRE_ADDITIONAL_PREFIX_HEADER_IGNORE_PATH)
  mark_as_advanced(COTIRE_DEBUG)
  mark_as_advanced(COTIRE_MAXIMUM_NUMBER_OF_UNITY_INCLUDES)
  mark_as_advanced(COTIRE_MINIMUM_NUMBER_OF_TARGET_SOURCES)
  mark_as_advanced(COTIRE_UNITY_SOURCE_EXCLUDE_EXTENSIONS)
  mark_as_advanced(COTIRE_VERBOSE)
endif()

######################################################################
# We have a super-build option. (Terminology comes from MITK/CTK).
######################################################################

option(BUILD_SUPERBUILD "Build NIFTK and the projects it depends on via SuperBuild.cmake." ON)

if (BUILD_SUPERBUILD)
  project(NIFTK-superbuild)
  set(NIFTK_SOURCE_DIR ${PROJECT_SOURCE_DIR})
  set(NIFTK_BINARY_DIR ${PROJECT_BINARY_DIR})
else()
  # Our version number. Edit this to generate a new version.
  # However, be warned, you need to set CMAKE_INSTALL_PREFIX manually.
  # Also, Trac 1592, version number must change if CTK changes significantly.
  project(NIFTK VERSION 15.09.0)
endif()

######################################################################
# Setting supported build types. Should ONLY be Release or Debug.
######################################################################

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Valid options are Release or Debug" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
endif()

if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
  message(FATAL_ERROR "Build type \"${CMAKE_BUILD_TYPE}\" is not supported.")
endif()

if(WIN32)
  # Restrict the generated configuration to be what we configured above.
  # No point creating project files for build types that will not compile.
  # Note: it's set to FORCE so that both CMAKE_BUILD_TYPE and CMAKE_CONFIGURATION_TYPES match up.
  set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE STRING "Build configurations to generate." FORCE)
  mark_as_advanced(CMAKE_CONFIGURATION_TYPES)
endif()


#-----------------------------------------------------------------------------
# Additional MITK Options (also shown during superbuild)
#-----------------------------------------------------------------------------

macro(env_option name doc value)
  set(_value $ENV{${name}})
  if("${_value}" STREQUAL "")
    set(_value ${value})
  endif()
  option(${name} "${doc}" ${_value})
endmacro()

######################################################################
# Configure GUI options from the choice of applications.
# The user cannot be expected to know the component parts of apps.
######################################################################

include("${CMAKE_CURRENT_SOURCE_DIR}/Code/Gui/MITK/Apps/Apps.cmake")
set(BUILD_GUI OFF)
mark_as_advanced(BUILD_GUI)
set(BUILD_IGI OFF)
mark_as_advanced(BUILD_IGI)
set(BUILD_MIDAS OFF)
mark_as_advanced(BUILD_MIDAS)

# Check each GUI application.
foreach(NIFTK_APP ${NIFTK_APPS})

  # extract option_name
  string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
  set(target_info_list ${target_info})
  list(GET target_info_list 1 option_name)
  list(GET target_info_list 0 app_name)

  if(${option_name})

    set(BUILD_GUI ON)

    if (${app_name} STREQUAL "NiftyIGI")
      set(BUILD_IGI ON)
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
option(BUILD_CAMINO "Build Camino." OFF)
option(BUILD_NiftyReg "Build NiftyReg." ON)
option(BUILD_NiftyRec "Build NiftyRec." OFF)
option(BUILD_NiftySim "Build NiftySim." ON)
option(BUILD_NiftySeg "Build NiftySeg." ON)
option(BUILD_PCL "Build PCL - Experimental." OFF)
option(BUILD_RTK "Build RTK - Experimental." OFF)
option(BUILD_VL "Build VisualizationLibrary - Experimental." OFF)
option(BUILD_ITKFFTW "Build ITK's FFTW - Not Supported." OFF)
option(BUILD_PROTOTYPE "Build prototype code in NifTK/Prototype." OFF)
option(BUILD_SHARED_LIBS "Build NifTK with shared libraries." ON)
option(BUILD_TESTING "Build Unit tests." OFF)
option(BUILD_UTILITIES "Build NifTK/Utilities." OFF)
option(NIFTK_CHECK_COVERAGE "Enable/Disable code coverage checking." OFF)
# the console window is normally shown only for debug builds. but can be explicitly enabled.
#if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(NIFTK_SHOW_CONSOLE_WINDOW_VAL ON)
#else()
#  set(NIFTK_SHOW_CONSOLE_WINDOW_VAL OFF)
#endif()
option(NIFTK_SHOW_CONSOLE_WINDOW "Use this to enable or disable the console window when starting GUI Applications on Windows" ${NIFTK_SHOW_CONSOLE_WINDOW_VAL})
option(NIFTK_BUILD_ALL_PLUGINS "Build all NIFTK plugins" OFF)
option(NIFTK_GENERATE_DOXYGEN_HELP "Use this to generate Doxygen help in GUI" OFF)
option(NIFTK_VERBOSE_COMPILER_WARNINGS "Add in all the warning flags that MITK does" OFF)
option(NIFTK_USE_FFTW "Use FFTW. Warning FFTW is GPL, so for binary external releases we can't use this. This is used for Kelvin's Fluid registration implementation." OFF)
option(NIFTK_USE_CUDA "Use CUDA. Experimental. " OFF)
option(NIFTK_DELAYLOAD_CUDA "(Windows only) If enabled the CUDA DLLs will be a soft delay-load dependency and NifTK can run without them." OFF)
option(NIFTK_NIFTYLINK_DEV "Build NiftyLink development branch" OFF)
option(NIFTK_RESOLVE_COMMAND_LINE_DEPENDENCIES "For make install, for command line apps, we compute all library dependencies and install them" ON)
option(BUILD_MESHING "Build meshing? Experimental, may nor may not work." OFF)

######################################################################
# Meshing module-related checks and options
######################################################################
include (CheckIncludeFiles)

# Meshing components require CGAL, which is a bit fiddly to get right.
if(BUILD_MESHING)
  if(NOT MSVC)
    # On certain Linux variants, the Superbuild can build CGAL for you.
    # This depends on MPFR/GMP, so this header check here will issue a warning
    # if a required file is missing.
    # The Superbuild will then download a patched version of CGAL 4.4. It's
    # been modified only wrt to VTK 6 compatibility.
    CHECK_INCLUDE_FILES(mpfr.h NIFTK_MESHING_CAN_BUILD)
    if (NOT NIFTK_MESHING_CAN_BUILD)
      message(WARNING "Meshing depends on CGAL depends on MPFR but cannot find mpfr.h. Build will likely fail.")
    endif()
  else()
    # On Windows, the situation is significantly more complicated.
    # GMP will not compile in Visual Studio due to lack of C99 support and/or gcc-isms.
    # Therefore you need to:
    # 1. Build a Superbuild with BUILD_MESHING=OFF
    # 2. Download the Windows version of CGAL 4.4 (or newer?) that includes precompiled GMP/MPFR.
    # 3. Apply the VTK 6 patch from Utilities/cgal-vtk6.diff
    # 4. CMake-configure CGAL, enabling WITH_VTK=ON, explicitly pointing VTK_DIR to the correct Superbuild
    #    VTK from step 1, explicitly set BOOST_ROOT to the correct Superbuild Boost from step 1.
    #    BOOST_ROOT may not show up in the cmake gui, set it anyway.
    # 5. Explicitly define CMAKE_INSTALL_PREFIX to a local path of your choice, e.g. .../CGAL-4.4-install,
    #    making sure you only have slashes in the path, no back-slashes.
    # 6. Compile CGAL, and install it via the install target.
    # 7. Reconfigure NifTK, setting BUILD_MESHING=ON and defining CGAL_DIR to .../CGAL-4.4-install/lib/CGAL.
    # 8. Compile, etc. Should now work fine.
    # x. You will need to rebuild CGAL every time VTK or Boost changes.
    if(NOT CGAL_DIR)
      message(WARNING "Meshing depends on CGAL for which you need to explicitly define CGAL_DIR! Configure will probably fail.")
    endif()
  endif()
endif(BUILD_MESHING)

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

# Further variables for the external project names and locations are defined
# in the external project CMake files under CMake/CMakeExternals.

set(NIFTK_EP_TARBALL_LOCATION "http://cmic.cs.ucl.ac.uk/platform/dependencies")

set(NIFTK_LOCATION_PROTOTYPE "https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NiftyPrototype" CACHE STRING  "Location of prototype code (internal access only)")

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
mark_as_advanced(NIFTK_LOCATION_PROTOTYPE)
mark_as_advanced(NIFTK_USE_FFTW)
mark_as_advanced(NIFTK_USE_CUDA)
mark_as_advanced(NIFTK_DELAYLOAD_CUDA)
mark_as_advanced(NIFTK_NIFTYLINK_DEV)
mark_as_advanced(BUILD_SHARED_LIBS)
mark_as_advanced(NIFTK_RESOLVE_COMMAND_LINE_DEPENDENCIES)
mark_as_advanced(BUILD_CAMINO)
mark_as_advanced(BUILD_PCL)
mark_as_advanced(BUILD_RTK)
mark_as_advanced(BUILD_ITKFFTW)
mark_as_advanced(BUILD_MESHING)

######################################################################
# Make sure Git is available.
######################################################################
find_package(Git REQUIRED)
if (WIN32)
  set(GITCOMMAND ${GIT_EXECUTABLE})
endif()

######################################################################
# Check if Doxygen is available for the GUI help pages.
######################################################################
find_package(Doxygen)

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
include(mitkFunctionCheckCompilerFlags)
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

include(mitkFunctionEnableBuildConfiguration)
include(mitkFunctionWhitelists)
include(mitkFunctionAddExternalProject)


# Standard CMake macros
#include(FeatureSummary)
include(CTestUseLaunchers)
#include(CMakeParseArguments)
#include(FindPackageHandleStandardArgs)

# MITK macros
include(mitkFunctionSuppressWarnings) # includes several functions
include(mitkMacroEmptyExternalProject)
#include(mitkFunctionGenerateProjectXml)

SUPPRESS_VC_DEPRECATED_WARNINGS()

######################################################################
# Test for some required system information. This came from old
# ITK CMake scripts, so I (Matt) am unsure if we still need it.
######################################################################
include(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityC.cmake)
include(${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityCXX.cmake)


#-----------------------------------------------------------------------------
# Check miminum Mac OS X version
#-----------------------------------------------------------------------------
# The minimum supported Mac OS X version is 10.9. If you use a version less than 10.9, there is no guarantee that the build still works.
if(APPLE)
  exec_program(sw_vers ARGS -productVersion OUTPUT_VARIABLE osx_version)
  if (osx_version VERSION_LESS "10.10")
    message(WARNING "Detected OS X version \"${osx_version}\" is not supported anymore. Minimum required OS X version is 10.10 or greater.")
  endif()
  if (CMAKE_OSX_DEPLOYMENT_TARGET AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 10.10)
    message(WARNING "Detected OS X deployment target \"${CMAKE_OSX_DEPLOYMENT_TARGET}\" is not supported anymore. Minimum required OS X version is 10.10 or greater.")
  endif()
endif()

#-----------------------------------------------------------------------------
# Check miminum compiler versions
#-----------------------------------------------------------------------------

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # require at least gcc 4.7.3 as provided by ppa:ubuntu-toolchain-r/test for Ubuntu 12.04
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7.3)
    message(FATAL_ERROR "GCC version must be at least 4.7.3
If you are using Ubuntu 12.04, you can easily install gcc and g++ 4.7.3 (or any later version available) in addition to your version ${CMAKE_CXX_COMPILER_VERSION}:
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  sudo apt-get update
  sudo apt-get install gcc-4.7 g++-4.7
Make sure to explicitly specify these compilers when configuring MITK:
  CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-4.7
  CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-4.7
For more information on the proposed PPA see the Toolchain Updates section of https://wiki.ubuntu.com/ToolChain.")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # require at least clang 3.4 as provided by Ubuntu 12.04
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.4)
    message(FATAL_ERROR "Clang version must be at least 3.4")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  # require at least clang 5.0
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "Apple Clang version must be at least 5.0")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # require at least Visual Studio 2012
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0.61030.0)
    message(FATAL_ERROR "Microsoft Visual Studio 2012 Update 4 or newer required (MSVC 17.0.61030.0)")
  endif()
else()
  message(WARNING "You are using an unsupported compiler! Compilation has only been tested with Clang (Linux or Apple), GCC and MSVC.")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  mitkFunctionGetGccVersion(${CMAKE_CXX_COMPILER} GCC_VERSION)
else()
  set(GCC_VERSION 0)
endif()

set(MITK_CXX_STANDARD 11)

set(CMAKE_CXX_EXTENSIONS 0)
set(CMAKE_CXX_STANDARD ${MITK_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED 1)

# This is necessary to avoid problems with compile feature checks.
# CMAKE_CXX_STANDARD seems to only set the -std=c++11 flag for targets.
# However, compile flag checks also need to be done with -std=c++11.
# The MITK_CXX11_FLAG variable is also used for external projects
# build during the MITK super-build.
mitkFunctionCheckCompilerFlags("-std=c++11" MITK_CXX11_FLAG)
if(NOT MITK_CXX11_FLAG)
  # Older gcc compilers use -std=c++0x
  mitkFunctionCheckCompilerFlags("-std=c++0x" MITK_CXX11_FLAG)
endif()


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

# -----------------------------------------
# Qt version related variables

if(APPLE)
  set(DESIRED_QT_VERSION 4 CACHE STRING "Pick a version of Qt to use: 4 or 5")
else()
  set(DESIRED_QT_VERSION 5 CACHE STRING "Pick a version of Qt to use: 4 or 5")
endif()

env_option(MITK_USE_QT "Use the Qt Company's Qt library" ON)
set(MITK_DESIRED_QT_VERSION ${DESIRED_QT_VERSION})

if(MITK_USE_QT)
  # find the package at the very beginning, so that QT4_FOUND is available
  if(DESIRED_QT_VERSION MATCHES 4)
    set(MITK_QT4_MINIMUM_VERSION 4.7)
    set(MITK_QT4_COMPONENTS QtCore QtGui QtXml)
    if(BUILD_IGI)
      list(APPEND MITK_QT4_COMPONENTS QtNetwork)
    endif()
    find_package(Qt4 ${MITK_QT4_MINIMUM_VERSION} COMPONENTS ${MITK_QT4_COMPONENTS} REQUIRED)
    set(MITK_USE_Qt4 TRUE)
    set(MITK_USE_Qt5 FALSE)
  endif()
  if(DESIRED_QT_VERSION MATCHES 5)
    set(MITK_QT5_MINIMUM_VERSION 5.0.0)
    set(MITK_USE_Qt4 FALSE)
    set(MITK_USE_Qt5 TRUE)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE PATH "")
    set(MITK_QT5_COMPONENTS Concurrent OpenGL PrintSupport Script Sql Svg Widgets WebKitWidgets Xml XmlPatterns UiTools Help)
    find_package(Qt5 ${MITK_QT5_MINIMUM_VERSION} COMPONENTS ${MITK_QT5_COMPONENTS} REQUIRED)
    if(Qt5_DIR)
      get_filename_component(_Qt5_DIR "${Qt5_DIR}/../../../" ABSOLUTE)
      list(FIND CMAKE_PREFIX_PATH "${_Qt5_DIR}" _result)
      if(_result LESS 0)
        set(CMAKE_PREFIX_PATH "${_Qt5_DIR};${CMAKE_PREFIX_PATH}" CACHE PATH "" FORCE)
      endif()
    endif()
  endif()
else()
  set(MITK_USE_Qt4 FALSE)
  set(MITK_USE_Qt5 FALSE)
endif()

if(QT_FOUND)
  set(QT_USE_QTXML 1)
  set(QT_USE_QTXMLPATTERNS 1)
  set(QT_USE_PHONON 0)
  include(${QT_USE_FILE})
else()
 message(FATAL "Qt is mandatory, but QT_FOUND is false.")
endif()

# ------------------------------------------------------------------------
# Register external projects which can be build with the MITK superbuild
# system. Each mitkFunctionAddExternalProject() call registers an external
# project for which a CMakeExternals/<project>.cmake file must exist. The
# call also creates a MITK_USE_<project> variable (appearing in the CMake
# UI if the NO_CACHE option is *not* given).
#
# Although the options are called 'MITK_USE_*', they mean whether NifTK uses
# those projects. To avoid confusion, we mark these options advanced so that
# they do not appear in the CMake GUI. We expose the 'BUILD_*' options to the
# users and set the 'MITK_USE_*' options accordingly.

# -----------------------------------------
# The following external projects must be ordered according to their inter-dependencies.

set_property(GLOBAL PROPERTY MITK_EXTERNAL_PROJECTS "")
#mitkFunctionAddExternalProject(NAME Camino OFF ADVANCED)
mitkFunctionAddExternalProject(NAME Boost ON ADVANCED)
mitkFunctionAddExternalProject(NAME Eigen ON ADVANCED)
mitkFunctionAddExternalProject(NAME OpenCV ON ADVANCED)
mitkFunctionAddExternalProject(NAME AprilTags OFF ADVANCED DEPENDS OpenCV Eigen)
mitkFunctionAddExternalProject(NAME ArUco OFF ADVANCED DEPENDS OpenCV)
mitkFunctionAddExternalProject(NAME FLANN OFF ADVANCED)
mitkFunctionAddExternalProject(NAME RTK OFF ADVANCED)
mitkFunctionAddExternalProject(NAME VL OFF ADVANCED)
mitkFunctionAddExternalProject(NAME CGAL OFF ADVANCED DEPENDS Boost)
mitkFunctionAddExternalProject(NAME NiftyLink OFF ADVANCED)
mitkFunctionAddExternalProject(NAME NiftySim OFF ADVANCED)
mitkFunctionAddExternalProject(NAME NiftySeg OFF ADVANCED Eigen)
mitkFunctionAddExternalProject(NAME NiftyReg OFF ADVANCED)
mitkFunctionAddExternalProject(NAME NiftyRec OFF ADVANCED)
mitkFunctionAddExternalProject(NAME NifTKData OFF ADVANCED)

mitkFunctionAddExternalProject(NAME PCRE ON ADVANCED NO_PACKAGE)
mitkFunctionAddExternalProject(NAME SWIG ON ADVANCED NO_PACKAGE DEPENDS PCRE)
mitkFunctionAddExternalProject(NAME ZLIB ON ADVANCED NO_PACKAGE NO_CACHE)
mitkFunctionAddExternalProject(NAME Python ON NO_PACKAGE DEPENDS ZLIB SWIG DOC "Use Python wrapping in NifTK")
mitkFunctionAddExternalProject(NAME Numpy ON ADVANCED NO_PACKAGE)

# These are "hard" dependencies and always set to ON.
mitkFunctionAddExternalProject(NAME DCMTK ON ADVANCED)
mitkFunctionAddExternalProject(NAME GDCM ON ADVANCED)
mitkFunctionAddExternalProject(NAME ITK ON ADVANCED NO_CACHE DEPENDS GDCM)
mitkFunctionAddExternalProject(NAME SlicerExecutionModel ON ADVANCED DEPENDS ITK)
mitkFunctionAddExternalProject(NAME VTK ON ADVANCED NO_CACHE)
mitkFunctionAddExternalProject(NAME CTK ON ADVANCED DEPENDS QT ITK VTK DCMTK)
mitkFunctionAddExternalProject(NAME MITK ON ADVANCED NO_CACHE DEPENDS ITK VTK GDCM DCMTK CTK Eigen)

# Further optional dependencies.
mitkFunctionAddExternalProject(NAME PCL OFF ADVANCED DEPENDS Boost Eigen FLANN VTK)
mitkFunctionAddExternalProject(NAME SimpleITK ON DEPENDS ITK GDCM SWIG)

# -----------------------------------------
# Other MITK_USE_* options not related to external projects build via the NifTK superbuild

#env_option(MITK_USE_OpenCL "Use OpenCL GPU-Computing library" OFF)

if (BUILD_IGI)
  set(MITK_USE_AprilTags ON CACHE BOOL "" FORCE)
  set(MITK_USE_ArUco ON CACHE BOOL "" FORCE)
  set(MITK_USE_NiftyLink ON CACHE BOOL "" FORCE)
  set(MITK_USE_OpenCV ON CACHE BOOL "" FORCE)
else()
  set(MITK_USE_AprilTags OFF CACHE BOOL "" FORCE)
  set(MITK_USE_ArUco OFF CACHE BOOL "" FORCE)
  set(MITK_USE_NiftyLink OFF CACHE BOOL "" FORCE)
  set(MITK_USE_OpenCV OFF CACHE BOOL "" FORCE)
endif()

if (BUILD_NiftyReg)
  set(MITK_USE_NiftyReg ON CACHE BOOL "" FORCE)
else()
  set(MITK_USE_NiftyReg OFF CACHE BOOL "" FORCE)
endif()

if (BUILD_NiftySeg)
  set(MITK_USE_NiftySeg ON CACHE BOOL "" FORCE)
else()
  set(MITK_USE_NiftySeg OFF CACHE BOOL "" FORCE)
endif()

if (BUILD_NiftySim)
  set(MITK_USE_NiftySim ON CACHE BOOL "" FORCE)
else()
  set(MITK_USE_NiftySim OFF CACHE BOOL "" FORCE)
endif()

if (BUILD_NiftyRec)
  set(MITK_USE_NiftyRec ON CACHE BOOL "" FORCE)
else()
  set(MITK_USE_NiftyRec OFF CACHE BOOL "" FORCE)
endif()

if (BUILD_VL)
  set(MITK_USE_VL ON CACHE BOOL "" FORCE)
else()
  set(MITK_USE_VL OFF CACHE BOOL "" FORCE)
endif()

######################################################################
# Python: Sanity check for supported Qt version. Only >= 5.3 is supported by CTK/PythonQt
######################################################################
if(MITK_USE_Qt5 AND MITK_USE_Python)
  set(minimum_required_python_qt5_version "5.3.0")
  find_package(Qt5 COMPONENTS Core REQUIRED)

  if(${Qt5Core_VERSION_STRING} VERSION_LESS ${minimum_required_python_qt5_version})
    message(WARNING "Can't build MITK Python with Qt version < ${minimum_required_python_qt5_version}. Disabling Python support")
    set(MITK_USE_Python OFF CACHE BOOL "Use python wrapping in MITK" FORCE)
  endif()
endif()

######################################################################
# Python: Only windows can't build python in debug mode
######################################################################
if(MITK_USE_Python AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND WIN32)
  message(WARNING "Disabling Python support. Building MITK Python in debug mode on Windowsis not supported!")
  set(MITK_USE_Python OFF CACHE BOOL "Use python wrapping in MITK" FORCE)
  set(MITK_USE_Numpy OFF CACHE BOOL "Use Numpy" FORCE)
  set(MITK_USE_SimpleITK OFF CACHE BOOL "Use SimpleITK" FORCE)
elseif(MITK_USE_Python)
  set(MITK_USE_ZLIB ON)
  if(NOT MITK_USE_Numpy)
    message("> Forcing MITK_USE_Numpy to ON because of MITK_USE_Python")
    set(MITK_USE_Numpy ON CACHE BOOL "Use Numpy" FORCE)
  endif()
  if(NOT MITK_USE_SimpleITK)
    message("> Forcing MITK_USE_SimpleITK to ON because of MITK_USE_Python")
    set(MITK_USE_SimpleITK ON CACHE BOOL "Use SimpleITK" FORCE)
  endif()
  if(NOT MITK_USE_SWIG)
    message("> Forcing MITK_USE_SWIG to ON because of MITK_USE_Python")
    set(MITK_USE_SWIG ON CACHE BOOL "Use SWIG" FORCE)
  endif()
  if(NOT MITK_USE_PCRE)
    message("> Forcing MITK_USE_PCRE to ON because of MITK_USE_Python")
    set(MITK_USE_PCRE ON CACHE BOOL "Use PCRE" FORCE)
  endif()
  if(NOT MITK_USE_ZLIB)
    message("> Forcing MITK_USE_ZLIB to ON because of MITK_USE_Python")
    set(MITK_USE_ZLIB ON CACHE BOOL "Use ZLIB" FORCE)
  endif()

  option(MITK_USE_SYSTEM_PYTHON "Use the system python runtime" OFF)
  if(MITK_USE_SYSTEM_PYTHON)
    find_package(PythonLibs REQUIRED)
    find_package(PythonInterp REQUIRED)
  endif()
elseif(MITK_USE_Python AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND WIN32)
  message(WARNING "Disabling Python support. Building MITK Python in debug mode on Windowsis not supported!")
  set(MITK_USE_Python OFF CACHE BOOL "Use python wrapping in MITK" FORCE)
endif()

#-----------------------------------------------------------------------------
# Build configurations
#-----------------------------------------------------------------------------

set(_buildConfigs "Custom")

file(GLOB _buildConfigFiles CMake/BuildConfigurations/*.cmake)

foreach(_buildConfigFile ${_buildConfigFiles})
  get_filename_component(_buildConfigFile ${_buildConfigFile} NAME_WE)
  list(APPEND _buildConfigs ${_buildConfigFile})
endforeach()

set(MITK_BUILD_CONFIGURATION "Custom" CACHE STRING "Use pre-defined MITK configurations")
mark_as_advanced(MITK_BUILD_CONFIGURATION)
set_property(CACHE MITK_BUILD_CONFIGURATION PROPERTY STRINGS ${_buildConfigs})

mitkFunctionEnableBuildConfiguration()

mitkFunctionCreateWhitelistPaths(NifTK)
mitkFunctionFindWhitelists(NifTK)

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

  # Setup continuous test script
  CONFIGURE_FILE(CMake/CTestContinuous.cmake.in ${CMAKE_BINARY_DIR}/CTestContinuous.cmake @ONLY)

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
if(WIN32)
  set(OPENCV_WITH_FFMPEG ON CACHE BOOL "Build OpenCV with FFMPEG support")
else(WIN32)
  set(OPENCV_WITH_FFMPEG OFF CACHE BOOL "Build OpenCV with FFMPEG support")
endif(WIN32)
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
    ${CMAKE_SOURCE_DIR}/Code/Libs/Meshing
  )
endif (BUILD_MESHING)
 
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
    if(APPLE)
      set(CMAKE_SHARED_LINKER_FLAGS "-F/Library/Frameworks ${CMAKE_SHARED_LINKER_FLAGS}")
    endif()
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
    ${aruco_DIR}/lib/cmake
    ${CMAKE_MODULE_PATH}
   )

######################################################################
# Configure CMake files before we need to use them in the FIND_XXX.
#
# So, the variables NIFTK_BOOSTINSTALL etc.
# should be passed in on cmake command line, or set up using ccmake.
######################################################################
find_package(aruco)
if(aruco_FOUND)
  configure_file(${CMAKE_SOURCE_DIR}/CMake/FindArUco.cmake ${CMAKE_BINARY_DIR}/FindArUco.cmake @ONLY)
endif()
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindFFTW.cmake ${CMAKE_BINARY_DIR}/FindFFTW.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftyReg.cmake ${CMAKE_BINARY_DIR}/FindNiftyReg.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftyRec.cmake ${CMAKE_BINARY_DIR}/FindNiftyRec.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftySim.cmake ${CMAKE_BINARY_DIR}/FindNiftySim.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindNiftySeg.cmake ${CMAKE_BINARY_DIR}/FindNiftySeg.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindEigen.cmake ${CMAKE_BINARY_DIR}/FindEigen.cmake @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/CMake/FindAprilTags.cmake ${CMAKE_BINARY_DIR}/FindAprilTags.cmake @ONLY)
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

# Configure module naming conventions
set(MITK_MODULE_NAME_REGEX_MATCH "^[A-Z].*$")
set(MITK_MODULE_NAME_REGEX_NOT_MATCH "^[Nn][Ii][Ff][Tt][Kk].*$")
set(MITK_MODULE_NAME_PREFIX "niftk")
set(MITK_MODULE_NAME_DEFAULTS_TO_DIRECTORY_NAME 1)
set(MITK_LEGACY_EXPORT_MACRO_NAME 0)

# We need this later for a DCMTK workaround
set(_dcmtk_dir_orig ${DCMTK_DIR})

# This property is populated at the top half of this file
get_property(MITK_EXTERNAL_PROJECTS GLOBAL PROPERTY MITK_EXTERNAL_PROJECTS)
foreach(ep ${MITK_EXTERNAL_PROJECTS})
  get_property(_package GLOBAL PROPERTY MITK_${ep}_PACKAGE)
  get_property(_components GLOBAL PROPERTY MITK_${ep}_COMPONENTS)
  if(MITK_USE_${ep} AND _package)
    if(_components)
      find_package(${_package} COMPONENTS ${_components} REQUIRED CONFIG)
    else()
      # Prefer config mode first because it finds external
      # <proj>Config.cmake files pointed at by <proj>_DIR variables.
      # Otherwise, existing Find<proj>.cmake files could fail.
      # (e.g. in the case of GLEW and the FindGLEW.cmake file shipped
      # with CMake).
      find_package(${_package} QUIET CONFIG)
      string(TOUPPER "${_package}" _package_uc)
      if(NOT (${_package}_FOUND OR ${_package_uc}_FOUND))
        find_package(${_package} REQUIRED)
      endif()
    endif()
  endif()
endforeach()

# MITKConfig.cmake sets this variable therefore we need to clear it again.
set(MITK_LEGACY_EXPORT_MACRO_NAME 0)

# Ensure that the MITK CMake module path comes first
set(CMAKE_MODULE_PATH
  ${MITK_CMAKE_DIR}
  ${CMAKE_MODULE_PATH}
  )

set(DCMTK_DIR ${_dcmtk_dir_orig})
if(MITK_USE_DCMTK)
  # Due to the preferred CONFIG mode in find_package calls above,
  # the DCMTKConfig.cmake file is read, which does not provide useful
  # package information. We explictly need MODULE mode to find DCMTK.
#  if(${_dcmtk_dir_orig} MATCHES "${MITK_EXTERNAL_PROJECT_PREFIX}.*")
#    # Help our FindDCMTK.cmake script find our super-build DCMTK
#    set(DCMTK_DIR ${MITK_EXTERNAL_PROJECT_PREFIX})
#  else()
    # Use the original value
    set(DCMTK_DIR ${_dcmtk_dir_orig})
#  endif()
  find_package(DCMTK REQUIRED MODULE)
endif()

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

    if(${option_name})
      set(MACOSX_BUNDLE_NAMES ${MACOSX_BUNDLE_NAMES} ${app_name})
    endif()

  endforeach()

endif()

if(BUILD_IGI)
  add_definitions(-DBUILD_IGI)
endif()

######################################################################
# Find Mandatory External packages
######################################################################



######################################################################
# (Optional) Find PCL.
# find it before boost because it will mess up Boost_LIBRARIES with its own requested components.
######################################################################
set(Boost_NO_SYSTEM_PATHS ON)
# off is the default, but our optional pcl dependency will try to use static libs if we dont define it explicitly.
set(Boost_USE_STATIC_LIBS OFF)

if(BUILD_PCL)
  # PCL's cmake bits are extremely chatty and fill the cmake window/log with noise.
  set(PCL_FIND_QUIETLY ON)
  find_package(PCL 1.7 REQUIRED)
  # do not add the include paths, etc to the global compiler settings!
  # instead, this is done for each module/plugin separately in PackageDepends/MITK_PCL_Config.cmake.
endif()


######################################################################
# Find Boost.
######################################################################
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

######################################################################
# Find OpenCV.
######################################################################
if(BUILD_IGI)
  set(OpenCV_FIND_QUIETLY ON)
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
set(MITK_DESIRED_QT_VERSION ${DESIRED_QT_VERSION})
set(MITK_USE_OpenCV ${BUILD_IGI})

if(NIFTK_GENERATE_DOXYGEN_HELP)
  set(BLUEBERRY_USE_QT_HELP ON)
endif(NIFTK_GENERATE_DOXYGEN_HELP)

# Note:
# We build NifTK with Boost but MITK without it. The next 'find_package(MITK REQUIRED)'
# loads values from MITKConfig.cmake that overwrites some Boost related variables, e.g.
# MITK_USE_Boost. This causes that if a module depends on the Boost package, the
# MITK_CREATE_MODULE macro will think that the software is being built without Boost,
# and it will not create the module because of the missing dependency.
# As a workaround, here we save the values of these variables and restore them after
# the find_package() call.

set(_boost_root ${BOOST_ROOT})
set(_mitk_use_boost ${MITK_USE_Boost})
set(_mitk_use_system_boost ${MITK_USE_SYSTEM_Boost})
set(_mitk_use_boost_libraries ${MITK_USE_Boost_LIBRARIES})
set(_mitk_legacy_export_macro_name ${MITK_LEGACY_EXPORT_MACRO_NAME})

find_package(MITK REQUIRED)
if(MITK_FOUND)
  message("Found MITK")
  set(CMAKE_MODULE_PATH
    ${MITK_SOURCE_DIR}/CMake
    ${CMAKE_MODULE_PATH}
  )
  link_directories(${MITK_LINK_DIRECTORIES})
endif(MITK_FOUND)

set(BOOST_ROOT ${_boost_root})
set(MITK_USE_Boost ${_mitk_use_boost})
set(MITK_USE_SYSTEM_Boost ${_mitk_use_system_boost})
set(MITK_USE_Boost_LIBRARIES ${_mitk_use_boost_libraries})
set(MITK_LEGACY_EXPORT_MACRO_NAME ${_mitk_legacy_export_macro_name})

######################################################################
# Find Optional External packages
######################################################################


######################################################################
# Find CGAL.
######################################################################

if (BUILD_MESHING)
  find_package(CGAL REQUIRED COMPONENTS Core ImageIO)
  include("${CGAL_USE_FILE}")

  if(NOT WIN32)
    # UseCGAL.cmake is buggy: despite reading the correct requested modules
    # it only adds the libCGAL.so base library to CGAL_LIBRARIES
    set(CGAL_LIBRARIES
      CGAL
      CGAL_Core
      CGAL_ImageIO)
  endif()
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
# Find NiftyReg - (version will vary according to whether CUDA found)
######################################################################
if(BUILD_NiftyReg)
  find_package(NiftyReg)
  if(NiftyReg_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NiftyReg_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYREG)
    set(NIFTYREG_NVCC_FLAGS "${NIFTYREG_NVCC_FLAGS};-DUSE_NIFTYREG")
    message("Found NiftyReg")
  else()
    message("Didn't find NiftyReg")
  endif()
endif()

if(NOT NiftyReg_FOUND)
  message("No NiftyReg found, so turning off plugin")
  set("NIFTK_Plugins/uk.ac.ucl.cmic.niftyreg" OFF CACHE BOOL "NiftyReg plugin" FORCE)
endif()

######################################################################
# Find NiftyRec - (version will vary according to whether CUDA found)
# NB: Is dependent on NiftyReg and at time of writing only has GPU version
######################################################################

if(BUILD_NiftyRec)
  find_package(NiftyRec)
  if(NiftyRec_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NiftyRec_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYREC)
    set(NIFTYREC_NVCC_FLAGS "${NIFTYREC_NVCC_FLAGS};-DUSE_NIFTYREC")
    message("Found NiftyRec")
  else()
    message("Didn't find NiftyRec")
  endif()
endif()

######################################################################
# Find NiftySim
######################################################################

if(BUILD_NiftySim)
  find_package(NiftySim)
  if(NiftySim_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NiftySim_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYSIM)
    set(NIFTYSIM_NVCC_FLAGS "${NIFTYSIM_NVCC_FLAGS};-DUSE_NIFTYSIM")
    message("Found NiftySim")
  else()
    message("Didn't find NiftySim")
  endif()
endif()

######################################################################
# Find NiftySeg
######################################################################

if(BUILD_NiftySeg)
  find_package(NiftySeg)
  if(NiftySeg_FOUND)
    set(NIFTK_INCLUDE_DIRS_BUILD_TREE ${NIFTK_INCLUDE_DIRS_BUILD_TREE} ${NiftySeg_INCLUDE_DIR})
    add_definitions(-DUSE_NIFTYSEG)
    set(NIFTYSEG_NVCC_FLAGS "${NIFTYSEG_NVCC_FLAGS};-DUSE_NIFTYSEG")
    message("Found NiftySeg")
  else()
    message("Didn't find NiftySeg")
  endif()
endif()

if(NOT NiftySeg_FOUND)
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
# Find VL
######################################################################

if(BUILD_VL)
  find_package(VL COMPONENTS VLCore VLGraphics VLVolume REQUIRED)
  if(VL_FOUND)
    message("Found VL")
    #include(${VL_USE_FILE})
  endif(VL_FOUND)
endif(BUILD_VL)

######################################################################
# Find SlicerExecutionModel
######################################################################
if(BUILD_COMMAND_LINE_PROGRAMS)

  set(_itk_dir ${ITK_DIR})

  ######################################################################
  # Warning: This process corrupts ITK_DIR
  ######################################################################
  find_package(SlicerExecutionModel REQUIRED)
  if(SlicerExecutionModel_FOUND)
    message("Found SlicerExecutionModel")
    include(${SlicerExecutionModel_USE_FILE})
  else()
    message("Didn't find SlicerExecutionModel")
  endif()

  ######################################################################
  # So, here we set ITK_DIR back
  ######################################################################
  set(ITK_DIR ${_itk_dir})

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
        ${PROTOTYPE_DIR}/InsightJournal/IJ_845_FuzzyClusteringForImageSegmentation
    )
  endif()
endif()


######################################################################
# NifTK uses KWStyle for checking the coding style
######################################################################
include(${CMAKE_SOURCE_DIR}/Utilities/KWStyle/NifTKKWStyle.cmake)


######################################################################
# NifTK uses CppCheck for static analysis
######################################################################
include(${CMAKE_SOURCE_DIR}/Utilities/CppCheck/NifTKCppCheck.cmake)


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

######################################################################
# For doxygen:
######################################################################

configure_file(${CMAKE_SOURCE_DIR}/Utilities/GenerateCommandLineDoxygen.sh.in
  ${EXECUTABLE_OUTPUT_PATH}/GenerateCommandLineDoxygen @ONLY)

configure_file(${CMAKE_SOURCE_DIR}/Utilities/GenerateTestingReports.sh.in
  ${EXECUTABLE_OUTPUT_PATH}/GenerateTestingReports @ONLY)

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
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/ArUco.txt ${CMAKE_BINARY_DIR}/LICENSE_ArUco.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/AprilTags.txt ${CMAKE_BINARY_DIR}/LICENSE_AprilTags.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/Boost.txt ${CMAKE_BINARY_DIR}/LICENSE_Boost.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/CTK.txt ${CMAKE_BINARY_DIR}/LICENSE_CTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/DCMTK.txt ${CMAKE_BINARY_DIR}/LICENSE_DCMTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/GDCM.txt ${CMAKE_BINARY_DIR}/LICENSE_GDCM.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/ITK.txt ${CMAKE_BINARY_DIR}/LICENSE_ITK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/RTK.txt ${CMAKE_BINARY_DIR}/LICENSE_RTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/VL.txt ${CMAKE_BINARY_DIR}/LICENSE_VL.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/VTK.txt ${CMAKE_BINARY_DIR}/LICENSE_VTK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/MITK.txt ${CMAKE_BINARY_DIR}/LICENSE_MITK.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftyReg.txt ${CMAKE_BINARY_DIR}/LICENSE_NiftyReg.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySeg.txt ${CMAKE_BINARY_DIR}/LICENSE_NiftySeg.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/NiftySim.txt ${CMAKE_BINARY_DIR}/LICENSE_NiftySim.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/OpenCV.txt ${CMAKE_BINARY_DIR}/LICENSE_OpenCV.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/Qt.txt ${CMAKE_BINARY_DIR}/LICENSE_Qt.txt)
configure_file(${CMAKE_SOURCE_DIR}/Doc/Licenses/Eigen.txt ${CMAKE_BINARY_DIR}/LICENSE_Eigen.txt)
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
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/VL.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
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
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/ArUco.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/AprilTags.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/Doc/Licenses/Eigen.txt DESTINATION ${NIFTK_INSTALL_LICENSES_DIR} COMPONENT documentation)
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
if(BUILD_NiftyReg AND NiftyReg_FOUND)
  file(GLOB _NiftyReg_LIB_FILES
       ${NiftyReg_DIR}/lib/*.so
       ${NiftyReg_DIR}/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${_NiftyReg_LIB_FILES})

  file(GLOB _NiftyReg_BIN_FILES ${NiftyReg_DIR}/bin/*)
  MITK_INSTALL(PROGRAMS ${_NiftyReg_BIN_FILES})

  foreach(prog reg_aladin reg_f3d reg_jacobian reg_resample)
    set(FULL_APP_NAME ${prog})
    NIFTK_GENERATE_CLI_SCRIPT(NAME ${prog})
  endforeach()
endif()

# NiftySeg
if(BUILD_NiftySeg)
  file(GLOB _NiftySeg_LIB_FILES
       ${NiftySeg_DIR}/lib/*.so
       ${NiftySeg_DIR}/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${_NiftySeg_LIB_FILES})

  file(GLOB _NiftySeg_BIN_FILES ${NiftySeg_DIR}/bin/*)
  MITK_INSTALL(PROGRAMS ${_NiftySeg_BIN_FILES})

  foreach(prog seg_EM seg_LabFusion)
    set(FULL_APP_NAME ${prog})
    NIFTK_GENERATE_CLI_SCRIPT(NAME ${prog})
  endforeach()

  file(GLOB _NiftySeg_PRIORS_FILES ${NiftySeg_DIR}/priors/*)
  install(PROGRAMS ${_NiftySeg_PRIORS_FILES} DESTINATION ${NIFTK_INSTALL_BASE_DIR}/priors COMPONENT applications)
endif()

# NiftyRec
if(BUILD_NiftyRec AND NiftyRec_FOUND)
  file(GLOB _NiftyRec_LIB_FILES
       ${NiftyRec_DIR}/lib/*.so
       ${NiftyRec_DIR}/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${_NiftyRec_LIB_FILES})

  file(GLOB _NiftyRec_INCL_FILES ${NiftyRec_DIR}/include/*)
  MITK_INSTALL(PROGRAMS ${_NiftyRec_INCL_FILES})
endif()

# NiftySim
if(BUILD_NiftySim AND NiftySim_FOUND)
  file(GLOB _NiftySim_LIB_FILES
       ${NiftySim_DIR}/lib/*.so
       ${NiftySim_DIR}/lib/*.dylib
       )
  MITK_INSTALL(PROGRAMS ${_NiftySim_LIB_FILES})

  file(GLOB _NiftySim_BIN_FILES ${NiftySim_DIR}/bin/*)
  MITK_INSTALL(PROGRAMS ${_NIFTY_SIM_BIN_FILES})
endif()

# Camino
if(BUILD_CAMINO AND NOT WIN32 AND NOT APPLE )
  file(GLOB CAMINO_BIN_FILES
       ${camino_DIR}/bin/*
      )
  file(GLOB CAMINO_MAN_FILES
       ${camino_DIR}/man/man1/*
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
      set(PCL_BIN_PATH_FOR_BATCH_FILE "${PCL_DIR}/bin;${PCL_DIR}/../bin")
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
    if(NiftySim_FOUND)
      set(NIFTYSIM_BIN_PATH_FOR_BATCH_FILE "${NiftySim_DIR}/bin")
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


# ---------------- External projects -----------------

get_property(MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS_CONFIG GLOBAL PROPERTY MITK_ADDITIONAL_LIBRARY_SEARCH_PATHS)

set(MITK_CONFIG_EXTERNAL_PROJECTS )
#string(REPLACE "^^" ";" _mitk_external_projects ${MITK_EXTERNAL_PROJECTS})

foreach(ep ${MITK_EXTERNAL_PROJECTS})
  get_property(_components GLOBAL PROPERTY MITK_${ep}_COMPONENTS)
  set(MITK_CONFIG_EXTERNAL_PROJECTS "${MITK_CONFIG_EXTERNAL_PROJECTS}
set(MITK_USE_${ep} ${MITK_USE_${ep}})
set(MITK_${ep}_DIR \"${${ep}_DIR}\")
set(MITK_${ep}_COMPONENTS ${_components})
")
endforeach()

foreach(ep ${MITK_EXTERNAL_PROJECTS})
  get_property(_package GLOBAL PROPERTY MITK_${ep}_PACKAGE)
  get_property(_components GLOBAL PROPERTY MITK_${ep}_COMPONENTS)
  if(_components)
      set(_components_arg COMPONENTS \${_components})
  else()
    set(_components_arg)
  endif()

  if(_package)
    set(MITK_CONFIG_EXTERNAL_PROJECTS "${MITK_CONFIG_EXTERNAL_PROJECTS}
if(MITK_USE_${ep})
  set(${ep}_DIR \${MITK_${ep}_DIR})
  if(MITK_${ep}_COMPONENTS)
    mitkMacroFindDependency(${_package} COMPONENTS \${MITK_${ep}_COMPONENTS})
  else()
    mitkMacroFindDependency(${_package})
  endif()
endif()")
  endif()
endforeach()

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
