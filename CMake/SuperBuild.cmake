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

include(mitkFunctionInstallExternalCMakeProject)

#-----------------------------------------------------------------------------
# Convenient macro allowing to download a file
#-----------------------------------------------------------------------------

macro(downloadFile url dest)
  file(DOWNLOAD ${url} ${dest} STATUS status)
  list(GET status 0 error_code)
  list(GET status 1 error_msg)
  if(error_code)
    message(FATAL_ERROR "error: Failed to download ${url} - ${error_msg}")
  endif()
endmacro()

# We need a proper patch program. On Linux and MacOS, we assume
# that "patch" is available. On Windows, we download patch.exe
# if not patch program is found.
find_program(PATCH_COMMAND patch)
if((NOT PATCH_COMMAND OR NOT EXISTS ${PATCH_COMMAND}) AND WIN32)
  downloadFile(${NIFTK_EP_TARBALL_LOCATION}/patch.exe
               ${CMAKE_CURRENT_BINARY_DIR}/patch.exe)
  find_program(PATCH_COMMAND patch ${CMAKE_CURRENT_BINARY_DIR})
endif()
if(NOT PATCH_COMMAND)
  message(FATAL_ERROR "No patch program found.")
endif()

include(ExternalProject)

set(EP_BASE "${CMAKE_BINARY_DIR}" CACHE PATH "Directory where the external projects are configured and built")
set_property(DIRECTORY PROPERTY EP_BASE ${EP_BASE})

# This option makes different versions of the same external project build in separate directories.
# This allows switching branches in the NifTK source code and build NifTK quickly, even if the
# branches use different versions of the same library. A given version of an EP will be built only
# once. A drawback is that the EP_BASE directory can become big easily.
# Note:
# If you switch branches that need different versions of EPs, you might need to delete the
# NifTK-configure timestamp manually before doing a superbuild. Without that the CMake cache is
# not regenerated and it may still store the paths to the EP versions that belong to the original
# branch (from which you switch). You have been warned.

set(EP_DIRECTORY_PER_VERSION FALSE CACHE BOOL "Use separate directories for different versions of the same external project.")

mark_as_advanced(EP_BASE)
mark_as_advanced(EP_DIRECTORY_PER_VERSION)

if(MITK_CTEST_SCRIPT_MODE)
  # Write a file containing the list of enabled external project targets.
  # This file can be read by a ctest script to separately build projects.
  set(SUPERBUILD_TARGETS )
  foreach(proj ${external_projects})
    if(MITK_USE_${proj})
      list(APPEND SUPERBUILD_TARGETS ${proj})
    endif()
  endforeach()
  file(WRITE "${CMAKE_BINARY_DIR}/SuperBuildTargets.cmake" "set(SUPERBUILD_TARGETS ${SUPERBUILD_TARGETS})")
endif()

# Compute -G arg for configuring external projects with the same CMake generator:
if(CMAKE_EXTRA_GENERATOR)
  set(gen "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
else()
  set(gen "${CMAKE_GENERATOR}")
endif()

if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj /MP /W0 /Zi")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /MP /W0 /Zi")
  # we want symbols, even for release builds!
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /debug")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /debug")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /debug")
  set(CMAKE_CXX_WARNING_LEVEL 0)
else()
  if(${BUILD_SHARED_LIBS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLINUX_EXTRA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX_EXTRA")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -DLINUX_EXTRA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -DLINUX_EXTRA")
  endif()
endif()


# This is a workaround for passing linker flags
# actually down to the linker invocation
set(_cmake_required_flags_orig ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_FLAGS "-Wl,-rpath")
mitkFunctionCheckCompilerFlags(${CMAKE_REQUIRED_FLAGS} _has_rpath_flag)
set(CMAKE_REQUIRED_FLAGS ${_cmake_required_flags_orig})

set(_install_rpath_linkflag )
if(_has_rpath_flag)
  if(APPLE)
    set(_install_rpath_linkflag "-Wl,-rpath,@loader_path/../lib")
  else()
    set(_install_rpath_linkflag "-Wl,-rpath='$ORIGIN/../lib'")
  endif()
endif()

set(_install_rpath)
if(APPLE)
  set(_install_rpath "@loader_path/../lib")
elseif(UNIX)
  # this works for libraries as well as executables
  set(_install_rpath "\$ORIGIN/../lib")
endif()

set(EP_COMMON_ARGS
  -DCMAKE_CXX_EXTENSIONS:STRING=${CMAKE_CXX_EXTENSIONS}
  -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
  -DCMAKE_DEBUG_POSTFIX:STRING=d
  -DCMAKE_MACOSX_RPATH:BOOL=TRUE
  "-DCMAKE_INSTALL_RPATH:STRING=${_install_rpath}"
  -DBUILD_TESTING:BOOL=OFF
  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  -DBUILD_SHARED_LIBS:BOOL=ON
  -DBUILD_TESTING:BOOL=OFF
  -DBUILD_EXAMPLES:BOOL=OFF
  -DDESIRED_QT_VERSION:STRING=${DESIRED_QT_VERSION}
  -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_CONFIGURATION_TYPES:STRING=${CMAKE_CONFIGURATION_TYPES}
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${CMAKE_VERBOSE_MAKEFILE}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} ${MITK_CXX11_FLAG}"
  #debug flags
  -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
  -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
  #release flags
  -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
  -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
  #relwithdebinfo
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
  -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}
  #link flags
  -DCMAKE_EXE_LINKER_FLAGS:STRING=${CMAKE_EXE_LINKER_FLAGS}
  -DCMAKE_SHARED_LINKER_FLAGS:STRING=${CMAKE_SHARED_LINKER_FLAGS}
  -DCMAKE_MODULE_LINKER_FLAGS:STRING=${CMAKE_MODULE_LINKER_FLAGS}
)

set(EP_COMMON_CACHE_ARGS
)

set(EP_COMMON_CACHE_DEFAULT_ARGS
  "-DCMAKE_PREFIX_PATH:PATH=<INSTALL_DIR>;${CMAKE_PREFIX_PATH}"
  "-DCMAKE_INCLUDE_PATH:PATH=${CMAKE_INCLUDE_PATH}"
  "-DCMAKE_LIBRARY_PATH:PATH=${CMAKE_LIBRARY_PATH}"
)


if(INSTALL_WITH_RPATH)
  set(EP_COMMON_ARGS
       -DCMAKE_MACOSX_RPATH:BOOL=TRUE
       "-DCMAKE_INSTALL_RPATH:STRING=${_install_rpath}"
       ${EP_COMMON_ARGS}
      )
endif()

if(APPLE)
  set(EP_COMMON_ARGS
       -DCMAKE_OSX_ARCHITECTURES:PATH=${CMAKE_OSX_ARCHITECTURES}
       -DCMAKE_OSX_DEPLOYMENT_TARGET:PATH=${CMAKE_OSX_DEPLOYMENT_TARGET}
       -DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}
       ${EP_COMMON_ARGS}
      )
endif()

set(NIFTK_APP_OPTIONS)
foreach(NIFTK_APP ${NIFTK_APPS})

  # extract option_name
  string(REPLACE "^^" "\\;" target_info ${NIFTK_APP})
  set(target_info_list ${target_info})
  list(GET target_info_list 1 option_name)
  list(GET target_info_list 0 app_name)

  # Set flag.
  set(option_string)
  if(${option_name})
    set(option_string "-D${option_name}:BOOL=ON")
  else()
    set(option_string "-D${option_name}:BOOL=OFF")
  endif()

  set(NIFTK_APP_OPTIONS
    ${NIFTK_APP_OPTIONS}
    ${option_string}
  )

  # Add to list.
endforeach()

######################################################################
# Include NifTK helper macros
######################################################################

include(niftkExternalProjectHelperMacros)

######################################################################
# External projects
######################################################################

get_property(external_projects GLOBAL PROPERTY MITK_EXTERNAL_PROJECTS)

if(MITK_CTEST_SCRIPT_MODE)
  # Write a file containing the list of enabled external project targets.
  # This file can be read by a ctest script to separately build projects.
  set(SUPERBUILD_TARGETS )
  foreach(proj ${external_projects})
    if(MITK_USE_${proj})
      list(APPEND SUPERBUILD_TARGETS ${proj})
    endif()
  endforeach()
  file(WRITE "${CMAKE_BINARY_DIR}/SuperBuildTargets.cmake" "set(SUPERBUILD_TARGETS ${SUPERBUILD_TARGETS})")
endif()

if(BUILD_IGI)
  if(OPENCV_WITH_CUDA)
    message("Beware: You are building with OPENCV_WITH_CUDA! This means OpenCV will have a hard dependency on CUDA and will not work without it!")
  endif(OPENCV_WITH_CUDA)
endif(BUILD_IGI)

foreach(p ${external_projects})
  include("CMake/CMakeExternals/${p}.cmake")

  list(APPEND NIFTK_APP_OPTIONS -DNIFTK_VERSION_${p}:STRING=${${p}_VERSION})
  list(APPEND NIFTK_APP_OPTIONS -DNIFTK_LOCATION_${p}:STRING=${${p}_LOCATION})
endforeach()

######################################################################
# Now compile NifTK, using the packages we just provided.
######################################################################
if(NOT DEFINED SUPERBUILD_EXCLUDE_NIFTKBUILD_TARGET OR NOT SUPERBUILD_EXCLUDE_NIFTKBUILD_TARGET)

  set(proj NifTK)
  set(proj_DEPENDENCIES ${Boost_DEPENDS} ${GDCM_DEPENDS} ${ITK_DEPENDS} ${SlicerExecutionModel_DEPENDS} ${VTK_DEPENDS} ${MITK_DEPENDS} ${Camino_DEPENDS})

  if(BUILD_TESTING)
    list(APPEND proj_DEPENDENCIES ${NifTKData_DEPENDS})
  endif()

  if(BUILD_IGI)
    list(APPEND proj_DEPENDENCIES ${NiftyLink_DEPENDS} ${OpenCV_DEPENDS} ${ArUco_DEPENDS} ${Eigen_DEPENDS} ${AprilTags_DEPENDS})
  endif()

  if(BUILD_IGI AND BUILD_PCL)
    list(APPEND proj_DEPENDENCIES ${FLANN_DEPENDS} ${PCL_DEPENDS})
  endif()

  if(BUILD_NiftyReg)
    list(APPEND proj_DEPENDENCIES ${NiftyReg_DEPENDS})
  endif()

  if(BUILD_NiftySeg)
    list(APPEND proj_DEPENDENCIES ${Eigen_DEPENDS} ${NiftySeg_DEPENDS})
  endif()

  if(BUILD_NiftySim)
    list(APPEND proj_DEPENDENCIES ${NiftySim_DEPENDS})
  endif()

  if(BUILD_NiftyRec)
    list(APPEND proj_DEPENDENCIES ${NiftyRec_DEPENDS})
  endif()

  if(BUILD_RTK)
    list(APPEND proj_DEPENDENCIES ${RTK_DEPENDS})
  endif()
  
  if(BUILD_VL)
    list(APPEND proj_DEPENDENCIES ${VL_DEPENDS})
  endif()

  if(MSVC)
    # if we dont do this then windows headers will define all sorts of "keywords"
    # and compilation will fail with the weirdest errors.
    set(NIFTK_ADDITIONAL_C_FLAGS "${NIFTK_ADDITIONAL_C_FLAGS} -DWIN32_LEAN_AND_MEAN")
    set(NIFTK_ADDITIONAL_CXX_FLAGS "${NIFTK_ADDITIONAL_CXX_FLAGS} -DWIN32_LEAN_AND_MEAN")
    # poco is picky! for some reason the sdk version is not defined for niftk.
    # so we define it here explicitly:
    # http://msdn.microsoft.com/en-us/library/aa383745.aspx
    # 0x0501  = Windows XP
    # 0x0601  = Windows 7
    set(NIFTK_ADDITIONAL_C_FLAGS "${NIFTK_ADDITIONAL_C_FLAGS} -D_WIN32_WINNT=0x0601")
    set(NIFTK_ADDITIONAL_CXX_FLAGS "${NIFTK_ADDITIONAL_CXX_FLAGS} -D_WIN32_WINNT=0x0601")
  endif()

  # unfortunately, putting this here means we cannot enable pch on a non-superbuild.
  # instead it must be initially enabled, and later on can be disabled/enabled at will.
  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    # visual studio needs an extra parameter to increase the pch heap size.
    if (MSVC)
      set(NIFTK_ADDITIONAL_C_FLAGS "${NIFTK_ADDITIONAL_C_FLAGS} -Zm450")
      set(NIFTK_ADDITIONAL_CXX_FLAGS "${NIFTK_ADDITIONAL_CXX_FLAGS} -Zm450")
    endif()
  endif()

  ExternalProject_Add(${proj}
    LIST_SEPARATOR ^^
    DOWNLOAD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    BINARY_DIR ${proj}-build
    PREFIX ${proj}-cmake
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      # The CTK PluginFramework cannot cope with
      # a non-empty CMAKE_DEBUG_POSTFIX for the plugin
      # libraries yet.
      -DCMAKE_DEBUG_POSTFIX:STRING=
      -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
    CMAKE_CACHE_ARGS
      ${EP_COMMON_CACHE_ARGS}
      -DCMAKE_LIBRARY_PATH:PATH=${CMAKE_LIBRARY_PATH}
      -DCMAKE_INCLUDE_PATH:PATH=${CMAKE_INCLUDE_PATH}
      ${NIFTK_APP_OPTIONS}
      -DMITK_WHITELIST:STRING=${MITK_WHITELIST}
      -DMITK_WHITELISTS_EXTERNAL_PATH:STRING=${MITK_WHITELISTS_EXTERNAL_PATH}
      -DMITK_WHITELISTS_INTERNAL_PATH:STRING=${MITK_WHITELISTS_INTERNAL_PATH}
      -DNIFTK_EXTERNAL_PROJECT_PREFIX:PATH=${ep_prefix}
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=${CMAKE_VERBOSE_MAKEFILE}
      -DBUILD_TESTING:BOOL=${BUILD_TESTING} # The value set in EP_COMMON_ARGS normally forces this off, but we may need NifTK to be on.
      -DBUILD_SUPERBUILD:BOOL=OFF           # Must force this to be off, or else you will loop forever.
      -DBUILD_PCL:BOOL=${BUILD_PCL}
      -DBUILD_RTK:BOOL=${BUILD_RTK}
      -DBUILD_VL:BOOL=${BUILD_VL}
      -DBUILD_ITKFFTW:BOOL=${BUILD_ITKFFTW}
      -DBUILD_CAMINO:BOOL=${BUILD_CAMINO}
      -DBUILD_COMMAND_LINE_PROGRAMS:BOOL=${BUILD_COMMAND_LINE_PROGRAMS}
      -DBUILD_COMMAND_LINE_SCRIPTS:BOOL=${BUILD_COMMAND_LINE_SCRIPTS}
      -DBUILD_GUI:BOOL=${BUILD_GUI}
      -DBUILD_IGI:BOOL=${BUILD_IGI}
      -DBUILD_PROTOTYPE:BOOL=${BUILD_PROTOTYPE}
      -DBUILD_PROTOTYPE_JHH:BOOL=${BUILD_PROTOTYPE_JHH}
      -DBUILD_PROTOTYPE_TM:BOOL=${BUILD_PROTOTYPE_TM}
      -DBUILD_PROTOTYPE_GY:BOOL=${BUILD_PROTOTYPE_GY}
      -DBUILD_PROTOTYPE_KKL:BOOL=${BUILD_PROTOTYPE_KKL}
      -DBUILD_PROTOTYPE_BE:BOOL=${BUILD_PROTOTYPE_BE}
      -DBUILD_PROTOTYPE_MJC:BOOL=${BUILD_PROTOTYPE_MJC}
      -DBUILD_TESTING:BOOL=${BUILD_TESTING}
      -DBUILD_MESHING:BOOL=${BUILD_MESHING}
      -DBUILD_NiftyReg:BOOL=${BUILD_NiftyReg}
      -DBUILD_NiftyRec:BOOL=${BUILD_NiftyRec}
      -DBUILD_NiftySim:BOOL=${BUILD_NiftySim}
      -DBUILD_NiftySeg:BOOL=${BUILD_NiftySeg}
      -DCUDA_SDK_ROOT_DIR:PATH=${CUDA_SDK_ROOT_DIR}
      -DCUDA_CUT_INCLUDE_DIR:PATH=${CUDA_CUT_INCLUDE_DIR}
      -DNVAPI_INCLUDE_DIR:PATH=${NVAPI_INCLUDE_DIR}
      -DNVAPI_LIBRARY:PATH=${NVAPI_LIBRARY}
      -DNIFTK_USE_FFTW:BOOL=${NIFTK_USE_FFTW}
      -DNIFTK_USE_CUDA:BOOL=${NIFTK_USE_CUDA}
      -DNIFTK_DELAYLOAD_CUDA:BOOL=${NIFTK_DELAYLOAD_CUDA}
      -DNIFTK_USE_COTIRE:BOOL=${NIFTK_USE_COTIRE}
      -DNIFTK_WITHIN_SUPERBUILD:BOOL=ON                    # Set this to ON, as some compilation flags rely on knowing if we are doing superbuild.
      -DNIFTK_VERSION_MAJOR:STRING=${NIFTK_VERSION_MAJOR}
      -DNIFTK_VERSION_MINOR:STRING=${NIFTK_VERSION_MINOR}
      -DNIFTK_VERSION_PATCH:STRING=${NIFTK_VERSION_PATCH}
      -DNIFTK_PLATFORM:STRING=${NIFTK_PLATFORM}
      -DNIFTK_COPYRIGHT:STRING=${NIFTK_COPYRIGHT}
      -DNIFTK_ORIGIN_URL:STRING=${NIFTK_ORIGIN_URL}
      -DNIFTK_ORIGIN_SHORT_TEXT:STRING=${NIFTK_ORIGIN_SHORT_TEXT}
      -DNIFTK_ORIGIN_LONG_TEXT:STRING=${NIFTK_ORIGIN_LONG_TEXT}
      -DNIFTK_USER_CONTACT:STRING=${NIFTK_USER_CONTACT}
      -DNIFTK_BASE_NAME:STRING=${NIFTK_BASE_NAME}
      -DNIFTK_VERSION_STRING:STRING=${NIFTK_VERSION_STRING}
      -DNIFTK_GENERATE_DOXYGEN_HELP:BOOL=${NIFTK_GENERATE_DOXYGEN_HELP}
      -DNIFTK_VERBOSE_COMPILER_WARNINGS:BOOL=${NIFTK_VERBOSE_COMPILER_WARNINGS}
      -DNIFTK_CHECK_COVERAGE:BOOL=${NIFTK_CHECK_COVERAGE}
      -DNIFTK_USE_KWSTYLE:BOOL=${NIFTK_USE_KWSTYLE}
      -DNIFTK_USE_CPPCHECK:BOOL=${NIFTK_USE_CPPCHECK}
      -DNIFTK_ADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
      -DNIFTK_ADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
      -DNIFTK_FFTWINSTALL:PATH=${NIFTK_LINK_PREFIX}/fftw     # We don't have CMake SuperBuild version of FFTW, so must rely on it already being there
      -DNIFTK_SHOW_CONSOLE_WINDOW:BOOL=${NIFTK_SHOW_CONSOLE_WINDOW}
      -DBOOST_ROOT:PATH=${BOOST_ROOT}
      -DBOOST_VERSION:STRING=${NIFTK_VERSION_Boost}
      -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}
      -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}
      -DMITK_DIR:PATH=${MITK_DIR}
      -DCTK_DIR:PATH=${CTK_DIR}
      -DDCMTK_CMAKE_DEBUG_POSTFIX:STRING=d
      -DDCMTK_DIR:PATH=${DCMTK_DIR}
      -DCTK_SOURCE_DIR:PATH=${CTK_SOURCE_DIR}
      -DNiftyLink_DIR:PATH=${NiftyLink_DIR}
      -DNiftyLink_SOURCE_DIR:PATH=${NiftyLink_SOURCE_DIR}
      -DEigen_ROOT:PATH=${Eigen_ROOT}
      -DEigen_INCLUDE_DIR:PATH=${Eigen_INCLUDE_DIR}
      -DFLANN_ROOT:PATH=${FLANN_ROOT}
      -DVL_ROOT:PATH=${VL_ROOT}
      -DNiftyReg_DIR:PATH=${NiftyReg_DIR}
      -DNiftyReg_DEBUG_POSTFIX:STRING=d
      -DNiftyRec_DIR:PATH=${NiftyRec_DIR}
      -DNiftyRec_DEBUG_POSTFIX:STRING=d
      -DNiftySim_DIR:PATH=${NiftySim_DIR}
      -DNiftySim_DEBUG_POSTFIX:STRING=d
      -DNiftySeg_DIR:PATH=${NiftySeg_DIR}
      -DNiftySeg_DEBUG_POSTFIX:STRING=d
      -DSlicerExecutionModel_DIR:PATH=${SlicerExecutionModel_DIR}
    DEPENDS ${proj_DEPENDENCIES}
  )

  mitkFunctionInstallExternalCMakeProject(${proj})

endif()
