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

include(ExternalProject)

set(EP_BASE "${CMAKE_BINARY_DIR}/CMakeExternals")
set_property(DIRECTORY PROPERTY EP_BASE ${EP_BASE})

# For external projects like ITK, VTK we always want to turn their testing targets off.
set(EP_BUILD_TESTING OFF)
set(EP_BUILD_EXAMPLES OFF)
set(EP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

if(MSVC90 OR MSVC10)
  set(EP_COMMON_C_FLAGS "${CMAKE_C_FLAGS} /bigobj /MP /W0 /Zi")
  set(EP_COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /MP /W0 /Zi")
  # we want symbols, even for release builds!
  set(EP_COMMON_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /debug")
  set(EP_COMMON_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /debug")
  set(EP_COMMON_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /debug")
  set(CMAKE_CXX_WARNING_LEVEL 0)
else()
  if(${BUILD_SHARED_LIBS})
    set(EP_COMMON_C_FLAGS "${CMAKE_C_FLAGS} -DLINUX_EXTRA")
    set(EP_COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX_EXTRA")
  else()
    set(EP_COMMON_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -DLINUX_EXTRA")
    set(EP_COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -DLINUX_EXTRA")
  endif()
  # These are not relevant for linux but we set them anyway to keep
  # the variable bits below symmetric.
  set(EP_COMMON_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  set(EP_COMMON_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
  set(EP_COMMON_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")  
endif()

set(EP_COMMON_ARGS
  -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
  -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
  -DDESIRED_QT_VERSION:STRING=4
  -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${CMAKE_VERBOSE_MAKEFILE}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_C_FLAGS:STRING=${EP_COMMON_C_FLAGS}
  -DCMAKE_CXX_FLAGS:STRING=${EP_COMMON_CXX_FLAGS}
  -DCMAKE_EXE_LINKER_FLAGS:STRING=${EP_COMMON_EXE_LINKER_FLAGS}
  -DCMAKE_MODULE_LINKER_FLAGS:STRING=${EP_COMMON_MODULE_LINKER_FLAGS}
  -DCMAKE_SHARED_LINKER_FLAGS:STRING=${EP_COMMON_SHARED_LINKER_FLAGS}
  #debug flags
  -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
  -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
  #release flags
  -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
  -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
  #relwithdebinfo
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
  -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}
)

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

# Compute -G arg for configuring external projects with the same CMake generator:
if(CMAKE_EXTRA_GENERATOR)
  set(GEN "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
else()
  set(GEN "${CMAKE_GENERATOR}")
endif()

######################################################################
# Include NifTK macro for md5 checking
######################################################################

include(niftkMacroGetChecksum)

######################################################################
# Loop round for each external project, compiling it
######################################################################

set(EXTERNAL_PROJECTS
  BOOST
  VTK
  GDCM
  ITK
  SlicerExecutionModel  
  DCMTK
  CTK
  NiftyLink
# OpenCV is inserted here, just before MITK, if BUILD_IGI is ON
  MITK
  CGAL
  NiftySim
  NiftyReg
  NiftyRec
  NiftySeg
  NifTKData
)

if(BUILD_IGI)
  list(FIND EXTERNAL_PROJECTS MITK MITK_POSITION_IN_EXTERNAL_PROJECTS)
  list(INSERT EXTERNAL_PROJECTS ${MITK_POSITION_IN_EXTERNAL_PROJECTS} OpenCV)
  if(OPENCV_WITH_CUDA)
    message("Beware: You are building with OPENCV_WITH_CUDA! This means OpenCV will have a hard dependency on CUDA and will not work without it!")
  endif(OPENCV_WITH_CUDA)
endif(BUILD_IGI)

if(BUILD_IGI)
  # aruco depends on opencv. Our root CMakeLists.txt should have taken care of validating BUILD_IGI thereby turning on OpenCV.
  list(APPEND EXTERNAL_PROJECTS aruco)
endif()

foreach(p ${EXTERNAL_PROJECTS})
  include("CMake/CMakeExternals/${p}.cmake")
endforeach()

######################################################################
# Now compile NifTK, using the packages we just provided.
######################################################################
if(NOT DEFINED SUPERBUILD_EXCLUDE_NIFTKBUILD_TARGET OR NOT SUPERBUILD_EXCLUDE_NIFTKBUILD_TARGET)

  set(proj NIFTK)
  set(proj_DEPENDENCIES ${BOOST_DEPENDS} ${GDCM_DEPENDS} ${ITK_DEPENDS} ${SlicerExecutionModel_DEPENDS} ${VTK_DEPENDS} ${MITK_DEPENDS} )

  if(BUILD_IGI)
    list(APPEND proj_DEPENDENCIES ${OPENCV_DEPENDS})
  endif(BUILD_IGI)
  
  if(BUILD_TESTING)
    list(APPEND proj_DEPENDENCIES ${NifTKData_DEPENDS})
  endif(BUILD_TESTING)

  if(BUILD_IGI)
    list(APPEND proj_DEPENDENCIES ${NIFTYLINK_DEPENDS} ${aruco_DEPENDS})
  endif(BUILD_IGI)

  if(BUILD_NIFTYREG)
    list(APPEND proj_DEPENDENCIES ${NIFTYREG_DEPENDS})
  endif(BUILD_NIFTYREG)

  if(BUILD_NIFTYSEG)
    list(APPEND proj_DEPENDENCIES ${NIFTYSEG_DEPENDS})
  endif(BUILD_NIFTYSEG)

  if(BUILD_NIFTYSIM)
    list(APPEND proj_DEPENDENCIES ${NIFTYSIM_DEPENDS})
  endif(BUILD_NIFTYSIM)

  if(BUILD_NIFTYREC)
    list(APPEND proj_DEPENDENCIES ${NIFTYREC_DEPENDS})
  endif(BUILD_NIFTYREC)

  if(MSVC)
    # if we dont do this then windows headers will define all sorts of "keywords"
	# and compilation will fail with the weirdest errors.
    set(NIFTK_ADDITIONAL_C_FLAGS "${NIFTK_ADDITIONAL_C_FLAGS} -DWIN32_LEAN_AND_MEAN")
	set(NIFTK_ADDITIONAL_CXX_FLAGS "${NIFTK_ADDITIONAL_CXX_FLAGS} -DWIN32_LEAN_AND_MEAN")
  endif()

  ExternalProject_Add(${proj}
    DOWNLOAD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/NifTK-build
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      ${NIFTK_APP_OPTIONS}
      -DNIFTK_BUILD_ALL_APPS:BOOL=${NIFTK_BUILD_ALL_APPS}      
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=${CMAKE_VERBOSE_MAKEFILE}
      -DBUILD_TESTING:BOOL=${BUILD_TESTING} # The value set in EP_COMMON_ARGS normally forces this off, but we may need NifTK to be on.
      -DBUILD_SUPERBUILD:BOOL=OFF           # Must force this to be off, or else you will loop forever.
      -DBUILD_COMMAND_LINE_PROGRAMS:BOOL=${BUILD_COMMAND_LINE_PROGRAMS}
      -DBUILD_COMMAND_LINE_SCRIPTS:BOOL=${BUILD_COMMAND_LINE_SCRIPTS}
      -DBUILD_GUI:BOOL=${BUILD_GUI}
      -DBUILD_IGI:BOOL=${BUILD_IGI}
      -DBUILD_SLS_TESTING:BOOL=${BUILD_SLS_TESTING}
      -DBUILD_PROTOTYPE:BOOL=${BUILD_PROTOTYPE}
      -DBUILD_PROTOTYPE_JHH:BOOL=${BUILD_PROTOTYPE_JHH}
      -DBUILD_PROTOTYPE_TM:BOOL=${BUILD_PROTOTYPE_TM}
      -DBUILD_PROTOTYPE_GY:BOOL=${BUILD_PROTOTYPE_GY}
      -DBUILD_PROTOTYPE_KKL:BOOL=${BUILD_PROTOTYPE_KKL}
      -DBUILD_PROTOTYPE_BE:BOOL=${BUILD_PROTOTYPE_BE}
      -DBUILD_PROTOTYPE_MJC:BOOL=${BUILD_PROTOTYPE_MJC}
      -DBUILD_TESTING:BOOL=${BUILD_TESTING}
      -DBUILD_MESHING:BOOL=${BUILD_MESHING}
      -DBUILD_NIFTYREG:BOOL=${BUILD_NIFTYREG}
      -DBUILD_NIFTYREC:BOOL=${BUILD_NIFTYREC}
      -DBUILD_NIFTYSIM:BOOL=${BUILD_NIFTYSIM}
      -DBUILD_NIFTYSEG:BOOL=${BUILD_NIFTYSEG}
      -DCUDA_SDK_ROOT_DIR:PATH=${CUDA_SDK_ROOT_DIR}
      -DCUDA_CUT_INCLUDE_DIR:PATH=${CUDA_CUT_INCLUDE_DIR}
      -DNVAPI_INCLUDE_DIR:PATH=${NVAPI_INCLUDE_DIR}
      -DNVAPI_LIBRARY:PATH=${NVAPI_LIBRARY}
      -DNIFTK_USE_FFTW:BOOL=${NIFTK_USE_FFTW}
      -DNIFTK_USE_CUDA:BOOL=${NIFTK_USE_CUDA}
      -DNIFTK_WITHIN_SUPERBUILD:BOOL=ON                    # Set this to ON, as some compilation flags rely on knowing if we are doing superbuild.
      -DNIFTK_VERSION_MAJOR:STRING=${NIFTK_VERSION_MAJOR}
      -DNIFTK_VERSION_MINOR:STRING=${NIFTK_VERSION_MINOR}
      -DNIFTK_PLATFORM:STRING=${NIFTK_PLATFORM}
      -DNIFTK_COPYRIGHT:STRING=${NIFTK_COPYRIGHT}
      -DNIFTK_ORIGIN_URL:STRING=${NIFTK_ORIGIN_URL}
      -DNIFTK_ORIGIN_SHORT_TEXT:STRING=${NIFTK_ORIGIN_SHORT_TEXT}
      -DNIFTK_ORIGIN_LONG_TEXT:STRING=${NIFTK_ORIGIN_LONG_TEXT}
      -DNIFTK_WIKI_URL:STRING=${NIFTK_WIKI_URL}
      -DNIFTK_WIKI_TEXT:STRING=${NIFTK_WIKI_TEXT}
      -DNIFTK_DASHBOARD_URL:STRING=${NIFTK_DASHBOARD_URL}
      -DNIFTK_DASHBOARD_TEXT:STRING=${NIFTK_DASHBOARD_TEXT}
      -DNIFTK_USER_CONTACT:STRING=${NIFTK_USER_CONTACT}
      -DNIFTK_BASE_NAME:STRING=${NIFTK_BASE_NAME}
      -DNIFTK_VERSION_STRING:STRING=${NIFTK_VERSION_STRING}
      -DNIFTK_GENERATE_DOXYGEN_HELP:BOOL=${NIFTK_GENERATE_DOXYGEN_HELP}
      -DNIFTK_VERBOSE_COMPILER_WARNINGS:BOOL=${NIFTK_VERBOSE_COMPILER_WARNINGS}
      -DNIFTK_CHECK_COVERAGE:BOOL=${NIFTK_CHECK_COVERAGE}
      -DNIFTK_ADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
      -DNIFTK_ADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
      -DNIFTK_FFTWINSTALL:PATH=${NIFTK_LINK_PREFIX}/fftw     # We don't have CMake SuperBuild version of FFTW, so must rely on it already being there
      -DNIFTK_DATA_DIR:PATH=${NIFTK_DATA_DIR}
      -DVTK_DIR:PATH=${VTK_DIR}
      -DITK_DIR:PATH=${ITK_DIR}
      -DSlicerExecutionModel_DIR:PATH=${SlicerExecutionModel_DIR}
      -DBOOST_ROOT:PATH=${BOOST_ROOT}
      -DBOOST_VERSION:STRING=${NIFTK_VERSION_BOOST}
      -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}
      -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}
      -DMITK_DIR:PATH=${MITK_DIR}
      -DCTK_DIR:PATH=${CTK_DIR}
      -DCTK_SOURCE_DIR:PATH=${CTK_SOURCE_DIR}
      -DNiftyLink_DIR:PATH=${NiftyLink_DIR}
      -DNiftyLink_SOURCE_DIR:PATH=${NiftyLink_SOURCE_DIR}
      -DNIFTYREG_DIR:PATH=${EP_BASE}/Install/NIFTYREG
      -DNIFTYREC_DIR:PATH=${EP_BASE}/Install/NIFTYREC
      -DNIFTYSIM_DIR:PATH=${EP_BASE}/Install/NIFTYSIM
      -DNIFTYSEG_DIR:PATH=${EP_BASE}/Install/NIFTYSEG
      -Daruco_DIR:PATH=${aruco_DIR}
      -DOpenCV_DIR:PATH=${OpenCV_DIR}
      DEPENDS ${proj_DEPENDENCIES}
  )

endif()
