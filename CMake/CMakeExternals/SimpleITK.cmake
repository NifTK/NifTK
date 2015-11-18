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

#-----------------------------------------------------------------------------
# SimpleITK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED SimpleITK_DIR AND NOT EXISTS ${SimpleITK_DIR})
  message(FATAL_ERROR "SimpleITK_DIR variable is defined but corresponds to non-existing directory")
endif()

set(version "0.8.1")
set(location "${NIFTK_EP_TARBALL_LOCATION}/SimpleITK-${version}.tar.gz")
niftkMacroDefineExternalProjectVariables(SimpleITK ${version} ${location})

if(MITK_USE_SimpleITK)

  set(proj_DEPENDENCIES ITK GDCM SWIG)
  if(MITK_USE_OpenCV)
    list(APPEND proj_DEPENDENCIES OpenCV)
  endif()

  if(NOT DEFINED SimpleITK_DIR)

    set(additional_cmake_args )

    list(APPEND additional_cmake_args
         -DWRAP_CSHARP:BOOL=OFF
         -DWRAP_TCL:BOOL=OFF
         -DWRAP_LUA:BOOL=OFF
         -DWRAP_PYTHON:BOOL=OFF
         -DWRAP_JAVA:BOOL=OFF
         -DWRAP_RUBY:BOOL=OFF
         -DWRAP_R:BOOL=OFF
        )

    if(MITK_USE_Python)
      list(APPEND proj_DEPENDENCIES Python)
      list(APPEND additional_cmake_args
           -DWRAP_PYTHON:BOOL=ON
           -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
           -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
           -DPYTHON_INCLUDE_DIR2:PATH=${PYTHON_INCLUDE_DIR2}
           -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
          )
    endif()

    if(CTEST_USE_LAUNCHERS)
      list(APPEND additional_cmake_args
        "-DCMAKE_PROJECT_${proj}_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake"
      )
    endif()

    #TODO: Installer and testing works only with static libs on MAC
    set(_build_shared ON)
    if(APPLE)
      set(_build_shared OFF)
    endif()

    ExternalProject_Add(${proj}
       LIST_SEPARATOR ^^
       PREFIX ${proj_CONFIG}
       SOURCE_DIR ${proj_SOURCE}
       BINARY_DIR ${proj_BUILD}
       INSTALL_DIR ${proj_INSTALL}
       URL ${proj_LOCATION}
       URL_MD5 ${proj_CHECKSUM}
       PATCH_COMMAND ${PATCH_COMMAND} -N -p1 -i ${CMAKE_CURRENT_LIST_DIR}/SimpleITK-0.8.1.patch
       CMAKE_ARGS
         ${EP_COMMON_ARGS}
         # -DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON
      CMAKE_CACHE_ARGS
         ${EP_COMMON_CACHE_ARGS}
         ${additional_cmake_args}
         -DBUILD_SHARED_LIBS:BOOL=${_build_shared}
         -DSimpleITK_BUILD_DISTRIBUTE:BOOL=ON
         -DSimpleITK_PYTHON_THREADS:BOOL=ON
         -DUSE_SYSTEM_ITK:BOOL=ON
         -DBUILD_EXAMPLES:BOOL=OFF
         -DGDCM_DIR:PATH=${GDCM_DIR}
         -DITK_DIR:PATH=${ITK_DIR}/lib/cmake/ITK-4.7/
         -DSWIG_DIR:PATH=${SWIG_DIR}
         -DSWIG_EXECUTABLE:FILEPATH=${SWIG_EXECUTABLE}
       CMAKE_CACHE_DEFAULT_ARGS
         ${EP_COMMON_CACHE_DEFAULT_ARGS}
       DEPENDS ${proj_DEPENDENCIES}
      )

    set(SimpleITK_DIR ${proj_INSTALL})

    if( MITK_USE_Python )
      set(_sitk_build_dir ${proj_BUILD})

      # Build python distribution with easy install. If a own runtime is used
      # embedd the egg into the site-package folder of the runtime

      # Note: Userbase install could also be relevant in some cases Probably windows want's to
      # install to Lib/python2.7/
      # Build egg into custom user base folder and deploy it later into installer
      # https://pythonhosted.org/setuptools/easy_install.html#use-the-user-option-and-customize-pythonuserbase
      # PYTHONUSERBASE=${_install_dir} ${PYTHON_EXECUTABLE} setup.py --user

      # PythonDir needs to be fixed for the python interpreter by
      # changing dir delimiter for Windows
      if(MITK_USE_SYSTEM_PYTHON)
        set(_install_dir ${EP_BASE}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
      else()
        set(_install_dir ${Python_DIR})
      endif()
      if(WIN32)
        STRING(REPLACE "/" "\\\\" _install_dir ${_install_dir})
      else()
        # escape spaces in the install path for linux
        STRING(REPLACE " " "\ " _install_dir ${_install_dir})
      endif()

      if( MITK_USE_SYSTEM_PYTHON )
        ExternalProject_Add_Step(${proj} sitk_python_install_step
          COMMAND PYTHONUSERBASE=${_install_dir} ${PYTHON_EXECUTABLE} setup.py install --user
          DEPENDEES install
          WORKING_DIRECTORY ${_sitk_build_dir}/Wrapping/PythonPackage
        )
      else()
        ExternalProject_Add_Step(${proj} sitk_python_install_step
          COMMAND ${PYTHON_EXECUTABLE} setup.py install --prefix=${_install_dir}
          DEPENDEES install
          WORKING_DIRECTORY ${_sitk_build_dir}/Wrapping/PythonPackage
        )
      endif()
    endif()

    mitkFunctionInstallExternalCMakeProject(${proj})
    message("SuperBuild loading SimpleITK from ${SimpleITK_DIR}")
    set(NifTK_PREFIX_PATH ${SimpleITK_DIR}^^${NifTK_PREFIX_PATH})

    # Still need to install the SimpleITK Python wrappings

  else()
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
    message("SuperBuild receiving SimpleITK from ${SimpleITK_DIR}")
  endif()
endif()

