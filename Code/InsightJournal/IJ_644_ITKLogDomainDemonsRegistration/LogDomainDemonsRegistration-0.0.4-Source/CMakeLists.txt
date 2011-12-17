#-----------------------------------------------------------------------------
# Name of the project
PROJECT(LogDomainDemonsRegistration)


#-----------------------------------------------------------------------------
# Requires a recent version of CMake
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)


#-----------------------------------------------------------------------------
# Include some useful macros
INCLUDE(${CMAKE_SOURCE_DIR}/IJMacros.txt)
INCLUDE(${CMAKE_SOURCE_DIR}/SDCMakeMacros.cmake)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})


#-----------------------------------------------------------------------------
# Set up some default paths
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)


#-----------------------------------------------------------------------------
#Declare any external dependencies that your project may have here.
#examples include: ITK, VTK, JPEG, PNG, OpenGL, ZLIB, Perl, Java
#If you're not sure what name to use, look in the Modules directory of your
#cmake install and check that a file named Find(Package).cmake exists
SET(Required_Packages
  ITK
  )


#-----------------------------------------------------------------------------
#this foreach loads all of the packages that you specified as required.
#It shouldn't need to be modified.
FOREACH(Package ${Required_Packages})
  LOAD_REQUIRED_PACKAGE(${Package})
ENDFOREACH(Package)


#-----------------------------------------------------------------------------
#Declare any optional external dependencies that your project may have here.
#examples include: ITK, VTK, JPEG, PNG, OpenGL, ZLIB, Perl, Java
#If you're not sure what name to use, look in the Modules directory of your
#cmake install and check that a file named Find(Package).cmake exists
SET(Optional_Packages
  Matlab
  )


#-----------------------------------------------------------------------------
#this foreach loads all of the packages that you specified as optional.
#It shouldn't need to be modified.
FOREACH(Package ${Optional_Packages})
  LOAD_OPTIONAL_PACKAGE(${Package})
ENDFOREACH(Package)


#-----------------------------------------------------------------------------
#Some packages options are required for this project. Check them now
SET(_ITKVersionString "${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR}.${ITK_VERSION_PATCH}" )
MATH(EXPR _ITKVersionNum "${ITK_VERSION_MAJOR}*100*100 + ${ITK_VERSION_MINOR}*100 + ${ITK_VERSION_PATCH}")
IF( _ITKVersionNum LESS 30800 )
  MESSAGE(SEND_ERROR "The ITK version you want to use (${_ITKVersionString}) is not supported by this project. Please use a more recent version of ITK. The minimum required version is 3.8.0. The recommended version is 3.14.0.")
ENDIF( _ITKVersionNum LESS 30800 )
IF(NOT ITK_USE_REVIEW)
  MESSAGE(SEND_ERROR "ITK_USE_REVIEW has not been set to on when compiling ITK. This option is mandatory for this project to build. Please recompile your ITK installation.")
ENDIF(NOT ITK_USE_REVIEW)


#-----------------------------------------------------------------------------
#Set any libraries that your project depends on.
#examples: ITKCommon, VTKRendering, etc
SET(Libraries
  ITKCommon
  ITKIO
  ITKStatistics
  )


#-----------------------------------------------------------------------------
#Set any extra compilation flags here
IF(CMAKE_COMPILER_IS_GNUCXX)
  # Add some useful warning flags
  # Note: -Wall and -Wno-deprecated are already pulled in by
  # UseITK.cmake
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wcast-qual")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow -Wcast-qual")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)


#-----------------------------------------------------------------------------
#Build the ITK code if necessary or simply add the corresponding include dirs
INCLUDE_DIRECTORIES(Code)
ADD_SUBDIRECTORY(Code)


#-----------------------------------------------------------------------------
#Build the matlab bindings if we have some
IF(MATLAB_FOUND)
  message("Found Matlab. Matlab wrappers (mex files) and unit tests will be built if there are some in the project.")
  IF(EXISTS ${CMAKE_SOURCE_DIR}/MatlabCode)
    ADD_SUBDIRECTORY(MatlabCode)
  ENDIF(EXISTS ${CMAKE_SOURCE_DIR}/MatlabCode)
ENDIF(MATLAB_FOUND)


#-----------------------------------------------------------------------------
#Build the unit tests / configure dashboard
INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)
MARK_AS_ADVANCED(TCL_TCLSH DART_ROOT)

SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
MARK_AS_ADVANCED(BUILDNAME)

ENABLE_TESTING()
ADD_SUBDIRECTORY(Testing)


#-----------------------------------------------------------------------------
#Build the applications
ADD_SUBDIRECTORY(Applications)


#-----------------------------------------------------------------------------
# CPack is only used to generate the source package
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "4")

INCLUDE(CPack)

SET(CPACK_SOURCE_IGNORE_FILES
  ${CPACK_SOURCE_IGNORE_FILES}
  "CTestConfig.cmake"
  "/Documents/.*/"
  "/Documents/.*/"
  "/MatlabCode/"
  "README-Matlab\\\\.txt"
  "/TestingData/[^R][^a][^t].*[^mr][^ha][^daw]"
  "/TestingData/.*\\\\.base\\\\.png$"
  "/TestingData/.*\\\\.test\\\\.png$"
  "/TestingData/.*\\\\.diff\\\\.png$"
  "\\\\.nii$"
  "~$"
  )

# reinclude cpack to take into account CPACK_SOURCE_IGNORE_FILES
INCLUDE(CPack)

