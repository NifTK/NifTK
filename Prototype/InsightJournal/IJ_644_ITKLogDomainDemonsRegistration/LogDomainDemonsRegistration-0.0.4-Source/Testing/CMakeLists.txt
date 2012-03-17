SD_UNIT_TEST(itkDisplacementFieldCompositionFilterTest.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkVelocityFieldLieBracketFilterTest.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkLogDomainDeformableRegistrationFilterTest.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkOppositeImageFilterTest.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkExponentialDeformationFieldImageFilterTest2.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkVelocityFieldBCHCompositionFilterTest.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkLogDomainDemonsRegistrationFilterTest.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkLogDomainDemonsRegistrationFilterTest2.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkSymmetricLogDomainDemonsRegistrationFilterTest.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkSymmetricLogDomainDemonsRegistrationFilterTest2.cxx EXTLIBS ${Libraries})
SD_UNIT_TEST(itkTransformToVelocityFieldSourceTest.cxx EXTLIBS ${Libraries})

IF(MATLAB_FOUND)
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})
  SD_UNIT_TEST(vnl_sd_matrix_tools_test.cxx EXTLIBS ${Libraries} MATLAB_ENG MATLAB_MX)
  SD_APPEND_TARGET_PROPERTIES(vnl_sd_matrix_tools_test COMPILE_FLAGS "-DMATLAB_FOUND")
  IF(APPLE)
    SET_TARGET_PROPERTIES(vnl_sd_matrix_tools_test PROPERTIES LINK_FLAGS "-L${MATLAB_SYS}")
  ENDIF(APPLE)
ELSE(MATLAB_FOUND)
  SD_UNIT_TEST(vnl_sd_matrix_tools_test.cxx EXTLIBS ${Libraries})
ENDIF(MATLAB_FOUND)


# Also do a unit test that checks whether the registration result is the same
# whether we use pixel spacing and image direction or not
ADD_TEST(RegisterImagesLD ${PROJECT_BINARY_DIR}/bin/LogDomainDemonsRegistration
  -f ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice1.mhd
  -m ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice2.mhd
  -o ${PROJECT_BINARY_DIR}/Testing/output-ld.mha
  -d -i 3x3x5 -s 1 -g 1 -e -a 0)

ADD_TEST(RegisterImagesLDs ${PROJECT_BINARY_DIR}/bin/LogDomainDemonsRegistration
  -f ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice1s.mhd
  -m ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice2s.mhd
  -o ${PROJECT_BINARY_DIR}/Testing/output-ld-s.mha
  -d -i 3x3x5 -s 1 -g 1 -e -a 0)

ADD_TEST(CompareImageLD ${PROJECT_BINARY_DIR}/bin/ImageCompare 
   ${PROJECT_BINARY_DIR}/Testing/output-ld.mha ${PROJECT_BINARY_DIR}/Testing/output-ld-s.mha)


ADD_TEST(RegisterImagesSLD ${PROJECT_BINARY_DIR}/bin/LogDomainDemonsRegistration
  -f ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice1.mhd
  -m ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice2.mhd
  -o ${PROJECT_BINARY_DIR}/Testing/output-sld.mha
  -d -i 3x3x5 -s 1 -g 1 -e -a 1)

ADD_TEST(RegisterImagesSLDs ${PROJECT_BINARY_DIR}/bin/LogDomainDemonsRegistration
  -f ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice1s.mhd
  -m ${CMAKE_SOURCE_DIR}/TestingData/RatLungSlice2s.mhd
  -o ${PROJECT_BINARY_DIR}/Testing/output-sld-s.mha
  -d -i 3x3x5 -s 1 -g 1 -e -a 1)

ADD_TEST(CompareImageSLD ${PROJECT_BINARY_DIR}/bin/ImageCompare 
  ${PROJECT_BINARY_DIR}/Testing/output-sld.mha ${PROJECT_BINARY_DIR}/Testing/output-sld-s.mha)


# Add a test to check the style of the source code
FIND_PROGRAM(KWSTYLE_EXECUTABLE
  NAMES KWStyle 
  PATHS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Kitware Inc.\\KWStyle 1.0.0]/bin"
  )
MARK_AS_ADVANCED(KWSTYLE_EXECUTABLE)
IF(KWSTYLE_EXECUTABLE)
  ADD_TEST(StyleTest ${CMAKE_COMMAND} 
    -E chdir ${CMAKE_SOURCE_DIR}; ${KWSTYLE_EXECUTABLE}
    -xml ${CMAKE_SOURCE_DIR}/sd_kwstyle.kws.xml
    -D ${CMAKE_SOURCE_DIR}/sd_kwstyle.txt
    -gcc )
ELSE(KWSTYLE_EXECUTABLE)
  MESSAGE("Did not find KWStyle - the style test will not be run")
ENDIF(KWSTYLE_EXECUTABLE)
