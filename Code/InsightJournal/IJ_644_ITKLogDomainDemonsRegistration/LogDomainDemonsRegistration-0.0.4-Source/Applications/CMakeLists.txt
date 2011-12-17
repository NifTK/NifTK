SD_EXECUTABLE(ImageCompare.cxx EXTLIBS ${Libraries})
SD_EXECUTABLE(DemonsRegistration.cxx EXTLIBS ${Libraries})
SD_EXECUTABLE(LogDomainDemonsRegistration.cxx EXTLIBS ${Libraries})

#ADD_TEST(RegisterImages ${MY_EXE_DIR}/DemonsRegistration -f ${CMAKE_SOURCE_DIR}/RatLungSlice1.mha -m ${CMAKE_SOURCE_DIR}/RatLungSlice2.mha -d -s 2 -g 1 -i 3x5 -e)
#ADD_TEST(CompareImage ${MY_EXE_DIR}/ImageCompare ${CMAKE_SOURCE_DIR}/RatLungSlice2-Reg.mha output.mha)