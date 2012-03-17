PROJECT(DigitalTopology)

INCLUDE(${CMAKE_SOURCE_DIR}/IJMacros.txt)

ENABLE_TESTING()
INCLUDE(Dart)

SET(Required_Packages ITK)

FOREACH(Package ${Required_Packages})
  LOADPACKAGE(${Package})
ENDFOREACH(Package)

SET(Libraries ITKCommon ITKIO)

SET(CurrentExe "DigitalTopology")
FILE(GLOB DigitalTopologyFiles *.cxx *.h *.txx)
ADD_EXECUTABLE(${CurrentExe} ${DigitalTopologyFiles} ) 
TARGET_LINK_LIBRARIES(${CurrentExe} ${Libraries})

ADD_TEST(Test DigitalTopology Data/bunnyPadded.hdr Data/output.hdr)


