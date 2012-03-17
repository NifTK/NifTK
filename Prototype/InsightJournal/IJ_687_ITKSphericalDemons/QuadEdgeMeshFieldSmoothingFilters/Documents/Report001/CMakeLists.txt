PROJECT(QuadEdgeMeshFieldSmoothingFiltersReport001)

FIND_PACKAGE(LATEX)

IF(NOT PDFLATEX_COMPILER)
  MESSAGE("pdflatex compiler was not found. Please pass to advanced mode and provide its full path")
ELSE(NOT PDFLATEX_COMPILER)


FIND_PROGRAM(FIG2DEV_EXECUTABLE NAMES fig2dev)


#
# These are source files needed for the report generation.
#
SET(REPORT_SRCS
  algorithmic.sty
  algorithm.sty
  amssymb.sty
  fancyhdr.sty
  floatflt.sty
  fncychap.sty
  InsightArticle.cls
  InsightJournal.bib
  InsightJournal.ist
  InsightJournal.sty
  QuadEdgeMeshFieldSmoothingFilters.tex
  times.sty
  picins.sty
  ../../Examples/SmoothingMeshWithScalars.cxx
  ../../Examples/SmoothingMeshWithVectors.cxx
  )

#
# This adds a custom command for each source file in REPORT_SRCS
# that copies the file from the source directory to the binary
# directory where the pdf will be generated.
#
FOREACH(SOURCE_FILE ${REPORT_SRCS})
   ADD_CUSTOM_COMMAND(
     OUTPUT   ${QuadEdgeMeshFieldSmoothingFiltersReport001_BINARY_DIR}/${SOURCE_FILE}
     DEPENDS  ${QuadEdgeMeshFieldSmoothingFiltersReport001_SOURCE_DIR}/${SOURCE_FILE}
     COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${QuadEdgeMeshFieldSmoothingFiltersReport001_SOURCE_DIR}/${SOURCE_FILE} 
        ${QuadEdgeMeshFieldSmoothingFiltersReport001_BINARY_DIR}/${SOURCE_FILE}
    )
   SET(COPY_RESULTS ${COPY_RESULTS} ${QuadEdgeMeshFieldSmoothingFiltersReport001_BINARY_DIR}/${SOURCE_FILE})
ENDFOREACH(SOURCE_FILE ${REPORT_SRCS})


FILE(GLOB XFIG_FIGURES *.fig)

FOREACH(image ${XFIG_FIGURES})
  GET_FILENAME_COMPONENT(Imagebase ${image} NAME)
  GET_FILENAME_COMPONENT(ImageBase ${Imagebase} NAME_WE)
  ADD_CUSTOM_COMMAND( 
    OUTPUT    ${PROJECT_BINARY_DIR}/${ImageBase}.pdf
    DEPENDS   ${PROJECT_BINARY_DIR}/${ImageBase}.fig
    COMMAND   ${FIG2DEV_EXECUTABLE} 
      -L pdf ${PROJECT_BINARY_DIR}/${ImageBase}.fig ${PROJECT_BINARY_DIR}/${ImageBase}.pdf
    )
  SET(EPS_DEPS ${EPS_DEPS} ${PROJECT_BINARY_DIR}/${ImageBase}.pdf)
ENDFOREACH(image)


#
# Copy all the images from the Testing/Temporary directory
#
FILE(GLOB PNG_FIGURES ${QuadEdgeMeshFieldSmoothingFilters_BINARY_DIR}/Testing/Temporary/*.png )

FOREACH(SOURCE_FILE ${PNG_FIGURES})
  GET_FILENAME_COMPONENT(nameBase ${SOURCE_FILE} NAME)
  GET_FILENAME_COMPONENT(ImageBase ${nameBase} NAME_WE)
  SET( DESTINATION_FILE ${QuadEdgeMeshFieldSmoothingFiltersReport001_BINARY_DIR}/${ImageBase}.png)
  ADD_CUSTOM_COMMAND(
    OUTPUT   ${DESTINATION_FILE}
    DEPENDS  ${DESTINATION_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${SOURCE_FILE} 
        ${DESTINATION_FILE}
    )
  SET(COPY_RESULTS ${COPY_RESULTS} ${DESTINATION_FILE})
ENDFOREACH(SOURCE_FILE)



SET(REPORT_ELEMENTS 
  ${EPS_DEPS}
  )

#
# This adds a custom target that generates QuadEdgeMeshFieldSmoothingFiltersReport001.pdf
# This target depends on the list of copied files created
# with the custom command above and the Plots target.
#
ADD_CUSTOM_TARGET(ReportIJ ALL 
   COMMAND ${PDFLATEX_COMPILER} 
        ${QuadEdgeMeshFieldSmoothingFiltersReport001_SOURCE_DIR}/QuadEdgeMeshFieldSmoothingFilters.tex 
        -output-directory ${QuadEdgeMeshFieldSmoothingFiltersReport001_BINARY_DIR}
   COMMAND ${PDFLATEX_COMPILER} 
        ${QuadEdgeMeshFieldSmoothingFiltersReport001_SOURCE_DIR}/QuadEdgeMeshFieldSmoothingFilters.tex 
        -output-directory ${QuadEdgeMeshFieldSmoothingFiltersReport001_BINARY_DIR}
   DEPENDS ${COPY_RESULTS} ${REPORT_ELEMENTS} 
   WORKING_DIRECTORY ${QuadEdgeMeshFieldSmoothingFiltersReport001_BINARY_DIR}
   )
 
ENDIF(NOT PDFLATEX_COMPILER)
