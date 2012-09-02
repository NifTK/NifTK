
set(CPP_FILES
  XnatRest.c
  XnatRestIoapi.c
  XnatRestMiniUnz.c
  XnatRestMiniZip.c
  XnatRestUnzip.c
  XnatRestZip.c
)

if(WIN32)
  set(CPP_FILES ${CPP_FILES} XnatRestIowin32.c)
endif(WIN32)
