#if !defined(XNATRESTMINIZIP_H)
#define XNATRESTMINIZIP_H

#include "XnatRestStatus.h"

XnatRestStatus minizipXnatRestFiles( const char *fileDir, int numFilenames, char const * const *filenames, 
                                     const char *zipFilename );

#endif
