#if !defined(XNATREST_H)
#define XNATREST_H

#include "XnatRestExports.h"

#include "XnatRestStatus.h"

/* ======== BASIC FUNCTIONS ======== */

/* get message for XnatRest status */
/* input: XnatRest status */
/* returns: XnatRest status message */
XnatRest_EXPORT const char *getXnatRestStatusMsg( const XnatRestStatus status );

/* ======== ZIP FILE UTILITY FUNCTIONS  ======== */

/* unzip downloaded files */
/* input: zip filename and name of existing directory where */
/*        files extracted from zip file will be put */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus unzipXnatRestFile( const char *zipFilename, const char *outputDirectory );

/* unzip downloaded files; ignore directories included with filenames in zip file */
/* input: zip filename and name of existing directory where */
/*        files extracted from zip file will be put */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus unzipXnatRestFileNoDirs( const char *zipFilename, const char *outputDirectory );

/* zip files for upload */
/* input: zip filename, name of existing directory containing files to be zipped, */
/*        number of files to be zipped, and array of names of files to be zipped */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus zipXnatRestFile( const char *zipFilename, const char *inputDirectory,
                                int numFilenames, char const * const *filenames );

#endif
