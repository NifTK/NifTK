#if !defined(XNATRESTMINIUNZ_H)
#define XNATRESTMINIUNZ_H

#include "XnatRestStatus.h"

#define MAXFILENAME (256)


/* unzip downloaded files */
/* input: zip filename and name of existing directory */
/*        where files extracted from zip file will be put */
/* returns: XnatRest status */
XnatRestStatus miniunzXnatRestFile( const char *zipFilename, const char *outputDir );

/* unzip downloaded files; ignore directories included with filenames in zip file */
/* input: zip filename and name of existing directory */
/*        where files extracted from zip file will be put */
/* returns: XnatRest status */
XnatRestStatus miniunzXnatRestFileNoDirs( const char *zipFilename, const char *outputDir );

#endif
