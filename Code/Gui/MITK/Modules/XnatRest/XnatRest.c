#include "XnatRest.h"

#include <stdlib.h>
#include <string.h>

#include "XnatRestMiniUnz.h"
#include "XnatRestMiniZip.h"

static char httpRespCodeMsg[128] = { "No error message available" };
static char ioErrorMsg[128] = { "No error message available" };
static char wsaStartupErrorMsg[128] = { "No error message available" };

static const char *XnatRestStatusMsg[] =
{
  "OK",
  "Memory allocation failure",
  "libcurl environment initialization failure",
  "Cannot get libcurl easy interface handle",
  "Cannot get libcurl multi interface handle",
  "Cannot set user ID and password",
  "Cannot set HTTP GET option",
  "Cannot set HTTP PUT option",
  "Cannot set HTTP DELETE option",
  "Cannot set HTTP REST URI",
  "Cannot connect to web site (Web site address may be wrong)",
  "HTTP error code 401 (User not authorized)",
  "HTTP error code 404 (Invalid query)",
  "HTTP error code >= 300 returned",
  "libcurl easy interface error",
  "libcurl multi interface error",
  "Another asynchronous file transfer is in progress",
  "No asynchronous file transfer is in progress",
  "Missing node name in REST URI for file in resource of reconstruction",
  "Missing file path in REST URI for file in resource of reconstruction",
  "XNAT REST Error",
  "Cannot open file for download",
  "Number of csv values overflow column values array size",
  "Missing quotation marks in column during csv line parsing",
  "Invalid characters outside quotation marks during csv line parsing",
  "Column not found during csv line parsing",
  "Invalid column number inpput to csv line parsing",
  "Memory allocation failure during csv line parsing",
  "Callback data buffer overflow",
  "XNAT REST Callback Error",
  "No zip filename specified",
  "Cannot open zip file to extract files",
  "Cannot get global information from zip file",
  "Cannot get information on next file to extract from zip file",
  "Cannot get name of current file to be extracted from zip file",
  "Cannot open current file to be extracted from zip file",
  "Cannot read contents of current file from zip file",
  "Cannot close current file extracted from zip file",
  "IO error opening output file for writing contents of current file from zip file",
  "IO error writing to output file the contents of current file from zip file",
  "Invalid directory name for output files from zip file",
  "Cannot create directory for output files from zip file",
  "Memory allocation failure during file extraction from zip file",
  "Cannot create new zip file",
  "Cannot open new compressed file in zip file",
  "Cannot write compressed contents into new file in zip file (err < 0)",
  "Cannot write compressed contents into new file in zip file (err > 0)",
  "Cannot close new compressed file in zip file",
  "IO error opening input file for copying compressed contents into zip file",
  "IO error reading input file for copying compressed contents into zip file",
  "Memory allocation failure during new zip file creation",
  "WSAStartup failed",
  "Invalid Python sequence"
};

#ifdef _WIN32
  static const char OS_PATH_SLASH = '\\';
#else
  static const char OS_PATH_SLASH = '/';
#endif

/* unzip downloaded files */
XnatRestStatus unzipXnatRestFile( const char *zipFilename, const char *outputDirectory )
{
  char outputDir[MAXFILENAME];
  int outputDirLen;
  char *cp;

  /* check input zip filename */
  if ( zipFilename == NULL )
  {
    return XNATREST_UNZ_NOZIPNAME;
  }

  /* check name of output directory */
  if ( outputDirectory == NULL || *outputDirectory == '\0' )
  {
    *outputDir = '\0';
  }
  else
  {
    outputDirLen = strlen( outputDirectory );
    if ( outputDirLen < ( MAXFILENAME - 1 ) )
    {
      strcpy( outputDir, outputDirectory );
      /* add (back)slash to end of output directory name, if not present */
      cp = outputDir + outputDirLen - 1;
      if ( *cp != OS_PATH_SLASH )
      {
        *++cp = OS_PATH_SLASH;
        *++cp = '\0';
      }
    }
    else
    {
      return XNATREST_UNZ_BADOUTDIR;
    }
  }

  /* unzip zip file into output directory */
  return miniunzXnatRestFile( zipFilename, outputDir );
}

/* zip files for upload */
XnatRestStatus zipXnatRestFile( const char *zipFilename, const char *inputDirectory,
                                int numFilenames, char const * const *filenames )
{
  char *fileDir;
  int inputDirLen;
  char noDirectory[] = "";
  char *cp;
  XnatRestStatus status = XNATREST_OK;

  if ( inputDirectory == NULL || *inputDirectory == '\0' )
  {
    fileDir = noDirectory;
  }
  else
  {
    inputDirLen = strlen( inputDirectory );
    fileDir = (char *) malloc( ( inputDirLen + 2 ) * sizeof( char ) );
    if ( fileDir == NULL )
    {
      return XNATREST_ZIP_NOMEM;
    }
    strcpy( fileDir, inputDirectory );
    /* add (back)slash to end of input directory name, if not present */
    cp = fileDir + inputDirLen - 1;
    if ( *cp != OS_PATH_SLASH )
    {
      *++cp = OS_PATH_SLASH;
      *++cp = '\0';
    }
  }

  status = minizipXnatRestFiles( fileDir, numFilenames, filenames, zipFilename );

  if ( fileDir != noDirectory )
  {
    free( fileDir );
  }

  return status;
}

/* get status message */
const char *getXnatRestStatusMsg( const XnatRestStatus status )
{
  static const char *unknownStatus = "No such XNAT REST status message";
  const char *cp;

  if ( status == XNATREST_HTTPERROR )
  {
    cp = httpRespCodeMsg;
  }
  else if ( status == XNATREST_IOERROR )
  {
    cp = ioErrorMsg;
  }
  else if ( status == XNATREST_WSASTARTUPERR )
  {
    cp = wsaStartupErrorMsg;
  }
  else if ( ( status >= XNATREST_OK ) && ( status < XNATREST_END ) )
  {
    cp = XnatRestStatusMsg[status];
  }
  else
  {
    cp = unknownStatus;
  }

  return cp;
}
