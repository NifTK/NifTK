#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <sys/types.h>
#include <winsock2.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif
#include <curl.h>
#include "XnatRestMiniUnz.h"
#include "XnatRestMiniZip.h"
#include "XnatRest.h"


static CURL *curl = NULL;
static CURLM *curlMulti = NULL;

static FILE *fpMulti = NULL;
static size_t moreBytes = 0;

enum XnatRestBooleanCodes
{
  XRBOOL_TRUE,
  XRBOOL_FALSE
};

typedef enum XnatRestBooleanCodes XnatRestBoolean;

static XnatRestBoolean asynXferInProgress = XRBOOL_FALSE;
static XnatRestBoolean firstXferCallMade = XRBOOL_FALSE;


/* before v7.17.0, strings were NOT copied by libcurl */
static char *userpwd = NULL;  /* XNAT user ID and password */
static char *xnatUrl = NULL;  /* URL address of XNAT web site */
static char *restUri = NULL;  /* XNAT REST HTTP URI */

#define MAX_BUF	1048576

static char wr_buf[MAX_BUF+1];
static int  wr_index = 0;

static CURLcode badCurlReturn = CURLE_OK;
static CURLMcode badCurlMultiReturn = CURLM_OK;
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

static size_t readNoDataCB( void *buffer, size_t size, size_t nmemb, FILE *stream );
static size_t readDataFileCB( void *buffer, size_t size, size_t nmemb, FILE *stream );
static size_t readDataFileMultiCB( void *buffer, size_t size, size_t nmemb, FILE *stream );
static size_t writeDataBufCB( void *buffer, size_t size, size_t nmemb, void *userp );
static size_t writeDataFileCB( void *buffer, size_t size, size_t nmemb, FILE *stream );
static size_t writeDataFileMultiCB( void *buffer, size_t size, size_t nmemb, FILE *stream );

static XnatRestStatus skipColValue( char **colPos );
static XnatRestStatus getColValue( char **colPos, char **valStart, int *len );
static XnatRestStatus parseCsvLine( const int col, char *csvLine, char **value, char **remainingLines );
static XnatRestStatus getColValuesFromCSV( const int col, int *numValues, char ***colValues );

static void removeDuplicateNames( int *numNames, char ***names );
static XnatRestStatus extractFilePathFromRestUri( int *numFilenames, char ***filenames );

static XnatRestStatus setHttpRestUri( const char **query );
static XnatRestStatus setHttpGetUri( const char **query );
static XnatRestStatus performXnatRestAction( const XnatRestStatus *wr_error );
static XnatRestStatus performXnatRestGet( const char **query, const int col, 
                                          int *numValues, char ***colValues );
static XnatRestStatus performXnatRestFileGet( const char **query, const char *filename );

static XnatRestStatus initXnatRestAsynFileGet( const char **query, const char *filename );
static XnatRestStatus waitForMoreAsynData( void );

static XnatRestStatus setHttpPutUri( const char **query );
static XnatRestStatus performXnatRestPut( const char **query );
static XnatRestStatus performXnatRestFilePut( const char **query, const char *filename );
static XnatRestStatus initXnatRestAsynFilePut( const char **query, const char *filename );

static XnatRestStatus performXnatRestDelete( const char **query );


/* read no data for upload on HTTP PUT */
static size_t readNoDataCB( void *buffer, size_t size, size_t nmemb, FILE *stream )
{
  return (size_t) 0;
}

/* read data from file for synchronouse upload on HTTP PUT */
static size_t readDataFileCB( void *buffer, size_t size, size_t nmemb, FILE *stream )
{
  size_t numBytes;  /* number of bytes read from input file */

  /* read contents of input file into buffer */
  numBytes = fread( buffer, size, nmemb, stream );

  return numBytes;
}

/* read data from file for asynchronous upload on HTTP PUT */
static size_t readDataFileMultiCB( void *buffer, size_t size, size_t nmemb, FILE *stream )
{
  size_t numBytes;  /* number of bytes read from input file */

  /* read contents of input file into buffer */
  numBytes = fread( buffer, size, nmemb, stream );

  /* increment number of bytes read */
  moreBytes += numBytes;

  return numBytes;
}

static size_t writeDataBufCB( void *buffer, size_t size, size_t nmemb, void *userp )
{
  int segsize = size * nmemb;

  /* printf( "writeDataBufCB received %d bytes\n", segsize ); */

  /* Check to see if this data exceeds the size of our buffer. If so, 
   * set the user-defined context value and return 0 to indicate a
   * problem to curl.
   */
  if ( wr_index + segsize > MAX_BUF ) {
    *(XnatRestStatus *)userp = XNATREST_CB_OVERFLOW;
    return 0;
  }

  /* Copy the data from the curl buffer into our buffer */
  memcpy( (void *)&wr_buf[wr_index], buffer, (size_t)segsize );

  /* Update the write index */
  wr_index += segsize;

  /* Null terminate the buffer */
  wr_buf[wr_index] = 0;

  /*
  printf("========\n");
  printf("%s\n", wr_buf );
  printf("========\n");
  */

  /* Return the number of bytes received, indicating to curl that all is okay */
  return segsize;
}

static size_t writeDataFileCB( void *buffer, size_t size, size_t nmemb, FILE *stream )
{
  size_t numBytes;  /* number of bytes written to output file */

  /* write contents of buffer to output file */
  numBytes = fwrite( buffer, size, nmemb, stream );
  return numBytes;
}

static size_t writeDataFileMultiCB( void *buffer, size_t size, size_t nmemb, FILE *stream )
{
  size_t numBytes;  /* number of bytes written to output file */

  /* write contents of buffer to output file */
  numBytes = fwrite( buffer, size, nmemb, stream );

  /* increment number of bytes written */
  moreBytes += numBytes;

  return numBytes;
}

static XnatRestStatus skipColValue( char **colPos )
{
  char *cp = *colPos;  /* current position in column */

  /* find open quote */
  while ( *cp != '\"' )
  {
    if ( *cp == ',' )
    {
      *colPos = ++cp;
      return XNATREST_OK;
    }
    else if ( ( *cp == '\n' ) || ( *cp == '\0' ) )
    {
      *colPos = cp;
      return XNATREST_OK;
    }
    cp++;
  }

  /* find close quote */
  cp++;
  while ( *cp != '\"' )
  {
    if ( *cp == '\0' )
    {
      *colPos = cp;
      return XNATREST_CSV_NOQUOTE;
    }
    cp++;
  }

  /* find comma at end of column */
  cp++;
  while ( *cp != ',' )
  {
    if ( ( *cp == '\n' ) || ( *cp == '\0' ) )
    {
      *colPos = cp;
      return XNATREST_OK;
    }
    else if ( !isspace( *cp ) )
    {
      *colPos = cp;
      return XNATREST_CSV_INVALID;
    }
    cp++;
  }

  /* output starting position for next column */
  *colPos = ++cp;

  return XNATREST_OK;
}

static XnatRestStatus getColValue( char **colPos, char **valStart, int *len )
{
  char *cp = *colPos;  /* current position in column */
  char *vp = NULL;     /* start of value in column */

  /* initialize output column value */
  *valStart = NULL;
  *len = 0;

  /* find start of value in column */
  while ( *cp != '\"' )
  {
    if ( *cp == ',' )
    {
      *colPos = ++cp;
      return XNATREST_CSV_NOCOL;
    }
    else if ( ( *cp == '\n' ) || ( *cp == '\0' ) )
    {
      *colPos = cp;
      return XNATREST_CSV_NOCOL;
    }
    else if ( !isspace( *cp ) )
    {
      break;
    }
    cp++;
  }

  if ( *cp == '\"' )  /* quoted column value */
  {
    vp = ++cp;

    /* find close quote */
    while ( *cp != '\"' )
    {
      if ( *cp == '\0' )
      {
        *colPos = cp;
        return XNATREST_CSV_NOQUOTE;
      }
      cp++;
    }

    /* set output column value */
    *valStart = vp;
    *len = cp++ - vp;

    /* find end of column */
    while ( ( *cp != ',' ) && ( *cp != '\n' ) && ( *cp != '\0' ) )
    {
      if ( !isspace( *cp ) )
      {
        *colPos = cp;
        return XNATREST_CSV_INVALID;
      }
      cp++;
    }
  }
  else  /* unquoted column value */
  {
    vp = cp++;

    /* find end of unquoted column value */
    while ( ( *cp != ',' ) && ( *cp != '\n' ) && ( *cp != '\0' ) )
    {
      cp++;
    }

    /* set output column value */
    *valStart = vp;
    *len = cp - vp;
  }

  /* output starting position for next column */
  *colPos = ( ( *cp == '\n' ) || ( *cp == '\0' ) ) ? cp : ++cp;

  return XNATREST_OK;
}

static XnatRestStatus parseCsvLine( const int col, char *csvLine, char **value, char **remainingLines )
{
  int n = 1;           /* current column in line */
  char *cp = csvLine;  /* current character in line */
  char *valStart;      /* pointer to start of value in specified column */
  char *valEnd;        /* pointer to character after end of value in specified column */
  int len;             /* length of value */
  char *vp;            /* current character in value */
  XnatRestStatus status;

  /* check input column number */
  if ( col < n )
  {
    return XNATREST_CSV_BADCOL;  /* invalid column number */
  }

  /* skip over columns before specified column on current line */
  while ( ( n < col ) && ( *cp != '\n' ) && ( *cp != '\0' ) )
  {
    status = skipColValue( &cp );
    if ( status != XNATREST_OK )
    {
      return status;
    }
    n++;
  }

  if ( ( *cp == '\n' ) || ( *cp == '\0' ) )
  {
    return XNATREST_CSV_NOCOL;  /* could not find column */
  }

  /* get value from specified column on current line */
  status = getColValue( &cp, &valStart, &len );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* copy specified column value */
  *value = (char *) malloc( ( len + 1 ) * sizeof(char) );
  if ( *value == NULL )
  {
    return XNATREST_CSV_NOMEM;
  }
  vp = *value;
  valEnd = valStart + len;
  while ( valStart < valEnd )
  {
    *vp++ = *valStart++;
  }
  *vp = '\0';

  /* skip over remaining columns on current line */
  while ( ( *cp != '\n' ) && ( *cp != '\0' ) )
  {
    status = skipColValue( &cp );
    if ( status != XNATREST_OK )
    {
      return status;
    }
  }

  /* output start of remaining csv lines */
  if ( *cp == '\n' )
  {
    cp++;
  }
  *remainingLines = cp;

  return XNATREST_OK;
}

static XnatRestStatus getColValuesFromCSV( const int col, int *numValues, char ***colValues )
{
  static const int MAX_COLVALUES = 8191;
  int lineNumber = 0;
  char *lineStart = wr_buf;  /* set pointer to beginning of buffer holding csv lines */
  char *colValuePtr;
  char *remainingLines;
  XnatRestStatus status;

  /* allocate memory for array of column values */
  *numValues = 0;
  *colValues = (char **) malloc( ( MAX_COLVALUES + 1 ) * sizeof(char *) );
  if ( *colValues == NULL )
  {
    return XNATREST_NOMEM;  /* cannot allocate memory for array of column values */
  }

  /* copy value from specified column in csv line to array of column values*/
  while ( *lineStart != '\0' )
  {
    /* extract specified column value from current csv line */
    status = parseCsvLine( col, lineStart, &colValuePtr, &remainingLines );
    if ( status != XNATREST_OK )
    {
      return status;
    }

    if ( lineNumber > 0 )  /* store column value from line in output array */
    {
      /* check for array overflow */
      if ( *numValues >= MAX_COLVALUES )
      {
        return XNATREST_COL_OVERFLOW;  /* number of csv values overflow column values array size */
      }

      (*colValues)[*numValues] = colValuePtr;
      *numValues += 1;
    }
    else /* free memory allocated for unneeded column value */
    {
      free( colValuePtr );
    }

    /* increment number of csv lines parsed */
    lineNumber++;

    /* set pointer to start of next line */
    lineStart = remainingLines;
  }

  /* add sentinel */
  (*colValues)[*numValues] = NULL;

  return XNATREST_OK;
}

/* remove duplicate names from array of names */
static void removeDuplicateNames( int *numNames, char ***names )
{
  int numUnique;
  char **cp;
  int i, j;

  /* check if more than one name in input array */
  if ( *numNames < 2 )
  {
    return;
  }

  /* set number of known unique names */
  numUnique = 1;

  /* remove duplicate names from array of names */
  cp = *names;
  for ( i = 1 ; i < *numNames ; i++ )
  {
    /* check if current name in known unique names */
    for ( j = 0 ; j < numUnique ; j++ )
    {
      if ( strcmp( cp[i], cp[j] ) == 0 )
      {
        break;
      }
    }

    if ( j < numUnique )  /* current name is duplicate */
    {
      free( cp[i] );
      cp[i] = NULL;
    }
    else  /* current name is unique */
    {
      if ( i > numUnique )
      {
        cp[numUnique] = cp[i];
        cp[i] = NULL;
      }
      numUnique++;
    }
  }

  /* reset number of names in array */
  *numNames = numUnique;
}

/* extract file paths from REST URIs for files in resource of reconstruction */
static XnatRestStatus extractFilePathFromRestUri( int *numFilenames, char ***filenames )
{
  static int numFirstElems = 2;
  static const char *firstElems[] =
  {
    "/data",
    "/REST"
  };
  static int numParts = 6;
  static const char *parts[] =
  {
    "/projects/",
    "/subjects/",
    "/experiments/",
    "/reconstructions/",
    "/resources/",
    "/files/"
  };
  char *cp, *pp;
  int i, j;

  for ( i = 0 ; i < *numFilenames ; i++ )
  {
    /* find file path in REST URI */
    cp = (*filenames)[i];
    if ( *cp != '\0' )
    {
      j = 0;
      while ( j < numFirstElems )
      {
        pp = strstr( cp, firstElems[j] );
        if ( pp != NULL )
        {
          cp = pp + strlen( firstElems[j] );
          break;
        }
        j++;
      }
      if ( j >= numFirstElems )
      {
        freeXnatRestArray( *numFilenames, *filenames );
        *numFilenames = 0;
        *filenames = NULL;
        return XNATREST_MISSINGNODE;
      }
    }

    j = 0;
    while ( ( *cp != '\0' ) && ( j < numParts ) )
    {
      pp = strstr( cp, parts[j] );
      if ( pp == NULL )
      {
        freeXnatRestArray( *numFilenames, *filenames );
        *numFilenames = 0;
        *filenames = NULL;
        return XNATREST_MISSINGNODE;
      }
      cp = pp + strlen( parts[j] );
      j++;
    }

    if ( *cp == '\0' )
    {
      freeXnatRestArray( *numFilenames, *filenames );
      *numFilenames = 0;
      *filenames = NULL;
      return XNATREST_MISSINGPATH;
    }

    /* replace REST URI with file path */
    pp = (*filenames)[i];
    (*filenames)[i] = (char *) malloc( ( strlen( cp ) + 1 ) * sizeof(char) );
    if ( (*filenames)[i] == NULL )
    {
      freeXnatRestArray( *numFilenames, *filenames );
      *numFilenames = 0;
      *filenames = NULL;
      return XNATREST_NOMEM;
    }
    strcpy( (*filenames)[i], cp );
    free( pp );
  }

  return XNATREST_OK;
}

/* initialize XnatRest -- call this function ONCE at beginning of program */
/* assumption: single-threaded program */
XnatRestStatus initXnatRest( void )
{
  CURLcode ret;

#ifdef _WIN32

  WSADATA wsaData;
  int err;

  /* initialize Windows socket interface */
  err = WSAStartup( MAKEWORD( 2, 2), &wsaData );
  if ( err != 0 )
  {
    sprintf( wsaStartupErrorMsg, "WSAStartup failed with error %d", err );
    return XNATREST_WSASTARTUPERR;
  }

#endif

  /* set program environment for libcurl */
  /* call ONCE for entire program */
  ret = curl_global_init( CURL_GLOBAL_ALL );
  if ( ret != CURLE_OK )
  {
    /* printf( "Cannot initialize program environment\n" ); */
#ifdef _WIN32
    WSACleanup();
#endif
    return XNATREST_NOGLOBAL;
  }

  /* get libcurl easy interface handle */
  /* (call ONCE for each thread) */
  curl = curl_easy_init();
  if ( curl == NULL )
  {
    /* printf( "Cannot get curl easy interface handle\n" ); */
    curl_global_cleanup();
#ifdef _WIN32
    WSACleanup();
#endif
    return XNATREST_NOHANDLE;
  }

  /* get libcurl multi interface handle */
  /* (for asynchronous file transfers) */
  curlMulti = curl_multi_init();
  if ( curlMulti == NULL )
  {
    /* printf( "Cannot get curl multi interface handle\n" ); */
    curl_easy_cleanup( curl );
    curl_global_cleanup();
#ifdef _WIN32
    WSACleanup();
#endif
    return XNATREST_NOMULTIHANDLE;
  }

  /* set flag for asynchronous file transfer NOT in progress */
  asynXferInProgress = XRBOOL_FALSE;

  /* disable SSL peer verification via certificates */
  curl_easy_setopt( curl, CURLOPT_SSL_VERIFYPEER, 0L );

  return XNATREST_OK;
}

/* clean up XnatRest -- call this function ONCE at end of program */
/* assumption: single-threaded program */
void cleanupXnatRest( void )
{
  /* free memory for URL address of XNAT web site, if previously allocated */
  if ( xnatUrl != NULL )
  {
    free( xnatUrl );
    xnatUrl = NULL;
  }

  /* free memory for user and password, if previously allocated */
  if ( userpwd != NULL )
  {
    free( userpwd );
    userpwd = NULL;
  }

  /* free memory for XNAT REST URI, if previously allocated */
  if ( restUri != NULL )
  {
    free( restUri );
    restUri = NULL;
  }

  /* close connections and free memory associated with curl multi interface handle */
  curl_multi_cleanup( curlMulti );

  /* close connections and free memory associated with curl easy interface handle */
  curl_easy_cleanup( curl );

  /* release resources acquired by curl_global_init */
  curl_global_cleanup();

#ifdef _WIN32

  /* release resources for Windows socket interface */
  WSACleanup();

#endif
}

/* set user and password for XnatRest */
XnatRestStatus setXnatRestUser( const char *user, const char *password )
{
  CURLcode ret;
  size_t userlen;
  char *cp;

  /* free memory if previously allocated */
  if ( userpwd != NULL )
  {
    free( userpwd );
    userpwd = NULL;
  }

  /* allocate memory for user:password phrase */
  userlen = strlen( user );
  userpwd = (char *) malloc( ( userlen + strlen( password ) + 2 ) * sizeof(char) );
  if ( userpwd == NULL )
  {
    /* printf( "Cannot allocate memory for user and password" ); */
    return XNATREST_NOMEM;
  }

  /* create user:password phrase */
  strcpy( userpwd, user );
  cp = userpwd + userlen;
  *cp = ':';
  strcpy( cp + 1, password );

  /* set user and password for libcurl */
  ret = curl_easy_setopt( curl, CURLOPT_USERPWD, userpwd );
  if ( ret != CURLE_OK )
  {
    /* printf( "Cannot set user ID and password\n" ); */
    return XNATREST_USERPWDERR;
  }

  return XNATREST_OK;
}

/* set URL address of XNAT web site */
XnatRestStatus setXnatRestUrl( const char *url )
{
  /* free memory if previously allocated */
  if ( xnatUrl != NULL )
  {
    free( xnatUrl );
    xnatUrl = NULL;
  }

  /* allocate memory for XNAT URL */
  xnatUrl = (char *) malloc( ( strlen( url ) + 1 ) * sizeof(char) );
  if ( xnatUrl == NULL )
  {
    /* printf( "Cannot allocate memory for XNAT URL" ); */
    return XNATREST_NOMEM;
  }

  /* set URL address of XNAT web site */
  strcpy( xnatUrl, url );

  /*
  printf( "xnatUrl: %s\n", xnatUrl );
  */

  return XNATREST_OK;
}

/* set HTTP REST URI */
static XnatRestStatus setHttpRestUri( const char **query )
{
  int restUriLen;
  const char **cp;
  CURLcode ret;

  /* free memory if previously allocated */
  if ( restUri != NULL )
  {
    free( restUri );
    restUri = NULL;
  }

  /* allocate memory for XNAT REST URI */
  restUriLen = strlen( xnatUrl );
  for ( cp = query ; *cp != NULL ; cp++ )
  {
    restUriLen += strlen( *cp );
  }
  restUri = (char *) malloc( ( restUriLen + 1 ) * sizeof(char) );
  if ( restUri == NULL )
  {
    /* printf( "Cannot allocate memory for XNAT REST URI" ); */
    return XNATREST_NOMEM;
  }

  /* generate XNAT REST URI */
  strcpy( restUri, xnatUrl );
  for ( cp = query ; *cp != NULL ; cp++ )
  {
    strcat( restUri, *cp );
  }

  /* printf( "URI: %s\n", restUri ); */

  /* set XNAT REST URI in libcurl */
  ret = curl_easy_setopt( curl, CURLOPT_URL, restUri );
  if ( ret != CURLE_OK )
  {
    /* printf( "Cannot set XNAT REST URI\n" ); */
    return XNATREST_NORESTURI;
  }

  return XNATREST_OK;
}

/* set HTTP GET URI */
static XnatRestStatus setHttpGetUri( const char **query )
{
  CURLcode ret;

  /* set HTTP GET request in libcurl */
  ret = curl_easy_setopt( curl, CURLOPT_HTTPGET, 1L );
  if ( ret != CURLE_OK )
  {
    /* printf( "Cannot set HTTP GET option\n" ); */
    return XNATREST_NOGET;
  }

  /* set XNAT REST GET URI in libcurl */
  return setHttpRestUri( query );
}

/* perform XNAT REST action */
static XnatRestStatus performXnatRestAction( const XnatRestStatus *wr_error )
{
  CURLcode ret;
  long respCode = 0L;

  badCurlReturn = CURLE_OK;

  /* Allow curl to perform the action */
  ret = curl_easy_perform( curl );

  /* printf( "ret = %d (write_error = %d)\n", ret, *wr_error ); */

  /* Emit the page if curl indicates that no errors occurred */
  if ( *wr_error != XNATREST_OK )
  {
    return *wr_error;
  }
  else if ( ret == CURLE_COULDNT_CONNECT )
  {
    return XNATREST_NOCONNECT;
  }
  else if ( ret != CURLE_OK )
  {
    badCurlReturn = ret;
    /* printf( "CURL Error %d: %s\n", ret, curl_easy_strerror( ret ) ); */
    return XNATREST_CURLERROR;
  }

  ret = curl_easy_getinfo( curl, CURLINFO_RESPONSE_CODE, &respCode );
  if ( ret != CURLE_OK )
  {
    badCurlReturn = ret;
    /* printf( "CURL Error %d: %s\n", ret, curl_easy_strerror( ret ) ); */
    return XNATREST_CURLERROR;
  }
  else if ( respCode == 401 )
  {
    /* printf("Error: Response code is %d (User not authorized)\n", respCode ); */
    return XNATREST_NOAUTH;
  }
  else if ( respCode == 404 )
  {
    /* printf("Error: Response code is %d (Invalid query)\n", respCode ); */
    return XNATREST_BADQUERY;
  }
  else if ( respCode >= 300 )
  {
    sprintf( httpRespCodeMsg, "HTTP error code %ld returned", respCode );
    /* printf("Error: Response code is %d\n", respCode ); */
    return XNATREST_HTTPERROR;
  }
  /*
  else
  {
    printf("Response code is %d\n", respCode );
  }
  */

  return XNATREST_OK;
}

/* get array of requested values */
static XnatRestStatus performXnatRestGet( const char **query, const int col, 
                                          int *numValues, char ***colValues )
{
  XnatRestStatus status;

  /* initialize callback function parameters */
  XnatRestStatus wr_error = XNATREST_OK;
  wr_index = 0;

  /* initialize output parameters */
  *numValues = 0;
  *colValues = NULL;

  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_TRUE )
  {
    return XNATREST_ASYNINPROGRESS;
  }

  /* set HTTP GET URI for query */
  status = setHttpGetUri( query );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* Tell curl that we'll receive data to the function writeDataBufCB,
   * and also provide it with a context pointer for our error return.
   */
  curl_easy_setopt( curl, CURLOPT_WRITEDATA, (void *)&wr_error );
  curl_easy_setopt( curl, CURLOPT_WRITEFUNCTION, writeDataBufCB );

  /* perform XNAT REST action */
  status = performXnatRestAction( &wr_error );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* extract column values from csv lines */
  return getColValuesFromCSV( col, numValues, colValues );
}

/* download requested file */
static XnatRestStatus performXnatRestFileGet( const char **query, const char *filename )
{
  FILE *fp;
  XnatRestStatus wr_error = XNATREST_OK;
  XnatRestStatus status;

  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_TRUE )
  {
    return XNATREST_ASYNINPROGRESS;
  }

  /* set HTTP GET URI for query */
  status = setHttpGetUri( query );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* open local file for downloaded data from XNAT */
  errno = 0;
  fp = fopen( filename, "wb" );
  if ( fp == NULL )
  {
    sprintf( ioErrorMsg, "Cannot open file: %s", strerror( errno ) );
    return XNATREST_IOERROR;
  }

  /* set callback function to write downloaded data to opened file */
  curl_easy_setopt( curl, CURLOPT_WRITEFUNCTION, writeDataFileCB );
  curl_easy_setopt( curl, CURLOPT_WRITEDATA, fp );

  /* perform XNAT REST action */
  status = performXnatRestAction( &wr_error );
  if ( status != XNATREST_OK )
  {
    fclose( fp );
    remove( filename );  /* delete output file holding error message */
    return status;
  }

  /* close file */
  fclose( fp );

  return XNATREST_OK;
}

/* initialize asynchronous download of requested file from XNAT */
static XnatRestStatus initXnatRestAsynFileGet( const char **query, const char *filename )
{
  CURLMcode retMulti;
  XnatRestStatus status;

  /* check if another asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_TRUE )
  {
    return XNATREST_ASYNINPROGRESS;
  }

  /* set HTTP GET URI for query */
  status = setHttpGetUri( query );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* open local file for downloaded data from XNAT */
  errno = 0;
  fpMulti = fopen( filename, "wb" );
  if ( fpMulti == NULL )
  {
    sprintf( ioErrorMsg, "Cannot open file: %s", strerror( errno ) );
    return XNATREST_IOERROR;
  }

  /* set callback function to write downloaded data to opened file */
  curl_easy_setopt( curl, CURLOPT_WRITEFUNCTION, writeDataFileMultiCB );
  curl_easy_setopt( curl, CURLOPT_WRITEDATA, fpMulti );

  /* add curl easy handle to curl multi stack */
  retMulti = curl_multi_add_handle( curlMulti, curl );
  if ( retMulti > CURLM_OK )
  {
    badCurlMultiReturn = retMulti;
    fclose( fpMulti );
    fpMulti = NULL;
    return XNATREST_CURLMULTIERROR;
  }

  /* set flag for asynchronous file transfer in progress */
  asynXferInProgress = XRBOOL_TRUE;
  firstXferCallMade = XRBOOL_FALSE;

  return XNATREST_OK;
}

/* transfer data between XNAT and local file */
XnatRestStatus moveXnatRestAsynData( unsigned long *numBytes, XnatRestAsynStatus *finished )
{
  int stillRunning;
  CURLMsg *msgMulti;
  int msgsLeft;
  long respCode = 0L;
  CURLcode ret;
  CURLMcode retMulti;
  XnatRestStatus status = XNATREST_OK;

  /* initialize outputs */
  *numBytes = 0;
  *finished = XNATRESTASYN_NOTDONE;

  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_FALSE )
  {
    *finished = XNATRESTASYN_DONE;
    return XNATREST_NOASYNINPROGRESS;
  }

  if ( firstXferCallMade == XRBOOL_TRUE )
  {
    /* wait for more data available for transfer */
    status = waitForMoreAsynData();
    if ( status != XNATREST_OK )
    {
      *finished = XNATRESTASYN_DONE;
      asynXferInProgress = XRBOOL_FALSE;
      curl_multi_remove_handle( curlMulti, curl );
      fclose( fpMulti );
      fpMulti = NULL;
      return status;
    }
  }
  else  /* skip first wait for more data */
  {
    firstXferCallMade = XRBOOL_TRUE;
  }

  /* asynchronous data transfer */
  stillRunning = 1;
  moreBytes = 0;
  do
  {
    retMulti = curl_multi_perform( curlMulti, &stillRunning );
    if ( retMulti > CURLM_OK )
    {
      *finished = XNATRESTASYN_DONE;
      asynXferInProgress = XRBOOL_FALSE;
      curl_multi_remove_handle( curlMulti, curl );
      fclose( fpMulti );
      fpMulti = NULL;
      badCurlMultiReturn = retMulti;
      return XNATREST_CURLMULTIERROR;
    }
  } while ( stillRunning && ( retMulti == CURLM_CALL_MULTI_PERFORM ) );
  *numBytes = moreBytes;

  /* get final status if last of data has been transferred */
  if ( !stillRunning )
  {
    *finished = XNATRESTASYN_DONE;
    asynXferInProgress = XRBOOL_FALSE;

    /* get last error status */
    ret = CURLE_OK;
    while ( ( msgMulti = curl_multi_info_read( curlMulti, &msgsLeft ) ) )
    {
      if ( msgMulti->msg == CURLMSG_DONE )
      {
        ret = msgMulti->data.result;
      }
    }

    curl_multi_remove_handle( curlMulti, curl );
    fclose( fpMulti );
    fpMulti = NULL;

    if ( ret == CURLE_COULDNT_CONNECT )
    {
      status = XNATREST_NOCONNECT;
    }
    else if ( ret != CURLE_OK )
    {
      badCurlReturn = ret;
      status = XNATREST_CURLERROR;
    }
    else
    {
      ret = curl_easy_getinfo( curl, CURLINFO_RESPONSE_CODE, &respCode );
      if ( ret != CURLE_OK )
      {
        badCurlReturn = ret;
        status = XNATREST_CURLERROR;
      }
      else if ( respCode == 401 )
      {
        status = XNATREST_NOAUTH;
      }
      else if ( respCode == 404 )
      {
        status = XNATREST_BADQUERY;
      }
      else if ( respCode >= 300 )
      {
        sprintf( httpRespCodeMsg, "HTTP error code %ld returned", respCode );
        status = XNATREST_HTTPERROR;
      }
    }
  }

  return status;
}

static XnatRestStatus waitForMoreAsynData( void )
{
  long curlTimeout = -1;  /* units of milliseconds */
  struct timeval timeout;
  fd_set fdRead;
  fd_set fdWrite;
  fd_set fdExcep;
  int maxfd = -1;
  CURLMcode retMulti;

  /* get suggested timeout before transferring more bytes */
  retMulti = curl_multi_timeout( curlMulti, &curlTimeout );
  if ( retMulti > CURLM_OK )
  {
    badCurlMultiReturn = retMulti;
    return XNATREST_CURLMULTIERROR;
  }

  if ( curlTimeout != 0 )
  {
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    if ( curlTimeout >= 0 )
    {
      timeout.tv_sec = curlTimeout / 1000;
      if ( timeout.tv_sec > 1 )
      {
        timeout.tv_sec = 1;
      }
      else
      {
        timeout.tv_usec = ( curlTimeout % 1000 ) * 1000;
      }
    }

    FD_ZERO(&fdRead);
    FD_ZERO(&fdWrite);
    FD_ZERO(&fdExcep);

    /* get file descriptor for data transfer */
    retMulti = curl_multi_fdset( curlMulti, &fdRead, &fdWrite, &fdExcep, &maxfd );
    if ( retMulti > CURLM_OK )
    {
      badCurlMultiReturn = retMulti;
      return XNATREST_CURLMULTIERROR;
    }

    /* wait for data to be transferred or timeout */
    select( ( maxfd + 1 ), &fdRead, &fdWrite, &fdExcep, &timeout );
  }

  return XNATREST_OK;
}

/* cancel transfer of data between XNAT and local file */
XnatRestStatus cancelXnatRestAsynTransfer( void )
{
  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_FALSE )
  {
    return XNATREST_NOASYNINPROGRESS;
  }

  curl_multi_remove_handle( curlMulti, curl );
  fclose( fpMulti );
  fpMulti = NULL;

  /* set flag for asynchronous file transfer to NOT in progress */
  asynXferInProgress = XRBOOL_FALSE;

  return XNATREST_OK;
}

static XnatRestStatus setHttpPutUri( const char **query )
{
  CURLcode ret;

  /* set HTTP PUT request in libcurl */
  ret = curl_easy_setopt( curl, CURLOPT_UPLOAD, 1L );
  if ( ret != CURLE_OK )
  {
    /* printf( "Cannot set HTTP PUT option\n" ); */
    return XNATREST_NOPUT;
  }

  ret = curl_easy_setopt( curl, CURLOPT_PUT, 1L );
  if ( ret != CURLE_OK )
  {
    /* printf( "Cannot set HTTP PUT option\n" ); */
    return XNATREST_NOPUT;
  }

  /* set XNAT REST PUT URI in libcurl */
  return setHttpRestUri( query );
}

/* create new value in XNAT project hierachy */
static XnatRestStatus performXnatRestPut( const char **query )
{
  struct curl_slist *slist = NULL;
  XnatRestStatus status;

  /* initialize callback error parameter */
  XnatRestStatus wr_error = XNATREST_OK;

  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_TRUE )
  {
    return XNATREST_ASYNINPROGRESS;
  }

  /* set HTTP PUT URI for query */
  status = setHttpPutUri( query );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* tell curl there is no data to upload */
  curl_easy_setopt( curl, CURLOPT_READDATA, NULL );
  curl_easy_setopt( curl, CURLOPT_READFUNCTION, readNoDataCB );
  curl_easy_setopt( curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t) 0 );

  /* set HTTP header for no data upload (override "Expect: 100-continue") */
  slist = curl_slist_append( slist, "Expect:" );
  if ( slist == NULL )
  {
    return XNATREST_NOMEM;
  }
  curl_easy_setopt( curl, CURLOPT_HTTPHEADER, slist );

  /* perform XNAT REST action */
  status = performXnatRestAction( &wr_error );

  /* reset HTTP headers */
  curl_easy_setopt( curl, CURLOPT_HTTPHEADER, NULL );
  curl_slist_free_all( slist );

  return status;
}

/* upload file to XNAT */
static XnatRestStatus performXnatRestFilePut( const char **query, const char *filename )
{
#ifdef _WIN32
  struct _stat fileInfo;
#else
  struct stat fileInfo;
#endif
  FILE *fp;
  XnatRestStatus wr_error = XNATREST_OK;
  XnatRestStatus status;

  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_TRUE )
  {
    return XNATREST_ASYNINPROGRESS;
  }

  /* set HTTP PUT URI for query */
  status = setHttpPutUri( query );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* determine size of input file */
#ifdef _WIN32
  _stat( filename, &fileInfo );
#else
  stat( filename, &fileInfo );
#endif

  /* open local file for reading data to be uploaded to XNAT */
  errno = 0;
  fp = fopen( filename, "rb" );
  if ( fp == NULL )
  {
    sprintf( ioErrorMsg, "Cannot open file: %s", strerror( errno ) );
    return XNATREST_IOERROR;
  }

  /* set callback function to read data for upload from opened file */
  curl_easy_setopt( curl, CURLOPT_READDATA, fp );
  curl_easy_setopt( curl, CURLOPT_READFUNCTION, readDataFileCB );
  curl_easy_setopt( curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t) fileInfo.st_size );

  /* perform XNAT REST action */
  status = performXnatRestAction( &wr_error );

  /* close local file */
  fclose( fp );

  return status;
}

/* initialize asynchronous upload of file to XNAT */
static XnatRestStatus initXnatRestAsynFilePut( const char **query, const char *filename )
{
#ifdef _WIN32
  struct _stat fileInfo;
#else
  struct stat fileInfo;
#endif
  CURLMcode retMulti;
  XnatRestStatus status;

  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_TRUE )
  {
    return XNATREST_ASYNINPROGRESS;
  }

  /* set HTTP PUT URI for query */
  status = setHttpPutUri( query );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* determine size of input file */
#ifdef _WIN32
  _stat( filename, &fileInfo );
#else
  stat( filename, &fileInfo );
#endif

  /* open local input file for reading data to be uploaded to XNAT */
  errno = 0;
  fpMulti = fopen( filename, "rb" );
  if ( fpMulti == NULL )
  {
    sprintf( ioErrorMsg, "Cannot open file: %s", strerror( errno ) );
    return XNATREST_IOERROR;
  }

  /* set callback function to read data from opened file for upload */
  curl_easy_setopt( curl, CURLOPT_READDATA, fpMulti );
  curl_easy_setopt( curl, CURLOPT_READFUNCTION, readDataFileMultiCB );
  curl_easy_setopt( curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t) fileInfo.st_size );

  /* add curl easy handle to curl multi stack */
  retMulti = curl_multi_add_handle( curlMulti, curl );
  if ( retMulti > CURLM_OK )
  {
    badCurlMultiReturn = retMulti;
    fclose( fpMulti );
    fpMulti = NULL;
    return XNATREST_CURLMULTIERROR;
  }

  /* set flag for asynchronous file transfer in progress */
  asynXferInProgress = XRBOOL_TRUE;
  firstXferCallMade = XRBOOL_FALSE;

  return XNATREST_OK;
}

/* delete value in XNAT project hierachy */
static XnatRestStatus performXnatRestDelete( const char **query )
{
  struct curl_slist *slist = NULL;
  CURLcode ret;
  XnatRestStatus status;

  /* initialize callback error parameter */
  XnatRestStatus wr_error = XNATREST_OK;

  /* check if asynchronous file transfer is in progress */
  if ( asynXferInProgress == XRBOOL_TRUE )
  {
    return XNATREST_ASYNINPROGRESS;
  }

  /* set XNAT REST DELETE URI in libcurl */
  status = setHttpRestUri( query );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* set HTTP DELETE request in libcurl */
  ret = curl_easy_setopt( curl, CURLOPT_CUSTOMREQUEST, "DELETE" );
  if ( ret != CURLE_OK )
  {
    /* printf( "Cannot set HTTP DELETE option\n" ); */
    return XNATREST_NODELETE;
  }

  /* tell curl there is no data to upload */
  curl_easy_setopt( curl, CURLOPT_READDATA, NULL );
  curl_easy_setopt( curl, CURLOPT_READFUNCTION, readNoDataCB );
  curl_easy_setopt( curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t) 0 );

  /* set HTTP header for no data upload (override "Expect: 100-continue") */
  slist = curl_slist_append( slist, "Expect:" );
  if ( slist == NULL )
  {
    curl_easy_setopt( curl, CURLOPT_CUSTOMREQUEST, NULL );
    return XNATREST_NOMEM;
  }
  curl_easy_setopt( curl, CURLOPT_HTTPHEADER, slist );

  /* perform XNAT REST action */
  status = performXnatRestAction( &wr_error );

  /* reset HTTP headers */
  curl_easy_setopt( curl, CURLOPT_HTTPHEADER, NULL );
  curl_slist_free_all( slist );
  curl_easy_setopt( curl, CURLOPT_CUSTOMREQUEST, NULL );

  return status;
}

/* get array of projects */
XnatRestStatus getXnatRestProjects( int *numProjects, char ***projects )
{
  /* csv column containing project names in GET response */
  static const int projectsCol = 1;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *projectsQuery[] = 
  {
    "/REST/projects?format=csv",
    NULL
  };

  /* get array of projects from XNAT REST query */
  return performXnatRestGet( projectsQuery, projectsCol, numProjects, projects );
}

/* get array of subjects for project */
XnatRestStatus getXnatRestSubjects( const char *project, int *numSubjects, char ***subjects )
{
  /* csv column containing subject names in GET response */
  static const int subjectsCol = 3;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *subjectsQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects?format=csv",
    NULL
  };

  /* put project name in subjects query */
  subjectsQuery[1] = project;

  /* get array of subjects for project from XNAT REST query */
  return performXnatRestGet( subjectsQuery, subjectsCol, numSubjects, subjects );
}

/* get array of experiments for subject */
XnatRestStatus getXnatRestExperiments( const char *project, const char *subject, 
                                       int *numExperiments, char ***experiments )
{
  /* csv column containing experiment names in GET response */
  static const int experimentsCol = 6;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *experimentsQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments?format=csv",
    NULL
  };

  /* put project name and subject name in experiments query */
  experimentsQuery[1] = project;
  experimentsQuery[3] = subject;

  /* get array of experiments for subject from XNAT REST query */
  return performXnatRestGet( experimentsQuery, experimentsCol, numExperiments, experiments );
}

/* get array of scans for experiment */
XnatRestStatus getXnatRestScans( const char *project, const char *subject, 
                                 const char *experiment, int *numScans, char ***scans )
{
  /* csv column containing scan numbers in GET response */
  static const int scansCol = 2;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *scansQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans?format=csv",
    NULL
  };

  /* put names for project, subject and experiment in scans query */
  scansQuery[1] = project;
  scansQuery[3] = subject;
  scansQuery[5] = experiment;

  /* get array of scans for experiment from XNAT REST query */
  return performXnatRestGet( scansQuery, scansCol, numScans, scans );
}

/* get array of resources in scan */
XnatRestStatus getXnatRestScanResources( const char *project, const char *subject, 
                                         const char *experiment, const char *scan,
                                         int *numResources, char ***resources )
{
  /* csv column containing resource names in GET response */
  static const int resourcesCol = 2;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *resourcesQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/resources?format=csv",
    NULL
  };

  /* put names for project, subject, experiment and scan in resources query */
  resourcesQuery[1] = project;
  resourcesQuery[3] = subject;
  resourcesQuery[5] = experiment;
  resourcesQuery[7] = scan;

  /* get array of resources in scan from XNAT REST query */
  return performXnatRestGet( resourcesQuery, resourcesCol, numResources, resources );
}

/* get array of filenames for resource in scan */
XnatRestStatus getXnatRestScanRsrcFilenames( const char *project, const char *subject, 
                                             const char *experiment, const char *scan,
                                             const char *resource, int *numFilenames, 
                                             char ***filenames )
{
  /* csv column containing filenames in GET response */
  static const int filenamesCol = 1;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *filenamesQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/resources/",
    NULL,
    "/files?format=csv",
    NULL
  };

  /* put names for project, subject, experiment, scan and resource in filenames query */
  filenamesQuery[1] = project;
  filenamesQuery[3] = subject;
  filenamesQuery[5] = experiment;
  filenamesQuery[7] = scan;
  filenamesQuery[9] = resource;

  /* get array of filenames for resource in scan from XNAT REST query */
  return performXnatRestGet( filenamesQuery, filenamesCol, numFilenames, filenames );
}

/* get names of resources for all scans in experiment for subject in project */
XnatRestStatus getXnatRestExprScanResources( const char *project, const char *subject, 
                                             const char *experiment, 
                                             int *numResources, char ***resources )
{
  XnatRestStatus status;

  /* csv column containing resource names in GET response */
  static const int resourcesCol = 2;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *resourcesQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/ALL/resources?format=csv",
    NULL
  };

  /* put names for project, subject, and experiment in resources query */
  resourcesQuery[1] = project;
  resourcesQuery[3] = subject;
  resourcesQuery[5] = experiment;

  /* get array of resources for all scans in experiment from XNAT REST query */
  status = performXnatRestGet( resourcesQuery, resourcesCol, numResources, resources );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* remove duplicate resource names */
  removeDuplicateNames( numResources, resources );

  return status;
}

/* download one file (zipped) from resource in scan */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestScanRsrcFile( const char *project, const char *subject,
                                        const char *experiment, const char *scan,
                                        const char *resource, const char *filename, 
                                        const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/resources/",
    NULL,
    "/files/",
    NULL,
    "?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, scan, resource and file in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = scan;
  downloadQuery[9] = resource;
  downloadQuery[11] = filename;

  /* download zipped file from resource in scan via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputFilename );
}

/* download one file (zipped) from resource in scan */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynScanRsrcFile( const char *project, const char *subject,
                                            const char *experiment, const char *scan,
                                            const char *resource, const char *filename, 
                                            const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/resources/",
    NULL,
    "/files/",
    NULL,
    "?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, scan, resource and file in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = scan;
  downloadQuery[9] = resource;
  downloadQuery[11] = filename;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from resource in scan for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllFilesInScanRsrc( const char *project, const char *subject,
                                              const char *experiment, const char *scan,
                                              const char *resource, const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, scan, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = scan;
  downloadQuery[9] = resource;

  /* download all files (zipped) from resource in scan via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from resource in scan for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllFilesInScanRsrc( const char *project, const char *subject,
                                                  const char *experiment, const char *scan,
                                                  const char *resource, const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, scan, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = scan;
  downloadQuery[9] = resource;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from scan for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllFilesInScan( const char *project, const char *subject,
                                          const char *experiment, const char *scan,
                                          const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and scan in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = scan;

  /* download all files (zipped) from scan via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from scan for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllFilesInScan( const char *project, const char *subject,
                                              const char *experiment, const char *scan,
                                              const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and scan in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = scan;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from resource in all scans in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllScanFilesInExprRsrc( const char *project, const char *subject,
                                                  const char *experiment, const char *resource,
                                                  const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/ALL/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = resource;

  /* download all files (zipped) from resource in experiment via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from resource in all scans in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllScanFilesInExprRsrc( const char *project, const char *subject,
                                                      const char *experiment, const char *resource,
                                                      const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/ALL/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = resource;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from all scans in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllScanFilesInExperiment( const char *project, const char *subject,
                                                    const char *experiment, const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/ALL/files?format=zip",
    NULL
  };

  /* put names for project, subject, and experiment in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;

  /* download all files (zipped) from experiment via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from all scans in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllScanFilesInExperiment( const char *project, const char *subject,
                                                        const char *experiment, const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/scans/ALL/files?format=zip",
    NULL
  };

  /* put names for project, subject, and experiment in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputFilename );
}

/* get names of reconstructions for experiment for subject in project */
XnatRestStatus getXnatRestReconstructions( const char *project, const char *subject, 
                                           const char *experiment, 
                                           int *numReconstructions, char ***reconstructions )
{
  /* csv column containing reconstruction names in GET response */
  static const int reconsCol = 2;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *reconsQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions?format=csv",
    NULL
  };

  /* put names for project, subject and experiment in reconstructions query */
  reconsQuery[1] = project;
  reconsQuery[3] = subject;
  reconsQuery[5] = experiment;

  /* get array of reconstruction names for experiment from XNAT REST query */
  return performXnatRestGet( reconsQuery, reconsCol, numReconstructions, reconstructions );
}

/* get names of resources in reconstruction for experiment for subject in project */
XnatRestStatus getXnatRestReconResources( const char *project, const char *subject, 
                                          const char *experiment, const char *reconstruction,
                                          int *numResources, char ***resources )
{
  /* csv column containing resource names in GET response */
  static const int resourcesCol = 2;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *resourcesQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources?format=csv",
    NULL
  };

  /* put names for project, subject, experiment and reconstruction in resources query */
  resourcesQuery[1] = project;
  resourcesQuery[3] = subject;
  resourcesQuery[5] = experiment;
  resourcesQuery[7] = reconstruction;

  /* get array of resources in reconstruction from XNAT REST query */
  return performXnatRestGet( resourcesQuery, resourcesCol, numResources, resources );
}

/* get filenames for resource in reconstruction for experiment for subject in project */
XnatRestStatus getXnatRestReconRsrcFilenames( const char *project, const char *subject, 
                                              const char *experiment, const char *reconstruction,
                                              const char *resource, int *numFilenames, 
                                              char ***filenames )
{
  XnatRestStatus status;

  /* csv column containing filenames in GET response */
  static const int filenamesCol = 3;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *filenamesQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources/",
    NULL,
    "/files?format=csv",
    NULL
  };

  /* put names for project, subject, experiment, reconstruction and resource in filenames query */
  filenamesQuery[1] = project;
  filenamesQuery[3] = subject;
  filenamesQuery[5] = experiment;
  filenamesQuery[7] = reconstruction;
  filenamesQuery[9] = resource;

  /* get array of filenames for resource in reconstruction from XNAT REST query */
  status = performXnatRestGet( filenamesQuery, filenamesCol, numFilenames, filenames );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* extract file paths from REST URIs */
  return extractFilePathFromRestUri( numFilenames, filenames );
}

/* get names of resources for all reconstructions in experiment for subject in project */
XnatRestStatus getXnatRestExprReconResources( const char *project, const char *subject, 
                                              const char *experiment, 
                                              int *numResources, char ***resources )
{
  XnatRestStatus status;

  /* csv column containing resource names in GET response */
  static const int resourcesCol = 2;

  /* acceptable formats are csv, xml, json, html */
  /* XnatRest assumes csv format is available    */
  const char *resourcesQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/ALL/ALL/resources?format=csv",
    NULL
  };

  /* put names for project, subject, and experiment in resources query */
  resourcesQuery[1] = project;
  resourcesQuery[3] = subject;
  resourcesQuery[5] = experiment;

  /* get array of resources for all reconstructions in experiment from XNAT REST query */
  status = performXnatRestGet( resourcesQuery, resourcesCol, numResources, resources );
  if ( status != XNATREST_OK )
  {
    return status;
  }

  /* remove duplicate resource names */
  removeDuplicateNames( numResources, resources );

  return status;
}

/* download one file (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestReconRsrcFile( const char *project, const char *subject,
                                         const char *experiment, const char *reconstruction,
                                         const char *resource, const char *filename, 
                                         const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources/",
    NULL,
    "/files/",
    NULL,
    "?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, reconstruction, resource and file in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = reconstruction;
  downloadQuery[9] = resource;
  downloadQuery[11] = filename;

  /* download zipped file from resource in reconstruction via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputFilename );
}

/* download one file (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynReconRsrcFile( const char *project, const char *subject,
                                             const char *experiment, const char *reconstruction,
                                             const char *resource, const char *filename, 
                                             const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources/",
    NULL,
    "/files/",
    NULL,
    "?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, reconstruction, resource and file in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = reconstruction;
  downloadQuery[9] = resource;
  downloadQuery[11] = filename;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllFilesInReconRsrc( const char *project, const char *subject,
                                               const char *experiment, const char *reconstruction,
                                               const char *resource, const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, reconstruction, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = reconstruction;
  downloadQuery[9] = resource;

  /* download all files (zipped) from resource in reconstruction via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllFilesInReconRsrc( const char *project, const char *subject,
                                                   const char *experiment, const char *reconstruction,
                                                   const char *resource, const char *outputFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, reconstruction, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = reconstruction;
  downloadQuery[9] = resource;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputFilename );
}

/* download all files (zipped) from reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllFilesInReconstruction( const char *project, const char *subject,
                                                    const char *experiment, const char *reconstruction,
                                                    const char *outputZipFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and reconstruction in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = reconstruction;

  /* download zipped file from reconstruction via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputZipFilename );
}

/* download all files (zipped) from reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllFilesInReconstruction( const char *project, const char *subject,
                                                        const char *experiment, const char *reconstruction,
                                                        const char *outputZipFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and reconstruction in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = reconstruction;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputZipFilename );
}

/* download all files (zipped) from resource for all reconstructions in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllReconFilesInExprRsrc( const char *project, const char *subject,
                                                   const char *experiment, const char *resource,
                                                   const char *outputZipFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/ALL/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = resource;

  /* download zipped file from resource for all reconstructions via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputZipFilename );
}

/* download all files (zipped) from resource for all reconstructions in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllReconFilesInExprRsrc( const char *project, const char *subject,
                                                       const char *experiment, const char *resource,
                                                       const char *outputZipFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/ALL/resources/",
    NULL,
    "/files?format=zip",
    NULL
  };

  /* put names for project, subject, experiment, and resource in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;
  downloadQuery[7] = resource;

  /* initialize download of all files (zipped) from resource in scan via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputZipFilename );
}

/* download all files (zipped) from all reconstructions in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
XnatRestStatus getXnatRestAllReconFilesInExperiment( const char *project, const char *subject,
                                                     const char *experiment, 
                                                     const char *outputZipFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/ALL/files?format=zip",
    NULL
  };

  /* put names for project, subject, and experiment in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;

  /* download zipped file from all reconstructions in experiment via XNAT REST query */
  return performXnatRestFileGet( downloadQuery, outputZipFilename );
}

/* download all files (zipped) from all reconstructions in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
XnatRestStatus getXnatRestAsynAllReconFilesInExperiment( const char *project, const char *subject,
                                                         const char *experiment, 
                                                         const char *outputZipFilename )
{
  /* XnatRest assumes zipped file format is available */
  const char *downloadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/ALL/files?format=zip",
    NULL
  };

  /* put names for project, subject, and experiment in query */
  downloadQuery[1] = project;
  downloadQuery[3] = subject;
  downloadQuery[5] = experiment;

  /* initialize download of files (zipped) from all reconstructions in experiment via XNAT REST query */
  return initXnatRestAsynFileGet( downloadQuery, outputZipFilename );
}

/* create reconstruction in experiment for subject in project */
XnatRestStatus putXnatRestReconstruction( const char *project, const char *subject, 
                                          const char *experiment, const char *reconstruction )
{
  const char *reconsQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    NULL
  };

  /* put names for project, subject, experiment and reconstruction in query */
  reconsQuery[1] = project;
  reconsQuery[3] = subject;
  reconsQuery[5] = experiment;
  reconsQuery[7] = reconstruction;

  /* create reconstruction in experiment via XNAT REST query */
  return performXnatRestPut( reconsQuery );
}

/* create resource for reconstruction in experiment for subject in project */
XnatRestStatus putXnatRestReconResource( const char *project, const char *subject, 
                                         const char *experiment, const char *reconstruction,
                                         const char *resource )
{
  const char *resourceQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/out/resources/",
    NULL,
    NULL
  };

  /* put names for project, subject, experiment, reconstruction and resource in query */
  resourceQuery[1] = project;
  resourceQuery[3] = subject;
  resourceQuery[5] = experiment;
  resourceQuery[7] = reconstruction;
  resourceQuery[9] = resource;

  /* create resource in reconstruction via XNAT REST query */
  return performXnatRestPut( resourceQuery );
}

/* delete reconstruction in experiment for subject in project */
/* NOTE: all resources and files within the reconstruction are also deleted */
XnatRestStatus deleteXnatRestReconstruction( const char *project, const char *subject, 
                                             const char *experiment, const char *reconstruction )
{
  const char *reconsQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "?removeFiles=true",
    NULL
  };

  /* put names for project, subject, experiment and reconstruction in query */
  reconsQuery[1] = project;
  reconsQuery[3] = subject;
  reconsQuery[5] = experiment;
  reconsQuery[7] = reconstruction;

  /* delete reconstruction in experiment via XNAT REST query */
  return performXnatRestDelete( reconsQuery );
}

/* delete resource for reconstruction in experiment for subject in project */
/* NOTE: all files within the resource are also deleted */
XnatRestStatus deleteXnatRestReconResource( const char *project, const char *subject, 
                                            const char *experiment, const char *reconstruction,
                                            const char *resource )
{
  const char *resourceQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources/",
    NULL,
    "?removeFiles=true",
    NULL
  };

  /* put names for project, subject, experiment, reconstruction and resource in query */
  resourceQuery[1] = project;
  resourceQuery[3] = subject;
  resourceQuery[5] = experiment;
  resourceQuery[7] = reconstruction;
  resourceQuery[9] = resource;

  /* delete resource in reconstruction via XNAT REST query */
  return performXnatRestDelete( resourceQuery );
}

/* delete file in resource for reconstruction in experiment for subject in project */
XnatRestStatus deleteXnatRestReconRsrcFile( const char *project, const char *subject, 
                                            const char *experiment, const char *reconstruction,
                                            const char *resource, const char *filename )
{
  const char *resourceQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/ALL/resources/",
    NULL,
    "/files/",
    NULL,
    "?removeFiles=true",
    NULL
  };

  /* put names for project, subject, experiment, reconstruction, resource and file in query */
  resourceQuery[1]  = project;
  resourceQuery[3]  = subject;
  resourceQuery[5]  = experiment;
  resourceQuery[7]  = reconstruction;
  resourceQuery[9]  = resource;
  resourceQuery[11] = filename;

  /* delete file in resource for reconstruction via XNAT REST query */
  return performXnatRestDelete( resourceQuery );
}

/* upload files (zipped) to resource in reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file upload is finished */
XnatRestStatus putXnatRestReconRsrcFiles( const char *project, const char *subject,
                                          const char *experiment, const char *reconstruction,
                                          const char *resource, const char *inputFilename )
{
  const char *filename;
  const char *cpSlash = NULL;
  const char *cpBackslash = NULL;

  /* XnatRest assumes zipped file format is available */
  const char *uploadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/out/resources/",
    NULL,
    "/files/",
    NULL,
    "?inbody=true&extract=true",
    NULL
  };

  /* extract filename and extension from input file path */
  filename = inputFilename;
  cpSlash = strrchr( inputFilename, '/' );
  cpBackslash = strrchr( inputFilename, '\\' );
  if ( cpSlash != NULL )
  {
    if ( cpBackslash == NULL )
    {
      filename = cpSlash + 1;
    }
    else if ( cpSlash > cpBackslash )
    {
      filename = cpSlash + 1;
    }
    else
    {
      filename = cpBackslash + 1;
    }
  }
  else if ( cpBackslash != NULL )
  {
    filename = cpBackslash + 1;
  }

  /* put names for project, subject, experiment, reconstruction, resource and file in query */
  uploadQuery[1] = project;
  uploadQuery[3] = subject;
  uploadQuery[5] = experiment;
  uploadQuery[7] = reconstruction;
  uploadQuery[9] = resource;
  uploadQuery[11] = filename;

  /* upload zipped file to resource in reconstruction via XNAT REST query */
  return performXnatRestFilePut( uploadQuery, inputFilename );
}

/* upload files (zipped) to resource in reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) upload of ZIP file and returns */
XnatRestStatus putXnatRestAsynReconRsrcFiles( const char *project, const char *subject,
                                              const char *experiment, const char *reconstruction,
                                              const char *resource, const char *inputFilename )
{
  const char *filename;
  const char *cpSlash = NULL;
  const char *cpBackslash = NULL;

  /* XnatRest assumes zipped file format is available */
  const char *uploadQuery[] =
  {
    "/REST/projects/",
    NULL,
    "/subjects/",
    NULL,
    "/experiments/",
    NULL,
    "/reconstructions/",
    NULL,
    "/out/resources/",
    NULL,
    "/files/",
    NULL,
    "?inbody=true&extract=true",
    NULL
  };

  /* extract filename and extension from input file path */
  filename = inputFilename;
  cpSlash = strrchr( inputFilename, '/' );
  cpBackslash = strrchr( inputFilename, '\\' );
  if ( cpSlash != NULL )
  {
    if ( cpBackslash == NULL )
    {
      filename = cpSlash + 1;
    }
    else if ( cpSlash > cpBackslash )
    {
      filename = cpSlash + 1;
    }
    else
    {
      filename = cpBackslash + 1;
    }
  }
  else if ( cpBackslash != NULL )
  {
    filename = cpBackslash + 1;
  }

  /* put names for project, subject, experiment, reconstruction, resource and file in query */
  uploadQuery[1] = project;
  uploadQuery[3] = subject;
  uploadQuery[5] = experiment;
  uploadQuery[7] = reconstruction;
  uploadQuery[9] = resource;
  uploadQuery[11] = filename;

  /* initialize upload of zipped file to resource in reconstruction via XNAT REST query */
  return initXnatRestAsynFilePut( uploadQuery, inputFilename );
}

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

/* unzip downloaded files without directories */
XnatRestStatus unzipXnatRestFileNoDirs( const char *zipFilename, const char *outputDirectory )
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
  return miniunzXnatRestFileNoDirs( zipFilename, outputDir );
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

/* free dynamically allocated memory for string array */
void freeXnatRestArray( const int numStrings, char **strings )
{
  int i;

  if ( strings != NULL )
  {
    /* deallocate memory for each string */
    for ( i = 0 ; i < numStrings ; i++ )
    {
      if ( strings[i] != NULL )
      {
        free( strings[i] );
      }
    }
  
    /* deallocate memory for array */
    free( strings );
  }
}

/* get status message */
const char *getXnatRestStatusMsg( const XnatRestStatus status )
{
  static const char *unknownStatus = "No such XNAT REST status message";
  const char *cp;

  if ( status == XNATREST_CURLERROR )
  {
    cp = curl_easy_strerror( badCurlReturn );
  }
  else if ( status == XNATREST_CURLMULTIERROR )
  {
    cp = curl_multi_strerror( badCurlMultiReturn );
  }
  else if ( status == XNATREST_HTTPERROR )
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

