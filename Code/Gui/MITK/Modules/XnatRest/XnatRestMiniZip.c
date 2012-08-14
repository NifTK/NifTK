#ifndef _WIN32
        #ifndef __USE_FILE_OFFSET64
                #define __USE_FILE_OFFSET64
        #endif
        #ifndef __USE_LARGEFILE64
                #define __USE_LARGEFILE64
        #endif
        #ifndef _LARGEFILE64_SOURCE
                #define _LARGEFILE64_SOURCE
        #endif
        #ifndef _FILE_OFFSET_BIT
                #define _FILE_OFFSET_BIT 64
        #endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>

#ifdef unix
# include <unistd.h>
# include <utime.h>
# include <sys/types.h>
# include <sys/stat.h>
#else
#if (defined(_WIN32))
# include <direct.h>
# include <io.h>
#else
# include <unistd.h>
# include <utime.h>
# include <sys/types.h>
# include <sys/stat.h>
#define fopen64 fopen
#define fseeko64 fseeko
#define ftello64 ftello
#endif
#endif

#include "XnatRestZip.h"

#ifdef _WIN32
        #define USEWIN32IOAPI
        #include "XnatRestIowin32.h"
#endif

#include "XnatRestMiniZip.h"

#define WRITEBUFFERSIZE (16384)
#define MAXFILENAME (256)

static int isLargeFile( const char *filename );
static XnatRestStatus openNewFileInZipFile( const char *infileAbsPath, const char *filename, zipFile zf );
static XnatRestStatus writeNewFileInZipFile( int size_buf, Bytef *buf, const char *infileAbsPath, zipFile zf );
static XnatRestStatus copyFilesToZipFile( const char *fileDir, int numFilenames, char const * const *filenames, 
                                          zipFile zf );


/**
    f:             name of file to get info on
    tmzip:         return value: access, modification and creation times
    dt:            dostime 
*/
uLong filetime(const char *filename, tm_zip *tmzip, uLong *dt)
{
  int ret = 0;

#ifdef _WIN32
  FILETIME ftLocal;
  HANDLE hFind;
  WIN32_FIND_DATAA ff32;

  hFind = FindFirstFileA(filename, &ff32);
  if (hFind != INVALID_HANDLE_VALUE)
  {
    FileTimeToLocalFileTime(&(ff32.ftLastWriteTime), &ftLocal);
    FileTimeToDosDateTime(&ftLocal, ((LPWORD) dt) + 1,((LPWORD) dt) + 0);
    FindClose(hFind);
    ret = 1;
  }
#elifdef unix
  struct stat s;        /* results of stat() */
  struct tm* filedate;
  time_t tm_t=0;

  if (strcmp(filename, "-") != 0)
  {
    char name[MAXFILENAME + 1];
    int len = strlen(filename);
    if (len > MAXFILENAME)
    {
      len = MAXFILENAME;
    }

    strncpy(name, filename, MAXFILENAME - 1);
    /* strncpy doesnt append the trailing NULL, of the string is too long. */
    name[ MAXFILENAME ] = '\0';

    if (name[len - 1] == '/')
      name[len - 1] = '\0';
    /* not all systems allow stat'ing a file with / appended */
    if (stat(name, &s) == 0)
    {
      tm_t = s.st_mtime;
      ret = 1;
    }
  }
  filedate = localtime(&tm_t);

  tmzip->tm_sec  = filedate->tm_sec;
  tmzip->tm_min  = filedate->tm_min;
  tmzip->tm_hour = filedate->tm_hour;
  tmzip->tm_mday = filedate->tm_mday;
  tmzip->tm_mon  = filedate->tm_mon ;
  tmzip->tm_year = filedate->tm_year;
#endif

  return ret;
}

static int isLargeFile( const char *filename )
{
  int largeFile = 0;
  ZPOS64_T pos = 0;

  /* open input file to be copied into zip file */
  FILE* pFile = fopen64( filename, "rb" );

  if ( pFile != NULL )
  {
    /* int n = fseeko64(pFile, 0, SEEK_END); */
    fseeko64( pFile, 0, SEEK_END );

    pos = ftello64( pFile );

    if ( pos >= 0xffffffff )
    {
      largeFile = 1;
    }

    fclose( pFile );
  }

  return largeFile;
}

static XnatRestStatus openNewFileInZipFile( const char *infileAbsPath, const char *filename, zipFile zf )
{
  zip_fileinfo zi;
  int zip64 = 0;
  const char *filenameInZip;
  int err = ZIP_OK;
  XnatRestStatus status = XNATREST_OK;

  /* set created, modified, and accessed times */
  zi.tmz_date.tm_sec = zi.tmz_date.tm_min = zi.tmz_date.tm_hour =
  zi.tmz_date.tm_mday = zi.tmz_date.tm_mon = zi.tmz_date.tm_year = 0;
  zi.dosDate = 0;
  zi.internal_fa = 0;
  zi.external_fa = 0;
  filetime( infileAbsPath, &zi.tmz_date, &zi.dosDate );

  /* determine if size of file is 4GB or larger */
  zip64 = isLargeFile( infileAbsPath );

  /* path name saved in zip file should not include a leading slash  */
  /* (If it did, windows/xp and dynazip couldn't read the zip file.) */
  filenameInZip = filename;
  while( *filenameInZip == '\\' || *filenameInZip == '/' )
  {
    filenameInZip++;
  }

  /* open new compressed file in zip file */
  err = zipOpenNewFileInZip3_64(
                   zf, filenameInZip, &zi,
                   NULL, 0, NULL, 0, NULL /* comment*/,
                   ( Z_DEFAULT_COMPRESSION != 0 ) ? Z_DEFLATED : 0,
                   Z_DEFAULT_COMPRESSION, 0,
                   -MAX_WBITS, DEF_MEM_LEVEL, Z_DEFAULT_STRATEGY,
                   NULL, 0UL, zip64 );

  if ( err != ZIP_OK )
  {
    status = XNATREST_ZIP_CUROPENERR;
  }

  return status;
}

static XnatRestStatus writeNewFileInZipFile( int size_buf, Bytef *buf, const char *infileAbsPath, zipFile zf )
{
  FILE *fin;
  int size_read;
  int err = ZIP_OK;
  XnatRestStatus status = XNATREST_OK;

  /* open input file to be copied into zip file */
  fin = fopen64( infileAbsPath, "rb" );
  if ( fin == NULL )
  {
    return XNATREST_ZIP_INOPENERR;
  }

  do
  {
    size_read = (int) fread( buf, 1, size_buf, fin );
    if ( size_read < size_buf )
    {
      if ( feof( fin ) == 0 )
      {
        status = XNATREST_ZIP_INREADERR;
      }
    }

    if ( size_read > 0 )
    {
      /* copy buffer from input file into zip file */
      err = zipWriteInFileInZip( zf, buf, size_read );
      if ( err < 0 )
      {
        status = XNATREST_ZIP_CURWRITERR_LTZ;
      }
      else if ( err > 0 )
      {
        status = XNATREST_ZIP_CURWRITERR_GTZ;
      }
    }
  } while ( ( status == XNATREST_OK ) && ( size_read > 0 ) );

  /* close input file after copying into zip file */
  fclose( fin );

  return status;
}
 
static XnatRestStatus copyFilesToZipFile( const char *fileDir, int numFilenames, char const * const *filenames, 
                                          zipFile zf )
{
  Bytef *buf = NULL;
  int size_buf = WRITEBUFFERSIZE;
  int fileDirLen = 0;
  const char *infileAbsPath;
  const char *filenameinzip;
  int i;
  char *cp = NULL;
  int err = ZIP_OK;
  XnatRestStatus status = XNATREST_OK;

  buf = (Bytef *) malloc( size_buf );
  if ( buf == NULL )
  {
    return XNATREST_ZIP_NOMEM;
  }

  fileDirLen = strlen( fileDir );
  for ( i = 0 ; ( i < numFilenames ) && ( status == XNATREST_OK ) ; i++ )
  {
    filenameinzip = filenames[i];

    /* build absolute path for input file */
    if ( fileDirLen > 0 )
    {
      cp = (char *) malloc( ( fileDirLen + strlen( filenameinzip ) + 1 ) * sizeof( char ) );
      if ( cp != NULL )
      {
        strcpy( cp, fileDir );
        strcat( cp, filenameinzip );
        infileAbsPath = cp;
      }
      else
      {
        status = XNATREST_ZIP_NOMEM;
      }
    }
    else
    {
      infileAbsPath = filenameinzip;
    }

    if ( status == XNATREST_OK )
    {
      /* open new compressed file in zip file */
      status = openNewFileInZipFile( infileAbsPath, filenameinzip, zf );

      if ( status == XNATREST_OK )
      {
        /* copy compressed contents from input file into zip file */
        status = writeNewFileInZipFile( size_buf, buf, infileAbsPath, zf );

        if ( status != XNATREST_ZIP_CURWRITERR_LTZ )
        {
          /* close new compressed file in zip file */
          err = zipCloseFileInZip( zf );
          if ( ( err != ZIP_OK ) && ( status == XNATREST_OK ) )
          {
            status = XNATREST_ZIP_CURCLOSERR;
          }
        }
      }
    }

    if ( ( fileDirLen > 0 ) && ( cp != NULL ) )
    {
      free( cp );
    }
  }

  free( buf );
  return status;
}

XnatRestStatus minizipXnatRestFiles( const char *fileDir, int numFilenames, char const * const *filenames, 
                                     const char *zipFilename )
{
  zipFile zf;
  XnatRestStatus status;

  /* open zip file */
#ifdef USEWIN32IOAPI
  zlib_filefunc64_def ffunc;
  fill_win32_filefunc64A( &ffunc );
  zf = zipOpen2_64( zipFilename, 0, NULL, &ffunc );
#else
  zf = zipOpen64( zipFilename, 0 );
#endif
  if ( zf == NULL )
  {
    zipClose( zf, NULL );
    return XNATREST_ZIP_ZIPOPENERR;
  }

  /* create compressed file copies in zip file */
  status = copyFilesToZipFile( fileDir, numFilenames, filenames, zf );

  /* close zip file */
  zipClose( zf, NULL );

  return status;
}
