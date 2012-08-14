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
# include <sys/stat.h>  /* for mkdir function */
#else
#if (defined(_WIN32))
# include <direct.h>
# include <io.h>
#else
# include <unistd.h>
# include <utime.h>
# include <sys/stat.h>  /* for mkdir function */
#define fopen64 fopen
#endif
#endif

#include "XnatRestUnzip.h"

#ifdef _WIN32
#define USEWIN32IOAPI
#include "XnatRestIowin32.h"
#endif

#include "XnatRestMiniUnz.h"

#define WRITEBUFFERSIZE (8192)
#define MKDIR_SUCCESS (0)

#define OPT_EXTRACT_WITH_PATH    (0)
#define OPT_EXTRACT_WITHOUT_PATH (1)

static void change_file_date( const char *filename, uLong dosdate, tm_unz tmu_date );
static int mymkdir( const char *dirname );
static XnatRestStatus makedir( const char *newdir, const char *outputDir );
static XnatRestStatus do_extract_currentfile( unzFile uf, int option_without_path, const char *outputDir );
static XnatRestStatus do_extract( unzFile uf, int option_without_path, const char *outputDir );


/* change_file_date : change the date/time of a file
    filename : the filename of the file where date/time must be modified
    dosdate : the new date at the MSDos format (4 bytes)
    tmu_date : the SAME new date at the tm_unz format */
static void change_file_date( const char *filename, uLong dosdate, tm_unz tmu_date )
{
#ifdef _WIN32
  HANDLE hFile;
  FILETIME ftm, ftLocal, ftCreate, ftLastAcc, ftLastWrite;

  hFile = CreateFileA( filename, GENERIC_READ | GENERIC_WRITE,
                       0, NULL, OPEN_EXISTING, 0, NULL );
  GetFileTime( hFile, &ftCreate, &ftLastAcc, &ftLastWrite );
  DosDateTimeToFileTime( (WORD) (dosdate>>16), (WORD) dosdate, &ftLocal );
  LocalFileTimeToFileTime( &ftLocal, &ftm );
  SetFileTime( hFile, &ftm, &ftLastAcc, &ftm );
  CloseHandle( hFile );
#else
#ifdef unix
  struct utimbuf ut;
  struct tm newdate;
  newdate.tm_sec = tmu_date.tm_sec;
  newdate.tm_min = tmu_date.tm_min;
  newdate.tm_hour = tmu_date.tm_hour;
  newdate.tm_mday = tmu_date.tm_mday;
  newdate.tm_mon = tmu_date.tm_mon;
  if ( tmu_date.tm_year > 1900 )
  {
    newdate.tm_year = tmu_date.tm_year - 1900;
  }
  else
  {
    newdate.tm_year = tmu_date.tm_year;
  }
  newdate.tm_isdst = -1;

  ut.actime = ut.modtime = mktime( &newdate );
  utime( filename, &ut );
#endif
#endif
}

static int mymkdir( const char *dirname )
{
  int status = MKDIR_SUCCESS;

#ifdef _WIN32
  status = _mkdir( dirname );
#else
#ifdef unix
  status = mkdir( dirname, 0775 );
#endif
#endif

  return status;
}

static XnatRestStatus makedir( const char *newdir, const char *outputDir )
{
  int len;
  char *buffer ;
  char *p;
  char hold;

  len = (int) strlen( newdir );
  if ( len <= 0 )
  {
    return XNATREST_OK;
  }

  buffer = (char *) malloc( len + 1 );
  if ( buffer == NULL )
  {
    return XNATREST_UNZ_NOMEM;
  }

  strcpy( buffer, newdir );
  if ( buffer[len-1] == '/' )
  {
    buffer[len-1] = '\0';
  }

  if ( mymkdir( buffer ) == MKDIR_SUCCESS )
  {
    free( buffer );
    return XNATREST_OK;
  }

  p = buffer + strlen( outputDir ) + 1;
  while ( 1 )
  {
    while( *p && ( *p != '\\' ) && ( *p != '/' ) )
    {
      p++;
    }
    hold = *p;
    *p = '\0';

    if ( ( mymkdir( buffer ) != MKDIR_SUCCESS ) && ( errno == ENOENT ) )
    {
      free( buffer );
      return XNATREST_UNZ_MKDIRERR;
    }

    if ( hold == '\0' )
    {
      break;
    }

    *p++ = hold;
  }

  free( buffer );

  return XNATREST_OK;
}

static XnatRestStatus do_extract_currentfile( unzFile uf, int option_without_path, const char *outputDir )
{
  int outputDirLen = 0;
  char filename_inzip[MAXFILENAME];
  char* filename_withoutpath;
  char* p;
  char* q;
  char c;
  FILE *fout = NULL;
  void* buf;
  uInt size_buf;
  unz_file_info64 file_info;
  int err = UNZ_OK;
  XnatRestStatus status = XNATREST_OK;
 
  /* prepend output directory to name of current file to be written */
  if ( *outputDir != '\0' )
  {
    strcpy( filename_inzip, outputDir );
    outputDirLen = strlen( outputDir );
  }

  /* get name of current file to extract from zip file */
  err = unzGetCurrentFileInfo64( uf, &file_info, ( filename_inzip + outputDirLen ), 
                                 ( sizeof(filename_inzip) - outputDirLen ), NULL, 0, NULL, 0 );
  if ( err != UNZ_OK )
  {
    return XNATREST_UNZ_CURINFOERR;
  }

  /* separate filename from directory path */
  p = filename_withoutpath = filename_inzip;
  while ( (*p) != '\0' )
  {
    if ( ( (*p) == '/' ) || ( (*p) == '\\' ) )
    {
      filename_withoutpath = p + 1;
    }
    p++;
  }

  if ( ( *filename_withoutpath ) == '\0' )  /* filename_inzip ends with (back)slash */
  {
    if ( option_without_path == OPT_EXTRACT_WITH_PATH )
    {
      /* create output directory */
      status = makedir( filename_inzip, outputDir );
      if ( status != XNATREST_OK )
      {
        return status;
      }
    }
  }
  else
  {
    /* open current file to be extracted from input zip file */
    err = unzOpenCurrentFile( uf );
    if ( err != UNZ_OK )
    {
      unzCloseCurrentFile( uf );
      return XNATREST_UNZ_CUROPENERR;
    }

    /* reset output file path if directory path from zip file to be ignored */
    if ( option_without_path == OPT_EXTRACT_WITHOUT_PATH )
    {
      if ( filename_inzip + outputDirLen != filename_withoutpath )
      {
        p = filename_inzip + outputDirLen;
        q = filename_withoutpath;
        while ( ( *p++ = *q++ ) )  /* extra () to silence (stupid) compiler warning */
        {
          /* empty loop; see K&R */
        }
      }
    }

    /* open output file to be written */
    fout = fopen64( filename_inzip, "wb" );

    /* some zipfiles don't contain directory alone before file */
    if ( ( fout == NULL ) && ( option_without_path == OPT_EXTRACT_WITH_PATH ) &&
         ( filename_withoutpath != (char *) filename_inzip ) )
    {
      /* fopen failed because directory for output file does not exist */
      /* create directory and try to fopen file again */
      c = *( filename_withoutpath - 1 );
      *( filename_withoutpath - 1 ) = '\0';
      if ( strlen( filename_inzip ) == ( outputDirLen - 1 ) )
      {
        unzCloseCurrentFile( uf );
        return XNATREST_UNZ_BADOUTDIR;
      }
      status = makedir( filename_inzip, outputDir );
      if ( status != XNATREST_OK )
      {
        unzCloseCurrentFile( uf );
        return status;
      }
      *( filename_withoutpath - 1 ) = c;
      fout = fopen64( filename_inzip, "wb" );
    }

    if ( fout == NULL )
    {
      unzCloseCurrentFile( uf );
      return XNATREST_UNZ_OUTOPENERR;
    }

    /* allocate memory for output buffer */
    size_buf = WRITEBUFFERSIZE;
    buf = (void *) malloc( size_buf );
    if ( buf == NULL )
    {
      fclose( fout );
      unzCloseCurrentFile( uf );
      return XNATREST_UNZ_NOMEM;
    }

    /* copy inflated contents of current file from input zip file to output file */
    do
    {
      err = unzReadCurrentFile( uf, buf, size_buf );
      if ( err < 0 )
      {
        status = XNATREST_UNZ_CURREADERR;  
        break;
      }
      else if ( err > 0 )
      {
        if ( fwrite( buf, err, 1, fout ) != 1 )
        {
          status = XNATREST_UNZ_OUTWRITERR;
          break;
        }
      }
    }
    while ( err > 0 );

    /* free memory allocated for output buffer */
    free( buf );

    /* close output file */
    fclose( fout );

    if ( err == 0 )
    {
      change_file_date( filename_inzip, file_info.dosDate, file_info.tmu_date );
    }

    err = unzCloseCurrentFile( uf );
    if ( ( err != UNZ_OK ) && ( status == XNATREST_OK ) )
    {
      status = XNATREST_UNZ_CURCLOSERR;
    }
  }

  return status;
}

static XnatRestStatus do_extract( unzFile uf, int option_without_path, const char *outputDir )
{
  uLong i;
  unz_global_info64 gi;
  int err;
  XnatRestStatus status;

  err = unzGetGlobalInfo64( uf, &gi );
  if ( err != UNZ_OK )
  {
    return XNATREST_UNZ_GBLINFOERR;
  }

  for ( i = 0 ; i < gi.number_entry ; i++ )
  {
    status = do_extract_currentfile( uf, option_without_path, outputDir );
    if ( status != XNATREST_OK )
    {
      return status;
    }

    if ( ( i + 1 ) < gi.number_entry )
    {
      err = unzGoToNextFile( uf );
      if ( err != UNZ_OK )
      {
        return XNATREST_UNZ_NXTFILEERR;
      }
    }
  }

  return XNATREST_OK;
}

XnatRestStatus miniunzXnatRestFile( const char *zipFilename, const char *outputDir )
{
  unzFile uf = NULL;
  XnatRestStatus status;

  /* open zip file */
#ifdef USEWIN32IOAPI
  zlib_filefunc64_def ffunc;
  fill_win32_filefunc64A( &ffunc );
  uf = unzOpen2_64( zipFilename, &ffunc );
#else
  uf = unzOpen64( zipFilename );
#endif
  if ( uf == NULL )
  {
    return XNATREST_UNZ_ZIPOPENERR;
  }

  /* extract files from zip file */
  status = do_extract( uf, OPT_EXTRACT_WITH_PATH, outputDir );

  /* close zip file */
  unzClose( uf );

  return status;
}

XnatRestStatus miniunzXnatRestFileNoDirs( const char *zipFilename, const char *outputDir )
{
  unzFile uf = NULL;
  XnatRestStatus status;

  /* open zip file */
#ifdef USEWIN32IOAPI
  zlib_filefunc64_def ffunc;
  fill_win32_filefunc64A( &ffunc );
  uf = unzOpen2_64( zipFilename, &ffunc );
#else
  uf = unzOpen64( zipFilename );
#endif
  if ( uf == NULL )
  {
    return XNATREST_UNZ_ZIPOPENERR;
  }

  /* extract files from zip file */
  status = do_extract( uf, OPT_EXTRACT_WITHOUT_PATH, outputDir );

  /* close zip file */
  unzClose( uf );

  return status;
}

