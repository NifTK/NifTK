#if !defined(XNATREST_H)
#define XNATREST_H

#include "XnatRestExports.h"

#include "XnatRestStatus.h"

#include "NifTKConfigure.h"

#define DICOM_RESOURCE "DICOM"

enum XnatRestAsynStatusCodes
{
  XNATRESTASYN_DONE,    /* set when asynchronous file upload/download is finished */
  XNATRESTASYN_NOTDONE
};

typedef enum XnatRestAsynStatusCodes XnatRestAsynStatus;


/* ======== BASIC FUNCTIONS ======== */

/* initialize XnatRest -- call this function ONCE at beginning of program */
/* assumption: single-threaded program */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus initXnatRest( void );

/* clean up XnatRest -- call this function ONCE at end of program */
/* assumption: single-threaded program */
XnatRest_EXPORT void cleanupXnatRest( void );

/* set URL address of XNAT web site */
/* input: XNAT web site URL */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus setXnatRestUrl( const char *url );

/* set user and password for XNAT web site */
/* input: user ID and password */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus setXnatRestUser( const char *user, const char *password );

/* free dynamically allocated memory for string array */
/* input: number of strings */
/* input/output: array of strings */
XnatRest_EXPORT void freeXnatRestArray( const int numStrings, char **strings );

/* get message for XnatRest status */
/* input: XnatRest status */
/* returns: XnatRest status message */
XnatRest_EXPORT const char *getXnatRestStatusMsg( const XnatRestStatus status );

/* ======== ASYNCHRONOUS TRANSFER FUNCTIONS  ======== */ 

/* transfer data between XNAT and local file */
/* output: number of bytes transferred, transfer status */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus moveXnatRestAsynData( unsigned long *numBytes, XnatRestAsynStatus *finished );

/* cancel transfer of data between XNAT and local file */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus cancelXnatRestAsynTransfer( void );

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

/* ======== PROJECT, SUBJECT, AND EXPERIMENT FUNCTIONS ======== */

/* get names of projects in XNAT */
/* output: number of projects and array of project names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestProjects( int *numProjects, char ***projects );

/* get names of subjects for project */
/* input: XNAT project name */
/* output: number of subjects and array of subject names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestSubjects( const char *project, int *numSubjects, char ***subjects );

/* get names of experiments for subject in project */
/* input: XNAT project name and subject name */
/* output: number of experiments and array of experiment names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestExperiments( const char *project, const char *subject, 
                                       int *numExperiments, char ***experiments );

/* ======== PRIMARY SCAN FUNCTIONS ======== */

/* get names of scans for experiment for subject in project */
/* input: XNAT project name, subject name, and experiment name */
/* output: number of scans and array of scan names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestScans( const char *project, const char *subject, 
                                 const char *experiment, int *numScans, char ***scans );

/* get names of resources in scan for experiment for subject in project */
/* input: XNAT project name, subject name, experiment name, and scan name */
/* output: number of resources and array of resource names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestScanResources( const char *project, const char *subject, 
                                         const char *experiment, const char *scan,
                                         int *numResources, char ***resources );

/* get array of filenames for resource in scan for experiment for subject in project */
/* input: XNAT project name, subject name, experiment name, scan name, and resource name */
/* output: number of filenames and array of filenames */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestScanRsrcFilenames( const char *project, const char *subject, 
                                             const char *experiment, const char *scan,
                                             const char *resource, int *numFilenames, 
                                             char ***filenames );

/* download one file (zipped) from resource in scan for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, scan name, resource name, */
/*        name of file to download, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestScanRsrcFile( const char *project, const char *subject,
                                        const char *experiment, const char *scan,
                                        const char *resource, const char *filename, 
                                        const char *outputZipFilename );

/* download one file (zipped) from resource in scan for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, scan name, resource name, */
/*        name of file to download, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynScanRsrcFile( const char *project, const char *subject,
                                            const char *experiment, const char *scan,
                                            const char *resource, const char *filename, 
                                            const char *outputZipFilename );

/* download all files (zipped) from resource in scan for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, scan name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllFilesInScanRsrc( const char *project, const char *subject,
                                              const char *experiment, const char *scan,
                                              const char *resource, const char *outputZipFilename );

/* download all files (zipped) from resource in scan for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, scan name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllFilesInScanRsrc( const char *project, const char *subject,
                                                  const char *experiment, const char *scan,
                                                  const char *resource, const char *outputZipFilename );

/* ======== AUXILIARY SCAN FUNCTIONS ======== */

/* download all files (zipped) from all resources in scan for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, scan name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllFilesInScan( const char *project, const char *subject,
                                          const char *experiment, const char *scan,
                                          const char *outputZipFilename );

/* download all files (zipped) from all resources in scan for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, scan name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllFilesInScan( const char *project, const char *subject,
                                              const char *experiment, const char *scan,
                                              const char *outputZipFilename );

/* get names of resources for all scans in experiment for subject in project */
/* input: XNAT project name, subject name, and experiment name */
/* output: number of resources and array of resource names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestExprScanResources( const char *project, const char *subject, 
                                             const char *experiment, 
                                             int *numResources, char ***resources );

/* download all files (zipped) from resource for all scans in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllScanFilesInExprRsrc( const char *project, const char *subject,
                                              const char *experiment, const char *resource,
                                              const char *outputZipFilename );

/* download all files (zipped) from resource for all scans in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllScanFilesInExprRsrc( const char *project, const char *subject,
                                                      const char *experiment, const char *resource,
                                                      const char *outputZipFilename );

/* download all files (zipped) from all scans in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllScanFilesInExperiment( const char *project, const char *subject,
                                                    const char *experiment, 
                                                    const char *outputZipFilename );

/* download all files (zipped) from all scans in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllScanFilesInExperiment( const char *project, const char *subject,
                                                        const char *experiment, 
                                                        const char *outputZipFilename );

/* ======== PRIMARY RECONSTRUCTION READ FUNCTIONS ======== */

/* get names of reconstructions for experiment for subject in project */
/* input: XNAT project name, subject name, and experiment name */
/* output: number of reconstructions and array of reconstruction names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestReconstructions( const char *project, const char *subject, 
                                           const char *experiment, int *numReconstructions, 
                                           char ***reconstructions );

/* get names of resources in reconstructions for experiment for subject in project */
/* input: XNAT project name, subject name, experiment name, and reconstruction name */
/* output: number of resources and array of resource names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestReconResources( const char *project, const char *subject, 
                                          const char *experiment, const char *reconstruction,
                                          int *numResources, char ***resources );

/* get array of filenames for resource in reconstruction for experiment for subject in project */
/* input: XNAT project name, subject name, experiment name, reconstruction name, and resource name */
/* output: number of filenames and array of filenames */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestReconRsrcFilenames( const char *project, const char *subject, 
                                              const char *experiment, const char *reconstruction,
                                              const char *resource, int *numFilenames, 
                                              char ***filenames );

/* download one file (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, reconstruction name, resource name, */
/*        name of file to download, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestReconRsrcFile( const char *project, const char *subject,
                                         const char *experiment, const char *reconstruction,
                                         const char *resource, const char *filename, 
                                         const char *outputZipFilename );

/* download one file (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, reconstruction name, resource name, */
/*        name of file to download, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynReconRsrcFile( const char *project, const char *subject,
                                             const char *experiment, const char *reconstruction,
                                             const char *resource, const char *filename, 
                                             const char *outputZipFilename );

/* download all files (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, reconstruction name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllFilesInReconRsrc( const char *project, const char *subject,
                                               const char *experiment, const char *reconstruction,
                                               const char *resource, const char *outputZipFilename );

/* download all files (zipped) from resource in reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, reconstruction name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllFilesInReconRsrc( const char *project, const char *subject,
                                                   const char *experiment, const char *reconstruction,
                                                   const char *resource, const char *outputZipFilename );

/* ======== PRIMARY RECONSTRUCTION WRITE FUNCTIONS ======== */

/* create reconstruction in experiment for subject in project */
/* input: XNAT project name, subject name, experiment name, and reconstruction name */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus putXnatRestReconstruction( const char *project, const char *subject, 
                                          const char *experiment, const char *reconstruction );

/* create resource for reconstruction in experiment for subject in project */
/* input: XNAT project name, subject name, experiment name, reconstruction name, and resource name */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus putXnatRestReconResource( const char *project, const char *subject, 
                                         const char *experiment, const char *reconstruction,
                                         const char *resource );

/* upload files (zipped) to resource in reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file upload is finished */
/* input: XNAT project name, subject name, experiment name, reconstruction name, resource name, */
/*        and input zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus putXnatRestReconRsrcFiles( const char *project, const char *subject,
                                          const char *experiment, const char *reconstruction,
                                          const char *resource, const char *inputZipFilename );

/* upload files (zipped) to resource in reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) upload of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, reconstruction name, resource name, */
/*        and input zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus putXnatRestAsynReconRsrcFiles( const char *project, const char *subject,
                                              const char *experiment, const char *reconstruction,
                                              const char *resource, const char *inputZipFilename );

/* ======== PRIMARY RECONSTRUCTION DELETE FUNCTIONS ======== */

/* delete reconstruction in experiment for subject in project */
/* NOTE: all resources and files within the reconstruction are also deleted */
/* input: XNAT project name, subject name, experiment name, and reconstruction name */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus deleteXnatRestReconstruction( const char *project, const char *subject, 
                                             const char *experiment, const char *reconstruction );

/* delete resource for reconstruction in experiment for subject in project */
/* NOTE: all files within the resource are also deleted */
/* input: XNAT project name, subject name, experiment name, reconstruction name, and resource name */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus deleteXnatRestReconResource( const char *project, const char *subject, 
                                            const char *experiment, const char *reconstruction,
                                            const char *resource );

/* delete file in resource for reconstruction in experiment for subject in project */
/* input: XNAT project name, subject name, experiment name, reconstruction name, resource name, and filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus deleteXnatRestReconRsrcFile( const char *project, const char *subject, 
                                            const char *experiment, const char *reconstruction,
                                            const char *resource, const char *filename );

/* ======== AUXILIARY RECONSTRUCTION READ FUNCTIONS ======== */

/* download all files (zipped) from all resources in reconstruction for experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, reconstruction name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllFilesInReconstruction( const char *project, const char *subject,
                                                    const char *experiment, const char *resconstruction,
                                                    const char *outputZipFilename );

/* download all files (zipped) from all resources in reconstruction for experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, reconstruction name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllFilesInReconstruction( const char *project, const char *subject,
                                                        const char *experiment, const char *resconstruction,
                                                        const char *outputZipFilename );

/* get names of resources for all reconstructions in experiment for subject in project */
/* input: XNAT project name, subject name, and experiment name */
/* output: number of resources and array of resource names */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestExprReconResources( const char *project, const char *subject, 
                                              const char *experiment, 
                                              int *numResources, char ***resources );

/* download all files (zipped) from resource for all reconstructions in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllReconFilesInExprRsrc( const char *project, const char *subject,
                                                   const char *experiment, const char *resource,
                                                   const char *outputZipFilename );

/* download all files (zipped) from resource for all reconstructions in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, resource name, */
/*        and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllReconFilesInExprRsrc( const char *project, const char *subject,
                                                       const char *experiment, const char *resource,
                                                       const char *outputZipFilename );

/* download all files (zipped) from all reconstructions in experiment for subject in project */
/* NOTE: function blocks until ZIP file download is finished */
/* input: XNAT project name, subject name, experiment name, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAllReconFilesInExperiment( const char *project, const char *subject,
                                                     const char *experiment, 
                                                     const char *outputZipFilename );

/* download all files (zipped) from all reconstructions in experiment for subject in project */
/* NOTE: function initializes asynchronous (non-blocking) download of ZIP file and returns */
/* input: XNAT project name, subject name, experiment name, and output zip filename */
/* returns: XnatRest status */
XnatRest_EXPORT XnatRestStatus getXnatRestAsynAllReconFilesInExperiment( const char *project, const char *subject,
                                                         const char *experiment, 
                                                         const char *outputZipFilename );

#endif
