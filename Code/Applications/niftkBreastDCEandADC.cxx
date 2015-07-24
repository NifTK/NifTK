/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

/*!
 * \file niftkBreastDCEandADC.cxx 
 * \page niftkBreastDCEandADC
 * \section niftkBreastDCEandADCSummary niftkBreastDCEandADC
 * 
 * Process breast DCE-MRI and ADC maps
 *
 */


#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iterator>

#include <QProcess>
#include <QString>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <niftkCSVRow.h>
#include <itkCommandLineHelper.h>

#include <itkLogHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageSeriesReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkWriteImage.h>
#include <itkReadImage.h>
#include <itkUCLN4BiasFieldCorrectionFilter.h>
#include <itkImageRegistrationFactory.h>
#include <itkIdentityTransform.h>
#include <itkBreastMaskSegmentationFromMRI.h>
#include <itkBreastMaskSegmForModelling.h>
#include <itkBreastMaskSegmForBreastDensity.h>
#include <itkITKImageToNiftiImage.h>
#include <itkRescaleImageUsingHistogramPercentilesFilter.h>


//#define LINK_TO_SEG_EM

#ifdef LINK_TO_SEG_EM
#include <_seg_EM.h>
#endif

#include <niftkBreastDCEandADCCLP.h>

#define SegPrecisionTYPE float

namespace fs = boost::filesystem;

namespace bio = boost::iostreams;
using bio::tee_device;
using bio::stream;


typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;

typedef float PixelType;
const unsigned int   Dimension = 3;

typedef itk::Image< PixelType, Dimension > ImageType;


// -------------------------------------------------------------------------
// GetTag()
// -------------------------------------------------------------------------

std::string GetTag( const DictionaryType &dictionary, 
                    std::string entryId, std::string *idTag=0 ) 
{
  std::string tagValue;
  std::string tagID;

  DictionaryType::ConstIterator tagItr;
  DictionaryType::ConstIterator end = dictionary.End();
    
  //  It is also possible to read a specific tag. In that case the string of the
  //  entry can be used for querying the MetaDataDictionary.
  
  tagItr = dictionary.Find( entryId );
      
  // If the entry is actually found in the Dictionary, then we can attempt to
  // convert it to a string entry by using a \code{dynamic\_cast}.
      
  if ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    // If the dynamic cast succeed, then we can print out the values of the label,
    // the tag and the actual value.
    
    if ( entryvalue )
    {
      std::string tagkey = tagItr->first;
      bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, tagID );
      
      tagValue = entryvalue->GetMetaDataObjectValue();
    }
  }

  if ( idTag )
  {
    *idTag = tagID;
  }

  return tagValue;
};


// -------------------------------------------------------------------------
// CreateRegisteredFilename()
// -------------------------------------------------------------------------

std::string CreateRegisteredFilename( std::string fileTarget, 
                                      std::string fileSource,
                                      std::string description ) 
{
  std::string fileTargetWithoutSuffix;
  std::string suffixTarget = niftk::ExtractImageFileSuffix( fileTarget, fileTargetWithoutSuffix );

  std::string fileSourceWithoutSuffix;
  std::string suffixSource = niftk::ExtractImageFileSuffix( fileSource, fileSourceWithoutSuffix );

  return fileSourceWithoutSuffix + "_" + description + "_" + fileTargetWithoutSuffix + suffixSource;
};


// -------------------------------------------------------------------------
// class InputParameters
// -------------------------------------------------------------------------

class InputParameters
{

public:

  bool flgVerbose;
  bool flgSaveImages;
  bool flgDebug;
  bool flgCompression;
  bool flgOverwrite;

  std::string dirInput;
  std::string dirOutput;

  std::string fileLog;
  std::string fileOutputCSV;

  std::string strSeriesDescT1W;
  std::string strSeriesDescT2W;
  std::string strSeriesDescADC;
  std::string strSeriesDescDCE;

  std::string progSegEM;
  std::string progRegAffine;
  std::string progRegNonRigid;

  QStringList argsSegEM;
  QStringList argsRegAffine;
  QStringList argsRegNonRigid;

  std::ofstream *foutLog;
  std::ofstream *foutOutputCSV;
  std::ostream *newCout;

  typedef tee_device<ostream, ofstream> TeeDevice;
  typedef stream<TeeDevice> TeeStream;

  TeeDevice *teeDevice;
  TeeStream *teeStream;

  InputParameters( TCLAP::CmdLine &commandLine, 
                   bool verbose, bool flgSave, 
                   bool compression, bool debug, bool overwrite,
                   std::string dInput, std::string dOutput,
                   std::string logfile, std::string csvfile,
                   std::string strT1,
                   std::string strT2,
                   std::string strADC,
                   std::string strDCE,
                   std::string segEM,       QStringList aSegEM,
                   std::string regAffine,   QStringList aRegAffine,
                   std::string regNonRigid, QStringList aRegNonRigid ) {

    std::stringstream message;

    flgVerbose = verbose;
    flgSaveImages = flgSave;
    flgDebug = debug;
    flgCompression = compression;
    flgOverwrite = overwrite;
    
    dirInput  = dInput;
    dirOutput = dOutput;

    fileLog = logfile;
    fileOutputCSV = csvfile;

    strSeriesDescT1W = strT1;
    strSeriesDescT2W = strT2;
    strSeriesDescADC = strADC;
    strSeriesDescDCE = strDCE;

    progSegEM       = segEM;
    progRegAffine   = regAffine;
    progRegNonRigid = regNonRigid;

    argsSegEM       = aSegEM;
    argsRegAffine   = aRegAffine;
    argsRegNonRigid = aRegNonRigid;

    if ( fileLog.length() > 0 )
    {
      foutLog = new std::ofstream( fileLog.c_str() );

      if ((! *foutLog) || foutLog->bad()) {
        message << "Could not open file: " << fileLog << std::endl;
        PrintErrorAndExit( message );
      }

      newCout = new std::ostream( std::cout.rdbuf() );

      teeDevice = new TeeDevice( *newCout, *foutLog); 
      teeStream = new TeeStream( *teeDevice );

      std::cout.rdbuf( teeStream->rdbuf() );
      std::cerr.rdbuf( teeStream->rdbuf() );
    }
    else
    {
      foutLog = 0;
      newCout = 0;
      teeDevice = 0;
      teeStream = 0;
    }

    if ( fileOutputCSV.length() > 0 )
    {
      foutOutputCSV = new std::ofstream( fileOutputCSV.c_str() );

      if ((! *foutOutputCSV) || foutOutputCSV->bad()) {
        message << "Could not open file: " << fileOutputCSV << std::endl;
        PrintErrorAndExit( message );
      }
    }
    else
    {
      foutOutputCSV = 0;
    }

    
    if ( dirInput.length() == 0 )
    {
      commandLine.getOutput()->usage( commandLine );
      message << "The input directory must be specified" << std::endl;
      PrintErrorAndExit( message );
    }
    
    if ( dirOutput.length() == 0 )
    {
      commandLine.getOutput()->usage( commandLine );
      message << "The output directory must be specified" << std::endl;
      PrintErrorAndExit( message );
    }

  }

  ~InputParameters() {
#if 0
    if ( teeStream )
    {
      teeStream->flush();
      teeStream->close();
      delete teeStream;
    }

    if ( teeDevice )
    {
      delete teeDevice;
    }

    if ( foutLog )
    {
      foutLog->close();
      delete foutLog;
    }

    if ( foutOutputCSV )
    {
      foutOutputCSV->close();
      delete foutOutputCSV;
    }    

    if ( newCout )
    {
      delete newCout;
    }
#endif
  }

  void Print(void) {

    std::stringstream message;

    message << std::endl
            << "Input DICOM directory: " << dirInput  << std::endl 
            << "Output directory: "      << dirOutput << std::endl 
            << std::endl
            << std::boolalpha
            << "Verbose output?: "             << flgVerbose     << std::endl
            << "Save images?: "                << flgSaveImages  << std::endl
            << "Compress images?: "            << flgCompression << std::endl
            << "Debugging output?: "           << flgDebug       << std::endl
            << "Overwrite previous results?: " << flgOverwrite   << std::endl
            << std::noboolalpha
            << std::endl
            << "Output log file: " << fileLog       << std::endl
            << "Output csv file: " << fileOutputCSV << std::endl
            << std::endl
            << "T1W series description:          " << strSeriesDescT1W << std::endl
            << "T2W series description:          " << strSeriesDescT2W << std::endl
            << "ADC map series description:      " << strSeriesDescADC << std::endl
            << "DCE sequence series description: " << strSeriesDescDCE << std::endl
            << std::endl
            << "Segmentation executable: "           
            << progSegEM       << " " << argsSegEM.join(" ").toStdString() << std::endl
            << "Affine registration executable: "    
            << progRegAffine   << " " << argsRegAffine.join(" ").toStdString() << std::endl
            << "Non-rigid registration executable: " 
            << progRegNonRigid << " " << argsRegNonRigid.join(" ").toStdString() << std::endl
            << std::endl;

    PrintMessage( message );
  }
    
  void PrintMessage( std::stringstream &message ) {

    std::cout << message.str();
    message.str( "" );
    teeStream->flush();
  }
    
  void PrintError( std::stringstream &message ) {

    std::cerr << "ERROR: " << message.str();
    message.str( "" );
    teeStream->flush();
  }
    
  void PrintErrorAndExit( std::stringstream &message ) {

    PrintError( message );

    exit( EXIT_FAILURE );
  }
    
  void PrintWarning( std::stringstream &message ) {

    std::cerr << "WARNING: " << message.str();
    message.str( "" );
    teeStream->flush();
  }

  void PrintTag( const DictionaryType &dictionary, 
                 std::string entryId ) {

    std::stringstream message;
    std::string tagID;
        
    std::string tagValue = GetTag( dictionary, entryId, &tagID );
          
    message << "   " << tagID <<  ": " << tagValue << std::endl;
    PrintMessage( message );
  }

  bool ReadImageFromFile( std::string inDir, std::string filename, 
                          std::string description, ImageType::Pointer &image ) {
  
    std::stringstream message;
    std::string fileInput = niftk::ConcatenatePath( inDir, filename );
           
    if ( itk::ReadImageFromFile< ImageType >( fileInput, image ) )
    {   
      message << std::endl << "Read " << description << " from file: " << fileInput << std::endl;
      PrintMessage( message );
      return true;
    }
    else
    {
      return false;
    }
  }

  bool ReadImageFromFile( std::string inDir, std::string filename, 
                          std::string description, nifti_image *&image ) {
  
    std::stringstream message;
    std::string fileInput = niftk::ConcatenatePath( inDir, filename );

    image = nifti_image_read( fileInput.c_str(), true );
      
    if ( image != NULL )
    {   
      message << std::endl << "Read " << description << " from file: " << fileInput << std::endl;
      PrintMessage( message );
      return true;
    }
    else
    {
      return false;
    }
  }

  void WriteImageToFile( std::string filename, 
                         std::string description, ImageType::Pointer image ) {
  
    std::stringstream message;
    std::string fileOutput = niftk::ConcatenatePath( dirOutput, filename );
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    itk::WriteImageToFile< ImageType >( fileOutput, image );
  }

  void WriteImageToFile( std::string filename, 
                         std::string description, nifti_image *image ) {
  
    std::stringstream message;
    std::string fileOutput = niftk::ConcatenatePath( dirOutput, filename );
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    nifti_image_write( image );
  }

  void DeleteFile( std::string filename ) {

    std::stringstream message;
    std::string filePath = niftk::ConcatenatePath( dirOutput, filename );

    if ( ! niftk::FileExists( filePath ) )
    {
      return;
    }

    if ( niftk::FileDelete( filePath ) )
    {
      message << std::endl << "Deleted file: " << filePath << std::endl;
      PrintMessage( message );
    }
    else
    {
      message << std::endl << "Failed to delete file: " << filePath << std::endl;
      PrintWarning( message );
    }      
  }

};


// -------------------------------------------------------------------------
// ConvertAffineTransformationMatrixToRegF3D()
// -------------------------------------------------------------------------

bool ConvertAffineTransformationMatrixToRegF3D( std::string fileAffineTransformFullPath,
                                                InputParameters &args ) 
{
  std::stringstream message;

  typedef itk::AffineTransform<double,3> DoubleAffineType;

  itk::TransformFactoryBase::RegisterDefaultTransforms();

  typedef itk::TransformFileReader TransformReaderType;
  TransformReaderType::Pointer transformFileReader = TransformReaderType::New();

  transformFileReader->SetFileName( fileAffineTransformFullPath );

  try
  {
    transformFileReader->Update();         
  }
  catch ( itk::ExceptionObject &e )
  {
    message << "ERROR: Failed to read " << fileAffineTransformFullPath << std::endl;
    args.PrintError( message );
    return false;
  }
 
  typedef TransformReaderType::TransformListType TransformListType;
  typedef TransformReaderType::TransformType BaseTransformType;
  
  TransformListType *list = transformFileReader->GetTransformList();
  BaseTransformType::Pointer transform = list->front();
  
  transform->Print( std::cout );

  DoubleAffineType *doubleAffine = static_cast< DoubleAffineType * >( transform.GetPointer() );
 
  if( doubleAffine == NULL )
  {
    message << "ERROR: Could not cast: " << fileAffineTransformFullPath << std::endl;
    args.PrintError( message );
    return false;
  }

  std::ofstream *foutMatrix;

  foutMatrix = new std::ofstream( fileAffineTransformFullPath.c_str() );
  
  if ( (! *foutMatrix) || foutMatrix->bad() ) {
    message << "ERROR: Could not open output file: " << fileAffineTransformFullPath << std::endl;
    args.PrintMessage( message );
    return false;
  }


  DoubleAffineType::MatrixType matrix = doubleAffine->GetMatrix();
  DoubleAffineType::OffsetType offset = doubleAffine->GetOffset();

  int i, j;

  for ( j=0; j<3; j++ )
  {
    for ( i=0; i<3; i++ )
    {
      *foutMatrix << matrix[j][i] << " ";
    }
  }

  for ( j=0; j<3; j++ )
  {
    *foutMatrix << offset[j] << " ";
  }

  *foutMatrix << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;

  message << "Affine matrix: " << matrix << std::endl;
  args.PrintMessage( message );

  foutMatrix->close();
  delete foutMatrix;
};


// -------------------------------------------------------------------------
// AffineRegisterImages()
// -------------------------------------------------------------------------

bool AffineRegisterImages( std::string fileTarget, 
                          std::string fileSource, 
                          std::string fileRegistered,
                          std::string &fileAffineTransform,
                          InputParameters &args ) 
{
  bool flgUsingNiftkAffine = false;
  std::stringstream message;

  fileAffineTransform = niftk::ModifyImageFileSuffix( fileRegistered, std::string( "_Transform.txt" ) );

  std::string fileAffineTransformFullPath 
    = niftk::ConcatenatePath( args.dirOutput, fileAffineTransform.c_str() );

  QStringList argsRegAffine = args.argsRegAffine; 

  if ( args.progRegAffine.find( "niftkAffine" ) != std::string::npos ) 
  {
    flgUsingNiftkAffine = true;

    argsRegAffine 
      << "--ti" << niftk::ConcatenatePath( args.dirOutput, fileTarget.c_str() ).c_str()
      << "--si" << niftk::ConcatenatePath( args.dirOutput, fileSource.c_str() ).c_str()
      << "--oi" << niftk::ConcatenatePath( args.dirOutput, fileRegistered.c_str() ).c_str()
      << "--om" << fileAffineTransformFullPath.c_str();
  }
  else if ( args.progRegAffine.find( "reg_aladin" ) != std::string::npos ) 
  {
    argsRegAffine 
      << "-target" << niftk::ConcatenatePath( args.dirOutput, fileTarget.c_str() ).c_str()
      << "-source" << niftk::ConcatenatePath( args.dirOutput, fileSource.c_str() ).c_str()
      << "-result" << niftk::ConcatenatePath( args.dirOutput, fileRegistered.c_str() ).c_str()
      << "-aff"    << fileAffineTransformFullPath.c_str();
  }

  message << std::endl << "Executing affine registration (QProcess): "
          << std::endl << "   " << args.progRegAffine;
  for(int i=0;i<argsRegAffine.size();i++)
  {
    message << " " << argsRegAffine[i].toStdString();
  }
  message << std::endl << std::endl;
  args.PrintMessage( message );
  
  QProcess callRegAffine;
  QString outRegAffine;
  
  callRegAffine.setProcessChannelMode( QProcess::MergedChannels );


  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();

  callRegAffine.start( args.progRegAffine.c_str(), argsRegAffine );
  
  bool flgFinished = callRegAffine.waitForFinished( 7200000 ); // Wait two hours
  
  boost::posix_time::ptime endTime = boost::posix_time::second_clock::local_time();
  boost::posix_time::time_duration duration = endTime - startTime;

  outRegAffine = callRegAffine.readAllStandardOutput();
  
  message << outRegAffine.toStdString();
  message << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl;
  
  if ( ! flgFinished )
  {
    message << "ERROR: Could not execute: " << args.progRegAffine << " ( " 
            << callRegAffine.errorString().toStdString() << " )" << std::endl;
    args.PrintMessage( message );
    return false;
  }
  
  args.PrintMessage( message );
  
  callRegAffine.close();

  // Convert the transformation to reg_f3d compatibility

  if ( flgUsingNiftkAffine )
  {
    return ConvertAffineTransformationMatrixToRegF3D( fileAffineTransformFullPath, args );
  }

  return true;
};


// -------------------------------------------------------------------------
// NonRigidRegisterImages()
// -------------------------------------------------------------------------

bool NonRigidRegisterImages( std::string fileTarget, 
                             std::string fileSource, 
                             std::string fileRegistered,
                             std::string fileAffineTransform,
                             InputParameters &args ) 
{
  std::stringstream message;

  std::string fileTransform = niftk::ModifyImageFileSuffix( fileRegistered, 
                                                            std::string( "_Transform.nii" ) );

  if ( args.flgCompression ) fileTransform.append( ".gz" );

  QStringList argsRegNonRigid = args.argsRegNonRigid; 

  argsRegNonRigid 
    << "-target" << niftk::ConcatenatePath( args.dirOutput, fileTarget.c_str() ).c_str()
    << "-source" << niftk::ConcatenatePath( args.dirOutput, fileSource.c_str() ).c_str()
    << "-res"    << niftk::ConcatenatePath( args.dirOutput, fileRegistered.c_str() ).c_str()
    << "-aff"    << niftk::ConcatenatePath( args.dirOutput, fileAffineTransform.c_str() ).c_str()
    << "-cpp"    << niftk::ConcatenatePath( args.dirOutput, fileTransform.c_str() ).c_str();
  
  message << std::endl << "Executing non-rigid registration (QProcess): "
          << std::endl << "   " << args.progRegNonRigid;
  for(int i=0;i<argsRegNonRigid.size();i++)
  {
    message << " " << argsRegNonRigid[i].toStdString();
  }
  message << std::endl << std::endl;
  args.PrintMessage( message );
  
  QProcess callRegNonRigid;
  QString outRegNonRigid;
  
  callRegNonRigid.setProcessChannelMode( QProcess::MergedChannels );


  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();

  callRegNonRigid.start( args.progRegNonRigid.c_str(), argsRegNonRigid );
  
  bool flgFinished = callRegNonRigid.waitForFinished( 10800000 ); // Wait three hours
  
  boost::posix_time::ptime endTime = boost::posix_time::second_clock::local_time();
  boost::posix_time::time_duration duration = endTime - startTime;
  
  outRegNonRigid = callRegNonRigid.readAllStandardOutput();
  
  message << outRegNonRigid.toStdString();
  message << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl;

  if ( ! flgFinished )
  {
    message << "ERROR: Could not execute: " << args.progRegNonRigid << " ( " 
            << callRegNonRigid.errorString().toStdString() << " )" << std::endl;
    args.PrintMessage( message );
    return false;
  }
  
  args.PrintMessage( message );
  
  callRegNonRigid.close();

  return true;
};


// -------------------------------------------------------------------------
// RegisterImages()
// -------------------------------------------------------------------------

bool RegisterImages( std::string fileTarget, 
                     std::string fileSource, 
                     std::string fileRegistered,
                     InputParameters &args ) 
{
  std::string fileAffineTransform;
  std::string fileAffineRegistered = CreateRegisteredFilename( fileTarget,
                                                               fileSource,
                                                               std::string( "AffineTo" ) );

  return ( AffineRegisterImages( fileTarget, 
                                fileSource, 
                                fileAffineRegistered,
                                fileAffineTransform, 
                                args ) 
           &&
           NonRigidRegisterImages( fileTarget, 
                                   fileSource, 
                                   fileRegistered,
                                   fileAffineTransform,
                                   args ) );
};


// -------------------------------------------------------------------------
// SplitStringIntoCommandAndArguments()
// -------------------------------------------------------------------------

std::string SplitStringIntoCommandAndArguments( std::string inString,
                                                QStringList &arguments )
{
  std::string command;

  std::stringstream ssString( inString );

  std::istream_iterator< std::string > itStringStream( ssString );
  std::istream_iterator< std::string > itEnd;

  command = *itStringStream;
  itStringStream++;
  
  for (; itStringStream != itEnd; itStringStream++)
  {
    arguments << (*itStringStream).c_str();
  }

  return command;
};


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  bool flgVeryFirstRow = true;

  float progress = 0.;
  float iTask = 0.;
  float nTasks;

  std::stringstream message;
  std::stringstream ss;

  std::vector< std::string > fileNamesT1;
  std::vector< std::string > fileNamesT2;
  std::vector< std::string > fileNamesADC;

  typedef std::map< unsigned int, std::vector< std::string > > SeriesDCEMapType;
  SeriesDCEMapType seriesDCE;

  std::map< unsigned int, std::string > fileNamesDCEImages;

  typedef itk::ImageSeriesReader< ImageType > SeriesReaderType;
  typedef itk::UCLN4BiasFieldCorrectionFilter< ImageType, ImageType > BiasFieldCorrectionType;

  fs::path pathExecutable( argv[0] );
  fs::path dirExecutables = pathExecutable.parent_path();


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  // Extract any arguments from the input commands

  QStringList argsSegEM;
  std::string progSegEM = SplitStringIntoCommandAndArguments( comSegEM, argsSegEM );

  QStringList argsRegAffine;
  std::string progRegAffine = SplitStringIntoCommandAndArguments( comRegAffine, argsRegAffine );

  QStringList argsRegNonRigid;
  std::string progRegNonRigid = SplitStringIntoCommandAndArguments( comRegNonRigid, argsRegNonRigid );

  InputParameters args( commandLine, 
                        flgVerbose, flgSaveImages, 
                        flgCompression, flgDebug, flgOverwrite,
                        dirInput, dirOutput,
                        fileLog, fileOutputCSV,
                        strSeriesDescT1W,
                        strSeriesDescT2W,
                        strSeriesDescADC,
                        strSeriesDescDCE,
                        progSegEM,       argsSegEM,
                        progRegAffine,   argsRegAffine,
                        progRegNonRigid, argsRegNonRigid );


  // Can we find the seg_EM executable or verify the path?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! niftk::FileExists( args.progSegEM ) )
  {
    std::string fileSearchSegEM = niftk::ConcatenatePath( dirExecutables.string(), 
                                                          args.progSegEM );
          
    if ( niftk::FileExists( fileSearchSegEM ) )
    {
      args.progSegEM = fileSearchSegEM;
    }
  }


  // Can we find the reg_aladin executable or verify the path?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! niftk::FileExists( args.progRegAffine ) )
  {
    std::string fileSearchRegAffine = niftk::ConcatenatePath( dirExecutables.string(), 
                                                              args.progRegAffine );
          
    if ( niftk::FileExists( fileSearchRegAffine ) )
    {
      args.progRegAffine = fileSearchRegAffine;
    }
  }


  // Can we find the reg_f3d executable or verify the path?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! niftk::FileExists( args.progRegNonRigid ) )
  {
    std::string fileSearchRegNonRigid = niftk::ConcatenatePath( dirExecutables.string(), 
                                                                args.progRegNonRigid );
          
    if ( niftk::FileExists( fileSearchRegNonRigid ) )
    {
      args.progRegNonRigid = fileSearchRegNonRigid;
    }
  }


  args.Print();


  // Initialise the output file names
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string fileI00_T1  = std::string( "I00_" ) + strSeriesDescT1W  + ".nii";
  std::string fileI00_T2  = std::string( "I00_" ) + strSeriesDescT2W  + ".nii";
  std::string fileI00_ADC = std::string( "I00_" ) + strSeriesDescADC + ".nii";

  std::string fileI01_T1_BiasFieldCorr
    = std::string( "I01_" ) + strSeriesDescT1W + "_BiasFieldCorrection.nii";
  std::string fileI01_T2_BiasFieldCorr
    = std::string( "I01_" ) + strSeriesDescT2W + "_BiasFieldCorrection.nii";

  std::string fileOutputVTKSurface( "I13_BreastSurface.vtk" );
  std::string fileOutputBreastMask( "I14_BreastMaskSegmentation.nii" );
  std::string fileOutputParenchyma( "I15_BreastParenchyma.nii" );

  std::string fileBIFs( "I03_OrientedBIFsSig3_Axial.nii" );

  std::string fileOutputSmoothedStructural;
  std::string fileOutputSmoothedFatSat;
  std::string fileOutputClosedStructural;

  std::string fileOutputMaxImage( "I04_FatSat_and_T2_MaximumIntensities.nii" );
  std::string fileOutputCombinedHistogram( "I05_FatSat_and_T2_CombinedHistogram.txt" );
  std::string fileOutputRayleigh( "I05_FatSat_and_T2_RayleighFit.txt" );
  std::string fileOutputFreqLessBgndCDF( "I05_FatSat_and_T2_FreqLessBgndCDF.txt" );
  std::string fileOutputBackground( "I05_BackgroundMask.nii" );
  std::string fileOutputSkinElevationMap;
  std::string fileOutputGradientMagImage( "I06_GradientMagImage.nii" );
  std::string fileOutputSpeedImage( "I07_SpeedImage.nii" );
  std::string fileOutputFastMarchingImage( "I08_FastMarchingImage.nii" );
  std::string fileOutputPectoral( "I09_PectoralMask.nii" );
  std::string fileOutputChestPoints( "I10_ChestPoints.nii" );
  std::string fileOutputPectoralSurfaceMask( "I11_PectoralSurfaceMask.nii" );

  std::string fileOutputPectoralSurfaceVoxels;

  std::string fileOutputFittedBreastMask( "I12_AnteriorSurfaceCropMask.nii" );

  std::string fileRegistrationTarget( fileI01_T1_BiasFieldCorr );



  if ( flgCompression )
  {
    fileI00_T1.append( ".gz" );
    fileI00_T2.append( ".gz" );
    fileI00_ADC.append( ".gz" );

    fileI01_T1_BiasFieldCorr.append( ".gz" );
    fileI01_T2_BiasFieldCorr.append( ".gz" );

    fileOutputBreastMask.append( ".gz" );
    fileOutputParenchyma.append( ".gz" );

    if ( fileBIFs.length() > 0 )                        fileBIFs.append( ".gz" );
        
    if ( fileOutputSmoothedStructural.length() > 0 )    fileOutputSmoothedStructural.append( ".gz" );
    if ( fileOutputSmoothedFatSat.length() > 0 )        fileOutputSmoothedFatSat.append( ".gz" );
    if ( fileOutputClosedStructural.length() > 0 )      fileOutputClosedStructural.append( ".gz" );
                                                                       
    if ( fileOutputMaxImage.length() > 0 )              fileOutputMaxImage.append( ".gz" );
    if ( fileOutputBackground.length() > 0 )            fileOutputBackground.append( ".gz" );
    if ( fileOutputSkinElevationMap.length() > 0 )      fileOutputSkinElevationMap.append( ".gz" );
    if ( fileOutputGradientMagImage.length() > 0 )      fileOutputGradientMagImage.append( ".gz" );
    if ( fileOutputSpeedImage.length() > 0 )            fileOutputSpeedImage.append( ".gz" );
    if ( fileOutputFastMarchingImage.length() > 0 )     fileOutputFastMarchingImage.append( ".gz" );
    if ( fileOutputPectoral.length() > 0 )              fileOutputPectoral.append( ".gz" );
    if ( fileOutputChestPoints.length() > 0 )           fileOutputChestPoints.append( ".gz" );
    if ( fileOutputPectoralSurfaceMask.length() > 0 )   fileOutputPectoralSurfaceMask.append( ".gz" );
                                                                       
    if ( fileOutputPectoralSurfaceVoxels.length() > 0 ) fileOutputPectoralSurfaceVoxels.append( ".gz" );
    if ( fileOutputFittedBreastMask.length() > 0 )      fileOutputFittedBreastMask.append( ".gz" );

    if ( fileRegistrationTarget.length() > 0 )          fileRegistrationTarget.append( ".gz" );
  }


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ImageType::Pointer imT1  = 0;
  ImageType::Pointer imT2  = 0;
  ImageType::Pointer imADC = 0;
  ImageType::Pointer imDCE = 0;

  ImageType::Pointer imSegmentedBreastMask = 0;
  ImageType::Pointer imParenchyma = 0;

  message << std::endl << "Directory: " << args.dirInput << std::endl << std::endl;
  args.PrintMessage( message );

  if ( ! niftk::DirectoryExists( args.dirOutput ) )
  {
    niftk::CreateDirAndParents( args.dirOutput );
    
    message << "Creating output directory: " << args.dirOutput << std::endl;
    args.PrintMessage( message );
  }


  // Find the DICOM files in this directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  fileNamesT1.clear();
  fileNamesT2.clear();
  fileNamesADC.clear();


  std::vector< std::string > fileNames;
  std::vector< std::string >::iterator iterFilenames;
     
  typedef itk::GDCMSeriesFileNames NamesGeneratorType;

  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

  nameGenerator->SetLoadSequences( true );
  nameGenerator->SetLoadPrivateTags( true ); 

  nameGenerator->SetUseSeriesDetails( true );
  
  nameGenerator->SetDirectory( args.dirInput );
  
  
  // The GDCMSeriesFileNames object first identifies the list of DICOM series
  // that are present in the given directory. We receive that list in a
  // reference to a container of strings and then we can do things like
  // printing out all the series identifiers that the generator had
  // found. Since the process of finding the series identifiers can
  // potentially throw exceptions, it is wise to put this code inside a
  // try/catch block.
  
  typedef std::vector<std::string> seriesIdContainer;
  const seriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

  seriesIdContainer::const_iterator seriesItr = seriesUID.begin();
  seriesIdContainer::const_iterator seriesEnd = seriesUID.end();

  while( seriesItr != seriesEnd )
  {
    message << std::endl << "Series: " << seriesItr->c_str() << std::endl;
    args.PrintMessage( message );

    try
    {
      fileNames = nameGenerator->GetFileNames( *seriesItr );
    }
    catch ( itk::ExceptionObject &e )
    {
      message << "Failed to get DICOM file names" << std::endl;
      args.PrintError( message );
      return EXIT_FAILURE;
    }
     
      
    // Read the first image in this series

    typedef itk::ImageFileReader< ImageType > ImageReaderType;

    ImageReaderType::Pointer imReader = ImageReaderType::New();
    imReader->SetFileName( fileNames[0] );
      
    typedef itk::GDCMImageIO ImageIOType;

    ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

    gdcmImageIO->SetMaxSizeLoadEntry(0xffff);

    imReader->SetImageIO( gdcmImageIO );

    try
    {
      imReader->Update();
    }
    catch ( itk::ExceptionObject &e )
    {
      message << "Failed to read file: " << fileNames[0] << std::endl;
      args.PrintError( message );
      return EXIT_FAILURE;
    }
 
    const  DictionaryType &dictionary = gdcmImageIO->GetMetaDataDictionary();
      
    args.PrintTag( dictionary, "0008|103e" ); // Series description
    args.PrintTag( dictionary, "0018|1030" ); // Protocol name
    args.PrintTag( dictionary, "0020|0011" ); // Series Number


    // Print all the images in this series

    if ( args.flgDebug )
    {
      for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) 
      {
        message << "      " << *iterFilenames << std::endl;        
        args.PrintMessage( message );
      }
    }

    std::string seriesDescription = GetTag( dictionary, "0008|103e" );

    if ( seriesDescription.find( args.strSeriesDescT1W ) != std::string::npos )
    {
      fileNamesT1 = fileNames;
    }
    else if ( seriesDescription.find( args.strSeriesDescT2W ) != std::string::npos )
    {
      fileNamesT2 = fileNames;
    }
    else if ( seriesDescription.find( args.strSeriesDescADC ) != std::string::npos )
    {
      fileNamesADC = fileNames;
    }
    else if ( seriesDescription.find( args.strSeriesDescDCE ) != std::string::npos )
    {
      unsigned int seriesNumber = atoi( GetTag( dictionary, "0020|0011" ).c_str() );

      seriesDCE.insert( std::pair< unsigned int, 
                                   std::vector< std::string > >( seriesNumber,
                                                                 fileNames ) );
    }

    seriesItr++;
  }

  // The total number of tasks T1W + Seg + T2W + ADC + N_DCE

  nTasks = 4 + seriesDCE.size();

  progress = iTask/nTasks;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;


  // Load the T1W image
  // ~~~~~~~~~~~~~~~~~~

  if ( flgOverwrite || 
       ( ! args.ReadImageFromFile( args.dirOutput, fileI01_T1_BiasFieldCorr, 
                                   std::string( "bias field corrected '") +
                                   args.strSeriesDescT1W + "' image", 
                                   imT1 ) ) )
  {
    if ( flgOverwrite || 
         ( ! args.ReadImageFromFile( args.dirOutput, fileI00_T1, 
                                     std::string( "T1W '" ) + args.strSeriesDescT1W 
                                     + "' image", imT1 ) ) )
    {
      if ( fileNamesT1.size() > 0 )
      {
            
        SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
        seriesReader->SetFileNames( fileNamesT1 );

        message << std::endl << "Reading '" << args.strSeriesDescT1W << "' image" << std::endl;  
        args.PrintMessage( message );

        seriesReader->UpdateLargestPossibleRegion();

        imT1 = seriesReader->GetOutput();
        imT1->DisconnectPipeline();
            
        args.WriteImageToFile( fileI00_T1, 
                               std::string( "T1W '" ) + args.strSeriesDescT1W +
                               "' image", imT1 );
      }
    }

    // Bias field correct it

    if ( imT1 ) 
    {
          
      message << std::endl << "Bias field correcting '" << args.strSeriesDescT1W << "' image"
              << std::endl;  
      args.PrintMessage( message );
          
      BiasFieldCorrectionType::Pointer biasFieldCorrector = BiasFieldCorrectionType::New();
          
      biasFieldCorrector->SetInput( imT1 );
      biasFieldCorrector->Update();
          
      imT1 = biasFieldCorrector->GetOutput();
      imT1->DisconnectPipeline();
          
      args.WriteImageToFile( fileI01_T1_BiasFieldCorr, 
                             std::string( "bias field corrected '" ) + 
                             args.strSeriesDescT1W + "' image", imT1 );
    }
  }

  iTask++;
  progress = iTask/nTasks;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;


  // Load the T2W image
  // ~~~~~~~~~~~~~~~~~~

  std::string fileRegisteredT2W = CreateRegisteredFilename( fileRegistrationTarget,
                                                            fileI01_T2_BiasFieldCorr,
                                                            std::string( "NonRigidTo" ) );

  

  if ( flgOverwrite || 
       ( ! args.ReadImageFromFile( args.dirOutput, fileRegisteredT2W, 
                                   std::string( "registered '") +
                                   args.strSeriesDescT2W + "' image", 
                                   imT2 ) ) )
  {
    if ( flgOverwrite || 
         ( ! args.ReadImageFromFile( args.dirOutput, fileI01_T2_BiasFieldCorr, 
                                     std::string( "bias field corrected '" ) + 
                                     args.strSeriesDescT2W + "' image", imT2 ) ) )
    {
      if ( flgOverwrite || 
           ( ! args.ReadImageFromFile( args.dirOutput, fileI00_T2,
                                       std::string( "T2W '" ) + 
                                       args.strSeriesDescT2W + "' image", 
                                       imT2 ) ) )
      {
        if ( fileNamesT2.size() > 0 )
        {
          
          SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
          seriesReader->SetFileNames( fileNamesT2 );

          message << std::endl << "Reading '" << args.strSeriesDescT2W << "' image" << std::endl;  
          args.PrintMessage( message );
          
          seriesReader->UpdateLargestPossibleRegion();
          
          imT2 = seriesReader->GetOutput();
          imT2->DisconnectPipeline();

          args.WriteImageToFile( fileI00_T2,
                                 std::string( "T2W '" ) + args.strSeriesDescT2W +
                                 "' image", imT2 );
        }
      }

      // Bias field correct it
          
      if ( imT2 )
      {
        message << std::endl << "Bias field correcting '" << args.strSeriesDescT2W << "' image" << std::endl;  
        args.PrintMessage( message );
        
        BiasFieldCorrectionType::Pointer biasFieldCorrector = BiasFieldCorrectionType::New();
        
        biasFieldCorrector->SetInput( imT2 );
        biasFieldCorrector->Update();
        
        imT2 = biasFieldCorrector->GetOutput();
        imT2->DisconnectPipeline();
        
        // Rescale the 98th percentile to 100
          
        message << std::endl << "Rescaling '" << args.strSeriesDescT2W << "' image to 100" << std::endl;  
        args.PrintMessage( message );

        typedef itk::RescaleImageUsingHistogramPercentilesFilter<ImageType, ImageType> RescaleFilterType;
        
        RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
        rescaleFilter->SetInput( imT2 );
  
        rescaleFilter->SetInLowerPercentile(  0. );
        rescaleFilter->SetInUpperPercentile( 98. );

        rescaleFilter->SetOutLowerLimit(   0. );
        rescaleFilter->SetOutUpperLimit( 100. );

        rescaleFilter->Update();

        imT2 = rescaleFilter->GetOutput();
        imT2->DisconnectPipeline();          
            
        args.WriteImageToFile( fileI01_T2_BiasFieldCorr, 
                               std::string( "bias field corrected '" ) +
                               args.strSeriesDescT2W + "' image", imT2 );
      }
    }
      
    // Register the T2W image to the T1W image

    if ( ! ( RegisterImages( fileRegistrationTarget,
                             fileI01_T2_BiasFieldCorr,
                             fileRegisteredT2W,
                             args ) 
             &&
             args.ReadImageFromFile( args.dirOutput, fileRegisteredT2W, 
                                     std::string( "registered '") +
                                     args.strSeriesDescT2W + "' image", 
                                     imT2 ) ) )
    {
      imT2 = 0;
    }
  }


  // Have we found both input images?

  if ( ! ( imT2 && imT1 ) )
  {
    message << "Both of T1W and T2W images not found" << std::endl << std::endl;
    args.PrintError( message );
    return EXIT_FAILURE;
  }
  
  iTask++;
  progress = iTask/nTasks;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;
          

  // Run the breast mask segmentation?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( flgOverwrite || 
       ( ! args.ReadImageFromFile( args.dirOutput, fileOutputBreastMask, 
                                   "segmented breast mask", 
                                   imSegmentedBreastMask ) ) )
  {

    bool flgSmooth = true;
    
    bool flgLeft = false;
    bool flgRight = false;
    
    bool flgExtInitialPect = false;

    int regGrowXcoord = 0;
    int regGrowYcoord = 0;
    int regGrowZcoord = 0;

    float bgndThresholdProb = 0.;
        
    float finalSegmThreshold = 0.49;

    float sigmaInMM = 1;

    float fMarchingK1   = 30.0;
    float fMarchingK2   = 7.5;
    float fMarchingTime = 0.;

    float sigmaBIF = 3.0;

    bool flgProneSupineBoundary = false;
    float cropProneSupineDistPostMidSternum  = 40.0;

    bool flgCropWithFittedSurface = true;


    ImageType::Pointer imBIFs;


    // Create the Breast Segmentation Object
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    typedef itk::BreastMaskSegmentationFromMRI< Dimension, PixelType > 
      BreastMaskSegmentationFromMRIType;
  
    typedef itk::BreastMaskSegmForModelling< Dimension, PixelType > 
      BreastMaskSegmForModellingType;
    
    typedef itk::BreastMaskSegmForBreastDensity< Dimension, PixelType > 
      BreastMaskSegmForBreastDensityType;
  
    BreastMaskSegmentationFromMRIType::Pointer breastMaskSegmentor;

    if ( flgProneSupineBoundary )
    {
      breastMaskSegmentor = BreastMaskSegmForModellingType::New();
    } 
    else
    {
      breastMaskSegmentor = BreastMaskSegmForBreastDensityType::New();
    } 


    // Pass Command Line Parameters to Segmentor

    breastMaskSegmentor->SetVerbose( flgVerbose );
    breastMaskSegmentor->SetSmooth(  flgSmooth );

    breastMaskSegmentor->SetLeftBreast(  flgLeft );
    breastMaskSegmentor->SetRightBreast( flgRight );
  
    breastMaskSegmentor->SetExtInitialPect( flgExtInitialPect );
  
    breastMaskSegmentor->SetRegionGrowX( regGrowXcoord );
    breastMaskSegmentor->SetRegionGrowY( regGrowYcoord );
    breastMaskSegmentor->SetRegionGrowZ( regGrowZcoord );

    breastMaskSegmentor->SetBackgroundThreshold( bgndThresholdProb );
    breastMaskSegmentor->SetFinalSegmThreshold( finalSegmThreshold );

    breastMaskSegmentor->SetSigmaInMM( sigmaInMM );

    breastMaskSegmentor->SetMarchingK1( fMarchingK1   );
    breastMaskSegmentor->SetMarchingK2( fMarchingK2   );
    breastMaskSegmentor->SetMarchingTime( fMarchingTime );

    breastMaskSegmentor->SetSigmaBIF( sigmaBIF );

    breastMaskSegmentor->SetCropFit( flgCropWithFittedSurface );
    breastMaskSegmentor->SetCropDistancePosteriorToMidSternum( cropProneSupineDistPostMidSternum );
          

    if ( args.flgDebug )
    {

      if ( fileOutputSmoothedStructural.length() > 0 )    breastMaskSegmentor->SetOutputSmoothedStructural(   niftk::ConcatenatePath( dirOutput, fileOutputSmoothedStructural ) );
      if ( fileOutputSmoothedFatSat.length() > 0 )        breastMaskSegmentor->SetOutputSmoothedFatSat(       niftk::ConcatenatePath( dirOutput, fileOutputSmoothedFatSat ) );
      if ( fileOutputClosedStructural.length() > 0 )      breastMaskSegmentor->SetOutputClosedStructural(     niftk::ConcatenatePath( dirOutput, fileOutputClosedStructural ) );
      if ( fileOutputCombinedHistogram.length() > 0 )     breastMaskSegmentor->SetOutputHistogram(            niftk::ConcatenatePath( dirOutput, fileOutputCombinedHistogram ) );
      if ( fileOutputRayleigh.length() > 0 )              breastMaskSegmentor->SetOutputFit(                  niftk::ConcatenatePath( dirOutput, fileOutputRayleigh ) );
      if ( fileOutputFreqLessBgndCDF.length() > 0 )       breastMaskSegmentor->SetOutputCDF(                  niftk::ConcatenatePath( dirOutput, fileOutputFreqLessBgndCDF ) );
      if ( fileOutputMaxImage.length() > 0 )              breastMaskSegmentor->SetOutputImageMax(             niftk::ConcatenatePath( dirOutput, fileOutputMaxImage ) );
      if ( fileOutputBackground.length() > 0 )            breastMaskSegmentor->SetOutputBackground(           niftk::ConcatenatePath( args.dirOutput, fileOutputBackground ) );
      if ( fileOutputSkinElevationMap.length() > 0 )      breastMaskSegmentor->SetOutputSkinElevationMap(     niftk::ConcatenatePath( args.dirOutput, fileOutputSkinElevationMap ) );
      if ( fileOutputChestPoints.length() > 0 )           breastMaskSegmentor->SetOutputChestPoints(          niftk::ConcatenatePath( args.dirOutput, fileOutputChestPoints ) );
      if ( fileOutputPectoral.length() > 0 )              breastMaskSegmentor->SetOutputPectoralMask(         niftk::ConcatenatePath( args.dirOutput, fileOutputPectoral ) );
      if ( fileOutputPectoralSurfaceMask.length() > 0 )   breastMaskSegmentor->SetOutputPecSurfaceMask(       niftk::ConcatenatePath( args.dirOutput, fileOutputPectoralSurfaceMask ) );

      if ( fileOutputGradientMagImage.length() > 0 )      breastMaskSegmentor->SetOutputGradientMagImage(     niftk::ConcatenatePath( args.dirOutput, fileOutputGradientMagImage ) );
      if ( fileOutputSpeedImage.length() > 0 )            breastMaskSegmentor->SetOutputSpeedImage(           niftk::ConcatenatePath( args.dirOutput, fileOutputSpeedImage ) );
      if ( fileOutputFastMarchingImage.length() > 0 )     breastMaskSegmentor->SetOutputFastMarchingImage(    niftk::ConcatenatePath( args.dirOutput, fileOutputFastMarchingImage ) );
  
      if ( fileOutputPectoralSurfaceVoxels.length() > 0 ) breastMaskSegmentor->SetOutputPectoralSurf(         niftk::ConcatenatePath( args.dirOutput, fileOutputPectoralSurfaceVoxels ) );
  
      if ( fileOutputFittedBreastMask.length() > 0 )      breastMaskSegmentor->SetOutputBreastFittedSurfMask( niftk::ConcatenatePath( args.dirOutput, fileOutputFittedBreastMask ) );
    }

    if ( fileOutputVTKSurface.length() > 0 )
    {
      breastMaskSegmentor->SetOutputVTKSurface( niftk::ConcatenatePath( args.dirOutput, 
                                                                        fileOutputVTKSurface ) );
    }

    if ( args.ReadImageFromFile( args.dirOutput, fileBIFs, "BIF image", imBIFs ) )
    {
      breastMaskSegmentor->SetBIFImage( imBIFs );
    }
    else
    {
      breastMaskSegmentor->SetOutputBIFS( niftk::ConcatenatePath( args.dirOutput, fileBIFs ) );
    }        

    breastMaskSegmentor->SetStructuralImage( imT2 );
    breastMaskSegmentor->SetFatSatImage( imT1 );

    breastMaskSegmentor->Execute();

    imSegmentedBreastMask = breastMaskSegmentor->GetSegmentedImage();

    args.WriteImageToFile( fileOutputBreastMask, 
                           "breast mask segmentation image", imSegmentedBreastMask );

  }
      

  // SEGMENT the parenchyma
  // ~~~~~~~~~~~~~~~~~~~~~~

  if ( flgOverwrite || 
       ( ! args.ReadImageFromFile( args.dirOutput, fileOutputParenchyma, 
                                   "breast parenchyma", imParenchyma ) ) )
  {
     
    QStringList argumentsNiftySeg; 
    argumentsNiftySeg 
      << "-v" << "2" 
      << "-bc_order" << "4" 
      << "-nopriors" << "2" 
      << "-in"   << niftk::ConcatenatePath( args.dirOutput, fileI01_T1_BiasFieldCorr ).c_str()
      << "-mask" << niftk::ConcatenatePath( args.dirOutput, fileOutputBreastMask ).c_str()
      << "-out"  << niftk::ConcatenatePath( args.dirOutput, fileOutputParenchyma ).c_str();

    message << std::endl << "Executing parenchyma segmentation (QProcess): "
            << std::endl << "   " << args.progSegEM;
    for(int i=0;i<argumentsNiftySeg.size();i++)
    {
      message << " " << argumentsNiftySeg[i].toStdString();
    }
    message << std::endl << std::endl;
    args.PrintMessage( message );
    
    QProcess callSegEM;
    QString outSegEM;
    
    callSegEM.setProcessChannelMode( QProcess::MergedChannels );
    callSegEM.start( args.progSegEM.c_str(), argumentsNiftySeg );
    
    bool flgFinished = callSegEM.waitForFinished( 600000 ); // Wait 10 mins
    
    outSegEM = callSegEM.readAllStandardOutput();
    
    message << outSegEM.toStdString();
    
    if ( ! flgFinished )
    {
      message << "ERROR: Could not execute: " << args.progSegEM << " ( " 
              << callSegEM.errorString().toStdString() << " )" << std::endl;
      args.PrintErrorAndExit( message );
    }
    
    args.PrintMessage( message );
    
    callSegEM.close();
    
    args.ReadImageFromFile( args.dirOutput, fileOutputParenchyma, 
                            "breast parenchyma", imParenchyma );
  }
  
  iTask++;
  progress = iTask/nTasks;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;


  // Load the ADC image
  // ~~~~~~~~~~~~~~~~~~

  std::string fileRegisteredADC = CreateRegisteredFilename( fileRegistrationTarget,
                                                            fileI00_ADC,
                                                            std::string( "NonRigidTo" ) );

  

  if ( flgOverwrite || 
       ( ! args.ReadImageFromFile( args.dirOutput, fileRegisteredADC, 
                                   std::string( "registered '") +
                                   args.strSeriesDescADC + "' image", 
                                   imADC ) ) )
  {        
    if ( flgOverwrite || 
         ( ! args.ReadImageFromFile( args.dirOutput, fileI00_ADC, 
                                     std::string( "ADC '" ) + args.strSeriesDescADC 
                                     + "' image", imADC ) ) )
    {
      if ( fileNamesADC.size() > 0 )
      {
            
        SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
        seriesReader->SetFileNames( fileNamesADC );

        message << std::endl << "Reading '" << args.strSeriesDescADC << "' image" << std::endl;  
        args.PrintMessage( message );

        seriesReader->UpdateLargestPossibleRegion();

        imADC = seriesReader->GetOutput();
        imADC->DisconnectPipeline();
            
        args.WriteImageToFile( fileI00_ADC, 
                               std::string( "ADC '" ) + args.strSeriesDescADC +
                               "' image", imADC );
      }
    }
      
    // Register the ADC image to the T1W image

    if ( ! ( RegisterImages( fileRegistrationTarget,
                             fileI00_ADC,
                             fileRegisteredADC,
                             args ) 
             &&
             args.ReadImageFromFile( args.dirOutput, fileRegisteredADC, 
                                     std::string( "registered '") +
                                     args.strSeriesDescADC + "' image", 
                                     imADC ) ) )
    {
      imADC = 0;
    }
  }
  
  iTask++;
  progress = iTask/nTasks;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;


  // Create the DCE images
  // ~~~~~~~~~~~~~~~~~~~~~

  SeriesDCEMapType::iterator itSeriesDCE;

  for ( itSeriesDCE=seriesDCE.begin(); itSeriesDCE!=seriesDCE.end(); ++itSeriesDCE )
  {

    message << std::endl << "DCE '" << args.strSeriesDescDCE << "' Series: " 
            << itSeriesDCE->first << std::endl;

    args.PrintMessage( message );


    fileNames = itSeriesDCE->second;

    if ( args.flgDebug )
    {
      for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) 
      {
        message << "      " << *iterFilenames << std::endl;        
        args.PrintMessage( message );
      }
    }

    ss.str( "" );
    ss << std::setw( 3 ) << std::setfill( '0' ) << itSeriesDCE->first;

    std::string fileI00_DCE = std::string( "I00_" ) + strSeriesDescDCE + "_" + ss.str() + ".nii";

    if ( flgCompression )
    {
      fileI00_DCE.append( ".gz" );
    }


    std::string fileRegisteredDCE = CreateRegisteredFilename( fileRegistrationTarget,
                                                              fileI00_DCE,
                                                              std::string( "NonRigidTo" ) );

  

    if ( flgOverwrite || 
         ( ! args.ReadImageFromFile( args.dirOutput, fileRegisteredDCE, 
                                     std::string( "registered '") +
                                     args.strSeriesDescDCE + "' image", 
                                     imDCE ) ) )
    {        
      if ( flgOverwrite || 
           ( ! args.ReadImageFromFile( args.dirOutput, fileI00_DCE, 
                                       std::string( "DCE '" ) + args.strSeriesDescDCE 
                                       + "' image", imDCE ) ) )
      {
        if ( fileNames.size() > 0 )
        {
            
          fileNamesDCEImages.insert( std::pair< unsigned int, 
                                     std::string >( itSeriesDCE->first, 
                                                    fileI00_DCE ) );

          SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
          seriesReader->SetFileNames( fileNames );

          message << std::endl << "Reading '" << args.strSeriesDescDCE << "' image" << std::endl;  
          args.PrintMessage( message );

          seriesReader->UpdateLargestPossibleRegion();

          imDCE = seriesReader->GetOutput();
          imDCE->DisconnectPipeline();
            
          args.WriteImageToFile( fileI00_DCE, 
                                 std::string( "DCE '" ) + args.strSeriesDescDCE +
                                 "' image", imDCE );
        }
      }
      
      // Register the DCE image to the T1W image

      if ( ! ( RegisterImages( fileRegistrationTarget,
                               fileI00_DCE,
                               fileRegisteredDCE,
                               args ) 
               &&
               args.ReadImageFromFile( args.dirOutput, fileRegisteredDCE, 
                                       std::string( "registered '") +
                                       args.strSeriesDescDCE + "' image", 
                                       imDCE ) ) )
      {
        imDCE = 0;
      }
    }
  
    iTask++;
    progress = iTask/nTasks;
    std::cout << "<filter-progress>" << std::endl
              << progress << std::endl
              << "</filter-progress>" << std::endl;
  }
  

  // Delete unwanted images
  // ~~~~~~~~~~~~~~~~~~~~~~

  if ( ! ( flgDebug || flgSaveImages ) )
  {
    args.DeleteFile( fileBIFs );

    args.DeleteFile( fileOutputSmoothedStructural );
    args.DeleteFile( fileOutputSmoothedFatSat );
    args.DeleteFile( fileOutputClosedStructural );

    args.DeleteFile( fileOutputMaxImage );
    args.DeleteFile( fileOutputCombinedHistogram );
    args.DeleteFile( fileOutputRayleigh );
    args.DeleteFile( fileOutputFreqLessBgndCDF );
    args.DeleteFile( fileOutputBackground );
    args.DeleteFile( fileOutputSkinElevationMap );
    args.DeleteFile( fileOutputGradientMagImage );
    args.DeleteFile( fileOutputSpeedImage );
    args.DeleteFile( fileOutputFastMarchingImage );
    args.DeleteFile( fileOutputPectoral );
    args.DeleteFile( fileOutputChestPoints );
    args.DeleteFile( fileOutputPectoralSurfaceMask );

    args.DeleteFile( fileOutputPectoralSurfaceVoxels );

    args.DeleteFile( fileOutputFittedBreastMask );
  }

  
  return EXIT_SUCCESS;
}
 
 

