/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

/*!
 * \file niftkBreastDensityFromMRIs.cxx 
 * \page niftkBreastDensityFromMRIs
 * \section niftkBreastDensityFromMRIsSummary niftkBreastDensityFromMRIs
 * 
 * Compute breast density for directories of DICOM MR images
 *
 */


#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

#include <QProcess>
#include <QString>
#include <QDebug>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <niftkCSVRow.h>
#include <niftkEnvironmentHelper.h>

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
#include <itkBreastMaskSegmForBreastDensity.h>
#include <itkITKImageToNiftiImage.h>
#include <itkRescaleImageUsingHistogramPercentilesFilter.h>
#include <itkOrientImageFilter.h>
#include <itkConversionUtils.h>
#include <itkTransformFileReader.h>
#include <itkTransformFactoryBase.h>
#include <itkAffineTransform.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkIsImageBinary.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkImageDuplicator.h>

#include <niftkBreastDensityFromMRIsCLP.h>

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
typedef itk::Image< unsigned char, Dimension > MaskImageType;


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
// GetOrientationInfo()
// -------------------------------------------------------------------------

std::string GetOrientationInfo( ImageType::Pointer image )
{
  itk::SpatialOrientationAdapter adaptor;
  ImageType::DirectionType direction;

  for (unsigned int i = 0; i < Dimension; i++)
  {
    for (unsigned int j = 0; j < Dimension; j++)
    {
      direction[i][j] = image->GetDirection()[i][j];
    }
  }

  return itk::ConvertSpatialOrientationToString(adaptor.FromDirectionCosines(direction));
}


// -------------------------------------------------------------------------
// PrintOrientationInfo()
// -------------------------------------------------------------------------

void PrintOrientationInfo( ImageType::Pointer image )
{
  std::cout << "ITK orientation: " << GetOrientationInfo( image ) << std::endl;
}


// -------------------------------------------------------------------------
// ReorientateImage()
// -------------------------------------------------------------------------

ImageType::Pointer ReorientateImage( ImageType::Pointer inImage, 
                                     std::string desiredOrientation )
{
  std::string inOrientation = GetOrientationInfo( inImage );
  std::cout << "Input ITK orientation: " << inOrientation << std::endl;

  if ( inOrientation.compare( desiredOrientation ) == 0 )
  {
    return inImage;
  }
  
  typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;

  OrientImageFilterType::FlipAxesArrayType flipAxes;
  OrientImageFilterType::PermuteOrderArrayType permuteAxes;

  OrientImageFilterType::Pointer orienter = OrientImageFilterType::New();

  orienter->UseImageDirectionOn();
  orienter->SetDesiredCoordinateOrientation( itk::ConvertStringToSpatialOrientation( desiredOrientation.c_str() ) );

  orienter->SetInput( inImage );

  try
  {
    orienter->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to reorientate image: " << err << std::endl; 
    return 0;
  }                

  flipAxes = orienter->GetFlipAxes();
  permuteAxes = orienter->GetPermuteOrder();

  std::cout << std::endl
            << "Permute Axes: " << permuteAxes << std::endl
            << "Flip Axes: "    << flipAxes << std::endl << std::endl;

  ImageType::Pointer reorientatedImage = orienter->GetOutput();
  reorientatedImage->DisconnectPipeline();
    
  std::cout << "Output ITK orientation: " << GetOrientationInfo( reorientatedImage ) << std::endl;

  return reorientatedImage;
};


// -------------------------------------------------------------------------
// class InputParameters
// -------------------------------------------------------------------------

class InputParameters
{

public:

  bool flgVerbose;
  bool flgRegisterImages;
  bool flgSaveImages;
  bool flgDebug;
  bool flgCompression;
  bool flgOverwrite;

  bool flgDoNotBiasFieldCorrectT1w;
  bool flgDoNotBiasFieldCorrectT2w;

  bool flgExcludeAxilla;
  bool flgCropFit;
  float coilCropDistance;

  std::string dirInput;
  std::string dirOutput;

  std::string fileLog;
  std::string fileT1wOutputCSV;
  std::string fileT2wOutputCSV;

  std::string dirSubMRI;  
  std::string dirSubData;  
  std::string dirPrefix;  

  std::string strSeriesDescStructuralT2;
  std::string strSeriesDescFatSatT1;
  std::string strSeriesDescDixonWater;
  std::string strSeriesDescDixonFat;

  std::string progSegEM;
  std::string progRegAffine;
  std::string progRegNonRigid;
  std::string progRegResample;

  QStringList argsSegEM;
  QStringList argsRegAffine;
  QStringList argsRegNonRigid;
  QStringList argsRegResample;

  std::ofstream *foutLog;
  std::ofstream *foutOutputT1wCSV;
  std::ofstream *foutOutputT2wCSV;

  std::ostream *newCout;

  typedef tee_device<ostream, ofstream> TeeDevice;
  typedef stream<TeeDevice> TeeStream;

  TeeDevice *teeDevice;
  TeeStream *teeStream;

  InputParameters( TCLAP::CmdLine &commandLine, 
                   bool verbose, bool flgRegister, bool flgSave, 
                   bool compression, bool debug, bool overwrite,
                   bool excludeAxilla, bool cropFit, float coilCropDist,
                   bool doNotBiasFieldCorrectT1w, bool doNotBiasFieldCorrectT2w,
                   std::string subdirMRI, std::string subdirData, 
                   std::string prefix, 
                   std::string dInput,
                   std::string logfile, 
                   std::string csvfileT1w,
                   std::string csvfileT2w,
                   std::string strStructuralT2,
                   std::string strFatSatT1,
                   std::string strDixonWater,
                   std::string strDixonFat,
                   std::string segEM,       QStringList aSegEM,
                   std::string regAffine,   QStringList aRegAffine,
                   std::string regNonRigid, QStringList aRegNonRigid,
                   std::string regResample, QStringList aRegResample ) {

    std::stringstream message;

    flgVerbose = verbose;
    flgRegisterImages = flgRegister;
    flgSaveImages = flgSave;
    flgDebug = debug;
    flgCompression = compression;
    flgOverwrite = overwrite;

    flgDoNotBiasFieldCorrectT1w = doNotBiasFieldCorrectT1w;
    flgDoNotBiasFieldCorrectT2w = doNotBiasFieldCorrectT2w;

    flgExcludeAxilla = excludeAxilla;
    flgCropFit = cropFit;
    coilCropDistance = coilCropDist;

    dirSubMRI  = subdirMRI;
    dirSubData = subdirData;
    dirPrefix  = prefix;
    dirInput   = dInput;

    fileLog = logfile;
    fileT1wOutputCSV = csvfileT1w;
    fileT2wOutputCSV = csvfileT2w;

    strSeriesDescStructuralT2 = strStructuralT2;
    strSeriesDescFatSatT1 = strFatSatT1;
    strSeriesDescDixonWater = strDixonWater;
    strSeriesDescDixonFat = strDixonFat;

    progSegEM       = segEM;
    progRegAffine   = regAffine;
    progRegNonRigid = regNonRigid;
    progRegResample = regResample;

    argsSegEM       = aSegEM;
    argsRegAffine   = aRegAffine;
    argsRegNonRigid = aRegNonRigid;
    argsRegResample = aRegResample;

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

    if ( fileT1wOutputCSV.length() > 0 )
    {
      foutOutputT1wCSV = new std::ofstream( fileT1wOutputCSV.c_str() );

      if ((! *foutOutputT1wCSV) || foutOutputT1wCSV->bad()) {
        message << "Could not open file: " << fileT1wOutputCSV << std::endl;
        PrintErrorAndExit( message );
      }
    }
    else
    {
      foutOutputT1wCSV = 0;
    }

    if ( fileT2wOutputCSV.length() > 0 )
    {
      foutOutputT2wCSV = new std::ofstream( fileT2wOutputCSV.c_str() );

      if ((! *foutOutputT2wCSV) || foutOutputT2wCSV->bad()) {
        message << "Could not open file: " << fileT2wOutputCSV << std::endl;
        PrintErrorAndExit( message );
      }
    }
    else
    {
      foutOutputT2wCSV = 0;
    }

    
    if ( dirInput.length() == 0 )
    {
      commandLine.getOutput()->usage( commandLine );
      message << "The input directory must be specified" << std::endl;
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

    if ( foutOutputT1wCSV )
    {
      foutOutputT1wCSV->close();
      delete foutOutputT1wCSV;
    }   

    if ( foutOutputT2wCSV )
    {
      foutOutputT2wCSV->close();
      delete foutOutputT2wCSV;
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
            << "Examining directory: " << dirInput << std::endl 
            << std::endl
            << std::boolalpha
            << "Verbose output?: "                        << flgVerbose        << std::endl
            << "Register the images?: "                   << flgRegisterImages << std::endl
            << "Save images?: "                           << flgSaveImages     << std::endl
            << "Compress images?: "                       << flgCompression    << std::endl
            << "Debugging output?: "                      << flgDebug          << std::endl
            << "Overwrite previous results?: "            << flgOverwrite      << std::endl       
            << std::noboolalpha
            << std::endl
            << "Input MRI sub-directory: " << dirSubMRI << std::endl
            << "Output data sub-directory: " << dirSubData << std::endl
            << "Study directory prefix: " << dirPrefix << std::endl
            << std::endl
            << "Output log file: " << fileLog << std::endl
            << "Output T1w csv file: " << fileT1wOutputCSV << std::endl
            << "Output T2w csv file: " << fileT2wOutputCSV << std::endl
            << std::endl
            << std::boolalpha
            << "Exclude the axilla?: " << flgExcludeAxilla << std::endl       
            << "Clip segmentation with fitted surface?: " << flgCropFit << std::endl       
            << std::noboolalpha
            << "MR coil coronal crop distance: " << coilCropDistance << std::endl       
            << std::endl
            << "Structural series description: " << strSeriesDescStructuralT2 << std::endl
            << "Complementary image series description" << strSeriesDescFatSatT1 << std::endl
            << "DIXON water image series description: " << strSeriesDescDixonWater << std::endl
            << "DIXON fat image series description: " << strSeriesDescDixonFat << std::endl
            << std::endl
            << "Segmentation executable: "
            << progSegEM       << " " << argsSegEM.join(" ").toStdString() << std::endl
            << "Affine registration executable: "
            << progRegAffine   << " " << argsRegAffine.join(" ").toStdString() << std::endl
            << "Non-rigid registration executable: "
            << progRegNonRigid << " " << argsRegNonRigid.join(" ").toStdString() << std::endl
            << "Resampling executable: "
            << progRegResample << " " << argsRegResample.join(" ").toStdString() << std::endl
            << std::endl;

    PrintMessage( message );
  }
    
  void PrintMessage( std::stringstream &message ) {

    std::cout << message.str();
    message.str( "" );
    if ( teeStream )
    {
      teeStream->flush();
    }
  }
    
  void PrintError( std::stringstream &message ) {

    std::cerr << "ERROR: " << message.str();
    message.str( "" );
    if ( teeStream )
    {
      teeStream->flush();
    }
  }
    
  void PrintErrorAndExit( std::stringstream &message ) {

    PrintError( message );

    exit( EXIT_FAILURE );
  }
    
  void PrintWarning( std::stringstream &message ) {

    std::cerr << "WARNING: " << message.str();
    message.str( "" );
    if ( teeStream )
    {
      teeStream->flush();
    }
  }

  void PrintTag( const DictionaryType &dictionary, 
                 std::string entryId ) {

    std::stringstream message;
    std::string tagID;
        
    std::string tagValue = GetTag( dictionary, entryId, &tagID );
          
    message << "   " << tagID <<  ": " << tagValue << std::endl;
    PrintMessage( message );
  }

  bool ReadImageFromFile( std::string dirInput, std::string filename, 
                          std::string description, ImageType::Pointer &image ) {
  
    std::stringstream message;
    std::string fileInput = niftk::ConcatenatePath( dirInput, filename );
           
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

  bool ReadImageFromFile( std::string dirInput, std::string filename, 
                          std::string description, nifti_image *&image ) {
  
    std::stringstream message;
    std::string fileInput = niftk::ConcatenatePath( dirInput, filename );

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

    if ( itk::IsImageBinary< ImageType >( image ) )
    {
      typedef unsigned char OutputPixelType;
      typedef itk::Image< OutputPixelType, ImageType::ImageDimension> OutputImageType;

      typedef itk::RescaleIntensityImageFilter< ImageType, OutputImageType > CastFilterType;

      CastFilterType::Pointer caster = CastFilterType::New();

      caster->SetInput( image );
      caster->SetOutputMinimum(   0 );
      caster->SetOutputMaximum( 255 );
      caster->Update();

      OutputImageType::Pointer imOut = caster->GetOutput();

      itk::WriteImageToFile< OutputImageType >( fileOutput, imOut );      
    }
    else
    {
      itk::WriteImageToFile< ImageType >( fileOutput, image );
    }
  }

  void WriteImageToFile( std::string filename, 
                         std::string description, MaskImageType::Pointer image ) {
  
    std::stringstream message;
    std::string fileOutput = niftk::ConcatenatePath( dirOutput, filename );
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    itk::WriteImageToFile< MaskImageType >( fileOutput, image );
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
    message << "Failed to read " << fileAffineTransformFullPath << std::endl;
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
    message << "Could not cast: " << fileAffineTransformFullPath << std::endl;
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
                             std::string &fileNonRigidTransform,
                             InputParameters &args ) 
{
  std::stringstream message;
  
  fileNonRigidTransform = niftk::ModifyImageFileSuffix( fileRegistered, 
                                                        std::string( "_Transform.nii" ) );

  if ( args.flgCompression ) fileNonRigidTransform.append( ".gz" );

  QStringList argsRegNonRigid = args.argsRegNonRigid; 

  argsRegNonRigid
    << "-target" << niftk::ConcatenatePath( args.dirOutput, fileTarget.c_str() ).c_str()
    << "-source" << niftk::ConcatenatePath( args.dirOutput, fileSource.c_str() ).c_str()
    << "-res"    << niftk::ConcatenatePath( args.dirOutput, fileRegistered.c_str() ).c_str()
    << "-aff"    << niftk::ConcatenatePath( args.dirOutput, fileAffineTransform.c_str() ).c_str()
    << "-cpp"    << niftk::ConcatenatePath( args.dirOutput, fileNonRigidTransform.c_str() ).c_str();
  
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

  std::string fileNonRigidTransform;

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
                                   fileNonRigidTransform,
                                   args ) );
};


// -------------------------------------------------------------------------
// ResampleImages()
// -------------------------------------------------------------------------

bool ResampleImages( std::string fileTarget, 
                     std::string fileResampleInput, 
                     std::string fileResampleOutput,
                     std::string fileNonRigidTransform,
                     std::string resamplingInterpolation,
                     InputParameters &args ) 
{
  std::stringstream message;
  
  QStringList argsRegResample; 
  argsRegResample 
    << resamplingInterpolation.c_str()
    << "-ref" << niftk::ConcatenatePath( args.dirOutput, fileTarget.c_str() ).c_str()
    << "-flo" << niftk::ConcatenatePath( args.dirOutput, fileResampleInput.c_str() ).c_str()
    << "-res"    << niftk::ConcatenatePath( args.dirOutput, fileResampleOutput.c_str() ).c_str()
    << "-cpp"    << niftk::ConcatenatePath( args.dirOutput, fileNonRigidTransform.c_str() ).c_str();
  
  message << std::endl << "Executing registration resampling (QProcess): "
          << std::endl << "   " << args.progRegResample;
  for(int i=0;i<argsRegResample.size();i++)
  {
    message << " " << argsRegResample[i].toStdString();
  }
  message << std::endl << std::endl;
  args.PrintMessage( message );
  
  QProcess callRegResample;
  QString outRegResample;
  
  callRegResample.setProcessChannelMode( QProcess::MergedChannels );
  callRegResample.start( args.progRegResample.c_str(), argsRegResample );
  
  bool flgFinished = callRegResample.waitForFinished( 3600000 ); // Wait one hour
  
  outRegResample = callRegResample.readAllStandardOutput();
  
  message << outRegResample.toStdString();
  
  if ( ! flgFinished )
  {
    message << "ERROR: Could not execute: " << args.progRegResample << " ( " 
            << callRegResample.errorString().toStdString() << " )" << std::endl;
    args.PrintMessage( message );
    return false;
  }
  
  args.PrintMessage( message );
  
  callRegResample.close();

  return true;
};


// -------------------------------------------------------------------------
// RegisterAndResample()
// -------------------------------------------------------------------------

bool RegisterAndResample( std::string fileTarget, 
                          std::string fileSource, 
                          std::string fileResampleInput,
                          std::string fileResampleOutput,
                          std::string resamplingInterpolation,
                          InputParameters &args ) 
{
  std::string fileAffineTransform;
  std::string fileAffineRegistered = CreateRegisteredFilename( fileTarget,
                                                               fileSource,
                                                               std::string( "AffineTo" ) );

  std::string fileNonRigidTransform;
  std::string fileNonRigidRegistered = CreateRegisteredFilename( fileTarget,
                                                                 fileSource,
                                                                 std::string( "NonRigidTo" ) );

  return ( AffineRegisterImages( fileTarget, 
                                 fileSource, 
                                 fileAffineRegistered,
                                 fileAffineTransform, 
                                 args )
           &&
           NonRigidRegisterImages( fileTarget, 
                                   fileSource, 
                                   fileNonRigidRegistered,
                                   fileAffineTransform,
                                   fileNonRigidTransform,
                                   args )
           &&
           ResampleImages( fileTarget, 
                           fileResampleInput,
                           fileResampleOutput,
                           fileNonRigidTransform,
                           resamplingInterpolation,
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
// NaiveParenchymaSegmentation()
// -------------------------------------------------------------------------

void NaiveParenchymaSegmentation( std::string label,
                                  InputParameters &args, 
                        
                                  ImageType::Pointer &imSegmentedBreastMask,
                                  ImageType::Pointer &image,
                                  
                                  bool flgFatIsBright,

                                  float &nLeftVoxels,
                                  float &nRightVoxels,                                  

                                  float &totalDensity,
                                  float &leftDensity,
                                  float &rightDensity,

                                  std::string fileOutputParenchyma )
{
  float minFraction = 0.02;

  std::stringstream message;

  ImageType::Pointer imParenchyma = 0;

  typedef itk::ImageDuplicator< ImageType > DuplicatorType; 
    
  DuplicatorType::Pointer duplicator = DuplicatorType::New();

  duplicator->SetInputImage( image );
  duplicator->Update();

  imParenchyma = duplicator->GetOutput();
  imParenchyma->DisconnectPipeline();

  imParenchyma->FillBuffer( 0. );

  nLeftVoxels = 0;
  nRightVoxels = 0;

  totalDensity = 0.;
  leftDensity = 0.;
  rightDensity = 0.;

  float meanOfHighProbIntensities = 0.;
  float meanOfLowProbIntensities = 0.;

  float nHighProbIntensities = 0.;
  float nLowProbIntensities = 0.;

  itk::ImageRegionIteratorWithIndex< ImageType > 
    itMask( imSegmentedBreastMask, imSegmentedBreastMask->GetLargestPossibleRegion() );

  itk::ImageRegionIterator< ImageType > 
    itSegmentation( imParenchyma, imParenchyma->GetLargestPossibleRegion() );

  itk::ImageRegionConstIterator< ImageType > 
    itImage( image, image->GetLargestPossibleRegion() );

  
  // Compute the range of intensities inside the mask
  
  float minIntensity = std::numeric_limits< float >::max();
  float maxIntensity = -std::numeric_limits< float >::max();

  for ( itMask.GoToBegin(), itImage.GoToBegin();
        ! itMask.IsAtEnd();
        ++itMask, ++itImage )
  {
    if ( itMask.Get() )
    {
      if ( itImage.Get() > maxIntensity )
      {
        maxIntensity = itImage.Get();
      }

      if ( itImage.Get() < minIntensity )
      {
        minIntensity = itImage.Get();
      }
    }
  }

  message  << std::endl
           << "Range of " << label << " is from: " 
           << minIntensity << " to: " << maxIntensity << std::endl;
  args.PrintMessage( message );
  

  // Compute 1st and 99th percentiles of the image from the image histogram

  unsigned int nBins = static_cast<unsigned int>( maxIntensity - minIntensity + 0.5 ) + 1;

  itk::Array< float > histogram( nBins );
    
  histogram.Fill( 0 );

  float nPixels = 0;
  float flIntensity;

  for ( itImage.GoToBegin(), itMask.GoToBegin();
        ! itImage.IsAtEnd();
        ++itImage, ++itMask )
  {
    if ( itMask.Get() )
    {
      flIntensity = itImage.Get() - minIntensity;
      
      if ( flIntensity < 0. )
      {
        flIntensity = 0.;
      }
      
      if ( flIntensity > static_cast<float>( nBins - 1 ) )
      {
        flIntensity = static_cast<float>( nBins - 1 );
      }
      
      nPixels++;
      histogram[ static_cast<unsigned int>( flIntensity ) ] += 1.;
    }
  }
    
  float sumProbability = 0.;
  unsigned int intensity;

  float pLowerBound = 0.;
  float pUpperBound = 0.;

  bool flgLowerBoundFound = false;
  bool flgUpperBoundFound = false;


  for ( intensity=0; intensity<nBins; intensity++ )
  {
    histogram[ intensity ] /= nPixels;
    sumProbability += histogram[ intensity ];

    if ( ( ! flgLowerBoundFound ) && ( sumProbability >= minFraction ) )
    {
      pLowerBound = intensity;
      flgLowerBoundFound = true;
    }

    if ( ( ! flgUpperBoundFound ) && ( sumProbability >= (1. - minFraction) ) )
    {
      pUpperBound = intensity;
      flgUpperBoundFound = true;
    }
    
    if ( args.flgDebug )
    {
      std::cout << std::setw( 18 ) << intensity << " " 
                << std::setw( 18 ) << histogram[ intensity ]  << " " 
                << std::setw( 18 ) << sumProbability << std::endl;
    }
  }
  
  message << std::endl
          << label << " density lower bound: " << pLowerBound 
          << " ( " << minFraction*100. << "% )" << std::endl
          << " upper bound: " << pUpperBound 
          << " ( " << (1. - minFraction)*100. << "% )" << std::endl;
  args.PrintMessage( message );
  

  // Compute the density

  ImageType::SpacingType spacing = imParenchyma->GetSpacing();

  float voxelVolume = spacing[0]*spacing[1]*spacing[2];

  ImageType::RegionType region;
  region = imSegmentedBreastMask->GetLargestPossibleRegion();

  ImageType::SizeType lateralSize;
  lateralSize = region.GetSize();
  lateralSize[0] = lateralSize[0]/2;

  ImageType::IndexType idx;
   
  for ( itMask.GoToBegin(), itSegmentation.GoToBegin(), itImage.GoToBegin();
        ! itMask.IsAtEnd();
        ++itMask, ++itSegmentation, ++itImage )
  {
    if ( itMask.Get() )
    {
      idx = itMask.GetIndex();

      flIntensity = ( itImage.Get() - pLowerBound )/( pUpperBound - pLowerBound );
      
      if ( flIntensity < 0. )
      {
        itSegmentation.Set( 0. );
      }
      else if ( flIntensity > 1. )
      {
        itSegmentation.Set( 1. );
      }
      else
      {
        itSegmentation.Set( flIntensity );
      }

      //std::cout << idx << " " << itImage.Get() << " -> " << flIntensity << std::endl;

      // Left breast

      if ( idx[0] < (int) lateralSize[0] )
      {
        nLeftVoxels++;
        leftDensity += flIntensity;
      }

      // Right breast
      
      else 
      {
        nRightVoxels++;
        rightDensity += flIntensity;
      }

      // Both breasts

      totalDensity += flIntensity;

      // Ensure we have the polarity correct by calculating the
      // mean intensities of each class

      if ( flIntensity > 0.5 )
      {
        meanOfHighProbIntensities += itImage.Get();
        nHighProbIntensities++;
      }
      else
      {
        meanOfLowProbIntensities += itImage.Get();
        nLowProbIntensities++;
      }              
    }
  }

  leftDensity /= nLeftVoxels;
  rightDensity /= nRightVoxels;
  totalDensity /= ( nLeftVoxels + nRightVoxels);

  // Calculate the mean intensities of each class
  
  if ( nHighProbIntensities > 0. )
  {
    meanOfHighProbIntensities /= nHighProbIntensities;
  }
  
  if ( nLowProbIntensities > 0. )
  {
    meanOfLowProbIntensities /= nLowProbIntensities;
  }

  message  << std::endl
           << "Mean intensity of high probability class = " << meanOfHighProbIntensities
           << " ( " << nHighProbIntensities << " voxels )" << std::endl
           << "Mean intensity of low probability class = " << meanOfLowProbIntensities
           << " ( " << nLowProbIntensities << " voxels )" << std::endl;
  args.PrintMessage( message );

  // Fat should be high intensity in the T2 image so if the dense
  // region (high prob) has a high intensity then it is probably fat
  // and we need to invert the density, whereas the opposite is true
  // for the fat-saturated T1w VIBE image.
  
  if ( (     flgFatIsBright   && ( meanOfHighProbIntensities > meanOfLowProbIntensities ) ) ||
       ( ( ! flgFatIsBright ) && ( meanOfHighProbIntensities < meanOfLowProbIntensities ) ) )
  {
    message << "Inverting the density estimation" << std::endl;
    args.PrintWarning( message );
    
    leftDensity  = 1. - leftDensity;
    rightDensity = 1. - rightDensity;
    totalDensity = 1. - totalDensity;
    
    typedef itk::InvertIntensityBetweenMaxAndMinImageFilter< ImageType > InvertFilterType;
    InvertFilterType::Pointer invertFilter = InvertFilterType::New();
    invertFilter->SetInput( imParenchyma );

    typedef itk::MaskImageFilter< ImageType, ImageType > MaskFilterType;
    MaskFilterType::Pointer maskFilter = MaskFilterType::New();

    maskFilter->SetInput( invertFilter->GetOutput() );
    maskFilter->SetMaskImage( imSegmentedBreastMask );
    maskFilter->Update();

    ImageType::Pointer imInverted = maskFilter->GetOutput();
    imInverted->DisconnectPipeline();
    imParenchyma = imInverted;
  }

  std::string fileOut = niftk::ModifyImageFileSuffix( fileOutputParenchyma, 
                                                      std::string( "_Naive.nii" ) );

  if ( args.flgCompression ) fileOut.append( ".gz" );

  args.WriteImageToFile( fileOut, 
                         std::string( "naive '" ) + label +
                         "' parenchyma image", imParenchyma );


  float leftBreastVolume = nLeftVoxels*voxelVolume;
  float rightBreastVolume = nRightVoxels*voxelVolume;

  message << label << " Naive - Number of left breast voxels: " << nLeftVoxels << std::endl
          << label << " Naive - Volume of left breast: " << leftBreastVolume << " mm^3" << std::endl
          << label << " Naive - Density of left breast (fraction of glandular tissue): " << leftDensity 
          << std::endl << std::endl
    
          << label << " Naive - Number of right breast voxels: " << nRightVoxels << std::endl
          << label << " Naive - Volume of right breast: " << rightBreastVolume << " mm^3" << std::endl
          << label << " Naive - Density of right breast (fraction of glandular tissue): " << rightDensity 
          << std::endl << std::endl
    
          << label << " Naive - Total number of breast voxels: " 
          << nLeftVoxels + nRightVoxels << std::endl
          << label << " Naive - Total volume of both breasts: " 
          << leftBreastVolume + rightBreastVolume << " mm^3" << std::endl
          << label << " Naive - Combined density of both breasts (fraction of glandular tissue): " 
          << totalDensity << std::endl << std::endl;
  args.PrintMessage( message );
};


// -------------------------------------------------------------------------
// SegmentParenchyma()
// -------------------------------------------------------------------------

bool SegmentParenchyma( std::string label,

                        InputParameters &args, 
                        std::ofstream *foutOutputCSV,
                        std::string fileOutputCSV,
                        bool &flgVeryFirstRow,
                        bool flgFatIsBright,

                        std::string dirBaseName,
                        std::string fileInputImage,
                        std::string fileOutputBreastMask,
                        std::string fileOutputParenchyma,
                        std::string fileDensityMeasurements,

                        ImageType::Pointer &imSegmentedBreastMask,
                        ImageType::Pointer &image )
{
  std::stringstream message;

  ImageType::Pointer imParenchyma = 0;
    

  if ( args.flgOverwrite || 
       ( ! args.ReadImageFromFile( args.dirOutput, fileOutputParenchyma, 
                                   "breast parenchyma", imParenchyma ) ) )
  {
          
    // QProcess call to seg_EM

    QStringList argumentsNiftySeg = args.argsSegEM; 

    argumentsNiftySeg 
      << "-in"   << niftk::ConcatenatePath( args.dirOutput, fileInputImage ).c_str()
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

    bool flgFinished = callSegEM.waitForFinished( 3600000 ); // Wait one hour

    outSegEM = callSegEM.readAllStandardOutput();

    message << outSegEM.toStdString();

    if ( ! flgFinished )
    {
      message << "ERROR: Could not execute: " << args.progSegEM << " ( " 
              << callSegEM.errorString().toStdString() << " )" << std::endl;
      args.PrintMessage( message );

      return false;
    }

    args.PrintMessage( message );

    callSegEM.close();

    args.ReadImageFromFile( args.dirOutput, fileOutputParenchyma, 
                            "breast parenchyma", imParenchyma );
  }


  // Compute the breast density
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( imParenchyma && imSegmentedBreastMask )
  {
    float nLeftVoxels = 0;
    float nRightVoxels = 0;

    float totalDensity = 0.;
    float leftDensity = 0.;
    float rightDensity = 0.;

    float meanOfHighProbIntensities = 0.;
    float meanOfLowProbIntensities = 0.;

    float nHighProbIntensities = 0.;
    float nLowProbIntensities = 0.;

    itk::ImageRegionIteratorWithIndex< ImageType > 
      itMask( imSegmentedBreastMask, imSegmentedBreastMask->GetLargestPossibleRegion() );

    itk::ImageRegionConstIterator< ImageType > 
      itSegmentation( imParenchyma, imParenchyma->GetLargestPossibleRegion() );

    itk::ImageRegionConstIterator< ImageType > 
      itImage( image, image->GetLargestPossibleRegion() );

    ImageType::SpacingType spacing = imParenchyma->GetSpacing();

    float voxelVolume = spacing[0]*spacing[1]*spacing[2];

    ImageType::RegionType region;
    region = imSegmentedBreastMask->GetLargestPossibleRegion();

    ImageType::SizeType lateralSize;
    lateralSize = region.GetSize();
    lateralSize[0] = lateralSize[0]/2;

    ImageType::IndexType idx;
   
    for ( itMask.GoToBegin(), itSegmentation.GoToBegin(), itImage.GoToBegin();
          ! itMask.IsAtEnd();
          ++itMask, ++itSegmentation, ++itImage )
    {
      if ( itMask.Get() )
      {
        idx = itMask.GetIndex();

        // Left breast

        if ( idx[0] < (int) lateralSize[0] )
        {
          nLeftVoxels++;
          leftDensity += itSegmentation.Get();
        }

        // Right breast

        else 
        {
          nRightVoxels++;
          rightDensity += itSegmentation.Get();
        }

        // Both breasts

        totalDensity += itSegmentation.Get();

        // Ensure we have the polarity correct by calculating the
        // mean intensities of each class

        if ( itSegmentation.Get() > 0.5 )
        {
          meanOfHighProbIntensities += itImage.Get();
          nHighProbIntensities++;
        }
        else
        {
          meanOfLowProbIntensities += itImage.Get();
          nLowProbIntensities++;
        }              
      }
    }

    leftDensity /= nLeftVoxels;
    rightDensity /= nRightVoxels;
    totalDensity /= ( nLeftVoxels + nRightVoxels);

    // Calculate the mean intensities of each class

    if ( nHighProbIntensities > 0. )
    {
      meanOfHighProbIntensities /= nHighProbIntensities;
    }

    if ( nLowProbIntensities > 0. )
    {
      meanOfLowProbIntensities /= nLowProbIntensities;
    }

    message  << std::endl
             << "Mean intensity of high probability class = " << meanOfHighProbIntensities
             << " ( " << nHighProbIntensities << " voxels )" << std::endl
             << "Mean intensity of low probability class = " << meanOfLowProbIntensities
             << " ( " << nLowProbIntensities << " voxels )" << std::endl;
    args.PrintMessage( message );

    // Fat should be high intensity in the T2 image so if the dense
    // region (high prob) has a high intensity then it is probably fat
    // and we need to invert the density, whereas the opposite is true
    // for the fat-saturated T1w VIBE image.

    if ( (     flgFatIsBright   && ( meanOfHighProbIntensities > meanOfLowProbIntensities ) ) ||
         ( ( ! flgFatIsBright ) && ( meanOfHighProbIntensities < meanOfLowProbIntensities ) ) )
    {
      message << "Inverting the density estimation" << std::endl;
      args.PrintWarning( message );
        
      leftDensity  = 1. - leftDensity;
      rightDensity = 1. - rightDensity;
      totalDensity = 1. - totalDensity;

      typedef itk::InvertIntensityBetweenMaxAndMinImageFilter< ImageType > InvertFilterType;
      InvertFilterType::Pointer invertFilter = InvertFilterType::New();
      invertFilter->SetInput( imParenchyma );

      typedef itk::MaskImageFilter< ImageType, ImageType > MaskFilterType;
      MaskFilterType::Pointer maskFilter = MaskFilterType::New();

      maskFilter->SetInput( invertFilter->GetOutput() );
      maskFilter->SetMaskImage( imSegmentedBreastMask );
      maskFilter->Update();

      ImageType::Pointer imInverted = maskFilter->GetOutput();
      imInverted->DisconnectPipeline();
      imParenchyma = imInverted;

      args.WriteImageToFile( fileOutputParenchyma, 
                             std::string( "inverted '" ) + label +
                             "' parenchyma image", imParenchyma );
    }
        
  
    float leftBreastVolume = nLeftVoxels*voxelVolume;
    float rightBreastVolume = nRightVoxels*voxelVolume;

    message << label << " Number of left breast voxels: " << nLeftVoxels << std::endl
            << label << " Volume of left breast: " << leftBreastVolume << " mm^3" << std::endl
            << label << " Density of left breast (fraction of glandular tissue): " << leftDensity 
            << std::endl << std::endl
        
            << label << " Number of right breast voxels: " << nRightVoxels << std::endl
            << label << " Volume of right breast: " << rightBreastVolume << " mm^3" << std::endl
            << label << " Density of right breast (fraction of glandular tissue): " << rightDensity 
            << std::endl << std::endl
        
            << label << " Total number of breast voxels: " 
            << nLeftVoxels + nRightVoxels << std::endl
            << label << " Total volume of both breasts: " 
            << leftBreastVolume + rightBreastVolume << " mm^3" << std::endl
            << label << " Combined density of both breasts (fraction of glandular tissue): " 
            << totalDensity << std::endl << std::endl;
    args.PrintMessage( message );


    // Compute a naive value of the breast density

    float nLeftVoxelsNaive;
    float nRightVoxelsNaive;                                  
      
    float totalDensityNaive;
    float leftDensityNaive;
    float rightDensityNaive;


    NaiveParenchymaSegmentation( label, args, 
                                 imSegmentedBreastMask, image,
                                 flgFatIsBright,
                                 nLeftVoxelsNaive, nRightVoxelsNaive,      
                                 totalDensityNaive, leftDensityNaive, rightDensityNaive,
                                 fileOutputParenchyma );


    if ( fileDensityMeasurements.length() != 0 ) 
    {
      std::string fileOutputDensityMeasurements 
        = niftk::ConcatenatePath( args.dirOutput, fileDensityMeasurements );

      std::ofstream fout( fileOutputDensityMeasurements.c_str() );

      fout.precision(16);

      if ((! fout) || fout.bad()) 
      {
        message << "Could not open file: " << fileDensityMeasurements << std::endl;
        args.PrintError( message );
        return false;
      }

      fout << "Study ID, "
           << label << " Number of left breast voxels, "
           << label << " Volume of left breast (mm^3), "
           << label << " Density of left breast (fraction of glandular tissue), "
      
           << label << " Number of right breast voxels, "
           << label << " Volume of right breast (mm^3), "
           << label << " Density of right breast (fraction of glandular tissue), "
      
           << label << " Total number of breast voxels, "
           << label << " Total volume of both breasts (mm^3), "
           << label << " Combined density of both breasts (fraction of glandular tissue), " 

           << label << " Naive density of left breast (fraction of glandular tissue), "
           << label << " Naive density of right breast (fraction of glandular tissue), "
           << label << " Naive combined density of both breasts (fraction of glandular tissue)" 
           << std::endl;

      fout << dirBaseName << ", "
           << nLeftVoxels << ", "
           << leftBreastVolume << ", "
           << leftDensity << ", "
      
           << nRightVoxels << ", "
           << rightBreastVolume << ", "
           << rightDensity << ", "
      
           << nLeftVoxels + nRightVoxels << ", "
           << leftBreastVolume + rightBreastVolume << ", "
           << totalDensity << ", " 

           << leftDensityNaive << ", "
           << rightDensityNaive << ", "
           << totalDensityNaive 

           << std::endl;
    
      fout.close();

      message  << label << " Density measurements written to file: " 
               << fileOutputDensityMeasurements  << std::endl << std::endl;
      args.PrintMessage( message );
    }

    // Write the data to the main collated csv file

    if ( foutOutputCSV )
    {
      if ( flgVeryFirstRow )    // Include the title row?
      {
        *foutOutputCSV << "Study ID, "
                       << label << " Number of left breast voxels, "
                       << label << " Volume of left breast (mm^3), "
                       << label << " Density of left breast (fraction of glandular tissue), "
              
                       << label << " Number of right breast voxels, "
                       << label << " Volume of right breast (mm^3), "
                       << label << " Density of right breast (fraction of glandular tissue), "
          
                       << label << " Total number of breast voxels, "
                       << label << " Total volume of both breasts (mm^3), "
                       << label << " Combined density of both breasts (fraction of glandular tissue), " 
          
                       << label << " Naive density of left breast (fraction of glandular tissue), "
                       << label << " Naive density of right breast (fraction of glandular tissue), "
                       << label << " Naive combined density of both breasts (fraction of glandular tissue)" 

                       << std::endl;

        flgVeryFirstRow = false;
      }

      *foutOutputCSV << dirBaseName << ", "
                     << nLeftVoxels << ", "
                     << leftBreastVolume << ", "
                     << leftDensity << ", "
            
                     << nRightVoxels << ", "
                     << rightBreastVolume << ", "
                     << rightDensity << ", "
      
                     << nLeftVoxels + nRightVoxels << ", "
                     << leftBreastVolume + rightBreastVolume << ", "
                     << totalDensity << ", "

                     << leftDensityNaive << ", "
                     << rightDensityNaive << ", "
                     << totalDensityNaive 

                     << std::endl;
    }
    else
    {
      message << "Collated csv data file: " << fileOutputCSV 
              << " is not open, data will not be written." << std::endl;
      args.PrintWarning( message );
    }
  }

  return true;
};


// -------------------------------------------------------------------------
// ReadFileCSV
// -------------------------------------------------------------------------

bool ReadFileCSV( InputParameters &args, 
                  std::string fileDensityMeasurements,
                  std::ofstream *foutOutputCSV,
                  bool &flgVeryFirstRow )
{
  std::stringstream message;

  if ( ! foutOutputCSV )
  {
    message << "Output csv stream is not open." << std::endl;
    args.PrintError( message );
    return false;
  }

  std::string fileInputDensityMeasurements  
    = niftk::ConcatenatePath( args.dirOutput, fileDensityMeasurements );

  if ( ! args.flgOverwrite) 
  {
    if ( niftk::FileExists( fileInputDensityMeasurements ) )
    {
      std::ifstream fin( fileInputDensityMeasurements.c_str() );

      if ((! fin) || fin.bad()) 
      {
        message << "Could not open file: " << fileDensityMeasurements << std::endl;
        args.PrintError( message );
        return false;
      }

      message << std::endl << "Reading CSV file: " << fileInputDensityMeasurements << std::endl;
      args.PrintMessage( message );

      niftk::CSVRow csvRow;

      bool flgFirstRowOfThisFile = true;

      while( fin >> csvRow )
      {
        message << csvRow << std::endl;
        args.PrintMessage( message );

        if ( flgFirstRowOfThisFile )
        {
          if ( flgVeryFirstRow )    // Include the title row?
          {
            *foutOutputCSV << csvRow << std::endl;
            flgVeryFirstRow = false;
          }
          flgFirstRowOfThisFile = false;
        }
        else
        {
          *foutOutputCSV << csvRow << std::endl;
        }
      }
        
      return true;
    }
    else
    {
      message << "Density measurements: " << fileInputDensityMeasurements 
              << " not found" << std::endl;
      args.PrintMessage( message );
    }     
  }

  return false;
};


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  bool flgVeryFirstRowT1w = true;
  bool flgVeryFirstRowT2w = true;

  float progress = 0.;
  float iDirectory = 0.;
  float nDirectories;

  std::stringstream message;

  std::vector< std::string > fileNamesStructuralT2;
  std::vector< std::string > fileNamesFatSatT1;
  std::vector< std::string > fileNamesDixonWater;
  std::vector< std::string > fileNamesDixonFat;

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

  QStringList argsRegResample;
  std::string progRegResample = SplitStringIntoCommandAndArguments( comRegResample, argsRegResample );

  InputParameters args( commandLine, 
                        flgVerbose, flgRegister, flgSaveImages, 
                        flgCompression, flgDebug, flgOverwrite,
                        flgExcludeAxilla, flgCropFit, coilCropDistance,
                        flgDoNotBiasFieldCorrectT1w, flgDoNotBiasFieldCorrectT2w,
                        dirSubMRI, dirSubData, dirPrefix, dirInput,
                        fileLog, fileT1wOutputCSV, fileT2wOutputCSV,
                        strSeriesDescStructuralT2,
                        strSeriesDescFatSatT1,
                        strSeriesDescDixonWater,
                        strSeriesDescDixonFat,
                        progSegEM,       argsSegEM,
                        progRegAffine,   argsRegAffine,
                        progRegNonRigid, argsRegNonRigid,
                        progRegResample, argsRegResample );


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

  std::string fileI00_T2w = std::string( "I00_" ) + strSeriesDescStructuralT2 + ".nii";
  std::string fileI00_T1w = std::string( "I00_" ) + strSeriesDescFatSatT1 + ".nii";

  std::string fileI00_sag_dixon_bilateral_W = std::string( "I00_") + strSeriesDescDixonWater + ".nii";
  std::string fileI00_sag_dixon_bilateral_F = std::string( "I00_") + strSeriesDescDixonFat + ".nii";

  std::string fileI01_T2w_BiasFieldCorrection;
  std::string fileI01_T1w_BiasFieldCorrection;

  std::string fileI01_T2w_BiasFieldCorrectionMask;
  std::string fileI01_T1w_BiasFieldCorrectionMask;

  std::string fileI01_T2w_BiasField;
  std::string fileI01_T1w_BiasField;

  std::string fileI01_T1w_BiasFieldCorrectionReorient;

  if ( flgDoNotBiasFieldCorrectT1w )
  {
    fileI01_T1w_BiasFieldCorrection = fileI00_T1w;
    fileI01_T1w_BiasFieldCorrectionReorient = std::string( "I01_" ) + strSeriesDescFatSatT1 + "_Reorient.nii";
  }
  else
  {
    fileI01_T1w_BiasFieldCorrection = std::string( "I01_" ) + strSeriesDescFatSatT1 + "_BiasFieldCorrection.nii";
    fileI01_T1w_BiasFieldCorrectionMask = std::string( "I01_" ) + strSeriesDescFatSatT1 + "_BiasFieldCorrectionMask.nii";
    fileI01_T1w_BiasField = std::string( "I01_" ) + strSeriesDescFatSatT1 + "_BiasField.nii";
    fileI01_T1w_BiasFieldCorrectionReorient = std::string( "I01_" ) + strSeriesDescFatSatT1 + "_BiasFieldCorrectionReorient.nii";
  }

  if ( flgDoNotBiasFieldCorrectT2w )
  {
    fileI01_T2w_BiasFieldCorrection = std::string( "I01_" ) + strSeriesDescStructuralT2 + ".nii" ;
  }
  else
  {
    fileI01_T2w_BiasFieldCorrection = std::string( "I01_" ) + strSeriesDescStructuralT2 + "_BiasFieldCorrection.nii" ;
    fileI01_T2w_BiasFieldCorrectionMask = std::string( "I01_" ) + strSeriesDescStructuralT2 + "_BiasFieldCorrectionMask.nii" ;
    fileI01_T2w_BiasField = std::string( "I01_" ) + strSeriesDescStructuralT2 + "_BiasField.nii" ;
  }
  


  std::string fileI02_T2w_Resampled;
  
  if ( args.flgRegisterImages )
  {
    fileI02_T2w_Resampled = 
      CreateRegisteredFilename( fileI01_T1w_BiasFieldCorrection,
                                fileI01_T2w_BiasFieldCorrection,
                                std::string( "NonRigidTo" ) );
  }
  else
  {
    fileI02_T2w_Resampled = std::string( "I02_" ) + strSeriesDescStructuralT2 + "_Resampled.nii";
  }

  std::string fileBIFs( "I03_OrientedBIFsSig3_Axial.nii" );

  std::string fileOutputSmoothedStructural;
  std::string fileOutputSmoothedFatSat;
  std::string fileOutputClosedStructural;

  std::string fileOutputMaxImage( "I04_FatSat_and_T2_MaximumIntensities.nii" );
  std::string fileOutputCombinedHistogram( "I05_FatSat_and_T2_CombinedHistogram.txt" );
  std::string fileOutputRayleigh( "I05_FatSat_and_T2_RayleighFit.txt" );
  std::string fileOutputFreqLessBgndCDF( "I05_FatSat_and_T2_FreqLessBgndCDF.txt" );
  std::string fileOutputBackground( "I05_BackgroundMask.nii" );
  std::string fileOutputGradientMagImage( "I06_GradientMagImage.nii" );
  std::string fileOutputSpeedImage( "I07_SpeedImage.nii" );
  std::string fileOutputFastMarchingImage( "I08_FastMarchingImage.nii" );
  std::string fileOutputPectoral( "I09_PectoralMask.nii" );
  std::string fileOutputChestPoints( "I10_ChestPoints.nii" );
  std::string fileOutputSkinElevationMap( "I10_SkinElevationMap.nii" );
  std::string fileOutputPectoralSurfaceMask( "I11_PectoralSurfaceMask.nii" );

  std::string fileOutputPectoralSurfaceVoxels;

  std::string fileOutputFittedBreastMask( "I12_AnteriorSurfaceCropMask.nii" );

  std::string fileOutputVTKSurface( "I13_BreastSurface.vtk" );

  std::string fileOutputBreastMask( "I14_BreastMaskSegmentation.nii" );
  std::string fileOutputBreastMaskReorient( "I14_BreastMaskSegmentationReorient.nii" );

  std::string fileOutputDixonMask( "I14_DixonMaskSegmentation.nii" );

  std::string fileOutputT1wParenchyma( "I15_T1wBreastParenchyma.nii" );
  std::string fileOutputT2wParenchyma( "I15_T2wBreastParenchyma.nii" );

  std::string fileT1wDensityMeasurements( "I16_T1wDensityMeasurements.csv" );
  std::string fileT2wDensityMeasurements( "I16_T2wDensityMeasurements.csv" );



  if ( flgCompression )
  {

    fileI00_T2w.append( ".gz" );
    fileI00_T1w.append( ".gz" );

    fileI00_sag_dixon_bilateral_W.append( ".gz" );
    fileI00_sag_dixon_bilateral_F.append( ".gz" );

    fileI01_T2w_BiasFieldCorrection.append( ".gz" );
    fileI01_T1w_BiasFieldCorrection.append( ".gz" );

    fileI01_T2w_BiasFieldCorrectionMask.append( ".gz" );
    fileI01_T1w_BiasFieldCorrectionMask.append( ".gz" );

    fileI01_T2w_BiasField.append( ".gz" );
    fileI01_T1w_BiasField.append( ".gz" );

    fileI01_T1w_BiasFieldCorrectionReorient.append( ".gz" );

    fileI02_T2w_Resampled.append( ".gz" );

    fileOutputBreastMask.append( ".gz" );
    fileOutputBreastMaskReorient.append( ".gz" );

    fileOutputDixonMask.append( ".gz" );


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

    if ( fileOutputT1wParenchyma.length() > 0 )         fileOutputT1wParenchyma.append( ".gz" );
    if ( fileOutputT2wParenchyma.length() > 0 )         fileOutputT2wParenchyma.append( ".gz" );
  }

  std::cout  << std::endl << "<filter-progress>" << std::endl
             << 0. << std::endl
             << "</filter-progress>" << std::endl << std::endl;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string dirFullPath;
  std::string dirBaseName;

  std::string dirMRI;

  std::vector< std::string > directoryNames;
  std::vector< std::string >::iterator iterDirectoryNames;       

  directoryNames = niftk::GetDirectoriesInDirectory( args.dirInput );

  nDirectories = directoryNames.size();

  for ( iterDirectoryNames = directoryNames.begin(); 
	iterDirectoryNames < directoryNames.end(); 
	++iterDirectoryNames, iDirectory += 1. )
  {

    ImageType::Pointer imStructuralT2 = 0;
    ImageType::Pointer imFatSatT1 = 0;

    ImageType::Pointer imDixonWater = 0;
    ImageType::Pointer imDixonFat = 0;

    ImageType::Pointer imSegmentedBreastMask = 0;
    ImageType::Pointer imDixonBreastMask = 0;
    
    dirFullPath = *iterDirectoryNames;
    dirBaseName = niftk::Basename( dirFullPath );

    if ( ! (dirBaseName.compare( 0, args.dirPrefix.length(), args.dirPrefix ) == 0) )
    {
      message << std::endl << "Skipping directory: " << dirFullPath << std::endl << std::endl;
      args.PrintMessage( message );
      continue;
    }

    message << std::endl << "Directory: " << dirFullPath << std::endl << std::endl;
    args.PrintMessage( message );

    if ( dirSubMRI.length() > 0 )
    {
      dirMRI = niftk::ConcatenatePath( dirFullPath, dirSubMRI );
    }
    else
    {
      dirMRI = dirFullPath;
    }

    if ( dirSubData.length() > 0 )
    {
      args.dirOutput = niftk::ConcatenatePath( dirFullPath, dirSubData );
    }
    else
    {
      args.dirOutput = dirFullPath;
    }

    if ( ! niftk::DirectoryExists( args.dirOutput ) )
    {
      niftk::CreateDirAndParents( args.dirOutput );

      message << "Creating output directory: " << args.dirOutput << std::endl;
      args.PrintMessage( message );
    }

    try
    {      

      // If the CSV file has already been generated then read it
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( ReadFileCSV( args, 
                        fileT1wDensityMeasurements, 
                        args.foutOutputT1wCSV, 
                        flgVeryFirstRowT1w ) 
           && 
           ReadFileCSV( args, 
                        fileT2wDensityMeasurements, 
                        args.foutOutputT2wCSV, 
                        flgVeryFirstRowT2w ) )
      {
        continue;
      }
      
      progress = ( iDirectory + 0.1 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


      // Find the DICOM files in this directory
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      fileNamesStructuralT2.clear();
      fileNamesFatSatT1.clear();

      fileNamesDixonWater.clear();
      fileNamesDixonFat.clear();


      std::vector< std::string > fileNames;
      std::vector< std::string >::iterator iterFilenames;
     
      typedef itk::GDCMSeriesFileNames NamesGeneratorType;

      NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

      nameGenerator->SetLoadSequences( true );
      nameGenerator->SetLoadPrivateTags( true ); 

      nameGenerator->SetUseSeriesDetails( true );
    
      nameGenerator->SetDirectory( dirMRI );
  
    
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

        fileNames = nameGenerator->GetFileNames( *seriesItr );
      
      
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
          continue;
        }
 
        const  DictionaryType &dictionary = gdcmImageIO->GetMetaDataDictionary();
      
        args.PrintTag( dictionary, "0008|103e" ); // Series description
        args.PrintTag( dictionary, "0018|1030" ); // Protocol name


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

        if ( seriesDescription.find( args.strSeriesDescStructuralT2 ) != std::string::npos )
        {
          fileNamesStructuralT2 = fileNames;
        }
        else if ( seriesDescription.find( args.strSeriesDescFatSatT1 ) != std::string::npos )
        {
          fileNamesFatSatT1 = fileNames;
        }
        else if ( seriesDescription.find( args.strSeriesDescDixonWater ) != std::string::npos )
        {
          fileNamesDixonWater = fileNames;
        }
        else if ( seriesDescription.find( args.strSeriesDescDixonFat ) != std::string::npos )
        {
          fileNamesDixonFat = fileNames;
        }


        seriesItr++;
      }

      progress = ( iDirectory + 0.2 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


      // Load the T1w image
      // ~~~~~~~~~~~~~~~~~~
        
      if ( flgOverwrite || 
           ( ! args.ReadImageFromFile( args.dirOutput, fileI01_T1w_BiasFieldCorrection, 
                                       std::string( "bias field corrected '") +
                                       args.strSeriesDescFatSatT1 + "' image", 
                                       imFatSatT1 ) ) )
      {
        if ( flgOverwrite || 
             ( ! args.ReadImageFromFile( args.dirOutput, fileI00_T1w, 
                                         std::string( "complementary '" ) +
                                         args.strSeriesDescFatSatT1 + "' image", imFatSatT1 ) ) )
        {
          if ( fileNamesFatSatT1.size() > 0 )
          {
            
            SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
            seriesReader->SetFileNames( fileNamesFatSatT1 );

            message << std::endl << "Reading '" << args.strSeriesDescFatSatT1 << "' image" << std::endl;  
            args.PrintMessage( message );

            seriesReader->UpdateLargestPossibleRegion();

            imFatSatT1 = seriesReader->GetOutput();
            imFatSatT1->DisconnectPipeline();
            
            args.WriteImageToFile( fileI00_T1w, 
                                   std::string( "complementary '" ) + args.strSeriesDescFatSatT1 +
                                   "' image", imFatSatT1 );
          }
        }

        // Bias field correct it?

        if ( ( ! flgDoNotBiasFieldCorrectT1w ) && imFatSatT1 )
        {
          
          message << std::endl << "Bias field correcting '" << args.strSeriesDescFatSatT1 << "' image"
                  << std::endl;  
          args.PrintMessage( message );
          
          BiasFieldCorrectionType::Pointer biasFieldCorrector = BiasFieldCorrectionType::New();
          
          biasFieldCorrector->SetInput( imFatSatT1 );
          biasFieldCorrector->Update();
          
          imFatSatT1 = biasFieldCorrector->GetOutput();
          imFatSatT1->DisconnectPipeline();
          
          args.WriteImageToFile( fileI01_T1w_BiasFieldCorrection, 
                                 std::string( "bias field corrected '" ) + 
                                 args.strSeriesDescFatSatT1 + "' image", imFatSatT1 );

          if ( args.flgDebug )
          {
            args.WriteImageToFile( fileI01_T1w_BiasFieldCorrectionMask, 
                                   std::string( "mask used for bias field corrected '" ) + 
                                   args.strSeriesDescFatSatT1 + "' image", 
                                   biasFieldCorrector->GetMask() );

            args.WriteImageToFile( fileI01_T1w_BiasField, 
                                   std::string( "bias field '" ) + 
                                   args.strSeriesDescFatSatT1 + "' image", 
                                   biasFieldCorrector->GetBiasField() );

          }
        }
      }

      progress = ( iDirectory + 0.3 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


      // Load the structural image
      // ~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( flgOverwrite || 
           ( ! args.ReadImageFromFile( args.dirOutput, fileI02_T2w_Resampled, 
                                       std::string( "resampled '" ) 
                                       + args.strSeriesDescStructuralT2 + "' image", 
                                       imStructuralT2 ) ) )
      {
        if  ( flgOverwrite || 
              ( ! args.ReadImageFromFile( args.dirOutput, fileI01_T2w_BiasFieldCorrection, 
                                          std::string( "bias field corrected '" ) + 
                                          args.strSeriesDescStructuralT2 + "' image", 
                                          imStructuralT2 ) ) )
        {
          if ( flgOverwrite || 
               ( ! args.ReadImageFromFile( args.dirOutput, fileI00_T2w,
                                           std::string( "structural '" ) + 
                                           args.strSeriesDescStructuralT2 + "' image", 
                                           imStructuralT2 ) ) )
          {
            if ( fileNamesStructuralT2.size() > 0 )
            {
            
              SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
              seriesReader->SetFileNames( fileNamesStructuralT2 );

              message << std::endl << "Reading '" << args.strSeriesDescStructuralT2 << "' image" << std::endl;  
              args.PrintMessage( message );

              seriesReader->UpdateLargestPossibleRegion();

              imStructuralT2 = seriesReader->GetOutput();
              imStructuralT2->DisconnectPipeline();

              args.WriteImageToFile( fileI00_T2w,
                                     std::string( "structural '" ) + args.strSeriesDescStructuralT2 +
                                     "' image", imStructuralT2 );
            }
          }

          if ( imStructuralT2 )
          {
          
            // Bias field correct it?
          
            if ( ! flgDoNotBiasFieldCorrectT2w )
            {
              message << std::endl << "Bias field correcting '" << args.strSeriesDescStructuralT2 << "' image" << std::endl;  
              args.PrintMessage( message );
              
              BiasFieldCorrectionType::Pointer biasFieldCorrector = BiasFieldCorrectionType::New();
              
              biasFieldCorrector->SetInput( imStructuralT2 );
              biasFieldCorrector->Update();
              
              imStructuralT2 = biasFieldCorrector->GetOutput();
              imStructuralT2->DisconnectPipeline();

              if ( args.flgDebug )
              {
                args.WriteImageToFile( fileI01_T2w_BiasFieldCorrectionMask, 
                                       std::string( "mask used for bias field corrected '" ) +
                                       args.strSeriesDescStructuralT2 + "' image",
                                       biasFieldCorrector->GetMask() );

                args.WriteImageToFile( fileI01_T2w_BiasField, 
                                       std::string( "bias field '" ) +
                                       args.strSeriesDescStructuralT2 + "' image",
                                       biasFieldCorrector->GetBiasField() );
              }            
            }

            // Rescale the 98th percentile to 100
          
            message << std::endl << "Rescaling '" << args.strSeriesDescStructuralT2 << "' image to 100" << std::endl;  
            args.PrintMessage( message );

            typedef itk::RescaleImageUsingHistogramPercentilesFilter<ImageType, ImageType> RescaleFilterType;

            RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
            rescaleFilter->SetInput( imStructuralT2 );
  
            rescaleFilter->SetInLowerPercentile(  0. );
            rescaleFilter->SetInUpperPercentile( 98. );

            rescaleFilter->SetOutLowerLimit(   0. );
            rescaleFilter->SetOutUpperLimit( 100. );

            rescaleFilter->Update();

            imStructuralT2 = rescaleFilter->GetOutput();
            imStructuralT2->DisconnectPipeline();          
            
            args.WriteImageToFile( fileI01_T2w_BiasFieldCorrection, 
                                   std::string( "bias field corrected '" ) +
                                   args.strSeriesDescStructuralT2 + "' image", imStructuralT2 );
          }
        }
      
        // Resample the T2 image to match the FatSat image

        if ( imStructuralT2 && imFatSatT1 ) 
        {

          // Also register 
          if ( args.flgRegisterImages )
          {
            if ( ! ( RegisterImages( fileI01_T1w_BiasFieldCorrection,
                                     fileI01_T2w_BiasFieldCorrection,
                                     fileI02_T2w_Resampled,
                                     args ) 
                     &&
                     args.ReadImageFromFile( args.dirOutput, fileI02_T2w_Resampled, 
                                             std::string( "registered '") +
                                             args.strSeriesDescStructuralT2 + "' image", 
                                             imStructuralT2 ) ) )
            {
              imStructuralT2 = 0;
            }
          }

          // Just resample
          else
          {
            typedef itk::IdentityTransform<double, Dimension> TransformType;
            TransformType::Pointer identityTransform = TransformType::New();

            typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
            ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

            typedef itk::NearestNeighborInterpolateImageFunction< ImageType, double >  InterpolatorType;
            InterpolatorType::Pointer interpolator = InterpolatorType::New();

            resampleFilter->SetUseReferenceImage( true ); 
            resampleFilter->SetReferenceImage( imFatSatT1 ); 

            resampleFilter->SetTransform( identityTransform );
            resampleFilter->SetInterpolator( interpolator );

            resampleFilter->SetInput( imStructuralT2 );

            resampleFilter->Update();

            imStructuralT2 = resampleFilter->GetOutput();
            imStructuralT2->DisconnectPipeline();

            args.WriteImageToFile( fileI02_T2w_Resampled, 
                                   std::string( "resampled '" ) + args.strSeriesDescStructuralT2 +
                                   "' image", imStructuralT2 );
          }
        }
      }


      // Have we found both input images?

      if ( ! ( imStructuralT2 && imFatSatT1 ) )
      {
        message << "Both of structural and complementary images not found, "
                << "skipping this directory: " << std::endl << std::endl;
        args.PrintWarning( message );
        continue;
      }
          
      progress = ( iDirectory + 0.4 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


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

        bool flgExtInitialPect = true;

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

        ImageType::Pointer imBIFs;


        // Create the Breast Segmentation Object
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        typedef itk::BreastMaskSegmForBreastDensity< Dimension, PixelType > 
          BreastMaskSegmForBreastDensityType;
  
        BreastMaskSegmForBreastDensityType::Pointer 
          breastMaskSegmentor = BreastMaskSegmForBreastDensityType::New();


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

        breastMaskSegmentor->SetCropFit( flgExcludeAxilla );
        breastMaskSegmentor->SetCropFit( flgCropFit );
        breastMaskSegmentor->SetCoilCropDistance( coilCropDistance );
          

        if ( args.flgDebug )
        {

          if ( fileOutputSmoothedStructural.length() > 0 )    breastMaskSegmentor->SetOutputSmoothedStructural(   niftk::ConcatenatePath( args.dirOutput, fileOutputSmoothedStructural ) );
          if ( fileOutputSmoothedFatSat.length() > 0 )        breastMaskSegmentor->SetOutputSmoothedFatSat(       niftk::ConcatenatePath( args.dirOutput, fileOutputSmoothedFatSat ) );
          if ( fileOutputClosedStructural.length() > 0 )      breastMaskSegmentor->SetOutputClosedStructural(     niftk::ConcatenatePath( args.dirOutput, fileOutputClosedStructural ) );
          if ( fileOutputCombinedHistogram.length() > 0 )     breastMaskSegmentor->SetOutputHistogram(            niftk::ConcatenatePath( args.dirOutput, fileOutputCombinedHistogram ) );
          if ( fileOutputRayleigh.length() > 0 )              breastMaskSegmentor->SetOutputFit(                  niftk::ConcatenatePath( args.dirOutput, fileOutputRayleigh ) );
          if ( fileOutputFreqLessBgndCDF.length() > 0 )       breastMaskSegmentor->SetOutputCDF(                  niftk::ConcatenatePath( args.dirOutput, fileOutputFreqLessBgndCDF ) );
          if ( fileOutputMaxImage.length() > 0 )              breastMaskSegmentor->SetOutputImageMax(             niftk::ConcatenatePath( args.dirOutput, fileOutputMaxImage ) );
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

        breastMaskSegmentor->SetStructuralImage( imStructuralT2 );
        breastMaskSegmentor->SetFatSatImage( imFatSatT1 );

        breastMaskSegmentor->Execute();

        imSegmentedBreastMask = breastMaskSegmentor->GetSegmentedImage();

        args.WriteImageToFile( fileOutputBreastMask, 
                               "breast mask segmentation image", imSegmentedBreastMask );

      }

      progress = ( iDirectory + 0.5 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


      // Load the Dixon Water Image
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( flgOverwrite || 
           ( ! args.ReadImageFromFile( args.dirOutput, fileI00_sag_dixon_bilateral_W, 
                                       std::string( "Dixon water '" ) + args.strSeriesDescDixonWater +
                                       "' image", imDixonWater ) ) )
      {
        if ( fileNamesDixonWater.size() > 0 )
        {
          
          SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
          seriesReader->SetFileNames( fileNamesDixonWater );
          
          message << std::endl << "Reading '" << args.strSeriesDescDixonWater << "' image" << std::endl;  
          args.PrintMessage( message );
          
          seriesReader->UpdateLargestPossibleRegion();
          
          imDixonWater = seriesReader->GetOutput();
          imDixonWater->DisconnectPipeline();
          
          args.WriteImageToFile( fileI00_sag_dixon_bilateral_W, 
                                 std::string( "Dixon water '" ) + args.strSeriesDescDixonWater +
                                 "' image", imDixonWater );
        }
      }

      progress = ( iDirectory + 0.6 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


      // Load the Dixon Fat Image
      // ~~~~~~~~~~~~~~~~~~~~~~~~

      if ( flgOverwrite || 
           ( ! args.ReadImageFromFile( args.dirOutput, fileI00_sag_dixon_bilateral_F, 
                                       std::string( "Dixon fat '" ) + args.strSeriesDescDixonFat +
                                       "' image", imDixonFat ) ) )
      {
        if ( fileNamesDixonFat.size() > 0 )
        {
          
          SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
          seriesReader->SetFileNames( fileNamesDixonFat );
          
          message << std::endl << "Reading '" << args.strSeriesDescDixonFat << "' image" << std::endl;  
          args.PrintMessage( message );
          
          seriesReader->UpdateLargestPossibleRegion();
          
          imDixonFat = seriesReader->GetOutput();
          imDixonFat->DisconnectPipeline();
          
          args.WriteImageToFile( fileI00_sag_dixon_bilateral_F, 
                                 std::string( "Dixon fat '" ) + args.strSeriesDescDixonFat +
                                 "' image", imDixonFat );
        }
      }

      progress = ( iDirectory + 0.7 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


      // Segment the parenchyma
      // ~~~~~~~~~~~~~~~~~~~~~~

      SegmentParenchyma( "T1w",

                         args,
                         args.foutOutputT1wCSV,
                         args.fileT1wOutputCSV,
                         flgVeryFirstRowT1w,
                         false,

                         dirBaseName,
                         fileI01_T1w_BiasFieldCorrection, 
                         fileOutputBreastMask, 
                         fileOutputT1wParenchyma,
                         fileT1wDensityMeasurements,

                         imSegmentedBreastMask,
                         imFatSatT1 );


      SegmentParenchyma( "T2w",

                         args,
                         args.foutOutputT2wCSV,
                         args.fileT2wOutputCSV,
                         flgVeryFirstRowT2w,
                         true,

                         dirBaseName,
                         fileI02_T2w_Resampled, 
                         fileOutputBreastMask, 
                         fileOutputT2wParenchyma,
                         fileT2wDensityMeasurements,

                         imSegmentedBreastMask,
                         imStructuralT2 );

      progress = ( iDirectory + 0.8 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;


      // Resample the breast mask to match the Dixon images
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( flgOverwrite || 
           ( ! args.ReadImageFromFile( args.dirOutput, fileOutputDixonMask, 
                                       "Dixon breast mask", imDixonBreastMask ) ) )
      {

        // Also register 
        if ( args.flgRegisterImages )
        {
          if ( flgOverwrite || 
               ( ! args.ReadImageFromFile( args.dirOutput, fileI01_T1w_BiasFieldCorrectionReorient, 
                                           std::string( "bias field corrected '") +
                                           args.strSeriesDescFatSatT1 + "' reorient image", 
                                           imFatSatT1 ) ) )
          {
            imFatSatT1 = ReorientateImage( imFatSatT1, 
                                           GetOrientationInfo( imDixonWater ) );

            args.WriteImageToFile( fileI01_T1w_BiasFieldCorrectionReorient, 
                                   std::string( "bias field corrected '" ) + 
                                   args.strSeriesDescFatSatT1 + "' reorient image", imFatSatT1 );
          }

          if ( flgOverwrite || 
               ( ! args.ReadImageFromFile( args.dirOutput, fileOutputBreastMaskReorient, 
                                           "segmented breast Reorient mask", 
                                           imSegmentedBreastMask ) ) )
          {
            imSegmentedBreastMask = ReorientateImage( imSegmentedBreastMask,
                                                      GetOrientationInfo( imDixonWater ) );

            args.WriteImageToFile( fileOutputBreastMaskReorient, 
                                   "breast mask segmentation reorient image", imSegmentedBreastMask );
          }

          RegisterAndResample( fileI00_sag_dixon_bilateral_W,
                               fileI01_T1w_BiasFieldCorrectionReorient,
                               fileOutputBreastMaskReorient,
                               fileOutputDixonMask,
                               std::string( "-NN" ),
                               args );
        }
        
        // Just resample
        else
        {
          
          if ( imDixonWater && imDixonFat ) 
          {
            typedef itk::IdentityTransform<double, Dimension> TransformType;
            TransformType::Pointer identityTransform = TransformType::New();
            
            typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
            ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
            
            typedef itk::NearestNeighborInterpolateImageFunction<ImageType, double> InterpolatorType;
            InterpolatorType::Pointer interpolator = InterpolatorType::New();
            
            resampleFilter->SetUseReferenceImage( true ); 
            resampleFilter->SetReferenceImage( imDixonWater ); 
            
            resampleFilter->SetTransform( identityTransform );
            resampleFilter->SetInterpolator( interpolator );
            
            resampleFilter->SetInput( imSegmentedBreastMask );
            
            resampleFilter->Update();
            
            imDixonBreastMask = resampleFilter->GetOutput();
            imDixonBreastMask->DisconnectPipeline();
            
            args.WriteImageToFile( fileOutputDixonMask, 
                                   "Dixon breast mask", imDixonBreastMask );
          }
        }
      }

      progress = ( iDirectory + 0.9 )/nDirectories;
      std::cout  << std::endl << "<filter-progress>" << std::endl
                 << progress << std::endl
                 << "</filter-progress>" << std::endl << std::endl;
    
      
      // Delete unwanted images
      // ~~~~~~~~~~~~~~~~~~~~~~

      if ( ! ( flgDebug || flgSaveImages ) )
      {
        args.DeleteFile( fileI01_T2w_BiasFieldCorrection );
        args.DeleteFile( fileI01_T1w_BiasFieldCorrection );

        args.DeleteFile( fileI01_T2w_BiasFieldCorrectionMask );
        args.DeleteFile( fileI01_T1w_BiasFieldCorrectionMask );

        args.DeleteFile( fileI01_T2w_BiasField );
        args.DeleteFile( fileI01_T1w_BiasField );

        args.DeleteFile( fileI02_T2w_Resampled );

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
    }
    catch (itk::ExceptionObject &ex)
    {
      message << ex << std::endl;
      args.PrintError( message );
      continue;
    }

    progress = ( iDirectory + 1. )/nDirectories;
    std::cout  << std::endl << "<filter-progress>" << std::endl
               << progress << std::endl
               << "</filter-progress>" << std::endl << std::endl;

  }


  progress = iDirectory/nDirectories;
  std::cout  << std::endl << "<filter-progress>" << std::endl
             << progress << std::endl
             << "</filter-progress>" << std::endl << std::endl;
  
  return EXIT_SUCCESS;
}
 
 

