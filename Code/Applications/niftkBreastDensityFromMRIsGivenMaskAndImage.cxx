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
#include <itkWriteImage.h>
#include <itkReadImage.h>
#include <itkConversionUtils.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkOrientImageFilter.h>
#include <itkSampleImageFilter.h>
#include <itkBinaryShapeBasedSuperSamplingFilter.h>
#include <itkIsImageBinary.h>


//#define LINK_TO_SEG_EM

#ifdef LINK_TO_SEG_EM
#include <_seg_EM.h>
#endif

#include <niftkBreastDensityFromMRIsGivenMaskAndImageCLP.h>

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
typedef itk::Image< PixelType, 4 > ImageType4D;




// -------------------------------------------------------------------------
// CreateFilename()
// -------------------------------------------------------------------------

std::string CreateFilename( std::string fileOne, 
                            std::string fileTwo,
                            std::string description,
                            std::string suffix ) 
{
  std::string fileOneWithoutSuffix;
  niftk::ExtractImageFileSuffix( fileOne, fileOneWithoutSuffix );

  std::string fileTwoWithoutSuffix;
  niftk::ExtractImageFileSuffix( fileTwo, fileTwoWithoutSuffix );

  return fileOneWithoutSuffix + "_" + description + "_" + fileTwoWithoutSuffix + suffix;
};


// -------------------------------------------------------------------------
// PrintOrientationInfo()
// -------------------------------------------------------------------------

void PrintOrientationInfo( ImageType::Pointer image )
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

  std::cout << "Image direction: " << std::endl
	    << direction;

  std::cout << "ITK orientation: " 
	    << itk::ConvertSpatialOrientationToString(adaptor.FromDirectionCosines(direction)) 
	    << std::endl;
}


// -------------------------------------------------------------------------
// GetOrientation()
// -------------------------------------------------------------------------

itk::SpatialOrientation::ValidCoordinateOrientationFlags GetOrientation( ImageType::Pointer image )
{
  ImageType::DirectionType direction;
  itk::SpatialOrientationAdapter adaptor;

  for (unsigned int i = 0; i < Dimension; i++)
  {
    for (unsigned int j = 0; j < Dimension; j++)
    {
      direction[i][j] = image->GetDirection()[i][j];
    }
  }

  return adaptor.FromDirectionCosines(direction);
}


// -------------------------------------------------------------------------
// ReorientateImage()
// -------------------------------------------------------------------------

ImageType::Pointer ReorientateImage( ImageType::Pointer image, 
                                     itk::SpatialOrientation::ValidCoordinateOrientationFlags desiredOrientation )
{
  std::cout << std::endl << "Input image:" << std::endl;
  //image->Print( std::cout );
  PrintOrientationInfo( image );

  std::cout << "Desired orientation: " << itk::ConvertSpatialOrientationToString( desiredOrientation )
            << std::endl;

  typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
  OrientImageFilterType::Pointer orienter = OrientImageFilterType::New();

  orienter->UseImageDirectionOn();
  orienter->SetDesiredCoordinateOrientation( desiredOrientation );
  orienter->SetInput( image );

  orienter->Update();

  typedef OrientImageFilterType::FlipAxesArrayType FlipAxesArrayType;
  typedef OrientImageFilterType::PermuteOrderArrayType PermuteOrderArrayType;

  FlipAxesArrayType flipAxes = orienter->GetFlipAxes();
  PermuteOrderArrayType permuteAxes = orienter->GetPermuteOrder();

  std::cout << std::endl
            << "Permute Axes: " << permuteAxes << std::endl
            << "Flip Axes: "    << flipAxes << std::endl;

  ImageType::Pointer reorientatedImage = orienter->GetOutput();
  //reorientatedImage->DisconnectPipeline();

  std::cout << std::endl << "Output image:" << std::endl;
  //reorientatedImage->Print( std::cout );
  PrintOrientationInfo( reorientatedImage );

  return reorientatedImage;
};


// -------------------------------------------------------------------------
// ReorientateImage()
// -------------------------------------------------------------------------

ImageType::Pointer ReorientateImage( ImageType::Pointer image, ImageType::Pointer reference )
{
  itk::SpatialOrientation::ValidCoordinateOrientationFlags desiredOrientation;

  std::cout << std::endl << "Reference image:" << std::endl;
  //reference->Print( std::cout );
  PrintOrientationInfo( reference );

  desiredOrientation = GetOrientation( reference );

  return ReorientateImage( image, desiredOrientation );
}


// -------------------------------------------------------------------------
// class InputParameters
// -------------------------------------------------------------------------

class InputParameters
{

public:

  bool flgVerbose;
  bool flgDebug;
  bool flgCompression;
  bool flgOverwrite;
  bool flgFatIsBright;

  std::string dirInput;
  std::string dirOutput;

  std::string fileLog;
  std::string fileOutputCSV;

  std::string fileMaskPattern;  
  std::string fileImagePattern;  

  std::string dirMask;  
  std::string dirImage;  

  std::string dirSubData;  
  std::string dirPrefix;  

  std::string progSegEM;

  QStringList argsSegEM;

  std::ofstream *foutLog;
  std::ofstream *foutOutputCSV;
  std::ostream *newCout;

  typedef tee_device<std::ostream, std::ofstream> TeeDevice;
  typedef stream<TeeDevice> TeeStream;

  TeeDevice *teeDevice;
  TeeStream *teeStream;

  InputParameters( TCLAP::CmdLine &commandLine, 

                   bool verbose, 
                   bool compression, 
                   bool debug, 
                   bool overwrite,
                   bool fatIsBright,

                   std::string subDirMask, 
                   std::string fileMask,
                   std::string subDirImage, 
                   std::string fileImage,

                   std::string subDirData, 
                   std::string prefix, 
                   std::string dInput,

                   std::string logfile, 
                   std::string csvfile,

                   std::string segEM,       
                   QStringList aSegEM ) {

    std::stringstream message;

    flgVerbose = verbose;
    flgDebug = debug;
    flgCompression = compression;
    flgOverwrite = overwrite;
    flgFatIsBright = fatIsBright;

    dirMask  = subDirMask;
    fileMaskPattern = fileMask;
    dirImage = subDirImage;
    fileImagePattern = fileImage;

    dirSubData = subDirData;
    dirPrefix  = prefix;
    dirInput   = dInput;

    fileLog = logfile;
    fileOutputCSV = csvfile;

    progSegEM       = segEM;
    argsSegEM       = aSegEM;

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
            << "Examining directory: " << dirInput << std::endl 
            << std::endl
            << std::boolalpha
            << "Verbose output?: "             << flgVerbose        << std::endl
            << "Compress images?: "            << flgCompression    << std::endl
            << "Debugging output?: "           << flgDebug          << std::endl
            << "Overwrite previous results?: " << flgOverwrite      << std::endl       
            << "Fat is bright?: "              << flgFatIsBright    << std::endl       
            << std::noboolalpha
            << std::endl
            << "Input mask sub-directory: " << dirMask << std::endl
            << "Input mask file name pattern: " << fileMaskPattern << std::endl
            << "Input image sub-directory: " << dirImage << std::endl
            << "Input image file name pattern: " << fileMaskPattern << std::endl
            << "Output data sub-directory: " << dirSubData << std::endl
            << "Study directory prefix: " << dirPrefix << std::endl
            << std::endl
            << "Output log file: " << fileLog << std::endl
            << "Output csv file: " << fileOutputCSV << std::endl
            << std::endl
            << "Segmentation executable: "
            << progSegEM       << " " << argsSegEM.join(" ").toStdString() << std::endl
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

  bool ReadImageFromFile( std::string fileInput, 
                          std::string description, ImageType::Pointer &image ) {
  
    std::stringstream message;

    if ( fileInput.find( ".nii" ) == std::string::npos )
    {
      return false;
    }

    if ( itk::ReadImageFromFile< ImageType >( fileInput, image ) )
    {   
      message << std::endl << "Read " << description << " from file: " << fileInput << std::endl;
      PrintMessage( message );
    }
    else if ( itk::ReadImageFromFile< ImageType >( fileInput + ".gz", image ) )
    {   
      message << std::endl << "Read " << description << " from file: " << fileInput << std::endl;
      PrintMessage( message );
    }
    else
    {
      return false;
    }

    itk::SpatialOrientation::ValidCoordinateOrientationFlags orientation;

    orientation = GetOrientation( image );

    if ( orientation !=  itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI )
    {
      image = ReorientateImage( image, itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI );
    }

    return true;
  }

  bool ReadImageFromFile( std::string dirInput, std::string filename, 
                          std::string description, ImageType::Pointer &image ) {
  
    std::string fileInput = niftk::ConcatenatePath( dirInput, filename );               

    return ReadImageFromFile( fileInput, description, image );
  }

  void WriteImageToFile( std::string filename, 
                         std::string description, ImageType::Pointer image,
                         bool flgConcatenatePath=true ) {
  
    std::stringstream message;
    std::string fileOutput;

    if ( flgConcatenatePath )
    {
      fileOutput = niftk::ConcatenatePath( dirOutput, filename );
    }
    else
    {
      fileOutput = filename;
    }
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    itk::WriteImageToFile< ImageType >( fileOutput, image );
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
// ResampleImageToIsotropicVoxels()
// -------------------------------------------------------------------------

bool ResampleImageToIsotropicVoxels( ImageType::Pointer &image, InputParameters &args )
{
  std::stringstream message;

  double factor[Dimension];

  ImageType::SpacingType spacing = image->GetSpacing();

  // Calculate the minimum spacing

  double minSpacing = std::numeric_limits<double>::max();

  for (unsigned int j = 0; j < ImageType::ImageDimension; j++)
  {
    if ( spacing[j] < minSpacing )
    {
      minSpacing = spacing[j];
    }
  }

  // Calculate the subsampling factors

  bool flgSamplingRequired = false;

  for (unsigned int j = 0; j < ImageType::ImageDimension; j++)
  {
    factor[j] = minSpacing/spacing[j];

    if ( factor[j] < 0.8 )
    {
      flgSamplingRequired = true;
    }
  }

  // Run the sampling filter

  if ( itk::IsImageBinary< ImageType >( image ) )
  {
    typedef itk::BinaryShapeBasedSuperSamplingFilter< ImageType, ImageType > SampleImageFilterType;

    SampleImageFilterType::Pointer sampler = SampleImageFilterType::New();

    sampler->SetIsotropicVoxels( true );

    sampler->SetInput( image );

    sampler->VerboseOn();

    message << "Computing sampled binary image" << std::endl;
    args.PrintMessage( message );

    sampler->Update();

    ImageType::Pointer sampledImage = sampler->GetOutput();
    sampledImage->DisconnectPipeline();

    image = sampledImage;
  }
  else
  {
    typedef itk::SampleImageFilter< ImageType, ImageType > SampleImageFilterType;

    SampleImageFilterType::Pointer sampler = SampleImageFilterType::New();

    sampler->SetIsotropicVoxels( true );
    sampler->SetInterpolationType( itk::NEAREST );

    sampler->SetInput( image );

    sampler->VerboseOn();

    message << "Computing sampled image" << std::endl;
    args.PrintMessage( message );

    sampler->Update();

    ImageType::Pointer sampledImage = sampler->GetOutput();
    sampledImage->DisconnectPipeline();

    image = sampledImage;
  }

  return true;
};


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  bool flgVeryFirstRow = true;

  float progress = 0.;
  float iDirectory = 0.;
  float nDirectories;

  std::stringstream message;

  fs::path pathExecutable( argv[0] );
  fs::path dirExecutables = pathExecutable.parent_path();



  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  // Extract any arguments from the input commands

  QStringList argsSegEM;
  std::string progSegEM = SplitStringIntoCommandAndArguments( comSegEM, argsSegEM );

  InputParameters args( commandLine, 
                        
                        flgVerbose,
                        flgCompression, 
                        flgDebug, 
                        flgOverwrite,
                        flgFatIsBright,

                        dirMask,
                        fileMaskPattern,
                        dirImage,
                        fileImagePattern,
                          
                        dirSubData,
                        dirPrefix, 
                        dirInput,
                          
                        fileLog, 
                        fileOutputCSV,
                          
                        progSegEM,       
                        argsSegEM );


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


  args.Print();
        
  std::cout  << std::endl << "<filter-progress>" << std::endl
             << 0. << std::endl
             << "</filter-progress>" << std::endl << std::endl;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string dirFullPath;
  std::string dirBaseName;

  std::string dirMaskFullPath;
  std::string dirImageFullPath;

  std::vector< std::string > directoryNames;
  std::vector< std::string >::iterator iterDirectoryNames;       

  std::vector< std::string > fileMaskNames;
  std::vector< std::string >::iterator iterFileMaskNames;       

  std::vector< std::string > fileImageNames;
  std::vector< std::string >::iterator iterFileImageNames;       

  directoryNames = niftk::GetDirectoriesInDirectory( args.dirInput );

  nDirectories = directoryNames.size();

  for ( iterDirectoryNames = directoryNames.begin(); 
	iterDirectoryNames < directoryNames.end(); 
	++iterDirectoryNames, iDirectory += 1. )
  {
    
    dirFullPath = *iterDirectoryNames;
    dirBaseName = niftk::Basename( dirFullPath );

    if ( ! (dirBaseName.compare( 0, args.dirPrefix.length(), args.dirPrefix ) == 0) )
    {
      message << std::endl << "Skipping directory: " << dirFullPath << std::endl;
      args.PrintMessage( message );
      continue;
    }

    message << std::endl << "Directory: " << dirFullPath << std::endl;
    args.PrintMessage( message );

    
    // The mask directory

    if ( args.dirMask.length() > 0 )
    {
      dirMaskFullPath = niftk::ConcatenatePath( dirFullPath, args.dirMask );
    }
    else
    {
      dirMaskFullPath = dirFullPath;
    }

    // The image directory

    if ( args.dirImage.length() > 0 )
    {
      dirImageFullPath = niftk::ConcatenatePath( dirFullPath, args.dirImage );
    }
    else
    {
      dirImageFullPath = dirFullPath;
    }

    // The output directory

    if ( dirSubData.length() > 0 )
    {
      args.dirOutput = niftk::ConcatenatePath( dirFullPath, args.dirSubData );
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


    // For each mask image
    // ~~~~~~~~~~~~~~~~~~~

    fileMaskNames = niftk::GetFilesInDirectory( dirMaskFullPath );

    float iMask = 0.;
    float nMasks = fileMaskNames.size();

    for ( iterFileMaskNames = fileMaskNames.begin(); 
          iterFileMaskNames < fileMaskNames.end(); 
          ++iterFileMaskNames, iMask += 1. )
    {

      std::string fileMaskFullPath = *iterFileMaskNames;
      std::string fileMaskBasename = niftk::Basename( fileMaskFullPath );

      if ( fileMaskBasename.find( args.fileMaskPattern ) == std::string::npos )
      {
        if ( args.flgDebug )
        {
          message << std::endl << "Skipping non-mask file: " << fileMaskBasename 
                  << std::endl;
          args.PrintMessage( message );
        }
        continue;
      }

      // Read the mask image

      ImageType::Pointer imMask = 0;

      if ( ! args.ReadImageFromFile( fileMaskFullPath, 
                                     std::string( "mask image" ), 
                                     imMask ) )
      {
        if ( args.flgDebug )
        {
          message << std::endl << "Cannot read mask: " << fileMaskFullPath << std::endl << std::endl;
          args.PrintMessage( message );
        }
        continue;
      }

      if ( ResampleImageToIsotropicVoxels( imMask, args ) )
      {
        fileMaskBasename = niftk::ModifyImageFileSuffix( fileMaskBasename, 
                                                         std::string( "_Isotropic.nii" ) );

        fileMaskFullPath = niftk::ConcatenatePath( args.dirOutput, fileMaskBasename );

        if ( flgCompression )
        {
          fileMaskFullPath.append( ".gz" );          
        }

        args.WriteImageToFile( fileMaskFullPath, 
                               std::string( "isotropic mask image" ), 
                               imMask, false );
      }


      // For each image
      // ~~~~~~~~~~~~~~

      fileImageNames = niftk::GetFilesInDirectory( dirImageFullPath );

      float iImage = 0.;
      float nImages = fileImageNames.size();

      for ( iterFileImageNames = fileImageNames.begin(); 
            iterFileImageNames < fileImageNames.end(); 
            ++iterFileImageNames, iImage += 1. )
      {

        std::string fileImageFullPath = *iterFileImageNames;
        std::string fileImageBasename = niftk::Basename( fileImageFullPath );

        if ( fileImageBasename.find( args.fileImagePattern ) == std::string::npos )
        {
          if ( args.flgDebug )
          {
            message << std::endl << "Skipping non-image file: " << fileImageBasename 
                    << std::endl;
            args.PrintMessage( message );
          }
          continue;
        }

        // Read the image

        ImageType::Pointer imInput = 0;

        if ( ! args.ReadImageFromFile( fileImageFullPath, 
                                       std::string( "input image" ), 
                                       imInput ) )
        {
          if ( args.flgDebug )
          {
            message << std::endl << "Cannot read image: " << fileImageFullPath 
                    << std::endl;
            args.PrintMessage( message );
          }
          continue;
        }

        if ( ResampleImageToIsotropicVoxels( imInput, args ) )
        {
          fileImageBasename = niftk::ModifyImageFileSuffix( fileImageBasename, 
                                                            std::string( "_Isotropic.nii" ) );

          fileImageFullPath = niftk::ConcatenatePath( args.dirOutput, fileImageBasename );

          if ( flgCompression )
          {
            fileImageFullPath.append( ".gz" );          
          }
          
          args.WriteImageToFile( fileImageFullPath, 
                                 std::string( "isotropic image" ), 
                                 imInput, false );
        }

        try
        {      

          // If the CSV file has already been generated then read it
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

          std::string fileDensityMeasurements = CreateFilename( fileImageBasename, 
                                                                fileMaskBasename,
                                                                std::string( "WithMask" ),
                                                                std::string( ".csv" ) );
          
          std::string fileInputDensityMeasurements  
            =  niftk::ConcatenatePath( args.dirOutput, fileDensityMeasurements );

          if ( ! args.flgOverwrite ) 
          {
            if ( niftk::FileExists( fileInputDensityMeasurements ) )
            {
              std::ifstream fin( fileInputDensityMeasurements.c_str() );

              if ((! fin) || fin.bad()) 
              {
                message << "ERROR: Could not open file: " << fileDensityMeasurements << std::endl;
                args.PrintError( message );
                continue;
              }
              
              message << std::endl << "Reading CSV file: " << fileInputDensityMeasurements << std::endl;
              args.PrintMessage( message );

              niftk::CSVRow csvRow;
              
              bool flgFirstRowOfThisFile = true;
              
              while( fin >> csvRow )
              {
                message << "" << csvRow << std::endl;
                args.PrintMessage( message );
                
                if ( flgFirstRowOfThisFile )
                {
                  if ( flgVeryFirstRow )    // Include the title row?
                  {
                    *args.foutOutputCSV << csvRow << std::endl;
                    flgVeryFirstRow = false;
                  }
                  flgFirstRowOfThisFile = false;
                }
                else
                {
                  *args.foutOutputCSV << csvRow << std::endl;
                }
              }
              
              continue;
            }
            else
            {
              message << "Density measurements: " << fileInputDensityMeasurements 
                      << " not found" << std::endl;
              args.PrintMessage( message );
            }     
          }
          
          progress = ( iDirectory + ( iMask + ( iImage + 0.3 )/nImages )/nMasks )/nDirectories;
          std::cout  << std::endl << "<filter-progress>" << std::endl
                     << progress << std::endl
                     << "</filter-progress>" << std::endl << std::endl;


          // Segment the parenchyma
          // ~~~~~~~~~~~~~~~~~~~~~~

          ImageType::Pointer imParenchyma = 0;
          
          std::string fileOutputParenchyma = CreateFilename( fileImageBasename, 
                                                             fileMaskBasename,
                                                             std::string( "WithMask" ),
                                                             std::string( "_Parenchyma.nii" ) );

          if ( args.flgOverwrite || 
               ( ! args.ReadImageFromFile( args.dirOutput, 
                                           fileOutputParenchyma, 
                                           "breast parenchyma", imParenchyma ) ) )
          {

            // QProcess call to seg_EM

            QStringList argumentsNiftySeg = args.argsSegEM; 

            argumentsNiftySeg 
              << "-in"   << fileImageFullPath.c_str()
              << "-mask" << fileMaskFullPath.c_str()
              << "-out"  << niftk::ConcatenatePath( args.dirOutput, fileOutputParenchyma ).c_str();

            message << std::endl << "Executing parenchyma segmentation (QProcess): "
                    << std::endl << " " << args.progSegEM;
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

            message << "" << outSegEM.toStdString();

            if ( ! flgFinished )
            {
              message << "ERROR: Could not execute: " << args.progSegEM << " ( " 
                      << callSegEM.errorString().toStdString() << " )" << std::endl;
              args.PrintMessage( message );
              
              continue;
            }
            
            args.PrintMessage( message );

            callSegEM.close();

            args.ReadImageFromFile( args.dirOutput, 
                                    fileOutputParenchyma, 
                                    "breast parenchyma", imParenchyma );
          }


          progress = ( iDirectory + ( iMask + ( iImage + 0.6 )/nImages )/nMasks )/nDirectories;
          std::cout  << std::endl << "<filter-progress>" << std::endl
                     << progress << std::endl
                     << "</filter-progress>" << std::endl << std::endl;


          // Compute the breast density
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~

          if ( imParenchyma && imMask )
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
              maskIterator( imMask, imMask->GetLargestPossibleRegion() );

            itk::ImageRegionConstIterator< ImageType > 
              segmIterator( imParenchyma, imParenchyma->GetLargestPossibleRegion() );

            itk::ImageRegionConstIterator< ImageType > 
              imIterator( imInput, imInput->GetLargestPossibleRegion() );

            ImageType::SpacingType spacing = imParenchyma->GetSpacing();

            float voxelVolume = spacing[0]*spacing[1]*spacing[2];

            ImageType::RegionType region;
            region = imMask->GetLargestPossibleRegion();

            ImageType::SizeType lateralSize;
            lateralSize = region.GetSize();
            lateralSize[0] = lateralSize[0]/2;

            ImageType::IndexType idx;
   
            for ( maskIterator.GoToBegin(), segmIterator.GoToBegin(), imIterator.GoToBegin();
                  ! maskIterator.IsAtEnd();
                  ++maskIterator, ++segmIterator, ++imIterator )
            {
              if ( maskIterator.Get() )
              {
                idx = maskIterator.GetIndex();

                // Left breast

                if ( idx[0] < (int) lateralSize[0] )
                {
                  nLeftVoxels++;
                  leftDensity += segmIterator.Get();
                }

                // Right breast

                else 
                {
                  nRightVoxels++;
                  rightDensity += segmIterator.Get();
                }

                // Both breasts

                totalDensity += segmIterator.Get();

                // Ensure we have the polarity correct by calculating the
                // mean intensities of each class

                if ( segmIterator.Get() > 0.5 )
                {
                  meanOfHighProbIntensities += imIterator.Get();
                  nHighProbIntensities++;
                }
                else
                {
                  meanOfLowProbIntensities += imIterator.Get();
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

            // Fat should be high intensity in the T2 image so if the
            // dense region (high prob) has a high intensity then it is
            // probably fat and we need to invert the density

            if ( (     args.flgFatIsBright   && ( meanOfHighProbIntensities > meanOfLowProbIntensities ) ) ||
                 ( ( ! args.flgFatIsBright ) && ( meanOfHighProbIntensities < meanOfLowProbIntensities ) ) )
            {
              message << std::endl << "Inverting the density estimation" << std::endl;
              args.PrintWarning( message );
        
              leftDensity  = 1. - leftDensity;
              rightDensity = 1. - rightDensity;
              totalDensity = 1. - totalDensity;

              itk::ImageRegionIterator< ImageType > 
              imIterator( imParenchyma, imParenchyma->GetLargestPossibleRegion() );
              
              itk::ImageRegionConstIterator< ImageType > 
              maskIterator( imMask, imMask->GetLargestPossibleRegion() );
   
              for ( maskIterator.GoToBegin(), imIterator.GoToBegin();
                    ! maskIterator.IsAtEnd();
                    ++maskIterator, ++imIterator )
              {
                if ( maskIterator.Get() )
                {
                  imIterator.Set( 1. - imIterator.Get() );
                }
              }

              args.WriteImageToFile( fileOutputParenchyma, 
                                     std::string( "inverted parenchyma image" ), 
                                     imParenchyma );
            }
        
  
            float leftBreastVolume = nLeftVoxels*voxelVolume;
            float rightBreastVolume = nRightVoxels*voxelVolume;

            message << "Number of left breast voxels: " << nLeftVoxels << std::endl
                    << "Volume of left breast: " << leftBreastVolume << " mm^3" << std::endl
                    << "Density of left breast (fraction of glandular tissue): " << leftDensity 
                    << std::endl << std::endl
        
                    << "Number of right breast voxels: " << nRightVoxels << std::endl
                    << "Volume of right breast: " << rightBreastVolume << " mm^3" << std::endl
                    << "Density of right breast (fraction of glandular tissue): " << rightDensity 
                    << std::endl << std::endl
        
                    << "Total number of breast voxels: " 
                    << nLeftVoxels + nRightVoxels << std::endl
                    << "Total volume of both breasts: " 
                    << leftBreastVolume + rightBreastVolume << " mm^3" << std::endl
                    << "Combined density of both breasts (fraction of glandular tissue): " 
                    << totalDensity << std::endl << std::endl;
            args.PrintMessage( message );


            if ( fileDensityMeasurements.length() != 0 ) 
            {
              std::string fileOutputDensityMeasurements 
                = niftk::ConcatenatePath( args.dirOutput, fileDensityMeasurements );

              std::ofstream fout( fileOutputDensityMeasurements.c_str() );

              fout.precision(16);

              if ((! fout) || fout.bad()) 
              {
                message << "ERROR: Could not open file: " << fileDensityMeasurements << std::endl;
                args.PrintError( message );
                continue;
              }

              fout << "Study ID, "
                   << "Image, "
                   << "Mask, "

                                    
                   << "Number of left breast voxels, "
                   << "Volume of left breast (mm^3), "
                   << "Density of left breast (fraction of glandular tissue), "
      
                   << "Number of right breast voxels, "
                   << "Volume of right breast (mm^3), "
                   << "Density of right breast (fraction of glandular tissue), "
      
                   << "Total number of breast voxels, "
                   << "Total volume of both breasts (mm^3), "
                   << "Combined density of both breasts (fraction of glandular tissue)" 
                   << std::endl;

              fout << dirBaseName << ", "
                   << fileImageBasename << ", "
                   << fileMaskBasename << ", "

                                  
                   << nLeftVoxels << ", "
                   << leftBreastVolume << ", "
                   << leftDensity << ", "
      
                   << nRightVoxels << ", "
                   << rightBreastVolume << ", "
                   << rightDensity << ", "
      
                   << nLeftVoxels + nRightVoxels << ", "
                   << leftBreastVolume + rightBreastVolume << ", "
                   << totalDensity << std::endl;
    
              fout.close();

              message  << "Density measurements written to file: " 
                       << fileOutputDensityMeasurements  << std::endl << std::endl;
              args.PrintMessage( message );
            }

            // Write the data to the main collated csv file

            if ( args.foutOutputCSV )
            {
              if ( flgVeryFirstRow )    // Include the title row?
              {
                *args.foutOutputCSV << "Study ID, "
                                    << "Image, "
                                    << "Mask, "

                                    << "Number of left breast voxels, "
                                    << "Volume of left breast (mm^3), "
                                    << "Density of left breast (fraction of glandular tissue), "
              
                                    << "Number of right breast voxels, "
                                    << "Volume of right breast (mm^3), "
                                    << "Density of right breast (fraction of glandular tissue), "
              
                                    << "Total number of breast voxels, "
                                    << "Total volume of both breasts (mm^3), "
                                    << "Combined density of both breasts (fraction of glandular tissue)" 
                                    << std::endl;
                flgVeryFirstRow = false;
              }

              *args.foutOutputCSV << dirBaseName << ", "
                                  << fileImageBasename << ", "
                                  << fileMaskBasename << ", "

                                  << nLeftVoxels << ", "
                                  << leftBreastVolume << ", "
                                  << leftDensity << ", "
            
                                  << nRightVoxels << ", "
                                  << rightBreastVolume << ", "
                                  << rightDensity << ", "
      
                                  << nLeftVoxels + nRightVoxels << ", "
                                  << leftBreastVolume + rightBreastVolume << ", "
                                  << totalDensity << std::endl;
            }
            else
            {
              message << "Collated csv data file: " << fileOutputCSV 
                      << " is not open, data will not be written." << std::endl;
              args.PrintWarning( message );
            }
          }

          progress = ( iDirectory + ( iMask + ( iImage + 0.9 )/nImages )/nMasks )/nDirectories;
          std::cout  << std::endl << "<filter-progress>" << std::endl
                     << progress << std::endl
                     << "</filter-progress>" << std::endl << std::endl;
    
          
        }
        catch (itk::ExceptionObject &ex)
        {
          message << ex << std::endl;
          args.PrintError( message );
          continue;
        }

        progress = ( iDirectory + ( iMask + iImage/nImages )/nMasks )/nDirectories;
        std::cout  << std::endl << "<filter-progress>" << std::endl
                   << progress << std::endl
                   << "</filter-progress>" << std::endl << std::endl;
      }
    }
  }


  progress = iDirectory/nDirectories;
  std::cout  << std::endl << "<filter-progress>" << std::endl
             << progress << std::endl
             << "</filter-progress>" << std::endl << std::endl;
  
  return EXIT_SUCCESS;
}
 
 

