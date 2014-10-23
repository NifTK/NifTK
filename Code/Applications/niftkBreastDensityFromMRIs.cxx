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


#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

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


//#define LINK_TO_SEG_EM

#ifdef LINK_TO_SEG_EM
#include <_seg_EM.h>
#endif

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
// class InputParameters
// -------------------------------------------------------------------------

class InputParameters
{

public:

  bool flgVerbose;
  bool flgSaveImages;
  bool flgDebug;
  bool flgCompression;

  std::string dirInput;

  std::string fileLog;
  std::string fileOutputCSV;

  std::string dirSub;  
  std::string dirPrefix;  

  std::ofstream *foutLog;
  std::ofstream *foutOutputCSV;
  std::ostream *newCout;

  typedef tee_device<ostream, ofstream> TeeDevice;
  typedef stream<TeeDevice> TeeStream;

  TeeDevice *teeDevice;
  TeeStream *teeStream;

  InputParameters( TCLAP::CmdLine commandLine, 
                   bool verbose, bool save, bool compression, bool debug,
                   std::string subdirectory, std::string prefix, std::string input,
                   std::string logfile, std::string csvfile ) {

    std::stringstream message;

    flgVerbose = verbose;
    flgSaveImages = save;
    flgDebug = debug;
    flgCompression = compression;

    dirSub = subdirectory;
    dirPrefix = prefix;
    dirInput = input;

    fileLog = logfile;
    fileOutputCSV = csvfile;
    
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
      newCout = 0;
      foutLog = 0;
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

   }

  void Print(void) {

    std::stringstream message;

    message << std::endl
            << "Examining directory: " << dirInput << std::endl 
            << std::endl
            << "Verbose output?: "   << std::boolalpha << flgVerbose     
            << std::noboolalpha << std::endl
            << "Save images?: "      << std::boolalpha << flgSaveImages  
            << std::noboolalpha << std::endl
            << "Compress images?: "  << std::boolalpha << flgCompression 
            << std::noboolalpha << std::endl
            << "Debugging output?: " << std::boolalpha << flgDebug       
            << std::noboolalpha << std::endl
            << std::endl
            << "Data sub-directory: " << dirSub << std::endl
            << "Study directory prefix: " << dirPrefix << std::endl
            << std::endl
            << "Output log file: " << fileLog << std::endl
            << "Output csv file: " << fileOutputCSV << std::endl
            << std::endl;

    PrintMessage( message );
  }
    
  void PrintMessage( std::stringstream &message ) {

    std::cout << message.str();
    message.str( "" );
    teeStream->flush();
  }
    
  void PrintErrorAndExit( std::stringstream &message ) {

    std::cerr << "ERROR: " << message.str();
    message.str( "" );
    teeStream->flush();

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

  bool ReadImageFromFile( std::string dirInput, std::string filename, 
                         const char *description, ImageType::Pointer &image ) {
  
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
                         const char *description, nifti_image *&image ) {
  
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

  void WriteImageToFile( std::string dirOutput, std::string filename, 
                         const char *description, ImageType::Pointer image ) {
  
    std::stringstream message;
    std::string fileOutput = niftk::ConcatenatePath( dirOutput, filename );
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    itk::WriteImageToFile< ImageType >( fileOutput, image );
  }

  void WriteImageToFile( std::string dirOutput, std::string filename, 
                         const char *description, nifti_image *image ) {
  
    std::stringstream message;
    std::string fileOutput = niftk::ConcatenatePath( dirOutput, filename );
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    nifti_image_write( image );
  }

  void DeleteFile( std::string dirOutput, std::string filename ) {

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
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  bool flgVeryFirstRow = true;

  float progress = 0.;
  float iDirectory = 1.;
  float nDirectories;

  std::stringstream message;

  std::vector< std::string > fileNamesStructuralT2;
  std::vector< std::string > fileNamesFatSatT1;
  std::vector< std::string > fileNamesDixonWater;
  std::vector< std::string > fileNamesDixonFat;

  typedef itk::ImageSeriesReader< ImageType > SeriesReaderType;
  typedef itk::UCLN4BiasFieldCorrectionFilter< ImageType, ImageType > BiasFieldCorrectionType;



  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  InputParameters args( commandLine, 
                        flgVerbose, flgSaveImages, flgCompression, flgDebug,
                        dirSub, dirPrefix, dirInput,
                        fileLog, fileOutputCSV );                     


  args.Print();



  // Initialise the output file names
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string fileI00_t2_tse_tra( "I00_t2_tse_tra.nii" );
  std::string fileI00_t1_fl3d_tra_VIBE( "I00_t1_fl3d_tra_VIBE.nii" );

  std::string fileI00_sag_dixon_bilateral_W( "I00_sag_dixon_bilateral_W.nii" );
  std::string fileI00_sag_dixon_bilateral_F( "I00_sag_dixon_bilateral_F.nii" );

  std::string fileI01_t2_tse_tra_BiasFieldCorrection( "I01_t2_tse_tra_BiasFieldCorrection.nii" );
  std::string fileI01_t1_fl3d_tra_VIBE_BiasFieldCorrection( "I01_t1_fl3d_tra_VIBE_BiasFieldCorrection.nii" );

  std::string fileI02_t2_tse_tra_Resampled( "I02_t2_tse_tra_Resampled.nii" );

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

  std::string fileOutputVTKSurface( "I13_BreastSurface.vtk" );

  std::string fileOutputBreastMask( "I14_BreastMaskSegmentation.nii" );
  std::string fileOutputDixonMask( "I14_DixonMaskSegmentation.nii" );

  std::string fileOutputParenchyma( "I15_BreastParenchyma.nii" );

  std::string fileDensityMeasurements( "I16_DensityMeasurements.csv" );



  if ( flgCompression )
  {

    fileI00_t2_tse_tra.append( ".gz" );
    fileI00_t1_fl3d_tra_VIBE.append( ".gz" );

    fileI00_sag_dixon_bilateral_W.append( ".gz" );
    fileI00_sag_dixon_bilateral_F.append( ".gz" );

    fileI01_t2_tse_tra_BiasFieldCorrection.append( ".gz" );
    fileI01_t1_fl3d_tra_VIBE_BiasFieldCorrection.append( ".gz" );

    fileI02_t2_tse_tra_Resampled.append( ".gz" );

    fileOutputBreastMask.append( ".gz" );
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

    if ( fileOutputParenchyma.length() > 0 )            fileOutputParenchyma.append( ".gz" );
  }


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string dirFullPath;
  std::string dirBaseName;

  std::string dirOutput;

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
    
    ImageType::Pointer imParenchyma = 0;
    
    progress = iDirectory/nDirectories;
    std::cout  << std::endl << "<filter-progress>" << std::endl
               << progress << std::endl
               << "</filter-progress>" << std::endl << std::endl;

    dirFullPath = *iterDirectoryNames;
    dirBaseName = niftk::Basename( dirFullPath );

    if ( ! dirBaseName.compare( 0, args.dirPrefix.length(), args.dirPrefix ) == 0 )
    {
      message << std::endl << "Skipping directory: " << dirFullPath << std::endl << std::endl;
      args.PrintMessage( message );
      continue;
    }

    message << std::endl << "Directory: " << dirFullPath << std::endl << std::endl;
    args.PrintMessage( message );

    if ( dirSub.length() > 0 )
    {
      dirOutput = niftk::ConcatenatePath( dirFullPath, dirSub );
    }
    else
    {
      dirOutput = dirFullPath;
    }

    if ( ! niftk::DirectoryExists( dirOutput ) )
    {
      niftk::CreateDirAndParents( dirOutput );

      message << "Creating output directory: " << dirOutput << std::endl;
      args.PrintMessage( message );
    }

    try
    {      

      // If the CSV file has already been generated then read it
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      std::string fileInputDensityMeasurements  
        = niftk::ConcatenatePath( dirOutput, fileDensityMeasurements );

      if ( niftk::FileExists( fileInputDensityMeasurements ) )
      {
        std::ifstream fin( fileInputDensityMeasurements.c_str() );

        if ((! fin) || fin.bad()) 
        {
          message << "ERROR: Could not open file: " << fileDensityMeasurements << std::endl;
          args.PrintErrorAndExit( message );
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
    
      nameGenerator->SetDirectory( dirFullPath );
  
    
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
          args.PrintErrorAndExit( message );
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

        if ( seriesDescription.find( std::string( "t2_tse_tra" ) ) != std::string::npos )
        {
          fileNamesStructuralT2 = fileNames;
        }
        else if ( seriesDescription.find( std::string( "t1_fl3d_tra_VIBE" ) ) != std::string::npos )
        {
          fileNamesFatSatT1 = fileNames;
        }
        else if ( seriesDescription.find( std::string( "sag_dixon_bilateral_W" ) ) != std::string::npos )
        {
          fileNamesDixonWater = fileNames;
        }
        else if ( seriesDescription.find( std::string( "sag_dixon_bilateral_F" ) ) != std::string::npos )
        {
          fileNamesDixonFat = fileNames;
        }


        seriesItr++;
      }


      // Load the fat sat image
      // ~~~~~~~~~~~~~~~~~~~~~~
        
      if ( ! args.ReadImageFromFile( dirOutput, fileI01_t1_fl3d_tra_VIBE_BiasFieldCorrection, 
                                     "bias field corrected 't1_fl3d_tra_VIBE' image", 
                                     imFatSatT1 ) )
      {
        if ( ! args.ReadImageFromFile( dirOutput, fileI00_t1_fl3d_tra_VIBE, 
                                       "fat sat 't1_fl3d_tra_VIBE' image", imFatSatT1 ) )
        {
          if ( fileNamesFatSatT1.size() > 0 )
          {
            
            SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
            seriesReader->SetFileNames( fileNamesFatSatT1 );

            message << std::endl << "Reading 't1_fl3d_tra_VIBE' image" << std::endl;  
            args.PrintMessage( message );

            seriesReader->UpdateLargestPossibleRegion();

            imFatSatT1 = seriesReader->GetOutput();
            imFatSatT1->DisconnectPipeline();
            
            args.WriteImageToFile( dirOutput, fileI00_t1_fl3d_tra_VIBE, 
                                   "fat sat 't1_fl3d_tra_VIBE' image", imFatSatT1 );
          }
        }

        // Bias field correct it

        if ( imFatSatT1 ) 
        {
          
          message << std::endl << "Bias field correcting 't1_fl3d_tra_VIBE' image"
                  << std::endl;  
          args.PrintMessage( message );
          
          BiasFieldCorrectionType::Pointer biasFieldCorrector = BiasFieldCorrectionType::New();
          
          biasFieldCorrector->SetInput( imFatSatT1 );
          biasFieldCorrector->Update();
          
          imFatSatT1 = biasFieldCorrector->GetOutput();
          imFatSatT1->DisconnectPipeline();
          
          args.WriteImageToFile( dirOutput, fileI01_t1_fl3d_tra_VIBE_BiasFieldCorrection, 
                                 "bias field corrected 't1_fl3d_tra_VIBE' image", imFatSatT1 );
        }
      }


      // Load the structural image
      // ~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( ! args.ReadImageFromFile( dirOutput, fileI02_t2_tse_tra_Resampled, 
                                     "resampled 't2_tse_tra' image", imStructuralT2 ) )
      {
        if ( ! args.ReadImageFromFile( dirOutput, fileI01_t2_tse_tra_BiasFieldCorrection, 
                                       "bias field corrected 't2_tse_tra' image", imStructuralT2 ) )
        {
          if ( ! args.ReadImageFromFile( dirOutput, fileI00_t2_tse_tra,
                                         "structural 't2_tse_tra' image", imStructuralT2 ) )
          {
            if ( fileNamesStructuralT2.size() > 0 )
            {
            
              SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
              seriesReader->SetFileNames( fileNamesStructuralT2 );

              message << std::endl << "Reading 't2_tse_tra' image" << std::endl;  
              args.PrintMessage( message );

              seriesReader->UpdateLargestPossibleRegion();

              imStructuralT2 = seriesReader->GetOutput();
              imStructuralT2->DisconnectPipeline();

              args.WriteImageToFile( dirOutput, fileI00_t2_tse_tra,
                                     "structural 't2_tse_tra' image", imStructuralT2 );
            }
          }

          // Bias field correct it
          
          if ( imStructuralT2 )
          {
            message << std::endl << "Bias field correcting 't2_tse_tra' image" << std::endl;  
            args.PrintMessage( message );

            BiasFieldCorrectionType::Pointer biasFieldCorrector = BiasFieldCorrectionType::New();

            biasFieldCorrector->SetInput( imStructuralT2 );
            biasFieldCorrector->Update();

            imStructuralT2 = biasFieldCorrector->GetOutput();
            imStructuralT2->DisconnectPipeline();
            
            args.WriteImageToFile( dirOutput, fileI01_t2_tse_tra_BiasFieldCorrection, 
                                   "bias field corrected 't2_tse_tra' image", imStructuralT2 );
          }
        }
      
        // Resample the T2 image to match the FatSat image

        if ( imStructuralT2 && imFatSatT1 ) 
        {
          typedef itk::IdentityTransform<double, Dimension> TransformType;
          TransformType::Pointer identityTransform = TransformType::New();

          typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
          ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

          resampleFilter->SetUseReferenceImage( true ); 
          resampleFilter->SetReferenceImage( imFatSatT1 ); 

          resampleFilter->SetTransform( identityTransform );

          resampleFilter->SetInput( imStructuralT2 );

          resampleFilter->Update();

          imStructuralT2 = resampleFilter->GetOutput();
          imStructuralT2->DisconnectPipeline();

          args.WriteImageToFile( dirOutput, fileI02_t2_tse_tra_Resampled, 
                                 "resampled 't2_tse_tra' image", imStructuralT2 );
        }
      }


      // Have we found both input images?

      if ( ! ( imStructuralT2 && imFatSatT1 ) )
      {
        message << "Both of structural T2 and fat saturated T1 images not found, "
                << "skipping this directory: " << std::endl << std::endl;
        args.PrintWarning( message );
        continue;
      }
          

      // Run the breast mask segmentation?
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( ! args.ReadImageFromFile( dirOutput, fileOutputBreastMask, 
                                     "segmented breast mask", imSegmentedBreastMask ) )
      {

        bool flgSmooth = true;

        bool flgLeft = false;
        bool flgRight = false;

        bool flgExtInitialPect = false;

        int regGrowXcoord = 0;
        int regGrowYcoord = 0;
        int regGrowZcoord = 0;

        float bgndThresholdProb = 0.5;

        float finalSegmThreshold = 0.49;

        float sigmaInMM = 5;

        float fMarchingK1   = 30.0;
        float fMarchingK2   = 15.0;
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
          if ( fileOutputBackground.length() > 0 )            breastMaskSegmentor->SetOutputBackground(           niftk::ConcatenatePath( dirOutput, fileOutputBackground ) );
          if ( fileOutputSkinElevationMap.length() > 0 )      breastMaskSegmentor->SetOutputSkinElevationMap(     niftk::ConcatenatePath( dirOutput, fileOutputSkinElevationMap ) );
          if ( fileOutputChestPoints.length() > 0 )           breastMaskSegmentor->SetOutputChestPoints(          niftk::ConcatenatePath( dirOutput, fileOutputChestPoints ) );
          if ( fileOutputPectoral.length() > 0 )              breastMaskSegmentor->SetOutputPectoralMask(         niftk::ConcatenatePath( dirOutput, fileOutputPectoral ) );
          if ( fileOutputPectoralSurfaceMask.length() > 0 )   breastMaskSegmentor->SetOutputPecSurfaceMask(       niftk::ConcatenatePath( dirOutput, fileOutputPectoralSurfaceMask ) );

          if ( fileOutputGradientMagImage.length() > 0 )      breastMaskSegmentor->SetOutputGradientMagImage(     niftk::ConcatenatePath( dirOutput, fileOutputGradientMagImage ) );
          if ( fileOutputSpeedImage.length() > 0 )            breastMaskSegmentor->SetOutputSpeedImage(           niftk::ConcatenatePath( dirOutput, fileOutputSpeedImage ) );
          if ( fileOutputFastMarchingImage.length() > 0 )     breastMaskSegmentor->SetOutputFastMarchingImage(    niftk::ConcatenatePath( dirOutput, fileOutputFastMarchingImage ) );
  
          if ( fileOutputPectoralSurfaceVoxels.length() > 0 ) breastMaskSegmentor->SetOutputPectoralSurf(         niftk::ConcatenatePath( dirOutput, fileOutputPectoralSurfaceVoxels ) );
  
          if ( fileOutputFittedBreastMask.length() > 0 )      breastMaskSegmentor->SetOutputBreastFittedSurfMask( niftk::ConcatenatePath( dirOutput, fileOutputFittedBreastMask ) );

          if ( fileOutputVTKSurface.length() > 0 )            breastMaskSegmentor->SetOutputVTKSurface(           niftk::ConcatenatePath( dirOutput, fileOutputVTKSurface ) );
        }

        if ( args.ReadImageFromFile( dirOutput, fileBIFs, "BIF image", imBIFs ) )
        {
          breastMaskSegmentor->SetBIFImage( imBIFs );
        }
        else
        {
          breastMaskSegmentor->SetOutputBIFS( niftk::ConcatenatePath( dirOutput, fileBIFs ) );
        }        

        breastMaskSegmentor->SetStructuralImage( imStructuralT2 );
        breastMaskSegmentor->SetFatSatImage( imFatSatT1 );

        breastMaskSegmentor->Execute();

        imSegmentedBreastMask = breastMaskSegmentor->GetSegmentedImage();

        args.WriteImageToFile( dirOutput, fileOutputBreastMask, 
                               "breast mask segmentation image", imSegmentedBreastMask );

      }


      // Load the Dixon Water Image
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( ! args.ReadImageFromFile( dirOutput, fileI00_sag_dixon_bilateral_W, 
                                     "Dixon water 'sag_dixon_bilateral_W' image", imDixonWater ) )
      {
        if ( fileNamesDixonWater.size() > 0 )
        {
          
          SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
          seriesReader->SetFileNames( fileNamesDixonWater );
          
          message << std::endl << "Reading 'sag_dixon_bilateral_W' image" << std::endl;  
          args.PrintMessage( message );
          
          seriesReader->UpdateLargestPossibleRegion();
          
          imDixonWater = seriesReader->GetOutput();
          imDixonWater->DisconnectPipeline();
          
          args.WriteImageToFile( dirOutput, fileI00_sag_dixon_bilateral_W, 
                                 "Dixon water 'sag_dixon_bilateral_W' image", imDixonWater );
        }
      }


      // Load the Dixon Fat Image
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( ! args.ReadImageFromFile( dirOutput, fileI00_sag_dixon_bilateral_F, 
                                     "Dixon fat 'sag_dixon_bilateral_F' image", imDixonFat ) )
      {
        if ( fileNamesDixonFat.size() > 0 )
        {
          
          SeriesReaderType::Pointer seriesReader = SeriesReaderType::New();
          seriesReader->SetFileNames( fileNamesDixonFat );
          
          message << std::endl << "Reading 'sag_dixon_bilateral_F' image" << std::endl;  
          args.PrintMessage( message );
          
          seriesReader->UpdateLargestPossibleRegion();
          
          imDixonFat = seriesReader->GetOutput();
          imDixonFat->DisconnectPipeline();
          
          args.WriteImageToFile( dirOutput, fileI00_sag_dixon_bilateral_F, 
                                 "Dixon fat 'sag_dixon_bilateral_F' image", imDixonFat );
        }
      }


      // Resample the breast mask to match the Dixon images
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      if ( ! args.ReadImageFromFile( dirOutput, fileOutputDixonMask, 
                                     "Dixon breast mask", imDixonBreastMask ) )
      {

        if ( imDixonWater && imDixonFat ) 
        {
          typedef itk::IdentityTransform<double, Dimension> TransformType;
          TransformType::Pointer identityTransform = TransformType::New();

          typedef itk::NearestNeighborInterpolateImageFunction<ImageType, double> InterpolatorType;
          InterpolatorType::Pointer interpolator = InterpolatorType::New();

          typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
          ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
        
          resampleFilter->SetUseReferenceImage( true ); 
          resampleFilter->SetReferenceImage( imDixonWater ); 
        
          resampleFilter->SetTransform( identityTransform );
          resampleFilter->SetInterpolator( interpolator );
        
          resampleFilter->SetInput( imSegmentedBreastMask );
        
          resampleFilter->Update();
        
          imDixonBreastMask = resampleFilter->GetOutput();
          imDixonBreastMask->DisconnectPipeline();
        
          args.WriteImageToFile( dirOutput, fileOutputDixonMask, 
                                 "Dixon breast mask", imDixonBreastMask );
        }
      }
    
      
      // Segment the parenchyma
      // ~~~~~~~~~~~~~~~~~~~~~~

#ifdef LINK_TO_SEG_EM

      nifti_image *niftiParenchyma;

      if ( ! args.ReadImageFromFile( dirOutput, fileOutputParenchyma, 
                                     "breast parenchyma", niftiParenchyma ) )
      {

        nifti_image *niftiStructuralT2 = 
          ConvertITKImageToNiftiImage< PixelType, PixelType, Dimension >( imStructuralT2 );
        
        nifti_image *niftiMask = 
          ConvertITKImageToNiftiImage< PixelType, PixelType, Dimension >( imSegmentedBreastMask );
        


        int nPriors = 2;
        int NumbMultiSpec = 1;
        int NumbTimePoints = 1;
        int verboseLevel = 2;
        
        seg_EM SEG( nPriors, NumbMultiSpec, NumbTimePoints);
        
        SEG.SetInputImage( niftiStructuralT2 );
        SEG.SetMaskImage( niftiMask );

        SEG.SetVerbose( verboseLevel );

        SEG.SetFilenameOut( niftk::ConcatenatePath( dirOutput, fileOutputParenchyma ) );

        SEG.Turn_MRF_ON( 0.4 );

        SEG.Run_EM();

        niftiParenchyma = SEG.GetResult();

        args.WriteImageToFile( dirOutput, fileOutputParenchyma, 
                               "breast mask segmentation image", niftiParenchyma );

      }

#else

      if ( ! args.ReadImageFromFile( dirOutput, fileOutputParenchyma, 
                                     "breast parenchyma", imParenchyma ) )
      {

        std::stringstream commandNiftySeg;

        commandNiftySeg 
          << "seg_EM -v 2 -bc_order 4 -nopriors 2" 
          << " -in \"" << niftk::ConcatenatePath( dirOutput, fileI02_t2_tse_tra_Resampled ) << "\" "
          << " -mask \"" << niftk::ConcatenatePath( dirOutput, fileOutputBreastMask ) << "\" "
          << " -out \"" << niftk::ConcatenatePath( dirOutput, fileOutputParenchyma ) << "\" ";

        message << std::endl << "Executing parenchyma segmentation: "
                << std::endl << "   " << commandNiftySeg.str() << std::endl << std::endl;
        args.PrintMessage( message );

        system( commandNiftySeg.str().c_str() );

      }

#endif


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
          maskIterator( imSegmentedBreastMask, imSegmentedBreastMask->GetLargestPossibleRegion() );

        itk::ImageRegionConstIterator< ImageType > 
          segmIterator( imParenchyma, imParenchyma->GetLargestPossibleRegion() );

        itk::ImageRegionConstIterator< ImageType > 
          imIterator( imStructuralT2, imStructuralT2->GetLargestPossibleRegion() );

        ImageType::SpacingType spacing = imParenchyma->GetSpacing();

        float voxelVolume = spacing[0]*spacing[1]*spacing[2];

        ImageType::RegionType region;
        region = imSegmentedBreastMask->GetLargestPossibleRegion();

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

        if ( meanOfHighProbIntensities > meanOfLowProbIntensities )
        {
          message << std::endl << "Inverting the density estimation" << std::endl;
          args.PrintWarning( message );
        
          leftDensity = 1. - leftDensity;
          rightDensity = 1. - rightDensity;
          totalDensity = 1. - totalDensity;
        }
        
  
        float leftBreastVolume = nLeftVoxels*voxelVolume;
        float rightBreastVolume = nRightVoxels*voxelVolume;

        leftDensity /= nLeftVoxels;
        rightDensity /= nRightVoxels;
        totalDensity /= ( nLeftVoxels + nRightVoxels);

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
            = niftk::ConcatenatePath( dirOutput, fileDensityMeasurements );

          std::ofstream fout( fileOutputDensityMeasurements .c_str() );

          fout.precision(16);

          if ((! fout) || fout.bad()) 
          {
            message << "ERROR: Could not open file: " << fileDensityMeasurements << std::endl;
            args.PrintErrorAndExit( message );
          }

          fout << "Study ID, "
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
      }

      // Delete unwanted images
      // ~~~~~~~~~~~~~~~~~~~~~~

      if ( ! ( flgDebug || flgSaveImages ) )
      {
        args.DeleteFile( dirOutput, fileI01_t2_tse_tra_BiasFieldCorrection );
        args.DeleteFile( dirOutput, fileI01_t1_fl3d_tra_VIBE_BiasFieldCorrection );
        args.DeleteFile( dirOutput, fileI02_t2_tse_tra_Resampled );

        args.DeleteFile( dirOutput, fileBIFs );

        args.DeleteFile( dirOutput, fileOutputSmoothedStructural );
        args.DeleteFile( dirOutput, fileOutputSmoothedFatSat );
        args.DeleteFile( dirOutput, fileOutputClosedStructural );

        args.DeleteFile( dirOutput, fileOutputMaxImage );
        args.DeleteFile( dirOutput, fileOutputCombinedHistogram );
        args.DeleteFile( dirOutput, fileOutputRayleigh );
        args.DeleteFile( dirOutput, fileOutputFreqLessBgndCDF );
        args.DeleteFile( dirOutput, fileOutputBackground );
        args.DeleteFile( dirOutput, fileOutputSkinElevationMap );
        args.DeleteFile( dirOutput, fileOutputGradientMagImage );
        args.DeleteFile( dirOutput, fileOutputSpeedImage );
        args.DeleteFile( dirOutput, fileOutputFastMarchingImage );
        args.DeleteFile( dirOutput, fileOutputPectoral );
        args.DeleteFile( dirOutput, fileOutputChestPoints );
        args.DeleteFile( dirOutput, fileOutputPectoralSurfaceMask );

        args.DeleteFile( dirOutput, fileOutputPectoralSurfaceVoxels );

        args.DeleteFile( dirOutput, fileOutputFittedBreastMask );

        args.DeleteFile( dirOutput, fileOutputVTKSurface );
      }
    }
    catch (itk::ExceptionObject &ex)
    {
      message << ex << std::endl;
      args.PrintErrorAndExit( message );
    }
  }
    
  progress = iDirectory/nDirectories;
  std::cout  << std::endl << "<filter-progress>" << std::endl
             << progress << std::endl
             << "</filter-progress>" << std::endl << std::endl;
  
  return EXIT_SUCCESS;
}
 
 

