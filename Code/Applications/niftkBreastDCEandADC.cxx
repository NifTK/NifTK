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
  std::string dirOutput;

  std::string fileLog;
  std::string fileOutputCSV;

  std::string strSeriesDescT1W;
  std::string strSeriesDescT2W;
  std::string strSeriesDescADC;
  std::string strSeriesDescDCE;

  std::string fileSegEM;

  std::ofstream *foutLog;
  std::ofstream *foutOutputCSV;
  std::ostream *newCout;

  typedef tee_device<ostream, ofstream> TeeDevice;
  typedef stream<TeeDevice> TeeStream;

  TeeDevice *teeDevice;
  TeeStream *teeStream;

  InputParameters( TCLAP::CmdLine &commandLine, 
                   bool verbose, bool flgSave, bool compression, bool debug,
                   std::string dInput, std::string dOutput,
                   std::string logfile, std::string csvfile,
                   std::string strT1,
                   std::string strT2,
                   std::string strADC,
                   std::string strDCE,
                   std::string segEM ) {

    std::stringstream message;

    flgVerbose = verbose;
    flgSaveImages = flgSave;
    flgDebug = debug;
    flgCompression = compression;

    dirInput  = dInput;
    dirOutput = dOutput;

    fileLog = logfile;
    fileOutputCSV = csvfile;

    strSeriesDescT1W = strT1;
    strSeriesDescT2W = strT2;
    strSeriesDescADC = strADC;
    strSeriesDescDCE = strDCE;

    fileSegEM = segEM;

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
            << "Input DICOM directory: " << dirInput << std::endl 
            << "Output directory: " << dirInput << std::endl 
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
            << "Output log file: " << fileLog << std::endl
            << "Output csv file: " << fileOutputCSV << std::endl
            << std::endl
            << "T1W series description:          " << strSeriesDescT1W << std::endl
            << "T2W series description:          " << strSeriesDescT2W << std::endl
            << "ADC map series description:      " << strSeriesDescADC << std::endl
            << "DCE sequence series description: " << strSeriesDescDCE << std::endl
            << std::endl
            << "NiftySeg 'seg_EM' executable: " << fileSegEM << std::endl
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

  void WriteImageToFile( std::string outDir, std::string filename, 
                         std::string description, ImageType::Pointer image ) {
  
    std::stringstream message;
    std::string fileOutput = niftk::ConcatenatePath( outDir, filename );
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    itk::WriteImageToFile< ImageType >( fileOutput, image );
  }

  void WriteImageToFile( std::string outDir, std::string filename, 
                         std::string description, nifti_image *image ) {
  
    std::stringstream message;
    std::string fileOutput = niftk::ConcatenatePath( outDir, filename );
              
    message << std::endl << "Writing " << description << " to file: "
            << fileOutput << std::endl;
    PrintMessage( message );

    nifti_image_write( image );
  }

  void DeleteFile( std::string outDir, std::string filename ) {

    std::stringstream message;
    std::string filePath = niftk::ConcatenatePath( outDir, filename );

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
  std::stringstream ss;

  std::vector< std::string > fileNamesT1;
  std::vector< std::string > fileNamesT2;
  std::vector< std::string > fileNamesADC;

  typedef std::map< unsigned int, std::vector< std::string > > SeriesDCEMapType;
  SeriesDCEMapType seriesDCE;

  std::map< unsigned int, std::string > fileNamesDCEImages;

  typedef itk::ImageSeriesReader< ImageType > SeriesReaderType;
  typedef itk::UCLN4BiasFieldCorrectionFilter< ImageType, ImageType > BiasFieldCorrectionType;



  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  InputParameters args( commandLine, 
                        flgVerbose, flgSaveImages, flgCompression, flgDebug,
                        dirInput, dirOutput,
                        fileLog, fileOutputCSV,
                        strSeriesDescT1W,
                        strSeriesDescT2W,
                        strSeriesDescADC,
                        strSeriesDescDCE,
                        fileSegEM );


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

  std::string fileI02_T2_Resampled( "I02_T2_Resampled.nii" );

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

  if ( flgCompression )
  {
    fileI00_T1.append( ".gz" );
    fileI00_T2.append( ".gz" );
    fileI00_ADC.append( ".gz" );

    fileI01_T1_BiasFieldCorr.append( ".gz" );
    fileI01_T2_BiasFieldCorr.append( ".gz" );

    fileI02_T2_Resampled.append( ".gz" );

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


  }


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ImageType::Pointer imT1  = 0;
  ImageType::Pointer imT2  = 0;
  ImageType::Pointer imADC = 0;
  ImageType::Pointer imDCE = 0;

  ImageType::Pointer imSegmentedBreastMask = 0;
  ImageType::Pointer imParenchyma = 0;

  std::cout  << std::endl << "<filter-progress>" << std::endl
             << 0.2 << std::endl
             << "</filter-progress>" << std::endl << std::endl;

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


  // Load the T1W image
  // ~~~~~~~~~~~~~~~~~~
        
  if ( ! args.ReadImageFromFile( args.dirOutput, fileI01_T1_BiasFieldCorr, 
                                 std::string( "bias field corrected '") +
                                 args.strSeriesDescT1W + "' image", 
                                 imT1 ) )
  {
    if ( ! args.ReadImageFromFile( args.dirOutput, fileI00_T1, 
                                   std::string( "T1W '" ) + args.strSeriesDescT1W 
                                   + "' image", imT1 ) )
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
            
        args.WriteImageToFile( args.dirOutput, fileI00_T1, 
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
          
      args.WriteImageToFile( args.dirOutput, fileI01_T1_BiasFieldCorr, 
                             std::string( "bias field corrected '" ) + 
                             args.strSeriesDescT1W + "' image", imT1 );
    }
  }


  // Load the T2W image
  // ~~~~~~~~~~~~~~~~~~

  if ( ! args.ReadImageFromFile( args.dirOutput, fileI02_T2_Resampled, 
                                 std::string( "resampled '" ) + args.strSeriesDescT2W +
                                 "' image", imT2 ) )
  {
    if ( ! args.ReadImageFromFile( args.dirOutput, fileI01_T2_BiasFieldCorr, 
                                   std::string( "bias field corrected '" ) + 
                                   args.strSeriesDescT2W + "' image", imT2 ) )
    {
      if ( ! args.ReadImageFromFile( args.dirOutput, fileI00_T2,
                                     std::string( "T2W '" ) + 
                                     args.strSeriesDescT2W + "' image", 
                                     imT2 ) )
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

          args.WriteImageToFile( args.dirOutput, fileI00_T2,
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
            
        args.WriteImageToFile( args.dirOutput, fileI01_T2_BiasFieldCorr, 
                               std::string( "bias field corrected '" ) +
                               args.strSeriesDescT2W + "' image", imT2 );
      }
    }
      
    // Resample the T2W image to match the T1W image

    if ( imT2 && imT1 ) 
    {
      typedef itk::IdentityTransform<double, Dimension> TransformType;
      TransformType::Pointer identityTransform = TransformType::New();

      typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;
      ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

      resampleFilter->SetUseReferenceImage( true ); 
      resampleFilter->SetReferenceImage( imT1 ); 

      resampleFilter->SetTransform( identityTransform );

      resampleFilter->SetInput( imT2 );

      resampleFilter->Update();

      imT2 = resampleFilter->GetOutput();
      imT2->DisconnectPipeline();

      args.WriteImageToFile( args.dirOutput, fileI02_T2_Resampled, 
                             std::string( "resampled '" ) + args.strSeriesDescT2W +
                             "' image", imT2 );
    }
  }


  // Have we found both input images?

  if ( ! ( imT2 && imT1 ) )
  {
    message << "Both of T1W and T2W images not found" << std::endl << std::endl;
    args.PrintError( message );
    return EXIT_FAILURE;
  }
          

  // Run the breast mask segmentation?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! args.ReadImageFromFile( args.dirOutput, fileOutputBreastMask, 
                                 "segmented breast mask", 
                                 imSegmentedBreastMask ) )
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

    args.WriteImageToFile( args.dirOutput, fileOutputBreastMask, 
                           "breast mask segmentation image", imSegmentedBreastMask );

  }
      
  // Segment the parenchyma
  // ~~~~~~~~~~~~~~~~~~~~~~

#ifdef LINK_TO_SEG_EM

  nifti_image *niftiParenchyma;

  if ( ! args.ReadImageFromFile( args.dirOutput, fileOutputParenchyma, 
                                 "breast parenchyma", niftiParenchyma ) )
  {

    nifti_image *niftiStructuralT2 = 
      ConvertITKImageToNiftiImage< PixelType, PixelType, Dimension >( imT2 );
        
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

    SEG.SetFilenameOut( niftk::ConcatenatePath( args.dirOutput, fileOutputParenchyma ) );

    SEG.Turn_MRF_ON( 0.4 );

    SEG.Run_EM();

    niftiParenchyma = SEG.GetResult();

    args.WriteImageToFile( args.dirOutput, fileOutputParenchyma, 
                           "breast mask segmentation image", niftiParenchyma );

  }

#else

  if ( ! args.ReadImageFromFile( args.dirOutput, fileOutputParenchyma, 
                                 "breast parenchyma", imParenchyma ) )
  {

    std::stringstream commandNiftySeg;

    commandNiftySeg 
      << "\"" << fileSegEM << "\""
      << " -v 2 -bc_order 4 -nopriors 2" 
      << " -in \"" << niftk::ConcatenatePath( args.dirOutput, fileI02_T2_Resampled ) << "\" "
      << " -mask \"" << niftk::ConcatenatePath( args.dirOutput, fileOutputBreastMask ) << "\" "
      << " -out \"" << niftk::ConcatenatePath( args.dirOutput, fileOutputParenchyma ) << "\" ";

    message << std::endl << "Executing parenchyma segmentation: "
            << std::endl << "   " << commandNiftySeg.str() << std::endl << std::endl;
    args.PrintMessage( message );

    int ret = system( commandNiftySeg.str().c_str() );
    message << std::endl << "Returned: " << ret << std::endl;

    args.ReadImageFromFile( args.dirOutput, fileOutputParenchyma, 
                            "breast parenchyma", imParenchyma );
  }

#endif


  // Load the ADC image
  // ~~~~~~~~~~~~~~~~~~
        
  if ( ! args.ReadImageFromFile( args.dirOutput, fileI00_ADC, 
                                 std::string( "ADC '" ) + args.strSeriesDescADC 
                                 + "' image", imADC ) )
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
            
      args.WriteImageToFile( args.dirOutput, fileI00_ADC, 
                             std::string( "ADC '" ) + args.strSeriesDescADC +
                             "' image", imADC );
    }
  }


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
            
      args.WriteImageToFile( args.dirOutput, fileI00_DCE, 
                             std::string( "DCE '" ) + args.strSeriesDescDCE +
                             "' image", imDCE );
    }
  }
  




  // Delete unwanted images
  // ~~~~~~~~~~~~~~~~~~~~~~

  if ( ! ( flgDebug || flgSaveImages ) )
  {
    args.DeleteFile( args.dirOutput, fileI01_T2_BiasFieldCorr );
    args.DeleteFile( args.dirOutput, fileI01_T1_BiasFieldCorr );
    args.DeleteFile( args.dirOutput, fileI02_T2_Resampled );

    args.DeleteFile( args.dirOutput, fileBIFs );

    args.DeleteFile( args.dirOutput, fileOutputSmoothedStructural );
    args.DeleteFile( args.dirOutput, fileOutputSmoothedFatSat );
    args.DeleteFile( args.dirOutput, fileOutputClosedStructural );

    args.DeleteFile( args.dirOutput, fileOutputMaxImage );
    args.DeleteFile( args.dirOutput, fileOutputCombinedHistogram );
    args.DeleteFile( args.dirOutput, fileOutputRayleigh );
    args.DeleteFile( args.dirOutput, fileOutputFreqLessBgndCDF );
    args.DeleteFile( args.dirOutput, fileOutputBackground );
    args.DeleteFile( args.dirOutput, fileOutputSkinElevationMap );
    args.DeleteFile( args.dirOutput, fileOutputGradientMagImage );
    args.DeleteFile( args.dirOutput, fileOutputSpeedImage );
    args.DeleteFile( args.dirOutput, fileOutputFastMarchingImage );
    args.DeleteFile( args.dirOutput, fileOutputPectoral );
    args.DeleteFile( args.dirOutput, fileOutputChestPoints );
    args.DeleteFile( args.dirOutput, fileOutputPectoralSurfaceMask );

    args.DeleteFile( args.dirOutput, fileOutputPectoralSurfaceVoxels );

    args.DeleteFile( args.dirOutput, fileOutputFittedBreastMask );
  }

  
  return EXIT_SUCCESS;
}
 
 

