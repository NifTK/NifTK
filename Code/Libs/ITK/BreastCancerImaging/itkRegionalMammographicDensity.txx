/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkRegionalMammographicDensity.h>

#include <itkImageFileWriter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSpatialObjectToImageFilter.h>
#include <itkGDCMImageIO.h>
#include <itkNeighborhoodIterator.h>
#include <itkCastImageFilter.h>
#include <itkLabelToRGBImageFilter.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkScalarConnectedComponentImageFilter.h>
 
namespace fs = boost::filesystem;


namespace itk
{


bool CmpCoordsAscending(PointOnBoundary c1, PointOnBoundary c2) { 
  return ( c1.id < c2.id ); 
}

bool CmpCoordsDescending(PointOnBoundary c1, PointOnBoundary c2) { 
  return ( c1.id > c2.id ); 
}



// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
RegionalMammographicDensity< InputPixelType, InputDimension >
::RegionalMammographicDensity() 
{
  thresholdDiagnostic = 0;
  thresholdPreDiagnostic = 0;

  tumourLeft = 0;
  tumourRight = 0;
  tumourTop = 0;
  tumourBottom = 0;
 
  tumourRegionValue = 0;

  regionSizeInMM = 10.;

  imDiagnostic = 0;
  imPreDiagnostic = 0;

  flgVerbose = false;
  flgOverwrite = false;
  flgDebug = false;
}


// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
RegionalMammographicDensity< InputPixelType, InputDimension >
::~RegionalMammographicDensity() 
{
}


// --------------------------------------------------------------------------
// LoadImages()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void 
RegionalMammographicDensity< InputPixelType, InputDimension >
::LoadImages( void ) {
  ReadImage( DIAGNOSTIC_MAMMO );
  ReadImage( PREDIAGNOSTIC_MAMMO );
}


// --------------------------------------------------------------------------
// UnloadImages()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void 
RegionalMammographicDensity< InputPixelType, InputDimension >
::UnloadImages( void )
{
  imDiagnostic = 0;
  imPreDiagnostic = 0;   

  imDiagnosticMask = 0;
  imPreDiagnosticMask = 0;   
}


// --------------------------------------------------------------------------
// Print()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::Print( bool flgVerbose )
{
  std::vector< PointOnBoundary >::iterator itPointOnBoundary;

  std::cout << std::endl
            << "Patient ID: " << id << std::endl;

  if ( flgVerbose )
    std::cout << "   Verbose output: YES" << std::endl;
  else
    std::cout << "   Verbose output: NO" << std::endl;

  if ( flgOverwrite )
    std::cout << "   Overwrite output: YES" << std::endl;
  else
    std::cout << "   Overwrite output: NO" << std::endl;

  if ( flgDebug )
    std::cout << "   Debug output: YES" << std::endl;
  else
    std::cout << "   Debug output: NO" << std::endl;


  if ( flgVerbose )
  {
    if ( imDiagnostic ) imDiagnostic->Print( std::cout );
    PrintDictionary( diagDictionary );

    if ( imPreDiagnostic ) imPreDiagnostic->Print( std::cout );
    PrintDictionary( preDiagDictionary );
  }

  std::cout << std::endl
            << "   Diagnostic ID: " << idDiagnosticImage << std::endl
            << "   Diagnostic file: " << fileDiagnostic << std::endl
            << "   Diagnostic threshold: " <<  thresholdDiagnostic << std::endl
            << std::endl;

  std::cout << "   Diagnostic breast edge points: " << std::endl;

  for ( itPointOnBoundary = diagBreastEdgePoints.begin(); 
        itPointOnBoundary != diagBreastEdgePoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }

  std::cout << std::endl
            << "   Diagnostic pectoral points: " << std::endl;

  for ( itPointOnBoundary = diagPectoralPoints.begin(); 
        itPointOnBoundary != diagPectoralPoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }
  
  std::cout << std::endl
            << "   Pre-diagnostic ID: " << idPreDiagnosticImage << std::endl
            << "   Pre-diagnostic file: " << filePreDiagnostic << std::endl
            << "   Pre-diagnostic threshold: " <<  thresholdPreDiagnostic << std::endl
            << std::endl;

  std::cout << "   Pre-diagnostic breast edge points: " << std::endl;

  for ( itPointOnBoundary = preDiagBreastEdgePoints.begin(); 
        itPointOnBoundary != preDiagBreastEdgePoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }

  std::cout << std::endl
            << "   Pre-diagnostic pectoral points: " << std::endl;

  for ( itPointOnBoundary = preDiagPectoralPoints.begin(); 
        itPointOnBoundary != preDiagPectoralPoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }
  
  std::cout << std::endl
            << "   Tumour ID: "       << strTumourID << std::endl
            << "   Tumour image ID: " << strTumourImageID << std::endl
            << "   Tumour left:   "   <<  tumourLeft << std::endl
            << "   Tumour right:  "   <<  tumourRight << std::endl
            << "   Tumour top:    "   <<  tumourTop << std::endl
            << "   Tumour bottom: "   <<  tumourBottom << std::endl
            << "   Tumour center: " 
            <<  tumourCenterIndex[0] << ", " <<  tumourCenterIndex[1] << std::endl
            << "   Tumour region value: " << tumourRegionValue << std::endl
            << std::endl;

  std::cout << "   Patch size: " 
            << tumourRegion.GetSize()[0] << " x " 
            << tumourRegion.GetSize()[1] << " pixels " << std::endl;

  float nPixelsInPatch = tumourRegion.GetSize()[0]*tumourRegion.GetSize()[1];
  

  std::map< LabelPixelType, Patch >::iterator itPatches;

  std::cout << "Diagnostic image patches: " << std::endl;

  for ( itPatches  = diagPatches.begin(); 
        itPatches != diagPatches.end(); 
        itPatches++ )
  {
    std::cout << "  patch: " << std::right << std::setw(6) << itPatches->first;
    itPatches->second.Print( "     ", nPixelsInPatch );
  }


  std::cout << "Pre-diagnostic image patches: " << std::endl;

  for ( itPatches  = preDiagPatches.begin(); 
        itPatches != preDiagPatches.end(); 
        itPatches++ )
  {
    std::cout << "  patch: " << std::right << std::setw(6) << itPatches->first;
    itPatches->second.Print( "     ", nPixelsInPatch );
  }

}


// --------------------------------------------------------------------------
// WriteToCSVFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::WriteDataToCSVFile( std::ofstream *foutOutputDensityCSV,
                      boost::random::mt19937 &gen )
{
  int i;
  float nMaxPixelsInPatch = tumourRegion.GetSize()[0]*tumourRegion.GetSize()[1];
  float nPixelsInPatch = 0;
  float randomPreDiagDensity = 0;

  if ( flgDebug )
    std::cout << "Max pixels in patch: " << nMaxPixelsInPatch << std::endl;

  std::map< LabelPixelType, Patch >::iterator itPatches;

  while ( nPixelsInPatch != nMaxPixelsInPatch )
  {

    boost::random::uniform_int_distribution<> dist(0, preDiagPatches.size() - 1);
    
    i =  dist( gen );

    itPatches = preDiagPatches.begin();
    std::advance( itPatches, i );
  
    nPixelsInPatch = itPatches->second.GetNumberOfPixels();
    
    randomPreDiagDensity = 
      itPatches->second.GetNumberOfDensePixels()/
      itPatches->second.GetNumberOfPixels();

    if ( flgDebug )
      std::cout << itPatches->first << ": " << setprecision( 6 )
                << setw(12) << itPatches->second.GetNumberOfDensePixels() << " / " 
                << setw(12) << itPatches->second.GetNumberOfPixels() << " = " 
                << setw(12) << randomPreDiagDensity 
                << std::endl;
  }

  *foutOutputDensityCSV 
    << setprecision( 6 )
    << std::right << std::setw(10) << id << ", "
                                   
    << std::right << std::setw(17) << idDiagnosticImage << ", "
    << std::right << std::setw(60) << fileDiagnostic << ", "
    << std::right << std::setw(18) << thresholdDiagnostic << ", "
                                   
    << std::right << std::setw(17) << idPreDiagnosticImage << ", "
    << std::right << std::setw(60) << filePreDiagnostic << ", "
    << std::right << std::setw(18) <<  thresholdPreDiagnostic << ", "
                                   
    << std::right << std::setw( 9) << strTumourID << ", "
    << std::right << std::setw(17) << strTumourImageID << ", "
    << std::right << std::setw(17) << tumourCenterIndex[0] << ", " 
    << std::right << std::setw(17) << tumourCenterIndex[1] << ", "
                                   
    << std::right << std::setw(11) << nPixelsInPatch << ", "

    << std::right << std::setw(29) << itPatches->first << ", "
    << std::right << std::setw(29) << randomPreDiagDensity

    << std::endl;
}


// --------------------------------------------------------------------------
// PrintDictionary()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::PrintDictionary( DictionaryType &dictionary ) 
{

  DictionaryType::ConstIterator tagItr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();
  
  while ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
  
    if ( entryvalue )
    {
      std::string tagkey = tagItr->first;
      std::string tagID;
      bool found =  GDCMImageIO::GetLabelFromTag( tagkey, tagID );
    
      std::string tagValue = entryvalue->GetMetaDataObjectValue();
    
      std::cout << tagkey << " " << tagID <<  ": " << tagValue << std::endl;
    }
  
    ++tagItr;
  }
}


// --------------------------------------------------------------------------
// ReadImage()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::ReadImage( MammogramType mammoType ) 
{
  
  std::string fileImage;

  if ( mammoType == DIAGNOSTIC_MAMMO )
  {
    fileImage = fileDiagnostic;
  }
  else if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    fileImage = filePreDiagnostic;
  }

  if ( ! fileImage.length() ) {
    std::cerr << "ERROR: Cannot read image, filename not set" << std::endl;
    exit( EXIT_FAILURE );
  }

  std::cout << "Reading image: " << fileImage << std::endl;


  typedef GDCMImageIO ImageIOType;
  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO( gdcmImageIO );

  reader->SetFileName( fileImage );

  try
  {
    reader->Update();
  }

  catch (ExceptionObject &ex)
  {
    std::cerr << "ERROR: Could not read file: " << fileImage << std::endl 
              << ex << std::endl;
    exit( EXIT_FAILURE );
  }
  
  if ( mammoType == DIAGNOSTIC_MAMMO )
  {
    imDiagnostic = reader->GetOutput();
    imDiagnostic->DisconnectPipeline();
  
    diagDictionary = imDiagnostic->GetMetaDataDictionary();
  }
  else if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    imPreDiagnostic = reader->GetOutput();
    imPreDiagnostic->DisconnectPipeline();
  
    preDiagDictionary = imPreDiagnostic->GetMetaDataDictionary();
  }
}


// --------------------------------------------------------------------------
// PushBackBreastEdgeCoord()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::PushBackBreastEdgeCoord( std::string strBreastEdgeImageID, 
                           int id, int x, int y )
{
  PointOnBoundary c;

  c.id = id;
  c.x = x;
  c.y = y;

  if ( strBreastEdgeImageID == idDiagnosticImage ) 
  {
    diagBreastEdgePoints.push_back( c );
  }
  else if ( strBreastEdgeImageID == idPreDiagnosticImage ) 
  {
    preDiagBreastEdgePoints.push_back( c );
  }
  else 
  {
    std::cerr << "ERROR: This patient doesn't have and image with id: " 
              << strBreastEdgeImageID << std::endl;
    exit( EXIT_FAILURE );
  }
}


// --------------------------------------------------------------------------
// PushBackPectoralCoord()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::PushBackPectoralCoord( std::string strPectoralImageID, 
                         int id, int x, int y ) 
{

  PointOnBoundary c;

  c.id = id;
  c.x = x;
  c.y = y;

  if ( strPectoralImageID == idDiagnosticImage ) 
  {
    diagPectoralPoints.push_back( c );
  }
  else if ( strPectoralImageID == idPreDiagnosticImage ) 
  {
    preDiagPectoralPoints.push_back( c );
  }
  else 
  {
    std::cerr << "ERROR: This patient doesn't have and image with id: " 
              << strPectoralImageID << std::endl;
    exit( EXIT_FAILURE );
  }
}


// --------------------------------------------------------------------------
// Compute()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::Compute( void )
{
  // Calculate the masks

  imDiagnosticMask = MaskWithPolygon( DIAGNOSTIC_MAMMO );

  WriteBinaryImageToUCharFile( fileDiagnostic, std::string( "_DiagMask.dcm" ), 
                               "diagnostic mask", 
                               imDiagnosticMask, diagDictionary );

  imPreDiagnosticMask = MaskWithPolygon( PREDIAGNOSTIC_MAMMO );

  WriteBinaryImageToUCharFile( filePreDiagnostic, std::string( "_PreDiagMask.dcm" ), 
                               "pre-diagnostic mask", 
                               imPreDiagnosticMask, preDiagDictionary );

  // Calculate the labels

  if ( flgVerbose ) 
    std::cout << "Computing diagnostic mammo labels." << std::endl;

  imDiagnosticLabels = GenerateRegionLabels( imDiagnostic, imDiagnosticMask, 
                                             diagPatches, thresholdDiagnostic );

  WriteImageFile<LabelImageType>( fileDiagnostic, 
                                        std::string( "_DiagLabels.dcm" ), 
                                        "diagnostic labels", 
                                  imDiagnosticLabels, diagDictionary );

  WriteLabelImageFile( fileDiagnostic, 
                       std::string( "_DiagLabels.jpg" ), 
                       "diagnostic labels", 
                       imDiagnosticLabels, diagDictionary );
  
  if ( flgVerbose ) 
    std::cout << "Computing pre-diagnostic mammo labels." << std::endl;

  imPreDiagnosticLabels = GenerateRegionLabels( imPreDiagnostic, imPreDiagnosticMask, 
                                                preDiagPatches, thresholdPreDiagnostic );
  
  WriteImageFile<LabelImageType>( filePreDiagnostic, 
                                        std::string( "_PreDiagLabels.dcm" ), 
                                        "pre-diagnostic labels", 
                                        imPreDiagnosticLabels, preDiagDictionary );

  WriteLabelImageFile( filePreDiagnostic, 
                       std::string( "_PreDiagLabels.jpg" ), 
                       "pre-diagnostic labels", 
                       imPreDiagnosticLabels, preDiagDictionary );

}


// --------------------------------------------------------------------------
// BuildOutputFilename()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
std::string
RegionalMammographicDensity< InputPixelType, InputDimension >
::BuildOutputFilename( std::string fileInput, std::string suffix )
{
  std::string fileOutput, dirOutputFullPath;

  fileOutput = niftk::ModifyImageFileSuffix( fileInput, suffix );

  fileOutput = niftk::ConcatenatePath( dirOutput, fileOutput );

  dirOutputFullPath = fs::path( fileOutput ).branch_path().string();
  
  if ( ! niftk::DirectoryExists( dirOutputFullPath ) )
  {
    std::cout << "Creating output directory: " << dirOutputFullPath << std::endl;
    niftk::CreateDirectoryAndParents( dirOutputFullPath );
  }
    
  std::cout << "Output filename: " << fileOutput << std::endl;

  return fileOutput;
}


// --------------------------------------------------------------------------
// WriteBinaryImageToUCharFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::WriteBinaryImageToUCharFile( std::string fileInput, 
                               std::string suffix,
                               const char *description,
                               typename ImageType::Pointer image,
                               DictionaryType &dictionary )
{
  if ( fileInput.length() ) 
  {
    typedef unsigned char OutputPixelType;
    typedef Image< OutputPixelType, InputDimension> OutputImageType;

    typedef RescaleIntensityImageFilter< ImageType, OutputImageType > CastFilterType;
    typedef ImageFileWriter< OutputImageType > FileWriterType;

    typename ImageType::Pointer pipeITKImageDataConnector;

    typename CastFilterType::Pointer caster = CastFilterType::New();

    std::string fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      if ( ! flgOverwrite ) 
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and can't be overwritten. Consider option: 'overwrite'."
                  << std::endl << std::endl;
        return;
      }
      else
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and will be overwritten."
                  << std::endl << std::endl;
      }
    }

    caster->SetInput( image );
    caster->SetOutputMinimum(   0 );
    caster->SetOutputMaximum( 255 );

    try
    {
      caster->UpdateLargestPossibleRegion();
    }    
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      exit( EXIT_FAILURE );
    }

    typename FileWriterType::Pointer writer = FileWriterType::New();

    typename OutputImageType::Pointer outImage = caster->GetOutput();

    outImage->DisconnectPipeline();

    typename ImageIOBase::Pointer imageIO;
    imageIO = ImageIOFactory::CreateImageIO(fileModifiedOutput.c_str(), 
                                            ImageIOFactory::WriteMode);

    imageIO->SetMetaDataDictionary( dictionary );

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( outImage );
    writer->SetImageIO( imageIO );
    writer->UseInputMetaDataDictionaryOff();

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
  
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not write file: " << fileModifiedOutput << std::endl 
                << ex << std::endl;
      exit( EXIT_FAILURE );
    }
  }
  else
  {
    std::cerr << "Failed to write " << description 
              << " to file - filename is empty " << std::endl;
    exit( EXIT_FAILURE );
  }
}


// --------------------------------------------------------------------------
// WriteImageFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
template <typename TOutputImageType>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::WriteImageFile( std::string fileInput, 
                  std::string suffix, 
                  const char *description,
                  typename TOutputImageType::Pointer image,
                  DictionaryType &dictionary )
{
  if ( fileInput.length() ) 
  {
    typedef ImageFileWriter< TOutputImageType > FileWriterType;

    std::string fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      if ( ! flgOverwrite ) 
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and can't be overwritten. Consider option: 'overwrite'."
                  << std::endl << std::endl;
        return;
      }
      else
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and will be overwritten."
                  << std::endl << std::endl;
      }
    }

    typename FileWriterType::Pointer writer = FileWriterType::New();

    image->DisconnectPipeline();

    typename ImageIOBase::Pointer imageIO;
    imageIO = ImageIOFactory::CreateImageIO(fileModifiedOutput.c_str(), 
                                            ImageIOFactory::WriteMode);

    imageIO->SetMetaDataDictionary( dictionary );

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( image );
    writer->SetImageIO( imageIO );
    writer->UseInputMetaDataDictionaryOff();

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
  
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could notwrite file: " << fileModifiedOutput << std::endl 
                << ex << std::endl;
      exit( EXIT_FAILURE );
    }
  }
  else
  {
    std::cerr << "Failed to write " << description 
              << " to file - filename is empty " << std::endl;
    exit( EXIT_FAILURE );
  }
}


// --------------------------------------------------------------------------
// WriteLabelImageFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::WriteLabelImageFile( std::string fileInput, 
                       std::string suffix, 
                       const char *description,
                       typename LabelImageType::Pointer image,
                       DictionaryType &dictionary )
{
  if ( fileInput.length() ) 
  {

    typedef itk::RGBPixel<unsigned char>             RGBPixelType;
    typedef itk::Image<RGBPixelType, InputDimension> RGBImageType;
 
#if 0 
    LabelPixelType distanceThreshold = 0;
 
    typedef itk::ScalarConnectedComponentImageFilter <LabelImageType, LabelImageType >
      ConnectedComponentImageFilterType;
 
    typename ConnectedComponentImageFilterType::Pointer connected =
      ConnectedComponentImageFilterType::New ();

    connected->SetInput(image);
    connected->SetDistanceThreshold(distanceThreshold);
 
    typedef itk::RelabelComponentImageFilter <LabelImageType, LabelImageType >
      RelabelFilterType;

    typename RelabelFilterType::Pointer relabel = RelabelFilterType::New();
    typename RelabelFilterType::ObjectSizeType minSize = 20;

    relabel->SetInput(connected->GetOutput());
    relabel->SetMinimumObjectSize(minSize);

    try {
      std::cout << "Computing connected labels" << std::endl;
      relabel->Update();
    }
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      exit( EXIT_FAILURE );
    }
#endif

    // Create a color image and set the background to black

    typedef itk::LabelToRGBImageFilter<LabelImageType, RGBImageType> RGBFilterType;

    typename RGBFilterType::Pointer rgbFilter = RGBFilterType::New();

    //rgbFilter->SetInput( relabel->GetOutput() );
    rgbFilter->SetInput( image );

    RGBPixelType rgbPixel;
    
    rgbPixel.SetRed(   0 );
    rgbPixel.SetGreen( 0 );
    rgbPixel.SetBlue(  0 );
    
    rgbFilter->SetBackgroundValue( 0 );
    rgbFilter->SetBackgroundColor( rgbPixel );
 
    try {
      rgbFilter->Update();
    }
    catch (ExceptionObject &ex)
    {
      std::cerr << ex << std::endl;
      exit( EXIT_FAILURE );
    }

    typename RGBImageType::Pointer rgbImage = rgbFilter->GetOutput();
    rgbImage->DisconnectPipeline();
  

    // Set the tumour region to white

    itk::ImageRegionIterator< LabelImageType > itLabel( image, tumourRegion );
    itk::ImageRegionIterator< RGBImageType > itRGBImage( rgbImage, tumourRegion );
      
    rgbPixel.SetRed(   255 );
    rgbPixel.SetGreen( 255 );
    rgbPixel.SetBlue(  255 );
    
    for ( itRGBImage.GoToBegin(), itLabel.GoToBegin(); 
          ! itRGBImage.IsAtEnd(); 
          ++itRGBImage, ++itLabel )
    {
      if ( itLabel.Get() )
      {
        itRGBImage.Set( rgbPixel );
      }
    }
   

    // Write the output to a file

    typedef ImageFileWriter< RGBImageType > FileWriterType;

    std::string fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      if ( ! flgOverwrite ) 
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and can't be overwritten. Consider option: 'overwrite'."
                  << std::endl << std::endl;
        return;
      }
      else
      {
        std::cerr << std::endl << "WARNING: File " << fileModifiedOutput << " exists"
                  << std::endl << "         and will be overwritten."
                  << std::endl << std::endl;
      }
    }

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( rgbImage );

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
  
    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could notwrite file: " << fileModifiedOutput << std::endl 
                << ex << std::endl;
      exit( EXIT_FAILURE );
    }
  }
  else
  {
    std::cerr << "Failed to write " << description 
              << " to file - filename is empty " << std::endl;
    exit( EXIT_FAILURE );
  }
}


// --------------------------------------------------------------------------
// AddPointToPolygon()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::AddPointToPolygon( typename PolygonType::Pointer &polygon, 
                     typename ImageType::Pointer &image, 
                     typename ImageType::SizeType &polySize, 
                     int x, int y )
{
  typename ImageType::IndexType index;
  typename PolygonType::PointType point;
  typename PolygonType::PointListType points = polygon->GetPoints();

  index[0] = x;
  index[1] = y;

  if ( x > polySize[0] ) polySize[0] = x;
  if ( y > polySize[1] ) polySize[1] = y;

  image->TransformIndexToPhysicalPoint( index, point );
  
  if ( (points.size() == 0) ||
       ( ( point[0] != points.back().GetPosition()[0] ) || 
         ( point[1] != points.back().GetPosition()[1] ) ) )
  {
    polygon->AddPoint( point );
  }
}


// --------------------------------------------------------------------------
// MaskWithPolygon()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::ImageType::Pointer 
RegionalMammographicDensity< InputPixelType, InputDimension >
::MaskWithPolygon( MammogramType mammoType, LocusType locusType )
{
  typename ImageType::Pointer imMask;
  typename ImageType::Pointer image;    
  typename ImageType::RegionType region;
  typename ImageType::SizeType polySize;

  typename std::vector< PointOnBoundary > *pPointOnBoundary;

  std::vector< PointOnBoundary >::iterator itPointOnBoundary;

  polySize[0] = 0;
  polySize[1] = 0;


  // Set the image and loci
   
  if ( mammoType == DIAGNOSTIC_MAMMO )
  {
    image = imDiagnostic;

    if ( locusType == BREAST_EDGE )
    {
      if ( flgVerbose ) 
        std::cout << "Creating diagnostic mammo breast edge mask." << std::endl;

      pPointOnBoundary = &diagBreastEdgePoints; 
    }
    else
    {
      if ( flgVerbose ) 
        std::cout << "Creating diagnostic mammo pectoral mask." << std::endl;

      pPointOnBoundary = &diagPectoralPoints;
    }
  }
  else if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    image = imPreDiagnostic;

    if ( locusType == BREAST_EDGE )
    {
      if ( flgVerbose ) 
        std::cout << "Creating prediagnostic mammo breast edge mask." << std::endl;

      pPointOnBoundary = &preDiagBreastEdgePoints; 
    }
    else {
      if ( flgVerbose ) 
        std::cout << "Creating prediagnostic mammo pectoral mask." << std::endl;

      pPointOnBoundary = &preDiagPectoralPoints;
    }
  }

  region = image->GetLargestPossibleRegion();


  // Sort the coordinates and check that the first point is closer to y = 0

  std::sort( pPointOnBoundary->begin(), pPointOnBoundary->end(), CmpCoordsAscending );

  pPointOnBoundary->front().Print( "  First point: " );
  pPointOnBoundary->back().Print(  "  Last point:  " );

  if ( pPointOnBoundary->front().y > pPointOnBoundary->back().y )
  {
    std::sort( pPointOnBoundary->begin(), pPointOnBoundary->end(), CmpCoordsDescending );

    pPointOnBoundary->front().Print( "  First point: " );
    pPointOnBoundary->back().Print(  "  Last point:  " );
  }

  typename PolygonType::Pointer polygon = PolygonType::New();

 
  polygon->ComputeObjectToWorldTransform();

  // First point is (x=0, y=0)   
  AddPointToPolygon( polygon, image, polySize, 0, 0 );

  // Second point is (x, y=0)   
  AddPointToPolygon( polygon, image, polySize, pPointOnBoundary->front().x, 0 );

  // Add all the user specified points
  for ( itPointOnBoundary  = pPointOnBoundary->begin(); 
        itPointOnBoundary != pPointOnBoundary->end(); 
        itPointOnBoundary++ )
  {
    (*itPointOnBoundary).Print( "     " );

    AddPointToPolygon( polygon, image, polySize, (*itPointOnBoundary).x, (*itPointOnBoundary).y );
  }

  // Last point is (x=0, y) for breast edge
  if ( locusType == BREAST_EDGE )
  {
    AddPointToPolygon( polygon, image, polySize, 0, pPointOnBoundary->back().y );
  }
  // or (x=0, y=max) for pectoral
  else {
    AddPointToPolygon( polygon, image, polySize, 0, region.GetSize()[1] );
  }

  if ( flgDebug )
  {
    typename PolygonType::PointListType vertexList = polygon->GetPoints();
    typename PolygonType::PointListType::const_iterator itPoints = vertexList.begin();

    std::cout << "  Polygon vertices:" << std::endl;

    for (itPoints=vertexList.begin(); itPoints<vertexList.end(); itPoints++)
    {
      std::cout << "     " 
                << std::right << std::setw(8) << (*itPoints).GetPosition()[0] << ", " 
                << std::right << std::setw(8) << (*itPoints).GetPosition()[1] << std::endl;
    }
  }

  // Create the mask

  typedef SpatialObjectToImageFilter< PolygonType, 
    ImageType > SpatialObjectToImageFilterType;

  typename SpatialObjectToImageFilterType::Pointer 
    polyMaskFilter = SpatialObjectToImageFilterType::New();
 
  polyMaskFilter->SetInput( polygon );
  polyMaskFilter->SetInsideValue( 1000 );
  polyMaskFilter->SetOutsideValue( 0 );
    
  polyMaskFilter->SetSize( region.GetSize() );

  polygon->SetThickness(1.0);

  try
  {
    polyMaskFilter->Update();
  }
  
  catch (ExceptionObject &ex)
  {
    std::cerr << ex << std::endl;
    exit( EXIT_FAILURE );
  }

  imMask = polyMaskFilter->GetOutput();
  imMask->DisconnectPipeline();

  imMask->SetSpacing( image->GetSpacing() );


  return imMask;
}


// --------------------------------------------------------------------------
// MaskWithPolygon()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::ImageType::Pointer 
RegionalMammographicDensity< InputPixelType, InputDimension >
::MaskWithPolygon( MammogramType mammoType )
{
  typename ImageType::Pointer imMask, imMaskPec;

  imMask = MaskWithPolygon( mammoType, BREAST_EDGE );
  imMaskPec = MaskWithPolygon( mammoType, PECTORAL );

  std::cout << "Breast Edge mask spacing: " << imMask->GetSpacing() << std::endl;
  std::cout << "Pectoral mask spacing: " << imMaskPec->GetSpacing() << std::endl;

  if ( flgDebug ) 
  {
    
    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      WriteBinaryImageToUCharFile( fileDiagnostic, 
                                   std::string( "_DiagBreastEdgeMask.dcm" ), 
                                   "diagnostic breast edge mask", imMask, diagDictionary );
    }
    else
    {
      WriteBinaryImageToUCharFile( filePreDiagnostic, 
                                   std::string( "_PreDiagBreastEdgeMask.dcm" ), 
                                   "pre-diagnostic mask", imMask, preDiagDictionary );
    }

    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      WriteBinaryImageToUCharFile( fileDiagnostic, 
                                   std::string( "_DiagPectoralMask.dcm" ), 
                                   "diagnostic pectoral mask", imMaskPec, diagDictionary );
    }
    else
    {
      WriteBinaryImageToUCharFile( filePreDiagnostic, 
                                   std::string( "_PreDiagPectoralMask.dcm" ), 
                                   "pre-diagnostic mask", imMaskPec, preDiagDictionary );
    }
  }

  // Iterate over the masks to compute the combined mask
  
  IteratorType itMask( imMask, imMask->GetLargestPossibleRegion() );
  IteratorType itMaskPec( imMaskPec, imMaskPec->GetLargestPossibleRegion() );
      
  for ( itMask.GoToBegin(), itMaskPec.GoToBegin(); 
        ! itMask.IsAtEnd(); 
        ++itMask, ++itMaskPec )
  {
    if ( itMaskPec.Get() )
      itMask.Set( 0 );
  }

  return imMask;
}


// --------------------------------------------------------------------------
// GenerateRegionLabels()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::LabelImageType::Pointer 
RegionalMammographicDensity< InputPixelType, InputDimension >
::GenerateRegionLabels( typename ImageType::Pointer &image,
                        typename ImageType::Pointer &imMask,
                        typename std::map< LabelPixelType, Patch > &listOfPatches,
                        int threshold )
{
  typename LabelImageType::Pointer imLabels;

  typename LabelImageType::PointType point;

  typename LabelImageType::IndexType index;
  typename LabelImageType::IndexType regionRadiusInPixels;
  typename LabelImageType::IndexType tumourRegionIndex;
  typename LabelImageType::IndexType labelOrigin;
  
  typename LabelImageType::SizeType regionSizeInPixels;
  typename LabelImageType::SizeType imSizeInPixels;
  typename LabelImageType::SizeType imSizeInROIs;

  typename LabelImageType::SpacingType imSpacing;

  typedef itk::CastImageFilter< ImageType, LabelImageType > CastFilterType;

   
  // Cast the image to int

  typename CastFilterType::Pointer caster = CastFilterType::New();
  
  caster->SetInput( imMask );

  try
  {
    caster->UpdateLargestPossibleRegion();
  }    
  catch (ExceptionObject &ex)
  {
    std::cerr << ex << std::endl;
    exit( EXIT_FAILURE );
  }

  imLabels = caster->GetOutput();
  imLabels->DisconnectPipeline();
  
  imSizeInPixels = imLabels->GetLargestPossibleRegion().GetSize();
  imSpacing = imLabels->GetSpacing();

  // Get the ROI size in pixels

  regionSizeInPixels[0] = (int) ceil(regionSizeInMM / imSpacing[0]);
  regionSizeInPixels[1] = (int) ceil(regionSizeInMM / imSpacing[1]);

  regionRadiusInPixels[0] = regionSizeInPixels[0]/2;
  regionRadiusInPixels[1] = regionSizeInPixels[1]/2;

  regionSizeInPixels[0] = 2*regionRadiusInPixels[0] + 1;
  regionSizeInPixels[1] = 2*regionRadiusInPixels[1] + 1;

  imSizeInROIs[0] = (int) ceil( ((float) imSizeInPixels[0]) 
                                / ((float) regionSizeInPixels[0]) );

  imSizeInROIs[1] = (int) ceil( ((float) imSizeInPixels[1]) 
                                / ((float) regionSizeInPixels[1]) );

  // Get the index of the tumour ROI

  tumourCenterIndex[0] = (tumourLeft +  tumourRight) / 2;
  tumourCenterIndex[1] = ( tumourTop + tumourBottom) / 2;

  tumourRegionIndex[0] = tumourCenterIndex[0] - regionRadiusInPixels[0];
  tumourRegionIndex[1] = tumourCenterIndex[1] - regionRadiusInPixels[1];

  // Hence the origin of the label grid

  labelOrigin[0] = (int) floor(((float) tumourRegionIndex[0]) - 
    ((float) regionSizeInPixels[0])*ceil( ((float) tumourRegionIndex[0]) 
                                          / ((float) regionSizeInPixels[0]) ) + 0.5);

  labelOrigin[1] = (int) floor(((float) tumourRegionIndex[1]) - 
    ((float) regionSizeInPixels[1])*ceil( ((float) tumourRegionIndex[1]) 
                                          / ((float) regionSizeInPixels[1]) ) + 0.5);

  tumourRegion.SetIndex( tumourRegionIndex );
  tumourRegion.SetSize( regionSizeInPixels );

  if ( flgDebug )
    std::cout << "  Region size (mm): " << regionSizeInMM << std::endl
              << "  Image resolution: " 
              << imSpacing[0] << ", " << imSpacing[1] << std::endl
              << "  Tumour index: " 
              << tumourRegionIndex[0] << ", " << tumourRegionIndex[1] << std::endl
              << "  Label grid origin: " 
              << labelOrigin[0] << ", " << labelOrigin[1] << std::endl
              << "  ROI size (pixels): " 
              << regionSizeInPixels[0] << " x " << regionSizeInPixels[1] << std::endl
              << "  Image size in ROIs: " 
              << imSizeInROIs[0] << " x " << imSizeInROIs[1] << std::endl;

  tumourRegionValue = imSizeInROIs[0]*((tumourCenterIndex[1] - labelOrigin[1]) 
                                       / regionSizeInPixels[1]) 
    + (tumourCenterIndex[0] - labelOrigin[0]) / regionSizeInPixels[0];

  if ( flgVerbose )
    std::cout << "  Tumour region will have value: " << tumourRegionValue << std::endl;

  // Iterate through the mask and the image estimating the density for each patch

  IteratorType itImage( image, image->GetLargestPossibleRegion() );
      
  itk::ImageRegionIteratorWithIndex< LabelImageType >
    itLabels( imLabels, imLabels->GetLargestPossibleRegion() );
      
  LabelPixelType iPatch;

  for ( itLabels.GoToBegin(), itImage.GoToBegin(); 
        ! itLabels.IsAtEnd(); 
        ++itLabels, ++itImage )
  {
    if ( itLabels.Get() )
    {
      index = itLabels.GetIndex();

      iPatch = imSizeInROIs[0]*((index[1] - labelOrigin[1]) / regionSizeInPixels[1]) 
        + (index[0] - labelOrigin[0]) / regionSizeInPixels[0];

      itLabels.Set( iPatch );      

      if ( itImage.Get() > threshold ) 
      {
        listOfPatches[ iPatch ].AddDensePixel();
      }
      else
      {
        listOfPatches[ iPatch ].AddNonDensePixel();
      }

#if 0      
      std::cout << "   index: " << std::right << setw(6) << index[0] << ", "  << index[1]
                << "   patch: " << std::right << setw(6) 
                << (index[0] - labelOrigin[0]) / regionSizeInPixels[0] << ", "  
                << (index[1] - labelOrigin[1]) / regionSizeInPixels[1] 
                << "   number: " << std::right << setw(6) << iPatch
                << std::endl;
#endif      
    }
  }

  return imLabels;
}


} // namespace itk
