/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <boost/algorithm/string.hpp>

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
  m_ThresholdDiagnostic = 0;
  m_ThresholdPreDiagnostic = 0;

  m_TumourLeft = 0;
  m_TumourRight = 0;
  m_TumourTop = 0;
  m_TumourBottom = 0;
 
  m_DiagTumourRegionValue = 0;

  m_DiagTumourCenterIndex[0] = 0;
  m_DiagTumourCenterIndex[1] = 0;

  m_PreDiagTumourRegionValue = 0;

  m_PreDiagCenterIndex[0] = 0;
  m_PreDiagCenterIndex[1] = 0;

  m_RegionSizeInMM = 10.;

  m_ImDiagnostic = 0;
  m_ImPreDiagnostic = 0;

  m_FlgOverwrite = false;
  m_FlgRegister = false;

  m_FlgVerbose = false;
  m_FlgDebug = false;

  m_Transform = 0;
};


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
  m_ImDiagnostic = 0;
  m_ImPreDiagnostic = 0;   

  m_ImDiagnosticMask = 0;
  m_ImPreDiagnosticMask = 0;   
};


// --------------------------------------------------------------------------
// Print()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::Print( bool m_FlgVerbose )
{
  std::vector< PointOnBoundary >::iterator itPointOnBoundary;

  std::cout << std::endl
            << "Patient ID: " << m_Id << std::endl;

  std::cout << std::endl
            << "    Input image directory: " << m_DirInput << std::endl
            << "    Output directory: "      << m_DirOutput << std::endl;

  if ( m_FlgVerbose )
    std::cout << "   Verbose output: YES" << std::endl;
  else
    std::cout << "   Verbose output: NO" << std::endl;

  if ( m_FlgOverwrite )
    std::cout << "   Overwrite output: YES" << std::endl;
  else
    std::cout << "   Overwrite output: NO" << std::endl;

  if ( m_FlgRegister )
    std::cout << "   Register the images: YES" << std::endl;
  else
    std::cout << "   Register the images: NO" << std::endl;

  if ( m_FlgDebug )
    std::cout << "   Debug output: YES" << std::endl;
  else
    std::cout << "   Debug output: NO" << std::endl;


  if ( m_FlgVerbose )
  {
    if ( m_ImDiagnostic ) m_ImDiagnostic->Print( std::cout );
    PrintDictionary( m_DiagDictionary );

    if ( m_ImPreDiagnostic ) m_ImPreDiagnostic->Print( std::cout );
    PrintDictionary( m_PreDiagDictionary );
  }

  std::cout << std::endl
            << "   Diagnostic ID: " << m_IdDiagnosticImage << std::endl
            << "   Diagnostic file: " << m_FileDiagnostic << std::endl
            << "   Diagnostic threshold: " <<  m_ThresholdDiagnostic << std::endl
            << std::endl;

  std::cout << "   Diagnostic breast edge points: " << std::endl;

  for ( itPointOnBoundary = m_DiagBreastEdgePoints.begin(); 
        itPointOnBoundary != m_DiagBreastEdgePoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }

  std::cout << std::endl
            << "   Diagnostic pectoral points: " << std::endl;

  for ( itPointOnBoundary = m_DiagPectoralPoints.begin(); 
        itPointOnBoundary != m_DiagPectoralPoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }
  
  std::cout << std::endl
            << "   Pre-diagnostic ID: " << m_IdPreDiagnosticImage << std::endl
            << "   Pre-diagnostic file: " << m_FilePreDiagnostic << std::endl
            << "   Pre-diagnostic threshold: " <<  m_ThresholdPreDiagnostic << std::endl
            << std::endl;

  std::cout << "   Pre-diagnostic breast edge points: " << std::endl;

  for ( itPointOnBoundary = m_PreDiagBreastEdgePoints.begin(); 
        itPointOnBoundary != m_PreDiagBreastEdgePoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }

  std::cout << std::endl
            << "   Pre-diagnostic pectoral points: " << std::endl;

  for ( itPointOnBoundary = m_PreDiagPectoralPoints.begin(); 
        itPointOnBoundary != m_PreDiagPectoralPoints.end(); 
        itPointOnBoundary++ )
  {
    std::cout << "     " 
              << std::right << std::setw(6) << (*itPointOnBoundary).id << ": "
              << std::right << std::setw(6) << (*itPointOnBoundary).x << ", "
              << std::right << std::setw(6) << (*itPointOnBoundary).y << std::endl;
  }
  
  std::cout << std::endl
            << "   Tumour ID: "       << m_StrTumourID << std::endl
            << "   Tumour image ID: " << m_StrTumourImageID << std::endl
            << "   Tumour left:   "   <<  m_TumourLeft << std::endl
            << "   Tumour right:  "   <<  m_TumourRight << std::endl
            << "   Tumour top:    "   <<  m_TumourTop << std::endl
            << "   Tumour bottom: "   <<  m_TumourBottom << std::endl
            << "   Tumour center in diag image: " 
            <<  m_DiagTumourCenterIndex[0] << ", " 
            <<  m_DiagTumourCenterIndex[1] << std::endl
            << "   Tumour diag region value: " 
            << m_DiagTumourRegionValue << std::endl
            << std::endl;

  std::cout << "   Tumour center in pre-diag image: " 
            <<  m_PreDiagCenterIndex[0] << ", " 
            <<  m_PreDiagCenterIndex[1] << std::endl 
            << "   Tumour pre-diag region value: " 
            << m_PreDiagTumourRegionValue << std::endl
            << std::endl;

  std::cout << "   Patch size: " 
            << m_DiagTumourRegion.GetSize()[0] << " x " 
            << m_DiagTumourRegion.GetSize()[1] << " pixels " << std::endl;

  float nPixelsInPatch = m_DiagTumourRegion.GetSize()[0]*m_DiagTumourRegion.GetSize()[1];
  

  std::map< LabelPixelType, Patch >::iterator itPatches;

  std::cout << "Diagnostic image patches: " << std::endl;

  for ( itPatches  = m_DiagPatches.begin(); 
        itPatches != m_DiagPatches.end(); 
        itPatches++ )
  {
    std::cout << "  patch: " << std::right << std::setw(6) << itPatches->first;
    itPatches->second.Print( "     ", nPixelsInPatch );
  }


  std::cout << "Pre-diagnostic image patches: " << std::endl;

  for ( itPatches  = m_PreDiagPatches.begin(); 
        itPatches != m_PreDiagPatches.end(); 
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
::WriteDataToCSVFile( std::ofstream *foutOutputDensityCSV )
{
  int i;
  float nPixelsInPatch = 0;
  float preDiagDensity = 0;

  std::map< LabelPixelType, Patch >::iterator itPatches;

  itPatches = m_PreDiagPatches.begin();

  while ( (itPatches != m_PreDiagPatches.end()) && 
          (itPatches->first != m_PreDiagTumourRegionValue) )
  {
    itPatches++;
  }
   
  nPixelsInPatch = itPatches->second.GetNumberOfPixels();
    
  preDiagDensity = 
    itPatches->second.GetNumberOfDensePixels()/
    itPatches->second.GetNumberOfPixels();
  
  if ( m_FlgDebug )
    std::cout << setw(6) << itPatches->first << ": " << setprecision( 6 )
              << setw(12) << itPatches->second.GetNumberOfDensePixels() << " / " 
              << setw(12) << itPatches->second.GetNumberOfPixels() << " = " 
              << setw(12) << preDiagDensity 
              << std::endl;

  *foutOutputDensityCSV 
    << setprecision( 6 )
    << std::right << std::setw(10) << m_Id << ", "
                                   
    << std::right << std::setw(17) << m_IdDiagnosticImage << ", "
    << std::right << std::setw(60) << m_FileDiagnostic << ", "
    << std::right << std::setw(18) << m_ThresholdDiagnostic << ", "
                                   
    << std::right << std::setw(17) << m_IdPreDiagnosticImage << ", "
    << std::right << std::setw(60) << m_FilePreDiagnostic << ", "
    << std::right << std::setw(18) << m_ThresholdPreDiagnostic << ", "
                                   
    << std::right << std::setw( 9) << m_StrTumourID << ", "
    << std::right << std::setw(17) << m_StrTumourImageID << ", "
    << std::right << std::setw(17) << m_DiagTumourCenterIndex[0] << ", " 
    << std::right << std::setw(17) << m_DiagTumourCenterIndex[1] << ", "
                                   
    << std::right << std::setw(11) << nPixelsInPatch << ", "

    << std::right << std::setw(22) << itPatches->first << ", "
    << std::right << std::setw(22) << preDiagDensity

    << std::endl;
};


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
  float nMaxPixelsInPatch = m_PreDiagTumourRegion.GetSize()[0]*m_PreDiagTumourRegion.GetSize()[1];
  float nPixelsInPatch = 0;
  float randomPreDiagDensity = 0;

  if ( m_FlgDebug )
    std::cout << "Max pixels in patch: " << nMaxPixelsInPatch << std::endl;

  std::map< LabelPixelType, Patch >::iterator itPatches;

  while ( nPixelsInPatch != nMaxPixelsInPatch )
  {

    boost::random::uniform_int_distribution<> dist(0, m_PreDiagPatches.size() - 1);
    
    i =  dist( gen );

    itPatches = m_PreDiagPatches.begin();
    std::advance( itPatches, i );
  
    nPixelsInPatch = itPatches->second.GetNumberOfPixels();
    
    randomPreDiagDensity = 
      itPatches->second.GetNumberOfDensePixels()/
      itPatches->second.GetNumberOfPixels();

    if ( m_FlgDebug )
      std::cout << setw(6) << itPatches->first << ": " << setprecision( 6 )
                << setw(12) << itPatches->second.GetNumberOfDensePixels() << " / " 
                << setw(12) << itPatches->second.GetNumberOfPixels() << " = " 
                << setw(12) << randomPreDiagDensity 
                << std::endl;
  }

  *foutOutputDensityCSV 
    << setprecision( 6 )
    << std::right << std::setw(10) << m_Id << ", "
                                   
    << std::right << std::setw(17) << m_IdDiagnosticImage << ", "
    << std::right << std::setw(60) << m_FileDiagnostic << ", "
    << std::right << std::setw(18) << m_ThresholdDiagnostic << ", "
                                   
    << std::right << std::setw(17) << m_IdPreDiagnosticImage << ", "
    << std::right << std::setw(60) << m_FilePreDiagnostic << ", "
    << std::right << std::setw(18) <<  m_ThresholdPreDiagnostic << ", "
                                   
    << std::right << std::setw( 9) << m_StrTumourID << ", "
    << std::right << std::setw(17) << m_StrTumourImageID << ", "
    << std::right << std::setw(17) << m_DiagTumourCenterIndex[0] << ", " 
    << std::right << std::setw(17) << m_DiagTumourCenterIndex[1] << ", "
                                   
    << std::right << std::setw(11) << nPixelsInPatch << ", "

    << std::right << std::setw(22) << itPatches->first << ", "
    << std::right << std::setw(22) << randomPreDiagDensity

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
// GetImageResolutionFromDictionary()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::ImageType::SpacingType
RegionalMammographicDensity< InputPixelType, InputDimension >
::GetImageResolutionFromDictionary( DictionaryType &dictionary ) 
{
  unsigned int i;
  typename ImageType::SpacingType spacing;
  std::string entryId = "0028|0030";
  
  typename DictionaryType::ConstIterator tagItr = dictionary.Find( entryId );

  for (i=0; i<InputDimension; i++)
    spacing[i] = 1.;

  if ( tagItr != dictionary.End() )
  {
    MetaDataStringType::ConstPointer entryvalue =
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagvalue = entryvalue->GetMetaDataObjectValue();
      std::vector<std::string> words;

      boost::split(words, tagvalue, boost::is_any_of(", \\"), 
                   boost::token_compress_on);
      
      if ( words.size() >= 2 )
      {
        spacing[0] = atof( words[0].c_str() );
        spacing[1] = atof( words[1].c_str() );
      }

      if ( m_FlgDebug )
      {
        std::cout << "Image resolution (" << entryId <<  ") "
                  << " is: " << tagvalue.c_str() 
                  << " ( " << spacing[0] << ", " << spacing[1] << " )"
                  << std::endl;
      }
    }
  }
  
  return spacing;
}


// --------------------------------------------------------------------------
// ReadImage()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::ReadImage( MammogramType mammoType ) 
{
  std::string fileInput;
  std::string fileImage;

  if ( mammoType == DIAGNOSTIC_MAMMO )
  {
    fileImage = m_FileDiagnostic;
  }
  else if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    fileImage = m_FilePreDiagnostic;
  }

  if ( ! fileImage.length() ) {
    std::cerr << "ERROR: Cannot read image, filename not set" << std::endl;
    exit( EXIT_FAILURE );
  }

  fileInput = niftk::ConcatenatePath( m_DirInput, fileImage );

  std::cout << "Reading image: " << fileInput << std::endl;

  typedef GDCMImageIO ImageIOType;
  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO( gdcmImageIO );

  reader->SetFileName( fileInput );

  try
  {
    reader->Update();
  }

  catch (ExceptionObject &ex)
  {
    std::cerr << "ERROR: Could not read file: " << fileInput << std::endl 
              << ex << std::endl;
    exit( EXIT_FAILURE );
  }

  if ( m_FlgDebug )
    std::cout << "  image spacing: " << reader->GetOutput()->GetSpacing()
              << std::endl;
  
  if ( mammoType == DIAGNOSTIC_MAMMO )
  {
    m_ImDiagnostic = reader->GetOutput();
    m_ImDiagnostic->DisconnectPipeline();
  
    m_DiagDictionary = m_ImDiagnostic->GetMetaDataDictionary();

    m_ImDiagnostic->SetSpacing( GetImageResolutionFromDictionary( m_DiagDictionary ) );
  }
  else if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    m_ImPreDiagnostic = reader->GetOutput();
    m_ImPreDiagnostic->DisconnectPipeline();
  
    m_PreDiagDictionary = m_ImPreDiagnostic->GetMetaDataDictionary();

    m_ImPreDiagnostic->SetSpacing( GetImageResolutionFromDictionary( m_PreDiagDictionary ) );
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

  if ( strBreastEdgeImageID == m_IdDiagnosticImage ) 
  {
    m_DiagBreastEdgePoints.push_back( c );
  }
  else if ( strBreastEdgeImageID == m_IdPreDiagnosticImage ) 
  {
    m_PreDiagBreastEdgePoints.push_back( c );
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

  if ( strPectoralImageID == m_IdDiagnosticImage ) 
  {
    m_DiagPectoralPoints.push_back( c );
  }
  else if ( strPectoralImageID == m_IdPreDiagnosticImage ) 
  {
    m_PreDiagPectoralPoints.push_back( c );
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
  std::string fileMask;
  std::string diagMaskSuffix( "_DiagMask.dcm" );
  std::string preDiagMaskSuffix( "_PreDiagMask.dcm" );

  // Generate the Diagnostic Mask

  fileMask = BuildOutputFilename( m_FileDiagnostic, diagMaskSuffix );

  if ( niftk::FileExists( fileMask ) && ( ! m_FlgOverwrite ) )
  {
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( fileMask );

    try
    {
      std::cout << "Reading the diagnostic mask: " << fileMask << std::endl;
      reader->Update();
    }

    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not read file: " 
                << fileMask << std::endl << ex << std::endl;
      exit( EXIT_FAILURE );
    }

    m_ImDiagnosticMask = reader->GetOutput();
    m_ImDiagnosticMask->DisconnectPipeline();
  }
  else
  {
    m_ImDiagnosticMask = MaskWithPolygon( DIAGNOSTIC_MAMMO );

    CastImageAndWriteToFile< unsigned char >( m_FileDiagnostic, 
                                              diagMaskSuffix, 
                                              "diagnostic mask", 
                                              m_ImDiagnosticMask, 
                                              m_DiagDictionary );
  }

  // Generate the Pre-diagnostic Mask

  fileMask = BuildOutputFilename( m_FilePreDiagnostic, preDiagMaskSuffix );

  if ( niftk::FileExists( fileMask ) && ( ! m_FlgOverwrite ) )
  {
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( fileMask );

    try
    {
      std::cout << "Reading the pre-diagnostic mask: " << fileMask << std::endl;
      reader->Update();
    }

    catch (ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not read file: " 
                << fileMask << std::endl << ex << std::endl;
      exit( EXIT_FAILURE );
    }

    m_ImPreDiagnosticMask = reader->GetOutput();
    m_ImPreDiagnosticMask->DisconnectPipeline();
  }
  else
  {
    m_ImPreDiagnosticMask = MaskWithPolygon( PREDIAGNOSTIC_MAMMO );

    CastImageAndWriteToFile< unsigned char >( m_FilePreDiagnostic, 
                                              preDiagMaskSuffix, 
                                              "pre-diagnostic mask", 
                                              m_ImPreDiagnosticMask, 
                                              m_PreDiagDictionary );
  }

  // Register the images?

  if ( m_FlgRegister )
  {
    RegisterTheImages();
  }

  // Calculate the labels

  if ( m_FlgVerbose ) 
    std::cout << "Computing diagnostic mammo labels." << std::endl;


  m_DiagTumourCenterIndex[0] = (m_TumourLeft +  m_TumourRight) / 2;
  m_DiagTumourCenterIndex[1] = ( m_TumourTop + m_TumourBottom) / 2;

  m_ImDiagnosticLabels = GenerateRegionLabels( m_DiagTumourCenterIndex,
                                               m_DiagTumourRegion,
                                               m_DiagTumourRegionValue,
                                               m_ImDiagnostic, 
                                               m_ImDiagnosticMask, 
                                               m_DiagPatches, 
                                               m_ThresholdDiagnostic );

  WriteImageFile<LabelImageType>( m_FileDiagnostic, 
                                  std::string( "_DiagLabels.dcm" ), 
                                  "diagnostic labels", 
                                  m_ImDiagnosticLabels, m_DiagDictionary );

  WriteLabelImageFile( m_FileDiagnostic, 
                       std::string( "_DiagLabels.jpg" ), 
                       "diagnostic labels", 
                       m_ImDiagnosticLabels,  m_DiagTumourRegion,
                       m_DiagDictionary );
  
  if ( m_FlgVerbose ) 
    std::cout << "Computing pre-diagnostic mammo labels." << std::endl;

  if ( m_FlgRegister ) 
  {
    m_PreDiagCenterIndex = TransformTumourPositionIntoPreDiagImage( m_DiagTumourCenterIndex );

    if ( m_FlgVerbose ) 
      std::cout << "   Tumour center in pre-diag image: " 
                << m_PreDiagCenterIndex[0] << ", " 
                << m_PreDiagCenterIndex[1] << std::endl;    
  }
  else
  {
    m_PreDiagCenterIndex = m_DiagTumourCenterIndex;
  }

  m_ImPreDiagnosticLabels = GenerateRegionLabels( m_PreDiagCenterIndex,
                                                  m_PreDiagTumourRegion,
                                                  m_PreDiagTumourRegionValue,                                                 
                                                  m_ImPreDiagnostic, 
                                                  m_ImPreDiagnosticMask, 
                                                  m_PreDiagPatches,
                                                  m_ThresholdPreDiagnostic );

  WriteImageFile<LabelImageType>( m_FilePreDiagnostic, 
                                  std::string( "_PreDiagLabels.dcm" ), 
                                  "pre-diagnostic labels", 
                                  m_ImPreDiagnosticLabels, m_PreDiagDictionary );
  
  WriteLabelImageFile( m_FilePreDiagnostic, 
                       std::string( "_PreDiagLabels.jpg" ), 
                       "pre-diagnostic labels", 
                       m_ImPreDiagnosticLabels, m_PreDiagTumourRegion,
                       m_PreDiagDictionary );
  
};


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

  fileOutput = niftk::ConcatenatePath( m_DirOutput, fileOutput );

  dirOutputFullPath = fs::path( fileOutput ).branch_path().string();
  
  if ( ! niftk::DirectoryExists( dirOutputFullPath ) )
  {
    std::cout << "Creating output directory: " << dirOutputFullPath << std::endl;
    niftk::CreateDirAndParents( dirOutputFullPath );
  }
    
  std::cout << "Output filename: " << fileOutput << std::endl;

  return fileOutput;
}


// --------------------------------------------------------------------------
// CastImageAndWriteToFile()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
template <typename TOutputImageType>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::CastImageAndWriteToFile( std::string fileInput, 
                            std::string suffix,
                            const char *description,
                            typename ImageType::Pointer image,
                            DictionaryType &dictionary )
{
  if ( fileInput.length() ) 
  {
    typedef RescaleIntensityImageFilter< ImageType, OutputImageType > CastFilterType;
    typedef ImageFileWriter< OutputImageType > FileWriterType;

    typename ImageType::Pointer pipeITKImageDataConnector;

    typename CastFilterType::Pointer caster = CastFilterType::New();

    std::string fileModifiedOutput = BuildOutputFilename( fileInput, suffix );

    if ( niftk::FileExists( fileModifiedOutput ) ) 
    {
      if ( ! m_FlgOverwrite ) 
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
      if ( ! m_FlgOverwrite ) 
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
                       typename LabelImageType::RegionType &tumourRegion,
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
      if ( ! m_FlgOverwrite ) 
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
    image = m_ImDiagnostic;

    if ( locusType == BREAST_EDGE )
    {
      if ( m_FlgVerbose ) 
        std::cout << "Creating diagnostic mammo breast edge mask." << std::endl;

      pPointOnBoundary = &m_DiagBreastEdgePoints; 
    }
    else
    {
      if ( m_FlgVerbose ) 
        std::cout << "Creating diagnostic mammo pectoral mask." << std::endl;

      pPointOnBoundary = &m_DiagPectoralPoints;
    }
  }
  else if ( mammoType == PREDIAGNOSTIC_MAMMO )
  {
    image = m_ImPreDiagnostic;

    if ( locusType == BREAST_EDGE )
    {
      if ( m_FlgVerbose ) 
        std::cout << "Creating prediagnostic mammo breast edge mask." << std::endl;

      pPointOnBoundary = &m_PreDiagBreastEdgePoints; 
    }
    else {
      if ( m_FlgVerbose ) 
        std::cout << "Creating prediagnostic mammo pectoral mask." << std::endl;

      pPointOnBoundary = &m_PreDiagPectoralPoints;
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

  if ( m_FlgDebug )
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
  polyMaskFilter->SetSpacing( image->GetSpacing() );

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

  if ( m_FlgDebug ) 
  {
    
    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      CastImageAndWriteToFile< unsigned char >( m_FileDiagnostic, 
                                                std::string( "_DiagBreastEdgeMask.dcm" ), 
                                                "diagnostic breast edge mask", 
                                                imMask, m_DiagDictionary );
    }
    else
    {
      CastImageAndWriteToFile< unsigned char >( m_FilePreDiagnostic, 
                                                std::string( "_PreDiagBreastEdgeMask.dcm" ), 
                                                "pre-diagnostic mask", 
                                                imMask, m_PreDiagDictionary );
    }

    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      CastImageAndWriteToFile< unsigned char >( m_FileDiagnostic, 
                                                std::string( "_DiagPectoralMask.dcm" ), 
                                                "diagnostic pectoral mask",
                                                imMaskPec, m_DiagDictionary );
    }
    else
    {
      CastImageAndWriteToFile< unsigned char >( m_FilePreDiagnostic, 
                                                std::string( "_PreDiagPectoralMask.dcm" ), 
                                                "pre-diagnostic mask", 
                                                imMaskPec, m_PreDiagDictionary );
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
// SetRegistrationParameterScales()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
template <typename ScalesType>
ScalesType
RegionalMammographicDensity< InputPixelType, InputDimension >
::SetRegistrationParameterScales( typename itk::TransformTypeEnum transformType,
                                  unsigned int nParameters )
{
  unsigned int i;
  ScalesType scales( nParameters);

  scales.Fill( 1.0 );      

  switch ( transformType )
  {

  // Translation only, 2 DOF in 2D and 3 DOF in 3D
  case TRANSLATION:
  {
    if ( nParameters != InputDimension )
    {
      std::cerr << "ERROR: Number of registration parameters does not equal dimension" << std::endl;
      exit( EXIT_FAILURE );
    }
    break;
  }

  // Rigid, so rotations and translations, 3 DOF in 2D and 6 DOF in 3D.
  case RIGID:
  {
    if (( InputDimension == 2) && (nParameters != 3) )
    {
      std::cerr << "ERROR: Rigid transformation should have 3 parameters in 2D" << std::endl;
      exit( EXIT_FAILURE );
    }
    else if (( InputDimension == 3) && (nParameters != 6) )
    {
      std::cerr << "ERROR: Rigid transformation should have 6 parameters in 3D" << std::endl;
      exit( EXIT_FAILURE );
    }
    break;
  }

  // Rigid plus scale, 5 DOF in 2D, 9 DOF in 3D.
  case RIGID_SCALE:
  {
    if ( InputDimension == 2 )
    {
      if ( nParameters != 5 )
      {
        std::cerr << "ERROR: Rigid plus scaletransformation should have 5 parameters in 2D" << std::endl;
        exit( EXIT_FAILURE );
      }

      for ( i=3; i<5; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    else if ( InputDimension == 3 )
    {
      if ( nParameters != 9 )
      {
        std::cerr << "ERROR: Rigid plus scale transformation should have 9 parameters in 3D" << std::endl;
        exit( EXIT_FAILURE );
      }
      for ( i=6; i<9; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    break;
  }

  // Affine. 6 DOF in 2D, 12 DOF in 3D.

  case AFFINE:
  {
    if ( InputDimension == 2 )
    {
      if ( nParameters != 6 )
      {
        std::cerr << "ERROR: Affine transformation should have 6 parameters in 2D" << std::endl;
        exit( EXIT_FAILURE );
      }

      for ( i=3; i<6; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    else if ( InputDimension == 3 )
    {
      if ( nParameters != 12 )
      {
        std::cerr << "ERROR: Affine transformation should have 12 parameters in 3D" << std::endl;
        exit( EXIT_FAILURE );
      }
      for ( i=6; i<12; i++ )
      {
        scales[i] = 100.0; 
      }
    }
    break;
  }

  default: {
    std::cerr << "ERROR: Unrecognised transformation type: " << transformType << std::endl;
    exit( EXIT_FAILURE );
  }
  }

  return scales;
};


// --------------------------------------------------------------------------
// RegisterTheImages()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
void
RegionalMammographicDensity< InputPixelType, InputDimension >
::RegisterTheImages( void )
{
  int finalInterpolator;
  int registrationInterpolator;
  int similarityMeasure;
  int transformation;
  int registrationStrategy;
  int optimizer;
  int bins;
  int iterations;
  int dilations;
  int levels;
  int startLevel;
  int stopLevel;
  double lowerIntensity;
  double higherIntensity;
  double dummyDefault;
  double paramTol;
  double funcTol;
  double maxStep;
  double minStep;
  double gradTol;
  double relaxFactor;
  double learningRate;
  double maskMinimumThreshold;
  double maskMaximumThreshold;
  double intensityFixedLowerBound;
  double intensityFixedUpperBound;
  double intensityMovingLowerBound;
  double intensityMovingUpperBound;
  double movingImagePadValue;
  int symmetricMetric;
  bool isRescaleIntensity;
  bool userSetPadValue;
  bool useWeighting; 
  double weightingThreshold; 
  double parameterChangeTolerance; 
  bool useCogInitialisation; 

  std::string outputMatrixTransformFile; 
  std::string outputUCLTransformFile;

  outputMatrixTransformFile = BuildOutputFilename( m_FileDiagnostic, "_PreDiagReg2Diag_Matrix.txt" );
  outputUCLTransformFile    = BuildOutputFilename( m_FileDiagnostic, "_PreDiagReg2Diag_UCLTransform.txt" );


  // Set defaults
  finalInterpolator = 4;
  registrationInterpolator = 2;
  similarityMeasure = 9;
  transformation = 4;
  registrationStrategy = 1;
  optimizer = 6;
  bins = 32;
  iterations = 300;
  dilations = 0;
  levels = 5;
  startLevel = 0;
  stopLevel = 2;
  lowerIntensity = 0;
  higherIntensity = 0;
  dummyDefault = -987654321;
  paramTol = 0.01;
  funcTol = 0.01;
  maxStep = 5.0;
  minStep = 0.01;
  gradTol = 0.01;
  relaxFactor = 0.5;
  learningRate = 0.5;
  maskMinimumThreshold = 0.5;
  maskMaximumThreshold = 255;
  intensityFixedLowerBound = dummyDefault;
  intensityFixedUpperBound = dummyDefault;
  intensityMovingLowerBound = dummyDefault;
  intensityMovingUpperBound = dummyDefault;
  movingImagePadValue = 0;
  symmetricMetric = 0;
  isRescaleIntensity = false;
  userSetPadValue = true;
  useWeighting = false; 
  useCogInitialisation = true; 
  
  // The factory.

  typename FactoryType::Pointer factory = FactoryType::New();
  
  // Start building.

  typename BuilderType::Pointer builder = BuilderType::New();

  builder->StartCreation( (itk::SingleResRegistrationMethodTypeEnum) registrationStrategy );
  builder->CreateInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );

  typename SimilarityMeasureType::Pointer metric = builder->CreateMetric( (itk::MetricTypeEnum) similarityMeasure );

  metric->SetSymmetricMetric( symmetricMetric );
  metric->SetUseWeighting( useWeighting ); 

  if (useWeighting)
  {
    metric->SetWeightingDistanceThreshold( weightingThreshold ); 
  }
  
  m_Transform = dynamic_cast< typename FactoryType::EulerAffineTransformType* >( builder->CreateTransform((itk::TransformTypeEnum) transformation, 
                                                                                                          static_cast<const ImageType * >( m_ImDiagnostic ) ).GetPointer() );
  int dof = m_Transform->GetNumberOfDOF(); 

  if ( niftk::FileExists( outputUCLTransformFile ) && ( ! m_FlgOverwrite ) )
  {
    m_Transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(builder->CreateTransform( outputUCLTransformFile ).GetPointer());
    m_Transform->SetNumberOfDOF(dof); 
    return;
  }
    
    
  typename ImageMomentCalculatorType::VectorType fixedImgeCOG; 
  typename ImageMomentCalculatorType::VectorType movingImgeCOG; 

  fixedImgeCOG.Fill(0.); 
  movingImgeCOG.Fill(0.); 
  
  // Calculate the CoG for the initialisation using CoG or for the symmetric transformation. 

  if (useCogInitialisation || symmetricMetric == 2)
  {
    typename ImageMomentCalculatorType::Pointer fixedImageMomentCalulator = ImageMomentCalculatorType::New(); 

    fixedImageMomentCalulator->SetImage(m_ImDiagnostic); 
    fixedImageMomentCalulator->Compute(); 
    fixedImgeCOG = fixedImageMomentCalulator->GetCenterOfGravity(); 

    typename ImageMomentCalculatorType::Pointer movingImageMomentCalulator = ImageMomentCalculatorType::New(); 

    movingImageMomentCalulator->SetImage(m_ImPreDiagnostic); 
    movingImageMomentCalulator->Compute(); 
    movingImgeCOG = movingImageMomentCalulator->GetCenterOfGravity(); 
  }
  
  if (symmetricMetric == 2)
  {
    builder->CreateFixedImageInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );
    builder->CreateMovingImageInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );
    
    // Change the center of the transformation for the symmetric transform. 

    typename ImageType::PointType centerPoint;

    for (unsigned int i = 0; i < InputDimension; i++)
      centerPoint[i] = (fixedImgeCOG[i] + movingImgeCOG[i])/2.; 

    typename FactoryType::EulerAffineTransformType::FullAffineTransformType* fullAffineTransform = m_Transform->GetFullAffineTransform();

    int dof = m_Transform->GetNumberOfDOF();
    m_Transform->SetCenter(centerPoint);
    m_Transform->SetNumberOfDOF(dof);
  }
  
  // Initialise the transformation using the CoG. 

  if (useCogInitialisation)
  {
    if (symmetricMetric == 2)
    {
      m_Transform->InitialiseUsingCenterOfMass(fixedImgeCOG/2.0, 
                                               movingImgeCOG/2.0); 
    }
    else
    {
      m_Transform->InitialiseUsingCenterOfMass(fixedImgeCOG, 
                                               movingImgeCOG); 

      typename ImageType::PointType centerPoint;

      centerPoint[0] = fixedImgeCOG[0];
      centerPoint[1] = fixedImgeCOG[1];

      m_Transform->SetCenter(centerPoint);
    }
  }
  
  builder->CreateOptimizer((itk::OptimizerTypeEnum)optimizer);

  // Get the single res method.

  typename SingleResImageRegistrationMethodType::Pointer singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
  typename MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  if ( m_FlgDebug )
  {
    singleResMethod->SetDebug( true );
    multiResMethod->SetDebug( true );
  }

  // Sort out metric and optimizer  

  typedef typename itk::SimilarityMeasure< ImageType, ImageType >  SimilarityType;
  typedef SimilarityType* SimilarityPointer;

  SimilarityPointer similarityPointer = dynamic_cast< SimilarityPointer >(singleResMethod->GetMetric());
  
  if (optimizer == itk::SIMPLEX)
  {
    typedef typename itk::UCLSimplexOptimizer OptimizerType;
    typedef OptimizerType*                    OptimizerPointer;

    OptimizerPointer op = dynamic_cast< OptimizerPointer >( singleResMethod->GetOptimizer() );

    op->SetMaximumNumberOfIterations (iterations );
    op->SetParametersConvergenceTolerance( paramTol );
    op->SetFunctionConvergenceTolerance( funcTol );
    op->SetAutomaticInitialSimplex( true );
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_Transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::GRADIENT_DESCENT)
  {
    typedef typename itk::GradientDescentOptimizer OptimizerType;
    typedef OptimizerType*                         OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetLearningRate(learningRate);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_Transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::REGSTEP_GRADIENT_DESCENT)
  {
    typedef typename itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
    typedef OptimizerType*                                       OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetMaximumStepLength(maxStep);
    op->SetMinimumStepLength(minStep);
    op->SetRelaxationFactor(relaxFactor);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_Transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::POWELL)
  {
    typedef typename itk::PowellOptimizer OptimizerType;
    typedef OptimizerType*                OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetMaximumIteration(iterations);
    op->SetStepLength(maxStep);
    op->SetStepTolerance(minStep);
    op->SetMaximumLineIteration(10);
    op->SetValueTolerance(0.0001);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());      

    OptimizerType::ScalesType scales = m_Transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::SIMPLE_REGSTEP)
  {
    typedef typename itk::UCLRegularStepOptimizer OptimizerType;
    typedef OptimizerType*                        OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetMaximumStepLength(maxStep);
    op->SetMinimumStepLength(minStep);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());
#if 1
    OptimizerType::ScalesType scales = m_Transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
#else
    OptimizerType::ScalesType scales;
    scales = SetRegistrationParameterScales< typename OptimizerType::ScalesType >((itk::TransformTypeEnum) transformation,
                                                                                  singleResMethod->GetTransform()->GetNumberOfParameters());
    op->SetScales(scales);      
#endif
  }
  else if (optimizer == itk::UCLPOWELL)
  {
    typedef itk::UCLPowellOptimizer OptimizerType;
    typedef OptimizerType*       OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetMaximumIteration(iterations);
    op->SetStepLength(maxStep);
    op->SetStepTolerance(minStep);
    op->SetMaximumLineIteration(15);
    op->SetValueTolerance(1.0e-14);
    op->SetParameterTolerance(parameterChangeTolerance);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());      

    OptimizerType::ScalesType scales = m_Transform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }

  // Finish configuring single-res object
  singleResMethod->SetNumberOfDilations(dilations);
  singleResMethod->SetThresholdFixedMask(true);
  singleResMethod->SetThresholdMovingMask(true);  
  singleResMethod->SetFixedMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(maskMaximumThreshold);
  
  if (isRescaleIntensity)
  {
    singleResMethod->SetRescaleFixedImage(true);
    singleResMethod->SetRescaleFixedMinimum((InputPixelType)lowerIntensity);
    singleResMethod->SetRescaleFixedMaximum((InputPixelType)higherIntensity);
    singleResMethod->SetRescaleMovingImage(true);
    singleResMethod->SetRescaleMovingMinimum((InputPixelType)lowerIntensity);
    singleResMethod->SetRescaleMovingMaximum((InputPixelType)higherIntensity);
  }
  
  // Finish configuring multi-res object.
  multiResMethod->SetInitialTransformParameters( singleResMethod->GetTransform()->GetParameters() );
  multiResMethod->SetSingleResMethod(singleResMethod);
  if (stopLevel > levels - 1)
  {
    stopLevel = levels - 1;
  }  
  multiResMethod->SetNumberOfLevels(levels);
  multiResMethod->SetStartLevel(startLevel);
  multiResMethod->SetStopLevel(stopLevel);

  if (intensityFixedLowerBound != dummyDefault ||
      intensityFixedUpperBound != dummyDefault ||
      intensityMovingLowerBound != dummyDefault ||
      intensityMovingUpperBound != dummyDefault)
  {
    if (isRescaleIntensity)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedBoundaryValue(lowerIntensity);
      singleResMethod->SetRescaleFixedLowerThreshold(intensityFixedLowerBound);
      singleResMethod->SetRescaleFixedUpperThreshold(intensityFixedUpperBound);
      singleResMethod->SetRescaleFixedMinimum((InputPixelType)lowerIntensity+1);
      singleResMethod->SetRescaleFixedMaximum((InputPixelType)higherIntensity);
          
      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingBoundaryValue(lowerIntensity);
      singleResMethod->SetRescaleMovingLowerThreshold(intensityMovingLowerBound);
      singleResMethod->SetRescaleMovingUpperThreshold(intensityMovingUpperBound);
      singleResMethod->SetRescaleMovingMinimum((InputPixelType)lowerIntensity+1);
      singleResMethod->SetRescaleMovingMaximum((InputPixelType)higherIntensity);

      metric->SetIntensityBounds(lowerIntensity+1, higherIntensity, lowerIntensity+1, higherIntensity);
    }
    else
    {
      metric->SetIntensityBounds(intensityFixedLowerBound, intensityFixedUpperBound, intensityMovingLowerBound, intensityMovingUpperBound);
    }
  }

  try
  {
    // The main filter.
    typename RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    filter->SetMultiResolutionRegistrationMethod(multiResMethod);

    std::cout << "Setting fixed image"<< std::endl;
    filter->SetFixedImage(m_ImDiagnostic);

    std::cout << "Setting moving image"<< std::endl;
    filter->SetMovingImage(m_ImPreDiagnostic);

    if (0)
    {
      std::cout << "Setting fixed mask"<< std::endl;
      filter->SetFixedMask(m_ImDiagnosticMask);  
    }
      
    if (0)
    {
      std::cout << "Setting moving mask"<< std::endl;
      filter->SetMovingMask(m_ImPreDiagnosticMask);
    }

    // If we havent asked for output, turn off reslicing.
    filter->SetDoReslicing(true);

    filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterpolator));
    
    // Set the padding value
    if (!userSetPadValue)
    {
      typename ImageType::IndexType index;
      for (unsigned int i = 0; i < InputDimension; i++)
      {
        index[i] = 0;  
      }
      movingImagePadValue = m_ImPreDiagnostic->GetPixel(index);
      std::cout << "Setting  moving image pad value to:" 
        + niftk::ConvertToString(movingImagePadValue)<< std::endl;
    }
    similarityPointer->SetTransformedMovingImagePadValue(movingImagePadValue);
    filter->SetResampledMovingImagePadValue(movingImagePadValue);
    
    // Run the registration
    filter->Update();
    
    // And write the output.
    WriteImageFile< OutputImageType >( m_FileDiagnostic, 
                                       std::string( "_PreDiagReg2Diag.dcm" ), 
                                       "registered diagnostic image", 
                                       filter->GetOutput(), 
                                       m_DiagDictionary );
    
    // Make sure we get the final one.
    m_Transform = dynamic_cast< typename FactoryType::EulerAffineTransformType* >(singleResMethod->GetTransform());
    m_Transform->SetFullAffine(); 
    
    // Save the transform (as 12 parameter UCLEulerAffine transform).
    typedef typename itk::TransformFileWriter TransformFileWriterType;
    typename TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    transformFileWriter->SetInput(m_Transform);
    transformFileWriter->SetFileName(outputUCLTransformFile);
    transformFileWriter->Update();         
    
    // Save the transform (as 16 parameter matrix transform).
    if (outputMatrixTransformFile.length() > 0)
    {
      transformFileWriter->SetInput(m_Transform->GetFullAffineTransform());
      transformFileWriter->SetFileName(outputMatrixTransformFile);
      transformFileWriter->Update(); 
    }
    
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception caught:" << std::endl;
    std::cerr << excp << std::endl;
    exit( EXIT_FAILURE );
  }
};


// --------------------------------------------------------------------------
// TransformTumourPositionIntoPreDiagImage()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::LabelImageType::IndexType
RegionalMammographicDensity< InputPixelType, InputDimension >
::TransformTumourPositionIntoPreDiagImage( typename LabelImageType::IndexType &idxTumourCenter )
{
  if ( ! m_Transform )
  {
    std::cerr << "ERROR: Cannot transform tumour position - no transform available" << std::endl;
    exit( EXIT_FAILURE );
  }

  typename LabelImageType::IndexType preDiagCenterIndex;
  typename LabelImageType::PointType inPoint;
  typename LabelImageType::PointType outPoint;


  m_ImDiagnostic->TransformIndexToPhysicalPoint( idxTumourCenter, inPoint );

  outPoint = m_Transform->TransformPoint( inPoint );

  m_ImPreDiagnostic->TransformPhysicalPointToIndex( outPoint, preDiagCenterIndex );

  if ( m_FlgDebug )
  {
    m_Transform->Print( std::cout );
    std::cout << "Tumour index: " << idxTumourCenter 
              << ", point: " << inPoint << std::endl
              << "  transforms to point: " << outPoint
              << ", index: " << preDiagCenterIndex << std::endl
              << std::endl;
  }

  return preDiagCenterIndex;
};


// --------------------------------------------------------------------------
// GenerateRegionLabels()
// --------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension>
typename RegionalMammographicDensity< InputPixelType, InputDimension >::LabelImageType::Pointer 
RegionalMammographicDensity< InputPixelType, InputDimension >
::GenerateRegionLabels( typename LabelImageType::IndexType &idxTumourCenter,
                        typename LabelImageType::RegionType &tumourRegion,
                        LabelPixelType &tumourRegionValue,
                        typename ImageType::Pointer &image,
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

  regionSizeInPixels[0] = (int) ceil(m_RegionSizeInMM / imSpacing[0]);
  regionSizeInPixels[1] = (int) ceil(m_RegionSizeInMM / imSpacing[1]);

  if ( (regionSizeInPixels[0] > imSizeInPixels[0]) ||
       (regionSizeInPixels[1] > imSizeInPixels[1]) )
  {
    std::cerr << "ERROR: Region size in pixels (" 
              << regionSizeInPixels[0] << "x"<< regionSizeInPixels[1]
              << " is larger than the image (" 
              << imSizeInPixels[0] << "x"<< imSizeInPixels[1] << ")"
              << std::endl;
    exit( EXIT_FAILURE );
  }

  regionRadiusInPixels[0] = regionSizeInPixels[0]/2;
  regionRadiusInPixels[1] = regionSizeInPixels[1]/2;

  regionSizeInPixels[0] = 2*regionRadiusInPixels[0] + 1;
  regionSizeInPixels[1] = 2*regionRadiusInPixels[1] + 1;

  imSizeInROIs[0] = (int) ceil( ((float) imSizeInPixels[0]) 
                                / ((float) regionSizeInPixels[0]) );

  imSizeInROIs[1] = (int) ceil( ((float) imSizeInPixels[1]) 
                                / ((float) regionSizeInPixels[1]) );

  // Get the index of the tumour ROI

  tumourRegionIndex[0] = idxTumourCenter[0] - regionRadiusInPixels[0];
  tumourRegionIndex[1] = idxTumourCenter[1] - regionRadiusInPixels[1];

  if ( (tumourRegionIndex[0] < 0) || (tumourRegionIndex[1] < 0) )
  {
    std::cerr << "ERROR: The corner of tumour region falls outside the image."
              << std::endl << "       The region size is probably too big."
              << std::endl;
    exit( EXIT_FAILURE );
  }

  // Hence the origin of the label grid

  labelOrigin[0] = (int) floor(((float) tumourRegionIndex[0]) - 
    ((float) regionSizeInPixels[0])*ceil( ((float) tumourRegionIndex[0]) 
                                          / ((float) regionSizeInPixels[0]) ) + 0.5);

  labelOrigin[1] = (int) floor(((float) tumourRegionIndex[1]) - 
    ((float) regionSizeInPixels[1])*ceil( ((float) tumourRegionIndex[1]) 
                                          / ((float) regionSizeInPixels[1]) ) + 0.5);

  tumourRegion.SetIndex( tumourRegionIndex );
  tumourRegion.SetSize( regionSizeInPixels );

  if ( m_FlgDebug )
    std::cout << "  Region size (mm): " << m_RegionSizeInMM << std::endl
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

  tumourRegionValue = imSizeInROIs[0]*((idxTumourCenter[1] - labelOrigin[1]) 
                                       / regionSizeInPixels[1]) 
    + (idxTumourCenter[0] - labelOrigin[0]) / regionSizeInPixels[0];

  if ( m_FlgVerbose )
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
