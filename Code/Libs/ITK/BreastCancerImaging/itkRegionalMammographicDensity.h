/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkRegionalMammographicDensity_h
#define itkRegionalMammographicDensity_h

#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision

#include <itkObject.h>
#include <itkImage.h>

#include <itkImageFileReader.h>
#include <itkPointSet.h>
#include <itkVector.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkPolygonSpatialObject.h>
 
#include <itkTransformFileWriter.h>
#include <itkImageMomentsCalculator.h>

#include <itkMammogramLeftOrRightSideCalculator.h>
#include <itkMammogramRegistrationFilter.h>

#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>

#include <boost/random/mersenne_twister.hpp>


/*!
 * \file niftkRegionalMammographicDensity.cxx
 * \page niftkRegionalMammographicDensity
 * \section niftkRegionalMammographicDensitySummary Calculates the density within regions on interest across a mammogram.
 *
 * \section niftkRegionalMammographicDensityCaveats Caveats
 * \li None
 */


namespace itk
{



// -----------------------------------------------------------------------------------
// Class to handle the boundary points i.e. pectoral muscle and breast edge
// -----------------------------------------------------------------------------------

class PointOnBoundary
{
public:
  int id;
  int x;
  int y;

  PointOnBoundary() {
    id = 0;
    x  = 0;
    y  = 0;
  }

  PointOnBoundary(int idIn, int xIn, int yIn) {
    id = idIn;
    x  = xIn;
    y  = yIn;
  }

  float DistanceTo( PointOnBoundary c ) {
    return sqrt(static_cast<double>( (x - c.x)*(x - c.x) + (y - c.y)*(y - c.y) ) );
  }

  void Print( const char *indent ) {
    std::cout << indent
              << std::right << std::setw(6) << id << ": "
              << std::right << std::setw(6) << x << ", "
              << std::right << std::setw(6) << y << std::endl;
  }
  
};


// -----------------------------------------------------------------------------------
// Class to handle a region or patch in the image
// -----------------------------------------------------------------------------------

class Patch
{
public:

  Patch() {
    iPatch = 0;
    jPatch = 0;

    nPixels = 0;
    nDensePixels  = 0;
    sumXindices = 0;
    sumYindices = 0;
  }

  void SetCoordinate( int i, int j ) {
    iPatch = i;
    jPatch = j;
  }

  void GetCoordinate( int &i, int &j ) {
    i = iPatch;
    j = jPatch;
  }

  void AddDensePixel( float xIndex, float yIndex ) {
    nDensePixels++;
    nPixels++;
    sumXindices += xIndex;
    sumYindices += yIndex;
  }

  void AddNonDensePixel( float xIndex, float yIndex ) {
    nPixels++;
    sumXindices += xIndex;
    sumYindices += yIndex;
  }

  float GetNumberOfPixels( void ) { return nPixels; }
  float GetNumberOfDensePixels( void ) { return nDensePixels; }

  void GetCenter( float &xIndex, float &yIndex ) { 
    if (nPixels > 0) { 
      xIndex = sumXindices/nPixels;
      yIndex = sumYindices/nPixels;
    }
    else {
      xIndex = 0;
      yIndex = 0;
    }
  }

  void Print( const char *indent, float maxNumberOfPixels ) {
    float xCenter, yCenter;
    GetCenter(  xCenter, yCenter );
    std::cout << indent
              << "index: (" << std::setw(4) << iPatch 
              << ", " << std::setw(4) << jPatch << ") "
              << "center: (" << xCenter << ", " << yCenter << ") "
              << indent
              << "no. of dense pixels: "
              << std::right << std::setprecision( 6 ) << std::setw(12) << nDensePixels << " ( "
              << std::fixed << std::setprecision( 2 )
              << std::right << std::setw(7) << 100.*nDensePixels/nPixels << "% )";

    std::cout.unsetf( std::ios_base::fixed );

    std::cout << indent
              << "no. of pixels in patch: "
              << std::right << std::setprecision( 6 ) << std::setw(12) << nPixels << " ( "
              << std::fixed << std::setprecision( 2 )
              << std::right << std::setw(7) << 100.*nPixels/maxNumberOfPixels << "% )"
              << std::setprecision( 0 ) << std::endl;

    std::cout.unsetf( std::ios_base::fixed );
  }
  
protected:

  int iPatch;
  int jPatch;

  float nPixels;
  float nDensePixels;
  float sumXindices;
  float sumYindices;

};


// -----------------------------------------------------------------------------------
// Class to store the data for diagnostic and pre-diagnostic images of a patient
// -----------------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension=2>
class ITK_EXPORT RegionalMammographicDensity
  : public Object
{
public:
    
  typedef RegionalMammographicDensity   Self;
  typedef Object                        Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;
  
  itkNewMacro(Self); 
  itkTypeMacro(RegionalMammographicDensity, Object);

  itkStaticConstMacro( ParametricDimension, unsigned int, 2 );
  itkStaticConstMacro( DataDimension, unsigned int, 1 );

  typedef itk::Image< InputPixelType, InputDimension > ImageType;
  typedef itk::ImageFileReader< ImageType > ReaderType;
  typedef itk::ImageRegionIterator< ImageType > IteratorType;
  typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorWithIndexType;

  typedef unsigned int LabelPixelType;
  typedef Image< LabelPixelType, InputDimension> LabelImageType;

  typedef short int OutputPixelType;
  typedef itk::Image< OutputPixelType , InputDimension >  OutputImageType;

  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  typedef float RealType;
    
  typedef itk::Vector<RealType,     DataDimension>        VectorType;
  typedef itk::Image<VectorType,    ParametricDimension>  VectorImageType;
  typedef itk::PointSet<VectorType, ParametricDimension>  PointSetType;

  typedef itk::Image< RealType, InputDimension >          RealImageType;

  typedef typename PointSetType::PointsContainer          PointsContainer;
  typedef typename PointsContainer::Iterator              PointsIterator;
  typedef typename PointSetType::PointDataContainer       PointDataContainer;
  typedef typename PointDataContainer::Iterator           PointDataIterator;

  typedef itk::PolygonSpatialObject< InputDimension > PolygonType;

  typedef typename itk::MammogramLeftOrRightSideCalculator< ImageType > 
    LeftOrRightSideCalculatorType;

  typedef typename LeftOrRightSideCalculatorType::BreastSideType BreastSideType;


  // Setup objects to build registration.

  typedef typename itk::ImageMomentsCalculator< ImageType > ImageMomentCalculatorType;
  typedef typename itk::SignedMaurerDistanceMapImageFilter< ImageType, RealImageType> DistanceTransformType;

  typedef typename itk::MammogramRegistrationFilter< ImageType, ImageType > RegistrationFilterType;

  typedef typename RegistrationFilterType::enumRegistrationImagesType enumRegistrationImagesType;
  
  /// Set the registration image type.
  void SetTypeOfInputImagesToRegister( enumRegistrationImagesType regImagesType ) { 
    m_TypeOfInputImagesToRegister = regImagesType;
  }

  enum MammogramType 
  { 
    UNKNOWN_MAMMO_TYPE,
    DIAGNOSTIC_MAMMO,
    PREDIAGNOSTIC_MAMMO,
    CONTROL_MAMMO
  };

  const static char* MammogramTypeNames[];

  enum LocusType 
  { 
    UNKNOWN_LOCUS_TYPE,
    BREAST_EDGE,
    PECTORAL
  };


  void SetPatientID( std::string idPatient ) { m_Id = idPatient; }

  void SetInputDirectory( std::string dirIn ) { m_DirInput = dirIn; }
  void SetOutputDirectory( std::string dirOut ) { m_DirOutput = dirOut; }

  void SetIDDiagnosticImage( std::string idDiagImage )       { m_IdDiagnosticImage    = idDiagImage; }
  void SetIDPreDiagnosticImage( std::string idPreDiagImage ) { m_IdPreDiagnosticImage = idPreDiagImage; }
  void SetIDControlImage( std::string idControlImage )       { m_IdControlImage       = idControlImage; }

  void SetFileDiagnostic( std::string fileDiag )       { m_FileDiagnostic    = fileDiag; }
  void SetFilePreDiagnostic( std::string filePreDiag ) { m_FilePreDiagnostic = filePreDiag; }
  void SetFileControl( std::string fileControl )       { m_FileControl       = fileControl; }

  void SetTumourID( std::string strTumID )           { m_StrTumourID          = strTumID; }
  void SetTumourImageID( std::string strTumImageID ) { m_StrTumourImageID     = strTumImageID; }

  void SetThresholdDiagnostic(    int thrDiag    ) { m_ThresholdDiagnostic    = thrDiag; }
  void SetThresholdPreDiagnostic( int thrPreDiag ) { m_ThresholdPreDiagnostic = thrPreDiag; }
  void SetThresholdControl(       int thrControl ) { m_ThresholdControl       = thrControl; }

  void SetTumourLeft(   int tumLeft )   { m_TumourLeft   = tumLeft; }
  void SetTumourRight(  int tumRight )  { m_TumourRight  = tumRight; }
  void SetTumourTop(    int tumTop )    { m_TumourTop    = tumTop; }
  void SetTumourBottom( int tumBottom ) { m_TumourBottom = tumBottom; }

  void SetTumourDiameter( float diameter ) { m_TumourDiameter = diameter; }

  void SetRegionSizeInMM( float roiSize ) { m_RegionSizeInMM = roiSize; }

  void SetRegisterOn( void ) { m_FlgRegister = true; }
  void SetRegisterOff( void ) { m_FlgRegister = false;
                                m_FlgRegisterNonRigid = false; }

  void SetVerboseOn( void ) { m_FlgVerbose = true; }
  void SetVerboseOff( void ) { m_FlgVerbose = false; }

  void SetOverwriteOn( void ) { m_FlgOverwrite = true; }
  void SetOverwriteOff( void ) { m_FlgOverwrite = false; }

  void SetDebugOn( void ) { m_FlgDebug = true; }
  void SetDebugOff( void ) { m_FlgDebug = false; }

  /// Specify whether to perform a non-rigid registration
  void SetRegisterNonRigidOn( void ) { m_FlgRegister = true; 
                                       m_FlgRegisterNonRigid = true; }
  void SetRegisterNonRigidOff( void ) { m_FlgRegisterNonRigid = false; }
 

  std::string GetPatientID( void ) { return m_Id; }

  std::string GetIDDiagnosticImage( void )    { return m_IdDiagnosticImage; }
  std::string GetIDPreDiagnosticImage( void ) { return m_IdPreDiagnosticImage; }
  std::string GetIDControlImage( void ) { return m_IdControlImage; }

  std::string GetFileDiagnostic( void )    { return m_FileDiagnostic; }
  std::string GetFilePreDiagnostic( void ) { return m_FilePreDiagnostic; }
  std::string GetFileControl( void ) { return m_FileControl; }

  std::string GetStrTumourID( void )      { return m_StrTumourID; }
  std::string GetStrTumourImageID( void ) { return m_StrTumourImageID; }

  int GetThresholdDiagnostic( void )    { return m_ThresholdDiagnostic; }
  int GetThresholdPreDiagnostic( void ) { return m_ThresholdPreDiagnostic; }
  int GetThresholdControl( void ) { return m_ThresholdControl; }

  int GetTumourLeft( void )   { return m_TumourLeft; }
  int GetTumourRight( void )  { return m_TumourRight; }
  int GetTumourTop( void )    { return m_TumourTop; }
  int GetTumourBottom( void ) { return m_TumourBottom; }

  float GetTumourDiameter( void ) { return m_TumourDiameter; }

  void LoadImages( void );
  void UnloadImages( void );

  void Print( bool flgVerbose = false );

  void PushBackBreastEdgeCoord( std::string strBreastEdgeImageID, 
                                int id, int x, int y );

  void PushBackPectoralCoord( std::string strPectoralImageID, 
                              int id, int x, int y ); 

  void WriteDataToCSVFile( std::ofstream *foutOutputDensityCSV );

  void Compute();

protected:

  /// Constructor
  RegionalMammographicDensity();

  /// Destructor
  virtual ~RegionalMammographicDensity();

  bool m_FlgVerbose;
  bool m_FlgDebug;

  bool m_FlgOverwrite;
  bool m_FlgRegister;
  
  /// Specify whether to perform a non-rigid registration
  bool m_FlgRegisterNonRigid;
  
  /// Specify the input images to register
  enumRegistrationImagesType m_TypeOfInputImagesToRegister;

  std::string m_Id;

  std::string m_DirInput;
  std::string m_DirOutput;

  // The diagnostic image

  std::string m_IdDiagnosticImage;
  std::string m_FileDiagnostic;
  std::string m_FileDiagnosticRegn;
  std::string m_FileDiagnosticRegnMask;

  int m_ThresholdDiagnostic;

  BreastSideType m_BreastSideDiagnostic;

  // The pre-diagnostic image

  std::string m_IdPreDiagnosticImage;
  std::string m_FilePreDiagnostic;
  std::string m_FilePreDiagnosticRegn;

  int m_ThresholdPreDiagnostic;

  BreastSideType m_BreastSidePreDiagnostic;

  // The control image

  std::string m_IdControlImage;
  std::string m_FileControl;
  std::string m_FileControlRegn;

  int m_ThresholdControl;

  BreastSideType m_BreastSideControl;

  // The tumour

  std::string m_StrTumourID;
  std::string m_StrTumourImageID;

  int m_TumourLeft;
  int m_TumourRight;
  int m_TumourTop;
  int m_TumourBottom;

  float m_TumourDiameter;

  LabelPixelType m_DiagTumourRegionValue;

  typename LabelImageType::IndexType  m_DiagTumourCenterIndex;
  typename LabelImageType::RegionType m_DiagTumourRegion;

  LabelPixelType m_PreDiagTumourRegionValue;

  typename LabelImageType::IndexType m_PreDiagCenterIndex;
  typename LabelImageType::RegionType m_PreDiagTumourRegion;

  LabelPixelType m_ControlTumourRegionValue;

  typename LabelImageType::IndexType m_ControlCenterIndex;
  typename LabelImageType::RegionType m_ControlTumourRegion;

  // The region of interest size

  float m_RegionSizeInMM;

  DictionaryType m_DiagDictionary;
  DictionaryType m_PreDiagDictionary;
  DictionaryType m_ControlDictionary;


  typename ImageType::Pointer m_ImDiagnostic;
  typename ImageType::Pointer m_ImPreDiagnostic;
  typename ImageType::Pointer m_ImControl;

  typename ImageType::Pointer m_ImDiagnosticMask;
  typename ImageType::Pointer m_ImPreDiagnosticMask;
  typename ImageType::Pointer m_ImControlMask;

  typename LabelImageType::Pointer m_ImDiagnosticLabels;
  typename LabelImageType::Pointer m_ImPreDiagnosticLabels;
  typename LabelImageType::Pointer m_ImControlLabels;

  typename ImageType::Pointer m_ImDiagnosticRegnMask;
  typename ImageType::Pointer m_ImPreDiagnosticRegnMask;
  typename ImageType::Pointer m_ImControlRegnMask;


  std::vector< PointOnBoundary > m_DiagBreastEdgePoints;
  std::vector< PointOnBoundary > m_PreDiagBreastEdgePoints;
  std::vector< PointOnBoundary > m_ControlBreastEdgePoints;
  
  std::vector< PointOnBoundary > m_DiagPectoralPoints;
  std::vector< PointOnBoundary > m_PreDiagPectoralPoints;
  std::vector< PointOnBoundary > m_ControlPectoralPoints;


  std::map< LabelPixelType, Patch > m_DiagPatches;
  std::map< LabelPixelType, Patch > m_PreDiagPatches;
  std::map< LabelPixelType, Patch > m_ControlPatches;


  typename RegistrationFilterType::Pointer m_RegistrationPreDiag;
  typename RegistrationFilterType::Pointer m_RegistrationControl;

  void PrintDictionary( DictionaryType &dictionary );

  typename ImageType::SpacingType GetImageResolutionFromDictionary( DictionaryType &dictionary );

  bool ReadImage( MammogramType mammoType );

  std::string BuildOutputFilename( std::string fileInput, std::string suffix );

  template < typename TOutputImageType >
  std::string CastImageAndWriteToFile( std::string fileInput, 
                                       std::string suffix,
                                       const char *description,
                                       typename ImageType::Pointer image,
                                       DictionaryType &dictionary );

  template < typename TOutputImageType >
  void WriteImageFile( std::string fileInput, 
                       std::string suffix, 
                       const char *description,
                       typename TOutputImageType::Pointer image,
                       DictionaryType &dictionary );

  void WriteLabelImageFile( std::string fileInput, 
                            std::string suffix, 
                            const char *description,
                            typename LabelImageType::Pointer image,
                            typename LabelImageType::RegionType &tumourRegion,
                            DictionaryType &dictionary );

  void WriteRegistrationDifferenceImage( std::string fileInput, 
                                         std::string suffix, 
                                         const char *description,
                                         typename ImageType::Pointer image,
                                         DictionaryType &dictionary );

  void AddPointToPolygon( typename PolygonType::Pointer &polygon, 
                          typename ImageType::Pointer &image, 
                          typename ImageType::SizeType &polySize, 
                          int x, int y );

  typename ImageType::Pointer MaskWithPolygon( MammogramType mammoType, 
                                               LocusType locusType );

  typename ImageType::Pointer MaskWithPolygon( MammogramType mammoType );

  void RemoveTumourFromRegnMask( void );

  template <typename ScalesType>
  ScalesType SetRegistrationParameterScales( typename itk::TransformTypeEnum transformType,
                                             unsigned int nParameters );

  void RegisterTheImages();

  typename RegistrationFilterType::Pointer 
    RegisterTheImages( typename ImageType::Pointer imSource,
                       std::string fileInputSource,
                       typename ImageType::Pointer maskSource,
                       
                       std::string fileOutputAffineTransformation,
                       std::string fileOutputAffineRegistered,
                       
                       std::string fileOutputNonRigidTransformation,
                       std::string fileOutputNonRigidRegistered );

  typename LabelImageType::IndexType
    TransformTumourPositionIntoImage( typename LabelImageType::IndexType &idxTumourCenter,
                                      typename ImageType::Pointer &image,
                                      typename RegistrationFilterType::Pointer registration );

  typename LabelImageType::Pointer 
       GenerateRegionLabels( BreastSideType breastSide,
                             typename LabelImageType::IndexType &idxTumourCenter,
                             typename LabelImageType::RegionType &tumourRegion,
                             LabelPixelType &tumourRegionValue,
                             typename ImageType::Pointer &image,
                             typename ImageType::Pointer &imMask,
                             typename std::map< LabelPixelType, Patch > &listOfPatches,
                             int threshold );

private:

  void GenerateRandomTumourPositionInImage( MammogramType mammoType );


  RegionalMammographicDensity(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  boost::random::mt19937 m_Gen;

};

} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRegionalMammographicDensity.txx"
#endif

#endif


