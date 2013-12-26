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

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <itkObject.h>
#include <vtkSmartPointer.h>
#include <itkImage.h>

#include <itkImageFileReader.h>
#include <itkPointSet.h>
#include <itkVector.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkPolygonSpatialObject.h>

#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>


/*!
 * \file niftkRegionalDensity.cxx
 * \page niftkRegionalDensity
 * \section niftkRegionalDensitySummary Calculates the density within regions on interest across a mammogram.
 *
 * \section niftkRegionalDensityCaveats Caveats
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
    nPixels = 0;
    nDensePixels  = 0;
  }

  void AddDensePixel( void ) {
    nDensePixels++;
    nPixels++;
  }

  void AddNonDensePixel( void ) {
    nPixels++;
  }

  float GetNumberOfPixels( void ) { return nPixels; }
  float GetNumberOfDensePixels( void ) { return nDensePixels; }

  void Print( const char *indent, float maxNumberOfPixels ) {
    std::cout << indent
              << "no. of dense pixels: "
              << std::right << setprecision( 6 )<< std::setw(12) << nDensePixels << " ( "
              << std::fixed << setprecision( 2 )
              << std::right << std::setw(7) << 100.*nDensePixels/nPixels << "% )";

    std::cout.unsetf( std::ios_base::fixed );

    std::cout << indent
              << "no. of pixels in patch: "
              << std::right << setprecision( 6 ) << std::setw(12) << nPixels << " ( "
              << std::fixed << setprecision( 2 )
              << std::right << std::setw(7) << 100.*nPixels/maxNumberOfPixels << "% )"
              << setprecision( 0 ) << std::endl;

    std::cout.unsetf( std::ios_base::fixed );
  }
  
protected:

  float nPixels;
  float nDensePixels;  

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

  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  typedef float RealType;
    
  typedef itk::Vector<RealType,     DataDimension>        VectorType;
  typedef itk::Image<VectorType,    ParametricDimension>  VectorImageType;
  typedef itk::PointSet<VectorType, ParametricDimension>  PointSetType;

  typedef typename PointSetType::PointsContainer          PointsContainer;
  typedef typename PointsContainer::Iterator              PointsIterator;
  typedef typename PointSetType::PointDataContainer       PointDataContainer;
  typedef typename PointDataContainer::Iterator           PointDataIterator;

  typedef itk::PolygonSpatialObject< InputDimension > PolygonType;


  enum MammogramType 
  { 
    UNKNOWN_MAMMO_TYPE,
    DIAGNOSTIC_MAMMO,
    PREDIAGNOSTIC_MAMMO
  };
  
  enum LocusType 
  { 
    UNKNOWN_LOCUS_TYPE,
    BREAST_EDGE,
    PECTORAL
  };


  void SetPatientID( std::string &idPatient ) { id = idPatient; }

  void SetOutputDirectory( std::string &dirOut ) { dirOutput = dirOut; }

  void SetIDDiagnosticImage( std::string &idDiagImage )       { idDiagnosticImage    = idDiagImage; }
  void SetIDPreDiagnosticImage( std::string &idPreDiagImage ) { idPreDiagnosticImage = idPreDiagImage; }

  void SetFileDiagnostic( std::string &fileDiag )       { fileDiagnostic       = fileDiag; }
  void SetFilePreDiagnostic( std::string &filePreDiag ) { filePreDiagnostic    = filePreDiag; }

  void SetTumourID( std::string &strTumID )           { strTumourID          = strTumID; }
  void SetTumourImageID( std::string &strTumImageID ) { strTumourImageID     = strTumImageID; }

  void SetThresholdDiagnostic(    int  thrDiag )   { thresholdDiagnostic    = thrDiag; }
  void SetThresholdPreDiagnostic( int thrPreDiag ) { thresholdPreDiagnostic = thrPreDiag; }

  void SetTumourLeft(   int tumLeft )   { tumourLeft   = tumLeft; }
  void SetTumourRight(  int tumRight )  { tumourRight  = tumRight; }
  void SetTumourTop(    int tumTop )    { tumourTop    = tumTop; }
  void SetTumourBottom( int tumBottom ) { tumourBottom = tumBottom; }

  void SetRegionSizeInMM( float roiSize ) { regionSizeInMM = roiSize; }

  void SetVerboseOn( void ) { flgVerbose = true; }
  void SetVerboseOff( void ) { flgVerbose = false; }

  void SetOverwriteOn( void ) { flgOverwrite = true; }
  void SetOverwriteOff( void ) { flgOverwrite = false; }

  void SetDebugOn( void ) { flgDebug = true; }
  void SetDebugOff( void ) { flgDebug = false; }


  std::string GetPatientID( void ) { return id; }

  std::string GetIDDiagnosticImage( void )    { return idDiagnosticImage; }
  std::string GetIDPreDiagnosticImage( void ) { return idPreDiagnosticImage; }

  std::string GetFileDiagnostic( void )    { return fileDiagnostic; }
  std::string GetFilePreDiagnostic( void ) { return filePreDiagnostic; }

  std::string GetStrTumourID( void )      { return strTumourID; }
  std::string GetStrTumourImageID( void ) { return strTumourImageID; }

  int GetThresholdDiagnostic( void )    { return thresholdDiagnostic; }
  int GetThresholdPreDiagnostic( void ) { return thresholdPreDiagnostic; }

  int GetTumourLeft( void )   { return tumourLeft; }
  int GetTumourRight( void )  { return tumourRight; }
  int GetTumourTop( void )    { return tumourTop; }
  int GetTumourBottom( void ) { return tumourBottom; }

  void LoadImages( void );
  void UnloadImages( void );

  void Print( bool flgVerbose = false );

  void PushBackBreastEdgeCoord( std::string strBreastEdgeImageID, 
                                int id, int x, int y );

  void PushBackPectoralCoord( std::string strPectoralImageID, 
                              int id, int x, int y ); 

  void WriteDataToCSVFile( std::ofstream *foutOutputDensityCSV,
                           boost::random::mt19937 &gen );


  void Compute( void );


protected:

  /// Constructor
  RegionalMammographicDensity();

  /// Destructor
  virtual ~RegionalMammographicDensity();

  bool flgVerbose;
  bool flgOverwrite;
  bool flgDebug;

  std::string id;

  std::string dirOutput;

  // The diagnostic image

  std::string idDiagnosticImage;
  std::string fileDiagnostic;

  int thresholdDiagnostic;

  // The pre-diagnostic image

  std::string idPreDiagnosticImage;
  std::string filePreDiagnostic;

  int thresholdPreDiagnostic;

  // The tumour

  std::string strTumourID;
  std::string strTumourImageID;

  int tumourLeft;
  int tumourRight;
  int tumourTop;
  int tumourBottom;

  LabelPixelType tumourRegionValue;

  typename LabelImageType::IndexType tumourCenterIndex;
  typename LabelImageType::RegionType tumourRegion;

  // The region of interest size

  float regionSizeInMM;

  DictionaryType diagDictionary;
  DictionaryType preDiagDictionary;


  typename ImageType::Pointer imDiagnostic;
  typename ImageType::Pointer imPreDiagnostic;


  typename ImageType::Pointer imDiagnosticMask;
  typename ImageType::Pointer imPreDiagnosticMask;

  typename LabelImageType::Pointer imDiagnosticLabels;
  typename LabelImageType::Pointer imPreDiagnosticLabels;


  std::vector< PointOnBoundary > diagBreastEdgePoints;
  std::vector< PointOnBoundary > preDiagBreastEdgePoints;
  
  std::vector< PointOnBoundary > diagPectoralPoints;
  std::vector< PointOnBoundary > preDiagPectoralPoints;


  std::map< LabelPixelType, Patch > diagPatches;
  std::map< LabelPixelType, Patch > preDiagPatches;

  
  void PrintDictionary( DictionaryType &dictionary );

  void ReadImage( MammogramType mammoType );

  std::string BuildOutputFilename( std::string fileInput, std::string suffix );

  void WriteBinaryImageToUCharFile( std::string fileInput, 
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
                            DictionaryType &dictionary );

  void AddPointToPolygon( typename PolygonType::Pointer &polygon, 
                          typename ImageType::Pointer &image, 
                          typename ImageType::SizeType &polySize, 
                          int x, int y );

  typename ImageType::Pointer MaskWithPolygon( MammogramType mammoType, 
                                               LocusType locusType );

  typename ImageType::Pointer MaskWithPolygon( MammogramType mammoType );

  typename LabelImageType::Pointer 
    GenerateRegionLabels( typename ImageType::Pointer &image,
                          typename ImageType::Pointer &imMask,
                          typename std::map< LabelPixelType, Patch > &listOfPatches,
                          int threshold );

private:

  RegionalMammographicDensity(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRegionalMammographicDensity.txx"
#endif

#endif


