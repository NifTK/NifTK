/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatEstimationFitMetricForImage_h
#define __itkMammogramFatEstimationFitMetricForImage_h

#include <itkMammogramFatEstimationFitMetric.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

namespace itk {
  
/** \class MammogramFatEstimationFitMetricForImage
 * \brief A metric to compute the similarity between an image and breast fat model.
 *
 * Computes the similarity to a shape model:
 *
 * y = {x < 0: 0}, {0 < x < a: b/a sqrt(a^2 - x^2)}, {x > a: b}
 *
 * \section itkMammogramFatEstimationFitMetricForImageCaveats Caveats
 * \li None
 */

template <class TInputImage>
class  ITK_EXPORT MammogramFatEstimationFitMetricForImage :
  public MammogramFatEstimationFitMetric
{
//  Software Guide : EndCodeSnippet
public:
  typedef MammogramFatEstimationFitMetricForImage  Self;
  typedef MammogramFatEstimationFitMetric   Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MammogramFatEstimationFitMetricForImage, MammogramFatEstimationFitMetric );

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  itkStaticConstMacro( ParametricSpaceDimension, unsigned int, 7 );

  virtual unsigned int GetNumberOfParameters(void) const override  
  {
    return ParametricSpaceDimension;
  }

  /** Type of the input image */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef typename InputImageType::SpacingType  InputImageSpacingType;
  typedef typename InputImageType::PointType    InputImagePointType;
  typedef typename InputImageType::IndexType    InputImageIndexType;
  typedef typename InputImageType::SizeType     InputImageSizeType;

  typedef typename itk::ImageRegionIterator< TInputImage >               IteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< TInputImage >      IteratorWithIndexType;
  typedef typename itk::ImageRegionConstIterator< TInputImage >          IteratorConstType;
  typedef typename itk::ImageRegionConstIteratorWithIndex< TInputImage > IteratorWithIndexConstType;


  /** Type of the distance image */
  typedef float                                                  DistancePixelType;
  typedef typename itk::Image<DistancePixelType, ImageDimension> DistanceImageType;
  typedef typename DistanceImageType::Pointer                    DistanceImagePointer;
  typedef typename DistanceImageType::ConstPointer               DistanceImageConstPointer;
  typedef typename DistanceImageType::RegionType                 DistanceImageRegionType;
  typedef typename DistanceImageType::PixelType                  DistanceImagePixelType;
  typedef typename DistanceImageType::SpacingType                DistanceImageSpacingType;
  typedef typename DistanceImageType::PointType                  DistanceImagePointType;
  typedef typename DistanceImageType::IndexType                  DistanceImageIndexType;
  typedef typename DistanceImageType::SizeType                   DistanceImageSizeType;

  typedef typename itk::ImageRegionIterator< DistanceImageType >          DistanceIteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< DistanceImageType > DistanceIteratorWithIndexType;

  /** Optional mask image */
  typedef unsigned char                                      MaskPixelType;
  typedef typename itk::Image<MaskPixelType, ImageDimension> MaskImageType;
  typedef typename MaskImageType::ConstPointer               MaskImageConstPointer;
  typedef typename MaskImageType::Pointer                    MaskImagePointer;
  typedef typename MaskImageType::RegionType                 MaskImageRegionType;
  typedef typename MaskImageType::SizeType                   MaskImageSizeType;
  typedef typename MaskImageType::IndexType                  MaskImageIndexType;
  typedef typename MaskImageType::SpacingType                MaskImageSpacingType;


  typedef typename itk::ImageRegionConstIterator< MaskImageType > MaskIteratorType;
  typedef typename itk::ImageLinearConstIteratorWithIndex< MaskImageType > MaskLineIteratorType;


  /** Connect the input image. */
  void SetInputImage( const InputImageType *imInput );

  /// Set the optional mask image
  void SetMask( const MaskImageType *imMask );

  /** Get the fat image. */
  itkGetObjectMacro( Fat, InputImageType );

  /** Get the maximum distance to the breast edge in mm. */
  DistancePixelType GetMaxDistance( void ) { return m_MaxDistance; }

  /** Get the distance image. */
  itkGetObjectMacro( Distance, DistanceImageType );

  MeasureType GetValue( const ParametersType &parameters ) const override;

  void ClearFatImage( void );

  void GenerateFatImage( const ParametersType &parameters );

  void WriteIntensityVsEdgeDistToFile( std::string fileOutputIntensityVsEdgeDist ) override;
  void WriteFitToFile( std::string fileOutputFit,
                       const ParametersType &parameters ) override;


protected:

  MammogramFatEstimationFitMetricForImage();
  virtual ~MammogramFatEstimationFitMetricForImage();
  MammogramFatEstimationFitMetricForImage(const Self &) {}
  void operator=(const Self &) {}
  void PrintSelf(std::ostream & os, Indent indent) const override;

  InputImageRegionType   m_ImRegion;
  InputImageSpacingType  m_ImSpacing;
  InputImagePointType    m_ImOrigin;
  InputImageSizeType     m_ImSize;
  InputImagePointType    m_ImSizeInMM;
  InputImageConstPointer m_InputImage;

  InputImagePointer      m_Fat;

  MaskImageRegionType    m_MaskRegion;
  MaskImageConstPointer  m_Mask;

  DistancePixelType      m_MaxDistance;
  DistanceImagePointer   m_Distance;

  double CalculateFit( double d, const ParametersType &parameters, DistanceImageIndexType index );

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramFatEstimationFitMetricForImage.txx"
#endif

#endif
