/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramPectoralisFitMetric_h
#define __itkMammogramPectoralisFitMetric_h

#include <itkSingleValuedCostFunction.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

namespace itk {
  
/** \class MammogramPectoralisFitMetric
 * \brief A metric to compute the similarity between an image and a pectoral shape model.
 *
 * \section itkMammogramPectoralisFitMetricCaveats Caveats
 * \li The mammogram is assumed to be have the chest wall of the left-hand
 * edge so right mammograms should be flipped first.
 */

template <class TInputImage>
class  ITK_EXPORT MammogramPectoralisFitMetric :
  public SingleValuedCostFunction
{
//  Software Guide : EndCodeSnippet
public:
  typedef MammogramPectoralisFitMetric  Self;
  typedef SingleValuedCostFunction   Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MammogramPectoralisFitMetric, SingleValuedCostFunction );

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

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


  /** Type of the template image */
  typedef float                                                  TemplatePixelType;
  typedef typename itk::Image<TemplatePixelType, ImageDimension> TemplateImageType;
  typedef typename TemplateImageType::Pointer                    TemplateImagePointer;
  typedef typename TemplateImageType::ConstPointer               TemplateImageConstPointer;
  typedef typename TemplateImageType::RegionType                 TemplateImageRegionType;
  typedef typename TemplateImageType::PixelType                  TemplateImagePixelType;
  typedef typename TemplateImageType::SpacingType                TemplateImageSpacingType;
  typedef typename TemplateImageType::PointType                  TemplateImagePointType;
  typedef typename TemplateImageType::IndexType                  TemplateImageIndexType;
  typedef typename TemplateImageType::SizeType                   TemplateImageSizeType;

  typedef typename itk::ImageRegionIterator< TemplateImageType >          TemplateIteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< TemplateImageType > TemplateIteratorWithIndexType;

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

  /** Get the template image. */
  itkGetObjectMacro( ImTemplate, TemplateImageType );

  typedef Superclass::ParametersType ParametersType;
  typedef Superclass::DerivativeType DerivativeType;
  typedef Superclass::MeasureType    MeasureType;

  itkStaticConstMacro( ParametricSpaceDimension, unsigned int, 8 );

  unsigned int GetNumberOfParameters(void) const  
  {
    return ParametricSpaceDimension;
  }

  void GetParameters( const InputImagePointType &pecInterceptInMM,
                      ParametersType &parameters );

  void GetDerivative( const ParametersType &parameters, 
                      DerivativeType &Derivative ) const
  {
    return;
  }

  MeasureType GetValue( const InputImagePointType &pecInterceptInMM );

  MeasureType GetValue( const ParametersType &parameters ) const;

  void GetValueAndDerivative( const ParametersType &parameters,
                              MeasureType &Value, 
                              DerivativeType &Derivative ) const
  {
    Value = this->GetValue( parameters );
    this->GetDerivative( parameters, Derivative );
  }

  void ClearTemplate( void );

  void GenerateTemplate( const ParametersType &parameters,
                         double &tMean, double &tStdDev, double &nInside, double &nPixels,
                         TemplateImageRegionType &templateRegion );

protected:

  MammogramPectoralisFitMetric();
  virtual ~MammogramPectoralisFitMetric();
  MammogramPectoralisFitMetric(const Self &) {}
  void operator=(const Self &) {}
  void PrintSelf(std::ostream & os, Indent indent) const;

  InputImageRegionType   m_ImRegion;
  InputImageSpacingType  m_ImSpacing;
  InputImagePointType    m_ImOrigin;
  InputImageSizeType     m_ImSize;
  InputImagePointType    m_ImSizeInMM;
  InputImageConstPointer m_InputImage;

  TemplateImagePointer   m_ImTemplate;

  MaskImageRegionType    m_MaskRegion;
  MaskImageConstPointer  m_Mask;

  double GradientAtMidpoint( const ParametersType &parameters ) const;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramPectoralisFitMetric.txx"
#endif

#endif
