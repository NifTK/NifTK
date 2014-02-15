/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramPectoralisSegmentationImageFilter_h
#define __itkMammogramPectoralisSegmentationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkMammogramLeftOrRightSideCalculator.h>
#include <itkMammogramPectoralisFitMetric.h>

namespace itk {
  
/** \class MammogramPectoralisSegmentationImageFilter
 * \brief 2D image filter class to segment the pectoral muscle from a mammogram.
 *
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT MammogramPectoralisSegmentationImageFilter:
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef MammogramPectoralisSegmentationImageFilter     Self;
  typedef ImageToImageFilter< TInputImage,TOutputImage > Superclass;
  typedef SmartPointer< Self >                           Pointer;
  typedef SmartPointer< const Self >                     ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MammogramPectoralisSegmentationImageFilter, ImageToImageFilter );

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

  /** Type of the output image */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;

  /** Type of the template image */
  typedef typename itk::MammogramPectoralisFitMetric<InputImageType>::TemplateImageType TemplateImageType;

  typedef typename TemplateImageType::Pointer      TemplateImagePointer;
  typedef typename TemplateImageType::ConstPointer TemplateImageConstPointer;
  typedef typename TemplateImageType::RegionType   TemplateImageRegionType;
  typedef typename TemplateImageType::PixelType    TemplateImagePixelType;
  typedef typename TemplateImageType::SpacingType  TemplateImageSpacingType;
  typedef typename TemplateImageType::PointType    TemplateImagePointType;
  typedef typename TemplateImageType::IndexType    TemplateImageIndexType;
  typedef typename TemplateImageType::SizeType     TemplateImageSizeType;

  typedef typename itk::ImageRegionIterator< TemplateImageType >          TemplateIteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< TemplateImageType > TemplateIteratorWithIndexType;

  /** Optional mask image */
  typedef unsigned char                                      MaskPixelType;
  typedef typename itk::Image<MaskPixelType, ImageDimension> MaskImageType;
  typedef typename MaskImageType::ConstPointer               MaskImageConstPointer;
  typedef typename MaskImageType::RegionType                 MaskImageRegionType;
  typedef typename MaskImageType::Pointer                    MaskImagePointer;
  typedef typename MaskImageType::SizeType                   MaskImageSizeType;
  typedef typename MaskImageType::SpacingType                MaskImageSpacingType;
  typedef typename MaskImageType::PointType                  MaskImagePointType;
  typedef typename MaskImageType::IndexType                  MaskImageIndexType;

  /// Set the optional mask image
  void SetMask( const MaskImageType *imMask );

  typedef typename itk::ImageRegionIterator< TInputImage > IteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< TInputImage > IteratorWithIndexType;
  typedef typename itk::ImageLinearIteratorWithIndex< MaskImageType > MaskLineIteratorType;
 
  typedef typename itk::ImageRegionConstIterator< TInputImage > IteratorConstType;
  typedef typename itk::ImageRegionConstIteratorWithIndex< TInputImage > IteratorWithIndexConstType;

  typedef typename itk::MammogramLeftOrRightSideCalculator< InputImageType > LeftOrRightSideCalculatorType;

  typedef typename LeftOrRightSideCalculatorType::BreastSideType BreastSideType;

  typedef typename itk::MammogramPectoralisFitMetric< TInputImage > FitMetricType;


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(DimensionShouldBe2,
		  (Concept::SameDimension<itkGetStaticConstMacro(InputImageDimension),2>));
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /** End concept checking */
#endif

  bool GetVerbose( void ) { return m_flgVerbose; }
  void SetVerbose( bool flag ) { m_flgVerbose = flag; }

  void SetVerboseOn( void ) { m_flgVerbose = true; }
  void SetVerboseOff( void ) { m_flgVerbose = false; }


protected:

  MammogramPectoralisSegmentationImageFilter();
  virtual ~MammogramPectoralisSegmentationImageFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;

  bool m_flgVerbose;
  BreastSideType m_BreastSide;

  InputImagePointer m_Image;
  MaskImagePointer m_Mask;

  template<typename ShrinkImageType>
    typename ShrinkImageType::Pointer 
    ShrinkTheInputImage( typename ShrinkImageType::ConstPointer &image,
                         unsigned int maxShrunkDimension,
                         typename ShrinkImageType::SizeType &outSize );
  
  /** Single threaded execution */
  void GenerateData();

  void GenerateTemplate( typename TInputImage::Pointer &imTemplate,
                         typename TInputImage::RegionType region,
                         double &tMean, double &tStdDev, double &nPixels );

  // Override since the filter produces the entire dataset
  void EnlargeOutputRequestedRegion(DataObject *output);

  // Run and exhaustive search over a region of interest
  void ExhaustiveSearch( InputImageIndexType pecInterceptStart, 
                         InputImageIndexType pecInterceptEnd, 
                         typename FitMetricType::Pointer &metric,
                         InputImagePointer &imPipelineConnector,
                         InputImagePointType &bestPecInterceptInMM,
                         typename FitMetricType::ParametersType &bestParameters );


private:

  MammogramPectoralisSegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramPectoralisSegmentationImageFilter.txx"
#endif

#endif
