/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatSubtractionImageFilter_h
#define __itkMammogramFatSubtractionImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkMammogramFatEstimationFitMetric.h>

namespace itk {
  
/** \class MammogramFatSubtractionImageFilter
 * \brief 2D image filter class to subtract the fat signal from a mammogram.
 *
 */

template<class TInputImage>
class ITK_EXPORT MammogramFatSubtractionImageFilter:
    public ImageToImageFilter< TInputImage, TInputImage >
{
public:
  /** Standard class typedefs. */
  typedef MammogramFatSubtractionImageFilter            Self;
  typedef ImageToImageFilter< TInputImage,TInputImage > Superclass;
  typedef SmartPointer< Self >                          Pointer;
  typedef SmartPointer< const Self >                    ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MammogramFatSubtractionImageFilter, ImageToImageFilter );

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

  /** Mask image */
  typedef unsigned char                                      MaskPixelType;
  typedef typename itk::Image<MaskPixelType, ImageDimension> MaskImageType;
  typedef typename MaskImageType::ConstPointer               MaskImageConstPointer;
  typedef typename MaskImageType::RegionType                 MaskImageRegionType;
  typedef typename MaskImageType::Pointer                    MaskImagePointer;
  typedef typename MaskImageType::SizeType                   MaskImageSizeType;
  typedef typename MaskImageType::SpacingType                MaskImageSpacingType;
  typedef typename MaskImageType::PointType                  MaskImagePointType;
  typedef typename MaskImageType::IndexType                  MaskImageIndexType;

  /// Set the mask image
  void SetMask( const MaskImageType *imMask );

  typedef typename itk::ImageRegionIterator< TInputImage > IteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< TInputImage > IteratorWithIndexType;
  typedef typename itk::ImageLinearIteratorWithIndex< MaskImageType > MaskLineIteratorType;
 
  typedef typename itk::ImageRegionConstIterator< TInputImage > IteratorConstType;
  typedef typename itk::ImageRegionConstIteratorWithIndex< TInputImage > IteratorWithIndexConstType;

  typedef typename itk::MammogramFatEstimationFitMetric< TInputImage > FitMetricType;


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(DimensionShouldBe2,
		  (Concept::SameDimension<itkGetStaticConstMacro(InputImageDimension),2>));
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  /** End concept checking */
#endif

  bool GetVerbose( void ) { return m_flgVerbose; }
  void SetVerbose( bool flag ) { m_flgVerbose = flag; }
  void SetVerboseOn( void ) { m_flgVerbose = true; }
  void SetVerboseOff( void ) { m_flgVerbose = false; }

  bool GetComputeFatEstimationFit( void ) { return m_flgComputeFatEstimationFit; }
  void SetComputeFatEstimationFit( bool flag ) { m_flgComputeFatEstimationFit = flag; }
  void SetComputeFatEstimationFitOn( void ) { m_flgComputeFatEstimationFit = true; }
  void SetComputeFatEstimationFitOff( void ) { m_flgComputeFatEstimationFit = false; }

  void SetFileOutputIntensityVsEdgeDist( std::string fn ) { m_fileOutputIntensityVsEdgeDist = fn; }
  void SetFileOutputFit( std::string fn ) { m_fileOutputFit = fn; }


protected:

  MammogramFatSubtractionImageFilter();
  virtual ~MammogramFatSubtractionImageFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;

  bool m_flgVerbose;
  bool m_flgComputeFatEstimationFit;

  std::string m_fileOutputIntensityVsEdgeDist;  
  std::string m_fileOutputFit;  

  InputImagePointer m_Image;
  MaskImagePointer m_Mask;

  /** Make a DataObject of the correct type to be used as the specified
   * output. */
  virtual DataObject::Pointer MakeOutput(unsigned int idx);

  template<typename ShrinkImageType>
  void ComputeShrinkFactors( typename ShrinkImageType::ConstPointer &image,
                             unsigned int maxShrunkDimension, 
                             itk::Array< double > &sampling,
                             typename ShrinkImageType::SpacingType &outSpacing,
                             typename ShrinkImageType::SizeType &outSize );

  template<typename ShrinkImageType>
    typename ShrinkImageType::Pointer 
    ShrinkTheInputImage( typename ShrinkImageType::ConstPointer &image,
                         unsigned int maxShrunkDimension,
                         typename ShrinkImageType::SizeType &outSize );
  
  template<typename ShrinkImageType>
    typename ShrinkImageType::Pointer 
    ShrinkTheInputImageViaMinResample( typename ShrinkImageType::ConstPointer &image,
                                       unsigned int maxShrunkDimension,
                                       typename ShrinkImageType::SizeType &outSize );
  
  /// Single threaded execution
  void GenerateData();

  /// Compute the fat subtraction via a fat estimation curve fit
  void ComputeFatEstimationFit();

  /// Compute the fat image via the minimum intensity at equal distances from the breast edge
  void ComputeMinIntensityVersusDistanceFromEdge();

  // Override since the filter produces the entire dataset
  void EnlargeOutputRequestedRegion(DataObject *output);

private:

  MammogramFatSubtractionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramFatSubtractionImageFilter.txx"
#endif

#endif
