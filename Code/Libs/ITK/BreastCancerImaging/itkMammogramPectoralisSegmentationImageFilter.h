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
#include <itkMammogramLeftOrRightSideCalculator.h>

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

  typedef typename NumericTraits<InputImagePixelType>::RealType    RealType;

  /** Type of the output image */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;

  typedef OutputImagePointType OriginType;


  /** Define the image type for internal computations 
      RealType is usually 'double' in NumericTraits. 
      Here we prefer float in order to save memory.  */

  typedef float InternalRealType;

  typedef Image< InternalRealType, TInputImage::ImageDimension > RealImageType;

  typedef typename RealImageType::Pointer RealImagePointer;


  typedef typename itk::ImageRegionIterator< TInputImage > IteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< TInputImage > IteratorWithIndexType;

  typedef typename itk::ImageRegionConstIterator< TInputImage > IteratorConstType;
  typedef typename itk::ImageRegionConstIteratorWithIndex< TInputImage > IteratorWithIndexConstType;

  typedef typename itk::MammogramLeftOrRightSideCalculator< InputImageType > LeftOrRightSideCalculatorType;

  typedef typename LeftOrRightSideCalculatorType::BreastSideType BreastSideType;


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

  bool m_flgVerbose;

  MammogramPectoralisSegmentationImageFilter();
  virtual ~MammogramPectoralisSegmentationImageFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  /** Single threaded execution */
  void GenerateData();

  void GenerateTemplate( typename TInputImage::Pointer &imTemplate,
                         typename TInputImage::RegionType region,
                         double &tMean, double &tStdDev, double &nPixels,
                         BreastSideType breastSide );

  // Override since the filter produces the entire dataset
  void EnlargeOutputRequestedRegion(DataObject *output);

private:

  MammogramPectoralisSegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramPectoralisSegmentationImageFilter.txx"
#endif

#endif
