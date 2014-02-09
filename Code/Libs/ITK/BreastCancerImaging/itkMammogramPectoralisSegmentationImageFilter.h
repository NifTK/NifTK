/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMammogramPectoralisSegmentationImageFilter_h
#define itkMammogramPectoralisSegmentationImageFilter_h

#include <itkImageToImageFilter.h>

#include <vnl/vnl_double_2.h>

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
  typedef MammogramPectoralisSegmentationImageFilter           Self;
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


protected:

  MammogramPectoralisSegmentationImageFilter();
  virtual ~MammogramPectoralisSegmentationImageFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  /** Single threaded execution */
  void GenerateData();

  // Override since the filter produces the entire dataset
  void EnlargeOutputRequestedRegion(DataObject *output);

  // Write an image to a file
  void WriteImageToFile( const char *fileOutput, const char *description,
                         typename TInputImage::Pointer image );

private:

  MammogramPectoralisSegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramPectoralisSegmentationImageFilter.txx"
#endif

#endif
