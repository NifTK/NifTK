/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMammogramMaskSegmentationImageFilter_h
#define itkMammogramMaskSegmentationImageFilter_h

#include <itkImageToImageFilter.h>

#include <vnl/vnl_double_2.h>

namespace itk {
  
/** \class MammogramMaskSegmentationImageFilter
 * \brief 2D image filter class to segment the breast area from a mammogram.
 *
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT MammogramMaskSegmentationImageFilter:
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef MammogramMaskSegmentationImageFilter           Self;
  typedef ImageToImageFilter< TInputImage,TOutputImage > Superclass;
  typedef SmartPointer< Self >                           Pointer;
  typedef SmartPointer< const Self >                     ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MammogramMaskSegmentationImageFilter, ImageToImageFilter );

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
  itkConceptMacro(InputHasFloatingPointPixelType,
                  (Concept::IsFloatingPoint<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /** End concept checking */
#endif

  /// For debugging purposes, set single threaded execution
  void SetSingleThreadedExecution(void) {m_FlagMultiThreadedExecution = false;}

protected:
  MammogramMaskSegmentationImageFilter();
  virtual ~MammogramMaskSegmentationImageFilter();
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  /** If an imaging filter needs to perform processing after the buffer
   * has been allocated but before threads are spawned, the filter can
   * can provide an implementation for BeforeThreadedGenerateData(). The
   * execution flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void BeforeThreadedGenerateData(void);
  
  /** If an imaging filter needs to perform processing after all
   * processing threads have completed, the filter can can provide an
   * implementation for AfterThreadedGenerateData(). The execution
   * flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void AfterThreadedGenerateData(void);
  
  /** Single threaded execution, for debugging purposes ( call
  SetSingleThreadedExecution() ) */
  void GenerateData();
  
  /** MammogramMaskSegmentationImageFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData()
   * routine which is called for each processing thread. The output
   * image data is allocated automatically by the superclass prior to
   * calling ThreadedGenerateData().  ThreadedGenerateData can only
   * write to the portion of the output image specified by the
   * parameter "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData() */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            ThreadIdType threadId );

  // Override since the filter produces the entire dataset
  void EnlargeOutputRequestedRegion(DataObject *output);

  /// Flag to turn multithreading on or off
  bool m_FlagMultiThreadedExecution;


private:
  MammogramMaskSegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramMaskSegmentationImageFilter.txx"
#endif

#endif
