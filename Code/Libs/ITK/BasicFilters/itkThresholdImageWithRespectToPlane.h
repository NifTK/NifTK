/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkThresholdImageWithRespectToPlane_h
#define __itkThresholdImageWithRespectToPlane_h

#include "itkImageToImageFilter.h"

namespace itk {
  
/** \class ThresholdImageWithRespectToPlane
 * \brief Image filter class to set all voxels on one side of plane to
 * a uer specified value (or zero by default).
 *
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT ThresholdImageWithRespectToPlane:
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef ThresholdImageWithRespectToPlane               Self;
  typedef ImageToImageFilter< TInputImage,TOutputImage > Superclass;
  typedef SmartPointer< Self >                           Pointer;
  typedef SmartPointer< const Self >                     ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( ThresholdImageWithRespectToPlane, ImageToImageFilter );

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
  typedef typename InputImageType::IndexType    InputImageIndexType;
  typedef typename InputImageType::PointType    InputImagePointType;

  /** Type of the output image */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;

  // Set the value to set voxels on the opposite side of the plane to the normal to
  itkSetMacro( ThresholdValue, OutputImagePixelType );
  // Get the value to set voxels on the opposite side of the plane to the normal to
  itkGetMacro( ThresholdValue, OutputImagePixelType );

  /** Set the position of the plane in mm. */
  void SetPlanePosition( double px, double py, double pz );
  /** Set the normal of the plane. */
  void SetPlaneNormal( double nx, double ny, double nz );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /** End concept checking */
#endif

protected:
  ThresholdImageWithRespectToPlane();
  virtual ~ThresholdImageWithRespectToPlane() {};
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
  
  /** ForwardImageProjector3Dto2D can be implemented as a multithreaded filter.
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
                            int threadId );

  // The value to set voxels on the opposite side of the plane to the normal to
  OutputImagePixelType m_ThresholdValue;

  /// The position of the plane
  Vector< double, 3 > m_PlanePosition;
  /// The normal of the plane
  Vector< double, 3 > m_PlaneNormal;

private:
  ThresholdImageWithRespectToPlane(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkThresholdImageWithRespectToPlane.txx"
#endif

#endif
