/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageProjector2D3D_h
#define __itkImageProjector2D3D_h

#include "itkImageProjectionBaseClass2D3D.h"

namespace itk
{
  
/** \class ImageProjector2D3D
 * \brief Class to project a 3D image into 2D.
 */

template <class IntensityType = float>
class ITK_EXPORT ImageProjector2D3D : 
    public ImageProjectionBaseClass2D3D<Image< IntensityType, 3>,  // Input image
					Image< IntensityType, 2> > // Output image
{
public:
  /** Standard class typedefs. */
  typedef ImageProjector2D3D                             Self;
  typedef SmartPointer<Self>                                      Pointer;
  typedef SmartPointer<const Self>                                ConstPointer;
  typedef ImageProjectionBaseClass2D3D<Image< IntensityType, 3>,
				       Image< IntensityType, 2> >   Superclass;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageProjector2D3D, ImageProjectionBaseClass2D3D);

  /** Some convenient typedefs. */
  typedef Image<IntensityType, 3>               InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::SizeType     InputImageSizeType;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef typename InputImageType::SpacingType  InputImageSpacingType;
  typedef typename InputImageType::PointType    InputImagePointType;
  typedef typename InputImageType::IndexType    InputImageIndexType;

  typedef Image<IntensityType, 2>               OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::SizeType    OutputImageSizeType;
  typedef typename OutputImageType::SpacingType OutputImageSpacingType;
  typedef typename OutputImageType::PointType   OutputImagePointType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;

  /** ImageDimension enumeration */
  itkStaticConstMacro(InputImageDimension, unsigned int, 3);
  itkStaticConstMacro(OutputImageDimension, unsigned int, 2);

  /** ImageProjector2D3D produces a 2D ouput image which is a different
   * resolution and with a different pixel spacing than its 3D input
   * image (obviously).  As such, ImageProjector2D3D needs to provide an
   * implementation for GenerateOutputInformation() in order to inform
   * the pipeline execution model. The original documentation of this
   * method is below.
   * \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation(void);

  /** Rather than calculate the input requested region for a
   * particular projection (which might take longer than the actual
   * projection), we simply set the input requested region to the
   * entire 3D input image region. Therefore needs to provide an implementation
   * for GenerateInputRequestedRegion() in order to inform the
   * pipeline execution model.  \sa
   * ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion(void);
  virtual void EnlargeOutputRequestedRegion(DataObject *output); 

  /// Set the size in pixels of the output projected image.
  void SetProjectedImageSize(OutputImageSizeType &outImageSize) {m_OutputImageSize = outImageSize;};
  /// Set the resolution in mm of the output projected image.
  void SetProjectedImageSpacing(OutputImageSpacingType &outImageSpacing) {
    m_OutputImageSpacing = outImageSpacing;
    this->GetOutput()->SetSpacing(m_OutputImageSpacing);
  };
  /// Set the origin of the output projected image.
  void SetProjectedImageOrigin(OutputImagePointType &outImageOrigin) {
    m_OutputImageOrigin = outImageOrigin;
    this->GetOutput()->SetOrigin(m_OutputImageOrigin);
  };

  /// Set the ray integration threshold
  void SetRayIntegrationThreshold(double threshold) {m_Threshold = threshold;}

  /// For debugging purposes, set single threaded execution
  void SetSingleThreadedExecution(void) {m_FlagMultiThreadedExecution = false;}

protected:
  ImageProjector2D3D();
  virtual ~ImageProjector2D3D(void) {};
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

  /** ImageProjector2D3D can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData()
   * routine which is called for each processing thread. The output
   * image data is allocated automatically by the superclass prior to
   * calling ThreadedGenerateData().  ThreadedGenerateData can only
   * write to the portion of the output image specified by the
   * parameter "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData() */
  void ThreadedGenerateData(const InputImageRegionType& inputRegionForThread,
                            int threadId );

  /** Split the output's RequestedRegion into "num" pieces, returning
   * region "i" as "splitRegion". This method is called "num" times. The
   * regions must not overlap. The method returns the number of pieces that
   * the routine is capable of splitting the input RequestedRegion,
   * i.e. return value is less than or equal to "num". */
  int SplitRequestedRegion(int i, int num, InputImageRegionType& splitRegion);

  /** Static function used as a "callback" by the MultiThreader.  The threading
   * library will call this routine for each thread, which will delegate the
   * control to ThreadedGenerateData(). */
  static ITK_THREAD_RETURN_TYPE ImageProjectorThreaderCallback( void *arg );

  /// The size of the output projected image
  OutputImageSizeType m_OutputImageSize;
  /// The resolution of the output projected image
  OutputImageSpacingType m_OutputImageSpacing;
  /// The origin of the output projected image
  OutputImagePointType m_OutputImageOrigin;

  /// The threshold above which voxels along the ray path are integrated.
  double m_Threshold;

  /// Flag to turn multithreading on or off
  bool m_FlagMultiThreadedExecution;

  /** Internal structure used for passing image data into the threading library */
  struct ImageProjectorThreadStruct
  {
   Pointer Filter;
  };

private:
  ImageProjector2D3D(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageProjector2D3D.txx"
#endif

#endif
