/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkForwardImageProjector3Dto2D_h
#define __itkForwardImageProjector3Dto2D_h

#include "itkRay.h"
#include "itkImageProjectionBaseClass2D3D.h"

namespace itk
{
  
/** \class ForwardImageProjector3Dto2D
 * \brief Class to project a 3D image into 2D.
 */

template <class IntensityType = float>
class ITK_EXPORT ForwardImageProjector3Dto2D : 
    public ImageProjectionBaseClass2D3D<Image< IntensityType, 3>,  // Input image
					Image< IntensityType, 2> > // Output image
{
public:
  /** Standard class typedefs. */
  typedef ForwardImageProjector3Dto2D                             Self;
  typedef SmartPointer<Self>                                      Pointer;
  typedef SmartPointer<const Self>                                ConstPointer;
  typedef ImageProjectionBaseClass2D3D<Image< IntensityType, 3>,
				       Image< IntensityType, 2> >   Superclass;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardImageProjector3Dto2D, ImageProjectionBaseClass2D3D);

  /** Some convenient typedefs. */
  typedef Image<IntensityType, 3>               InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;

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

  /** ForwardImageProjector3Dto2D produces a 2D ouput image which is a different
   * resolution and with a different pixel spacing than its 3D input
   * image (obviously).  As such, ForwardImageProjector3Dto2D needs to provide an
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
  ForwardImageProjector3Dto2D();
  virtual ~ForwardImageProjector3Dto2D(void) {};
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

private:
  ForwardImageProjector3Dto2D(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkForwardImageProjector3Dto2D.txx"
#endif

#endif
