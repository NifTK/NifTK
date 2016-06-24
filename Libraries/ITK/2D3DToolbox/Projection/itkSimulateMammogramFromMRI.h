/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSimulateMammogramFromMRI_h
#define itkSimulateMammogramFromMRI_h

#include "itkForwardImageProjector3Dto2D.h"

namespace itk
{
  
/** \class SimulateMammogramFromMRI
 * \brief Class to project a 3D image into 2D.
 */

template <class IntensityType = float>
class ITK_EXPORT SimulateMammogramFromMRI : 
    public ForwardImageProjector3Dto2D< IntensityType >
{
public:
  /** Standard class typedefs. */
  typedef SimulateMammogramFromMRI                       Self;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;
  typedef ForwardImageProjector3Dto2D< IntensityType >   Superclass;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimulateMammogramFromMRI, ForwardImageProjector3Dto2D);

  /** Some convenient typedefs. */
  typedef typename Superclass::InputImageType          InputImageType;
  typedef typename Superclass::InputImagePointer       InputImagePointer;
  typedef typename Superclass::InputImageConstPointer  InputImageConstPointer;
  typedef typename Superclass::InputImageRegionType    InputImageRegionType;
  typedef typename Superclass::InputImagePixelType     InputImagePixelType;
			                               
  typedef typename Superclass::OutputImageType         OutputImageType;
  typedef typename Superclass::OutputImagePointer      OutputImagePointer;
  typedef typename Superclass::OutputImageRegionType   OutputImageRegionType;
  typedef typename Superclass::OutputImageSizeType     OutputImageSizeType;
  typedef typename Superclass::OutputImageSpacingType  OutputImageSpacingType;
  typedef typename Superclass::OutputImagePointType    OutputImagePointType;
  typedef typename Superclass::OutputImagePixelType    OutputImagePixelType;
  typedef typename Superclass::OutputImageIndexType    OutputImageIndexType;   

protected:
  SimulateMammogramFromMRI();
  virtual ~SimulateMammogramFromMRI(void) {};
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

  /** SimulateMammogramFromMRI can be implemented as a multithreaded filter.
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

private:
  SimulateMammogramFromMRI(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSimulateMammogramFromMRI.txx"
#endif

#endif
