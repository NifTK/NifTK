/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkBackwardImageProjector2Dto3D_h
#define __itkBackwardImageProjector2Dto3D_h

#include "itkRay.h"
#include "itkImageProjectionBaseClass2D3D.h"

namespace itk
{
  
/** \class BackwardImageProjector2Dto3D
 * \brief Class to project a 3D image into 2D.
 */

template <class IntensityType = float>
class ITK_EXPORT BackwardImageProjector2Dto3D : 
  public ImageProjectionBaseClass2D3D<Image<IntensityType, 2>,  // Input image
				      Image<IntensityType, 3> > // Output image
{
public:
  /** Standard class typedefs. */
  typedef BackwardImageProjector2Dto3D                         Self;
  typedef SmartPointer<Self>                                   Pointer;
  typedef SmartPointer<const Self>                             ConstPointer;
  typedef ImageProjectionBaseClass2D3D<Image< IntensityType, 2>,
				       Image< IntensityType, 3> > Superclass;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BackwardImageProjector2Dto3D, ImageProjectionBaseClass2D3D);

  /** Some convenient typedefs. */
  typedef Image<IntensityType, 2>                InputImageType;
  typedef typename InputImageType::Pointer       InputImagePointer;
  typedef typename InputImageType::ConstPointer  InputImageConstPointer;
  typedef typename InputImageType::RegionType    InputImageRegionType;
  typedef typename InputImageType::SizeType      InputImageSizeType;
  typedef typename InputImageType::SpacingType   InputImageSpacingType;
  typedef typename InputImageType::PointType     InputImagePointType;
  typedef typename InputImageType::IndexType     InputImageIndexType;
  typedef typename InputImageType::PixelType     InputImagePixelType;

  typedef Image<IntensityType, 3>                OutputImageType;
  typedef typename OutputImageType::Pointer      OutputImagePointer;
  typedef typename OutputImageType::ConstPointer OutputImageConstPointer;
  typedef typename OutputImageType::RegionType   OutputImageRegionType;
  typedef typename OutputImageType::SizeType     OutputImageSizeType;
  typedef typename OutputImageType::SpacingType  OutputImageSpacingType;
  typedef typename OutputImageType::PixelType    OutputImagePixelType;
  typedef typename OutputImageType::PointType    OutputImagePointType;

  /** ImageDimension enumeration */
  itkStaticConstMacro(InputImageDimension, unsigned int, 2);
  itkStaticConstMacro(OutputImageDimension, unsigned int, 3);

  /** BackwardImageProjector2Dto3D produces a 3D ouput image which is a different
   * resolution and with a different pixel spacing than its 2D input
   * image (obviously).  As such, BackwardImageProjector3Dto2D needs to provide an
   * implementation for GenerateOutputInformation() in order to inform
   * the pipeline execution model. The original documentation of this
   * method is below.
   * \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation();

  /** Rather than calculate the input requested region for a
   * particular back-projection (which might take longer than the actual
   * projection), we simply set the input requested region to the
   * entire 2D input image region. Therefore needs to provide an implementation
   * for GenerateInputRequestedRegion() in order to inform the
   * pipeline execution model.  \sa
   * ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion();
  virtual void EnlargeOutputRequestedRegion(DataObject *output); 

  /// Set the size in voxels of the output back-projected image.
  void SetBackProjectedImageSize(OutputImageSizeType &outImageSize) {m_OutputImageSize = outImageSize;};
  /// Set the resolution in mm of the output back-projected image.
  void SetBackProjectedImageSpacing(OutputImageSpacingType &outImageSpacing) {
    m_OutputImageSpacing = outImageSpacing;
    this->GetOutput()->SetSpacing(m_OutputImageSpacing);
  };
  /// Set the origin of the output back-projected image.
  void SetBackProjectedImageOrigin(OutputImagePointType &outImageOrigin) {
    m_OutputImageOrigin = outImageOrigin;
    this->GetOutput()->SetOrigin(m_OutputImageOrigin);
  };

  /// For debugging purposes, set single threaded execution
  void SetSingleThreadedExecution(void) {m_FlagMultiThreadedExecution = false;}

  /// Set the backprojection volume to zero prior to the next back-projection
  void ClearVolumePriorToNextBackProjection(void) {m_ClearBackProjectedVolume = true;}


protected:

  BackwardImageProjector2Dto3D();
  virtual ~BackwardImageProjector2Dto3D() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /// Do we need to set the volume to zero?
  void ClearVolume(void);
  
  /** Initialise data prior to back projection (such as filling the
      back-projected volume with zeros). */
  virtual void BeforeThreadedGenerateData(void);

  /** Perform some operation immediately after the back projection. */
  virtual void AfterThreadedGenerateData(void);

  /** The version in ImageSource is overloaded here because we want to
      thread execution by splitting the input 2D image not the output 3D volume.*/
  void GenerateData();

  /** This function is modified to thread based on the input 2D image */
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
  static ITK_THREAD_RETURN_TYPE BackwardImageProjectorThreaderCallback( void *arg );

  /// The size of the output projected image
  OutputImageSizeType m_OutputImageSize;
  /// The resolution of the output projected image
  OutputImageSpacingType m_OutputImageSpacing;
  /// The origin of the output projected image
  OutputImagePointType m_OutputImageOrigin;

  /// Flag to turn multithreading on or off
  bool m_FlagMultiThreadedExecution;

  /// Flag that back-projected volume should be filled with zeros
  bool m_ClearBackProjectedVolume;

  /** Internal structure used for passing image data into the threading library */
  struct BackwardImageProjectorThreadStruct
  {
   Pointer Filter;
  };
  
  /** A volume equal in size to the back-projected volume which keeps
      a count of the number of rays contributing to each voxel in the back-projected volume.*/
  OutputImagePointer m_VoxelRayCount;


private:
  BackwardImageProjector2Dto3D(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBackwardImageProjector2Dto3D.txx"
#endif

#endif
