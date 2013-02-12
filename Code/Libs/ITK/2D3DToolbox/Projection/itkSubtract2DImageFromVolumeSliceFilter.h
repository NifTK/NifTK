/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSubtract2DImageFromVolumeSliceFilter_h
#define __itkSubtract2DImageFromVolumeSliceFilter_h

#include "itkInPlaceImageFilter.h"

namespace itk
{
  
/** \class Subtract2DImageFromVolumeSliceFilter
 * \brief Implements an operator for pixel-wise subtraction of a slice
 * of a 3D volume from a 2D image.
 *
 * Output(i=0..Nx-1, j=0..Ny-1) = Image2D(i=0..Nx-1, j=0..Ny-1) - Volume3D(i=0..Nx-1, j=0..Ny-1, k).
 */

template <class IntensityType = float>
class ITK_EXPORT Subtract2DImageFromVolumeSliceFilter : 
    public ImageToImageFilter< Image<IntensityType, 2>, Image<IntensityType, 2> > 
{
public:

  /** Standard class typedefs. */
  typedef Subtract2DImageFromVolumeSliceFilter          Self;
  typedef ImageToImageFilter< Image< IntensityType, 2>, Image<IntensityType, 2> > Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(Subtract2DImageFromVolumeSliceFilter, ImageToImageFilter);

  /** Some convenient typedefs. */
  typedef Image<IntensityType, 2>                  InputImageType;
  typedef typename    InputImageType::Pointer      InputImagePointer;
  typedef typename    InputImageType::RegionType   InputImageRegionType;
  typedef typename    InputImageType::PixelType    InputImagePixelType;
  typedef typename    InputImageType::SizeType     InputImageSizeType;
  typedef typename    InputImageType::SpacingType  InputImageSpacingType;
  typedef typename    InputImageType::PointType    InputImagePointType;
  typedef typename    InputImageType::IndexType    InputImageIndexType;

  typedef Image<IntensityType, 3>                             InputProjectionVolumeType;
  typedef typename    InputProjectionVolumeType::Pointer      InputProjectionVolumePointer;
  typedef typename    InputProjectionVolumeType::RegionType   InputProjectionVolumeRegionType;
  typedef typename    InputProjectionVolumeType::PixelType    InputProjectionVolumePixelType;
  typedef typename    InputProjectionVolumeType::SizeType     InputProjectionVolumeSizeType;
  typedef typename    InputProjectionVolumeType::SpacingType  InputProjectionVolumeSpacingType;
  typedef typename    InputProjectionVolumeType::PointType    InputProjectionVolumePointType;
  typedef typename    InputProjectionVolumeType::IndexType    InputProjectionVolumeIndexType;

  typedef Image<IntensityType, 2>                             OutputSubtractedImageType;
  typedef typename     OutputSubtractedImageType::Pointer     OutputSubtractedImagePointer;
  typedef typename     OutputSubtractedImageType::RegionType  OutputSubtractedImageRegionType;
  typedef typename     OutputSubtractedImageType::PixelType   OutputSubtractedImagePixelType;
  typedef typename     OutputSubtractedImageType::SizeType    OutputSubtractedImageSizeType;
  typedef typename     OutputSubtractedImageType::SpacingType OutputSubtractedImageSpacingType;
  typedef typename     OutputSubtractedImageType::PointType   OutputSubtractedImagePointType;

  /// Set the input 2D image
  void SetInputImage2D( const InputImageType *im2D);

  /// Set the input 3D volume of slices to subtract
  void SetInputVolume3D( const InputProjectionVolumeType * im3D);

  /** ImageDimension enumeration */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      InputImageType::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      OutputSubtractedImageType::ImageDimension);

  /// Set the current subtraction slice number
  itkSetMacro( SliceNumber, unsigned int);


protected:

  Subtract2DImageFromVolumeSliceFilter();
  virtual ~Subtract2DImageFromVolumeSliceFilter() {};

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

  /// We overload GenerateData() purely as a means of outputing debug info 
  void GenerateData(void);

  /** Subtract2DImageFromVolumeSliceFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData() routine
   * which is called for each processing thread. The output image data is
   * allocated automatically by the superclass prior to calling
   * ThreadedGenerateData().  ThreadedGenerateData can only write to the
   * portion of the output image specified by the parameter
   * "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData()  */
  void ThreadedGenerateData(const OutputSubtractedImageRegionType &outputRegionForThread,
                            int threadId );

  /// The slice number of the 3D volume to be subtracted from the 2D image
  unsigned int m_SliceNumber;

private:
  Subtract2DImageFromVolumeSliceFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSubtract2DImageFromVolumeSliceFilter.txx"
#endif

#endif
