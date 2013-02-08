/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkInterpolateVectorFieldFilter_h
#define __itkInterpolateVectorFieldFilter_h

#include "itkImage.h"
#include "itkVector.h"
#include "itkImageToImageFilter.h"
#include "itkVectorInterpolateImageFunction.h"


namespace itk {
/** 
 * \class InterpolateVectorFieldFilter
 * \brief This class takes a vector field as input 1, and a vector field as
 * input 2, and the output is a vector field, of the same dimensions as input 2,
 * where the vector at each location is interpolated from input 1.
 * 
 * Defaults to Linear Interpolation, but you can inject an interpolator.
 * 
 * \sa VectorResampleImageFilter
 */
template <
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions = 3>        // Number of Dimensions i.e. 2D or 3D
class ITK_EXPORT InterpolateVectorFieldFilter :
  public ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input image
                             Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                           >
{
public:

  /** Standard "Self" typedef. */
  typedef InterpolateVectorFieldFilter                                                  Self;
  typedef ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>,
                              Image< Vector<TScalarType, NDimensions>,  NDimensions>
                            >                                                           Superclass;
  typedef SmartPointer<Self>                                                            Pointer;
  typedef SmartPointer<const Self>                                                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(InterpolateVectorFieldFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Standard typedefs. */
  typedef Vector<TScalarType, NDimensions>                                            OutputPixelType;
  typedef typename OutputPixelType::ValueType                                         OutputPixelComponentType;
  typedef Image< OutputPixelType, NDimensions >                                       OutputImageType;
  typedef typename OutputImageType::Pointer                                           OutputImagePointer;
  typedef typename OutputImageType::ConstPointer                                      OutputImageConstPointer;
  typedef typename OutputImageType::RegionType                                        OutputImageRegionType;
  typedef typename OutputImageType::SpacingType                                       OutputImageSpacingType;
  typedef typename OutputImageType::PointType                                         OutputImageOriginType;
  typedef typename OutputImageRegionType::SizeType                                    OutputImageSizeType;
  typedef typename OutputImageRegionType::IndexType                                   OutputImageIndexType;
  typedef typename Superclass::InputImageType                                         InputImageType;
  typedef typename InputImageType::Pointer                                            InputImagePointer;
  typedef typename InputImageType::ConstPointer                                       InputImageConstPointer;
  typedef typename InputImageType::RegionType                                         InputImageRegionType;
  typedef VectorInterpolateImageFunction<
                                         Image< 
                                               Vector<TScalarType, NDimensions>,  
                                               NDimensions
                                              >, 
                                         TScalarType>                                 InterpolatorType;
  typedef typename InterpolatorType::Pointer                                          InterpolatorPointer;
  typedef typename InterpolatorType::PointType                                        PointType;
  
  /** Set the interpolated field at position 0. */
  virtual void SetInterpolatedField(const InputImageType *image) { this->SetNthInput(0, image); }

  /** Set the interpolating field at position 1 */
  virtual void SetInterpolatingField(const InputImageType *image) { this->SetNthInput(1, image); }

  /** We set the input images by number. */
  virtual void SetNthInput(unsigned int idx, const InputImageType *);

  /** Set the interpolator. */
  itkSetObjectMacro( Interpolator, InterpolatorType );

  /** Get a pointer to the interpolator function. */
  itkGetConstObjectMacro( Interpolator, InterpolatorType );

  /** Set the pixel value when a transformed pixel is outside of the image.  The default default pixel value is 0. */
  itkSetMacro(DefaultPixelValue, OutputPixelType);

  /** Get the pixel value when a transformed pixel is outside of the image */
  itkGetMacro(DefaultPixelValue, OutputPixelType);

  /** Method Compute the Modified Time based on changed to the components. */
  unsigned long GetMTime( void ) const;

protected:
  InterpolateVectorFieldFilter();
  ~InterpolateVectorFieldFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** This method is used to set the state of the filter before multi-threading. i.e. Connect interpolator. */
  virtual void BeforeThreadedGenerateData();

  /** This method is used to set the state of the filter after multi-threading. i.e. Disconnect interpolator. */
  virtual void AfterThreadedGenerateData();

  /** Force the filter to request LargestPossibleRegion on all inputs, which may be a different size. */
  virtual void GenerateInputRequestedRegion();

  /** Force filter to create the output buffer at LargestPossibleRegion, same size as 2nd input.  */
  virtual void GenerateOutputInformation();

  /** Yes, this one's multi-threaded. */
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId );

private:
  
  /**
   * Prohibited copy and assignment. 
   */
  InterpolateVectorFieldFilter(const Self&); 
  void operator=(const Self&); 

  /** The interpolator. */
  InterpolatorPointer m_Interpolator;
  
  /** Default pixel value. */
  OutputPixelType m_DefaultPixelValue;
  
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkInterpolateVectorFieldFilter.txx"
#endif

#endif
