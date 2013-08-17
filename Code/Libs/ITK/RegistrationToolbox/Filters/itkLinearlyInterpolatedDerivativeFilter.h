/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkLinearlyInterpolatedDerivativeFilter_h
#define itkLinearlyInterpolatedDerivativeFilter_h
#include <itkImage.h>
#include <itkVector.h>
#include <itkImageToImageFilter.h>

namespace itk {
/** 
 * \class LinearlyInterpolatedDerivativeFilter
 * \brief This class takes as input 2 input images, the Fixed and Moving image,
 * as you would have in a registration pipeline, and outputs the derivative
 * of the moving image resampled by a transformation.
 *
 * The output is a vector image, of the same size as the input image, where 
 * each pixel is a vector with as many components as image dimensions.  
 * 
 * Effectively, this is a cross between itkResampleImageFilter which resamples
 * an image using an interpolator and outputs a scalar value and also a derivative
 * filter. Except that we don't inject an interpolator, as this is done internally.
 *
 */
template< class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
class ITK_EXPORT LinearlyInterpolatedDerivativeFilter :
  public ImageToImageFilter<TFixedImage, 
                            Image<  
                              Vector< TDeformationScalar, ::itk::GetImageDimension<TFixedImage>::ImageDimension>, 
                              ::itk::GetImageDimension<TFixedImage>::ImageDimension> >
{
public:

  /** Standard "Self" typedef. */
  typedef LinearlyInterpolatedDerivativeFilter                                          Self;
  typedef ImageToImageFilter<TFixedImage, 
                             Image<  
                               Vector< TDeformationScalar, ::itk::GetImageDimension<TFixedImage>::ImageDimension>, 
                               ::itk::GetImageDimension<TFixedImage>::ImageDimension> > Superclass;
  typedef SmartPointer<Self>                                                            Pointer;
  typedef SmartPointer<const Self>                                                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LinearlyInterpolatedDerivativeFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Standard typedefs. */
  typedef TDeformationScalar                                                          OutputDataType;
  typedef Vector< OutputDataType, itkGetStaticConstMacro(Dimension) >                 OutputPixelType;
  typedef Image< OutputPixelType, itkGetStaticConstMacro(Dimension) >                 OutputImageType;
  typedef typename OutputImageType::Pointer                                           OutputImagePointer;
  typedef typename OutputImageType::RegionType                                        OutputImageRegionType; 
  typedef typename OutputImageType::IndexType                                         OutputImageIndexType;
  typedef typename OutputImageType::SizeType                                          OutputImageSizeType;
  typedef typename OutputImageType::SpacingType                                       OutputImageSpacingType;
  typedef typename OutputImageType::PointType                                         OutputImageOriginType;
  typedef typename OutputImageType::DirectionType                                     OutputImageDirectionType;
  typedef TFixedImage                                                                 FixedImageType;
  typedef typename FixedImageType::Pointer                                            FixedImagePointer;
  typedef typename FixedImageType::PixelType                                          FixedImagePixelType;
  typedef typename FixedImageType::RegionType                                         FixedImageRegionType; 
  typedef typename FixedImageType::IndexType                                          FixedImageIndexType;
  typedef typename FixedImageType::SizeType                                           FixedImageSizeType;
  typedef typename FixedImageType::SpacingType                                        FixedImageSpacingType;
  typedef typename FixedImageType::PointType                                          FixedImageOriginType;
  typedef typename FixedImageType::DirectionType                                      FixedImageDirectionType;
  typedef TMovingImage                                                                MovingImageType;
  typedef typename MovingImageType::SizeType                                          MovingImageSizeType;
  typedef typename MovingImageType::Pointer                                           MovingImagePointer;
  typedef typename MovingImageType::PixelType                                         MovingImagePixelType;
  typedef typename MovingImageType::RegionType                                        MovingImageRegionType;
  typedef typename MovingImageType::IndexType                                         MovingImageIndexType;
  
  /** Transform typedef. */
  typedef Transform<TScalarType, itkGetStaticConstMacro(Dimension), 
                                 itkGetStaticConstMacro(Dimension)>                   TransformType;
  typedef typename TransformType::ConstPointer                                        TransformPointerType;

  /** Image size typedef. */
  typedef Size<itkGetStaticConstMacro(Dimension)> SizeType;

  /** Image index typedef. */
  typedef typename OutputImageType::IndexType IndexType;

  /** Image point type. */
  typedef typename MovingImageType::PointType MovingImagePointType;
  
  /** Set the fixed image at position 0. */
  virtual void SetFixedImage(const FixedImageType *image);

  /** Set the moving image at position 1. */
  virtual void SetMovingImage(const MovingImageType *image);

  /** 
   * Set the pixel value when a transformed pixel is outside of the
   * image.  The default default pixel value is a vector of zeros. 
   */
  itkSetMacro( DefaultPixelValue, OutputPixelType );

  /** Get the pixel value when a transformed pixel is outside of the image */
  itkGetConstReferenceMacro( DefaultPixelValue, OutputPixelType );

  /** Mainly for debugging, write image to file. */
  void WriteDerivativeImage(std::string filename);
  
  /** Connect the Transform. */
  itkSetObjectMacro( Transform, TransformType );

  /** Get a pointer to the Transform.  */
  itkGetConstObjectMacro( Transform, TransformType );
  
  /** Set a lower limit on voxels we consider for gradient calcs. */
  itkSetMacro(MovingImageLowerPixelValue, MovingImagePixelType);
  itkGetMacro(MovingImageLowerPixelValue, MovingImagePixelType);

  /** Set an upper limit on voxels we consider for gradient calcs. */
  itkSetMacro(MovingImageUpperPixelValue, MovingImagePixelType);
  itkGetMacro(MovingImageUpperPixelValue, MovingImagePixelType);

protected:
  LinearlyInterpolatedDerivativeFilter();
  ~LinearlyInterpolatedDerivativeFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** ResampleImageFilter produces an image which is a different size
   * than its input.  As such, it needs to provide an implementation
   * for GenerateOutputInformation() in order to inform the pipeline
   * execution model.  The original documentation of this method is
   * below. \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation( void );

  /** ResampleImageFilter needs a different input requested region than
   * the output requested region.  As such, ResampleImageFilter needs
   * to provide an implementation for GenerateInputRequestedRegion()
   * in order to inform the pipeline execution model.
   * \sa ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion( void );

  /** Check before we start. */
  virtual void BeforeThreadedGenerateData();

  /** Just Do it. */
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, int threadId );

  /** (mainly for debugging purposes). */
  virtual void AfterThreadedGenerateData();

  /** Method Compute the Modified Time based on changed to the components. */
  unsigned long GetMTime( void ) const;

private:
  
  /**
   * Prohibited copy and assingment. 
   */
  LinearlyInterpolatedDerivativeFilter(const Self&); 
  void operator=(const Self&); 
  
  TransformPointerType m_Transform;
  
  OutputPixelType m_DefaultPixelValue;
  
  MovingImagePixelType m_MovingImageLowerPixelValue;
  
  MovingImagePixelType m_MovingImageUpperPixelValue;
  
}; /* End class */

} /* End namespace */

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLinearlyInterpolatedDerivativeFilter.txx"
#endif

#endif
