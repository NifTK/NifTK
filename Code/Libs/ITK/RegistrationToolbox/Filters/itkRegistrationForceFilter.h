/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkRegistrationForceFilter_h
#define itkRegistrationForceFilter_h
#include <itkHistogramSimilarityMeasure.h>
#include <itkImageToImageFilter.h>
#include <itkVector.h>
#include <itkImage.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>

namespace itk {
/** 
 * \class RegistrationForceFilter
 * \brief This class takes as input 2 input images, and outputs the registration force.
 *
 * All input images are expected to have the same template parameters and have
 * the same size and origin. The output is a vector image, of the same size
 * as the input image, where each pixel is a vector with as many components as 
 * image dimensions.  ie. you put in 2 images, the fixed image, and the moving
 * image resampled into the same coordinate system as the fixed. Each input 
 * image is a scalar image.  Lets say that are 2D images of 256 x 256 pixels.
 * The output is 256 x 256 of type vector 2D.
 *
 */

template< class TFixedImage, class TMovingImage, class TScalarType>
class ITK_EXPORT RegistrationForceFilter :
  public ImageToImageFilter<TFixedImage, 
                            Image<  
                              Vector< TScalarType, ::itk::GetImageDimension<TFixedImage>::ImageDimension>, 
                              ::itk::GetImageDimension<TFixedImage>::ImageDimension> >
{
public:

  /** Standard "Self" typedef. */
  typedef RegistrationForceFilter                                                       Self;
  typedef ImageToImageFilter<TFixedImage, 
                             Image<  
                               Vector< TScalarType, ::itk::GetImageDimension<TFixedImage>::ImageDimension>, 
                               ::itk::GetImageDimension<TFixedImage>::ImageDimension> > Superclass;
  typedef SmartPointer<Self>                                                            Pointer;
  typedef SmartPointer<const Self>                                                      ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(RegistrationForceFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Standard typedefs. */
  typedef TScalarType                                                                 OutputDataType;
  typedef Vector< OutputDataType, itkGetStaticConstMacro(Dimension) >                 OutputPixelType;
  typedef Image< OutputPixelType, itkGetStaticConstMacro(Dimension) >                 OutputImageType;
  typedef typename Superclass::InputImageType                                         InputImageType;
  typedef typename InputImageType::PixelType                                          InputImagePixelType;
  typedef typename Superclass::InputImageRegionType                                   RegionType;
  typedef TScalarType                                                                 MeasureType;
  
  /** This can be refactored later if necessary, and moved to a derived class. */
  typedef HistogramSimilarityMeasure<TFixedImage, TMovingImage>                       MetricType;
  typedef typename MetricType::Pointer                                                MetricPointer;
  typedef typename MetricType::HistogramType                                          HistogramType;
  typedef typename MetricType::HistogramPointer                                       HistogramPointer;
  typedef typename MetricType::HistogramSizeType                                      HistogramSizeType;
  typedef typename MetricType::HistogramMeasurementVectorType                         HistogramMeasurementVectorType;
  typedef typename MetricType::HistogramFrequencyType                                 HistogramFrequencyType;
  typedef typename MetricType::HistogramIteratorType                                  HistogramIteratorType;
  
  typedef SpatialObject< Dimension >                                                  FixedImageMaskType; 
  
  //typedef Image<Vector<TScalarType, TFixedImage::ImageDimension>, TFixedImage::ImageDimension>  DeformationFieldType; 
  //typedef DisplacementFieldJacobianDeterminantFilter<DeformationFieldType, TScalarType> JacobianDeterminantFilterType;
  typedef Image<double, TFixedImage::ImageDimension> JacobianImageType; 
  
  /** Set the fixed image at position 0. */
  virtual void SetFixedImage(const InputImageType *image) { this->SetNthInput(0, image); }

  /** Set the transformed moving image at position 1. */
  virtual void SetTransformedMovingImage(const InputImageType *image) { this->SetNthInput(1, image); }

  /** Set the un-transformed moving image at position 2. */
  virtual void SetUnTransformedMovingImage(const InputImageType *image) { this->SetNthInput(2, image); }

  /** We set the input images by number. */
  virtual void SetNthInput(unsigned int idx, const InputImageType *);
  
  /** Set/Get the Metric. */
  itkSetObjectMacro( Metric, MetricType );
  itkGetObjectMacro( Metric, MetricType );

  /** 
   * Set/Get the ScaleToSizeOfVoxelAxis flag. 
   * Subclasses or clients decide what to do with it. 
   */
  itkSetMacro( ScaleToSizeOfVoxelAxis, bool );
  itkGetMacro( ScaleToSizeOfVoxelAxis, bool );

  /** Set/Get the FixedUpperPixelValue. */
  itkSetMacro( FixedUpperPixelValue, InputImagePixelType);
  itkGetMacro( FixedUpperPixelValue, InputImagePixelType);

  /** Set/Get the FixedUpperPixelValue. */
  itkSetMacro( FixedLowerPixelValue, InputImagePixelType);
  itkGetMacro( FixedLowerPixelValue, InputImagePixelType);

  /** Set/Get the MovingUpperPixelValue. */
  itkSetMacro( MovingUpperPixelValue, InputImagePixelType);
  itkGetMacro( MovingUpperPixelValue, InputImagePixelType);

  /** Set/Get the MovingUpperPixelValue. */
  itkSetMacro( MovingLowerPixelValue, InputImagePixelType);
  itkGetMacro( MovingLowerPixelValue, InputImagePixelType);
  
  /**
   * Set/Get. 
   */
  itkSetMacro(IsSymmetric, bool); 
  itkGetMacro(IsSymmetric, bool); 

  /** Set fixed image mask. */
  virtual void SetFixedImageMask(const FixedImageMaskType* fixedImageMask) { this->m_FixedImageMask = fixedImageMask; } 
  
  /** Mainly for debugging, write image to file. */
  void WriteForceImage(std::string filename);
  
  /**
   * Set m_FixedImageTransformJacobian. 
   */
  virtual void SetFixedImageTransformJacobian(typename JacobianImageType::Pointer jacobian)
  {
    this->m_FixedImageTransformJacobian = jacobian; 
  }
      
  /**
   * Set m_MovingImageTransformJacobian. 
   */
  virtual void SetMovingImageTransformJacobian(typename JacobianImageType::Pointer jacobian)
  {
    this->m_MovingImageTransformJacobian = jacobian; 
  }
  
protected:
  RegistrationForceFilter();
  ~RegistrationForceFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  // Check before we start.
  virtual void BeforeThreadedGenerateData();
  
  // (mainly for debugging purposes).
  virtual void AfterThreadedGenerateData();
  
  /** We need this to calculate entropies, so it should be a histogram based one. */
  MetricPointer m_Metric;  

  /** Multiply force by size of voxels. */
  bool m_ScaleToSizeOfVoxelAxis;

  /** So we can mask out intensities to generate zero force. */
  InputImagePixelType m_FixedUpperPixelValue;
  
  /** So we can mask out intensities to generate zero force. */
  InputImagePixelType m_FixedLowerPixelValue;

  /** So we can mask out intensities to generate zero force. */
  InputImagePixelType m_MovingUpperPixelValue;
  
  /** So we can mask out intensities to generate zero force. */
  InputImagePixelType m_MovingLowerPixelValue;

  /** Fixed image mask. */
  const FixedImageMaskType* m_FixedImageMask; 
  
  /**
   * Symmetric? 
   */
  bool m_IsSymmetric; 
  
  /**
   * Jacobian of the fixed image transform. 
   */
  typename JacobianImageType::Pointer m_FixedImageTransformJacobian; 
  
  /**
   * Jacobian of the moving image transform. 
   */
  typename JacobianImageType::Pointer m_MovingImageTransformJacobian; 

private:
  
  /**
   * Prohibited copy and assingment. 
   */
  RegistrationForceFilter(const Self&); 
  void operator=(const Self&); 
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRegistrationForceFilter.txx"
#endif

#endif
