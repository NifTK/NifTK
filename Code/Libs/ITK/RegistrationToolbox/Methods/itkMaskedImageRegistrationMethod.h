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

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkMaskedImageRegistrationMethod_h
#define __itkMaskedImageRegistrationMethod_h


#include "itkSingleResolutionImageRegistrationMethod.h"
#include "itkImageMaskSpatialObject.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkBoundaryValueRescaleIntensityImageFilter.h"
#include "itkMultiplyImageFilter.h"

namespace itk
{

/** 
 * \class MaskedImageRegistrationMethod
 * \brief Base class for NifTK Image Registration Methods employing a binary mask.
 *
 * This Class extends the SingleResolutionImageRegistrationMethod, with inputs for
 * a fixed and moving mask. This class provides for thresholding the mask,
 * dilating the mask a number of times and optionally multiplying it by the
 * input image, or just setting it on the similarity measure.
 * 
 * The threshold, dilating, masking is all applied within the Initialize()
 * method, at then end of which, the Initialize methods calls SetFixedImage
 * and SetMovingImage with the resultant images.  
 * 
 * We also take and store a copy of the original Fixed and Moving image so
 * that subclasses can have access to it. For example, block matching
 * needs a masked image to decide which sets of points to use, but then
 * when its doing the match, it always wants unmasked images.
 *
 * \sa MultiResolutionImageRegistrationWrapper
 * 
 * \ingroup RegistrationFilters
 */
template <typename TInputImageType>
class ITK_EXPORT MaskedImageRegistrationMethod 
: public SingleResolutionImageRegistrationMethod<TInputImageType, TInputImageType> 
{
public:

  /** Standard class typedefs. */
  typedef MaskedImageRegistrationMethod                                             Self;
  typedef SingleResolutionImageRegistrationMethod<TInputImageType, TInputImageType> Superclass;
  typedef SmartPointer<Self>                                                        Pointer;
  typedef SmartPointer<const Self>                                                  ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(MaskedImageRegistrationMethod, SingleResolutionImageRegistrationMethod);

  /**  Type of the input image. */
  typedef          TInputImageType                                                  InputImageType;
  typedef typename InputImageType::PixelType                                        InputImagePixelType;
  typedef typename InputImageType::Pointer                                          InputImagePointer;
  typedef typename InputImageType::ConstPointer                                     InputImageConstPointer;
  typedef typename InputImageType::RegionType                                       InputImageRegionType;
  typedef typename InputImageType::SizeType                                         InputImageSizeType;
  typedef typename InputImageType::IndexType                                        InputImageIndexType;
  typedef typename InputImageType::SpacingType                                      InputImageSpacingType;
  typedef typename InputImageType::PointType                                        InputImageOriginType;
  typedef typename InputImageType::DirectionType                                    InputImageDirectionType;

  /** Set/Get the Fixed mask. */
  void SetFixedMask( const InputImageType * fixedMask );
  itkGetConstObjectMacro( FixedMask, InputImageType ); 

  /** Set/Get the Moving mask. */
  void SetMovingMask( const InputImageType * movingMask );
  itkGetConstObjectMacro( MovingMask, InputImageType );

  /**
   * This class provides rescaling of inputs. This is useful, for example if you
   * are doing NMI based registration, and want to rescale inputs to match
   * a histogram size, or for example, if you are doing similarity measures
   * like CC, RIU, or PIU where the magnitude of the scalar values will impact 
   * the sensitivity of the registration algorithm.
   */
  typedef BoundaryValueRescaleIntensityImageFilter<InputImageType>                  RescaleFilterType;
  typedef typename RescaleFilterType::Pointer                                       RescaleFilterPointer;
  
  /** Now added smoothing, after rescaling. */
  typedef SmoothingRecursiveGaussianImageFilter<InputImageType, InputImageType >    SmoothingFilterType;
  typedef typename SmoothingFilterType::Pointer                                     SmoothingFilterPointer;
  
  /** Threshold mask to 0 and 1, just in case its not binary already. */
  typedef BinaryThresholdImageFilter<InputImageType, InputImageType>                ThresholdFilterType;
  typedef typename ThresholdFilterType::Pointer                                     ThresholdFilterPointer;
  
  /** Then we dilate it a number of times. */
  typedef BinaryCrossStructuringElement<InputImagePixelType, 
                                        InputImageType::ImageDimension>             StructuringType;
  typedef BinaryDilateImageFilter<InputImageType, 
                                  InputImageType, 
                                  StructuringType>                                  DilateMaskFilterType;
  typedef typename DilateMaskFilterType::Pointer                                    DilateMaskFilterPointer;

  /** Optionally, we multiply the mask with the input, or we pass it into ImageMaskSpatialObject. */
  typedef MultiplyImageFilter<InputImageType, InputImageType>                       MultiplyFilterType;
  typedef typename MultiplyFilterType::Pointer                                      MultiplyFilterPointer;
  
  /** Note that the ImageMaskSpatialObject types need unsigned char. */
  typedef unsigned char                                                             MaskPixelType;
  typedef Image<MaskPixelType, InputImageType::ImageDimension>                      MaskImageType;
  typedef CastImageFilter<InputImageType, MaskImageType >                           CastToMaskImageTypeFilterType;
  typedef typename CastToMaskImageTypeFilterType::Pointer                           CastToMaskImageTypeFilterPointer;
  typedef ImageMaskSpatialObject< InputImageType::ImageDimension >                  MaskFilterType;
  typedef typename MaskFilterType::Pointer                                          MaskFilterPointer;
  typedef typename Superclass::TransformType                                        TransformType;
  /** 
   * Turns whether we are using the fixed mask at all.
   * 
   * Default is off.
   */
  itkSetMacro( UseFixedMask, bool );
  itkGetConstMacro( UseFixedMask, bool );

  /** 
   * Turns whether we are using the moving mask at all.
   * 
   * Default is off.
   */
  itkSetMacro( UseMovingMask, bool );
  itkGetConstMacro( UseMovingMask, bool );

  /**
   * Set the number of mask dilations. 
   * 
   * Default is zero. 
   */
  itkSetMacro( NumberOfDilations, unsigned int );
  itkGetMacro( NumberOfDilations, unsigned int );
  
  /** 
   * If this is true, we multiply the mask by the image, and only give the single resolution
   * method, an already masked image. If this is false, we leave the image as it is, and
   * set the mask onto the single res method.  You probably want true for deformable stuff,
   * and false for affine stuff. Default false.
   */
  itkSetMacro( MaskImageDirectly, bool );
  itkGetMacro( MaskImageDirectly, bool );
  
  /** 
   * Turns image rescaling of the fixed image off or on. 
   * If you turn it on, you should also set the Min/Max values you want to rescale to,
   * as they default to 0,255.
   * 
   * Default is off.
   * 
   */
  itkSetMacro( RescaleFixedImage, bool );
  itkGetConstMacro( RescaleFixedImage, bool );

  /**
   * Set the minimum rescale value for the fixed image.
   */
  itkSetMacro( RescaleFixedMinimum, InputImagePixelType );
  itkGetConstMacro( RescaleFixedMinimum, InputImagePixelType );

  /**
   * Set the Maximum rescale value for the fixed image.
   */
  itkSetMacro( RescaleFixedMaximum, InputImagePixelType );
  itkGetConstMacro( RescaleFixedMaximum, InputImagePixelType );

  /**
   * Set the boundary value for if we want to do thresholding.
   */
  itkSetMacro( RescaleFixedBoundaryValue, InputImagePixelType );
  itkGetConstMacro( RescaleFixedBoundaryValue, InputImagePixelType );

  /**
   * Set the lower threshold, so we can mask these values out before rescaling.
   */
  itkSetMacro( RescaleFixedLowerThreshold, InputImagePixelType );
  itkGetConstMacro( RescaleFixedLowerThreshold, InputImagePixelType );

  /**
   * Set the upper threshold, so we can mask these values out before rescaling.
   */
  itkSetMacro( RescaleFixedUpperThreshold, InputImagePixelType );
  itkGetConstMacro( RescaleFixedUpperThreshold, InputImagePixelType );

  /** 
   * Turns image rescaling of the moving image off or on. 
   * If you turn it on, you should also set the Min/Max values you want to rescale to
   * as they default to 0,255.
   * 
   * Default is off.
   * 
   */
  itkSetMacro( RescaleMovingImage, bool );
  itkGetConstMacro( RescaleMovingImage, bool );

  /**
   * Set the minimum rescale value for the moving image.
   */
  itkSetMacro( RescaleMovingMinimum, InputImagePixelType );
  itkGetConstMacro( RescaleMovingMinimum, InputImagePixelType );

  /**
   * Set the Maximum rescale value for the moving image.
   */
  itkSetMacro( RescaleMovingMaximum, InputImagePixelType );
  itkGetConstMacro( RescaleMovingMaximum, InputImagePixelType );

  /**
   * Set the boundary value for if we want to do thresholding.
   */
  itkSetMacro( RescaleMovingBoundaryValue, InputImagePixelType );
  itkGetConstMacro( RescaleMovingBoundaryValue, InputImagePixelType );

  /**
   * Set the lower threshold, so we can mask these values out before rescaling.
   */
  itkSetMacro( RescaleMovingLowerThreshold, InputImagePixelType );
  itkGetConstMacro( RescaleMovingLowerThreshold, InputImagePixelType );

  /**
   * Set the upper threshold, so we can mask these values out before rescaling.
   */
  itkSetMacro( RescaleMovingUpperThreshold, InputImagePixelType );
  itkGetConstMacro( RescaleMovingUpperThreshold, InputImagePixelType );

  /** 
   * Threshold fixed mask. 
   * 
   * Defaults to true. 
   */
  itkSetMacro( ThresholdFixedMask, bool );
  itkGetConstMacro( ThresholdFixedMask, bool );

  /** 
   * Lower limit to threshold mask at.
   * 
   * Defaults to 1. 
   */
  itkSetMacro( FixedMaskMinimum, InputImagePixelType );
  itkGetConstMacro( FixedMaskMinimum, InputImagePixelType );

  /** 
   * Upper limit to threshold at.
   * 
   * Defaults to maximum for pixel type. 
   */
  itkSetMacro( FixedMaskMaximum, InputImagePixelType );
  itkGetConstMacro( FixedMaskMaximum, InputImagePixelType );

  /** 
   * Threshold moving mask. 
   * 
   * Defaults to true. 
   */
  itkSetMacro( ThresholdMovingMask, bool );
  itkGetConstMacro( ThresholdMovingMask, bool );

  /** 
   * Lower limit to threshold at. 
   * 
   * Defaults to 1. 
   */
  itkSetMacro( MovingMaskMinimum, InputImagePixelType );
  itkGetConstMacro( MovingMaskMinimum, InputImagePixelType );

  /** 
   * Upper limit to threshold at. 
   * 
   * Defaults to maximum for pixel type. 
   */
  itkSetMacro( MovingMaskMaximum, InputImagePixelType );
  itkGetConstMacro( MovingMaskMaximum, InputImagePixelType );

  /** 
   * Set the Sigma for Gaussian smoothing. 
   * 
   * Default 0. 
   */
  itkSetMacro( Sigma, float);
  itkGetMacro( Sigma, float);
  
  /** Get the fixed image copy. Lazy Initialized. */
  InputImageType* GetFixedImageCopy();

  /** Get the moving image copy. Lazy Initialized. */
  InputImageType* GetMovingImageCopy();
  
protected:

  MaskedImageRegistrationMethod();
  virtual ~MaskedImageRegistrationMethod() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Override this method to wire everything together. */
  virtual void Initialize() throw (ExceptionObject);

  /** Just to copy image. */
  void CopyImage(const InputImageType* source, InputImageType* target);
  
private:
  
  MaskedImageRegistrationMethod(const Self&); // purposely not implemented
  void operator=(const Self&);                // purposely not implemented
  
  InputImageConstPointer                m_FixedMask;
  InputImageConstPointer                m_MovingMask;

  bool                                  m_UseFixedMask;
  bool                                  m_UseMovingMask;
  bool                                  m_MaskImageDirectly;
  bool                                  m_RescaleFixedImage;
  bool                                  m_RescaleMovingImage;
  bool                                  m_ThresholdFixedMask;
  bool                                  m_ThresholdMovingMask;  
  unsigned int                          m_NumberOfDilations;

  InputImagePixelType                   m_RescaleFixedMinimum;
  InputImagePixelType                   m_RescaleFixedMaximum;
  InputImagePixelType                   m_RescaleFixedBoundaryValue;
  InputImagePixelType                   m_RescaleFixedLowerThreshold;
  InputImagePixelType                   m_RescaleFixedUpperThreshold;
  InputImagePixelType                   m_RescaleMovingMinimum;
  InputImagePixelType                   m_RescaleMovingMaximum;
  InputImagePixelType                   m_RescaleMovingBoundaryValue;
  InputImagePixelType                   m_RescaleMovingLowerThreshold;
  InputImagePixelType                   m_RescaleMovingUpperThreshold;

  RescaleFilterPointer                  m_FixedRescaler;
  RescaleFilterPointer                  m_MovingRescaler;

  float                                 m_Sigma;
  SmoothingFilterPointer                m_FixedSmoother;
  SmoothingFilterPointer                m_MovingSmoother;
  
  InputImagePixelType                   m_FixedMaskMinimum;
  InputImagePixelType                   m_FixedMaskMaximum;
  InputImagePixelType                   m_MovingMaskMinimum;
  InputImagePixelType                   m_MovingMaskMaximum;
  ThresholdFilterPointer                m_FixedMaskThresholder;
  ThresholdFilterPointer                m_MovingMaskThresholder;
  
  DilateMaskFilterPointer               m_FixedMaskDilater;
  DilateMaskFilterPointer               m_MovingMaskDilater;
  
  MultiplyFilterPointer                 m_FixedImageMuliplier;
  MultiplyFilterPointer                 m_MovingImageMultiplier;
  
  CastToMaskImageTypeFilterPointer      m_FixedMaskCaster;
  CastToMaskImageTypeFilterPointer      m_MovingMaskCaster;

  MaskFilterPointer                     m_FixedMasker;
  MaskFilterPointer                     m_MovingMasker;
  
  InputImagePointer                     m_FixedImageCopy;
  InputImagePointer                     m_MovingImageCopy;

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMaskedImageRegistrationMethod.txx"
#endif

#endif



