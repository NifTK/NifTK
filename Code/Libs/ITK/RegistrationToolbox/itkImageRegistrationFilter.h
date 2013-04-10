/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageRegistrationFilter_h
#define __itkImageRegistrationFilter_h


#include "itkImageToImageFilter.h"

#include "itkCastImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkSingleResolutionImageRegistrationMethod.h"
#include "itkMultiResolutionImageRegistrationWrapper.h"
#include "itkImageRegistrationFactory.h"
#include "itkAbsImageFilter.h"

namespace itk
{
/** 
 * \class ImageRegistrationFilter
 * \brief Used to plug registration methods into a filter based pipeline.
 * 
 * The purpose of this filter is simply to run a fully configured
 * multi-resolution image registration method, and make sure the outputs
 * come out in a consistent order.
 *  
 * Inputs:
 * 
 * 1. Fixed Image
 * 
 * 2. Moving image
 * 
 * 3. Fixed Mask
 * 
 * 4. Moving Mask
 * 
 * Outputs:
 * 
 * 1. Transformed image
 * 
 * 2. Transformation
 * 
 * 3. If Transform is a subclass of itkDeformableTransform.h, then jacobian image.
 * 
 * 4. If Transform is a subclass of itkDeformableTransform.h, then vector (deformation) image.
 */
template <typename TInputImageType, typename TOutputImageType, unsigned int Dimension, class TScalarType, typename TDeformationScalar, typename TPyramidFilter = RecursiveMultiResolutionPyramidImageFilter< TInputImageType, TInputImageType > >
class ITK_EXPORT ImageRegistrationFilter : public ImageToImageFilter< TInputImageType, TOutputImageType > 
{
  public:

    /** Standard class typedefs. */
    typedef ImageRegistrationFilter                                        Self;
    typedef ImageToImageFilter< TInputImageType, TOutputImageType >        Superclass;
    typedef SmartPointer<Self>                                             Pointer;
    typedef SmartPointer<const Self>                                       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(ImageRegistrationFilter, ImageToImageFilter);

    /** Any additional type defs. */
    typedef typename TInputImageType::PixelType                            InputPixelType;    
    typedef typename TInputImageType::Pointer                              InputImagePointer;
    typedef typename TInputImageType::ConstPointer                         InputImageConstPointer;

    /**
     * This is for the underlying multi-resolution registration.
     */
    typedef itk::MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>  MultiResolutionRegistrationType;
    typedef typename MultiResolutionRegistrationType::Pointer              MultiResolutionRegistrationPointer;
    typedef typename MultiResolutionRegistrationType::SingleResType        SingleResType;
    typedef typename SingleResType::TransformType                          TransformType;
    typedef itk::ImageRegistrationFactory<TInputImageType, Dimension, 
                                                           TScalarType>    ImageRegistrationFactoryType;
    typedef typename ImageRegistrationFactoryType::Pointer                 ImageRegistrationFactoryPointer;
    
    /**
     * And, as this is a filter, we resample the moving image
     * into the coordinates of the fixed image, so we
     * resample and produce an image as first output,
     * and the transformation as the second output.
     */
    typedef itk::InterpolateImageFunction< TInputImageType, TScalarType >  InterpolatorType;
    typedef typename InterpolatorType::Pointer                             InterpolatorPointer;
    typedef itk::ResampleImageFilter<TInputImageType, TInputImageType >    ResampleFilterType;
    typedef typename ResampleFilterType::Pointer                           ResampleFilterPointer;
    typedef itk::AbsImageFilter<TInputImageType, TInputImageType >         AbsImageFilterType;
    typedef itk::CastImageFilter<TInputImageType, TOutputImageType >       CastToOutputFilterType;
    typedef typename CastToOutputFilterType::Pointer                       CastToOutputFilterPointer;
    
    typedef itk::FluidDeformableTransform<TInputImageType, TScalarType, Dimension, TDeformationScalar > FluidDeformableTransformType;
    
    /** Set/Get the Multi-res method. */
    itkSetObjectMacro( MultiResolutionRegistrationMethod, MultiResolutionRegistrationType );
    itkGetObjectMacro( MultiResolutionRegistrationMethod, MultiResolutionRegistrationType );

    /** Set/Get the Interpolator used for final resampling (not registration). */
    itkSetObjectMacro( Interpolator, InterpolatorType );
    itkGetObjectMacro( Interpolator, InterpolatorType );

    /** Sets the fixed image at input position 0. */
    void SetFixedImage(InputImagePointer fixedImage) { this->SetInput(0, fixedImage); }

    /** Sets the moving image at input position 1. */
    void SetMovingImage(InputImagePointer  movingImage) { this->m_MovingImage =  movingImage; }

    /** Sets the fixed mask at input position 2. */
    void SetFixedMask(InputImagePointer fixedMask) {this->SetInput(1, fixedMask); }

    /** Sets the moving mask at input position 3. */
    void SetMovingMask(InputImagePointer movingMask) { this->m_MovingMask = movingMask; }

    /** Turn off/on reslicng, default ON */
    itkSetMacro(DoReslicing, bool);
    itkGetMacro(DoReslicing, bool);
    
    /** Set/Get m_IsOutputAbsIntensity */
    itkSetMacro(IsOutputAbsIntensity, bool); 
    itkGetMacro(IsOutputAbsIntensity, bool); 
    
    /** Set/Get m_IsotropicVoxelSize */
    itkSetMacro(IsotropicVoxelSize, double); 
    itkGetMacro(IsotropicVoxelSize, double); 
    
    /** Set/Get m_ResampleImageInterpolation */
    itkSetMacro(ResampleImageInterpolation, InterpolationTypeEnum); 
    itkGetMacro(ResampleImageInterpolation, InterpolationTypeEnum); 
    
    /** Set/Get m_ResampleMaskInterpolation */
    itkSetMacro(ResampleMaskInterpolation, InterpolationTypeEnum); 
    itkGetMacro(ResampleMaskInterpolation, InterpolationTypeEnum); 
    
    /** Set/Get the m_ResampledMovingImagePadValue. */
    itkSetMacro(ResampledMovingImagePadValue, InputPixelType);
    itkGetMacro(ResampledMovingImagePadValue, InputPixelType);

    /** Set/Get the m_FixedImagePadValue. */
    itkSetMacro(ResampledFixedImagePadValue, InputPixelType);
    itkGetMacro(ResampledFixedImagePadValue, InputPixelType);

    /** Resample the given image to a different voxel size using the given interpolation */
    InputImagePointer ResampleToVoxelSize(const TInputImageType* image, const InputPixelType defaultPixelValue, InterpolationTypeEnum interpolation, typename TInputImageType::SpacingType voxelSize); 

  protected:

    ImageRegistrationFilter();
    virtual ~ImageRegistrationFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** This is called by base class to run filter. */
    void GenerateData();

    /**
     * For setting up the pipeline. This is called by GenerateData() method.
     */
    virtual void Initialize();
    
    /** Reference to the factory, to help us build stuff. */
    ImageRegistrationFactoryPointer m_ImageRegistrationFactory;

    /** To cast to output type, as this class is Templated based on output type. */
    CastToOutputFilterPointer m_FinalCaster;

    /** For resampling the image once we have finished registration. */
    ResampleFilterPointer m_FinalResampler;

    /** This interpolator is JUST used for the final resampling, default is Linear. */
    InterpolatorPointer m_Interpolator;
    
    /** For running the multi-resolution bit. */
    MultiResolutionRegistrationPointer m_MultiResolutionRegistrationMethod;
    
    /** Perform the optional abs value to the resliced image */
    typename AbsImageFilterType::Pointer m_AbsImageFilter; 
    
    /** Resampled fixed image */
    typename TInputImageType::ConstPointer m_ResampledFixedImage; 
    
    /** Resampled moving image */
    typename TInputImageType::ConstPointer m_ResampledMovingImage; 
    
    /** Resampled fixed mask */
    typename TInputImageType::ConstPointer m_ResampledFixedMask; 
    
    /** Resampled moving image */
    typename TInputImageType::ConstPointer m_ResampledMovingMask; 
    
    /** Image resampling interpolation mode */
    InterpolationTypeEnum m_ResampleImageInterpolation;
        
    /** Image resampling interpolation mode */
    InterpolationTypeEnum m_ResampleMaskInterpolation;
    
  private:
    
    ImageRegistrationFilter(const Self&); // purposefully not implemented
    void operator=(const Self&);          // purposefully not implemented
    
    /** Keeps the moving image */
    InputImagePointer m_MovingImage; 
    
    /** Keeps the moving image mask */
    InputImagePointer m_MovingMask; 

    /** So, when we resample the moving image, we pad with a given value. Default 0. */
    InputPixelType m_ResampledMovingImagePadValue;
    
    /** When we resample the fixed image (only done if we ask for isotropic voxels), pad with a given value. Default 0. */
    InputPixelType m_ResampledFixedImagePadValue;
    
    /** Turns off the reslicing. */
    bool m_DoReslicing;
    
    /** Output non-negative intensity values. */
    bool m_IsOutputAbsIntensity; 
    
    /** Resample the input images to this isotropic size if > 0. */
    double m_IsotropicVoxelSize; 
    
};
  
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRegistrationFilter.txx"
#endif

#endif
