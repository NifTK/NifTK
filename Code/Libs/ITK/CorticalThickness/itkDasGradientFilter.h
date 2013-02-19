/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDasGradientFilter_h
#define __itkDasGradientFilter_h

#include "itkVector.h"
#include "itkImage.h"
#include "itkImageToImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkImageFileWriter.h"
#include "itkDasTransformImageFilter.h"

namespace itk {
/**
 * \class DasGradientFilter
 * \brief This class calculates the gradient as per equation 3 in Das et. al. Neuroimage 45 (2009) 867-879.
 * 
 * The inputs should be exactly as follows
 * 
 * <pre>
 * 1. WM PV map, set using SetInput(0, image)
 * 2. G+W PV map, set using SetInput(1, image)
 * 3. Thickness prior map, set using SetInput(2, image)
 * 4. Thickness image, set using SetInput(3, image)
 * 5. Phi transformation, set using SetTransformation(image) as this is a vector image.
 * </pre>
 * 
 * \sa VectorVPlusLambdaUImageFilter.
 * 
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT DasGradientFilter : 
public ImageToImageFilter<
                           Image< TScalarType, NDimensions>,                      // Input image
                           Image< Vector<TScalarType, NDimensions>,  NDimensions> // Output image
                         >
{
  public:

    /** Standard "Self" typedef. */
    typedef DasGradientFilter                                                Self;
    typedef ImageToImageFilter<
                               Image< TScalarType, 
                                      NDimensions>,
                               Image< Vector<TScalarType, NDimensions>,  
                                      NDimensions>  
                              >                                              Superclass;
    typedef SmartPointer<Self>                                               Pointer;
    typedef SmartPointer<const Self>                                         ConstPointer;

    /** Standard typedefs. */
    typedef Image< TScalarType, NDimensions >                                InputImageType;
    typedef typename InputImageType::IndexType                               InputImageIndexType;
    typedef typename InputImageType::RegionType                              InputImageRegionType;
    typedef Vector< TScalarType, NDimensions >                               VectorPixelType;
    typedef Image< VectorPixelType, NDimensions >                            VectorImageType;
    typedef VectorPixelType                                                  OutputPixelType;
    typedef VectorImageType                                                  OutputImageType;
    typedef VectorLinearInterpolateImageFunction <VectorImageType, 
                                                     TScalarType >           VectorInterpolatorType;
    typedef LinearInterpolateImageFunction< InputImageType, TScalarType >    LinearInterpolatorType;
    typedef ScalarImageToNormalizedGradientVectorImageFilter<InputImageType, 
                                                             TScalarType>    GradientFilterType;
    typedef DasTransformImageFilter<TScalarType, NDimensions>                TransformImageFilterType;
    
    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(DasGradientFilter, ImageToImageFilter);

    /** Get the number of dimensions we are working in. */
    itkStaticConstMacro(Dimension, unsigned int, NDimensions);

    /** Set the current value of the transformation (phi). */
    void SetTransformation(VectorImageType* image) { m_PhiTransformation = image; }
    
    /** Set/Get flag to switch gradient round. Default false */
    itkSetMacro(ReverseGradient, bool);
    itkGetMacro(ReverseGradient, bool);

    /** 
     * Set/Get flag where we multiply by either the gradient of the
     * moving image, or the gradient of the transformed moving image. 
     */
    itkSetMacro(UseGradientTransformedMovingImage, bool);
    itkGetMacro(UseGradientTransformedMovingImage, bool);
    
  protected:
    DasGradientFilter();
    ~DasGradientFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    // Before we start multi-threaded section.
    virtual void BeforeThreadedGenerateData();

    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);
    
  private:
    
    /**
     * Prohibited copy and assignment. 
     */
    DasGradientFilter(const Self&); 
    void operator=(const Self&); 

    /** The transformation phi. */
    typename VectorImageType::Pointer m_PhiTransformation;

    /** To interpolate the thickness map at the transformed position. */
    typename LinearInterpolatorType::Pointer m_ThicknessInterpolator;

    /** To interpolate the white matter PV map */
    typename LinearInterpolatorType::Pointer m_WhiteMatterInterpolator;

    /** To interpolate the gradient map */
    typename VectorInterpolatorType::Pointer m_GradientInterpolator;

    /** To take gradient image. */
    typename GradientFilterType::Pointer m_GradientFilter;
    
    /** To make a transformed moving image. */
    typename TransformImageFilterType::Pointer m_TransformImageFilter;
    
    /** To switch gradient round. */
    bool m_ReverseGradient;

    /** We multiply by gradient of moving image, or gradient of transformed moving image. Default true. */
    bool m_UseGradientTransformedMovingImage;
    
    /** To make sure we only take gradient of moving image once. */
    bool m_GradientFilterInitialized;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDasGradientFilter.txx"
#endif

#endif
