/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkVectorPhiPlusDeltaTTimesVFilter_h
#define __itkVectorPhiPlusDeltaTTimesVFilter_h

#include <itkVector.h>
#include <itkImage.h>
#include <itkImageToImageFilter.h>
#include <itkVectorLinearInterpolateImageFunction.h>

namespace itk {
/**
 * \class VectorPhiPlusDeltaTTimesVFilter
 * \brief This class basically takes two vector images as input (Phi and V) and calculates Phi + (dt * V).
 * 
 * Input 0 is assumed to be Phi, and input 1 is assumed to be V.
 * 
 * Furthermore, Phi should be an image, where each voxel contains a vector which is the
 * absolute position (not the displacement) of a deformation field transformation.
 * V should be a velocity field.
 * 
 * This filter was originally developed to implement step 3(a) in Das et. al. NeuroImage 45 (2009) 867-879.
 * This filter is different to VectorVPlusLambdaUImageFilter, as VectorVPlusLambdaUImageFilter is just
 * adding two vectors together (at integer voxel positions) with a scale factor. This filter is
 * effectively ray tracing, as it adds on the velocity field, interpolating as it goes.
 * As such, this filter can be considered an implementation of Euler's method for integrating
 * a PDE. However, you are advised to use something more sophisticated like
 * itkFourthOrderRungeKuttaVelocityFieldIntegrator.
 * 
 * An additional two inputs are required. Input 2 should be the phi transformation at time zero,
 * specified using SetTimeZeroTransformation. This means we can calculate the displacement internally, 
 * which gives us the current thickness. Input 3 should be a scalar image indicating a thickness prior,
 * and set using SetThicknessPrior. These must both be specified, and then we only update the
 * output value if the calculated thickness is less than the thickness prior.
 * 
 * Furthermore, we do store the thickness value at each point, and expose this map via a pointer.
 * 
 * \sa VectorVPlusLambdaUImageFilter.
 * \sa FourthOrderRungeKuttaVelocityFieldIntegrator
 * 
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT VectorPhiPlusDeltaTTimesVFilter : 
public ImageToImageFilter<
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input images
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                         >
{
  public:

    /** Standard "Self" typedef. */
    typedef VectorPhiPlusDeltaTTimesVFilter                                             Self;
    typedef ImageToImageFilter<Image< Vector<TScalarType, NDimensions>,  NDimensions>,
                               Image< Vector<TScalarType, NDimensions>,  NDimensions>
                              >                                                         Superclass;
    typedef SmartPointer<Self>                                                          Pointer;
    typedef SmartPointer<const Self>                                                    ConstPointer;

    /** Standard typedefs. */
    typedef Vector< TScalarType, NDimensions >                                          InputPixelType;
    typedef Image< InputPixelType, NDimensions >                                        InputImageType;
    typedef Image< TScalarType, NDimensions >                                           InputScalarImageType;
    typedef typename InputImageType::IndexType                                          InputImageIndexType;
    typedef typename InputImageType::RegionType                                         InputImageRegionType;
    typedef InputPixelType                                                              OutputPixelType;
    typedef InputImageType                                                              OutputImageType;
    typedef VectorLinearInterpolateImageFunction< InputImageType, 
                                                  TScalarType >                         VectorLinearInterpolatorType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(VectorPhiPlusDeltaTTimesVFilter, ImageToImageFilter);

    /** Get the number of dimensions we are working in. */
    itkStaticConstMacro(Dimension, unsigned int, NDimensions);

    /** Set/Get flag to subtract steps. i.e. output = input 0 - (delta t * input 1). Default false. */
    itkSetMacro(SubtractSteps, bool);
    itkGetMacro(SubtractSteps, bool);

    /** Set/Get Lambda. Default 1. */
    itkSetMacro(DeltaT, double);
    itkGetMacro(DeltaT, double);
    
    /** Set/Get m, the number of steps for this velocity field. Default 1. */
    itkSetMacro(NumberOfSteps, unsigned int);
    itkGetMacro(NumberOfSteps, unsigned int);
    
    /** Set the time zero transformation. */
    void SetTimeZeroTransformation(InputImageType* image) { m_PhiZeroTransformation = image; }
    
    /** Set the thickness prior. */
    void SetThicknessPrior(InputScalarImageType* image) { m_ThicknessPriorImage = image; }
    
    /** Returns a pointer to the thickness image held internally. */
    InputScalarImageType* GetThicknessImage() { return m_ThicknessImage.GetPointer(); }
    
  protected:
    VectorPhiPlusDeltaTTimesVFilter();
    ~VectorPhiPlusDeltaTTimesVFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    // Check before we start.
    virtual void BeforeThreadedGenerateData();
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);
    
  private:
    
    /**
     * Prohibited copy and assignment. 
     */
    VectorPhiPlusDeltaTTimesVFilter(const Self&); 
    void operator=(const Self&); 
    
    /** So we can run the filter backwards. */
    bool m_SubtractSteps;
    
    /** DeltaT. Default 1. */
    double m_DeltaT;
    
    /** The number of steps to take. Default 1. */
    unsigned int m_NumberOfSteps;
    
    /** To interpolate the velocity field. */
    typename VectorLinearInterpolatorType::Pointer m_Interpolator;
    
    /** Transformation at time zero. */
    typename InputImageType::Pointer m_PhiZeroTransformation;
    
    /** Thickness prior image. */
    typename InputScalarImageType::Pointer m_ThicknessPriorImage;
    
    /** Thickness image. */
    typename InputScalarImageType::Pointer m_ThicknessImage;
    
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVectorPhiPlusDeltaTTimesVFilter.txx"
#endif

#endif
