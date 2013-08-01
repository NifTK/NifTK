/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkVectorVPlusLambdaUImageFilter_h
#define itkVectorVPlusLambdaUImageFilter_h

#include <itkVector.h>
#include <itkImage.h>
#include <itkImageToImageFilter.h>

namespace itk {
/**
 * \class VectorVPlusLambdaUImageFilter
 * \brief This class takes two vector images as input (V and U), and calculates V + (lambda * U).
 * 
 * Input 0 is assumed to be V, and input 1 is assumed to be U.
 * 
 * So output is Input1 + (lambda * Input2).
 * 
 * It was originally made to implement step 4 in Das et. al. NeuroImage 45 (2009) 867-879.
 * 
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT VectorVPlusLambdaUImageFilter : 
public ImageToImageFilter<
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input images
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                         >
{
  public:

    /** Standard "Self" typedef. */
    typedef VectorVPlusLambdaUImageFilter                                               Self;
    typedef ImageToImageFilter<Image< Vector<TScalarType, NDimensions>,  NDimensions>,
                               Image< Vector<TScalarType, NDimensions>,  NDimensions>
                              >                                                         Superclass;
    typedef SmartPointer<Self>                                                          Pointer;
    typedef SmartPointer<const Self>                                                    ConstPointer;

    /** Standard typedefs. */
    typedef Vector< TScalarType, NDimensions >                                          InputPixelType;
    typedef Image< InputPixelType, NDimensions >                                        InputImageType;
    typedef typename InputImageType::IndexType                                          InputImageIndexType;
    typedef typename InputImageType::RegionType                                         InputImageRegionType;
    typedef InputPixelType                                                              OutputPixelType;
    typedef InputImageType                                                              OutputImageType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(VectorVPlusLambdaUImageFilter, ImageToImageFilter);

    /** Get the number of dimensions we are working in. */
    itkStaticConstMacro(Dimension, unsigned int, NDimensions);

    /** Set/Get Lambda. Default 1. */
    itkSetMacro(Lambda, double);
    itkGetMacro(Lambda, double);

    /** If false (the default), we compute Output = V + Lambda * U, if true, we compute Output = Lambda * U. */
    itkSetMacro(IgnoreInputV, bool);
    itkGetMacro(IgnoreInputV, bool);
    
    /** Set/Get flag to subtract lambda u instead of adding. Default false, so we add lambda u */
    itkSetMacro(SubtractSteps, bool);
    itkGetMacro(SubtractSteps, bool);

  protected:
    VectorVPlusLambdaUImageFilter();
    ~VectorVPlusLambdaUImageFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    // Check before we start.
    virtual void BeforeThreadedGenerateData();
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);
    
  private:
    
    /**
     * Prohibited copy and assignment. 
     */
    VectorVPlusLambdaUImageFilter(const Self&); 
    void operator=(const Self&); 

    /** Flag so we can ignore input V. Default false. */
    bool m_IgnoreInputV;
    
    /** Flag so we can subtract instead of adding. Default false. */
    bool m_SubtractSteps;

    /** Lambda. Default 1. */
    double m_Lambda;
    
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVectorVPlusLambdaUImageFilter.txx"
#endif

#endif
