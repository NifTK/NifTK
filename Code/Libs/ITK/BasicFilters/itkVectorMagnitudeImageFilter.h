/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkVectorMagnitudeImageFilter_h
#define __itkVectorMagnitudeImageFilter_h

#include <itkVector.h>
#include <itkImage.h>
#include <itkImageToImageFilter.h>

namespace itk {
/**
 * \class VectorMagnitudeImageFilter
 * \brief This class takes a vector image as input, and outputs the Euclidean magnitude.
 * 
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT VectorMagnitudeImageFilter : 
public ImageToImageFilter<
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input image
                           Image< TScalarType,  NDimensions>                       // Output image
                         >
{
  public:

    /** Standard "Self" typedef. */
    typedef VectorMagnitudeImageFilter                                                  Self;
    typedef ImageToImageFilter<Image< Vector<TScalarType, NDimensions>,  NDimensions>,
                               Image< TScalarType,  NDimensions>
                              >                                                         Superclass;
    typedef SmartPointer<Self>                                                          Pointer;
    typedef SmartPointer<const Self>                                                    ConstPointer;

    /** Standard typedefs. */
    typedef Vector< TScalarType, NDimensions >                                          InputPixelType;
    typedef Image< InputPixelType, NDimensions >                                        InputImageType;
    typedef typename InputImageType::IndexType                                          InputImageIndexType;
    typedef typename InputImageType::RegionType                                         InputImageRegionType;
    typedef float                                                                       OutputPixelType;
    typedef Image< OutputPixelType, NDimensions >                                       OutputImageType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(VectorMagnitudeImageFilter, ImageToImageFilter);

    /** Get the number of dimensions we are working in. */
    itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  protected:
    VectorMagnitudeImageFilter();
    ~VectorMagnitudeImageFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    // Check before we start.
    virtual void BeforeThreadedGenerateData();
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);
    
  private:
    
    /**
     * Prohibited copy and assignment. 
     */
    VectorMagnitudeImageFilter(const Self&); 
    void operator=(const Self&); 
    
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVectorMagnitudeImageFilter.txx"
#endif

#endif
