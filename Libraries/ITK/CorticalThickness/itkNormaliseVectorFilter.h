/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkNormaliseVectorFilter_h
#define itkNormaliseVectorFilter_h

#include <itkVector.h>
#include <itkImage.h>
#include <itkImageToImageFilter.h>

namespace itk {
/**
 * \class NormaliseVectorFilter
 * \brief This class takes a vector field and normalises each vector to unit length.
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT NormaliseVectorFilter : 
public ImageToImageFilter<
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input images
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                         >
{
  public:

    /** Standard "Self" typedef. */
    typedef NormaliseVectorFilter                                                       Self;
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

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(NormaliseVectorFilter, ImageToImageFilter);

    /** Get the number of dimensions we are working in. */
    itkStaticConstMacro(Dimension, unsigned int, NDimensions);

    /** Set/Get the normalise flag. If off/false, then filter simply passess vectors through. Default true. */
    itkSetMacro(Normalise, bool);
    itkGetMacro(Normalise, bool);
    
    /** Set/Get the tolerance of the length. If the length of a vector is less than this tolerance, we just set the output to zero. */
    itkSetMacro(LengthTolerance, double);
    itkGetMacro(LengthTolerance, double);
    
  protected:
    NormaliseVectorFilter();
    ~NormaliseVectorFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, ThreadIdType threadId);
    
  private:
    
    /**
     * Prohibited copy and assignment. 
     */
    NormaliseVectorFilter(const Self&); 
    void operator=(const Self&); 
    
    /** Flag to turn normalisation on/off. */
    bool m_Normalise;
    
    /** Set a tolerance on the length of the vector. */
    double m_LengthTolerance;
    
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNormaliseVectorFilter.txx"
#endif

#endif
