/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSetOutputVectorToCurrentPositionFilter_h
#define itkSetOutputVectorToCurrentPositionFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkVector.h>
#include <itkImage.h>

namespace itk {
/**
 * \class SetOutputVectorToCurrentPositionFilter
 * \brief This class takes a vector image as input, and outputs a vector image, where
 * the vector data value at each voxel is equal to the millimetre or voxel position of that voxel.
 * 
 * This class is also subclassing InPlaceImageFilter meaning that the input buffer
 * and output buffer can be the same. If you are using this filter in place, you
 * run the filter, then us the output pointer, not the input pointer.
 * 
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT SetOutputVectorToCurrentPositionFilter : 
public InPlaceImageFilter<
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input image
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                         >
{
  public:

    /** Standard "Self" typedef. */
    typedef SetOutputVectorToCurrentPositionFilter                                      Self;
    typedef InPlaceImageFilter<Image< Vector<TScalarType, NDimensions>,  NDimensions>,
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
    typedef typename OutputImageType::PointType                                         OutputImagePointType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(SetOutputVectorToCurrentPositionFilter, InPlaceImageFilter);

    /** Get the number of dimensions we are working in. */
    itkStaticConstMacro(Dimension, unsigned int, NDimensions);

    /** Set/Get a flag controlling output, if true, output is millimetre coordinates, if false, output is in voxels. */
    itkSetMacro(OutputIsInMillimetres, bool);
    itkGetMacro(OutputIsInMillimetres, bool);
    
  protected:
    SetOutputVectorToCurrentPositionFilter();
    ~SetOutputVectorToCurrentPositionFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    // Check before we start.
    virtual void BeforeThreadedGenerateData();
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);
    
  private:
    
    /**
     * Prohibited copy and assignment. 
     */
    SetOutputVectorToCurrentPositionFilter(const Self&); 
    void operator=(const Self&); 
    
    /** Default to true, so output is millimetre coordinates, with respect to origin. If false, output is voxel coordinates. */
    bool m_OutputIsInMillimetres;
    
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSetOutputVectorToCurrentPositionFilter.txx"
#endif

#endif
