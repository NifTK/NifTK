/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkDisplacementFieldJacobianVectorFilter_h
#define itkDisplacementFieldJacobianVectorFilter_h

#include <itkImageToImageFilter.h>

namespace itk {
/** 
 * \class DisplacementFieldJacobianVectorFilter
 * \brief This class calculates the Jacobian matrix of a deformation field and store in a row-major vector. 
 */
template <class InputScalarType, class OutputScalarType, unsigned int NDimensions, unsigned int NJDimensions>  
class DisplacementFieldJacobianVectorFilter :
  public ImageToImageFilter< Image< Vector<InputScalarType, NDimensions>,  NDimensions>, 
                             Image< Vector<OutputScalarType, NJDimensions>, NDimensions> >
{
public:
  /** 
   * Standard "Self" typedef. 
   */
  typedef DisplacementFieldJacobianVectorFilter Self;
  typedef ImageToImageFilter< Image< Vector<InputScalarType, NDimensions>,  NDimensions>,
                              Image< Vector<OutputScalarType, NJDimensions>, NDimensions> > Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);
  /** 
   * Run-time type information (and related methods). 
   */
  itkTypeMacro(DisplacementFieldJacobianVectorFilter, ImageToImageFilter);
  /** 
   * Get the number of dimensions we are working in. 
   */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);
  /** 
   * Standard typedefs. 
   */
  typedef Vector<OutputScalarType, NJDimensions> OutputPixelType;
  typedef Image<OutputPixelType, NDimensions> OutputImageType;
  typedef typename Superclass::InputImageType InputImageType;
  typedef Image<OutputScalarType, NDimensions> OutputDeterminantImageType; 
  /**
   * Get/Set. 
   */
  typename OutputDeterminantImageType::Pointer GetDeterminant() { return this->m_Determinant;  }
  
protected:
  DisplacementFieldJacobianVectorFilter() {} 
  ~DisplacementFieldJacobianVectorFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const {}
  /**
   * The main filter method. 
   */
  virtual void GenerateData();
  /**
   * The Jacobian determinant. 
   */
  typename OutputDeterminantImageType::Pointer m_Determinant; 

private:
  /**
   * Prohibited copy and assignment. 
   */
  DisplacementFieldJacobianVectorFilter(const Self&); 
  void operator=(const Self&); 
  
}; 

} // End of namespace. 

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDisplacementFieldJacobianVectorFilter.txx"
#endif
  
#endif  

             
             
