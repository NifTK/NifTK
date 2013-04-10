/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDisplacementFieldJacobianFilter_h
#define __itkDisplacementFieldJacobianFilter_h

#include "itkImageToImageFilter.h"

namespace itk {
/** 
 * \class DisplacementFieldJacobianFilter
 * \brief This class calculates the Jacobian matrix of a deformation field. 
 */
template <class InputScalarType, class OutputScalarType, unsigned int NDimensions>  
class DisplacementFieldJacobianFilter :
  public ImageToImageFilter< Image< Vector<InputScalarType, NDimensions>,  NDimensions>, 
                             Image< Matrix<OutputScalarType, NDimensions, NDimensions>, NDimensions> >
{
public:
  /** 
   * Standard "Self" typedef. 
   */
  typedef DisplacementFieldJacobianFilter Self;
  typedef ImageToImageFilter< Image< Vector<InputScalarType, NDimensions>,  NDimensions>,
                              Image< Matrix<OutputScalarType, NDimensions, NDimensions>, NDimensions> > Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);
  /** 
   * Run-time type information (and related methods). 
   */
  itkTypeMacro(DisplacementFieldJacobianFilter, ImageToImageFilter);
  /** 
   * Get the number of dimensions we are working in. 
   */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);
  /** 
   * Standard typedefs. 
   */
  typedef Matrix<OutputScalarType, NDimensions, NDimensions> OutputPixelType;
  typedef Image<OutputPixelType, NDimensions> OutputImageType;
  typedef typename Superclass::InputImageType InputImageType;
  typedef Image<OutputScalarType, NDimensions> OutputDeterminantImageType; 
  /**
   * Get/Set. 
   */
  typename OutputDeterminantImageType::Pointer GetDeterminant() { return this->m_Determinant;  }
  
protected:
  DisplacementFieldJacobianFilter() {} 
  ~DisplacementFieldJacobianFilter() {}
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
  DisplacementFieldJacobianFilter(const Self&); 
  void operator=(const Self&); 
  
}; 

} // End of namespace. 

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDisplacementFieldJacobianFilter.txx"
#endif
  
#endif  

             
             
