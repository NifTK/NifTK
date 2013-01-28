/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-06-28 14:20:20 +0100 (Tue, 28 Jun 2011) $
 Revision          : $Revision: 6588 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkDisplacementFieldJacobianVectorFilter_h
#define __itkDisplacementFieldJacobianVectorFilter_h

#include "itkImageToImageFilter.h"

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

             
             
