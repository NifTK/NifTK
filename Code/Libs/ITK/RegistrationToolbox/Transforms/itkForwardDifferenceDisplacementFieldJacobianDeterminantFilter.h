/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-06-01 09:38:00 +0100 (Wed, 01 Jun 2011) $
 Revision          : $Revision: 6322 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKForwardDifferenceisplacementFieldJacobianDeterminantFilter_H_
#define ITKForwardDifferenceisplacementFieldJacobianDeterminantFilter_H_

#include "itkDisplacementFieldJacobianDeterminantFilter.h"

namespace itk
{
  
/** 
 * \class ForwardDifferenceDisplacementFieldJacobianDeterminantFilter
 * 
 */
template <typename TInputImage, typename TRealType, typename TOutputImage>
class ITK_EXPORT ForwardDifferenceDisplacementFieldJacobianDeterminantFilter :
  public DisplacementFieldJacobianDeterminantFilter<TInputImage, TRealType, TOutputImage>
{    
public:  
  /** 
   * Standard class typedefs. 
   */
  typedef ForwardDifferenceDisplacementFieldJacobianDeterminantFilter Self;
  typedef DisplacementFieldJacobianDeterminantFilter<TInputImage, TRealType, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  /** 
   * Method for creation through the object factory. 
   */
  itkNewMacro(Self);
  /** 
   * Run-time type information (and related methods) 
   */
  itkTypeMacro(ForwardDifferenceDisplacementFieldJacobianDeterminantFilter, DisplacementFieldJacobianDeterminantFilter);
  /** 
   * The dimensionality of the input and output images. 
   */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension);
  /** 
   * Length of the vector pixel type of the input image. 
   */
  itkStaticConstMacro(VectorDimension, unsigned int, TInputImage::PixelType::Dimension);
  /** 
   * Type of the iterator that will be used to move through the image.  Also
   * the type which will be passed to the evaluate function. 
   */
  typedef typename Superclass::ConstNeighborhoodIteratorType ConstNeighborhoodIteratorType;
  
protected:
  ForwardDifferenceDisplacementFieldJacobianDeterminantFilter() { }
  virtual ~ForwardDifferenceDisplacementFieldJacobianDeterminantFilter() { }
  /**
   * Override to use asymmetric forward difference instead of symmetric centered difference to 
   * calculate the Jacobian. 
   */
  virtual TRealType EvaluateAtNeighborhood(const ConstNeighborhoodIteratorType& it) const;
  
private:
  ForwardDifferenceDisplacementFieldJacobianDeterminantFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  
}; 
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkForwardDifferenceDisplacementFieldJacobianDeterminantFilter.txx"
#endif


#endif 



