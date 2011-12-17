/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkNondirectionalDerivativeOperator_h
#define __itkNondirectionalDerivativeOperator_h

#include "itkNeighborhood.h"
#include "itkVector.h"
#include "itkDerivativeOperator.h"

namespace itk {

/**
 * \class SingleDerivativeTermInfo
 * \brief Information about a single derivative term (derivative order and the multiplicative term).
 */
template< unsigned int VDimension=2 > 
class SingleDerivativeTermInfo
{
public:
  /**
   * Derivative order in each dimension. 
   */
  typedef Vector < unsigned int, VDimension > DerivativeOrderType; 
  /**
   * Constructor.
   */  
  SingleDerivativeTermInfo() { m_Constant = 1.0; }
  /**
   * Destructor. 
   */
  ~SingleDerivativeTermInfo() {}; 
  /**
   * Get/Set derivative order. 
   */
  void SetDervativeOrder(const DerivativeOrderType& order) { this->m_DervativeOrder = order; }
  const DerivativeOrderType& GetDerivativeOrder() const { return this->m_DervativeOrder; }
  /**
   * Get/Set constant.
   */
  void SetConstant(double constant) { this->m_Constant = constant; }
  double GetConstant(void) const { return this->m_Constant; }

protected:
  /**
   * The order of the derivative in each dimension. 
   */
  DerivativeOrderType m_DervativeOrder; 
  /**
   * The constant multiplicative term. 
   */
  double m_Constant;
};

/**
 * \class NondirectionalDerivativeOperator
 * \brief Construct a multi-directional n-th derivative operator. 
 *
 * \ingroup Operators
 */
template<class TPixel,unsigned int VDimension=2,
  class TAllocator = NeighborhoodAllocator<TPixel> >
class ITK_EXPORT NondirectionalDerivativeOperator
  : public Neighborhood<TPixel, VDimension, TAllocator>
{
public:
  /** 
   * Standard class typedefs. 
   */
  typedef NondirectionalDerivativeOperator Self;
  typedef Neighborhood< TPixel, VDimension, TAllocator > Superclass;
  typedef SingleDerivativeTermInfo < VDimension > SingleDerivativeTermInfoType; 
  itkTypeMacro(NondirectionalDerivativeOperator, Neighborhood);
  
  NondirectionalDerivativeOperator();
  virtual ~NondirectionalDerivativeOperator() { };
  /**
   * Add a derivative term to the operator. 
   */
  void AddSingleDerivativeTerm(const SingleDerivativeTermInfoType& term) { this->m_DervativeTermInfo.push_back(term); }
  /**
   * Clear all the derivative terms. 
   */
  void ClearDerivativeTerm(void) { this->m_DervativeTermInfo.clear(); }
  /**
   * Create the operator. 
   */
  void CreateToRadius(unsigned int radius);
  
protected:
  typedef DerivativeOperator< TPixel, VDimension, TAllocator > DerivativeOperatorType; 

protected:
  /** 
   * The order of derivative in each dimension. 
   */
  std::vector< SingleDerivativeTermInfoType > m_DervativeTermInfo; 
  
private:
};

}
#endif

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNondirectionalDerivativeOperator.txx"
#endif

