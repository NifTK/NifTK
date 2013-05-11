/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBSplineOperator_h
#define __itkBSplineOperator_h

#include <itkNeighborhoodOperator.h>
#include <math.h>

namespace itk {

/**
 * \class BSplineOperator
 * \brief A NeighborhoodOperator whose coefficients are a one
 * dimensional, discrete BSpline kernel.
 *
 * \sa NeighborhoodOperator
 * \sa NeighborhoodIterator
 * \sa Neighborhood
 * \sa GaussianOperator
 * 
 * \ingroup Operators
 */
template<class TPixel,unsigned int VDimension=2,
  class TAllocator = NeighborhoodAllocator<TPixel> >
class ITK_EXPORT BSplineOperator
  : public NeighborhoodOperator<TPixel, VDimension, TAllocator>
{
public:
  
  /** Standard class typedefs. */
  typedef BSplineOperator                                         Self;
  typedef NeighborhoodOperator<TPixel, VDimension, TAllocator>    Superclass;

  /** Constructor. */
  BSplineOperator() { this->m_Spacing = 1.0; };

  /** Copy constructor */
  BSplineOperator(const Self &other)
    : NeighborhoodOperator<TPixel, VDimension, TAllocator>(other)
    {
      m_Spacing = other.m_Spacing;
    }

  /** Assignment operator */
  Self &operator=(const Self &other)
  {
    Superclass::operator=(other);
    m_Spacing = other.m_Spacing;
    return *this;
  }

  /** Set/Get Spacing. */
  void SetSpacing(double spacing) { this->m_Spacing = spacing; }
  double GetSpacing() { return this->m_Spacing; }
  
protected:

  typedef typename Superclass::CoefficientVector CoefficientVector;

  /** Calculates operator coefficients. */
  CoefficientVector GenerateCoefficients();

  /** Arranges coefficients spatially in the memory buffer. */
  void Fill(const CoefficientVector& coeff)
    {
      this->FillCenteredDirectional(coeff);  
    }

private:

  /** Desired pixel size */
  double m_Spacing;

  /** For compatibility with itkWarningMacro */
  const char *GetNameOfClass() { return "itkBSplineOperator"; }

};

template <class TPixel, unsigned int VDimension, class TContainer>
std::ostream & operator<<(std::ostream &os, const BSplineOperator<TPixel,VDimension,TContainer> &bspline)
{
  os << "BSplineOperator:" << std::endl;
  os << "    Radius:" << bspline.GetRadius() << std::endl;
  os << "    Size:" << bspline.GetSize() << std::endl;
  os << "    DataBuffer:" << bspline.GetBufferReference() << std::endl;

  for (unsigned int i = 0; i < bspline.Size(); i++)
    {
      os << "       " << i << ":" << bspline[i] << std::endl;  
    }
  return os;
}

} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineOperator.txx"
#endif

#endif
