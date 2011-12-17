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
#ifndef __itkGaussianSmoothVectorFieldFilter_h
#define __itkGaussianSmoothVectorFieldFilter_h
#include "itkSmoothVectorFieldFilter.h"
#include "itkGaussianOperator.h"
#include "itkFixedArray.h"

namespace itk {
/** 
 * \class GaussianSmoothVectorFieldFilter
 * \brief Class that takes a vector field, and applies Gaussian smoothing.
 *
 * As of 16/01/2010, I have extended this to cope with time varying vector fields.
 * This means that you have an image dimension and a vector dimension, which can be
 * different. For example, you could have images of 2D vectors (x,y), but the image
 * could be 2D, or for example 3D (time varying vector field).
 */
template <
    class TScalarType,                     // Data type for the vectors
    unsigned int NumberImageDimensions,    // Number of image dimensions
    unsigned int NumberVectorDimensions>   // Number of dimensions in the vectors
class ITK_EXPORT GaussianSmoothVectorFieldFilter :
  public SmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
{
public:

  /** Standard "Self" typedef. */
  typedef GaussianSmoothVectorFieldFilter                      Self;
  typedef SmoothVectorFieldFilter<TScalarType,
                                 NumberImageDimensions,
                                 NumberVectorDimensions>       Superclass;
  typedef SmartPointer<Self>                                   Pointer;
  typedef SmartPointer<const Self>                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(GaussianSmoothVectorFieldFilter, SmoothVectorFieldFilter);

  /** Standard typedefs. */
  typedef typename Superclass::NeighborhoodOperatorType            NeighborhoodOperatorType;
  typedef GaussianOperator<TScalarType, NumberImageDimensions>     GaussianOperatorType;
  typedef FixedArray<TScalarType, NumberImageDimensions>           SigmaType;
  
  /** Set/Get the Sigma to apply. */
  itkGetMacro(Sigma, SigmaType);
  itkSetMacro(Sigma, SigmaType);
  
protected:

  GaussianSmoothVectorFieldFilter();
  ~GaussianSmoothVectorFieldFilter() {};
  
  /** Create the gaussian operator. Called by baseclass. */
  NeighborhoodOperatorType* CreateOperator(int dimension);

private:

  /**
   * Prohibited copy and assingment. 
   */
  GaussianSmoothVectorFieldFilter(const Self&); 
  void operator=(const Self&); 

  /** To store sigma. */
  SigmaType m_Sigma;
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGaussianSmoothVectorFieldFilter.txx"
#endif

#endif
