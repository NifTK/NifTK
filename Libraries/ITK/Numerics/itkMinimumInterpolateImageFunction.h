/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMinimumInterpolateImageFunction_h
#define __itkMinimumInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"
#include "itkTransform.h"

namespace itk
{
/** \class MinimumInterpolateImageFunction
 * \brief Minimum interpolation of an image at specified positions.
 *
 * \ingroup ImageFunctions
 * \ingroup ITKImageFunction
 */
template< class TInputImage, class TCoordRep = double >
class ITK_EXPORT MinimumInterpolateImageFunction :
  public InterpolateImageFunction< TInputImage, TCoordRep >
{
public:
  /** Standard class typedefs. */
  typedef MinimumInterpolateImageFunction                    Self;
  typedef InterpolateImageFunction< TInputImage, TCoordRep > Superclass;
  typedef SmartPointer< Self >                               Pointer;
  typedef SmartPointer< const Self >                         ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MinimumInterpolateImageFunction, InterpolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType typedef support. */
  typedef typename Superclass::OutputType OutputType;

  /** InputImageType typedef support. */
  typedef typename Superclass::InputImageType InputImageType;

  /** InputPixelType typedef support. */
  typedef typename Superclass::InputPixelType InputPixelType;

  /** RealType typedef support. */
  typedef typename Superclass::RealType RealType;

  /** Dimension underlying input image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

  /** Index typedef support. */
  typedef typename Superclass::IndexType      IndexType;

  /** ContinuousIndex typedef support. */
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  /** Point typedef support. */
  typedef typename Superclass::PointType PointType;


  /** Evalulate the function at specified index */
  virtual RealType EvaluateAtIndex(const IndexType & index) const;

  /** Evaluate the function at non-integer positions */
  virtual RealType Evaluate(const PointType & point) const
  {
    IndexType index;

    this->ConvertPointToNearestIndex(point, index);
    return this->EvaluateAtIndex(index);
  }

  virtual RealType EvaluateAtContinuousIndex(
    const ContinuousIndexType & cindex) const
  {
    IndexType index;

    this->ConvertContinuousIndexToNearestIndex(cindex, index);
    return this->EvaluateAtIndex(index);
  }

  /** Get/Set the radius of the neighborhood over which the
      statistics are evaluated */
  itkSetMacro(NeighborhoodRadius, unsigned int);
  itkGetConstReferenceMacro(NeighborhoodRadius, unsigned int);

protected:

  /// Constructor
  MinimumInterpolateImageFunction();

  /// Destructor
  ~MinimumInterpolateImageFunction();

  /// Print the object
  void PrintSelf(std::ostream & os, Indent indent) const;

private:
  MinimumInterpolateImageFunction(const Self &); //purposely not implemented
  void operator=(const Self &);                  //purposely not implemented

  unsigned int m_NeighborhoodRadius;

};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMinimumInterpolateImageFunction.txx"
#endif

#endif
