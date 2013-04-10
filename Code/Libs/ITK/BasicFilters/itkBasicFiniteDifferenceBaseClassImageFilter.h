/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBasicFiniteDifferenceBaseClassImageFilter_h
#define __itkBasicFiniteDifferenceBaseClassImageFilter_h
#include "itkImageToImageFilter.h"

namespace itk
{

/**
 * \class BasicFiniteDifferenceBaseClassImageFilter
 * \brief Abstract base class to provide first, second, mixed derivatives, which
 * can be subclassed for things like calculating Mean / Gaussian curvature.
 *
 * \sa itkGaussianCurvatureImageFilter
 * \sa itkMeanCurvatureImageFilter
 * \sa itkMinimumCurvatureImageFilter
 * \sa itkMaximumCurvatureImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT BasicFiniteDifferenceBaseClassImageFilter
  : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BasicFiniteDifferenceBaseClassImageFilter         Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage>     Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;
  typedef typename TInputImage::IndexType                   IndexType;
  typedef typename TInputImage::SizeType                    SizeType;
  typedef typename TInputImage::RegionType                  ImageRegionType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(BasicFiniteDifferenceBaseClassImageFilter, ImageToImageFilter);

  /** Print internal ivars */
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TInputImage::ImageDimension);

protected:

  BasicFiniteDifferenceBaseClassImageFilter();
  virtual ~BasicFiniteDifferenceBaseClassImageFilter();

  /** Check before we start - fill image with zeros. */
  virtual void BeforeThreadedGenerateData();

  /** Calculates the first derivative in the given dimension, at the given voxel location. */
  double d(int dimension, IndexType& location, TInputImage* image);

  /** Calculates the second derivative in the given dimension, at the given voxel location. */
  double dd(int dimension, IndexType& location, TInputImage* image);

  /** Calculates the mixed partial derivative in the two dimensions, at the given voxel location. */
  double dd(int dimension1, int dimension2, IndexType& location, TInputImage* image);

  /** Returns a new region, that is not too close to the image. */
  ImageRegionType CheckAndAdjustRegion(const ImageRegionType& region, TInputImage* image);

private:

  BasicFiniteDifferenceBaseClassImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBasicFiniteDifferenceBaseClassImageFilter.txx"
#endif

#endif
