/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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
