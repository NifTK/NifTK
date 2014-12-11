/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkThinPlateSplineScatteredDataPointSetToImageFilter_h
#define __itkThinPlateSplineScatteredDataPointSetToImageFilter_h

#include <itkPointSetToImageFilter.h>
#include <itkThinPlateR2LogRSplineKernelTransform.h>

#include <itkVectorContainer.h>

#include <vnl/vnl_matrix.h>

namespace itk
{
/** \class ThinPlateSplineScatteredDataPointSetToImageFilter
 * \brief Image filter which provides a thin plate spline mask approximation to a set of landmarks.
 *
 * \sa LandmarkDisplacementFieldSource
 */

template< typename TInputPointSet, typename TOutputImage >
class ThinPlateSplineScatteredDataPointSetToImageFilter:
  public PointSetToImageFilter< TInputPointSet, TOutputImage >
{
public:
  typedef ThinPlateSplineScatteredDataPointSetToImageFilter     Self;
  typedef PointSetToImageFilter<TInputPointSet, TOutputImage>   Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

   /** Extract dimension from the output image. */
  itkStaticConstMacro( ImageDimension, unsigned int, TOutputImage::ImageDimension );

  typedef TInputPointSet                                 LandmarkPointSetType;


 /** Image typedef support. */
  typedef TOutputImage                                   OutputImageType;
  typedef typename OutputImageType::Pointer              OutputImagePointer;
  typedef typename OutputImageType::RegionType           OutputImageRegionType;

  typedef typename OutputImageType::PixelType            OutputPixelType;
  typedef typename OutputImageType::SizeType             OutputSizeType;
  typedef typename OutputImageType::IndexType            OutputIndexType;

  typedef typename OutputImageType::SpacingType          SpacingType;
  typedef typename OutputImageType::PointType            OriginPointType;
  typedef typename OutputImageType::DirectionType        DirectionType;

  typedef typename LandmarkPointSetType::PointType          LandmarkPointType;
  typedef typename LandmarkPointSetType::PointDataContainer PointDataContainerType;

  typedef typename LandmarkPointSetType::CoordRepType    CoordRepType;

  /** The KernelBased spline transform type. */
  typedef ThinPlateR2LogRSplineKernelTransform< CoordRepType, itkGetStaticConstMacro(ImageDimension) > KernelTransformType;
  typedef typename KernelTransformType::Pointer          KernelTransformPointerType;
  typedef typename KernelTransformType::PointsContainer  LandmarkContainer;

  typedef typename LandmarkContainer::ConstPointer       LandmarkContainerPointer;


  /** Get/Set the coordinate transformation.
   * Set the KernelBase spline used for resampling the displacement grid.
   * */
  itkSetObjectMacro(KernelTransform, KernelTransformType);
  itkGetModifiableObjectMacro(KernelTransform, KernelTransformType);

  itkSetMacro(Stiffness, double);
  itkGetMacro(Stiffness, double);


  /** Method Compute the Modified Time based on changed to the components. */
  ModifiedTimeType GetMTime(void) const;


protected:
  ThinPlateSplineScatteredDataPointSetToImageFilter();
  virtual ~ThinPlateSplineScatteredDataPointSetToImageFilter() {};

  void PrintSelf(std::ostream & os, Indent indent) const;

  /**
   * GenerateData() computes the internal KernelBase spline.
   */
  void GenerateData();

  void PrepareKernelBaseSpline();

private:

  //purposely not implemented
  ThinPlateSplineScatteredDataPointSetToImageFilter( const Self & );
  void operator=( const Self & );

  KernelTransformPointerType m_KernelTransform;      // Coordinate transform to use

  /// The spline stiffness
  double m_Stiffness;

};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkThinPlateSplineScatteredDataPointSetToImageFilter.txx"
#endif

#endif
