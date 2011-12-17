/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkNodeScalarGradientCalculator.h,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkResampleDestinationPointsQuadEdgeMeshFilter_h
#define __itkResampleDestinationPointsQuadEdgeMeshFilter_h

#include "itkMeshToMeshFilter.h"
#include "itkLinearInterpolateDeformationFieldMeshFunction.h"
#include "itkTransform.h"

namespace itk
{

/**
 * \class ResampleDestinationPointsQuadEdgeMeshFilter
 * \brief This filter resamples a collection of destination points.
 *
 * This filter takes as input a PointSet, and a fixed Mesh, and assumes that
 * the points in the PointSet are one-to-one destination points for the points
 * in the fixed Mesh. Then, it computes via linear interpolation the destination
 * points that would correspond to the locations indicated by the points of the
 * reference mesh.
 *
 * \ingroup MeshFilters
 *
 */
template< class TInputPointSet, class TFixedMesh, class TReferenceMesh, class TOutputPointSet >
class ResampleDestinationPointsQuadEdgeMeshFilter :
  public MeshToMeshFilter< TInputPointSet, TOutputPointSet >
{
public:
  typedef ResampleDestinationPointsQuadEdgeMeshFilter   Self;
  typedef MeshToMeshFilter< 
    TInputPointSet, TOutputPointSet >                   Superclass;
  typedef SmartPointer< Self >                          Pointer;
  typedef SmartPointer< const Self >                    ConstPointer;

  /** Run-time type information (and related methods).   */
  itkTypeMacro( ResampleDestinationPointsQuadEdgeMeshFilter, MeshToMeshFilter );

  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro( Self );

  typedef TInputPointSet                                     InputPointSetType;
  typedef typename InputPointSetType::Pointer                InputPointSetPointer;
  typedef typename InputPointSetType::PointsContainer        InputPointsContainer;

  typedef TFixedMesh                                         FixedMeshType;
  typedef typename FixedMeshType::PointType                  FixedMeshPointType;

  typedef TReferenceMesh                                     ReferenceMeshType;
  typedef typename ReferenceMeshType::PointsContainer        ReferencePointsContainer;
  typedef typename ReferencePointsContainer::ConstIterator   ReferencePointsContainerConstIterator;

  typedef TOutputPointSet                                    OutputPointSetType;
  typedef typename OutputPointSetType::Pointer               OutputPointSetPointer;
  typedef typename OutputPointSetType::ConstPointer          OutputPointSetConstPointer;
  typedef typename OutputPointSetType::PointType             OutputPointType;
  typedef typename OutputPointSetType::PointsContainer       OutputPointsContainer;

  typedef typename OutputPointSetType::PointsContainerConstPointer    OutputPointsContainerConstPointer;
  typedef typename OutputPointSetType::PointsContainerPointer         OutputPointsContainerPointer;
  typedef typename OutputPointSetType::PointsContainerIterator        OutputPointsContainerIterator;

  itkStaticConstMacro( PointDimension, unsigned int, OutputPointSetType::PointDimension );

  /** Transform typedef. */
  typedef Transform<double, 
    itkGetStaticConstMacro(PointDimension), 
    itkGetStaticConstMacro(PointDimension)>         TransformType;
  typedef typename TransformType::ConstPointer      TransformPointerType;

  /** Interpolator typedef. */
  typedef LinearInterpolateDeformationFieldMeshFunction< 
    ReferenceMeshType, InputPointsContainer >             InterpolatorType;
  typedef typename InterpolatorType::Pointer              InterpolatorPointerType;

  /** Set Mesh whose grid defines the geometry and topology of the input PointSet.
   *  In a multi-resolution registration scenario, this will typically be the Fixed
   *  mesh at the current higher resolution level. */
  void SetFixedMesh ( const FixedMeshType * mesh );
  const FixedMeshType * GetFixedMesh( void ) const;

  /** Set Mesh whose grid will define the geometry of the output PointSet.
   *  In a multi-resolution registration scenario, this will typically be 
   *  the Fixed mesh at the next higher resolution level. */
  void SetReferenceMesh ( const ReferenceMeshType * mesh );
  const ReferenceMeshType * GetReferenceMesh( void ) const;

  /** Set the coordinate transformation.
   * Set the coordinate transform to use for resampling.  Note that this must
   * be in physical coordinates and it is the output-to-input transform, NOT
   * the input-to-output transform that you might naively expect.  By default
   * the filter uses an Identity transform. You must provide a different
   * transform here, before attempting to run the filter, if you do not want to
   * use the default Identity transform. */
  itkSetConstObjectMacro( Transform, TransformType ); 

  /** Get a pointer to the coordinate transform. */
  itkGetConstObjectMacro( Transform, TransformType );

  /** Set the interpolator function.  The default is
   * itk::LinearInterpolateMeshFunction<InputPointSetType, TInterpolatorPrecisionType>. Some
   * other options are itk::NearestNeighborInterpolateMeshFunction
   * (useful for binary masks and other images with a small number of
   * possible pixel values), and itk::BSplineInterpolateMeshFunction
   * (which provides a higher order of interpolation).  */
  itkSetObjectMacro( Interpolator, InterpolatorType );

  /** Get a pointer to the interpolator function. */
  itkGetConstObjectMacro( Interpolator, InterpolatorType );


protected:
  ResampleDestinationPointsQuadEdgeMeshFilter();
  ~ResampleDestinationPointsQuadEdgeMeshFilter();

  void GenerateData();

private:

  ResampleDestinationPointsQuadEdgeMeshFilter( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented


  TransformPointerType     m_Transform;         // Coordinate transform to use
  InterpolatorPointerType  m_Interpolator;      // Image function for

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkResampleDestinationPointsQuadEdgeMeshFilter.txx"
#endif

#endif
