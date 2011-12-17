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
#ifndef __itkHighResRelaxStreamlinesFilter_h
#define __itkHighResRelaxStreamlinesFilter_h

#include "itkLagrangianInitializedRelaxStreamlinesFilter.h"
#include "itkHighResLaplacianSolverImageFilter.h"
#include "itkVectorInterpolateImageFunction.h"

namespace itk {
/** 
 * \class HighResRelaxStreamlinesFilter
 * \brief Prototype high res version of RelaxStreamlines Filter.
 * 
 * \sa RelaxStreamlinesFilter
 * \sa LagrangianInitializedRelaxStreamlinesFilter
 */
template < class TImageType, typename TScalarType, unsigned int NDimensions> 
class ITK_EXPORT HighResRelaxStreamlinesFilter :
  public LagrangianInitializedRelaxStreamlinesFilter< TImageType, TScalarType, NDimensions>
{
public:

  /** Standard "Self" typedef. */
  typedef HighResRelaxStreamlinesFilter                                                      Self;
  typedef LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>  Superclass;
  typedef SmartPointer<Self>                                                                 Pointer;
  typedef SmartPointer<const Self>                                                           ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RelaxStreamlinesFilter, LagrangianInitializedRelaxStreamlinesFilter);

  /** Standard typedefs. */
  typedef typename Superclass::InputVectorImagePixelType      InputVectorImagePixelType;
  typedef typename Superclass::InputVectorImageType           InputVectorImageType;
  typedef typename Superclass::InputVectorImagePointer        InputVectorImagePointer;
  typedef typename Superclass::InputVectorImageConstPointer   InputVectorImageConstPointer;
  typedef typename Superclass::InputVectorImageIndexType      InputVectorImageIndexType; 
  typedef typename Superclass::InputScalarImagePixelType      InputScalarImagePixelType;
  typedef typename Superclass::InputScalarImageType           InputScalarImageType;
  typedef typename Superclass::InputScalarImagePointType      InputScalarImagePointType;
  typedef typename Superclass::InputScalarImagePointer        InputScalarImagePointer;
  typedef typename Superclass::InputScalarImageIndexType      InputScalarImageIndexType;
  typedef typename Superclass::InputScalarImageConstPointer   InputScalarImageConstPointer;
  typedef typename Superclass::InputScalarImageRegionType     InputScalarImageRegionType;
  typedef typename Superclass::InputScalarImageSpacingType    InputScalarImageSpacingType;
  typedef typename InputScalarImageType::SizeType             InputScalarImageSizeType;
  typedef typename InputScalarImageType::PointType            InputScalarImageOriginType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename Superclass::OutputImagePixelType           OutputImagePixelType;
  typedef typename Superclass::OutputImagePointer             OutputImagePointer;
  typedef typename Superclass::OutputImageConstPointer        OutputImageConstPointer;
  typedef typename Superclass::OutputImageIndexType           OutputImageIndexType;
  typedef typename Superclass::OutputImageSpacingType         OutputImageSpacingType;
  typedef typename OutputImageType::RegionType                OutputImageRegionType;
  typedef typename OutputImageType::SizeType                  OutputImageSizeType;
  typedef typename OutputImageType::DirectionType             OutputImageDirectionType;
  typedef typename OutputImageType::PointType                 OutputImageOriginType;  
  typedef typename Superclass::VectorInterpolatorType         VectorInterpolatorType;
  typedef typename Superclass::VectorInterpolatorPointer      VectorInterpolatorPointer;
  typedef typename Superclass::VectorInterpolatorPointType    VectorInterpolatorPointType;
  typedef typename Superclass::ScalarInterpolatorType         ScalarInterpolatorType;
  typedef typename Superclass::ScalarInterpolatorPointer      ScalarInterpolatorPointer;
  typedef typename Superclass::ScalarInterpolatorPointType    ScalarInterpolatorPointType;
  typedef typename HighResLaplacianSolverImageFilter<TImageType, TScalarType>::MapType MapType;
  typedef typename HighResLaplacianSolverImageFilter<TImageType, TScalarType>::IteratorType MapIteratorType;
  typedef typename HighResLaplacianSolverImageFilter<TImageType, TScalarType>::PairType MapPairType;
  typedef typename HighResLaplacianSolverImageFilter<TImageType, TScalarType>::FiniteDifferenceVoxelType FiniteDifferenceVoxelType;
  typedef ContinuousIndex<TScalarType, TImageType::ImageDimension> ContinuousIndexType;
  typedef Point<TScalarType, TImageType::ImageDimension> PointType;
  
  /** Set the map from the Laplacian. */
  void SetHighResLaplacianMap(MapType *map) { m_LaplacianMap = map; }
  
  /** Set/Get the VoxelMultiplicationFactor. */
  itkSetMacro(VoxelMultiplicationFactor, int);
  itkGetMacro(VoxelMultiplicationFactor, int);

protected:
  HighResRelaxStreamlinesFilter();
  ~HighResRelaxStreamlinesFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Called once to initialize both boundaries. */
  virtual void IntializeBoundaries(
      InputScalarImageType* gmpvImage,
      InputVectorImageType* vectorImage
      );
  
  /** This is called twice, once for L0 boundary, and once for L1 boundary. */
  virtual void SolvePDE(
      int boundaryNumber,
      InputScalarImageSpacingType& virtualSpacing,
      InputScalarImageType* scalarImage,
      InputScalarImageType* gmpvImage,
      InputVectorImageType* vectorImage
      );

private:
  
  /**
   * Prohibited copy and assignment. 
   */
  HighResRelaxStreamlinesFilter(const Self&); 
  void operator=(const Self&); 

  // The main filter method. Note, single threaded.
  virtual void GenerateData();

  /** Pointer to the map in the Laplacian high res solver. */
  MapType *m_LaplacianMap;
  
  /** Pointer to thickness maps */
  MapType* m_L0L1;

  /** Controls how many more voxels we use. */
  int m_VoxelMultiplicationFactor;
  
  /** Vector interpolator. */
  VectorInterpolatorPointer m_VectorInterpolator;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHighResRelaxStreamlinesFilter.txx"
#endif

#endif
