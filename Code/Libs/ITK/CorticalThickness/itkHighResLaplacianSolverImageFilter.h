/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkHighResLaplacianSolverImageFilter_h
#define itkHighResLaplacianSolverImageFilter_h

#include "itkLaplacianSolverImageFilter.h"
#include <itkNearestNeighborInterpolateImageFunction.h>
#include "itkFiniteDifferenceVoxel.h"

namespace itk
{
/**
 * \class HighResLaplacianSolverImageFilter
 * \brief Solves Laplace equation over the cortical volume, but can be run at
 * very high resolution, by setting the VoxelMultiplicationFactor to determine 
 * an effective voxel size.
 * 
 * So, if you set a VoxelMultiplicationFactor to 3, you have 3 times as many 
 * voxels in each dimension, and hence 1/3 the voxel dimension in each dimension.
 * 
 * The output is an image of voltage potentials, of the same size as the input.
 * In addition, we expose the internal list of pixels used in the high resolution
 * computations, so that subsequent pipeline steps can have access to them.
 * 
 * \ingroup ImageFeatureExtraction */
template <class TInputImage, typename TScalarType=double>
class ITK_EXPORT HighResLaplacianSolverImageFilter : 
    public LaplacianSolverImageFilter< TInputImage >
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef HighResLaplacianSolverImageFilter         Self;
  typedef LaplacianSolverImageFilter< TInputImage > Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;

  /** Run-time type information (and related methods)  */
  itkTypeMacro(HighResLaplacianSolverImageFilter, LaplacianSolverImageFilter);
  
  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  typedef TScalarType                                         OutputPixelType;
  typedef typename TInputImage::PixelType                     InputPixelType;
  
  /** Image typedef support. */
  typedef TInputImage                                         InputImageType;
  typedef typename InputImageType::Pointer                    InputImagePointer;
  typedef typename InputImageType::IndexType                  InputImageIndexType;
  typedef typename InputImageType::PixelType                  InputImagePixelType;
  typedef typename InputImageType::RegionType                 InputImageRegionType;
  typedef typename InputImageType::SizeType                   InputImageSizeType;
  typedef typename InputImageType::PointType                  InputImageOriginType;
  typedef typename InputImageType::PointType                  InputImagePointType;
  typedef typename InputImageType::SpacingType                InputImageSpacingType;
  typedef Image<OutputPixelType, TInputImage::ImageDimension> OutputImageType;
  typedef typename OutputImageType::Pointer                   OutputImagePointer;
  typedef typename OutputImageType::SpacingType               OutputImageSpacing;
  typedef typename OutputImageType::RegionType                OutputImageRegionType;
  typedef typename OutputImageType::SizeType                  OutputImageSizeType;
  typedef typename OutputImageType::IndexType                 OutputImageIndexType;
  typedef typename OutputImageType::DirectionType             OutputImageDirectionType;
  typedef typename OutputImageType::PointType                 OutputImageOriginType;
  typedef ContinuousIndex<TScalarType, InputImageType::ImageDimension> ContinuousIndexType;
  typedef Point<TScalarType, InputImageType::ImageDimension> PointType;
  typedef itk::NearestNeighborInterpolateImageFunction< TInputImage, TScalarType> NearestNeighbourInterpolatorType;
  typedef FiniteDifferenceVoxel<InputImageType::ImageDimension, 2, InputImagePixelType, InputImagePixelType> FiniteDifferenceVoxelType;
  typedef std::map<unsigned long int, FiniteDifferenceVoxelType*> MapType;
  typedef std::pair<unsigned long int, FiniteDifferenceVoxelType*> PairType;
  typedef typename MapType::const_iterator IteratorType;
  
  /** Set/Get the VoxelMultiplicationFactor. */
  itkSetMacro(VoxelMultiplicationFactor, int);
  itkGetMacro(VoxelMultiplicationFactor, int);
  
  /** To expose the internal high res map. */
  MapType* GetMapOfVoxels() { return &m_MapOfVoxels; }
  
protected:
  HighResLaplacianSolverImageFilter();
  virtual ~HighResLaplacianSolverImageFilter();

  /** Standard Print Self. */
  virtual void PrintSelf(std::ostream&, Indent) const;

  // The main filter method. Note, single threaded.
  virtual void GenerateData();
  
private:
  HighResLaplacianSolverImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
    
  /** Controls how many more voxels we use. */
  int m_VoxelMultiplicationFactor;
  
  bool GetValue(InputImageType* highResImage, 
      InputImageType* lowResImage, 
      NearestNeighbourInterpolatorType* interpolator, 
      InputImageIndexType& index,
      InputImagePixelType& result);
  
  void InsertNeighbour(
      FiniteDifferenceVoxelType* greyVoxel,
      InputImagePixelType& neighbourValue,  
      unsigned long int& mapIndexOfVoxel,
      InputImageIndexType& itkImageIndexOfVoxel,
      MapType& map,
      unsigned long int& numberOfDuplicates,
      unsigned long int& numberOfBoundaryPoints
      );
  
  MapType m_MapOfVoxels;
  
  float m_Tolerance;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHighResLaplacianSolverImageFilter.txx"
#endif

#endif
