/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkOrderedTraversalStreamlinesFilter_h
#define itkOrderedTraversalStreamlinesFilter_h

#include <itkImage.h>
#include <itkVector.h>
#include "itkBaseCTEStreamlinesFilter.h"
#include <itkVectorInterpolateImageFunction.h>
#include <itkInterpolateImageFunction.h>

#include <queue>

namespace itk {
/** 
 * \class OrderedTraversalStreamlinesFilter
 * \brief Calculates length between two boundaries, solving PDE by ordered traversal.
 * 
 * This filter implements algorithm 2) in Yezzi and Prince 2003 , IEEE TMI
 * Vol. 22, No. 10, p 1332-1339. The first input should be a scalar image,
 * such as the output of itkLaplacianSolverImageFilter. The second image 
 * should be the vector field of the gradient of the first input.
 * 
 * In this implementation, you specify the the voltage potentials that
 * your Laplacian was solved on. This enables the filter to set the 
 * boundaries correctly. Only voxels that are > LowVoltage and
 * < HighVoltage are solved.
 * 
 * \sa BaseStreamlinesFilter
 * \sa IntegrateStreamlinesFilter
 * \sa RelaxStreamlinesFilter
 */
template <  class TImageType, typename TScalarType=double, unsigned int NDimensions=3 > 
class ITK_EXPORT OrderedTraversalStreamlinesFilter :
  public BaseCTEStreamlinesFilter< TImageType, TScalarType, NDimensions>
{
public:

  /** Standard "Self" typedef. */
  typedef OrderedTraversalStreamlinesFilter                               Self;
  typedef BaseCTEStreamlinesFilter<TImageType, TScalarType, NDimensions>  Superclass;
  typedef SmartPointer<Self>                                              Pointer;
  typedef SmartPointer<const Self>                                        ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(OrderedTraversalStreamlinesFilter, BaseStreamlinesFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Standard typedefs. */
  typedef Vector< TScalarType, NDimensions >                     InputVectorImagePixelType;
  typedef Image< InputVectorImagePixelType, NDimensions >        InputVectorImageType;
  typedef typename InputVectorImageType::Pointer                 InputVectorImagePointer;
  typedef typename InputVectorImageType::ConstPointer            InputVectorImageConstPointer;
  typedef typename InputVectorImageType::IndexType               InputVectorImageIndexType; 
  typedef TScalarType                                            InputScalarImagePixelType;
  typedef Image< InputScalarImagePixelType, NDimensions >        InputScalarImageType;
  typedef typename InputScalarImageType::PointType               InputScalarImagePointType;
  typedef typename InputScalarImageType::Pointer                 InputScalarImagePointer;
  typedef typename InputScalarImageType::IndexType               InputScalarImageIndexType;
  typedef typename InputScalarImageType::ConstPointer            InputScalarImageConstPointer;
  typedef typename InputScalarImageType::RegionType              InputScalarImageRegionType;
  typedef typename InputScalarImageType::SizeType                InputScalarImageSizeType;
  typedef InputScalarImageType                                   OutputImageType;
  typedef typename OutputImageType::PixelType                    OutputImagePixelType;
  typedef typename OutputImageType::Pointer                      OutputImagePointer;
  typedef typename OutputImageType::ConstPointer                 OutputImageConstPointer;
  typedef typename OutputImageType::IndexType                    OutputImageIndexType;
  typedef typename OutputImageType::SpacingType                  OutputImageSpacingType;
  
  typedef VectorInterpolateImageFunction<InputVectorImageType
                                         ,TScalarType
                                        >                        VectorInterpolatorType;
  typedef typename VectorInterpolatorType::Pointer               VectorInterpolatorPointer;
  typedef typename VectorInterpolatorType::PointType             VectorInterpolatorPointType;
  typedef InterpolateImageFunction< InputScalarImageType
                                    ,TScalarType >               ScalarInterpolatorType;
  typedef typename ScalarInterpolatorType::Pointer               ScalarInterpolatorPointer;
  typedef typename ScalarInterpolatorType::PointType             ScalarInterpolatorPointType;
  typedef Image<unsigned char, Dimension>                        StatusImageType;

  /** Sets the scalar (Laplacian) image, at input 0. */
  void SetScalarImage(const InputScalarImageType *image) {this->SetNthInput(0, const_cast<InputScalarImageType *>(image)); }

  /** Sets the vector image, at input 1. */
  void SetVectorImage(const InputVectorImageType* image) { this->SetNthInput(1, const_cast<InputVectorImageType *>(image)); }

protected:
  OrderedTraversalStreamlinesFilter();
  ~OrderedTraversalStreamlinesFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  // The main filter method. Note, single threaded.
  virtual void GenerateData();

  // Typedefs used internally.
  typedef std::pair<TScalarType, InputScalarImageIndexType> Pair;
  typedef std::multimap<TScalarType, InputScalarImageIndexType> MinMap;
  typedef typename MinMap::iterator MinMapIterator;
  typedef typename MinMap::reverse_iterator MinMapReverseIterator;
  
private:
  
  /**
   * Prohibited copy and assingment. 
   */
  OrderedTraversalStreamlinesFilter(const Self&); 
  void operator=(const Self&); 

  /** 
   * Gets the Laplacian value at a given index.
   * If offset is 1, we flip Laplacian round to 1-Laplacian, as this method 
   * is used for the ordered heap. If its -1, we leave it.
   */
  OutputImagePixelType GetLaplacian(const InputScalarImageIndexType& index,
                                    const int& offset,
                                    const typename InputScalarImageType::Pointer& scalarImage);
  /**
   * Solve equation 8 or 9 in paper. 
   * If offset is -1, we are doing eqn 8.
   * If offset is 1, we are doing eqn 9.
   */
  OutputImagePixelType Solve(const InputScalarImageIndexType& index, 
                             const int& offset, 
                             const typename InputVectorImageType::Pointer& vectorImage,
                             const typename OutputImageType::Pointer& distanceImage);

  /** 
   * Simply checks if we are next to the boundary or not. 
   * We check the threshold, as then you can select either the L0 or L1 boundary.
   */
  bool IsNextToBoundary(const InputScalarImageIndexType& index, 
                        const typename InputScalarImageType::Pointer& scalarImage,
                        const typename StatusImageType::Pointer& statusImage,
                        const InputScalarImagePixelType& threshold);

  /**
   * We have to solve Algorithm 2 for L0 and L1, i.e. do it twice.
   * So this method runs step 1 - 5 twice, you just pass the right
   * images in.
   */
  void DoOrderedTraversal(const int& offset,
                          const InputScalarImagePixelType& threshold,
                          const typename InputScalarImageType::Pointer& scalarImage,
                          const typename InputVectorImageType::Pointer& vectorImage,
                          const typename OutputImageType::Pointer& distanceImage);
                          
  // TODO: Sort these out, make them const static.
  unsigned char BOUNDARY;
  unsigned char FIRST_PASS;
  unsigned char UNVISITED;
  unsigned char VISITED;
  unsigned char SOLVED;

};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkOrderedTraversalStreamlinesFilter.txx"
#endif

#endif
