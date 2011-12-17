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
#ifndef __itkSkeletonizeBinaryImageFilter_h
#define __itkSkeletonizeBinaryImageFilter_h
#include <iostream>
#include "itkBinaryThinningImageFilter3D.h"

namespace itk
{
/** 
 * \class SkeletonizeBinaryImageFilter
 * \brief Implements Pudney, CVIU Vol 72, No 3 December 1998, pp 404-413.
 *
 * The input images is a binary image, with values 0 and 1.
 * You must also SetDistanceTransform(image), where image is a float valued image.
 * 
 * The output is a skeletonized binary image. We extend itkBinaryThinningFilter3D, to use
 * the isSimplePoint and isEulerInvariant, so credit goes to Hanno Homann, who implemented:
 *
 * T.C. Lee, R.L. Kashyap, and C.N. Chu.
 * Building skeleton models via 3-D medial surface/axis thinning algorithms.
 * Computer Vision, Graphics, and Image Processing, 56(6):462--478, 1994.
 * 
 * as described in this article: http://www.insight-journal.org/browse/publication/181
 * 
 * This filter implements both Rule A and Rule B, with methods UseRuleA() and UseRuleB()
 * to switch between the modes. This could clearly have been written using two sub-classes,
 * but I decided to put it all in one class to keep it all in one place.
 * 
 * This implementation requires that the distance transform is a Chamfer distance, 
 * with weights for 3D set to 3, 4 and 5. (See Pudney paper).
 */
template <class TInputImage,class TOutputImage>
class ITK_EXPORT SkeletonizeBinaryImageFilter : public BinaryThinningImageFilter3D<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SkeletonizeBinaryImageFilter                           Self;
  typedef BinaryThinningImageFilter3D<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(SkeletonizeBinaryImageFilter, BinaryThinningImageFilter3D);

  /** Type for input image. */
  typedef   TInputImage       InputImageType;

  /** Type for output image: Skeleton of the object.  */
  typedef   TOutputImage      OutputImageType;

  /** Type for the region of the input image. */
  typedef typename InputImageType::RegionType RegionType;

  /** Type for the index of the input image. */
  typedef typename RegionType::IndexType  IndexType;

  /** Type for the pixel type of the input image. */
  typedef typename InputImageType::PixelType InputImagePixelType ;

  /** Type for the pixel type of the input image. */
  typedef typename OutputImageType::PixelType OutputImagePixelType ;

  /** Type for the size of the input image. */
  typedef typename RegionType::SizeType SizeType;

  /** Pointer Type for input image. */
  typedef typename InputImageType::ConstPointer InputImagePointer;

  /** Pointer Type for the output image. */
  typedef typename OutputImageType::Pointer OutputImagePointer;
  
  /** Boundary condition type for the neighborhood iterator */
  typedef ConstantBoundaryCondition< TInputImage > ConstBoundaryConditionType;
  
  /** Neighborhood iterator type */
  typedef NeighborhoodIterator<TInputImage, ConstBoundaryConditionType> NeighborhoodIteratorType;
  
  /** Neighborhood type */
  typedef typename NeighborhoodIteratorType::NeighborhoodType NeighborhoodType;

  /** ImageDimension enumeration   */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension );
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension );

  /** Typedef for distance transform image. */
  typedef Image<float, InputImageDimension> DistanceImageType;
  
  /** Typedef for distance image pixel type. */
  typedef typename DistanceImageType::PixelType DistanceImagePixelType;
  
  /** Set the distance transform (managed externally). */
  void SetDistanceTransform(DistanceImageType* image) { m_DistanceTransform = image; }
  
  /** Set the filter to use Rule A, checking medial axis, and medial surface, as mentioned in section 3.2.1. */
  void UseRuleA() { this->m_RuleA = true; }
  
  /** Set the filter to use Rule B, checking the centres of maximal balls, as mentioned in section 3.2.2. */
  void UseRuleB() { this->m_RuleA = false; }
  
  /** Set/Get the flag to check the medial axis. Default true. */
  itkSetMacro(CheckMedialAxis, bool);
  itkGetMacro(CheckMedialAxis, bool);

  /** Set/Get the flag to check the medial surface. Default true. */
  itkSetMacro(CheckMedialSurface, bool);
  itkGetMacro(CheckMedialSurface, bool);

  /** Set/Get the Tau value described in section 3.2.2 which is the minimum radius for the centre of maximal balls. */
  itkSetMacro(Tau, double);
  itkGetMacro(Tau, double);

protected:

  SkeletonizeBinaryImageFilter();
  virtual ~SkeletonizeBinaryImageFilter();
  
  // Just struct to hold data.
  class QueueData {

    public:

       float distance;
       IndexType index;

       QueueData()
         {
           distance = 0;
           index.Fill(0);
         }
       
       QueueData(const QueueData& another)
         {
         index = another.index;
         distance = another.distance;
         }

       QueueData(const float& aDistance, const IndexType& anIndex)
         {
           distance = aDistance;
           index = anIndex;
         };

       bool operator< (const QueueData& right) const
         {
           return distance > right.distance;
         }
  };

  // The main filter method. Note, single threaded.
  virtual void ComputeThinImage();

  /** Checks if the given position is deletable. */
  virtual bool Deletable(const OutputImageType* image, const DistanceImageType* distanceImage, const IndexType& index);

  /** Checks if a point is simple deletable. */
  virtual bool SimpleDeletable(OutputImageType* image, const IndexType& index);

  /** Checks if a point is the end of a medial axis or medial surface. */
  virtual bool EndOfMedialAxisOrMedialSurface(OutputImageType* image, const IndexType& index);
  
  /** Checks if a point is the centre of a maximal ball. */
  virtual bool CentreOfMaximalBall(OutputImageType* image, const DistanceImageType *distanceImage, const IndexType& index);

  /** Checks if the given position is on the boundary. */
  virtual bool Exterior(const InputImageType* image, const IndexType& index);

private:
  SkeletonizeBinaryImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  /** Hold pointer to distance transform. */
  typename DistanceImageType::Pointer m_DistanceTransform;
  
  /** Flag to switch between Rule A and Rule B. Default true. */
  bool m_RuleA;
  
  /** Flag to determine if we are checking medial axis. Default true. */
  bool m_CheckMedialAxis;
  
  /** Flag to determine if we are checking medial surface. Default true. */
  bool m_CheckMedialSurface;
  
  /** Distance threshold for the Rule B. Default 6. */
  double m_Tau;
  
  /** 
   * We simply iterate through arrayOfIndexes, counting the number
   * of neighbourhood elements that are non-zero.
   * lengthOfArray is the length of arrayOfIndexes.
   */
  int CheckNumberOfNeighbours(const NeighborhoodType& neighbourhood, const int *arrayOfIndexes, unsigned int lengthOfArray); 

  /** NOT USED. Checks if a point is simple, according to Theorem 2 in section 2.1. NOT USED. */
  virtual bool Simple(OutputImageType* image, const IndexType& index);
  
  /** NOT USED. Returns the number of connected components. NOT USED. */
  int CheckNumberOfConnectedComponents(const NeighborhoodType& neighbors);
  
  // These planes are in Figure 7 in paper.
  const static int m_P1[];
  const static int m_P2[];
  const static int m_P3[];
  const static int m_P4[];
  const static int m_P5[];
  const static int m_P6[];
  const static int m_P7[];
  const static int m_P8[];
  const static int m_P9[];
  const static int m_26[];
  int* m_EulerLUT;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSkeletonizeBinaryImageFilter.txx"
#endif

#endif
