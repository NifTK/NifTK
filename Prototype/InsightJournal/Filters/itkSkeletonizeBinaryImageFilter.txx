/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkSkeletonizeBinaryImageFilter_txx
#define __itkSkeletonizeBinaryImageFilter_txx

#include <queue>
#include "itkSkeletonizeBinaryImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkLogHelper.h"


namespace itk
{

/** These m_P1 to m_P9 define the 9 planes shown in figure 7 in paper. */
template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P1[8] = {3, 4,  5,  12, 14, 21, 22, 23};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P2[8] = {9, 10, 11, 12, 14, 15, 16, 17};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P3[8] = {7, 4,  1,  16, 10, 25, 22, 19};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P4[8] = {0, 4,  8,  9,  17, 18, 22, 26};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P5[8] = {6, 4,  2,  15, 11, 24, 22, 20};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P6[8] = {6, 7, 8, 12, 14, 18, 19, 20};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P7[8] = {0, 1,  2,  12, 14, 24, 25, 26};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P8[8] = {0, 3,  6,  10, 16, 20, 23, 26};

template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_P9[8] = {2, 5,  8,  10, 16, 18, 21, 24};

/** This defines the indexes of all 26 neighbours. */
template <class TInputImage,class TOutputImage>
const int SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>::m_26[26] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};

template <class TInputImage,class TOutputImage>
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::SkeletonizeBinaryImageFilter()
{
  m_RuleA = true;
  m_CheckMedialAxis = true;
  m_CheckMedialSurface = true;
  m_Tau = 6;
  m_EulerLUT = new int[256];
   
  this->fillEulerLUT( m_EulerLUT );

}

template <class TInputImage,class TOutputImage>
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::~SkeletonizeBinaryImageFilter()
{
  if (m_EulerLUT != NULL)
    {
      delete [] m_EulerLUT;
    }
}

template <class TInputImage,class TOutputImage>
bool 
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::Exterior(const InputImageType* image, const IndexType& index)
{
  // This method checks the 6-neighbourhood for connectivity.
  // From paper, section 1.1, suggests (to me at least), 
  // "foreground points n-adjacent to the background are called exterior points"
  // and as I read it, n = 6, so in 3D we need 6 connected neighbourhood.
  
  bool isExterior = false;
  bool isBoundary = false;

  IndexType plusIndex;
  IndexType minusIndex;

  // The current pixel has to be a foreground value.
  if (image->GetPixel(index) == 1)
    {
      SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
      
      for (unsigned int i = 0; i < InputImageDimension; i++)
        {
          if ((int)(index[i]) == (int)0 || (int)(index[i]) >= (int)(imageSize[i] - 1))
            {
              isBoundary = true;
            } 
        }
      
      if (!isBoundary)
        {
               
          for (unsigned int i = 0; i < InputImageDimension; i++)
            {
              plusIndex = index;
              minusIndex = index;
                  
              plusIndex[i] = index[i] + 1;
              minusIndex[i] = index[i] - 1;
                  
              if (image->GetPixel(plusIndex) == 0 || image->GetPixel(minusIndex) == 0)
                {
                  isExterior = true;
                  break;
                }
            } // end for each InputImageDimension
        } // end if not on boundary
    } // end if foreground value
  
  return isExterior;
}

template <class TInputImage,class TOutputImage>
bool 
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::Deletable(const OutputImageType* image, const DistanceImageType* distanceImage, const IndexType& index) 
{ 
  OutputImageType* im = const_cast<OutputImageType*>(image);
  
  bool isDeletable = false;
  
  if (m_RuleA)
    {
      // Rule A
      if (SimpleDeletable(im, index) && !EndOfMedialAxisOrMedialSurface(im, index))
        {
          isDeletable = true;
        }
    }
  else
    {
      // Rule B
      if (SimpleDeletable(im, index) && !CentreOfMaximalBall(im, distanceImage, index))
        {
          isDeletable = true;
        }
    }
  
  // niftkitkDebugMacro(<<"Deletable():Index:" << index << ", isDeletable=" << isDeletable);

  return isDeletable; 
}

template <class TInputImage,class TOutputImage>
bool 
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::SimpleDeletable(OutputImageType* image, const IndexType& index)
{
  bool isSimple = false;
  bool isEulerInvariant = false;
  bool isSimpleDeletable = false;
  
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);  
  
  NeighborhoodIteratorType it( radius, image, image->GetLargestPossibleRegion() );
  it.SetLocation(index);
  
  // using the methods in base class
  isSimple = this->isSimplePoint(it.GetNeighborhood());
  isEulerInvariant = this->isEulerInvariant( it.GetNeighborhood(), m_EulerLUT );
  
  if (isSimple && isEulerInvariant) 
    {
      isSimpleDeletable = true;
    }
  
  // niftkitkDebugMacro(<<"SimpleDeletable():Index:" << index << ", isSimple=" << isSimple << ", isEulerInvariant=" << isEulerInvariant << ", isSimpleDeletable=" << isSimpleDeletable);

  return isSimpleDeletable;
}

template <class TInputImage,class TOutputImage>
int 
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::CheckNumberOfNeighbours(const NeighborhoodType& neighbourhood, const int *arrayOfIndexes, unsigned int length)
{
  unsigned int count = 0;
  
  for (unsigned int i = 0; i < length; i++)
    {
      if (neighbourhood[arrayOfIndexes[i]] != 0)
        {
          count++;
        }
    }
  return count;
}

template <class TInputImage,class TOutputImage>
bool 
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::EndOfMedialAxisOrMedialSurface(OutputImageType* image, const IndexType& index)
{
  bool isEndOfMedialAxis = false;
  bool isEndOfMedialSurface = false;
  bool isEndOfMedialAxisOrMedialSurface = false;
  
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  
  NeighborhoodIteratorType it( radius, image, image->GetLargestPossibleRegion() );
  it.SetLocation(index);
  
  NeighborhoodType neighbors = it.GetNeighborhood();
  
  if (m_CheckMedialAxis)
    {
      if (CheckNumberOfNeighbours(neighbors, m_26, 26) < 2)
        {
          isEndOfMedialAxis = true;    
        }      
    }
  
  if (m_CheckMedialSurface)
    {
      if (
          CheckNumberOfNeighbours(neighbors, m_P1,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P2,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P3,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P4,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P5,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P6,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P7,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P8,  8) < 2 ||    // end of medial surface
          CheckNumberOfNeighbours(neighbors, m_P9,  8) < 2)      // end of medial surface
        {
          isEndOfMedialSurface = true;
        }      
    }
  
  if (isEndOfMedialSurface || isEndOfMedialAxis)
    {
      isEndOfMedialAxisOrMedialSurface = true;
    }

  // niftkitkDebugMacro(<<"EndOfMedialAxisOrMedialSurface():Index:" << index << ", isEndOfMedialAxis=" << isEndOfMedialAxis << ", isEndOfMedialSurface=" << isEndOfMedialSurface << ", isEndOfMedialAxisOrMedialSurface=" << isEndOfMedialAxisOrMedialSurface);

  return isEndOfMedialAxisOrMedialSurface;
}

template <class TInputImage,class TOutputImage>
bool 
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::CentreOfMaximalBall(OutputImageType* image, const DistanceImageType* distanceImage, const IndexType& index)
{
  bool isNonWitness = true;
  bool isCentreOfMaximalBall = false;
  
  SizeType size;
  size.Fill(3);

  IndexType movingIndex;
  
  for (unsigned int i = 0; i < InputImageDimension; i++)
    {
      movingIndex[i] = index[i] - 1;  
    }
  
  RegionType region;
  region.SetSize(size);
  region.SetIndex(movingIndex);
      
  ImageRegionConstIteratorWithIndex<OutputImageType> neighbourhoodIterator(image, region);
  
  unsigned int diff = 0;
  unsigned int bigD = 0;
  DistanceImagePixelType p;
  DistanceImagePixelType q;

  p = distanceImage->GetPixel(index);

  if (p > m_Tau)
    {
      for(neighbourhoodIterator.GoToBegin(); !neighbourhoodIterator.IsAtEnd(); ++neighbourhoodIterator)
        {
          movingIndex = neighbourhoodIterator.GetIndex();
          
          if (movingIndex != index)
            {
              diff = 0;
              
              for (unsigned int i = 0; i < InputImageDimension; i++)
                {
                  diff += abs(movingIndex[i] - index[i]);
                }
              
              // Diff will be either 1, 2 or 3.
              switch (diff)
                {
                  case 1:
                    bigD = 3;
                    break;
                  case 2:
                    bigD = 4;
                    break;
                  case 3:
                    bigD = 5;
                    break;
                  default:
                    bigD = 0;
                    niftkitkErrorMacro( "CentreOfMaximalBall():For index=" << index \
                        << ", at movingIndex=" << movingIndex \
                        << ", diff was=" << diff \
                        );
                }
              
              q = distanceImage->GetPixel(movingIndex);
              
              if (q - bigD == p)
                {
                  isNonWitness = false;
                  break;
                }
            } // end if not centre index
        } // end for each neighbour  
      
      if (isNonWitness)
        {
          isCentreOfMaximalBall = true;  
        }
    } // end if p > tau
  
  return isCentreOfMaximalBall;    
}

template <class TInputImage,class TOutputImage>
void
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::ComputeThinImage()
{
  niftkitkDebugMacro(<<"ComputeThinImage():Started");
  
  const unsigned char UNQUEUED = 0;
  const unsigned char QUEUED = 1;
  const unsigned char NONDELETABLE = 2;
  
  // Make sure the output is generated.
  this->AllocateOutputs();
  
  typename InputImageType::Pointer inputImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename DistanceImageType::Pointer distanceImage = m_DistanceTransform;

  if (distanceImage.IsNull())
    {
      itkExceptionMacro(<< "The distance transform has not been set");
    }
  
  typename OutputImageType::Pointer outputImage = 
        static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  typename OutputImageType::RegionType region = outputImage->GetLargestPossibleRegion();

  ConstBoundaryConditionType boundaryCondition;
  boundaryCondition.SetConstant( 0 );

  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  NeighborhoodIteratorType ot( radius, outputImage, region );
  ot.SetBoundaryCondition( boundaryCondition );

  niftkitkDebugMacro(<<"ComputeThinImage():Creating label image");
  
  typedef unsigned char LabelType;
  typedef Image<LabelType, InputImageDimension> LabelImageType;
  typename LabelImageType::Pointer labelImage = LabelImageType::New();
  labelImage->SetRegions(inputImage->GetLargestPossibleRegion());
  labelImage->SetOrigin(inputImage->GetOrigin());
  labelImage->SetDirection(inputImage->GetDirection());
  labelImage->SetSpacing(inputImage->GetSpacing());
  labelImage->Allocate();
  labelImage->FillBuffer(UNQUEUED);
  
  niftkitkDebugMacro(<<"ComputeThinImage():Adding exterior points to queue");
  
  std::priority_queue< QueueData > queue;
  std::vector< QueueData> listOfDeletable;
  std::vector< QueueData> listOfNonDeletable;
  
  ImageRegionConstIteratorWithIndex< InputImageType > inputImageIterator(inputImage, inputImage->GetLargestPossibleRegion());
  ImageRegionConstIterator< DistanceImageType > distanceImageIterator(distanceImage, distanceImage->GetLargestPossibleRegion());
  ImageRegionIterator< LabelImageType > labelImageIterator(labelImage, labelImage->GetLargestPossibleRegion());
  
  IndexType index;
  for (inputImageIterator.GoToBegin(),
       distanceImageIterator.GoToBegin(),
       labelImageIterator.GoToBegin(); 
       !inputImageIterator.IsAtEnd(); 
       ++inputImageIterator,
       ++distanceImageIterator,
       ++labelImageIterator)
    {
      index = inputImageIterator.GetIndex();
      
      if (Exterior(inputImage, index))
        {
          queue.push(QueueData(distanceImageIterator.Get(), index));          
          labelImageIterator.Set(QUEUED);           
        }
      else
        {
          labelImageIterator.Set(UNQUEUED);
        }
    }

  niftkitkDebugMacro(<<"ComputeThinImage():Added " << queue.size() << " exterior points");
    
  // Keep track of the number of iterations
  QueueData topItem;
  unsigned long int iterations = 0;
  unsigned long int totalNumberDeleted = 0;
  unsigned long int actuallyDeletedAtThisIteration = 0;
  unsigned long int neighboursAddedToListAtThisIteration = 0;
  
  while(queue.size() > 0)
    {
      actuallyDeletedAtThisIteration = 0;
      neighboursAddedToListAtThisIteration = 0;
      
      topItem = queue.top();
      queue.pop();
      
      labelImage->SetPixel(topItem.index, UNQUEUED);
      
      if (Deletable(outputImage, distanceImage, topItem.index))
        {
          listOfDeletable.push_back(topItem);  
          
          // DOHT sequential re-checking, only on points with distance
          // values equal to the current minimum distance, section 3.2.3.
          
          double minDistance = topItem.distance;
          bool sameDistance = true;
          
          while (queue.size() > 0 && sameDistance)
            {

              topItem = queue.top();
              
              if (fabs(topItem.distance - minDistance) < 0.00001)
                {
                  
                  queue.pop();                  
                  if (Deletable(outputImage, distanceImage, topItem.index))
                    {
                      
                      listOfDeletable.push_back(topItem);  
                    }
                  else
                    {                  
                      listOfNonDeletable.push_back(topItem);
                    }                  
                }
              else
                {
                  sameDistance = false;  
                }

              

            } // end while (all points with same minimum distance value).

          // Replace all non-deletable ones, so that the only place where
          // we mark something as NONDELETABLE is the containing loop
          
          for (unsigned int i = 0; i < listOfNonDeletable.size(); i++)
            {
              // niftkitkDebugMacro(<<"ComputeThinImage():Replacing non-deletable:" << listOfNonDeletable[i].index);
              queue.push(listOfNonDeletable[i]);
            }
          listOfNonDeletable.clear();
          
          // For all deletable ones, we go through list again, checking if we can delete them.
          // Again, this bit inspired by functionality in the base class.
          
          for (unsigned int i = 0; i < listOfDeletable.size(); i++)
            {

              // 1. Set the pixel to zero.
              outputImage->SetPixel(listOfDeletable[i].index, 0);
              totalNumberDeleted++;
              actuallyDeletedAtThisIteration++;
              
              // 2. Check if neighborhood is still connected
              ot.SetLocation( listOfDeletable[i].index );
              
              if( !isSimplePoint(ot.GetNeighborhood()) )
                {
                  // 3. We cannot delete current point, so reset back to 1
                  outputImage->SetPixel( listOfDeletable[i].index, 1 );
                  totalNumberDeleted--;
                  actuallyDeletedAtThisIteration--;
                  
                  // Replace all non-deletable ones on main queue, so that the 
                  // only place where we mark something as NONDELETABLE is the containing loop                
                  queue.push(listOfDeletable[i]);
                  
                }
              else
                {
                  // 4. We can delete: Need to iterate round 3x3x3 region, and pick up the neighbours.
         
                  SizeType size;
                  size.Fill(3);

                  for (unsigned int j = 0; j < InputImageDimension; j++)
                    {
                      index[j] = listOfDeletable[i].index[j] - 1;
                    }
          
                  RegionType region;
                  region.SetSize(size);
                  region.SetIndex(index);
                      
                  ImageRegionConstIteratorWithIndex<OutputImageType> neighbourhoodIterator(outputImage, region);
                  for(neighbourhoodIterator.GoToBegin();
                     !neighbourhoodIterator.IsAtEnd();
                     ++neighbourhoodIterator)
                    {
                      index = neighbourhoodIterator.GetIndex();

                      if (index != listOfDeletable[i].index)
                        {
                          if (outputImage->GetPixel(index) == 1 && 
                          labelImage->GetPixel(index) != QUEUED &&
                          labelImage->GetPixel(index) != NONDELETABLE)
                            {
                              queue.push(QueueData(distanceImage->GetPixel(index), index));
                              labelImage->SetPixel(index, QUEUED); 
                              
                              neighboursAddedToListAtThisIteration++;
                              
                            }
                        }
                    } // end for each neighbour                  
                } // end else (is deletable).
            } // end for list of deletable

          listOfDeletable.clear();
        }
      else
        {
          labelImage->SetPixel(topItem.index, NONDELETABLE);
        }
/*            
      niftkitkDebugMacro(<<"ComputeThinImage():Iteration " << iterations \
        << ", queue size=" << queue.size() \
        << ", neighboursAdded=" << neighboursAddedToListAtThisIteration \
        << ", deleted=" << actuallyDeletedAtThisIteration \
        << ", totalDeleted=" << totalNumberDeleted \
        );
*/      
      iterations++;
    }

  niftkitkInfoMacro(<<"ComputeThinImage():Iterations " << iterations \
    << ", totalDeleted=" << totalNumberDeleted \
    );
  
  niftkitkDebugMacro(<<"ComputeThinImage():Finished");
}

/** NOT USED. */

template <class TInputImage,class TOutputImage>
int
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::CheckNumberOfConnectedComponents(const NeighborhoodType& neighbors)
{
  // copy neighbors for labeling
  int cube[26];
  int i;
  for( i = 0; i < 13; i++ )  // i =  0..12 -> cube[0..12]
    cube[i] = neighbors[i];
  // i != 13 : ignore center pixel when counting (see [Lee94])
  for( i = 14; i < 27; i++ ) // i = 14..26 -> cube[13..25]
    cube[i-1] = neighbors[i];
  // set initial label
  int label = 2;
  // for all points in the neighborhood
  for( int i = 0; i < 26; i++ )
  {
    if( cube[i]==1 )     // voxel has not been labelled yet
    {
      // start recursion with any octant that contains the point i
      switch( i )
      {
      case 0:
      case 1:
      case 3:
      case 4:
      case 9:
      case 10:
      case 12:
        this->Octree_labeling(1, label, cube );
        break;
      case 2:
      case 5:
      case 11:
      case 13:
        this->Octree_labeling(2, label, cube );
        break;
      case 6:
      case 7:
      case 14:
      case 15:
        this->Octree_labeling(3, label, cube );
        break;
      case 8:
      case 16:
        this->Octree_labeling(4, label, cube );
        break;
      case 17:
      case 18:
      case 20:
      case 21:
        this->Octree_labeling(5, label, cube );
        break;
      case 19:
      case 22:
        this->Octree_labeling(6, label, cube );
        break;
      case 23:
      case 24:
        this->Octree_labeling(7, label, cube );
        break;
      case 25:
        this->Octree_labeling(8, label, cube );
        break;
      }
      label++;
    }
  }
  return label-2;
}

/** NOT USED. */

template <class TInputImage,class TOutputImage>
bool 
SkeletonizeBinaryImageFilter<TInputImage,TOutputImage>
::Simple(OutputImageType* image, const IndexType& index)
{
  bool isSimple = false; 

  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);  
  
  NeighborhoodIteratorType it( radius, image, image->GetLargestPossibleRegion() );
  it.SetLocation(index);

  NeighborhoodType foregroundIsSetToOne = it.GetNeighborhood();
  NeighborhoodType backgroundIsSetToOne = it.GetNeighborhood();
  
  for (unsigned int i = 0; i < 26; i++)
    {
      if (foregroundIsSetToOne[i] == 1)
        {
          backgroundIsSetToOne[i] = 0;  
        }
      else
        {
          backgroundIsSetToOne[i] = 1;
        }
    }
  
  int numberOfConnectedForgroundComponents = CheckNumberOfConnectedComponents(foregroundIsSetToOne);  
  int numberOfConnectedBackgroundComponents = CheckNumberOfConnectedComponents(backgroundIsSetToOne) - 1;
  
  bool IsMAdjacent = false;
  bool isNAdjacent = false;
  
  if (numberOfConnectedForgroundComponents == 1)
    {
      IsMAdjacent = true; 
    }
  
  IndexType plusIndex;
  IndexType minusIndex;
  
  for (unsigned int i = 0; i < InputImageDimension; i++)
    {
      plusIndex = index;
      minusIndex = index;
      
      plusIndex[i] = index[i] + 1;
      minusIndex[i] = index[i] - 1;
      
      if (image->GetPixel(plusIndex) == 0 || image->GetPixel(minusIndex) == 0)
        {
          isNAdjacent = true;  
        }
    }
  
  if (numberOfConnectedBackgroundComponents > 1)
    {
      isNAdjacent = false;  
    }
  
  isSimple = IsMAdjacent && isNAdjacent;
  
  niftkitkDebugMacro(<< "Simple():#fore=" << numberOfConnectedForgroundComponents \
      << ", #back=" << numberOfConnectedBackgroundComponents \
      << ", IsMAdjacent=" << IsMAdjacent \
      << ", isNAdjacent=" << isNAdjacent \
      << ", isSimple=" << isSimple \
      );
  
  return isSimple;
}

} /** End namespace. */

#endif
