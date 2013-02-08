/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkSimilarityMeasure_txx
#define _itkSimilarityMeasure_txx
#include "ConversionUtils.h"
#include "itkSimilarityMeasure.h"
#include "itkStatisticsImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkUCLMacro.h"

namespace itk
{
template <typename TFixedImage, typename TMovingImage> 
const int SimilarityMeasure<TFixedImage,TMovingImage>::SYMMETRIC_METRIC_AVERAGE = 1; 

template <typename TFixedImage, typename TMovingImage> 
const int SimilarityMeasure<TFixedImage,TMovingImage>::SYMMETRIC_METRIC_MID_WAY = 2; 

template <typename TFixedImage, typename TMovingImage> 
const int SimilarityMeasure<TFixedImage,TMovingImage>::SYMMETRIC_METRIC_BOTH_FIXED_AND_MOVING_TRANSFORM = 3; 

/*
 * Constructor
 */
template <class TFixedImage, class TMovingImage> 
SimilarityMeasure<TFixedImage,TMovingImage>
::SimilarityMeasure()
{
  m_BoundsSetByUser = false;
  m_TwoSidedMetric = false;
  
  m_WriteFixedImage = false;
  m_FixedImageFileName = "tmp.similarity.fixed";
  m_FixedImageFileExt = "nii";
  
  m_WriteTransformedMovingImage = false;
  m_TransformedMovingImageFileName = "tmp.similarity.moving";
  m_TransformedMovingImageFileExt = "nii";
  
  m_IterationNumber = 0;
  
  this->m_FixedLowerBound = NumericTraits< FixedImagePixelType >::Zero;
  this->m_FixedUpperBound = NumericTraits< FixedImagePixelType >::Zero;
  this->m_MovingLowerBound = NumericTraits< MovingImagePixelType >::Zero;
  this->m_MovingUpperBound = NumericTraits< MovingImagePixelType >::Zero;
  
  m_DirectVoxelComparison = false;
  m_SymmetricMetric = 0; 
  m_IsUpdateMatrix = true; 
  m_TransformedMovingImagePadValue = 0;
  m_UseWeighting = false; 
  m_WeightingDistanceThreshold = 2.0; 
  m_InitialiseIntensityBoundsUsingMask = false; 
  m_IsResampleWholeImage = false; 
  
  niftkitkDebugMacro("SimilarityMeasure():Constructed");
}

/*
 * PrintSelf
 */
template <class TFixedImage, class TMovingImage> 
void
SimilarityMeasure<TFixedImage, TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "FixedLowerBound = " << this->m_FixedLowerBound << std::endl;
  os << indent << "FixedUpperBound = " << this->m_FixedUpperBound << std::endl;
  os << indent << "MovingLowerBound = " << this->m_MovingLowerBound << std::endl;
  os << indent << "MovingUpperBound = " << this->m_MovingUpperBound << std::endl;
  os << indent << "BoundsSetByUser = " << this->m_BoundsSetByUser << std::endl;
  os << indent << "TwoSidedMetric = " << this->m_TwoSidedMetric  << std::endl;
  os << indent << "FixedImageFileName = " << this->m_FixedImageFileName  << std::endl;
  os << indent << "FixedImageFileExt = " << this->m_FixedImageFileExt  << std::endl;
  os << indent << "TransformedMovingImageFileName = " << this->m_TransformedMovingImageFileName  << std::endl;
  os << indent << "TransformedMovingImageFileExt = " << this->m_TransformedMovingImageFileExt  << std::endl;  
  os << indent << "DirectVoxelComparison = " << this->m_DirectVoxelComparison  << std::endl;  
  os << indent << "TransformedMovingImagePadValue = " << this->m_TransformedMovingImagePadValue  << std::endl;
}

template <class TFixedImage, class TMovingImage>
void 
SimilarityMeasure<TFixedImage, TMovingImage>
::SetIntensityBounds(  const FixedImagePixelType fixedLower, 
                       const FixedImagePixelType fixedUpper,
                       const MovingImagePixelType movingLower,
                       const MovingImagePixelType movingUpper)
{
   m_FixedLowerBound = fixedLower;
   m_FixedUpperBound = fixedUpper;
   m_MovingLowerBound = movingLower;
   m_MovingUpperBound = movingUpper;
   m_BoundsSetByUser = true;
   this->Modified();
   
   niftkitkDebugMacro("Set bounds:fixedLower:"
     << niftk::ConvertToString(fixedLower)
   << ",fixedUpper:" << niftk::ConvertToString(fixedUpper)
   << ",movingLower:" << niftk::ConvertToString(movingLower)
   << ",movingUpper:" << niftk::ConvertToString(movingUpper));
}

template <class TFixedImage, class TMovingImage>
void SimilarityMeasure<TFixedImage, TMovingImage>
::InitializeIntensityBounds() throw (ExceptionObject)
{
  niftkitkDebugMacro("InitializeIntensityBounds():Started");
  
  if( m_BoundsSetByUser == false )
    {
      // Calculate min and max image values in fixed image.
      FixedImageConstPointer pFixedImage = this->m_FixedImage;
      ImageRegionConstIteratorWithIndex<FixedImageType> fiIt(pFixedImage,
                                                  pFixedImage->
                                                      GetLargestPossibleRegion());
      FixedMaskType* fixedMask = dynamic_cast<FixedMaskType*>(this->m_FixedImageMask.GetPointer()); 
      int numberOfVoxels = 0; 
      
      fiIt.GoToBegin();
      FixedImagePixelType minFixed = fiIt.Value();
      FixedImagePixelType maxFixed = fiIt.Value();
      ++fiIt;
      while ( !fiIt.IsAtEnd() )
        {
          FixedImagePixelType value = fiIt.Value();
          
          // Ignore values outside the mask. 
          if (m_InitialiseIntensityBoundsUsingMask)
          {
             typename TFixedImage::IndexType index = fiIt.GetIndex(); 
             if (fixedMask->GetImage()->GetPixel(index) == 0)
             {
               ++fiIt;
               continue; 
             }
          }
          numberOfVoxels++; 
          if (value < minFixed)
            {
              minFixed = value;
            }
              else if (value > maxFixed)
            {
              maxFixed = value;
            }
          ++fiIt;
        }
      niftkitkInfoMacro("numberOfVoxels=" << numberOfVoxels); 
      
      // Calculate min and max image values in moving image.
      MovingImageConstPointer pMovingImage = this->m_MovingImage;
      ImageRegionConstIteratorWithIndex<MovingImageType> miIt(pMovingImage,
                                                   pMovingImage->
                                                       GetLargestPossibleRegion());
      miIt.GoToBegin();
      MovingImagePixelType minMoving = miIt.Value();
      MovingImagePixelType maxMoving = miIt.Value();
      ++miIt;
      numberOfVoxels = 0; 
      while ( !miIt.IsAtEnd() )
        {
          MovingImagePixelType value = miIt.Value();

          // Ignore values outside the mask. NOTE: the fixed image mask is used. 
          if (m_InitialiseIntensityBoundsUsingMask)
          {
             typename TMovingImage::IndexType index = miIt.GetIndex(); 
             if (fixedMask->GetImage()->GetPixel(index) == 0)
             {
               ++miIt;
               continue; 
             }
          }
          numberOfVoxels++; 
          if (value < minMoving)
            {
              minMoving = value;
            }
          else if (value > maxMoving)
            {
              maxMoving = value;
            }
          ++miIt;
        }
      niftkitkInfoMacro("numberOfVoxels=" << numberOfVoxels); 

      // Initialize the upper and lower bounds of the histogram.
      m_FixedLowerBound = minFixed;
      m_FixedUpperBound = maxFixed;
      m_MovingLowerBound = minMoving;
      m_MovingUpperBound = maxMoving;

      niftkitkDebugMacro("InitializeIntensityBounds():Automatically set bounds:"
    		  << "fixedLower:" << niftk::ConvertToString((double)m_FixedLowerBound)
        << ",fixedUpper:" << niftk::ConvertToString((double)m_FixedUpperBound)
        << ",movingLower:" << niftk::ConvertToString((double)m_MovingLowerBound)
        << ",movingUpper:" << niftk::ConvertToString((double)m_MovingUpperBound));

    }
  else
    {
      niftkitkDebugMacro("InitializeIntensityBounds():Bounds have already been set by user");
    }
  niftkitkDebugMacro("InitializeIntensityBounds():Finished");
}

template <class TFixedImage, class TMovingImage>
void SimilarityMeasure<TFixedImage, TMovingImage>
::Initialize() throw (ExceptionObject)
{
  this->SetFixedImageRegion(this->m_FixedImage->GetLargestPossibleRegion());
  
  Superclass::Initialize();

  if (!this->m_FixedImage)
    {
      itkExceptionMacro("Fixed image has not been set.");
    }
  else if (!this->m_MovingImage)
    {
      itkExceptionMacro("Moving image has not been set.");
    }

  if( !this->m_Transform)
    {
      FixedImageSizeType fixedSize = this->m_FixedImage->GetLargestPossibleRegion().GetSize();
      MovingImageSizeType movingSize = this->m_MovingImage->GetLargestPossibleRegion().GetSize();
  
      if (fixedSize != movingSize)
        {
          itkExceptionMacro("Transform has not been assigned, and fixed and moving size are different, so I can't do a voxel wise comparison either." );
        }
      else
        {
          m_DirectVoxelComparison = true;  
        }      
    }
  
  // Allocate transformed image.
  this->m_TransformedMovingImage = TFixedImage::New();
  this->m_TransformedMovingImage->SetRegions(this->m_FixedImage->GetLargestPossibleRegion());
  this->m_TransformedMovingImage->SetSpacing(this->m_FixedImage->GetSpacing());
  this->m_TransformedMovingImage->SetOrigin(this->m_FixedImage->GetOrigin());
  this->m_TransformedMovingImage->SetDirection(this->m_FixedImage->GetDirection());
  this->m_TransformedMovingImage->Allocate();  
  
  if (this->m_SymmetricMetric == SYMMETRIC_METRIC_MID_WAY)
  {
    InitializeSymmetricMetric(); 
  }
  
  if (this->m_SymmetricMetric == SYMMETRIC_METRIC_BOTH_FIXED_AND_MOVING_TRANSFORM)
  {
    this->m_TransformedFixedImage = TFixedImage::New();
    this->m_TransformedFixedImage->SetRegions(this->m_FixedImage->GetLargestPossibleRegion());
    this->m_TransformedFixedImage->SetSpacing(this->m_FixedImage->GetSpacing());
    this->m_TransformedFixedImage->SetOrigin(this->m_FixedImage->GetOrigin());
    this->m_TransformedFixedImage->SetDirection(this->m_FixedImage->GetDirection());
    this->m_TransformedFixedImage->Allocate();  
  }
  
  if (this->m_UseWeighting)
  {
    InitializeDistanceWeightings(); 
  }

  this->InitializeIntensityBounds();
}



template <class TFixedImage, class TMovingImage>
void 
SimilarityMeasure<TFixedImage, TMovingImage>
::InitializeSymmetricMetric()
{
  this->m_MidwayImage = MidwayImageType::New(); 
  typename TFixedImage::SpacingType fixedImageSpacing = this->m_FixedImage->GetSpacing(); 
  typename TFixedImage::SpacingType movingImageSpacing = this->m_MovingImage->GetSpacing(); 
  typename TFixedImage::SpacingType minImageSpacing;  
  typename TFixedImage::IndexType centerIndex; 
  for (unsigned int i = 0; i < TFixedImage::ImageDimension; i++)
  {
    minImageSpacing[i] = std::min<double>(fixedImageSpacing[i], movingImageSpacing[i]);
  }
  this->m_MidwayImage->SetSpacing(minImageSpacing);
  
  typename TFixedImage::RegionType fixedRegion = this->m_FixedImage->GetLargestPossibleRegion(); 
  typename TFixedImage::RegionType movingRegion = this->m_MovingImage->GetLargestPossibleRegion(); 
  for (unsigned int i = 0; i < TFixedImage::ImageDimension; i++)
  {
    double maxPhysicalSize = std::max<double>(fixedRegion.GetSize(i)*fixedImageSpacing[i], movingRegion.GetSize(i)*movingImageSpacing[i]); 
    unsigned int imageSize = static_cast<unsigned int>(maxPhysicalSize/minImageSpacing[i]+0.5); 
    fixedRegion.SetSize(i, imageSize);
    centerIndex[i] = imageSize/2; 
  }
  this->m_MidwayImage->SetRegions(fixedRegion); 
  niftkitkDebugMacro("m_MidwayImage region:" << fixedRegion);
  niftkitkDebugMacro("m_MidwayImage spacing:" << this->m_MidwayImage->GetSpacing());
  
  typename TFixedImage::PointType midPoint; 
  midPoint.SetToMidPoint(this->m_FixedImage->GetOrigin(), this->m_MovingImage->GetOrigin());  
  this->m_MidwayImage->SetOrigin(midPoint);
  niftkitkDebugMacro("m_MidwayImage origin:" << this->m_MidwayImage->GetOrigin());
  // Should use the middle origin and direction as well. Later.....
  this->m_MidwayImage->SetDirection(this->m_FixedImage->GetDirection());

  this->m_MidwayImage->Allocate();  
}

template <class TFixedImage, class TMovingImage>
void 
SimilarityMeasure<TFixedImage, TMovingImage>
::InitializeDistanceWeightings()
{
  FixedMaskType* fixedMask = dynamic_cast<FixedMaskType*>(this->m_FixedImageMask.GetPointer()); 
  MovingMaskType* movingMask = dynamic_cast<MovingMaskType*>(this->m_MovingImageMask.GetPointer()); 
  
  niftkitkDebugMacro("Preparing m_FixedDistanceMap...");
  this->m_FixedDistanceMap = FixedDistanceMapImageFilterType::New(); 
  this->m_FixedDistanceMap->SetInput(fixedMask->GetImage()); 
  this->m_FixedDistanceMap->SetUseImageSpacing(true); 
  this->m_FixedDistanceMap->SetBackgroundValue(0); 
  this->m_FixedDistanceMap->SetInsideIsPositive(true); 
  this->m_FixedDistanceMap->Update(); 
  
  niftkitkDebugMacro("Preparing m_MovingDistanceMap...");
  this->m_MovingDistanceMap = MovingDistanceMapImageFilterType::New(); 
  this->m_MovingDistanceMap->SetInput(movingMask->GetImage()); 
  this->m_MovingDistanceMap->SetUseImageSpacing(true); 
  this->m_MovingDistanceMap->SetBackgroundValue(0); 
  this->m_MovingDistanceMap->SetInsideIsPositive(true); 
  this->m_MovingDistanceMap->Update(); 
  
  this->m_FixedDistanceMapInterpolator = DistanceMapLinearInterpolatorType::New();
  this->m_FixedDistanceMapInterpolator->SetInputImage(this->m_FixedDistanceMap->GetOutput()); 
  this->m_MovingDistanceMapInterpolator = DistanceMapLinearInterpolatorType::New();
  this->m_MovingDistanceMapInterpolator->SetInputImage(this->m_MovingDistanceMap->GetOutput()); 
}


template <class TFixedImage, class TMovingImage>
void
SimilarityMeasure<TFixedImage, TMovingImage>
::WriteImage(const TFixedImage* image, std::string filename) const
{
  niftkitkDebugMacro("WriteImage():Writing image:" << image << ", to file:" << filename);
  typename ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
  writer->SetFileName(filename);
  writer->SetInput(image);
  writer->Modified();
  writer->Update();
  niftkitkDebugMacro("WriteImage():Done");
}

template <class TFixedImage, class TMovingImage> 
typename SimilarityMeasure<TFixedImage,TMovingImage>::MeasureType 
SimilarityMeasure<TFixedImage, TMovingImage>
::GetSimilarity( const TransformParametersType & parameters ) const
{
  if (this->m_SymmetricMetric == SYMMETRIC_METRIC_AVERAGE)
    {
      return const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->GetSymmetricSimilarity(parameters);  
    }
  else if (this->m_SymmetricMetric == SYMMETRIC_METRIC_MID_WAY)
    {
      return const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->GetSymmetricSimilarityAtHalfway(parameters);  
    }
  else if (this->m_SymmetricMetric == SYMMETRIC_METRIC_BOTH_FIXED_AND_MOVING_TRANSFORM)
    {
      return const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->GetSimilarityUsingFixedAndMovingImageTransforms(parameters);  
    }

  typename TFixedImage::RegionType fixedRegion;
  fixedRegion = this->GetFixedImage()->GetLargestPossibleRegion();
    
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  FixedImagePixelType fixedValue = 0;
  MovingImagePixelType movingValue = 0;
  typename TFixedImage::IndexType fixedMaskTransformedIndex; 
  typename TMovingImage::IndexType movingMaskTransformedIndex; 

  // Reset the derived class. We need to throw away constness, as this function shouldn't be const.
  const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->ResetCostFunction();
  // clock_t start = clock(); 
   
  if (m_DirectVoxelComparison)
    {
      typename TMovingImage::RegionType movingRegion;
      typename TMovingImage::SizeType   movingRegionSize;
      typename TMovingImage::IndexType  movingRegionIndex;

      for (int i = 0; i < TFixedImage::ImageDimension; i++)
        {
          movingRegionSize[i] = (unsigned long int)parameters.GetElement(i + 1) ;
          movingRegionIndex[i] = (long int)parameters.GetElement(i + 1 + TFixedImage::ImageDimension);
        }
      movingRegion.SetSize(movingRegionSize);
      movingRegion.SetIndex(movingRegionIndex);
      
      typedef itk::ImageRegionConstIterator<TFixedImage> NonIndexIteratorType;
        
      NonIndexIteratorType fixedImageNonIndexIterator(this->m_FixedImage, fixedRegion);
      NonIndexIteratorType movingImageNonIndexIterator(this->m_MovingImage, movingRegion);
      
      fixedImageNonIndexIterator.GoToBegin();
      movingImageNonIndexIterator.GoToBegin();
      
      while(!fixedImageNonIndexIterator.IsAtEnd())
        {
          fixedValue = fixedImageNonIndexIterator.Get();
      
          if (!m_BoundsSetByUser || (fixedValue > this->m_FixedLowerBound && fixedValue <= this->m_FixedUpperBound))
            {
              movingValue = movingImageNonIndexIterator.Get();
      
              if (!m_BoundsSetByUser || (movingValue > this->m_MovingLowerBound && movingValue <= this->m_MovingUpperBound))
                {
                  const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->AggregateCostFunctionPair(fixedValue, movingValue);
                }
            }

          ++fixedImageNonIndexIterator;
          ++movingImageNonIndexIterator;
        }

      // Now sum up the measure in derived class.
      measure = const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->FinalizeCostFunction();

      if(this->m_PrintOutMetricEvaluation)
        {
          niftkitkDebugMacro("GetValue(), value of metric:" << niftk::ConvertToString(measure));
        }
    }
  else
    {
      typedef itk::ImageRegionConstIteratorWithIndex<TFixedImage> IndexIteratorType;
      typedef itk::ImageRegionIterator<TFixedImage> NonIndexIteratorType;
      
      FixedImagePixelType zero = NumericTraits< FixedImagePixelType >::Zero;

      InputPointType inputPoint;
      OutputPointType transformedPoint;
      ContinuousIndex<double, TMovingImage::ImageDimension> movingImageTransformedIndex; 

      // We will count the number of samples from looping over the image masks.  
      this->m_NumberOfFixedSamples = 0;
      this->m_NumberOfMovingSamples = 0;
      
      typename FixedImageType::IndexType index;
      IndexIteratorType    fixedImageIterator(this->m_FixedImage, fixedRegion);
      
      // We iterate over m_TransformedMovingImage.
      // This is the same size as FixedImage, and is the moving 
      // image resampled into fixed image coordinate space.
      NonIndexIteratorType transformedMovingImageIterator(this->m_TransformedMovingImage, fixedRegion); 
      
      // To replace the slow SpatialObject::IsInside by directly testing it on the image masks. 
      FixedMaskType* fixedMask = dynamic_cast<FixedMaskType*>(this->m_FixedImageMask.GetPointer()); 
      MovingMaskType* movingMask = dynamic_cast<MovingMaskType*>(this->m_MovingImageMask.GetPointer()); 

      // Set transform. Only do this iff !m_DirectVoxelComparison    
      TransformType *transform = NULL;
      this->SetTransformParameters( parameters );
      transform = this->m_Transform;

      this->m_Interpolator->SetInputImage(this->m_MovingImage);

      fixedImageIterator.GoToBegin();
      transformedMovingImageIterator.GoToBegin();

      while(!fixedImageIterator.IsAtEnd() && !transformedMovingImageIterator.IsAtEnd())
        {
          index = fixedImageIterator.GetIndex();
      
          this->m_FixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

          // Use input mask.
          if(!this->m_FixedImageMask.IsNull() && 
             (!fixedMask->GetImage()->TransformPhysicalPointToIndex(inputPoint, fixedMaskTransformedIndex) ||
               fixedMask->GetImage()->GetPixel(fixedMaskTransformedIndex) == 0))
            {
              transformedMovingImageIterator.Set(m_TransformedMovingImagePadValue);
              
              ++fixedImageIterator;
              ++transformedMovingImageIterator;
              continue;
            }

          transformedPoint = transform->TransformPoint( inputPoint );
          //std::cout << "fixed index=" << index << std::endl; 
          //std::cout << "inputPoint=" << inputPoint << std::endl;
          //std::cout << "transformedPoint=" << transformedPoint << std::endl; 
          
          //OutputPointType outputPoint;
          //this->m_MovingImage->TransformIndexToPhysicalPoint( index, outputPoint );
          //std::cout << "outputPoint=" << outputPoint << std::endl;

          /*
          ContinuousIndex< double, TFixedImage::ImageDimension > transformedVoxel;
          this->m_MovingImage->TransformPhysicalPointToContinuousIndex(transformedPoint, transformedVoxel);
          printf("Matt:\tresample\t:%f, %f, %f, %f, %f, %f\n", (float)index[0], (float)index[1], (float)index[2], transformedVoxel[0], transformedVoxel[1], transformedVoxel[2] );
          */
          
          if(!this->m_TwoSidedMetric && 
             !this->m_MovingImageMask.IsNull() && 
             (!movingMask->GetImage()->TransformPhysicalPointToIndex(transformedPoint, movingMaskTransformedIndex) || 
               movingMask->GetImage()->GetPixel(movingMaskTransformedIndex) == 0))
            {
              transformedMovingImageIterator.Set(m_TransformedMovingImagePadValue);
              
              ++fixedImageIterator;
              ++transformedMovingImageIterator;
              continue;
            }

          fixedValue = zero;
          movingValue = m_TransformedMovingImagePadValue;
          
          if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
            {
              fixedValue = fixedImageIterator.Get();
          
              if (!m_BoundsSetByUser || (fixedValue > this->m_FixedLowerBound && fixedValue <= this->m_FixedUpperBound))
                {
                  this->m_MovingImage->TransformPhysicalPointToContinuousIndex(transformedPoint, movingImageTransformedIndex);
                  movingValue = (MovingImagePixelType)(this->m_Interpolator->EvaluateAtContinuousIndex(movingImageTransformedIndex));
                  // movingValue = (MovingImagePixelType)(this->m_Interpolator->Evaluate( transformedPoint ));
          
                  if (!m_BoundsSetByUser || (movingValue > this->m_MovingLowerBound && movingValue <= this->m_MovingUpperBound))
                    {
                      // Ask derived class to store the pair.
                      if (!m_UseWeighting)
                      {
                        //printf("Matt: index=[%d, %d, %d], target=%f, resultValue=%f\n", (int)index[0], (int)index[1], (int)index[2], fixedValue, movingValue);
                        const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->AggregateCostFunctionPair(fixedValue, movingValue);
                      }
                      else
                      {
                        double weight = std::min<double>(fabs(this->m_FixedDistanceMapInterpolator->EvaluateAtIndex(index)), 
                                                         fabs(this->m_MovingDistanceMapInterpolator->EvaluateAtContinuousIndex(movingImageTransformedIndex))); 
                        
                        weight = std::min<double>(weight, m_WeightingDistanceThreshold)/m_WeightingDistanceThreshold; 
                        const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->AggregateCostFunctionPairWithWeighting(fixedValue, movingValue, weight); 
                      }
                      this->m_NumberOfFixedSamples++;
                    }       
                }
            }      

          // As we have transformed point, we can store the value. 
          transformedMovingImageIterator.Set((FixedImagePixelType)movingValue);

          // And increment the index.
          ++fixedImageIterator;
          ++transformedMovingImageIterator;
          
        } // end while

      // Now sum up the measure in derived class.
      measure = const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->FinalizeCostFunction();

      if(this->m_PrintOutMetricEvaluation)
        {
          niftkitkDebugMacro("GetValue(), Number of fixed samples:" << this->m_NumberOfFixedSamples << ", value of metric:" << niftk::ConvertToString(measure));
        }
      // We can optionally evaluate the measure the other way round.
      if (this->m_TwoSidedMetric && !this->m_MovingImageMask.IsNull())
        {
          // this means we need the inverse.
          LightObject::Pointer anotherTransform = this->GetTransform()->CreateAnother();
          try 
            {
              UCLBaseTransformType* transformCopy = dynamic_cast< UCLBaseTransformType * >( anotherTransform.GetPointer() );  
              UCLBaseTransformType* uclTransformPointer = dynamic_cast<UCLBaseTransformType*>(this->m_Transform.GetPointer());
              if (transformCopy == 0 || uclTransformPointer == 0)
                {
                  niftkitkErrorMacro(<< "Failed to cast transform:" << this->GetTransform() << ", so skipping two sided cost function evaluation");
                }
                else if (!uclTransformPointer->GetInv(transformCopy))
                {
                  niftkitkErrorMacro(<< "Transform:" << this->GetTransform() << ", is not invertible, so skipping two sided cost function evaluation");
                }
            else
              {
                MeasureType movingMeasure = NumericTraits< MeasureType >::Zero;
                const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->ResetCostFunction();
                typedef itk::ImageRegionConstIteratorWithIndex<TMovingImage> MovingIteratorType;
                MovingIteratorType movingIterator(this->m_MovingImage, this->m_MovingImage->GetLargestPossibleRegion());
                typename MovingImageType::IndexType movingIndex;
                this->m_Interpolator->SetInputImage(this->m_FixedImage);
                
                movingIterator.GoToBegin();
                while(!movingIterator.IsAtEnd())
                  {
                    movingIndex = movingIterator.GetIndex();
                    this->m_MovingImage->TransformIndexToPhysicalPoint( movingIndex, inputPoint );
                    
                    // Use input mask.
                    if(!this->m_MovingImageMask.IsNull() && 
                       !this->m_MovingImageMask->IsInside( inputPoint ) )
                      {
                        ++movingIterator;
                        continue;
                      }
      
                    transformCopy->TransformPoint( inputPoint, transformedPoint );
                  
                    if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
                      {
                        movingValue = movingIterator.Get();
                        if (!m_BoundsSetByUser || (movingValue > this->m_MovingLowerBound && movingValue <= this->m_MovingUpperBound))
                          {
                            fixedValue = (FixedImagePixelType)(this->m_Interpolator->Evaluate( transformedPoint ));
                    
                            if (!m_BoundsSetByUser || (fixedValue > this->m_FixedLowerBound && fixedValue <= this->m_FixedUpperBound))
                              {
                                // Ask derived class to store the pair.
                                const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->AggregateCostFunctionPair(fixedValue, movingValue);
                                this->m_NumberOfMovingSamples++;
                              }      
                          }
                      }        
                   ++movingIterator;
                }
          
                // Now sum up the measure in derived class.
                movingMeasure = const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->FinalizeCostFunction();
      
                if(this->m_PrintOutMetricEvaluation)
                  {
                    niftkitkDebugMacro("GetValue(), Number of moving samples:" << this->m_NumberOfMovingSamples << ", value of metric:" << niftk::ConvertToString(movingMeasure));
                  }
                  
                // Take mean of the two values;
                measure = (measure + movingMeasure)/2.0;            
              }
            }
          catch( ExceptionObject& err )
            {
              niftkitkErrorMacro(<< "Caught exception " << err);
              niftkitkErrorMacro(<< "Caught exception while casting to invertible transform, so skipping two sided cost function evaluation");
            }
        }

      m_TransformedMovingImage->Modified();
      
      if (m_WriteTransformedMovingImage)
        {
          this->WriteImage(this->m_TransformedMovingImage,
              m_TransformedMovingImageFileName 
              + "." + niftk::ConvertToString((int)m_IterationNumber)
              + "." + m_TransformedMovingImageFileExt);
        }
      
      if (m_WriteFixedImage)
        {
          this->WriteImage(this->m_FixedImage.GetPointer(), 
              m_FixedImageFileName
              + "." + niftk::ConvertToString((int)m_IterationNumber)
              + "." + m_FixedImageFileExt);
        }
      
      m_IterationNumber++;

    } // end if (m_DirectVoxelComparison)
    
  // niftkitkDebugMacro("time spent=" << ((clock()-start)*1000)/CLOCKS_PER_SEC);
    
  return measure;
}

template <class TFixedImage, class TMovingImage> 
typename SimilarityMeasure<TFixedImage,TMovingImage>::MeasureType 
SimilarityMeasure<TFixedImage, TMovingImage>
::GetSymmetricSimilarity(const TransformParametersType & parameters)
{
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  FixedImagePixelType zero = NumericTraits< FixedImagePixelType >::Zero;
  
  // Reset the derived class. 
  ResetCostFunction();

  // We will count the number of samples from looping over the image masks.  
  this->m_NumberOfFixedSamples = 0;
  this->m_NumberOfMovingSamples = 0;
  
  FixedImagePixelType fixedValue = 0;
  MovingImagePixelType movingValue = 0;
  
  InputPointType inputPoint;
  OutputPointType transformedPoint;
  
  typedef itk::ImageRegionConstIteratorWithIndex<TFixedImage> IteratorType;
  typedef itk::ImageRegionIterator<TFixedImage> TransformedIteratorType;
  
  typename TFixedImage::RegionType fixedRegion;
  typename FixedImageType::IndexType index;
  
  // First we will iterate over the whole of the fixed image.
  fixedRegion = this->GetFixedImage()->GetLargestPossibleRegion();
  IteratorType fixedImageIterator(this->m_FixedImage, fixedRegion);
  
  // We also iterate over m_TransformedMovingImage.
  // This is the same size as FixedImage, and is the moving 
  // image resampled into fixed image coordinate space.
  TransformedIteratorType transformedMovingImageIterator(this->m_TransformedMovingImage, fixedRegion); 
  
  TransformType *transform = NULL;
       
  if (m_IsUpdateMatrix)
  {
    this->SetTransformParameters( parameters );
  }
  transform = this->m_Transform;

  this->m_Interpolator->SetInputImage(this->m_MovingImage);
  
  for (fixedImageIterator.GoToBegin(), transformedMovingImageIterator.GoToBegin(); 
       !fixedImageIterator.IsAtEnd() && !transformedMovingImageIterator.IsAtEnd(); 
       ++fixedImageIterator, ++transformedMovingImageIterator)
  {
    transformedMovingImageIterator.Set(zero); 
    index = fixedImageIterator.GetIndex();
    this->m_FixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

    // Use input mask.
    if(!this->m_FixedImageMask.IsNull() && 
        !this->m_FixedImageMask->IsInside( inputPoint ) )
    {
      transformedMovingImageIterator.Set(m_TransformedMovingImagePadValue);
      continue;
    }

    transformedPoint = transform->TransformPoint( inputPoint );

    if(!this->m_MovingImageMask.IsNull() && 
        !this->m_MovingImageMask->IsInside( transformedPoint ) )
    {
      transformedMovingImageIterator.Set(m_TransformedMovingImagePadValue);
      continue;
    }

    fixedValue = 0;
    movingValue = m_TransformedMovingImagePadValue;
          
    if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
    {
      fixedValue = fixedImageIterator.Get();
  
      if (fixedValue > this->m_FixedLowerBound && fixedValue <= this->m_FixedUpperBound)
        {
          movingValue = (MovingImagePixelType)(this->m_Interpolator->Evaluate( transformedPoint ));
  
          if (movingValue > this->m_MovingLowerBound && movingValue <= this->m_MovingUpperBound)
            {
              // Ask derived class to store the pair.
              AggregateCostFunctionPair(fixedValue, movingValue);
              this->m_NumberOfFixedSamples++;
            }       
        }
    }         
          
    // As we have transformed point, we can store the value. 
    transformedMovingImageIterator.Set((FixedImagePixelType)movingValue);
  }
    
  m_TransformedMovingImage->Modified();
  // Now sum up the measure in derived class.
  measure = FinalizeCostFunction();

  niftkitkDebugMacro("GetValue(), Number of fixed samples:" << this->m_NumberOfFixedSamples << ", value of metric:" << niftk::ConvertToString(measure));

  typename AffineTransformType::Pointer inverseTransform = AffineTransformType::New(); 
  AffineTransformType* currentTransform = dynamic_cast<AffineTransformType*>(this->m_Transform.GetPointer());  
  currentTransform->GetInv(inverseTransform.GetPointer()); 
  
  MeasureType movingMeasure = NumericTraits< MeasureType >::Zero;
  ResetCostFunction();
  typedef itk::ImageRegionConstIteratorWithIndex<TMovingImage> MovingIteratorType;
  MovingIteratorType movingIterator(this->m_MovingImage, this->m_MovingImage->GetLargestPossibleRegion());
  typename MovingImageType::IndexType movingIndex;
  this->m_Interpolator->SetInputImage(this->m_FixedImage);
  
  for (movingIterator.GoToBegin(); !movingIterator.IsAtEnd(); ++movingIterator)
  {
    movingIndex = movingIterator.GetIndex();
    this->m_MovingImage->TransformIndexToPhysicalPoint( movingIndex, inputPoint );
      
    // Use input mask.
    if(!this->m_MovingImageMask.IsNull() && 
        !this->m_MovingImageMask->IsInside( inputPoint ) )
    {
      continue;
    }

    transformedPoint = inverseTransform->TransformPoint(inputPoint);
    
    if(!this->m_FixedImageMask.IsNull() && 
        !this->m_FixedImageMask->IsInside( transformedPoint ) )
    {
        continue;
    }
    
    if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
    {
      movingValue = movingIterator.Get();
      if (movingValue > this->m_MovingLowerBound && movingValue <= this->m_MovingUpperBound)
        {
          fixedValue = (FixedImagePixelType)(this->m_Interpolator->Evaluate( transformedPoint ));
  
          if (fixedValue > this->m_FixedLowerBound && fixedValue <= this->m_FixedUpperBound)
            {
              // Ask derived class to store the pair.
              AggregateCostFunctionPair(fixedValue, movingValue);
              this->m_NumberOfMovingSamples++;
            }      
        }
    }        
  }

  // Now sum up the measure in derived class.
  movingMeasure = FinalizeCostFunction();

  niftkitkDebugMacro("GetValue(), Number of moving samples:" << this->m_NumberOfMovingSamples << ", value of metric:" << niftk::ConvertToString(movingMeasure));
    
  // Take mean of the two values;
  measure = (measure + movingMeasure)/2.0;            

  if (m_WriteTransformedMovingImage)
    {
      this->WriteImage(this->m_TransformedMovingImage,
          m_TransformedMovingImageFileName 
          + "." + niftk::ConvertToString((int)m_IterationNumber)
          + "." + m_TransformedMovingImageFileExt);
    }
  
  if (m_WriteFixedImage)
    {
      this->WriteImage(this->m_FixedImage.GetPointer(),
          m_FixedImageFileName
          + "." + niftk::ConvertToString((int)m_IterationNumber) 
          + "." + m_FixedImageFileExt);
    }
  
  m_IterationNumber++;
  return measure;
}


template <class TFixedImage, class TMovingImage> 
typename SimilarityMeasure<TFixedImage,TMovingImage>::MeasureType 
SimilarityMeasure<TFixedImage, TMovingImage>
::GetSymmetricSimilarityAtHalfway( const TransformParametersType & parameters )
{
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  typedef itk::ImageRegionConstIteratorWithIndex<MidwayImageType> IteratorType;
  
  this->SetTransformParameters(parameters); 
  
  // Reset the derived class. 
  ResetCostFunction();

  // We will count the number of samples from looping over the image masks.  
  this->m_NumberOfFixedSamples = 0;
  this->m_NumberOfMovingSamples = 0;
  
  FixedImagePixelType fixedValue = 0;
  MovingImagePixelType movingValue = 0;
  
  InputPointType inputPoint;
  OutputPointType fixedImageTransformedPoint;
  OutputPointType movingImageTransformedPoint;
  ContinuousIndex<double, TFixedImage::ImageDimension> fixedImageTransformedIndex; 
  ContinuousIndex<double, TMovingImage::ImageDimension> movingImageTransformedIndex; 
  typename TFixedImage::IndexType fixedMaskTransformedIndex; 
  typename TMovingImage::IndexType movingMaskTransformedIndex; 
      
  this->m_FixedImageInterpolator->SetInputImage(this->m_FixedImage);
  this->m_MovingImageInterpolator->SetInputImage(this->m_MovingImage);
  
  // Create the inverse transforms for the fixed image. 
  typename AffineTransformType::Pointer inverseTransform = AffineTransformType::New(); 
  AffineTransformType* transform = dynamic_cast<AffineTransformType*>(this->m_Transform.GetPointer());  
  transform->GetInv(inverseTransform.GetPointer()); 
  
  FixedMaskType* fixedMask = dynamic_cast<FixedMaskType*>(this->m_FixedImageMask.GetPointer()); 
  MovingMaskType* movingMask = dynamic_cast<MovingMaskType*>(this->m_MovingImageMask.GetPointer()); 
  
  // clock_t start = clock(); 
  
  // We will iterate over the mid-way image.
  IteratorType midwayImageIterator(this->m_MidwayImage, this->m_MidwayImage->GetLargestPossibleRegion());
  for (midwayImageIterator.GoToBegin(); !midwayImageIterator.IsAtEnd(); ++midwayImageIterator)
  {
    this->m_MidwayImage->TransformIndexToPhysicalPoint(midwayImageIterator.GetIndex(), inputPoint);
    
    // Transform the point to the fixed image. 
    fixedImageTransformedPoint = inverseTransform->TransformPoint(inputPoint);
    if (this->m_FixedImageMask.IsNull())
    {
      if (!this->m_FixedImageInterpolator->IsInsideBuffer(fixedImageTransformedPoint))
        continue; 
    }
    else 
    {
      if (!fixedMask->GetImage()->TransformPhysicalPointToIndex(fixedImageTransformedPoint, fixedMaskTransformedIndex) ||
         fixedMask->GetImage()->GetPixel(fixedMaskTransformedIndex) == 0)
      {
          continue; 
      } 
    }
    // Transform the point to the moving image. 
    movingImageTransformedPoint = transform->TransformPoint(inputPoint);
    if (this->m_MovingImageMask.IsNull())
    {
      if (this->m_MovingImageInterpolator->IsInsideBuffer(movingImageTransformedPoint))
        continue; 
    }
    else 
    {
      if (!movingMask->GetImage()->TransformPhysicalPointToIndex(movingImageTransformedPoint, movingMaskTransformedIndex) || 
           movingMask->GetImage()->GetPixel(movingMaskTransformedIndex) == 0)
      {
        continue; 
      } 
    } 
    
    this->m_FixedImage->TransformPhysicalPointToContinuousIndex(fixedImageTransformedPoint, fixedImageTransformedIndex);
    fixedValue = static_cast<FixedImagePixelType>(this->m_FixedImageInterpolator->EvaluateAtContinuousIndex(fixedImageTransformedIndex)); 
    if (fixedValue > this->m_FixedLowerBound && fixedValue <= this->m_FixedUpperBound)
    {
      this->m_MovingImage->TransformPhysicalPointToContinuousIndex(movingImageTransformedPoint, movingImageTransformedIndex);
      movingValue = (MovingImagePixelType)(this->m_MovingImageInterpolator->EvaluateAtContinuousIndex(movingImageTransformedIndex));
      if (movingValue > this->m_MovingLowerBound && movingValue <= this->m_MovingUpperBound)
      {
        // Ask derived class to store the pair.
        if (!m_UseWeighting)
        {
          AggregateCostFunctionPair(fixedValue, movingValue);
        }
        else
        {
          double weight = std::min<double>(fabs(this->m_FixedDistanceMapInterpolator->EvaluateAtContinuousIndex(fixedImageTransformedIndex)), 
                                           fabs(this->m_MovingDistanceMapInterpolator->EvaluateAtContinuousIndex(movingImageTransformedIndex))); 
          
          weight = std::min<double>(weight, m_WeightingDistanceThreshold)/m_WeightingDistanceThreshold; 
          AggregateCostFunctionPairWithWeighting(fixedValue, movingValue, weight); 
        }
        this->m_NumberOfFixedSamples++;
      }       
    }
  }       
    
  // niftkitkDebugMacro("time spent=" << ((clock()-start)*1000)/CLOCKS_PER_SEC);
  
  // Now sum up the measure in derived class.
  measure = FinalizeCostFunction();
  if (this->m_PrintOutMetricEvaluation)
  {
    niftkitkDebugMacro("GetValue(), Number of fixed samples:" << this->m_NumberOfFixedSamples << ", value of metric:" << niftk::ConvertToString(measure));
  }
  this->m_IterationNumber++; 
    
  return measure;
}
  
template <class TFixedImage, class TMovingImage>
double
SimilarityMeasure<TFixedImage, TMovingImage>
::GetMeasureOfParameterChange(TransformParametersType lastP, TransformParametersType p)
{
  const double Radius = 100.0; 
  typename AffineTransformType::Pointer lastTransform = AffineTransformType::New();
  typename AffineTransformType::Pointer currentTransform = AffineTransformType::New();

  lastTransform->SetNumberOfDOF(lastP.GetSize()); 
  currentTransform->SetNumberOfDOF(p.GetSize()); 

  lastTransform->SetParameters(lastP); 
  currentTransform->SetParameters(p); 
 
  typename AffineTransformType::FullAffineMatrixType identityMatrix; 
  identityMatrix.SetIdentity(); 
  typename AffineTransformType::FullAffineMatrixType lastMatrix = lastTransform->GetFullAffineMatrix();
  typename AffineTransformType::FullAffineMatrixType currentMatrix = currentTransform->GetFullAffineMatrix();
  typename AffineTransformType::FullAffineMatrixType lastMatrixInverse(lastMatrix.GetInverse()); 
  typename AffineTransformType::FullAffineMatrixType diffMatrix = currentMatrix*lastMatrixInverse-identityMatrix;  

  // Divide the difference into matrix M and vector T.
  const unsigned int ColumnDimensions = AffineTransformType::FullAffineMatrixType::ColumnDimensions; 
  const unsigned int RowDimensions = AffineTransformType::FullAffineMatrixType::RowDimensions; 
  // Affine matrix without translations, its transpose and the product. 
  vnl_matrix_fixed<typename AffineTransformType::FullAffineMatrixType::ValueType, ColumnDimensions-1, RowDimensions-1> matrixM; 
  vnl_matrix_fixed<typename AffineTransformType::FullAffineMatrixType::ValueType, ColumnDimensions-1, RowDimensions-1> matrixMTranspose; 
  vnl_matrix_fixed<typename AffineTransformType::FullAffineMatrixType::ValueType, ColumnDimensions-1, RowDimensions-1> matrixMTransposeM; 
  // Translation vector. 
  vnl_vector<typename AffineTransformType::FullAffineMatrixType::ValueType> translation(RowDimensions-1); 
  
  for (unsigned int row = 0; row < RowDimensions-1; row++)
  {
    for (unsigned int col = 0; col < ColumnDimensions-1; col++)
    {
      matrixM(row, col) = diffMatrix(row, col); 
      matrixMTranspose(col, row) = diffMatrix(row, col); 
    }
    translation[row] = diffMatrix(row, ColumnDimensions-1); 
  }
  matrixMTransposeM = matrixMTranspose*matrixM; 
  niftkitkDebugMacro("diffMatrix=" << std::endl << diffMatrix);
  niftkitkDebugMacro("matrixM=" << std::endl << matrixM);
  niftkitkDebugMacro("translation=" << translation);

  // Change the origin to the centre of the image. 
  vnl_vector<typename AffineTransformType::FullAffineMatrixType::ValueType> centre(RowDimensions-1); 
  vnl_vector<typename AffineTransformType::FullAffineMatrixType::ValueType> shiftedTranslation(RowDimensions-1); 
  AffineTransformType* transform = dynamic_cast<AffineTransformType*>(this->m_Transform.GetPointer());
  InputPointType centrePoint = transform->GetCenter();
  niftkitkDebugMacro("relative to center=" << centrePoint);

  for (unsigned int i = 0; i < RowDimensions-1; i++)
  {
    centre[i] = centrePoint[i]; 
  }
  shiftedTranslation = translation + matrixM*centre; 

  double trace = 0.0;  
  double translationError = 0.0; 
  for (unsigned int row = 0; row < RowDimensions-1; row++)
  {
    trace += matrixMTransposeM(row, row); 
    translationError += shiftedTranslation[row]*shiftedTranslation[row]; 
  }
  return vcl_sqrt(0.2*Radius*Radius*trace + translationError); 
 

}



template <class TFixedImage, class TMovingImage> 
typename SimilarityMeasure<TFixedImage,TMovingImage>::MeasureType 
SimilarityMeasure<TFixedImage, TMovingImage>
::GetSimilarityUsingFixedAndMovingImageTransforms(const TransformParametersType & parameters)
{
  // niftkitkInfoMacro("GetSimilarityUsingFixedAndMovingImageTransforms(): started");
  
  // Reset the derived class. 
  ResetCostFunction();
  
  typedef itk::ImageRegionConstIteratorWithIndex<TFixedImage> IndexIteratorType;
  typedef itk::ImageRegionIterator<TFixedImage> NonIndexIteratorType;
  
  typename TFixedImage::RegionType fixedRegion;
  fixedRegion = this->GetFixedImage()->GetLargestPossibleRegion();
    
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  FixedImagePixelType fixedValue = 0;
  MovingImagePixelType movingValue = 0;
  typename TFixedImage::IndexType fixedMaskTransformedIndex; 
  typename TMovingImage::IndexType movingMaskTransformedIndex; 
  FixedImagePixelType zero = NumericTraits< FixedImagePixelType >::Zero;
  InputPointType inputPoint;
  OutputPointType fixedImageTransformedPoint;
  OutputPointType movingImageTransformedPoint;
  ContinuousIndex<double, TMovingImage::ImageDimension> fixedImageTransformedIndex; 
  ContinuousIndex<double, TMovingImage::ImageDimension> movingImageTransformedIndex; 

  // We will count the number of samples from looping over the image masks.  
  this->m_NumberOfFixedSamples = 0;
  this->m_NumberOfMovingSamples = 0;
  
  typename FixedImageType::IndexType index;
  IndexIteratorType fixedImageIterator(this->m_FixedImage, fixedRegion);
    
  // We iterate over m_TransformedMovingImage and m_TransformedFixedImage.
  // This is the same size as FixedImage, and is the moving 
  // image resampled into fixed image coordinate space.
  NonIndexIteratorType transformedMovingImageIterator(this->m_TransformedMovingImage, fixedRegion); 
  NonIndexIteratorType transformedFixedImageIterator(this->m_TransformedFixedImage, fixedRegion); 
    
  // To replace the slow SpatialObject::IsInside by directly testing it on the image masks. 
  FixedMaskType* fixedMask = dynamic_cast<FixedMaskType*>(this->m_FixedImageMask.GetPointer()); 
  MovingMaskType* movingMask = dynamic_cast<MovingMaskType*>(this->m_MovingImageMask.GetPointer()); 

  TransformType* movingImageTransform = this->m_Transform;
  this->m_MovingImageInterpolator->SetInputImage(this->m_MovingImage);
  
  TransformType* fixedImageTransform = this->m_FixedImageTransform; 
  this->m_FixedImageInterpolator->SetInputImage(this->m_FixedImage); 
  
  // Go through the image grid.   
  for (fixedImageIterator.GoToBegin(), transformedMovingImageIterator.GoToBegin(), transformedFixedImageIterator.GoToBegin();
       !fixedImageIterator.IsAtEnd(); 
       ++fixedImageIterator, ++transformedMovingImageIterator, ++transformedFixedImageIterator)
  {
    bool isInsideMask = true; 
    index = fixedImageIterator.GetIndex();
    this->m_FixedImage->TransformIndexToPhysicalPoint(index, inputPoint);

    movingImageTransformedPoint = movingImageTransform->TransformPoint(inputPoint);
    fixedImageTransformedPoint = fixedImageTransform->TransformPoint(inputPoint);
    
    // Use input mask. 
    if (!this->m_FixedImageMask.IsNull() && 
        (!fixedMask->GetImage()->TransformPhysicalPointToIndex(fixedImageTransformedPoint, fixedMaskTransformedIndex) ||
          fixedMask->GetImage()->GetPixel(fixedMaskTransformedIndex) == 0))
    {
      transformedMovingImageIterator.Set(m_TransformedMovingImagePadValue);
      transformedFixedImageIterator.Set(m_TransformedMovingImagePadValue);
      isInsideMask = false; ; 
    }
    if (!this->m_MovingImageMask.IsNull() && isInsideMask && 
        (!movingMask->GetImage()->TransformPhysicalPointToIndex(movingImageTransformedPoint, movingMaskTransformedIndex) || 
          movingMask->GetImage()->GetPixel(movingMaskTransformedIndex) == 0))
    {
      transformedMovingImageIterator.Set(m_TransformedMovingImagePadValue);
      transformedFixedImageIterator.Set(m_TransformedMovingImagePadValue);
      isInsideMask = false; ; 
    }
    
    if (!this->m_IsResampleWholeImage && !isInsideMask)
    {
      continue;
    }

    fixedValue = zero;
    movingValue = zero;
          
    if (this->m_MovingImageInterpolator->IsInsideBuffer(movingImageTransformedPoint) &&
        this->m_FixedImageInterpolator->IsInsideBuffer(fixedImageTransformedPoint))
    {
      fixedValue = fixedImageIterator.Get();
      
      this->m_FixedImage->TransformPhysicalPointToContinuousIndex(fixedImageTransformedPoint, fixedImageTransformedIndex);
      fixedValue = (FixedImagePixelType)(this->m_FixedImageInterpolator->EvaluateAtContinuousIndex(fixedImageTransformedIndex));
      
      this->m_MovingImage->TransformPhysicalPointToContinuousIndex(movingImageTransformedPoint, movingImageTransformedIndex);
      movingValue = (MovingImagePixelType)(this->m_MovingImageInterpolator->EvaluateAtContinuousIndex(movingImageTransformedIndex));
      
      if ((!m_BoundsSetByUser || (fixedValue > this->m_FixedLowerBound && fixedValue <= this->m_FixedUpperBound)) && 
          (!m_BoundsSetByUser || (movingValue > this->m_MovingLowerBound && movingValue <= this->m_MovingUpperBound)))
      {
        if (isInsideMask)
        {
          const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->AggregateCostFunctionPair(fixedValue, movingValue);
        }
      }       
    }

    this->m_NumberOfFixedSamples++; 
    // As we have transformed point, we can store the value. 
    transformedMovingImageIterator.Set((FixedImagePixelType)movingValue);
    transformedFixedImageIterator.Set((FixedImagePixelType)fixedValue);

  } 

  // Now sum up the measure in derived class.
  measure = const_cast< SimilarityMeasure<TFixedImage, TMovingImage>* >(this)->FinalizeCostFunction();
  
  // niftkitkInfoMacro("GetSimilarityUsingFixedAndMovingImageTransforms(): done, this->m_NumberOfFixedSamples=" << this->m_NumberOfFixedSamples);
  return measure;  
}
















} // end namespace itk

#endif
