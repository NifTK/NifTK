/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkBlockMatchingMethod_txx
#define _itkBlockMatchingMethod_txx

#include "itkLogHelper.h"
#include "itkBlockMatchingMethod.h"
#include "itkImageFileWriter.h"
#include "ConversionUtils.h"

namespace itk
{
/*
 * Constructor
 */
template < typename TImageType, class TScalarType >
BlockMatchingMethod<TImageType, TScalarType>
::BlockMatchingMethod()
: MaskedImageRegistrationMethod<TImageType>()
{
  m_MovingImageResampler = ResampleFilterType::New();
  m_FixedImageRegionFilter = RegionOfInterestFilterType::New();
  m_DummyInterpolator = DummyInterpolatorType::New();
  m_FixedPointSet = PointSetType::New();
  m_MovingPointSet = PointSetType::New();
  m_FixedPointSetContainer = PointsContainerType::New();
  m_MovingPointSetContainer = PointsContainerType::New();
  m_FixedImageListAdaptor = ImageTypeListAdaptorType::New();
  m_FixedImageCovarianceCalculator = ImageTypeCovarianceCalculatorType::New();
  m_GradientMagnitudeImageFilter = GradientMagnitudeFilterType::New();
  m_GradientMagnitudeListAdaptor = GradientImageTypeListAdaptorType::New();
  m_GradientMagnitudeCovarianceCalculator = GradientImageTypeCovarianceCalculatorType::New();
  m_MinMaxCalculator = MinimumMaximumImageCalculatorType::New();
  
  m_MaximumNumberOfIterationsRoundMainLoop = 10;
  m_BlockSize = -1;
  m_BlockHalfWidth = -1;
  m_BlockSpacing = -1;
  m_BlockSubSampling = -1;
  m_Epsilon = 1;
  m_ParameterReductionFactor = 0.5;
  m_MinimumBlockSize = 4.0;
  m_PercentageOfPointsToKeep = 100;
  m_PercentageOfPointsInLeastTrimmedSquares = 100;
  m_ScaleByMillimetres = false;
  m_WriteTransformedMovingImage = false;
  m_TransformedMovingImageFileName = "tmp.block.resampled";
  m_TransformedMovingImageFileExt = "nii";
  m_PointSetFileNameWithoutExtension = "tmp.block.points";
  m_WritePointSet = false;
  m_NoZero = false;
  m_TransformedMovingImagePadValue = 0;
  
  niftkitkDebugMacro(<<"BlockMatchingMethod():Constructed, with m_MaximumNumberOfIterationsRoundMainLoop=" << m_MaximumNumberOfIterationsRoundMainLoop \
    << ", m_BlockSize:" << m_BlockSize \
    << ", m_BlockHalfWidth:" << m_BlockHalfWidth \
    << ", m_BlockSpacing:" << m_BlockSpacing \
    << ", m_BlockSubSampling:" << m_BlockSubSampling \
    << ", m_Epsilon:" << m_Epsilon \
    << ", m_ParameterReductionFactor:" << m_ParameterReductionFactor \
    << ", m_MinimumBlockSize:" << m_MinimumBlockSize \
    << ", m_PercentageOfPointsToKeep:" << m_PercentageOfPointsToKeep \
    << ", m_PercentageOfPointsInLeastTrimmedSquares=" << m_PercentageOfPointsInLeastTrimmedSquares \
    << ", m_ScaleByMillimetres:" << m_ScaleByMillimetres \
    << ", m_WriteTransformedMovingImage:" << m_WriteTransformedMovingImage \
    << ", m_TransformedMovingImageFileName:" << m_TransformedMovingImageFileName \
    << ", m_TransformedMovingImageFileExt:" << m_TransformedMovingImageFileExt \
    << ", m_PointSetFileNameWithoutExtension=" << m_PointSetFileNameWithoutExtension \
    << ", m_WritePointSet=" << m_WritePointSet \
    << ", m_NoZero=" << m_NoZero \
    << ", m_TransformedMovingImagePadValue=" << m_TransformedMovingImagePadValue \
    );
}

/*
 * PrintSelf
 */
template < typename TImageType, class TScalarType  >
void
BlockMatchingMethod<TImageType, TScalarType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  if (!m_MovingImageResampler.IsNull())
    {
      os << indent << "MovingImageResampler="<< m_MovingImageResampler << std::endl;
    }
  if (!m_FixedImageRegionFilter.IsNull())
    {
      os << indent << "FixedImageRegionFilter="<< m_FixedImageRegionFilter << std::endl;
    }
  os << indent << "MaximumNumberOfIterationsRoundMainLoop="<< m_MaximumNumberOfIterationsRoundMainLoop << std::endl;  
  os << indent << "BlockSize="<< m_BlockSize << std::endl;  
  os << indent << "BlockHalfWidth="<< m_BlockHalfWidth << std::endl;  
  os << indent << "BlockSpacing="<< m_BlockSpacing << std::endl;    
  os << indent << "BlockSubSampling="<< m_BlockSubSampling << std::endl;
  os << indent << "Epsilon="<< m_Epsilon << std::endl;
  os << indent << "ParameterReductionFactor="<< m_ParameterReductionFactor << std::endl;
  os << indent << "MinimumBlockSize="<< m_MinimumBlockSize << std::endl;
  os << indent << "PercentageOfPointsToKeep=" << m_PercentageOfPointsToKeep << std::endl;
  os << indent << "PercentageOfPointsInLeastTrimmedSquares=" << m_PercentageOfPointsInLeastTrimmedSquares << std::endl;
  os << indent << "ScaleByMillimetres=" << m_ScaleByMillimetres << std::endl;
  os << indent << "PercentageOfPointsToKeep="<< m_PercentageOfPointsToKeep << std::endl;
  os << indent << "WriteTransformedMovingImage="<< m_WriteTransformedMovingImage << std::endl;
  os << indent << "TransformedMovingImageFileName="<< m_TransformedMovingImageFileName << std::endl;
  os << indent << "TransformedMovingImageFileExt="<< m_TransformedMovingImageFileExt << std::endl;   
  os << indent << "PointSetFileNameWithoutExtension="<< m_PointSetFileNameWithoutExtension << std::endl;
  os << indent << "WritePointSet="<< m_WritePointSet << std::endl;
  os << indent << "NoZero="<< m_NoZero << std::endl;
  os << indent << "TransformedMovingImagePadValue="<< m_TransformedMovingImagePadValue << std::endl;
}

template < typename TImageType, class TScalarType >
void
BlockMatchingMethod<TImageType, TScalarType >
::SetBlockParameters(
  double blockSize,
  double blockHalfWidth,
  double blockSpacing,
  double blockSubSampling)
  {
    this->m_BlockSize = blockSize;
    this->m_BlockHalfWidth = blockHalfWidth;
    this->m_BlockSpacing = blockSpacing;
    this->m_BlockSubSampling = blockSubSampling;
  }

/*
 * The bit that does the wiring together.
 */
template < typename TImageType, class TScalarType >
void
BlockMatchingMethod<TImageType, TScalarType >
::Initialize() throw (ExceptionObject)
{

  niftkitkDebugMacro(<<"Initialize():Started.");
  
  Superclass::Initialize();
  
  // Sanity checks
  if (!this->GetMetric())
    {
      itkExceptionMacro(<<"Metric is not present" );
    }

  if (m_PointSetMetric.IsNull())
    {
      itkExceptionMacro(<<"Point set metric is not present" );
    }

  if(!this->GetTransform())
    {
      itkExceptionMacro(<<"Transform is not present");
    }

  if(!this->GetInterpolator())
    {
      itkExceptionMacro(<<"Interpolator is not present");
    }

  if(!this->GetOptimizer())
    {
      itkExceptionMacro(<<"Optimizer is not present");
    }
  
  m_MovingImageResampler->SetInput(this->GetMovingImageCopy()); // This forces us to use a copy of the non-masked, original moving image.

#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  m_MovingImageResampler->SetOutputParametersFromImage(const_cast<TImageType*>(this->GetFixedImage()));
#else
  m_MovingImageResampler->SetOutputParametersFromImage(this->GetFixedImage());  
#endif
  m_MovingImageResampler->SetTransform(this->GetTransform());
  m_MovingImageResampler->SetInterpolator(this->GetInterpolator());
  m_MovingImageResampler->SetDefaultPixelValue(m_TransformedMovingImagePadValue);

  m_FixedImageRegionFilter->SetInput(this->GetFixedImage());
  m_FixedImageRegionFilter->SetNumberOfThreads(1);

  // We will either be calculating covariance based on image intensity.
  m_FixedImageListAdaptor->SetImage(m_FixedImageRegionFilter->GetOutput());
  m_FixedImageCovarianceCalculator->SetInputSample(m_FixedImageListAdaptor);
  
  // Or we will be calculating covariance based on gradient magnitude image intensity.
  m_GradientMagnitudeImageFilter->SetInput(m_FixedImageRegionFilter->GetOutput());
  m_GradientMagnitudeListAdaptor->SetImage(m_GradientMagnitudeImageFilter->GetOutput());
  m_GradientMagnitudeCovarianceCalculator->SetInputSample(m_GradientMagnitudeListAdaptor);
  
  this->GetMetric()->SetTransform( this->GetTransform() );
  this->GetMetric()->SetInterpolator( this->m_DummyInterpolator);
  this->GetMetric()->SetComputeGradient(false);
  
  // We set the original fixed image and moving image onto the metric, so it can set the intensity bounds.
  this->GetMetric()->SetFixedImage(this->GetFixedImageCopy());
  this->GetMetric()->SetMovingImage(this->GetMovingImageCopy());
  
  (static_cast<SimilarityMeasurePointer>(this->GetMetric()))->SetDirectVoxelComparison(true);
  (static_cast<SimilarityMeasurePointer>(this->GetMetric()))->InitializeIntensityBounds();

  // And then we set it back, so its connected only to the region filters for fixed image.
  this->GetMetric()->SetFixedImage( m_FixedImageRegionFilter->GetOutput() );
  this->GetMetric()->SetMovingImage( m_MovingImageResampler->GetOutput() );
  
  niftkitkDebugMacro(<<"Initialize():Finished.");
}

template < typename TImageType, class TScalarType >
double
BlockMatchingMethod<TImageType, TScalarType>
::CheckSinglePoint(ParametersType& previousParameters, ParametersType& currentParameters, ImageIndexType& index)
{
  ContinuousIndex< TScalarType, TImageType::ImageDimension > movingPointInVoxelCoordinates;
  PointType                                                  movingPointInMillimetreCoordinates;
  PointType                                                  previousMovingPointInMillimetreCoordinates;
  PointType                                                  currentMovingPointInMillimetreCoordinates;
  
  for (unsigned int i = 0; i < TImageType::ImageDimension; i++)
    {
      movingPointInVoxelCoordinates[i] = index[i];
    }
  
  this->GetMovingImage()->TransformContinuousIndexToPhysicalPoint(movingPointInVoxelCoordinates, movingPointInMillimetreCoordinates);
  
  // Transform by previous transform
  this->GetTransform()->SetParameters(previousParameters);
  previousMovingPointInMillimetreCoordinates = this->GetTransform()->TransformPoint(movingPointInMillimetreCoordinates);
  
  // Transform by current parameters
  this->GetTransform()->SetParameters(currentParameters);
  currentMovingPointInMillimetreCoordinates = this->GetTransform()->TransformPoint(movingPointInMillimetreCoordinates);
  
  // Measure the Euclidean Distance between them.
  double result = previousMovingPointInMillimetreCoordinates.EuclideanDistanceTo(currentMovingPointInMillimetreCoordinates);

  niftkitkDebugMacro(<<"CheckSinglePoint():index:" << index \
      << ", in millimetres:" << movingPointInMillimetreCoordinates \
      << ", gives previous:" << previousMovingPointInMillimetreCoordinates \
      << ", and current:" << currentMovingPointInMillimetreCoordinates \
      << ", and a distance of:" << result);
  
  return result;
}

template < typename TImageType, class TScalarType >
bool
BlockMatchingMethod<TImageType, TScalarType>
::CheckEpsilon(ParametersType& previousParameters, ParametersType& currentParameters)
{
  double smallDelta = 0;
  ImageIndexType index;
  
  unsigned int     dimensions = TImageType::ImageDimension;
  ImageSizeType    size = this->GetMovingImage()->GetLargestPossibleRegion().GetSize();
  ImageSpacingType spacing = this->GetMovingImage()->GetSpacing();
  
  niftkitkDebugMacro(<<"ShouldChangeScale():Moving image size=" << size << ", spacing=" << spacing);
  
  if (dimensions == 2)
    {
      index[0] = 0;            index[1] = 0;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = size[0] - 1;  index[1] = 0;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = 0;            index[1] = size[1] - 1;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = size[0] - 1;  index[1] = size[1] - 1;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);      
    }
  else
    {
      index[0] = 0;            index[1] = 0;            index[2] = 0;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = size[0] - 1;  index[1] = 0;            index[2] = 0;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = 0;            index[1] = size[1] - 1;  index[2] = 0;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = size[0] - 1;  index[1] = size[1] - 1;  index[2] = 0;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);      

      index[0] = 0;            index[1] = 0;            index[2] = size[2] - 1;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = size[0] - 1;  index[1] = 0;            index[2] = size[2] - 1;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = 0;            index[1] = size[1] - 1;  index[2] = size[2] - 1;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);
      
      index[0] = size[0] - 1;  index[1] = size[1] - 1;  index[2] = size[2] - 1;
      smallDelta += CheckSinglePoint(previousParameters, currentParameters, index);      
    }
  smallDelta /= (dimensions * dimensions);

  niftkitkDebugMacro(<<"CheckEpsilon():previousParameters=" << previousParameters \
      << ", currentParameters=" << currentParameters \
      << ", gave:" <<  smallDelta \
      << ", with m_Epsilon=" << m_Epsilon \
      << ", and m_ParameterReductionFactor=" << m_ParameterReductionFactor);

  if (smallDelta <= m_Epsilon && m_ParameterReductionFactor == 1)
    {
      niftkitkDebugMacro(<<"CheckEpsilon():Below epsilon tolerance and we aren't reducing block size, so stop");
      return false;
    }
  else if (smallDelta <= m_Epsilon)
    {
      m_BlockSize        = m_ParameterReductionFactor * m_BlockSize;
      m_BlockHalfWidth   = m_ParameterReductionFactor * m_BlockHalfWidth;
      m_BlockSpacing     = m_ParameterReductionFactor * m_BlockSpacing;
      m_BlockSubSampling = m_ParameterReductionFactor * m_BlockSubSampling;
      
      if (m_BlockSpacing < 1) 
        {
          m_BlockSpacing = 1;  
        }
      
      if (m_BlockSubSampling < 1)
        {
          m_BlockSubSampling = 1;
        }
      
      niftkitkDebugMacro(<<"CheckEpsilon():Changing scale, m_BlockSize=" << m_BlockSize \
          << ", m_BlockHalfWidth=" << m_BlockHalfWidth \
          << ", m_BlockSpacing=" << m_BlockSpacing \
          << ", m_BlockSubSampling=" << m_BlockSubSampling);
      return true;
    }
  else 
    {
      niftkitkDebugMacro(<<"CheckEpsilon():Above tolerance, so keep going");
      return true;
    }
  
}

template < typename TImageType, class TScalarType >
void
BlockMatchingMethod<TImageType, TScalarType>
::WritePointSet(const PointsContainerPointer& fixedPointContainer, const PointsContainerPointer& movingPointContainer)
{
  PointType fixedPointInMillimetreCoordinates;
  PointType movingPointInMillimetreCoordinates;
  PointType vector;
  
  //
  // Write to output file
  //
  std::string fileName = this->m_PointSetFileNameWithoutExtension + ".vtk";
  std::ofstream outputFile( fileName.c_str() );

  if( !outputFile.is_open() )
    {
    itkExceptionMacro("Unable to open file\n"
        "outputFilename= " << fileName );
    return;
    }

  outputFile << "# vtk DataFile Version 2.0" << std::endl;
  outputFile << "File written by itkBlockMatchingMethod.txx" << std::endl;
  outputFile << "ASCII" << std::endl;
  outputFile << "DATASET POLYDATA" << std::endl;

  // POINTS go first

  unsigned int numberOfPoints = fixedPointContainer->Size();
  outputFile << "POINTS " << numberOfPoints << " float" << std::endl;

  for (unsigned int i = 0; i < numberOfPoints; i++)
    {
      fixedPointInMillimetreCoordinates = fixedPointContainer->GetElement(i);
      for (unsigned int j = 0; j < TImageType::ImageDimension; j++)
        {
          outputFile << fixedPointInMillimetreCoordinates[j];
          
          if (j == TImageType::ImageDimension-1)
            { 
              outputFile << std::endl;  
            }
          else
            {
              outputFile << " ";  
            }
        }      
    }
  
  // VECTORS go next.
  outputFile << "POINT_DATA " << numberOfPoints << std::endl;
  outputFile << "VECTORS vectors float" << std::endl;
  
  for (unsigned int i = 0; i < numberOfPoints; i++)
    {
      fixedPointInMillimetreCoordinates = fixedPointContainer->GetElement(i);
      movingPointInMillimetreCoordinates = movingPointContainer->GetElement(i);
      
      for (unsigned int j = 0; j < TImageType::ImageDimension; j++)
        {
          vector[j] = movingPointInMillimetreCoordinates[j] - fixedPointInMillimetreCoordinates[j];    
        }
      
      for (unsigned int j = 0; j < TImageType::ImageDimension; j++)
        {
          outputFile << vector[j];
          
          if (j == TImageType::ImageDimension-1)
            { 
              outputFile << std::endl;  
            }
          else
            {
              outputFile << " ";  
            }
        }      
    }
  niftkitkDebugMacro(<<"WritePointSet():Written to:" << fileName);
}

template < typename TImageType, class TScalarType >
void
BlockMatchingMethod<TImageType, TScalarType>
::GetPointCorrespondencies2D(
    ImageSizeType& size,
    ImageSizeType& bigN,
    ImageSizeType& bigOmega,
    ImageSizeType& bigDeltaOne,
    ImageSizeType& bigDeltaTwo,
    PointsContainerPointer& fixedPointContainer,
    PointsContainerPointer& movingPointContainer
    )
{
  unsigned int i, j, k, l;

  ImageRegionType fixedRegion;
  ImageRegionType movingRegion;
  
  ImageIndexType  minFixed; minFixed.Fill(0);
  ImageIndexType  maxFixed; maxFixed.Fill(0);
  ImageIndexType  minMoving; minMoving.Fill(0);
  ImageIndexType  maxMoving; maxMoving.Fill(0);
  ImageIndexType  fixedIndex; fixedIndex.Fill(0);
  ImageIndexType  movingIndex; movingIndex.Fill(0);
  ImageIndexType  bestMovingIndex; bestMovingIndex.Fill(0);
  
  ContinuousIndex< TScalarType, TImageType::ImageDimension > fixedPointInVoxelCoordinates;
  ContinuousIndex< TScalarType, TImageType::ImageDimension > movingPointInVoxelCoordinates;
  PointType       fixedPointInMillimetreCoordinates;
  PointType       movingPointInMillimetreCoordinates;
  PointType       movingPointInMillimetresInOriginalMovingImage;
      
  double similarityMeasure;
  double bestSimilarityMeasure;
  
  ParametersType dummyParametersContainingRegionSize(2*TImageType::ImageDimension + 1);
  dummyParametersContainingRegionSize.SetElement(0, TImageType::ImageDimension);
  for (i = 0; i < TImageType::ImageDimension; i++)
    {
      dummyParametersContainingRegionSize.SetElement(i+1, bigN[i]);  
    }
  
  bool shouldBeMaximized = (static_cast<SimilarityMeasurePointer>(this->GetMetric()))->ShouldBeMaximized();
  
  for (i = 0; i < TImageType::ImageDimension; i++)
    {
      minFixed[i] = bigOmega[i];
      maxFixed[i] = size[i] - bigN[i] - bigOmega[i] - 1;
      minMoving[i] = minFixed[i] - bigOmega[i];
      maxMoving[i] = maxFixed[i] + bigOmega[i];
    }

  niftkitkDebugMacro(<<"GetPointCorrespondencies2D():minFixed=" << minFixed \
    << ", maxFixed=" << maxFixed \
    << ", minMoving=" << minMoving \
    << ", maxMoving=" << maxMoving \
    << ", bigN=" << bigN \
    << ", bigOmega=" << bigOmega \
    << ", bigDeltaOne=" << bigDeltaOne \
    << ", bigDeltaTwo=" << bigDeltaTwo);

  for (i = 0; i < TImageType::ImageDimension; i++)
    {
      if (maxFixed[i] <= minFixed[i] || maxMoving[i] <= minMoving[i])
        {
          itkExceptionMacro(<< "The maximum block bounds are less than or equal to the minimum, this is wrong. You probably have too small images for your block size.");
        }
    }

  fixedRegion.SetSize(bigN);
  movingRegion.SetSize(bigN);

  // Two step. 
  // 1.) Find points with significant variance, sort into a list, and keep the most meaningful points.
  // 2.) Do block matching only for each of those points.
  
  // Step 1, use a priority_queue (heap) to store index and variance.
  VarianceHeap heap;
  double variance;
  
  if (m_UseGradientMagnitudeVariance)
    {
      niftkitkDebugMacro(<<"GetPointCorrespondencies2D():Using fixed image gradient magnitude for variance");
    }
  else
    {
      niftkitkDebugMacro(<<"GetPointCorrespondencies2D():Using fixed image intensity for variance");
    }

  // Before doing this, we need to make sure we are using the FixedImage, (which may be
  // masked with a dilated, thresholded mask, or even smoothed in some way),
  // rather than the FixedImageCopy which is a copy of the original.
  m_FixedImageRegionFilter->SetInput(this->GetFixedImage());

  for (i = minFixed[0]; i < (unsigned int)maxFixed[0]; i += bigDeltaOne[0])
    {
      for(j = minFixed[1]; j < (unsigned int)maxFixed[1]; j += bigDeltaOne[1])
        {
          fixedIndex[0] = i;
          fixedIndex[1] = j;
          fixedRegion.SetIndex(fixedIndex);        
          m_FixedImageRegionFilter->SetRegionOfInterest(fixedRegion);
          m_FixedImageRegionFilter->Update();

          if (m_UseGradientMagnitudeVariance)
            {
              m_GradientMagnitudeImageFilter->Update();
              m_GradientMagnitudeCovarianceCalculator->Update();
              variance = (*(m_GradientMagnitudeCovarianceCalculator->GetOutput()))(0,0);
            }
          else
            {
              m_FixedImageCovarianceCalculator->Update();
              variance = (*(m_FixedImageCovarianceCalculator->GetOutput()))(0,0);
            }
          if (variance > 0)
            {
              heap.push(VarianceHeapDataType(variance, fixedIndex));    
            }
        }
    }
  
  // Step 2, make sure list is most variance -> least variance, and go through list
  // until we hit the threshold determined by m_PercentageOfPointsToKeep
  // Also, we need to make sure we are working with the original fixed image, not a masked one.
  m_FixedImageRegionFilter->SetInput(this->GetFixedImageCopy());
  
  unsigned long int totalNumberOfFixedImagePoints = heap.size();
  unsigned long int numberOfFixedImagePointsThatWeWillUse = (unsigned long int)(totalNumberOfFixedImagePoints * (m_PercentageOfPointsToKeep/100.0));
  unsigned long int currentPoint = 0;
  unsigned long int actualPointNumberInContainer = 0;
  
  niftkitkDebugMacro(<<"GetPointCorrespondencies2D():Using " << totalNumberOfFixedImagePoints \
      << " x " << m_PercentageOfPointsToKeep << "% = " << numberOfFixedImagePointsThatWeWillUse << " points");
  
  currentPoint = 0;
  
  while(currentPoint < numberOfFixedImagePointsThatWeWillUse && currentPoint < totalNumberOfFixedImagePoints)
    {
      fixedIndex = (heap.top()).GetIndex();
      
      heap.pop();

      fixedRegion.SetIndex(fixedIndex);

      m_FixedImageRegionFilter->SetRegionOfInterest(fixedRegion);
      m_FixedImageRegionFilter->Update();

      this->GetMetric()->SetFixedImageRegion( m_FixedImageRegionFilter->GetOutput()->GetBufferedRegion() );

      bestMovingIndex = fixedIndex;

      if (shouldBeMaximized)
        {
          bestSimilarityMeasure = std::numeric_limits<double>::min();  
        }
      else
        {
          bestSimilarityMeasure = std::numeric_limits<double>::max();  
        }
                                 
      for (k = fixedIndex[0] - bigOmega[0]; k < fixedIndex[0] + bigOmega[0]; k += bigDeltaTwo[0])
        {
          for (l = fixedIndex[1] - bigOmega[1]; l < fixedIndex[1] + bigOmega[1]; l += bigDeltaTwo[1])
            {
            
              movingIndex[0] = k;
              movingIndex[1] = l;

              dummyParametersContainingRegionSize.SetElement(3, k);
              dummyParametersContainingRegionSize.SetElement(4, l);

              similarityMeasure = this->GetMetric()->GetValue(dummyParametersContainingRegionSize);

              if ((shouldBeMaximized && similarityMeasure > bestSimilarityMeasure)
                || (!shouldBeMaximized && similarityMeasure < bestSimilarityMeasure)
                )
                {
                  bestMovingIndex = movingIndex;
                  bestSimilarityMeasure = similarityMeasure;
                }
            } // end for l
        } // end for k
      
      // Now we add to list.
      for (k = 0; k < TImageType::ImageDimension; k++)
        {
          fixedPointInVoxelCoordinates[k] = fixedIndex[k] + ((bigN[k]-1)/2.0);
          movingPointInVoxelCoordinates[k] = bestMovingIndex[k] + ((bigN[k]-1)/2.0);
        }

      // Check if its zero displacement.
      bool isZeroDisplacement = true;
      for (k = 0; k < TImageType::ImageDimension; k++)
        {
          if (fixedPointInVoxelCoordinates[k] != movingPointInVoxelCoordinates[k])
            {
              isZeroDisplacement = false; 
            }
        }
      
      if (!(isZeroDisplacement && m_NoZero))
        {
          // For fixed point, we simply convert to millimetres.            
          this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(fixedPointInVoxelCoordinates, fixedPointInMillimetreCoordinates);
            
          // transformed moving image has already been resampled by transform
          // So we need to convert the voxel coordinate to the millimetre coordinate in original moving image
          
          this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(movingPointInVoxelCoordinates, movingPointInMillimetreCoordinates);
          movingPointInMillimetresInOriginalMovingImage = this->GetTransform()->TransformPoint( movingPointInMillimetreCoordinates );

          fixedPointContainer->InsertElement(actualPointNumberInContainer, fixedPointInMillimetreCoordinates);
          movingPointContainer->InsertElement(actualPointNumberInContainer, movingPointInMillimetresInOriginalMovingImage);
          actualPointNumberInContainer++;          
        }
      currentPoint++;
    }  

  niftkitkDebugMacro(<<"GetPointCorrespondencies2D():Actually did " << actualPointNumberInContainer << " points");
  
  if (m_WritePointSet) 
    {
      this->WritePointSet(fixedPointContainer, movingPointContainer);
    }
}

template < typename TImageType, class TScalarType  >
void
BlockMatchingMethod<TImageType, TScalarType >
::GetPointCorrespondencies3D(
    ImageSizeType& size,
    ImageSizeType& bigN,
    ImageSizeType& bigOmega,
    ImageSizeType& bigDeltaOne,
    ImageSizeType& bigDeltaTwo,
    PointsContainerPointer& fixedPointContainer,
    PointsContainerPointer& movingPointContainer
    )    
{
  // TODO: Refactor the 2D and 3D version to remove code duplication.
  unsigned int i, j, k, l, m, n;

  ImageRegionType fixedRegion;
  ImageRegionType movingRegion;
  ImageIndexType  minFixed;
  ImageIndexType  maxFixed;
  ImageIndexType  minMoving;
  ImageIndexType  maxMoving;
  ImageIndexType  fixedIndex;
  ImageIndexType  movingIndex;
  ImageIndexType  bestMovingIndex;
  ImageIndexType  intermediateBestMovingIndex;
  ContinuousIndex< TScalarType, TImageType::ImageDimension > fixedPointInVoxelCoordinates;
  ContinuousIndex< TScalarType, TImageType::ImageDimension > movingPointInVoxelCoordinates;
  PointType       fixedPointInMillimetreCoordinates;
  PointType       movingPointInMillimetreCoordinates;
  PointType       movingPointInMillimetresInOriginalMovingImage;
      
  double similarityMeasure;
  double bestSimilarityMeasure;
  bool canBeOptimized;
  
  ParametersType dummyParametersContainingRegionSize(2*TImageType::ImageDimension + 1);
  dummyParametersContainingRegionSize.SetElement(0, TImageType::ImageDimension);
  for (i = 0; i < TImageType::ImageDimension; i++)
    {
      dummyParametersContainingRegionSize.SetElement(i+1, bigN[i]);  
    }
  
  bool shouldBeMaximized = (static_cast<SimilarityMeasurePointer>(this->GetMetric()))->ShouldBeMaximized();
  
  canBeOptimized = true;
  for (i = 0; i < TImageType::ImageDimension; i++)
    {
      minFixed[i] = bigOmega[i];
      maxFixed[i] = size[i] - bigN[i] - bigOmega[i] - 1;
      minMoving[i] = minFixed[i] - bigOmega[i];
      maxMoving[i] = maxFixed[i] + bigOmega[i];
      
      if (bigN[i] != 4) canBeOptimized = false;
      if (bigOmega[i] != 4) canBeOptimized = false;
      if (bigDeltaTwo[i] != 1) canBeOptimized = false;
    }

  niftkitkDebugMacro(<<"GetPointCorrespondencies3D():minFixed=" << minFixed \
    << ", maxFixed=" << maxFixed \
    << ", minMoving=" << minMoving \
    << ", maxMoving=" << maxMoving \
    << ", bigN=" << bigN \
    << ", bigOmega=" << bigOmega \
    << ", bigDeltaOne=" << bigDeltaOne \
    << ", bigDeltaTwo=" << bigDeltaTwo \
    << ", canBeOptimized=" << canBeOptimized
    );

  for (i = 0; i < TImageType::ImageDimension; i++)
    {
      if (maxFixed[i] <= minFixed[i] || maxMoving[i] <= minMoving[i])
        {
          itkExceptionMacro(<< "The maximum block bounds are less than or equal to the minimum, this is wrong. You probably have too small images for your block size.");
        }
    }

  fixedRegion.SetSize(bigN);
  movingRegion.SetSize(bigN);

  // Two step. 
  // 1.) Find points with significant variance, sort into a list, and keep the most meaningful points.
  // 2.) Do block matching only for each of those points.
  
  // Step 1, use a priority_queue (heap) to store index and variance.
  VarianceHeap heap;
  double variance;

  if (m_UseGradientMagnitudeVariance)
    {
      niftkitkDebugMacro(<<"GetPointCorrespondencies3D():Using fixed image gradient magnitude for variance");
    }
  else
    {
      niftkitkDebugMacro(<<"GetPointCorrespondencies3D():Using fixed image intensity for variance");
    }

  // Before doing this, we need to make sure we are using the FixedImage, (which may be
  // masked with a dilated, thresholded mask, or even smoothed in some way),
  // rather than the FixedImageCopy which is a copy of the original.
  m_FixedImageRegionFilter->SetInput(this->GetFixedImage());
  
  for (i = minFixed[0]; i < (unsigned int)maxFixed[0]; i += bigDeltaOne[0])
    {
      for(j = minFixed[1]; j < (unsigned int)maxFixed[1]; j += bigDeltaOne[1])
        {
          for(k = minFixed[2]; k < (unsigned int)maxFixed[2]; k += bigDeltaOne[2])
            {
              fixedIndex[0] = i;
              fixedIndex[1] = j;
              fixedIndex[2] = k;
              fixedRegion.SetIndex(fixedIndex);        
              m_FixedImageRegionFilter->SetRegionOfInterest(fixedRegion);
              m_FixedImageRegionFilter->Update();
              
              if (m_UseGradientMagnitudeVariance)
                {
                  m_GradientMagnitudeImageFilter->Update();
                  m_GradientMagnitudeCovarianceCalculator->Update();
                  variance = (*(m_GradientMagnitudeCovarianceCalculator->GetOutput()))(0,0);
                }
              else
                {
                  m_FixedImageCovarianceCalculator->Update();
                  variance = (*(m_FixedImageCovarianceCalculator->GetOutput()))(0,0);
                }
              if (variance > 0)
                {
                  heap.push(VarianceHeapDataType(variance, fixedIndex));    
                }
            }          
        }
    }
  
  // Step 2, make sure list is most variance -> least variance, and go through list
  // until we hit the threshold determined by m_PercentageOfPointsToKeep
  // Also, we need to make sure we are working with the original fixed image, not a masked one.
  m_FixedImageRegionFilter->SetInput(this->GetFixedImageCopy());
  
  unsigned long int totalNumberOfFixedImagePoints = heap.size();
  unsigned long int numberOfFixedImagePointsThatWeWillUse = (unsigned long int)(totalNumberOfFixedImagePoints * (m_PercentageOfPointsToKeep/100.0));
  unsigned long int currentPoint = 0;
  unsigned long int actualPointNumberInContainer = 0;
  
  niftkitkDebugMacro(<<"GetPointCorrespondencies3D():Using " << totalNumberOfFixedImagePoints \
      << " x " << m_PercentageOfPointsToKeep << "% = " << numberOfFixedImagePointsThatWeWillUse << " points");
  
  currentPoint = 0;

  while(currentPoint < numberOfFixedImagePointsThatWeWillUse && currentPoint < totalNumberOfFixedImagePoints)
    {
      fixedIndex = (heap.top()).GetIndex();    
      
      heap.pop();

      fixedRegion.SetIndex(fixedIndex);        

      m_FixedImageRegionFilter->SetRegionOfInterest(fixedRegion);
      m_FixedImageRegionFilter->Update();

      this->GetMetric()->SetFixedImageRegion( m_FixedImageRegionFilter->GetOutput()->GetBufferedRegion() );

      bestMovingIndex = fixedIndex;
      
      if (shouldBeMaximized)
        {
          bestSimilarityMeasure = std::numeric_limits<double>::min();  
        }
      else
        {
          bestSimilarityMeasure = std::numeric_limits<double>::max();  
        }

      // Experimental section, trying to avoid evaluating each position exhaustively.
      // Alternatively we could do a mini-registration here.
      if (false)
        {
          // Take steps of 4
          for (l = fixedIndex[0] - 4; l <= (unsigned int)(fixedIndex[0] + 4); l += 4)
            {
              for (m = fixedIndex[1] - 4; m <= (unsigned int)(fixedIndex[1] + 4); m += 4)
                {
                  for (n = fixedIndex[2] - 4; n <= (unsigned int)(fixedIndex[2] + 4); n += 4)
                    {
            
                      movingIndex[0] = l;
                      movingIndex[1] = m;
                      movingIndex[2] = n;

                      dummyParametersContainingRegionSize.SetElement(4, l);
                      dummyParametersContainingRegionSize.SetElement(5, m);
                      dummyParametersContainingRegionSize.SetElement(6, n);
                      
                      similarityMeasure = this->GetMetric()->GetValue(dummyParametersContainingRegionSize);

                      if ((shouldBeMaximized && similarityMeasure > bestSimilarityMeasure)
                      || (!shouldBeMaximized && similarityMeasure < bestSimilarityMeasure)
                      )
                        {
                          intermediateBestMovingIndex = movingIndex;
                          bestSimilarityMeasure = similarityMeasure;                          
                        }
                    } // end for n
                } // end for m
            } // end for l          

          bestMovingIndex = intermediateBestMovingIndex;
          
          // Take steps of 2, starting at best index
          for (l = bestMovingIndex[0] - 2; l <= (unsigned int)(bestMovingIndex[0] + 2); l += 2)
            {
              for (m = bestMovingIndex[1] - 2; m <= (unsigned int)(bestMovingIndex[1] + 2); m += 2)
                {
                  for (n = bestMovingIndex[2] - 2; n <= (unsigned int)(bestMovingIndex[2] + 2); n += 2)
                    {
            
                      movingIndex[0] = l;
                      movingIndex[1] = m;
                      movingIndex[2] = n;

                      dummyParametersContainingRegionSize.SetElement(4, l);
                      dummyParametersContainingRegionSize.SetElement(5, m);
                      dummyParametersContainingRegionSize.SetElement(6, n);
                      
                      similarityMeasure = this->GetMetric()->GetValue(dummyParametersContainingRegionSize);

                      if ((shouldBeMaximized && similarityMeasure > bestSimilarityMeasure)
                      || (!shouldBeMaximized && similarityMeasure < bestSimilarityMeasure)
                      )
                        {
                          intermediateBestMovingIndex = movingIndex;
                          bestSimilarityMeasure = similarityMeasure;                          
                        }
                    } // end for n
                } // end for m
            } // end for l          

          bestMovingIndex = intermediateBestMovingIndex;
          
          // Take steps of 1, starting at best index
          for (l = bestMovingIndex[0] - 1; l <= (unsigned int)(bestMovingIndex[0] + 1); l += 1)
            {
              for (m = bestMovingIndex[1] - 1; m <= (unsigned int)(bestMovingIndex[1] + 1); m += 1)
                {
                  for (n = bestMovingIndex[2] - 1; n <= (unsigned int)(bestMovingIndex[2] + 1); n += 1)
                    {
            
                      movingIndex[0] = l;
                      movingIndex[1] = m;
                      movingIndex[2] = n;

                      dummyParametersContainingRegionSize.SetElement(4, l);
                      dummyParametersContainingRegionSize.SetElement(5, m);
                      dummyParametersContainingRegionSize.SetElement(6, n);
                      
                      similarityMeasure = this->GetMetric()->GetValue(dummyParametersContainingRegionSize);

                      if ((shouldBeMaximized && similarityMeasure > bestSimilarityMeasure)
                      || (!shouldBeMaximized && similarityMeasure < bestSimilarityMeasure)
                      )
                        {
                          intermediateBestMovingIndex = movingIndex;
                          bestSimilarityMeasure = similarityMeasure;
                        }
                    } // end for n
                } // end for m
            } // end for l          
          
          bestMovingIndex = intermediateBestMovingIndex;
        }
      else
        {
          for (l = fixedIndex[0] - bigOmega[0]; l < fixedIndex[0] + bigOmega[0]; l += bigDeltaTwo[0])
            {
              for (m = fixedIndex[1] - bigOmega[1]; m < fixedIndex[1] + bigOmega[1]; m += bigDeltaTwo[1])
                {
                  for (n = fixedIndex[2] - bigOmega[2]; n < fixedIndex[2] + bigOmega[2]; n += bigDeltaTwo[2])
                    {
            
                      movingIndex[0] = l;
                      movingIndex[1] = m;
                      movingIndex[2] = n;

                      dummyParametersContainingRegionSize.SetElement(4, l);
                      dummyParametersContainingRegionSize.SetElement(5, m);
                      dummyParametersContainingRegionSize.SetElement(6, n);
                      
                      similarityMeasure = this->GetMetric()->GetValue(dummyParametersContainingRegionSize);

                      if ((shouldBeMaximized && similarityMeasure > bestSimilarityMeasure)
                      || (!shouldBeMaximized && similarityMeasure < bestSimilarityMeasure)
                      )
                        {
                          bestMovingIndex = movingIndex;
                          bestSimilarityMeasure = similarityMeasure;
                        }
                    } // end for n
                } // end for m
            } // end for l          
        }

      // Now we add to list.
      for (l = 0; l < TImageType::ImageDimension; l++)
        {
          fixedPointInVoxelCoordinates[l] = fixedIndex[l] + ((bigN[l]-1)/2.0);
          movingPointInVoxelCoordinates[l] = bestMovingIndex[l] + ((bigN[l]-1)/2.0);
        }

      // Check if its zero displacement.
      bool isZeroDisplacement = true;
      for (k = 0; k < TImageType::ImageDimension; k++)
        {
          if (fixedPointInVoxelCoordinates[k] != movingPointInVoxelCoordinates[k])
            {
              isZeroDisplacement = false; 
            }
        }
      
      if (!(isZeroDisplacement && m_NoZero))
        {
          // For fixed point, we simply convert to millimetres.            
          this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(fixedPointInVoxelCoordinates, fixedPointInMillimetreCoordinates);
        
          // transformed moving image has already been resampled by transform
          // So we need to convert the voxel coordinate to the millimetre coordinate in original moving image
      
          this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(movingPointInVoxelCoordinates, movingPointInMillimetreCoordinates);
          movingPointInMillimetresInOriginalMovingImage = this->GetTransform()->TransformPoint( movingPointInMillimetreCoordinates );

          fixedPointContainer->InsertElement(actualPointNumberInContainer, fixedPointInMillimetreCoordinates);
          movingPointContainer->InsertElement(actualPointNumberInContainer, movingPointInMillimetresInOriginalMovingImage);
          actualPointNumberInContainer++;          
        }
      currentPoint++;
  
    } // end while

  niftkitkDebugMacro(<<"GetPointCorrespondencies2D():Actually did " << actualPointNumberInContainer << " points");

  if (m_WritePointSet) 
    {
      this->WritePointSet(fixedPointContainer, movingPointContainer);
    }

}

template < typename TImageType, class TScalarType  >
void
BlockMatchingMethod<TImageType, TScalarType >
::TrimPoints(const TransformType* transform, 
    const PointsContainerType* fixedPoints,
    const PointsContainerType* movingPoints,
    PointsContainerType* trimmedFixedPoints,
    PointsContainerType* trimmedMovingPoints)
{
  trimmedFixedPoints->Initialize();
  trimmedMovingPoints->Initialize();
  
  ResidualHeap heap;

  PointIterator fixedPointItr = fixedPoints->Begin();
  PointIterator fixedPointEnd = fixedPoints->End();
  PointIterator movingPointItr = movingPoints->Begin();
  PointIterator movingPointEnd = movingPoints->End();

  PointType fixedPoint;
  PointType movingPoint;
  PointType transformedFixedPoint;

  double residual;
  unsigned long int i = 0;

  while( fixedPointItr != fixedPointEnd && movingPointItr != movingPointEnd)
    {
      
      fixedPoint = fixedPointItr.Value();
      movingPoint = movingPointItr.Value();
      transformedFixedPoint = this->GetTransform()->TransformPoint(fixedPoint);

      // Compute the squared distance
      residual = 0;
      for (i = 0; i < PointSetType::PointDimension; i++)
        {
          residual += ((movingPoint[i] - transformedFixedPoint[i])*(movingPoint[i] - transformedFixedPoint[i])); 
        }

      // Stick it in the queue.
      heap.push(ResidualHeapDataType(residual, fixedPoint, movingPoint));
      
      ++fixedPointItr;
      ++movingPointItr;
    }
  
  unsigned long int totalSize    = heap.size();
  unsigned long int numberToKeep = (unsigned long int)(totalSize * (m_PercentageOfPointsInLeastTrimmedSquares/100.0));
  
  while (i < totalSize && i < numberToKeep)
    {
      trimmedFixedPoints->InsertElement(i, heap.top().GetFixed());
      trimmedMovingPoints->InsertElement(i, heap.top().GetMoving());
      i++;
      //std::cerr << "Inserted i=" << i << "," << heap.top().GetFixed() << "," << heap.top().GetMoving() << ", residual=" << heap.top().GetResidual() << std::endl;
      heap.pop();
    }
  
  niftkitkDebugMacro(<<"TrimPoints():Started with[" << fixedPoints->Size() \
      << "," << movingPoints->Size() \
      << "], kept " << m_PercentageOfPointsInLeastTrimmedSquares \
      << "% which is " << numberToKeep \
      << ", resulting in [" << trimmedFixedPoints->Size() \
      << "," << trimmedMovingPoints->Size() \
      << "]");
}

/*
 * The bit that does the registration.
 */
template < typename TImageType, class TScalarType  >
void
BlockMatchingMethod<TImageType, TScalarType >
::DoRegistration() throw (ExceptionObject)
{
  niftkitkDebugMacro(<<"DoRegistration():Started");

  ParametersType localParameters = this->GetInitialTransformParameters();
  ParametersType previousParameters = this->GetInitialTransformParameters();
  
  unsigned long int currentIteration = 0;
  unsigned int i;
  
  double beforeValue;
  double afterValue;
  
  // In general, this will be called with few parameters.... so.. lets print them.
  niftkitkInfoMacro(<<"DoRegistration():Initial transform parameters:" << localParameters);

  // Initialize point sets.
  PointsContainerPointer fixedPointContainer  = PointsContainerType::New();
  PointsContainerPointer trimmedFixedPointContainer = PointsContainerType::New();
  PointsContainerPointer movingPointContainer = PointsContainerType::New();
  PointsContainerPointer trimmedMovingPointContainer = PointsContainerType::New();
  PointSetPointer        fixedPointSet        = PointSetType::New();
  PointSetPointer        movingPointSet       = PointSetType::New();
  
  unsigned int     dimensions = TImageType::ImageDimension;
  ImageSizeType    size = this->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  ImageSpacingType spacing = this->GetFixedImage()->GetSpacing();

  niftkitkDebugMacro(<<"DoRegistration():Dimensions=" << dimensions \
    << ", size=" << size \
    << ", spacing=" << spacing \
    << ", N=" << m_BlockSize \
    << ", Omega=" << m_BlockHalfWidth \
    << ", DeltaOne=" << m_BlockSpacing \
    << ", DeltaTwo=" << m_BlockSubSampling);

  // Check we are initialized, and good to go.
  if (m_BlockSize == -1 || m_BlockHalfWidth == -1 || m_BlockSpacing == -1 || m_BlockSubSampling == -1)
    {
      itkExceptionMacro(<<"Please initialise N, Omega, DeltaOne and DeltaTwo" );
    }
  
  // Multi-scale bit here: The scale is the size of the block, not the size of the image.
  bool stillImprovingSoKeepGoing = true;
  while(m_BlockSize >= m_MinimumBlockSize && currentIteration < m_MaximumNumberOfIterationsRoundMainLoop && stillImprovingSoKeepGoing)
    {
      // Adjust parameter values, to take into account anisotropic voxels.
      ImageSizeType bigN;
      ImageSizeType bigOmega;
      ImageSizeType bigDeltaOne;
      ImageSizeType bigDeltaTwo;

      double minimumSize = spacing[0];
      
      for (i = 1; i < dimensions; i++)
        {
          if (spacing[1] < minimumSize)
            {
              minimumSize = spacing[i];
            }
        }

      for (i = 0; i < dimensions; i++)
        {
          if (m_ScaleByMillimetres)
            {
              double sizeOfThisAxis = spacing[i];
              double scaleFactor = sizeOfThisAxis / minimumSize;
              
              niftkitkDebugMacro(<<"DoRegistration():axis[" << i << "], sizeOfThisAxis=" << sizeOfThisAxis << ", scaleFactor=" << scaleFactor);
              bigN[i] = (long unsigned int)(m_BlockSize / scaleFactor);
              bigOmega[i] = (long unsigned int)(m_BlockHalfWidth / scaleFactor);
              bigDeltaOne[i] = (long unsigned int)(m_BlockSpacing / scaleFactor);  
              bigDeltaTwo[i] = (long unsigned int)(m_BlockSubSampling / scaleFactor);                  
            }
          else
            {
              bigN[i] = (long unsigned int)(m_BlockSize);
              bigOmega[i] = (long unsigned int)(m_BlockHalfWidth);
              bigDeltaOne[i] = (long unsigned int)(m_BlockSpacing);  
              bigDeltaTwo[i] = (long unsigned int)(m_BlockSubSampling);
            }
        }
      
      niftkitkDebugMacro(<<"DoRegistration():bigN=" << bigN \
        << ", bigOmega=" << bigOmega \
        << ", bigDeltaOne=" << bigDeltaOne \
        << ", bigDeltaTwo=" << bigDeltaTwo);

      for (i = 0; i < dimensions; i++)
        {
          if (bigN[i] < 1) bigN[i] = 1;
          if (bigOmega[i] < 1) bigOmega[i] = 1;
          if (bigDeltaOne[i] < 1 ) bigDeltaOne[i] = 1;
          if (bigDeltaTwo[i] < 1 ) bigDeltaTwo[i] = 1;
        }

      niftkitkDebugMacro(<<"DoRegistration():bigN=" << bigN \
        << ", bigOmega=" << bigOmega \
        << ", bigDeltaOne=" << bigDeltaOne \
        << ", bigDeltaTwo=" << bigDeltaTwo \
        );
      
      niftkitkDebugMacro(<<"DoRegistration():Resampling for parameters:" << localParameters);
             
      this->GetTransform()->SetParameters(localParameters);
      this->m_MovingImageResampler->SetTransform(this->GetTransform());
      this->m_MovingImageResampler->Modified();
      this->m_MovingImageResampler->UpdateLargestPossibleRegion();
      this->m_MovingImageResampler->Update();
      
      this->m_MinMaxCalculator->SetImage(this->m_MovingImageResampler->GetOutput());
      this->m_MinMaxCalculator->Compute();
      if (this->m_MinMaxCalculator->GetMinimum() == this->m_MinMaxCalculator->GetMaximum())
        {
          itkExceptionMacro(<<"The transformed moving image has the same minimum and maximum, which suggests it is empty!" );
        }

      if (m_WriteTransformedMovingImage)
        {
          std::string filename = m_TransformedMovingImageFileName + "." + niftk::ConvertToString((int)currentIteration) + "." + m_TransformedMovingImageFileExt;
          niftkitkDebugMacro(<<"DoRegistration():Writing to " << filename);
          typedef ImageFileWriter<TImageType> WriterType;
          typename WriterType::Pointer writer = WriterType::New();
          writer->SetInput(this->m_MovingImageResampler->GetOutput());
          writer->SetFileName(filename);
          writer->Update(); 
        }

      niftkitkDebugMacro(<<"DoRegistration():Fetching points for iteration:" << currentIteration);
      
      fixedPointContainer->Initialize();
      movingPointContainer->Initialize();
      
      if (dimensions == 2)
        {
          GetPointCorrespondencies2D(
            size,
            bigN,
            bigOmega,
            bigDeltaOne,
            bigDeltaTwo,
            fixedPointContainer,
            movingPointContainer
            );
        }
      else if (dimensions == 3)
        {
          GetPointCorrespondencies3D(
            size,
            bigN,
            bigOmega,
            bigDeltaOne,
            bigDeltaTwo,
            fixedPointContainer,
            movingPointContainer
            );        
        }
      else 
        {
          itkExceptionMacro(<<"This class is only suitable for 2D or 3D images.");
        }

      if (fixedPointContainer->Size() < 7 || movingPointContainer->Size() < 7)
        {
          niftkitkDebugMacro(<<"DoRegistration():Less than seven points, so finishing");
          stillImprovingSoKeepGoing = false;
        }
      else
        {
          fixedPointSet->SetPoints(fixedPointContainer);
          movingPointSet->SetPoints(movingPointContainer);                  
          this->m_PointSetMetric->SetTransform(this->GetTransform());
          this->m_PointSetMetric->SetFixedPointSet(fixedPointSet);
          this->m_PointSetMetric->SetMovingPointSet(movingPointSet);
          this->GetOptimizer()->SetCostFunction(m_PointSetMetric);
          this->GetOptimizer()->SetInitialPosition(localParameters);

          niftkitkDebugMacro(<<"DoRegistration():Iteration:" << currentIteration << ", solving points, fixedPoints=" << fixedPointSet->GetNumberOfPoints() << ", movingPoints=" << movingPointSet->GetNumberOfPoints());

          previousParameters = localParameters;
          beforeValue = this->GetOptimizer()->GetValue(previousParameters);
          niftkitkDebugMacro(<<"DoRegistration():Iteration:" << currentIteration << ", before optimising points, beforeValue=" << beforeValue << ", at position:" << previousParameters);

          // This bit is the Least Trimmed Squares part.

          for (i = 0; i < 3; i++)
            {
              if (i == 0)
                {
                  fixedPointSet->SetPoints(fixedPointContainer);
                  movingPointSet->SetPoints(movingPointContainer);                  
                }
              else
                {
                  // This is where we trim the point set:
                  this->TrimPoints(this->GetTransform(),
                      fixedPointContainer,          // input
                      movingPointContainer,         // input
                      trimmedFixedPointContainer,   // output
                      trimmedMovingPointContainer); // output
                  
                  // Set them onto the metric.
                  fixedPointSet->SetPoints(trimmedFixedPointContainer);
                  movingPointSet->SetPoints(trimmedMovingPointContainer);
                }

              // Assume that the optimizer that was plugged into this registration
              // method is suitable.  The paper uses Powell, but any non-gradient 
              // based one should do the trick.
              
              this->GetOptimizer()->StartOptimization();
              
              localParameters = this->GetOptimizer()->GetCurrentPosition();
              afterValue = this->GetOptimizer()->GetValue(localParameters);
              
              niftkitkDebugMacro(<<"DoRegistration():Iteration:" << currentIteration << ", similarity=" << niftk::ConvertToString(afterValue) << ",\t position:" << localParameters);

            } // end for 3 loops

          niftkitkInfoMacro(<<"DoRegistration():Iteration:" << currentIteration << ", similarity=" << niftk::ConvertToString(afterValue) << ",\t position:" << localParameters);

          stillImprovingSoKeepGoing = this->CheckEpsilon(previousParameters, localParameters);
          currentIteration++;
          
          niftkitkDebugMacro(<<"DoRegistration():Before while conditions, currentIteration=" << currentIteration \
              << ", m_MaximumNumberOfIterationsRoundMainLoop=" << m_MaximumNumberOfIterationsRoundMainLoop \
              << ", m_BlockSize=" << m_BlockSize \
              << ", m_MinimumBlockSize=" << m_MinimumBlockSize \
              << ", stillImprovingSoKeepGoing=" << stillImprovingSoKeepGoing);          
        }
      
    } // end while m_BlockSize > m_MinimumBlockSize
  
  this->SetLastTransformParameters(localParameters);
  this->GetTransform()->SetParameters(localParameters);
  
  niftkitkDebugMacro(<<"DoRegistration():Finished with parameters:" << localParameters);
}

} // end namespace itk


#endif
