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
#ifndef __itkHighResLaplacianSolverImageFilter_txx
#define __itkHighResLaplacianSolverImageFilter_txx

#include "itkHighResLaplacianSolverImageFilter.h"
#include "ConversionUtils.h"
#include <cmath>

#include "itkLogHelper.h"

namespace itk
{

template <typename TInputImage, typename TScalarType > 
HighResLaplacianSolverImageFilter<TInputImage, TScalarType>
::HighResLaplacianSolverImageFilter()
{
  m_Tolerance = 0.001;
  m_VoxelMultiplicationFactor = 2;
  niftkitkDebugMacro(<<"HighResLaplacianSolverImageFilter():Constructed" \
      << ", m_VoxelMultiplicationFactor=" << m_VoxelMultiplicationFactor \
      );
}

template <typename TInputImage, typename TScalarType > 
HighResLaplacianSolverImageFilter<TInputImage, TScalarType>
::~HighResLaplacianSolverImageFilter()
{
  unsigned long int counter = 0;
  IteratorType iterator;
  FiniteDifferenceVoxelType *tmp;
  
  for (iterator = m_MapOfVoxels.begin(); iterator != m_MapOfVoxels.end(); iterator++)
    {
      tmp = (*iterator).second;
      if (tmp != NULL)
        {
          delete tmp;
          counter++;
        }
    }
  niftkitkDebugMacro(<<"HighResLaplacianSolverImageFilter():Destroyed " << counter << " voxels");
}

template <typename TInputImage, typename TScalarType >
void 
HighResLaplacianSolverImageFilter<TInputImage, TScalarType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "VoxelMultiplicationFactor:" << m_VoxelMultiplicationFactor << std::endl;
}

template <typename TInputImage, typename TScalarType >
bool
HighResLaplacianSolverImageFilter<TInputImage, TScalarType>
::GetValue(
    InputImageType* highResImage,
    InputImageType* lowResImage,
    NearestNeighbourInterpolatorType* interpolator, 
    InputImageIndexType& index,
    InputImagePixelType& result)
{
  PointType point;
  ContinuousIndexType continuousIndex;

  highResImage->TransformIndexToPhysicalPoint( index, point );
  lowResImage->TransformPhysicalPointToContinuousIndex(point, continuousIndex);
  
  if (interpolator->IsInsideBuffer(continuousIndex))
    {
      result = interpolator->EvaluateAtContinuousIndex(continuousIndex);
      return true;
    }
  else
    {
      return false;
    }
}

template <typename TInputImage, typename TScalarType >
void 
HighResLaplacianSolverImageFilter<TInputImage, TScalarType>
::InsertNeighbour(
    FiniteDifferenceVoxelType* greyVoxel,
    InputImagePixelType& neighbourValue,  
    unsigned long int& mapIndexOfVoxel,
    InputImageIndexType& itkImageIndexOfVoxel,
    MapType& map,
    unsigned long int& numberOfDuplicates,
    unsigned long int& numberOfBoundaryPoints
    )
{

  if (fabs(neighbourValue - this->GetWhiteMatterLabel()) < m_Tolerance
      || fabs(neighbourValue - this->GetExtraCerebralMatterLabel()) < m_Tolerance)
    {

      FiniteDifferenceVoxelType* neighbourVoxel = new FiniteDifferenceVoxelType();
      neighbourVoxel->SetVoxelArrayIndex(mapIndexOfVoxel);
      
      if (fabs(neighbourValue - this->GetWhiteMatterLabel()) < m_Tolerance)
        {
          neighbourVoxel->SetValue(0, (InputImagePixelType)this->GetLowVoltage());  
          greyVoxel->SetIsNextToWM(true);
        }
      else
        {
          neighbourVoxel->SetValue(0, (InputImagePixelType)this->GetHighVoltage());
          greyVoxel->SetIsNextToCSF(true);
        }
      
      neighbourVoxel->SetVoxelIndex(itkImageIndexOfVoxel);
      neighbourVoxel->SetBoundary(true);
      
      if (map.find(mapIndexOfVoxel) != map.end())
        {
          numberOfDuplicates++;
          
          // We have got one for this position, so we call operator=
          *(map[mapIndexOfVoxel]) = *neighbourVoxel;
          delete neighbourVoxel;
        }
      else
        {
          // We havent got one for this position, so insert this new one.
          map.insert(PairType(mapIndexOfVoxel, neighbourVoxel));    
        }
      
      numberOfBoundaryPoints++;
    }
}

template <typename TInputImage, typename TScalarType > 
void
HighResLaplacianSolverImageFilter<TInputImage, TScalarType>
::GenerateData()
{
  this->AllocateOutputs();

  OutputPixelType meanVoltage = (this->GetHighVoltage() + this->GetLowVoltage())/2.0;
  
  niftkitkDebugMacro(<<"GenerateData():Started, grey Label=" << this->GetGreyMatterLabel()
      << ", white label=" << this->GetWhiteMatterLabel()
      << ", extra cerebral label=" << this->GetExtraCerebralMatterLabel() \
      << ", low voltage=" << this->GetLowVoltage() \
      << ", high voltage=" << this->GetHighVoltage() \
      << ", mean voltage=" << meanVoltage \
      );

  typename InputImageType::Pointer inputImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  
  // Set up the new spacing dimension, and a virtual size. 
  // Note we never actually create the image's memory
  InputImageSizeType inputSize = inputImage->GetLargestPossibleRegion().GetSize();
  InputImageSpacingType inputSpacing = inputImage->GetSpacing();
  InputImageOriginType inputOrigin = inputImage->GetOrigin();

  niftkitkDebugMacro(<<"GenerateData():Input image size=" << inputSize \
      << ", spacing=" << inputSpacing \
      << ", origin=" << inputOrigin \
      );

  InputImageRegionType virtualRegion;
  InputImageIndexType virtualIndex;
  InputImageSizeType virtualSize;
  InputImageSpacingType virtualSpacing;
  InputImageOriginType virtualOrigin;
  
  for (unsigned int i =0; i < this->Dimension; i++)
    {
      virtualSpacing[i] = inputSpacing[i]/(float)m_VoxelMultiplicationFactor;
      virtualSize[i] = (int)(inputSize[i]*inputSpacing[i]/virtualSpacing[i]);
      virtualOrigin[i] = inputOrigin[i];//+(((float)(inputSize[i]-1.0)/2.0)*inputSpacing[i])-(((float)(virtualSize[i]-1.0)/2.0)*virtualSpacing[i]);
    }
  virtualIndex.Fill(0);
  virtualRegion.SetSize(virtualSize);
  virtualRegion.SetIndex(virtualIndex);
  
  InputImagePointer virtualImage = InputImageType::New();
  virtualImage->SetRegions(virtualRegion);
  virtualImage->SetSpacing(virtualSpacing);
  virtualImage->SetOrigin(virtualOrigin);
  virtualImage->SetDirection(inputImage->GetDirection());
  
  niftkitkDebugMacro(<<"GenerateData():Virtual image size=" << virtualImage->GetLargestPossibleRegion().GetSize() \
      << ", spacing=" << virtualImage->GetSpacing() \
      << ", origin=" << virtualImage->GetOrigin() \
      );

  m_MapOfVoxels.clear();
  IteratorType iterator;
  
  typename NearestNeighbourInterpolatorType::Pointer interpolator = NearestNeighbourInterpolatorType::New();
  interpolator->SetInputImage(inputImage);
  InputImageIndexType neighbourIndex;
  InputImagePixelType thisPixel;
  InputImagePixelType neighbourPixel;
  FiniteDifferenceVoxelType *fdVox = NULL;
  
  unsigned long int numberOfGrey = 0;
  unsigned long int numberOfBoundary = 0;
  unsigned long int numberOfDuplicates = 0;
  unsigned long int indexOfNeighbour = 0;
  unsigned long int indexOfCurrentVoxel = 0;
    
  /** 
   * Do something more clever about this 2D/3D bit! 
   * This monstrous bit is just to build up a data
   * structure that isnt an image, so we can solve 
   * laplaces equation, at a very high resolution.
   */
  PointType voxelPointInMillimetres;
  
  if (this->Dimension == 2)
    {
      for (unsigned long int y = 0; y < virtualSize[1]; y++)
        {
          for (unsigned long int x = 0; x < virtualSize[0]; x++)
            {
              virtualIndex[0] = x;
              virtualIndex[1] = y;
              indexOfCurrentVoxel = virtualIndex[1]*virtualSize[0] + virtualIndex[0];
              
              if (this->GetValue(virtualImage, inputImage, interpolator, virtualIndex, thisPixel))
                {

                  if (fabs(thisPixel - this->GetGreyMatterLabel()) < m_Tolerance) 
                    {
                      
                      fdVox = new FiniteDifferenceVoxelType();
                      
                      virtualImage->TransformIndexToPhysicalPoint( virtualIndex, voxelPointInMillimetres );
                      fdVox->SetValue(0, meanVoltage);
                      fdVox->SetVoxelIndex(virtualIndex);
                      fdVox->SetVoxelPointInMillimetres(voxelPointInMillimetres);
                      fdVox->SetBoundary(false);
                      fdVox->SetVoxelArrayIndex(indexOfCurrentVoxel);
                      
                      neighbourIndex[0] = x-1;
                      neighbourIndex[1] = y;
                      indexOfNeighbour = neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetMinus(0, indexOfNeighbour);
                      
                      this->GetValue(virtualImage, inputImage, interpolator, neighbourIndex, neighbourPixel);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);
                      
                      neighbourIndex[0] = x+1;
                      neighbourIndex[1] = y;
                      indexOfNeighbour = neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetPlus(0, indexOfNeighbour);

                      this->GetValue(virtualImage, inputImage, interpolator, neighbourIndex, neighbourPixel);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      neighbourIndex[0] = x;
                      neighbourIndex[1] = y-1;
                      indexOfNeighbour = neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetMinus(1, indexOfNeighbour);

                      this->GetValue(virtualImage, inputImage, interpolator, neighbourIndex, neighbourPixel);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      neighbourIndex[0] = x;
                      neighbourIndex[1] = y+1;
                      indexOfNeighbour = neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetPlus(1, indexOfNeighbour);

                      this->GetValue(virtualImage, inputImage, interpolator, neighbourIndex, neighbourPixel);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        
                      
                      m_MapOfVoxels.insert(PairType(indexOfCurrentVoxel, fdVox));
                      numberOfGrey++;
                      
                    } // end if (thisPixel == this->GetGreyMatterLabel()) 
                } // end if (this->GetValue(virtualImage, interpolator, virtualIndex, thisPixel))
              
            } // end for x
          
          niftkitkInfoMacro(<<"GenerateData():Row=" << y << ", total voxels=" << m_MapOfVoxels.size());
          
        } // end for y
    }
  else
    {
      
      InputImageSizeType virtualSizeMinus;
      virtualSizeMinus[0] = virtualSize[0]-m_VoxelMultiplicationFactor-m_VoxelMultiplicationFactor;
      virtualSizeMinus[1] = virtualSize[1]-m_VoxelMultiplicationFactor-m_VoxelMultiplicationFactor;
      virtualSizeMinus[2] = virtualSize[2]-m_VoxelMultiplicationFactor-m_VoxelMultiplicationFactor;
      
      ContinuousIndexType originInLowResImage;
      ContinuousIndexType dxInLowResImage;
      ContinuousIndexType dyInLowResImage;
      ContinuousIndexType dzInLowResImage;
      
      InputImageIndexType originIndex;
      originIndex[0] = m_VoxelMultiplicationFactor;
      originIndex[1] = m_VoxelMultiplicationFactor;
      originIndex[2] = m_VoxelMultiplicationFactor;
      
      InputImageIndexType indexForXAxis;
      indexForXAxis[0] = originIndex[0]+1;
      indexForXAxis[1] = originIndex[1];
      indexForXAxis[2] = originIndex[2];

      InputImageIndexType indexForYAxis;
      indexForYAxis[0] = originIndex[0];
      indexForYAxis[1] = originIndex[1]+1;
      indexForYAxis[2] = originIndex[2];

      InputImageIndexType indexForZAxis;
      indexForZAxis[0] = originIndex[0];
      indexForZAxis[1] = originIndex[1];
      indexForZAxis[2] = originIndex[2]+1;

      PointType tmpPoint;
      
      virtualImage->TransformIndexToPhysicalPoint( originIndex, tmpPoint );
      inputImage->TransformPhysicalPointToContinuousIndex(tmpPoint, originInLowResImage);

      virtualImage->TransformIndexToPhysicalPoint( indexForXAxis, tmpPoint );
      inputImage->TransformPhysicalPointToContinuousIndex(tmpPoint, dxInLowResImage);

      virtualImage->TransformIndexToPhysicalPoint( indexForYAxis, tmpPoint );
      inputImage->TransformPhysicalPointToContinuousIndex(tmpPoint, dyInLowResImage);

      virtualImage->TransformIndexToPhysicalPoint( indexForZAxis, tmpPoint );
      inputImage->TransformPhysicalPointToContinuousIndex(tmpPoint, dzInLowResImage);

      niftkitkInfoMacro(<<"GenerateData():virtualSizeMinus=" << virtualSizeMinus \
          << ", originInLowResImage=" << originInLowResImage \
          );
          
      for (unsigned int i = 0; i < InputImageType::ImageDimension; i++)
        {
          dxInLowResImage[i] -= originInLowResImage[i];
          dyInLowResImage[i] -= originInLowResImage[i];
          dzInLowResImage[i] -= originInLowResImage[i];
        }
      niftkitkInfoMacro(<<"GenerateData():dx=" << dxInLowResImage << ", dy=" << dyInLowResImage << ", dz=" << dzInLowResImage);
      
      ContinuousIndexType currentPoint;
      ContinuousIndexType neighbourPointInVoxels;
            
      for (unsigned long int z = 0; z < virtualSizeMinus[2]; z++)
        {
          for (unsigned long int y = 0; y < virtualSizeMinus[1]; y++)
            {
              for (unsigned long int x = 0; x < virtualSizeMinus[0]; x++)
                {

                  currentPoint[0] = originInLowResImage[0] + (x * dxInLowResImage[0]);
                  currentPoint[1] = originInLowResImage[1] + (y * dyInLowResImage[1]);
                  currentPoint[2] = originInLowResImage[2] + (z * dzInLowResImage[2]);
                  
                  thisPixel = interpolator->EvaluateAtContinuousIndex(currentPoint);
                  
                  if (thisPixel == this->GetGreyMatterLabel())
                    {

                      virtualIndex[0] = x + m_VoxelMultiplicationFactor;
                      virtualIndex[1] = y + m_VoxelMultiplicationFactor;
                      virtualIndex[2] = z + m_VoxelMultiplicationFactor;

                      indexOfCurrentVoxel = virtualIndex[2]*virtualSize[0]*virtualSize[1] + virtualIndex[1]*virtualSize[0] + virtualIndex[0];

                      fdVox = new FiniteDifferenceVoxelType();
                      
                      virtualImage->TransformIndexToPhysicalPoint( virtualIndex, voxelPointInMillimetres );
                      fdVox->SetValue(0, meanVoltage);
                      fdVox->SetVoxelIndex(virtualIndex);
                      fdVox->SetVoxelPointInMillimetres(voxelPointInMillimetres);
                      fdVox->SetBoundary(false);
                      
                      neighbourIndex[0] = virtualIndex[0]-1;
                      neighbourIndex[1] = virtualIndex[1];
                      neighbourIndex[2] = virtualIndex[2];                         
                      indexOfNeighbour = neighbourIndex[2]*virtualSize[0]*virtualSize[1] + neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetMinus(0, indexOfNeighbour);

                      neighbourPointInVoxels[0] = originInLowResImage[0] + ((x-1) * dxInLowResImage[0]);
                      neighbourPointInVoxels[1] = originInLowResImage[1] + ((y)   * dyInLowResImage[1]);
                      neighbourPointInVoxels[2] = originInLowResImage[2] + ((z)   * dzInLowResImage[2]);
                      neighbourPixel = interpolator->EvaluateAtContinuousIndex(neighbourPointInVoxels);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      neighbourIndex[0] = virtualIndex[0]+1;
                      neighbourIndex[1] = virtualIndex[1];
                      neighbourIndex[2] = virtualIndex[2];                         
                      indexOfNeighbour = neighbourIndex[2]*virtualSize[0]*virtualSize[1] + neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetPlus(0, indexOfNeighbour);

                      neighbourPointInVoxels[0] = originInLowResImage[0] + ((x+1) * dxInLowResImage[0]);
                      neighbourPointInVoxels[1] = originInLowResImage[1] + ((y)   * dyInLowResImage[1]);
                      neighbourPointInVoxels[2] = originInLowResImage[2] + ((z)   * dzInLowResImage[2]);
                      neighbourPixel = interpolator->EvaluateAtContinuousIndex(neighbourPointInVoxels);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      neighbourIndex[0] = virtualIndex[0];
                      neighbourIndex[1] = virtualIndex[1]-1;
                      neighbourIndex[2] = virtualIndex[2];                         
                      indexOfNeighbour = neighbourIndex[2]*virtualSize[0]*virtualSize[1] + neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetMinus(1, indexOfNeighbour);

                      neighbourPointInVoxels[0] = originInLowResImage[0] + ((x)   * dxInLowResImage[0]);
                      neighbourPointInVoxels[1] = originInLowResImage[1] + ((y-1) * dyInLowResImage[1]);
                      neighbourPointInVoxels[2] = originInLowResImage[2] + ((z)   * dzInLowResImage[2]);
                      neighbourPixel = interpolator->EvaluateAtContinuousIndex(neighbourPointInVoxels);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      neighbourIndex[0] = virtualIndex[0];
                      neighbourIndex[1] = virtualIndex[1]+1;
                      neighbourIndex[2] = virtualIndex[2];                         
                      indexOfNeighbour = neighbourIndex[2]*virtualSize[0]*virtualSize[1] + neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetPlus(1, indexOfNeighbour);

                      neighbourPointInVoxels[0] = originInLowResImage[0] + ((x)   * dxInLowResImage[0]);
                      neighbourPointInVoxels[1] = originInLowResImage[1] + ((y+1) * dyInLowResImage[1]);
                      neighbourPointInVoxels[2] = originInLowResImage[2] + ((z)   * dzInLowResImage[2]);
                      neighbourPixel = interpolator->EvaluateAtContinuousIndex(neighbourPointInVoxels);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      neighbourIndex[0] = virtualIndex[0];
                      neighbourIndex[1] = virtualIndex[1];
                      neighbourIndex[2] = virtualIndex[2]-1;                         
                      indexOfNeighbour = neighbourIndex[2]*virtualSize[0]*virtualSize[1] + neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetMinus(2, indexOfNeighbour);

                      neighbourPointInVoxels[0] = originInLowResImage[0] + ((x)   * dxInLowResImage[0]);
                      neighbourPointInVoxels[1] = originInLowResImage[1] + ((y)   * dyInLowResImage[1]);
                      neighbourPointInVoxels[2] = originInLowResImage[2] + ((z-1) * dzInLowResImage[2]);
                      neighbourPixel = interpolator->EvaluateAtContinuousIndex(neighbourPointInVoxels);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      neighbourIndex[0] = virtualIndex[0];
                      neighbourIndex[1] = virtualIndex[1];
                      neighbourIndex[2] = virtualIndex[2]+1;                         
                      indexOfNeighbour = neighbourIndex[2]*virtualSize[0]*virtualSize[1] + neighbourIndex[1]*virtualSize[0] + neighbourIndex[0];
                      fdVox->SetPlus(2, indexOfNeighbour);

                      neighbourPointInVoxels[0] = originInLowResImage[0] + ((x)   * dxInLowResImage[0]);
                      neighbourPointInVoxels[1] = originInLowResImage[1] + ((y)   * dyInLowResImage[1]);
                      neighbourPointInVoxels[2] = originInLowResImage[2] + ((z+1) * dzInLowResImage[2]);
                      neighbourPixel = interpolator->EvaluateAtContinuousIndex(neighbourPointInVoxels);
                      this->InsertNeighbour(fdVox, neighbourPixel, indexOfNeighbour, neighbourIndex, m_MapOfVoxels, numberOfDuplicates, numberOfBoundary);                        

                      fdVox->SetVoxelArrayIndex(indexOfCurrentVoxel);
                      
                      m_MapOfVoxels.insert(PairType(indexOfCurrentVoxel, fdVox));
                      numberOfGrey++;

                    } // end if this is a grey matter voxel

                } // end for x
              
            } // end for y
          
          niftkitkInfoMacro(<<"GenerateData():Slice=" << z << ", total voxels=" << m_MapOfVoxels.size());
          
        } // end for z
    }

  niftkitkInfoMacro(<<"GenerateData():Inserted numberOfGrey=" << numberOfGrey \
      << ", numberOfBoundary=" << numberOfBoundary \
      << ", numberOfDuplicates=" << numberOfDuplicates \
      << ", size=" << m_MapOfVoxels.size() \
      << ", sizeof=" << fdVox->GetSizeofObject() \
      << ", mem for map=" << m_MapOfVoxels.size() * fdVox->GetSizeofObject() \
      << " b or " << m_MapOfVoxels.size() * fdVox->GetSizeofObject() / (double) 1024 / (double) 1024 / (double) 1024 \
      << " Gb" \
      );

  /**
   * Sanity check, that all values are:  lowVoltage <= val <= highVoltage.
   * Check all pointers for non-boundary are connected, as the next stage relies on this.
   */
  indexOfCurrentVoxel = 0;
  IteratorType checkIterator;
  
  for (iterator = m_MapOfVoxels.begin(); iterator != m_MapOfVoxels.end(); iterator++)
    {
      fdVox = (*iterator).second;

      if (fdVox->GetValue(0) < this->GetLowVoltage())
        {
    	  niftkitkErrorMacro("GenerateData():Found erroneous Laplacian value " << fdVox->GetValue(0) \
              << ", at index " << fdVox->GetVoxelIndex() \
              << ", when lower limit is " << this->GetLowVoltage() \
              << ", so clamping to lower value");
              fdVox->SetValue(0, this->GetLowVoltage());
        }
      if (fdVox->GetValue(0) > this->GetHighVoltage())
        {
    	  niftkitkErrorMacro("GenerateData():Found erroneous Laplacian value " << fdVox->GetValue(0) \
              << ", at index " << fdVox->GetVoxelIndex() \
              << ", when upper limit is " << this->GetHighVoltage() \
              << ", so clamping to upper value");
              fdVox->SetValue(0, this->GetHighVoltage());
        }
      
      if (!fdVox->GetBoundary()) 
        {
          std::string listOfIndexes;
          for (unsigned int i = 0; i < this->Dimension; i++)
            {
              listOfIndexes = listOfIndexes + niftk::ConvertToString((int)(fdVox->GetMinus(i))) + ", ";
              listOfIndexes = listOfIndexes + niftk::ConvertToString((int)(fdVox->GetPlus(i))) + ", ";
            }
          
          for (unsigned int i = 0; i < this->Dimension; i++)
            {
              checkIterator = m_MapOfVoxels.find(fdVox->GetPlus(i));
              if (checkIterator == m_MapOfVoxels.end())
                {
                  itkExceptionMacro(<< "Found invalid pointer at map position=" << indexOfCurrentVoxel << ", index=" << (*iterator).first << ", i.e. index=" << fdVox->GetVoxelIndex() << ", size=" << virtualSize << ", arrayIndex=" << fdVox->GetVoxelArrayIndex() << ", with neighbours=" << listOfIndexes << ", and +dimension=" << i);
                }
              checkIterator = m_MapOfVoxels.find(fdVox->GetMinus(i));
              if (checkIterator == m_MapOfVoxels.end())
                {
                  itkExceptionMacro(<< "Found invalid pointer at map position=" << indexOfCurrentVoxel << ", index=" << (*iterator).first << ", i.e. index=" << fdVox->GetVoxelIndex() << ", size=" << virtualSize << ", arrayIndex=" << fdVox->GetVoxelArrayIndex() << ", with neighbours=" << listOfIndexes << ", and -dimension=" << i);
                }
            }
        }
      indexOfCurrentVoxel++;
    }

  /** Precalculate multipliers and denominators. */
  OutputImageSpacing multipliers;
  OutputPixelType multiplier = 0;
  OutputPixelType denominator = 0;
  unsigned int dimensionIndex = 0;
  unsigned int dimensionIndexForAnisotropicScaleFactors = 0;
  
  for (dimensionIndex = 0; dimensionIndex < this->Dimension; dimensionIndex++)
    {
      multiplier = 1;
      
      for (dimensionIndexForAnisotropicScaleFactors = 0; dimensionIndexForAnisotropicScaleFactors < this->Dimension; dimensionIndexForAnisotropicScaleFactors++)
        {
          if (dimensionIndexForAnisotropicScaleFactors != dimensionIndex)
            {
              multiplier *= (virtualSpacing[dimensionIndexForAnisotropicScaleFactors] * virtualSpacing[dimensionIndexForAnisotropicScaleFactors]);
            }
        }
      multipliers[dimensionIndex] = multiplier;
      denominator += multiplier;
      niftkitkDebugMacro(<<"GenerateData():Anisotropic multiplier[" << dimensionIndex << "]=" <<  multipliers[dimensionIndex]);
    }
  denominator *= 2.0;
  niftkitkDebugMacro(<<"GenerateData():Denominator:" << denominator);

  /** Start of main Laplace bit. */
  
  OutputPixelType epsilonRatio = 1;
  OutputPixelType currentPixelValue = 0;
  OutputPixelType currentPixelValuePlus = 0;  
  OutputPixelType currentPixelValueMinus = 0;  
  OutputPixelType currentPixelEnergy = 0;
  OutputPixelType currentFieldEnergy = 0;
  OutputPixelType previousFieldEnergy = 0;
  
  this->SetCurrentIteration(0);
  
  while (this->GetCurrentIteration() < this->GetMaximumNumberOfIterations() 
      && epsilonRatio >= this->GetEpsilonConvergenceThreshold())
    {
      currentFieldEnergy = 0;
      
      for (iterator = m_MapOfVoxels.begin(); iterator != m_MapOfVoxels.end(); iterator++)
        {
          currentPixelValue = 0;
          currentPixelEnergy = 0;
          
          if (!((*iterator).second)->GetBoundary())
            {
              
              for (dimensionIndex = 0; dimensionIndex < this->Dimension; dimensionIndex++)
                {                  
                  currentPixelValuePlus = m_MapOfVoxels[((*iterator).second)->GetPlus(dimensionIndex)]->GetValue(0);
                  currentPixelValueMinus = m_MapOfVoxels[((*iterator).second)->GetMinus(dimensionIndex)]->GetValue(0);
                  
                  currentPixelValue += (multipliers[dimensionIndex] * (currentPixelValuePlus + currentPixelValueMinus));
                  
                  currentPixelEnergy += (((currentPixelValuePlus - currentPixelValueMinus)/virtualSpacing[dimensionIndex])
                                        *((currentPixelValuePlus - currentPixelValueMinus)/virtualSpacing[dimensionIndex]));                  
                }
              
              currentPixelValue /= denominator;
              currentPixelEnergy = sqrt(currentPixelEnergy);
              currentFieldEnergy += currentPixelEnergy;

              indexOfCurrentVoxel = ((*iterator).second)->GetVoxelArrayIndex();
              m_MapOfVoxels[indexOfCurrentVoxel]->SetValue(0, currentPixelValue); 
            }          
        }
      if (this->GetCurrentIteration() != 0)
        {
          epsilonRatio = fabs((previousFieldEnergy - currentFieldEnergy) / previousFieldEnergy);  
        }

      niftkitkInfoMacro(<<"GenerateData():[" << this->GetCurrentIteration() \
          << "] maxIterations=" << this->GetMaximumNumberOfIterations()  \
          << ", currentFieldEnergy=" << currentFieldEnergy \
          << ", previousFieldEnergy=" << previousFieldEnergy 
          << ", epsilonRatio=" << epsilonRatio 
          << ", epsilonTolerance=" << this->GetEpsilonConvergenceThreshold() \
          );
      previousFieldEnergy = currentFieldEnergy;
      this->SetCurrentIteration(this->GetCurrentIteration() + 1);
    }
    
  /**
   * Sanity check, that all values are:  lowVoltage <= val <= highVoltage.
   * Check all pointers for non-boundary are connected, as the next stage relies on this.
   */
  indexOfCurrentVoxel = 0;
  
  for (iterator = m_MapOfVoxels.begin(); iterator != m_MapOfVoxels.end(); iterator++)
    {
      fdVox = (*iterator).second;
      if (fdVox->GetValue(0) < this->GetLowVoltage())
        {
    	  niftkitkErrorMacro("GenerateData():Found erroneous Laplacian value " << fdVox->GetValue(0) \
              << ", at index " << fdVox->GetVoxelIndex() \
              << ", when lower limit is " << this->GetLowVoltage() \
              << ", so clamping to lower value");
              fdVox->SetValue(0, this->GetLowVoltage());
        }
      if (fdVox->GetValue(0) > this->GetHighVoltage())
        {
    	  niftkitkErrorMacro("GenerateData():Found erroneous Laplacian value " << fdVox->GetValue(0) \
              << ", at index " << fdVox->GetVoxelIndex() \
              << ", when upper limit is " << this->GetHighVoltage() \
              << ", so clamping to upper value");
              fdVox->SetValue(0, this->GetHighVoltage());
        }
      if (!fdVox->GetBoundary()) 
        {
          std::string listOfIndexes;
          for (unsigned int i = 0; i < this->Dimension; i++)
            {
              listOfIndexes = listOfIndexes + niftk::ConvertToString((int)(fdVox->GetMinus(i))) + ", ";
              listOfIndexes = listOfIndexes + niftk::ConvertToString((int)(fdVox->GetPlus(i))) + ", ";
            }
          
          for (unsigned int i = 0; i < this->Dimension; i++)
            {
              checkIterator = m_MapOfVoxels.find(fdVox->GetPlus(i));
              if (checkIterator == m_MapOfVoxels.end())
                {
                  itkExceptionMacro(<< "Found invalid pointer at map position=" << indexOfCurrentVoxel << ", index=" << (*iterator).first << ", i.e. index=" << fdVox->GetVoxelIndex() << ", size=" << virtualSize << ", arrayIndex=" << fdVox->GetVoxelArrayIndex() << ", with neighbours=" << listOfIndexes << ", and +dimension=" << i);
                }
              checkIterator = m_MapOfVoxels.find(fdVox->GetMinus(i));
              if (checkIterator == m_MapOfVoxels.end())
                {
                  itkExceptionMacro(<< "Found invalid pointer at map position=" << indexOfCurrentVoxel << ", index=" << (*iterator).first << ", i.e. index=" << fdVox->GetVoxelIndex() << ", size=" << virtualSize << ", arrayIndex=" << fdVox->GetVoxelArrayIndex() << ", with neighbours=" << listOfIndexes << ", and -dimension=" << i);
                }
            }
        }
      indexOfCurrentVoxel++;
    }

  /** 
   * Now the big question. How to get the result out?
   */
  
  OutputImagePointer counterImage = OutputImageType::New();
  counterImage->SetRegions(inputImage->GetLargestPossibleRegion());
  counterImage->SetDirection(inputImage->GetDirection());
  counterImage->SetSpacing(inputImage->GetSpacing());
  counterImage->SetOrigin(inputImage->GetOrigin());
  counterImage->Allocate();
  counterImage->FillBuffer(0);

  OutputImagePointer accumulationImage = OutputImageType::New();
  accumulationImage->SetRegions(inputImage->GetLargestPossibleRegion());
  accumulationImage->SetDirection(inputImage->GetDirection());
  accumulationImage->SetSpacing(inputImage->GetSpacing());
  accumulationImage->SetOrigin(inputImage->GetOrigin());
  accumulationImage->Allocate();
  accumulationImage->FillBuffer(0);

  /*
   * Iterate through map, plotting points into the accumulation image, and incrementing counter image.
   * The output will be accumulationImage / counterImage. i.e. take an average.
   */

  PointType point;
  ContinuousIndexType continuousIndex;
  OutputImageIndexType outputIndex;
  InputImageIndexType inputIndex;
  InputImagePixelType inputValue;

  for (iterator = m_MapOfVoxels.begin(); iterator != m_MapOfVoxels.end(); iterator++)
    {
      fdVox = (*iterator).second;
      continuousIndex = fdVox->GetVoxelIndex();
      virtualImage->TransformContinuousIndexToPhysicalPoint(continuousIndex, point);
      outputImage->TransformPhysicalPointToIndex( point, outputIndex );
      accumulationImage->SetPixel(outputIndex, accumulationImage->GetPixel(outputIndex) + fdVox->GetValue(0));
      counterImage->SetPixel(outputIndex, counterImage->GetPixel(outputIndex) + 1);
    }

  ImageRegionConstIteratorWithIndex<InputImageType> inputIterator(inputImage, inputImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> accumulationIterator(accumulationImage, accumulationImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> counterIterator(counterImage, counterImage->GetLargestPossibleRegion());
  for (outputIterator.GoToBegin(), inputIterator.GoToBegin(), accumulationIterator.GoToBegin(), counterIterator.GoToBegin();
       !outputIterator.IsAtEnd();
       ++outputIterator, ++inputIterator, ++accumulationIterator, ++counterIterator)
    {
      inputValue = inputIterator.Get();
      inputIndex = inputIterator.GetIndex();
      
      if(inputValue == this->GetWhiteMatterLabel())
        {
          outputIterator.Set(this->GetLowVoltage());
        }
      else if (inputValue == this->GetExtraCerebralMatterLabel())
        {
          outputIterator.Set(this->GetHighVoltage());
        }
      else
        {
          if (counterIterator.Get() > 0)
            {
              outputIterator.Set(accumulationIterator.Get() / counterIterator.Get());
            }
          else
            {
              outputIterator.Set(0);
            }
        }      
    }

  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif // __itkImageRegistrationFilter_txx
