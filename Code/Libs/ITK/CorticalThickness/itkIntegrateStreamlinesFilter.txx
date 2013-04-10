/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkIntegrateStreamlinesFilter_txx
#define __itkIntegrateStreamlinesFilter_txx

#include "itkIntegrateStreamlinesFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

namespace itk
{
template <class TImageType, typename TScalarType, unsigned int NDimensions> 
IntegrateStreamlinesFilter<TImageType, TScalarType, NDimensions>
::IntegrateStreamlinesFilter()
{
  m_MinIterationVoltage = 0;
  m_MaxIterationVoltage = 10000;
  m_StepSize = 0.1;
  m_MaxIterationLength = 10.0;
  
  niftkitkDebugMacro(<<"IntegrateStreamlinesFilter():Constructed, " \
    << "m_MinIterationVoltage=" << m_MinIterationVoltage \
    << ", m_MaxIterationVoltage=" << m_MaxIterationVoltage \
    << ", m_MaxIterationLength=" << m_MaxIterationLength \
    << ", m_StepSize=" << m_StepSize );
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
IntegrateStreamlinesFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "MinIterationVoltage:" << m_MinIterationVoltage << std::endl;
  os << indent << "MaxIterationVoltage:" << m_MaxIterationVoltage << std::endl;  
  os << indent << "MaxIterationLength:" << m_MaxIterationLength << std::endl;  
  os << indent << "StepSize:" << m_StepSize << std::endl;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
double
IntegrateStreamlinesFilter<TImageType, TScalarType, NDimensions>
::GetLengthToThreshold(
                       const InputScalarImagePointType &startingPoint,
                       const InputScalarImagePixelType &initialValue,
                       const InputScalarImagePixelType &threshold,
                       const InputVectorImagePointer   &vectorImage,
                       const InputScalarImagePointer   &scalarImage,
                       const VectorInterpolatorPointer &vectorInterpolator,
                       const ScalarInterpolatorPointer &scalarInterpolator,                       
                       const double &multiplier, // 1 or -1
                       const bool &debug,
                       bool &maxLengthExceeded
                      )
{
  unsigned long int           i = 0;
  double                      length = 0;
  double                      vectorLength = 0;
  InputScalarImagePixelType   scalarPixel;
  InputVectorImagePixelType   vectorPixel;
  InputVectorImagePixelType   vectorOffset;
  VectorInterpolatorPointType imagePoint;
  
  // Default return value
  maxLengthExceeded = false;
  bool hitThreshold = false;
  
  for (i = 0; i < NDimensions; i++)
    {
      imagePoint[i] = startingPoint[i];  
    }  
  
  scalarPixel = initialValue;

  while (!hitThreshold && !maxLengthExceeded)
    {

      if(vectorInterpolator->IsInsideBuffer(imagePoint))
        {
        
          vectorPixel = vectorInterpolator->Evaluate(imagePoint);

          // Here, im calculating the interpolated vector length,
          // and then calculating the step size based on a unit vector.
          // The reason is that even though the vectors in the vector image
          // are all of unit length, when they are linearly interpolated,
          // the boundaries have zero length vectors, so the interpolated
          // vector tends to come out smaller as you approach a boundary.
                            
          vectorLength = vcl_sqrt(vectorPixel.GetSquaredNorm());
          
          if (vectorLength == 0)
            {
              // No point continuing
              return 0;
            }
          else
            {
              for (i = 0; i < NDimensions; i++)
                {
                  vectorOffset[i] = (vectorPixel[i]*m_StepSize*multiplier/vectorLength);
                  imagePoint[i] += vectorOffset[i]; 
                }
              length += vcl_sqrt(vectorOffset.GetSquaredNorm());
              
              if (scalarInterpolator->IsInsideBuffer(imagePoint))
                {
                  scalarPixel = scalarInterpolator->Evaluate(imagePoint);
                }
              else
                {
                  // must have just gone out of bounds
                  return length;
                }

              if (debug)
                {
            	  niftkitkDebugMacro(<<"GetLengthToThreshold():sp=" << startingPoint \
                      << ", iv=" << initialValue \
                      << ", vec=" << vectorPixel \
                      << ", cur=" << imagePoint \
                      << ", val=" << scalarPixel \
                      << ", m=" << multiplier \
                      << ", t=" << threshold \
                      << ", l=" << length \
                      );
                }

              if (   (multiplier == 1 && scalarPixel >= threshold)
                   ||(multiplier == -1 && scalarPixel <= threshold)
                 )
                {
                  hitThreshold = true;
                }
              
              if (length > m_MaxIterationLength)
                {
                  maxLengthExceeded = true;
                }
            }
        }
      else
        {
    	  niftkitkErrorMacro("imagePoint:" << imagePoint << ", is outside buffer??? This shouldnt happen, and suggests a programming bug, or a contrived image example.");
          return length;    
        }
                    
    }  
  if (debug)
    {
      double euclideanLength = 0;
      for (i = 0; i < NDimensions; i++)
        {
          vectorOffset[i] = imagePoint[i] - startingPoint[i];  
        }
      euclideanLength = vcl_sqrt(vectorOffset.GetSquaredNorm());
      niftkitkDebugMacro(<<"GetLengthToThreshold():startingPoint=" << startingPoint \
          << ", euclideanLength=" << euclideanLength \
          << ", laplacianLength=" << length \
          );
    }
  return length;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
IntegrateStreamlinesFilter<TImageType, TScalarType, NDimensions>
::ThreadedGenerateData(const InputScalarImageRegionType& outputRegionForThread, int threadNumber) 
{
 
  // Iterate through each pixel in input region.
  // If > low threshold and < high threshold, start integrating.
  // Integrate in both directions, adding up the length.
  // store length in the corresponding voxel in output image.
  // otherwise length = zero.

  unsigned long int pixelNumber = 0;
  unsigned long int totalNumberOfPixels = 1;
  unsigned long int numberThatExceededMaxLength = 0;
  bool maxLengthExceeded;
  bool debug = false;
  
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      totalNumberOfPixels *= outputRegionForThread.GetSize()[i];
    }

  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread " << threadNumber << " with " << totalNumberOfPixels << " voxels");
  
  typename InputScalarImageType::Pointer scalarImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(0));
  typename InputVectorImageType::Pointer vectorImage = static_cast< InputVectorImageType * >(this->ProcessObject::GetInput(1));
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  niftkitkDebugMacro(<<"ThreadedGenerateData():scalarImage:" << scalarImage.GetPointer() \
    << ", vectorImage:" << vectorImage.GetPointer() \
    << ", outputImage:" << outputImage.GetPointer());
    
  // Create these locally, per thread.
  typename VectorInterpolatorType::Pointer vectorInterpolator = VectorInterpolatorType::New();
  typename ScalarInterpolatorType::Pointer scalarInterpolator = ScalarInterpolatorType::New();
  
  vectorInterpolator->SetInputImage(vectorImage);
  scalarInterpolator->SetInputImage(scalarImage);
  
  ImageRegionConstIteratorWithIndex<InputScalarImageType> scalarIterator(scalarImage, outputRegionForThread);
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputRegionForThread);

  InputScalarImagePixelType scalarPixel;
  InputScalarImageIndexType scalarIndex;
  InputScalarImagePointType startingPoint;
  OutputImagePixelType      resultPixel;
  
  double lengthToHighThreshold;
  double lengthToLowThreshold;
  
  pixelNumber = 0;
  unsigned int tenthPercentIndicator = 1;
  unsigned long int tenthPercentProgressCounter = totalNumberOfPixels / 10;
  
  for (scalarIterator.GoToBegin(), outputIterator.GoToBegin(); 
      !scalarIterator.IsAtEnd() ; 
      ++scalarIterator, ++outputIterator)
    {
      scalarPixel = scalarIterator.Get(); 
      scalarIndex = scalarIterator.GetIndex();

      /*
      niftkitkDebugMacro(<<"ThreadedGenerateData():scalarValue=" << scalarPixel \
          << ", m_LowVoltage=" << this->m_LowVoltage \
          << ", m_HighVoltage=" << this->m_HighVoltage \
          );
      */
      
      if (scalarPixel > this->m_LowVoltage && scalarPixel < this->m_HighVoltage)
        {
          // This implicitly assumes that the coordinate 
          // system of the images is the same. If for some
          // reason, you have different origins, spacing
          // or grid size etc. then it needs changing.
          
          scalarImage->TransformIndexToPhysicalPoint( scalarIndex, startingPoint );

          maxLengthExceeded = false;
          debug = false;
/*          
          if (scalarIndex[0] == 67 && scalarIndex[1] == 128 && scalarIndex[2] == 106)
            {
              niftkitkDebugMacro(<<"ThreadedGenerateData():Index=" << scalarIndex \
                  << ", setting debug to true");
              debug = true;
            }
*/          
          lengthToHighThreshold = this->GetLengthToThreshold(
                                    startingPoint,
                                    scalarPixel,
                                    m_MaxIterationVoltage,
                                    vectorImage, 
                                    scalarImage,
                                    vectorInterpolator,
                                    scalarInterpolator,
                                    1,
                                    debug,
                                    maxLengthExceeded
                                    );
          
          lengthToLowThreshold  = this->GetLengthToThreshold(
                                    startingPoint,
                                    scalarPixel,
                                    m_MinIterationVoltage,
                                    vectorImage,
                                    scalarImage,
                                    vectorInterpolator,
                                    scalarInterpolator,                                    
                                    -1,
                                    debug,
                                    maxLengthExceeded
                                    );
                                    
          resultPixel = lengthToHighThreshold + lengthToLowThreshold;
             
          if (resultPixel > m_MaxIterationLength)
          {
            numberThatExceededMaxLength++;
            resultPixel = m_MaxIterationLength;
          }
        }
      else
        {
          resultPixel = 0;    
        }

      
      outputIterator.Set(resultPixel);
      pixelNumber++;
      
      if (pixelNumber >= tenthPercentProgressCounter)
        {
    	  niftkitkInfoMacro(<<"ThreadedGenerateData():Thread[" << threadNumber \
            << "], completed " << tenthPercentIndicator*10 << " percent i.e. " << pixelNumber << " out of " << totalNumberOfPixels);
          tenthPercentIndicator++;
          tenthPercentProgressCounter = totalNumberOfPixels * tenthPercentIndicator / 10;
        }
    }

  if (numberThatExceededMaxLength > 0)
    {
	  niftkitkWarningMacro("ThreadedGenerateData():Thread[" << threadNumber \
          << "], number that exceeded max length=" << numberThatExceededMaxLength);
    }
  
  niftkitkInfoMacro(<<"ThreadedGenerateData():Thread[" << threadNumber \
          << "], Finished");
}

} // end namespace

#endif // __itkImageRegistrationFilter_txx
