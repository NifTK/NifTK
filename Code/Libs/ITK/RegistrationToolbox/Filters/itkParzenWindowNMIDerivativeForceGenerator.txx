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
#ifndef __itkParzenWindowNMIDerivativeForceGenerator_txx
#define __itkParzenWindowNMIDerivativeForceGenerator_txx

#include "itkParzenWindowNMIDerivativeForceGenerator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkVector.h"

#include "itkLogHelper.h"

namespace itk {

template< class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar >
ParzenWindowNMIDerivativeForceGenerator< TFixedImage, TMovingImage, TScalarType, TDeformationScalar >
::ParzenWindowNMIDerivativeForceGenerator()
{
}

template< class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar > 
void
ParzenWindowNMIDerivativeForceGenerator< TFixedImage, TMovingImage, TScalarType, TDeformationScalar >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar > 
void
ParzenWindowNMIDerivativeForceGenerator<TFixedImage, TMovingImage, TScalarType, TDeformationScalar > 
::BeforeThreadedGenerateData()
{
  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Started, with fixed intensity range:" << this->GetFixedLowerPixelValue() \
      << ", " << this->GetFixedUpperPixelValue() \
      << ", and moving range:" << this->GetMovingLowerPixelValue() \
      << ", " << this->GetMovingUpperPixelValue() \
  );
  
  if( !m_ScalarImageGradientFilter )
    {
      itkExceptionMacro(<< "ScalarImageGradientFilter not set");
    }

  Superclass::BeforeThreadedGenerateData();
  
  // Force an up-to-date computation of image gradient.
  InputImageType *fixedImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  InputImageType *movingImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(2));
  
  this->m_ScalarImageGradientFilter->SetFixedImage(fixedImage);
  this->m_ScalarImageGradientFilter->SetMovingImage(movingImage);
  this->m_ScalarImageGradientFilter->UpdateLargestPossibleRegion();

  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Finished");
}

template< class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar > 
void
ParzenWindowNMIDerivativeForceGenerator< TFixedImage, TMovingImage, TScalarType, TDeformationScalar >
::ThreadedGenerateData(const RegionType& outputRegionForThread, int threadNumber) 
{
  niftkitkDebugMacro(<<"ThreadedGenerateData():Computing histogram force, using Marc Modat's method, thread:" << threadNumber);

  OutputDataType fixedImageEntropy = 0.0;
  OutputDataType transformedMovingImageEntropy = 0.0;
  OutputDataType jointEntropy = 0.0;
  OutputDataType totalFrequency = 0.0;
  OutputDataType NMI;
  OutputDataType jointLog;
  OutputDataType fixedLog;
  OutputDataType movingLog;
  OutputDataType commonValue;
  OutputDataType temp;
  InputPixelType fixedValue;
  InputPixelType movingValue;
  OutputPixelType movingGradientValue;
  OutputPixelType outputValue;
  OutputPixelType zeroValue;
  unsigned int dimension;
  typename HistogramType::MeasurementVectorType histogramSample;
  typename HistogramType::IndexType histogramIndex; 
  int f;
  int m;

  InputPixelType fixedLowerBound = this->GetFixedLowerPixelValue();
  InputPixelType fixedUpperBound = this->GetFixedUpperPixelValue();
  InputPixelType movingLowerBound = this->GetMovingLowerPixelValue();
  InputPixelType movingUpperBound = this->GetMovingUpperPixelValue();
  
  // Pointers to images.
  typename InputImageType::ConstPointer inputFixedImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename InputImageType::ConstPointer transformedMovingImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));

  typename OutputImageType::Pointer gradientImage 
    = this->m_ScalarImageGradientFilter->GetOutput();

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  
  // Get Histogram
  HistogramPointer histogram = this->GetMetric()->GetHistogram();
  
  // This assumes that the similarity measure has already been run!
  transformedMovingImageEntropy = histogram->EntropyMoving();
  fixedImageEntropy             = histogram->EntropyFixed();
  jointEntropy                  = histogram->JointEntropy();
  totalFrequency                = histogram->GetTotalFrequency();
  NMI = (fixedImageEntropy + transformedMovingImageEntropy) / jointEntropy;

  niftkitkDebugMacro(<<"ThreadedGenerateData():H(f)=" << fixedImageEntropy \
      << ", H(m)=" << transformedMovingImageEntropy \
      << ", H(f,m)=" << jointEntropy \
      << ", NMI=" << NMI \
      << ", frequency=" << totalFrequency);
  
  // We build a simpler histogram of all marginal probabilities
  std::vector<double> fastFixedImageHistogram;
  std::vector<double> fastTransformedImageHistogram;
  fastFixedImageHistogram.clear();
  fastTransformedImageHistogram.clear();
  
  unsigned int fixedSize = histogram->GetSize()[0];
  unsigned int movingSize = histogram->GetSize()[1];
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():fixedSize=" << fixedSize << ", " << movingSize);
  
  for (unsigned int i = 0; i < fixedSize; i++)
  {
    HistogramFrequencyType freq = histogram->GetFrequency(i, 0);
    double value = 0;
    
    if (freq > 0)
      {
        value = vcl_log((double)freq/(double)totalFrequency);
        fastFixedImageHistogram.push_back(value);  
      }
    else
      {
        fastFixedImageHistogram.push_back(0); 
      }
  }
  
  // Transformed moving image marginal entropy.
  for (unsigned int i = 0; i < movingSize; i++)
  {
    HistogramFrequencyType freq = histogram->GetFrequency(i, 1);
    double value = 0;
    
    if (freq > 0)
      {
        value = vcl_log((double)freq/(double)totalFrequency);
        fastTransformedImageHistogram.push_back(value);  
      }
    else
      {
        fastTransformedImageHistogram.push_back(0);
      }
  }
  
  // Get some iterators
  typedef ImageRegionConstIterator<InputImageType> InputIteratorType;
  typedef ImageRegionIterator<OutputImageType> OutputIteratorType;
  
  InputIteratorType fixedImageIterator(inputFixedImage, outputRegionForThread);
  InputIteratorType transformedMovingImageIterator(transformedMovingImage,  outputRegionForThread);
  OutputIteratorType gradientImageIterator(gradientImage, outputRegionForThread);
  OutputIteratorType outputImageIterator(outputImage, outputRegionForThread);
  
  fixedImageIterator.GoToBegin();
  transformedMovingImageIterator.GoToBegin();
  gradientImageIterator.GoToBegin();
  outputImageIterator.GoToBegin();
  
  // Technically shouldnt be a 'Spacing' type, but we know its an array of floating point
  OutputImageSpacingType fixedImageEntropyDerivative;
  OutputImageSpacingType movingImageEntropyDerivative;
  OutputImageSpacingType jointEntropyDerivative;
  OutputImageSpacingType spacing = outputImage->GetSpacing();
  
  zeroValue.Fill(0);
  
  unsigned long int counter=0;
  
  while (!fixedImageIterator.IsAtEnd() 
      && !transformedMovingImageIterator.IsAtEnd() 
      && !gradientImageIterator.IsAtEnd()
      && !outputImageIterator.IsAtEnd())
    {
      fixedValue = fixedImageIterator.Get();
      movingValue = transformedMovingImageIterator.Get();
      movingGradientValue = gradientImageIterator.Get();
      
      if (fixedValue > fixedLowerBound && fixedValue <= fixedUpperBound 
          && movingValue > movingLowerBound && movingValue <= movingUpperBound)
        {

          //printf("Matt, target=%f, result=%f, gradient=%f, %f, %f\n", fixedValue, movingValue, movingGradientValue[0], movingGradientValue[1], movingGradientValue[2]);
          
          for (dimension = 0; dimension < Dimension; dimension++)
            {
              movingGradientValue[dimension] /= (OutputDataType)totalFrequency;
              fixedImageEntropyDerivative[dimension] = 0;
              movingImageEntropyDerivative[dimension] = 0;
              jointEntropyDerivative[dimension] = 0;	
            }  

          for(f = (int)(fixedValue - 2); f < (int)(fixedValue + 3); f++)
            {
              for (m = (int)(movingValue - 2); m < (int)(movingValue + 3); m++)
                {
                	            
                  if (f >= (int)0 && m >= (int)0 && f < (int)fixedSize && m < (int)movingSize)
                    {
                      commonValue = this->GetMetric()->GetParzenValue(f - fixedValue)
                                  * this->GetMetric()->GetParzenDerivative(m - movingValue);

                      fixedLog = fastFixedImageHistogram[f];
                      movingLog = fastTransformedImageHistogram[m];
                      
                      histogramIndex[0] = f;
                      histogramIndex[1] = m;
                      
                      jointLog = histogram->GetFrequency(histogramIndex);
                      
                      if (jointLog > 0)
                        {
                          jointLog = vcl_log(jointLog/(OutputDataType)totalFrequency);  
                        }
                      else
                        {
                          jointLog = 0;  
                        }
                      
                      for (dimension = 0; dimension < Dimension; dimension++)
                        {
                          temp = commonValue * movingGradientValue[dimension];
                          jointEntropyDerivative[dimension] -= temp*jointLog;
                          fixedImageEntropyDerivative[dimension] -= temp*fixedLog;
                          movingImageEntropyDerivative[dimension] -= temp*movingLog;

                        }

                    } // if inside histogram
                } // for m
            } // for f
            
          for (dimension = 0; dimension < Dimension; dimension++)
            {
              outputValue[dimension] = 
                                       (
                                         fixedImageEntropyDerivative[dimension] 
                                         + movingImageEntropyDerivative[dimension] 
                                         - NMI * jointEntropyDerivative[dimension]
                                       ) 
                                       / jointEntropy;
            }

          if (this->GetScaleToSizeOfVoxelAxis())
            {
              for (dimension = 0; dimension < Dimension; dimension++)
                { 
                  outputValue[dimension] *= spacing[dimension]; 
                }      
            }

          outputImageIterator.Set(outputValue);

//          printf("Matt: counter=%d: fixedValue=%f, movingValue=%f, movingGradientValue=[%f, %f, %f], outputValue=[%f, %f, %f] \n", counter, fixedValue, movingValue, movingGradientValue[0], movingGradientValue[1], movingGradientValue[2], outputValue[0], outputValue[1], outputValue[2] );
          
        } // if fixed and moving value > m_PadValue (i.e. inside mask).
        
      else
        {
          outputImageIterator.Set(zeroValue);  
        }

      // Don't forget these.
      ++fixedImageIterator;
      ++transformedMovingImageIterator;
      ++gradientImageIterator;
      ++outputImageIterator; 
      counter++;
    }
          
  niftkitkDebugMacro(<<"ThreadedGenerateData():Computing histogram force, using Marc Modat's method, thread:" << threadNumber << ", DONE");
}

} // end namespace itk

#endif
