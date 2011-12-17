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
#ifndef __itkLaplacianSolverImageFilter_txx
#define __itkLaplacianSolverImageFilter_txx

#include "itkLaplacianSolverImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

namespace itk
{
template <typename TInputImage, typename TScalarType > 
LaplacianSolverImageFilter<TInputImage, TScalarType>
::LaplacianSolverImageFilter()
{
  m_LowVoltage = 0;
  m_HighVoltage = 10000;
  m_EpsilonConvergenceThreshold = 0.00001;
  m_MaximumNumberOfIterations = 200;
  m_UseGaussSeidel = true;
  m_CurrentIteration = 0;
  niftkitkDebugMacro(<<"LaplacianSolverImageFilter():Constructed" << ", LowVoltage=" << m_LowVoltage << ", HighVoltage=" << m_HighVoltage << ", EpsilonConvergenceThreshold=" << m_EpsilonConvergenceThreshold << ", MaximumNumberOfIterations=" << m_MaximumNumberOfIterations << ", m_UseGaussSeidel=" << m_UseGaussSeidel << ", m_CurrentIteration" << m_CurrentIteration);
}

template <typename TInputImage, typename TScalarType >
void 
LaplacianSolverImageFilter<TInputImage, TScalarType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "LowVoltage:" << m_LowVoltage << std::endl;
  os << indent << "HighVoltage:" << m_HighVoltage << std::endl;
  os << indent << "EpsilonConvergenceThreshold:" << m_EpsilonConvergenceThreshold << std::endl;        
  os << indent << "MaximumNumberOfIterations:" << m_MaximumNumberOfIterations << std::endl;          
  os << indent << "UseGaussSeidel:" << m_UseGaussSeidel << std::endl;
}

template <typename TInputImage, typename TScalarType > 
void
LaplacianSolverImageFilter<TInputImage, TScalarType>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Started, grey Label=" << this->m_GreyMatterLabel
      << ", white label=" << this->m_WhiteMatterLabel
      << ", extra cerebral label=" << this->m_ExtraCerebralMatterLabel );
  
  typename InputImageType::Pointer inputImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  InputImageIndexType index;
  InputImageIndexType indexPlus;
  InputImageIndexType indexMinus;
  InputPixelType      tmp;
  int                 i;
  
  typename OutputImageType::Pointer tmpOutput1 = OutputImageType::New();
  typename OutputImageType::Pointer tmpOutput2 = OutputImageType::New();
  
  typename OutputImageType::Pointer tmpImages[2];
  tmpImages[0] = tmpOutput1;
  tmpImages[1] = tmpOutput2;
  
  // Set temp images to the right size.
  for (i = 0; i < 2; i++)
    {
      tmpImages[i]->SetDirection(inputImage->GetDirection());
      tmpImages[i]->SetSpacing(inputImage->GetSpacing());
      tmpImages[i]->SetOrigin(inputImage->GetOrigin());
      tmpImages[i]->SetRegions(inputImage->GetLargestPossibleRegion());
      tmpImages[i]->Allocate();
      niftkitkDebugMacro(<<"GenerateData():Set tmp image[" << i << "] to size:" << tmpImages[i]->GetLargestPossibleRegion().GetSize() \
          << ", spacing:" << tmpImages[i]->GetSpacing() \
          << ", origin:" << tmpImages[i]->GetOrigin() \
          << ", direction:\n" << tmpImages[i]->GetDirection() \
          );
    }
    
  // Now we initialize the first image to the right voltage.
  // Note: We are also creating a list of indexes that are grey matter.
  ImageRegionConstIteratorWithIndex<InputImageType> inputIterator(
                                                      inputImage,
                                                      inputImage->
                                                      GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> outputIterator1(tmpOutput1, 
                                                      tmpOutput1->
                                                      GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> outputIterator2(tmpOutput2, 
                                                      tmpOutput2->
                                                      GetLargestPossibleRegion());

  std::vector<InputImageIndexType> listOfGreyMatterPixels;                                                    
  listOfGreyMatterPixels.clear();
  
  OutputPixelType meanVoltage = (m_HighVoltage + m_LowVoltage)/2.0;
  
  for (outputIterator1.GoToBegin(), outputIterator2.GoToBegin(), inputIterator.GoToBegin(); 
       !inputIterator.IsAtEnd(); 
       ++inputIterator, ++outputIterator2, ++outputIterator1)
    {
      tmp = inputIterator.Get();   
      if (tmp == this->m_ExtraCerebralMatterLabel)
        {
          outputIterator1.Set(m_HighVoltage);
          outputIterator2.Set(m_HighVoltage);
        }
      else if (tmp == this->m_WhiteMatterLabel)
        {
          outputIterator1.Set(m_LowVoltage);
          outputIterator2.Set(m_LowVoltage);
        }
      else
        {
          // I didnt find this mean voltage bit in the paper, but it
          // seems to help. If I left it as zero, algorithm wasnt converging properly.
          outputIterator1.Set(meanVoltage);
          outputIterator2.Set(meanVoltage);
          listOfGreyMatterPixels.push_back(inputIterator.GetIndex());
        }
    }
  niftkitkDebugMacro(<<"GenerateData():Found " << listOfGreyMatterPixels.size() << " grey matter pixels, set lowVoltage:" << m_LowVoltage << ", highVoltage:" << m_HighVoltage);

  // Performance Note.
  // In the iteration that follows, I could 
  //
  // a.) Define a custom iterator that ONLY visits the grey matter pixels.
  // i.e. by initializing it with the list of indexes listOfGreyMatterPixels
  //
  // b.) Use the SetPixel / GetPixel methods, which are known to be slow,
  // but its simpler to get working.
  //
  // Currently I opted for option B for now. Lets see if its fast enough.
  // I'm hoping that the percentage of grey matter pixels will be so small
  // that this method will be fast enough in practice.
  // It also means that, if your grey matter pixels don't touch the 
  // edge of the image, we don't have to worry about boundary conditions.
  // If your grey matter pixels do touch the edge, this might crash.
  niftkitkDebugMacro(<<"GenerateData():Starting iteration with epsilon threshold:" << m_EpsilonConvergenceThreshold << ", and max iteration threshold:" << m_MaximumNumberOfIterations);
    
  OutputPixelType currentPixelValue;
  OutputPixelType currentPixelValuePlus;  
  OutputPixelType currentPixelValueMinus;  
  OutputPixelType currentPixelEnergy;
  OutputPixelType currentFieldEnergy;
  OutputPixelType previousFieldEnergy = 0;
  OutputPixelType epsilonRatio = 1;
  OutputPixelType denominator = 0;
  OutputPixelType multiplier = 0;
  OutputImageSpacing spacing;
  
  m_CurrentIteration = 0;
  unsigned long int pixelNumber = 0;
  unsigned long int totalNumberOfPixels = listOfGreyMatterPixels.size();
  unsigned int dimensionIndex = 0;
  unsigned int dimensionIndexForAnisotropicScaleFactors = 0;
  
  // We can pre-calculate these multipliers, and denominator.
  OutputImageSpacing multipliers;
  spacing = tmpImages[0]->GetSpacing(); // these images should always be same size.
  denominator = 0;

  for (dimensionIndex = 0; dimensionIndex < this->Dimension; dimensionIndex++)
    {
      multiplier = 1;
      
      for (dimensionIndexForAnisotropicScaleFactors = 0; dimensionIndexForAnisotropicScaleFactors < this->Dimension; dimensionIndexForAnisotropicScaleFactors++)
        {
          if (dimensionIndexForAnisotropicScaleFactors != dimensionIndex)
            {
              multiplier *= (spacing[dimensionIndexForAnisotropicScaleFactors] * spacing[dimensionIndexForAnisotropicScaleFactors]);
            }
        }
      multipliers[dimensionIndex] = multiplier;
      denominator += multiplier;
      niftkitkDebugMacro(<<"GenerateData():Anisotropic multiplier[" << dimensionIndex << "]=" <<  multipliers[dimensionIndex]);
    }
  denominator *= 2.0;
  niftkitkDebugMacro(<<"GenerateData():Denominator:" << denominator);
  
  while (m_CurrentIteration < m_MaximumNumberOfIterations && epsilonRatio >= m_EpsilonConvergenceThreshold)
    {
      // Sort out which image we are reading from / writing to.
      if (m_UseGaussSeidel)
        {
            // Read and write to the same image.
            tmpImages[0] = tmpOutput1;
            tmpImages[1] = tmpOutput1;    
        }
      else 
        {
          if (m_CurrentIteration % 2 == 0)
            {
              tmpImages[0] = tmpOutput1;
              tmpImages[1] = tmpOutput2;  
            }
          else
            {
              tmpImages[0] = tmpOutput2;
              tmpImages[1] = tmpOutput1;
            }
        }

      // So, from here-on-in, 
      // reading from tmpImages[0]
      // writing to   tmpImages[1];
      
      currentFieldEnergy = 0;
      
      for (pixelNumber = 0; pixelNumber < totalNumberOfPixels; pixelNumber++)
        {
          index = listOfGreyMatterPixels[pixelNumber];

          currentPixelValue = 0;
          currentPixelEnergy = 0;
          
          for (dimensionIndex = 0; dimensionIndex < this->Dimension; dimensionIndex++)
            {
              indexPlus = index;
              indexMinus = index;
              
              indexPlus[dimensionIndex] += 1;
              indexMinus[dimensionIndex] -= 1;  
              
              currentPixelValuePlus = tmpImages[0]->GetPixel(indexPlus);
              currentPixelValueMinus = tmpImages[0]->GetPixel(indexMinus);

              currentPixelValue += (multipliers[dimensionIndex] * (currentPixelValuePlus + currentPixelValueMinus));
              
              currentPixelEnergy += (((currentPixelValuePlus - currentPixelValueMinus)/spacing[dimensionIndex])
                                    *((currentPixelValuePlus - currentPixelValueMinus)/spacing[dimensionIndex]));

            }
          currentPixelValue /= denominator;
          tmpImages[1]->SetPixel(index, currentPixelValue);
          
          currentPixelEnergy = sqrt(currentPixelEnergy);
          currentFieldEnergy += currentPixelEnergy;
        }
      if (m_CurrentIteration != 0)
        {
          epsilonRatio = fabs((previousFieldEnergy - currentFieldEnergy) / previousFieldEnergy);  
        }

      niftkitkInfoMacro(<<"GenerateData():[" << m_CurrentIteration << "] currentFieldEnergy=" << currentFieldEnergy << ", previousFieldEnergy=" << previousFieldEnergy << ", epsilonRatio=" << epsilonRatio << ", epsilonTolerance=" << m_EpsilonConvergenceThreshold);
      previousFieldEnergy = currentFieldEnergy;
                  
      m_CurrentIteration++;
    }

  niftkitkDebugMacro(<<"GenerateData():Grafting output from:" << tmpImages[1].GetPointer());
  this->GraftOutput( tmpImages[1] );

  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif // __itkImageRegistrationFilter_txx
