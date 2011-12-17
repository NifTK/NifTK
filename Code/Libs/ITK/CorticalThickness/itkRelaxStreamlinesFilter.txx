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
#ifndef __itkRelaxStreamlinesFilter_txx
#define __itkRelaxStreamlinesFilter_txx

#include "itkRelaxStreamlinesFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

namespace itk
{

template <class TImageType, typename TScalarType, unsigned int NDimensions> 
RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::~RelaxStreamlinesFilter()
{
}

template <class TImageType, typename TScalarType, unsigned int NDimensions> 
RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::RelaxStreamlinesFilter()
{
  m_MaximumLength = 10;
  m_EpsilonConvergenceThreshold = 0.00001;
  m_MaximumNumberOfIterations = 200;
  m_InitializeBoundaries = false;
  m_L0Image = OutputImageType::New();
  m_L1Image = OutputImageType::New();
  
  niftkitkDebugMacro(<<"RelaxStreamlinesFilter():Constructed, " \
    << "m_EpsilonConvergenceThreshold=" << m_EpsilonConvergenceThreshold \
    << ", m_MaximumNumberOfIterations=" << m_MaximumNumberOfIterations \
    << ", m_InitializeBoundaries=" << m_InitializeBoundaries \
    );
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "EpsilonConvergenceThreshold:" << m_EpsilonConvergenceThreshold << std::endl;
  os << indent << "MaximumNumberOfIterations:" << m_MaximumNumberOfIterations << std::endl;
  os << indent << "MaximumLength:" << m_MaximumLength << std::endl;    
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::SolvePDE(
    bool isInnerBoundary,
    std::vector<InputScalarImageIndexType>& listOfGreyPixels,
    InputScalarImageType* scalarImage,
    InputVectorImageType* vectorImage,
    OutputImageType* outputImage
    )
{
  // [STEP 2] Use eqn (8) and (9) from paper to update L_0 and L_1.
  unsigned long int currentIteration = 0;
  unsigned long int pixelNumber = 0;
  unsigned long int totalNumberOfPixels = 0;
  unsigned int dimensionIndex = 0;
  unsigned int dimensionIndexForAnisotropicScaleFactors = 0;
  double epsilonRatio = 1;
  
  OutputImageIndexType index;
  OutputImageIndexType indexPlus;
  OutputImageIndexType indexMinus;
  OutputImagePixelType divisor;
  OutputImagePixelType multiplier;
  OutputImagePixelType initialValue;
  OutputImagePixelType componentMagnitude;
  OutputImagePixelType currentPixelEnergy;
  OutputImagePixelType currentFieldEnergy;
  OutputImagePixelType previousFieldEnergy = 0;
  OutputImagePixelType pixelPlus;
  OutputImagePixelType pixelMinus;  
  OutputImagePixelType value;
  OutputImageSpacingType spacing = scalarImage->GetSpacing();
  OutputImageSpacingType multipliers;
  InputVectorImagePixelType vectorPixel;
  
  // Pre-calculate this initial value.
  initialValue = 1;
  for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
    {
      initialValue *= spacing[dimensionIndex];
    }
  
  // Pre-calculate multipliers
  for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
    {
      multiplier = 1;
      for (dimensionIndexForAnisotropicScaleFactors = 0; dimensionIndexForAnisotropicScaleFactors < Dimension; dimensionIndexForAnisotropicScaleFactors++)
        {
          if (dimensionIndexForAnisotropicScaleFactors != dimensionIndex)
            {
              multiplier *= (spacing[dimensionIndexForAnisotropicScaleFactors]);
            }
        }
      multipliers[dimensionIndex] = multiplier;
    }
  
  totalNumberOfPixels = listOfGreyPixels.size();
  
  niftkitkDebugMacro(<<"GenerateData():initialValue=" << initialValue << ", multipliers=" << multipliers << ", listOfGreyPixels.size()=" << listOfGreyPixels.size());
  
  // Start of main loop
  niftkitkDebugMacro(<<"GenerateData():currentIteration=" << currentIteration \
      << ", m_MaximumNumberOfIterations=" << m_MaximumNumberOfIterations \
      << ", epsilonRatio=" << epsilonRatio \
      << ", m_EpsilonConvergenceThreshold=" << m_EpsilonConvergenceThreshold \
      );
  
  while (currentIteration < m_MaximumNumberOfIterations && epsilonRatio >= m_EpsilonConvergenceThreshold)
    {
      currentFieldEnergy = 0;
      
      for (pixelNumber = 0; pixelNumber < totalNumberOfPixels; pixelNumber++)
        {
          value = initialValue;
          divisor = 0;
          
          currentPixelEnergy = 0;

          index = listOfGreyPixels[pixelNumber];
          
          vectorPixel = vectorImage->GetPixel(index);
          
          for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
            {
              indexPlus = index;
              indexMinus = index;

              indexPlus[dimensionIndex] += 1;
              indexMinus[dimensionIndex] -= 1;  

              componentMagnitude = fabs(vectorPixel[dimensionIndex]);
              
              multiplier = (multipliers[dimensionIndex]* componentMagnitude);
              
              if (isInnerBoundary)
                {
                  // Note, the indexPlus and indexMinus
                  // are MEANT to be different between L1 and L0.    
                  
                  if (vectorPixel[dimensionIndex] >= 0)
                    {
                      pixelMinus = outputImage->GetPixel(indexMinus);
                      value += multiplier * pixelMinus; 
                      currentPixelEnergy += (pixelMinus * pixelMinus);
                    }
                  else
                    {
                      pixelPlus = outputImage->GetPixel(indexPlus);
                      value += multiplier * pixelPlus;
                      currentPixelEnergy += (pixelPlus * pixelPlus);
                    }
                   
                }
              else
                {
                  // Note, the indexPlus and indexMinus
                  // are MEANT to be different between L1 and L0.    
                  
                  if (vectorPixel[dimensionIndex] >= 0)
                    {
                      pixelPlus = outputImage->GetPixel(indexPlus);
                      value += multiplier * pixelPlus; 
                      currentPixelEnergy += (pixelPlus * pixelPlus); 
                    }
                  else
                    {
                      pixelMinus = outputImage->GetPixel(indexMinus);
                      value += multiplier * pixelMinus;
                      currentPixelEnergy += (pixelMinus * pixelMinus);
                    }
                  
                }
              
              // And dont forget the divisor
              divisor += multiplier;            
            }
            
          if (divisor != 0)
            {
              value /= divisor;
            }
          else
            {
              value = 0;
            }          
          outputImage->SetPixel(index, value);
          currentPixelEnergy = sqrt(currentPixelEnergy);
          currentFieldEnergy += currentPixelEnergy;
              
        }  

      if (currentIteration != 0)
        {
          epsilonRatio = fabs((previousFieldEnergy - currentFieldEnergy) / previousFieldEnergy);  
        }

      niftkitkInfoMacro(<<"GenerateData():[" << currentIteration << "] currentFieldEnergy=" << currentFieldEnergy << ", previousFieldEnergy=" << previousFieldEnergy << ", epsilonRatio=" << epsilonRatio << ", epsilonTolerance=" << m_EpsilonConvergenceThreshold);
      previousFieldEnergy = currentFieldEnergy;
                  
      currentIteration++;
             
    }    
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Started");
  
  int i;
  
  this->AllocateOutputs();
  
  InputScalarImagePointer scalarImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(0));
  InputVectorImagePointer vectorImage = static_cast< InputVectorImageType * >(this->ProcessObject::GetInput(1));
  InputScalarImagePointer segmentedImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(2));
  OutputImagePointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  typename OutputImageType::Pointer tmpImages[2];
  tmpImages[0] = m_L0Image;
  tmpImages[1] = m_L1Image;
  
  // Make sure m_L0Image and m_L1Image are the right size.
  if (m_L0Image->GetLargestPossibleRegion() != scalarImage->GetLargestPossibleRegion()
   || m_L1Image->GetLargestPossibleRegion() != scalarImage->GetLargestPossibleRegion()
   )
    {
      // Set temp images to the right size.
      for (i = 0; i < 2; i++)
        {
          tmpImages[i]->SetDirection(segmentedImage->GetDirection());
          tmpImages[i]->SetSpacing(segmentedImage->GetSpacing());
          tmpImages[i]->SetOrigin(segmentedImage->GetOrigin());
          tmpImages[i]->SetRegions(segmentedImage->GetLargestPossibleRegion());
          tmpImages[i]->Allocate();
          niftkitkDebugMacro(<<"GenerateData():Set tmp image[" << i << "] to size:" << tmpImages[i]->GetLargestPossibleRegion().GetSize());
        } 
    }

  // [STEP 1]. Whizz through image, setting L_0 = L_1 = 0 at all grid points,
  // and building up a list of indexes that are within grey matter.
  
  ImageRegionConstIteratorWithIndex<InputScalarImageType> 
    segmentedIterator(segmentedImage, segmentedImage->GetLargestPossibleRegion());
    
  ImageRegionIterator<OutputImageType> 
    outputIterator1(m_L0Image, m_L0Image->GetLargestPossibleRegion());
    
  ImageRegionIterator<OutputImageType> 
    outputIterator2(m_L1Image, m_L1Image->GetLargestPossibleRegion());
    
  std::vector<InputScalarImageIndexType> listOfGreyMatterPixels;                                                    
  listOfGreyMatterPixels.clear();

  InputScalarImagePixelType segmentedPixel;
  
  for (outputIterator1.GoToBegin(), 
       outputIterator2.GoToBegin(), 
       segmentedIterator.GoToBegin(); 
       !segmentedIterator.IsAtEnd(); 
       ++segmentedIterator, ++outputIterator2, ++outputIterator1)
    {
      outputIterator1.Set(0);
      outputIterator2.Set(0);
      segmentedPixel = segmentedIterator.Get();
      if (segmentedPixel == this->GetGreyMatterLabel() )
        {
          listOfGreyMatterPixels.push_back(segmentedIterator.GetIndex());  
        }
    }
  
  niftkitkDebugMacro(<<"GenerateData():Initially found " << listOfGreyMatterPixels.size() << " grey matter voxels");
  
  // Additional Step not in Yezzi 2003 paper, we may want to 
  // initialize the boundaries like Bourgeat ISBI 2007, or Diep 2007.
  // Subclasses should redefine this method, as its virtual.

  std::vector<InputScalarImageIndexType> L0greyList;
  L0greyList.clear();

  std::vector<InputScalarImageIndexType> L1greyList;
  L1greyList.clear();

  this->InitializeBoundaries(
    listOfGreyMatterPixels,
    scalarImage.GetPointer(),
    vectorImage.GetPointer(),
    m_L0Image.GetPointer(),
    m_L1Image.GetPointer(),
    L0greyList,
    L1greyList);
  
  niftkitkDebugMacro(<<"GenerateData():After InitializeBoundaries(), complete list of grey voxels has " << listOfGreyMatterPixels.size() << " voxels, whereas L0 list has " << L0greyList.size() << " and L1 list has " << L1greyList.size() << " that need solving");
  
  // Solve for L0 and L1 separately, as this allows us to have different regions for L0 and L1.
  this->SolvePDE(true, L0greyList, scalarImage.GetPointer(), vectorImage.GetPointer(), m_L0Image.GetPointer());
  this->SolvePDE(false, L1greyList, scalarImage.GetPointer(), vectorImage.GetPointer(), m_L1Image.GetPointer());
  
  // [Step 3] Repeat step 2 until values L0 and L1 converge. - Done
  // [Step 4] We need to add L0 and L1 to get the distance, and write it to output.
  
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());

  for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
    {
      outputIterator.Set(0);
    }
  
  OutputImageIndexType index;
  OutputImagePixelType outputValue;
  
  for (unsigned long int i = 0; i < listOfGreyMatterPixels.size(); i++)
    {
      index = listOfGreyMatterPixels[i];
      outputValue = m_L0Image->GetPixel(index) + m_L1Image->GetPixel(index);
      if (outputValue < 0) 
      {
        outputValue = 0;
      }
      else if (outputValue > m_MaximumLength)
      {
        outputValue = m_MaximumLength;
      }
      
      outputImage->SetPixel(index, outputValue);      
    }

  niftkitkDebugMacro(<<"GenerateData():Finished");
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::InitializeBoundaries(
    std::vector<InputScalarImageIndexType>& completeListOfGreyMatterPixels,
    InputScalarImageType* scalarImage,
    InputVectorImageType* vectorImage,
    OutputImageType* L0Image,
    OutputImageType* L1Image,
    std::vector<InputScalarImageIndexType>& L0greyList,
    std::vector<InputScalarImageIndexType>& L1greyList
    )
{
  
  if (m_InitializeBoundaries)
    {

      double initializer = 0;
      InputScalarImagePointer segmentedImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(2));
      OutputImageIndexType greyIndex;
      OutputImageIndexType indexOffset;
      
      for (unsigned int i = 0; i < NDimensions; i++)
        {
          initializer += scalarImage->GetSpacing()[i];
        }
      initializer = -1.0 * (initializer/(2.0*(double)NDimensions));
      
      niftkitkDebugMacro(<<"InitializeBoundaries():Initializing boundaries to " << initializer);
      
      for (unsigned int i = 0; i < completeListOfGreyMatterPixels.size(); i++)
        {
          greyIndex = completeListOfGreyMatterPixels[i];
          
          for (unsigned int j = 0; j < NDimensions; j++)
            {
              for (int k = -1; k <= 1; k += 2)
                {
                  indexOffset = greyIndex;
                  indexOffset[j] = indexOffset[j] + k;
                  
                  if (segmentedImage->GetPixel(indexOffset) == this->GetWhiteMatterLabel())
                    {
                      L0Image->SetPixel(indexOffset, initializer);
                    }
                  else if (segmentedImage->GetPixel(indexOffset) == this->GetExtraCerebralMatterLabel())
                    {
                      L1Image->SetPixel(indexOffset, initializer);
                    }

                } // end for k
            } // end for j
        } // end for i
    }
  else
    {
      niftkitkDebugMacro(<<"InitializeBoundaries():Not initializing boundaries, so they stay at zero.");
    }
  
  // In this method, we are just setting the boundary to the grey matter
  // (i.e. initializing the adjacent WM and CSF voxels), so
  // the number of pixels that need solving is always the same,
  // (i.e. the complete grey matter mask).
  L0greyList = completeListOfGreyMatterPixels;
  L1greyList = completeListOfGreyMatterPixels;
  
  niftkitkDebugMacro(<<"InitializeBoundaries():greyMatter.size()=" << completeListOfGreyMatterPixels.size() \
      << ", L0greyList.size()=" << L0greyList.size() \
      << ", L1greyList.size()=" << L1greyList.size() \
      );
}

} // end namespace

#endif // __itkImageRegistrationFilter_txx
