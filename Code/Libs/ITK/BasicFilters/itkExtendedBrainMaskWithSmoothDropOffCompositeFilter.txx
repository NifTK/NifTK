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
#ifndef __itkExtendedBrainMaskWithSmoothDropOffCompositeFilter_txx
#define __itkExtendedBrainMaskWithSmoothDropOffCompositeFilter_txx

#include "itkExtendedBrainMaskWithSmoothDropOffCompositeFilter.h"

#include "itkLogHelper.h"

namespace itk
{
template <typename TImageType>
ExtendedBrainMaskWithSmoothDropOffCompositeFilter<TImageType>
::ExtendedBrainMaskWithSmoothDropOffCompositeFilter()
{
  // Initialise these parameters to reasonable values.
  SetFirstNumberOfDilations(0);
  SetSecondNumberOfDilations(0);
  SetGaussianFWHM(1);
  SetInitialThreshold(1);
  
  // We create the filters here, as the composite filter "owns" them.
  this->m_ThresholdFilter = BinaryThresholdFilterType::New();
  this->m_FirstDilateFilter = BinaryDilateFilterType::New();
  this->m_SecondDilateFilter = BinaryDilateFilterType::New();
  this->m_GaussianFilter = GaussianFilterType::New();
  this->m_InjectorFilter = InjectSourceImageGreaterThanZeroIntoTargetImageFilterType::New();
}

template <typename TImageType>
void 
ExtendedBrainMaskWithSmoothDropOffCompositeFilter<TImageType>
::GenerateData()
{
  niftkitkDebugMacro(<<"Started");

  this->m_ThresholdFilter->SetInput(this->GetInput());
  this->m_ThresholdFilter->SetLowerThreshold(m_InitialThreshold);
  this->m_ThresholdFilter->SetUpperThreshold(std::numeric_limits<double>::max());
  this->m_ThresholdFilter->SetInsideValue(1);
  this->m_ThresholdFilter->SetOutsideValue(0);
  this->m_ThresholdFilter->Update();
  
  StructuringElementType element;
  element.SetRadius(1);
  element.CreateStructuringElement();
  
  this->m_FirstDilateFilter->SetInput(this->m_ThresholdFilter->GetOutput());
  this->m_FirstDilateFilter->SetKernel(element);
  this->m_FirstDilateFilter->SetDilateValue(1);
  this->m_FirstDilateFilter->SetBackgroundValue(0);
  this->m_FirstDilateFilter->SetBoundaryToForeground(false);

  if (m_FirstNumberOfDilations > 1)
    {
      for (unsigned int i = 0; i < m_FirstNumberOfDilations - 1; i++)
        {
          this->m_FirstDilateFilter->Update();
          ImagePointer image = this->m_FirstDilateFilter->GetOutput();
          image->DisconnectPipeline();
          this->m_FirstDilateFilter->SetInput(image);
        }
      this->m_FirstDilateFilter->Update();
    }
  else if (this->m_FirstNumberOfDilations == 1)
    {
      this->m_FirstDilateFilter->Update();  
    }
  
  if (this->m_FirstNumberOfDilations > 0)
    {
      this->m_SecondDilateFilter->SetInput(this->m_FirstDilateFilter->GetOutput());    
    }
  else
    {
      this->m_SecondDilateFilter->SetInput(this->m_ThresholdFilter->GetOutput());
    }
  
  this->m_SecondDilateFilter->SetKernel(element);
  this->m_SecondDilateFilter->SetDilateValue(1);
  this->m_SecondDilateFilter->SetBackgroundValue(0);
  this->m_SecondDilateFilter->SetBoundaryToForeground(false);

  if (this->m_SecondNumberOfDilations > 1)
    {
      for (unsigned int i = 0; i < m_SecondNumberOfDilations - 1; i++)
        {
          this->m_SecondDilateFilter->Update();
          ImagePointer image = this->m_SecondDilateFilter->GetOutput();
          image->DisconnectPipeline();
          this->m_SecondDilateFilter->SetInput(image);
        }
      this->m_SecondDilateFilter->Update();
    }
  else if (this->m_SecondNumberOfDilations == 1)
    {
      this->m_SecondDilateFilter->Update();  
    }
  
  if (this->m_SecondNumberOfDilations > 0)
    {
      this->m_GaussianFilter->SetInput(this->m_SecondDilateFilter->GetOutput());  
    }
  else
    {
      if (this->m_FirstNumberOfDilations > 0)
        {
          this->m_GaussianFilter->SetInput(this->m_FirstDilateFilter->GetOutput());      
        }
      else
        {
          this->m_GaussianFilter->SetInput(this->m_ThresholdFilter->GetOutput());
        }
    }

  typename ImageType::SpacingType inputSpacing = this->GetInput()->GetSpacing();
  
  // http://mathworld.wolfram.com/GaussianFunction.html
  double variance = ((m_GaussianFWHM * m_GaussianFWHM) / (8.0 * log(2.0)));

  // http://mathworld.wolfram.com/GaussianFunction.html
  double stdDev = m_GaussianFWHM / (2.0 * sqrt(2.0 * log(2.0)));
  
  int kernelWidth = (int)niftk::Round(4.0 * stdDev / inputSpacing[0]);
  
  this->m_GaussianFilter->SetVariance(variance);
  this->m_GaussianFilter->SetMaximumKernelWidth(kernelWidth);
  this->m_GaussianFilter->Update();
  
  if (this->m_FirstNumberOfDilations > 0)
    {
      this->m_InjectorFilter->SetInput1(this->m_FirstDilateFilter->GetOutput());
    }
  else
    {
      this->m_InjectorFilter->SetInput1(this->m_ThresholdFilter->GetOutput());    
    }
  this->m_InjectorFilter->SetInput2(this->m_GaussianFilter->GetOutput());
  this->m_InjectorFilter->Update();
  
  // Get output
  this->GraftOutput(this->m_InjectorFilter->GetOutput());

  niftkitkDebugMacro(<<"Finished");
}

template <typename TImageType>
void 
ExtendedBrainMaskWithSmoothDropOffCompositeFilter<TImageType>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "InitialThreshold:" << this->m_InitialThreshold << std::endl;
  os << indent << "FirstNumberOfDilations:" << this->m_FirstNumberOfDilations << std::endl;
  os << indent << "SecondNumberOfDilations:" << this->m_SecondNumberOfDilations << std::endl;
  os << indent << "GaussianFWHM:" << this->m_GaussianFWHM << std::endl;
}

} /** End namespace. */

#endif
