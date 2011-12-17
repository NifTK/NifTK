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
#ifndef __itkJonesThicknessFilter_txx
#define __itkJonesThicknessFilter_txx

#include "itkJonesThicknessFilter.h"

namespace itk
{

template <typename TImageType, typename TScalarType, unsigned int NDimensions>
JonesThicknessFilter<TImageType, TScalarType, NDimensions>
::JonesThicknessFilter()
{
  m_CheckFilter = CheckFilterType::New();
  m_LaplaceFilter = LaplaceFilterType::New();
  m_NormalsFilter = NormalsFilterType::New(); 
  m_IntegrateFilter = IntegrateFilterType::New();
  m_NormalsOverrideImage = VectorNormalImageType::New();
  m_NormalsOverrideImage = NULL;
  
  m_LowVoltage = 0;
  m_HighVoltage = 10000;
  m_LaplaceEpsionRatio = 0.00001;
  m_LaplaceMaxIterations = 200;
  m_WhiteMatterLabel = 1;
  m_GreyMatterLabel = 2;
  m_CSFMatterLabel = 3;
  m_MinimumStepSize = 0.1;
  m_MaximumLength = 10;
  m_Sigma = 0;
  m_DontUseGaussSiedel = false;
  m_UseLabels = false;
  m_UseSmoothing = false;

  niftkitkDebugMacro(<<"JonesThicknessFilter():Constructed");
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions>
void
JonesThicknessFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "LowVoltage = " << m_LowVoltage << std::endl;
  os << indent << "HighVoltage = " << m_HighVoltage << std::endl;
  os << indent << "LaplaceEpsionRatio = " << m_LaplaceEpsionRatio << std::endl;  
  os << indent << "LaplaceMaxIterations = " << m_LaplaceMaxIterations << std::endl;
  os << indent << "WhiteMatterLabel = " << m_WhiteMatterLabel << std::endl;
  os << indent << "GreyMatterLabel = " << m_GreyMatterLabel << std::endl;
  os << indent << "CSFMatterLabel = " << m_CSFMatterLabel << std::endl;
  os << indent << "MinimumStepSize = " << m_MinimumStepSize << std::endl;
  os << indent << "MaximumLength = " << m_MaximumLength << std::endl;
  os << indent << "Sigma = " << m_Sigma << std::endl;
  os << indent << "DontUseGaussSiedel = " << m_DontUseGaussSiedel << std::endl;
  os << indent << "UseLabels = " << m_UseLabels << std::endl;
  os << indent << "UseSmoothing = " << m_UseSmoothing << std::endl;
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions>
void 
JonesThicknessFilter<TImageType, TScalarType, NDimensions>
::SetVectorNormalsOverrideImage(VectorNormalImageType* v)
{
  m_NormalsOverrideImage = v;
  niftkitkDebugMacro(<<"SetVectorNormalsOverrideImage(" << v << ")");
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions>
typename JonesThicknessFilter<TImageType, TScalarType, NDimensions>::VectorNormalImageType* 
JonesThicknessFilter<TImageType, TScalarType, NDimensions>
::GetVectorNormalsOverrideImage()
{
  return m_NormalsOverrideImage.GetPointer();   
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions>
typename JonesThicknessFilter<TImageType, TScalarType, NDimensions>::VectorNormalImageType* 
JonesThicknessFilter<TImageType, TScalarType, NDimensions>
::GetVectorNormalsFilterImage()
{
  return m_NormalsFilter->GetOutput();   
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions>
typename JonesThicknessFilter<TImageType, TScalarType, NDimensions>::LaplacianImageType* 
JonesThicknessFilter<TImageType, TScalarType, NDimensions>
::GetLaplacianFilterImage()
{
  return m_LaplaceFilter->GetOutput();   
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions>
void
JonesThicknessFilter<TImageType, TScalarType, NDimensions>
::GenerateData()
{
  niftkitkDebugMacro(<<"Started");
  
  typename TImageType::Pointer inputImage = static_cast< TImageType * >(this->ProcessObject::GetInput(0));
  
  m_CheckFilter->SetInput(inputImage);
  m_CheckFilter->SetLabelThresholds(m_GreyMatterLabel, m_WhiteMatterLabel, m_CSFMatterLabel); 
  
  m_LaplaceFilter->SetInput(m_CheckFilter->GetOutput());
  m_LaplaceFilter->SetLowVoltage(m_LowVoltage);
  m_LaplaceFilter->SetHighVoltage(m_HighVoltage);
  m_LaplaceFilter->SetMaximumNumberOfIterations(m_LaplaceMaxIterations);
  m_LaplaceFilter->SetEpsilonConvergenceThreshold(m_LaplaceEpsionRatio);
  m_LaplaceFilter->SetLabelThresholds(m_GreyMatterLabel, m_WhiteMatterLabel, m_CSFMatterLabel); 
  m_LaplaceFilter->SetUseGaussSeidel(!m_DontUseGaussSiedel);
  m_LaplaceFilter->UpdateLargestPossibleRegion();
  
  m_NormalsFilter->SetInput(m_LaplaceFilter->GetOutput());
  m_NormalsFilter->SetUseMillimetreScaling(true);
  m_NormalsFilter->SetDivideByTwo(true);
  m_NormalsFilter->SetNormalize(true);
  m_NormalsFilter->SetDerivativeType(NormalsFilterType::DERIVATIVE_OF_GAUSSIAN);
  if (m_UseSmoothing)
    {
      m_NormalsFilter->SetSigma(m_Sigma);
    }  
  else
    {
      m_NormalsFilter->SetSigma(0);
    }
  
  VectorNormalImagePointer vectorImage = m_NormalsFilter->GetOutput();
  if (m_NormalsOverrideImage.IsNotNull())
    {
	  niftkitkDebugMacro(<<"GenerateData():Using vector normals override image.");
      vectorImage = m_NormalsOverrideImage;
    }
  
  if (m_UseLabels)
    {
      double averageWMGM = (m_WhiteMatterLabel + m_GreyMatterLabel)/2.0;
      double averageGMCSF = (m_GreyMatterLabel + m_CSFMatterLabel)/2.0;
      
      niftkitkDebugMacro(<<"GenerateData():Integrating towards thresholds WM/GM=" << averageWMGM \
          << ", and GM/CSF=" << averageGMCSF);
      
      m_IntegrateFilter->SetScalarImage(m_CheckFilter->GetOutput());
      m_IntegrateFilter->SetLowVoltage(m_WhiteMatterLabel);
      m_IntegrateFilter->SetMinIterationVoltage(averageWMGM);
      m_IntegrateFilter->SetHighVoltage(m_CSFMatterLabel);  
      m_IntegrateFilter->SetMaxIterationVoltage(averageGMCSF);
    }
  else
    {
      m_IntegrateFilter->SetScalarImage(m_LaplaceFilter->GetOutput());
      m_IntegrateFilter->SetLowVoltage(m_LowVoltage);
      m_IntegrateFilter->SetMinIterationVoltage(m_LowVoltage);
      m_IntegrateFilter->SetHighVoltage(m_HighVoltage);
      m_IntegrateFilter->SetMaxIterationVoltage(m_HighVoltage);
    }  
  
  m_IntegrateFilter->SetVectorImage(vectorImage);  
  m_IntegrateFilter->SetStepSize(m_MinimumStepSize);
  m_IntegrateFilter->SetMaxIterationLength(m_MaximumLength);
  m_IntegrateFilter->UpdateLargestPossibleRegion();
  
  // Get output
  this->GraftOutput(this->m_IntegrateFilter->GetOutput());

  niftkitkDebugMacro(<<"Finished");
  
}

} // end namespace

#endif
