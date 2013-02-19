/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKMULTIPLEDILATEIMAGEFILTER_TXX_
#define ITKMULTIPLEDILATEIMAGEFILTER_TXX_

#include "itkImageDuplicator.h"

namespace itk 
{

template <class TImageType>
MultipleDilateImageFilter<TImageType>
::MultipleDilateImageFilter()
{
  this->m_StructuringElement.SetRadius(1);
  this->m_StructuringElement.CreateStructuringElement();
  this->m_NumberOfDilations = 1;
  this->m_DilateImageFilter = DilateImageFilterType::New();
  this->m_DilateImageFilter->SetKernel(this->m_StructuringElement);
  this->m_DilateImageFilter->SetBoundaryToForeground(false);
  this->m_DilateValue = 1;
  this->m_BackgroundValue = 0;
}

template <class TImageType>
void
MultipleDilateImageFilter<TImageType>
::GenerateData()
{
  this->m_DilateImageFilter->SetDilateValue(this->m_DilateValue);
  this->m_DilateImageFilter->SetBackgroundValue(this->m_BackgroundValue);
  
  typedef ImageDuplicator<TImageType> DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  
  // Get a copy of the input image.
  duplicator->SetInputImage(this->GetInput());
  duplicator->Update();
  this->m_DilatedImage = duplicator->GetOutput();
  this->m_DilatedImage->DisconnectPipeline();
  
  // Dilate it repeatly. 
  for (unsigned int dilationCount = 0; dilationCount < this->m_NumberOfDilations; dilationCount++)
  {
    this->m_DilateImageFilter->SetInput(this->m_DilatedImage);
    this->m_DilateImageFilter->Update();
    this->m_DilatedImage = this->m_DilateImageFilter->GetOutput();
    this->m_DilatedImage->DisconnectPipeline();
  }
  
  // Set it to be the output.  
  this->GraftOutput(this->m_DilatedImage);
}


}

#endif /*ITKMULTIPLEDILATEIMAGEFILTER_TXX_*/
