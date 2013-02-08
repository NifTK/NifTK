/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKMULTIPLEERODEIMAGEFILTER_TXX_
#define ITKMULTIPLEERODEIMAGEFILTER_TXX_

#include "itkImageDuplicator.h"

namespace itk 
{

template <class TImageType>
MultipleErodeImageFilter<TImageType>
::MultipleErodeImageFilter()
{
  this->m_StructuringElement.SetRadius(1);
  this->m_StructuringElement.CreateStructuringElement();
  this->m_NumberOfErosions = 1;
  this->m_ErodeImageFilter = ErodeImageFilterType::New();
  this->m_ErodeImageFilter->SetKernel(this->m_StructuringElement);
  this->m_ErodeImageFilter->SetBoundaryToForeground(false);
  this->m_ErodeValue = 1;
  this->m_BackgroundValue = 0;
}

template <class TImageType>
void
MultipleErodeImageFilter<TImageType>
::GenerateData()
{
  this->m_ErodeImageFilter->SetErodeValue(this->m_ErodeValue);
  this->m_ErodeImageFilter->SetBackgroundValue(this->m_BackgroundValue);
  
  typedef ImageDuplicator<TImageType> DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  
  // Get a copy of the input image. 
  duplicator->SetInputImage(this->GetInput());
  duplicator->Update();
  this->m_ErodedImage = duplicator->GetOutput();
  this->m_ErodedImage->DisconnectPipeline();
  
  // Erode it repeatly. 
  for (unsigned int erosionCount = 0; erosionCount < this->m_NumberOfErosions; erosionCount++)
  {
    this->m_ErodeImageFilter->SetInput(this->m_ErodedImage);
    this->m_ErodeImageFilter->Update();
    this->m_ErodedImage = this->m_ErodeImageFilter->GetOutput();
    this->m_ErodedImage->DisconnectPipeline();
  }
  
  // Set it to be the output.  
  this->GraftOutput(this->m_ErodedImage);
}

}

#endif /*ITKMULTIPLEERODEIMAGEFILTER_TXX_*/
