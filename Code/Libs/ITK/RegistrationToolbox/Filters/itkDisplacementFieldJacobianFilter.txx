/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDisplacementFieldJacobianFilter_txx
#define __itkDisplacementFieldJacobianFilter_txx

#include "itkDisplacementFieldJacobianFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"

namespace itk {

  
template <class InputScalarType, class OutputScalarType, unsigned int NDimensions>  
void
DisplacementFieldJacobianFilter<InputScalarType, OutputScalarType, NDimensions>
::GenerateData()
{
  this->AllocateOutputs();
  
  // Input image is the displacement field. 
  typename InputImageType::Pointer displacementField = dynamic_cast<InputImageType*>(this->ProcessObject::GetInput(0));
  // Output image is the Jacobian matrix. 
  typename OutputImageType::Pointer jacobianImage = dynamic_cast<OutputImageType*>(this->ProcessObject::GetOutput(0));
  
  this->m_Determinant = OutputDeterminantImageType::New(); 
  this->m_Determinant->SetRegions(displacementField->GetRequestedRegion());
  this->m_Determinant->SetOrigin(displacementField->GetOrigin());
  this->m_Determinant->SetDirection(displacementField->GetDirection());
  this->m_Determinant->SetSpacing(displacementField->GetSpacing());
  this->m_Determinant->Allocate();
  this->m_Determinant->Update(); 
  
  // Prepare the neighbourhood iterator for the numerical differential in the Jacobian calculation. 
  typedef ConstNeighborhoodIterator<InputImageType> NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  NeighborhoodIteratorType displacementFieldIterator(radius, displacementField, displacementField->GetRequestedRegion()); 
  
  // Prepare the iterator for filling the Jacobian and determinant image. 
  typedef ImageRegionIterator<OutputImageType> OuputImageIteratorType; 
  OuputImageIteratorType jacobianImageIterator(jacobianImage, jacobianImage->GetRequestedRegion()); 
  typedef ImageRegionIterator<OutputDeterminantImageType> OuputDeterminantImageIteratorType; 
  OuputDeterminantImageIteratorType jacobianDeterminantImageIterator(this->m_Determinant, displacementField->GetRequestedRegion()); 
  
  for (displacementFieldIterator.GoToBegin(), jacobianImageIterator.GoToBegin(), jacobianDeterminantImageIterator.GoToBegin(); 
       !displacementFieldIterator.IsAtEnd();       
       ++displacementFieldIterator, ++jacobianImageIterator, ++jacobianDeterminantImageIterator)
  {
    OutputPixelType jacobian; 
            
    // Calculate the Jacobian matrix. 
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      typename NeighborhoodIteratorType::OffsetType minusOffset;
      typename NeighborhoodIteratorType::OffsetType plusOffset;
        
      minusOffset.Fill(0);
      plusOffset.Fill(0);
      minusOffset[i] += -1;
      plusOffset[i] += 1;
        
      // Numerical differentiation along the i-th direction. This corresponds to the i-th column vector in the Jacobian matrix. 
      typename InputImageType::PixelType derivative = displacementFieldIterator.GetPixel(plusOffset)-displacementFieldIterator.GetPixel(minusOffset); 
      for (unsigned int j = 0; j < NDimensions; j++)
      {
        // Since this is a displacement field, we need to add 1 to the diagonal elements, 
        // because the Jacobian matrix of a zero displacement field (identify transform) is an identify matrix. 
        if (i == j) 
          jacobian(j, i) = derivative[j]/2.+1.; 
        else
          jacobian(j, i) = derivative[j]/2.; 
      }
    }
    jacobianImageIterator.Set(jacobian); 
    
    float jacobianDet = vnl_determinant(jacobian.GetVnlMatrix(), false); 
    jacobianDeterminantImageIterator.Set(jacobianDet); 
  }
}

  
  
  
  
  
  
  
  
}



#endif


















