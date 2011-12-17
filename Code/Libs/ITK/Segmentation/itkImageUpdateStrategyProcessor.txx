/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 05:46:51 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7494 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkImageUpdateStrategyProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
ImageUpdateStrategyProcessor<TPixel, VImageDimension>
::ImageUpdateStrategyProcessor()
: m_Algorithm(0)
{
  m_Calculator = CalculatorType::New();
}

template<class TPixel, unsigned int VImageDimension>
void 
ImageUpdateStrategyProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  if (m_Algorithm.IsNotNull())
  {
    os << indent << "m_Algorithm=" << m_Algorithm << std::endl;
  }  
  else
  {
    os << indent << "m_Algorithm=NULL" << std::endl;
  }
  if (m_Calculator.IsNotNull())
  {
    os << indent << "m_Calculator=" << m_Calculator << std::endl;
  }  
  else
  {
    os << indent << "m_Calculator=NULL" << std::endl;
  }   
}

template<class TPixel, unsigned int VImageDimension>
void
ImageUpdateStrategyProcessor<TPixel, VImageDimension> 
::ApplyUpdateToAfterImage()
{
  RegionType regionOfInterest = this->GetDestinationRegionOfInterest();
  ImagePointer targetImage = this->GetAfterImage();
  
  if (targetImage.IsNull())
  {
    itkExceptionMacro(<< "Target image is NULL");
  }
  
  if (m_Algorithm.IsNull())
  {
    itkExceptionMacro(<< "Algorithm has not been set");
  }
  
  // The region of interest should match the target image
  // but in the general case, as long as it is smaller, we are ok.
  
  if (!targetImage->GetLargestPossibleRegion().IsInside(regionOfInterest))
  {
    itkExceptionMacro("Region of interest=\n" << regionOfInterest << ", is not inside target region=\n" << targetImage->GetLargestPossibleRegion() );
  }

  // Delegate to the algorithm and the algorithm decides what to do with it.
  ImagePointer outputImage = m_Algorithm->Execute(targetImage);
  this->SetAfterImage(outputImage);
}

} // end namespace
