/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkHessianToRaundahlStripinessMeasureImageFilter_txx
#define __itkHessianToRaundahlStripinessMeasureImageFilter_txx

#include "itkHessianToRaundahlStripinessMeasureImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "vnl/vnl_math.h"
#include "itkUCLMacro.h"

namespace itk
{

/**
 * Constructor
 */
template < typename TInputImage, typename TOutputImage > 
HessianToRaundahlStripinessMeasureImageFilter< TInputImage, TOutputImage>
::HessianToRaundahlStripinessMeasureImageFilter()
{
  m_SymmetricEigenValueFilter = EigenAnalysisFilterType::New();
  m_SymmetricEigenValueFilter->SetDimension( ImageDimension );
  m_SymmetricEigenValueFilter->OrderEigenValuesBy(EigenAnalysisFilterType::FunctorType::OrderByValue);

  m_ScaleStripinessMeasure = false; 

  // by default extract bright lines (equivalent to vesselness)
  m_BrightObject = true;
}

template < typename TInputImage, typename TOutputImage > 
void
HessianToRaundahlStripinessMeasureImageFilter< TInputImage, TOutputImage>
::GenerateData()
{
  niftkitkDebugMacro("HessianToRaundahlStripinessMeasureImageFilter generating data ");

  m_SymmetricEigenValueFilter->SetInput( this->GetInput() );
  
  typename OutputImageType::Pointer output = this->GetOutput();

  m_SymmetricEigenValueFilter->Update();
  
  const typename EigenValueImageType::ConstPointer eigenImage = m_SymmetricEigenValueFilter->GetOutput();
  
  // walk the region of eigen values and get the stripiness measure
  EigenValueArrayType eigenValues;

  ImageRegionConstIterator<EigenValueImageType> it;
  it = ImageRegionConstIterator<EigenValueImageType>(eigenImage, eigenImage->GetRequestedRegion());

  ImageRegionIterator<OutputImageType> oit;

  this->AllocateOutputs();

  oit = ImageRegionIterator<OutputImageType>(output,output->GetRequestedRegion());
  oit.GoToBegin();

  it.GoToBegin();

  while (!it.IsAtEnd())
    {
    // Get the eigenvalues
    eigenValues = it.Get();

    // Sort the eigenvalues by magnitude but retain their sign
    EigenValueArrayType sortedEigenValues = eigenValues;
    bool done = false;
    while (!done)
      {
      done = true;
      for (unsigned int i=0; i<ImageDimension-1; i++)
        {
        if (vnl_math_abs(sortedEigenValues[i]) > vnl_math_abs(sortedEigenValues[i+1]))
          {
          EigenValueType temp = sortedEigenValues[i+1];
          sortedEigenValues[i+1] = sortedEigenValues[i];
          sortedEigenValues[i] = temp;
          done = false;
          }
        }
      }

    EigenValueArrayType sortedAbsEigenValues;
    for (unsigned int i=0; i<ImageDimension; i++)
      {
      sortedAbsEigenValues[i] = vnl_math_abs(sortedEigenValues[i]);
      }

    /* compute stripiness from eigenvalue ratios:

                  |e2| - |e1|
         qs = ------------------
              |e1| + |e2| + 1e-5

    */

    double stripinessMeasure 
      = ( sortedAbsEigenValues[1] - sortedAbsEigenValues[0] )
      / ( sortedAbsEigenValues[0] + sortedAbsEigenValues[1] + 1e-5 );

    //std::cout << it.GetIndex() << ": " << stripinessMeasure << ", " <<  sortedAbsEigenValues[0] << ", " << sortedAbsEigenValues[1] << std::endl;

    // in case, scale by largest absolute eigenvalue
    if (m_ScaleStripinessMeasure)
      {
      stripinessMeasure *= sortedAbsEigenValues[ImageDimension-1];
      }

    oit.Set( static_cast< OutputPixelType >(stripinessMeasure));
    
    ++it;
    ++oit;
    }
}

template < typename TInputImage, typename TOutputImage > 
void
HessianToRaundahlStripinessMeasureImageFilter< TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  
  os << indent << "ScaleStripinessMeasure: " << m_ScaleStripinessMeasure << std::endl;
  os << indent << "BrightObject: " << m_BrightObject << std::endl;
}


} // end namespace itk
  
#endif
