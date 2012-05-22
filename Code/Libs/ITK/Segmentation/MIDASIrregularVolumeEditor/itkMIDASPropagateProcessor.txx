/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7494 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkMIDASPropagateProcessor.h"

namespace itk
{

template<class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
MIDASPropagateProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension>
::MIDASPropagateProcessor()
: m_Strategy(NULL)
, m_Algorithm(NULL)
{
  this->m_Algorithm = AlgorithmType::New();
  this->m_Strategy = StrategyProcessorType::New();
  this->m_Processor = this->m_Strategy;
  
  // Strategy pattern: Inject algorithm into strategy class.
  this->m_Strategy->SetAlgorithm(m_Algorithm);
}

template<class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
void 
MIDASPropagateProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  if (m_Strategy.IsNotNull()) 
  {
    os << indent << "m_Strategy=" << m_Algorithm << std::endl;
  }
  else
  {
    os << indent << "m_Strategy=NULL" << std::endl;
  }  
  if (m_Algorithm.IsNotNull()) 
  {
    os << indent << "m_Algorithm=" << m_Algorithm << std::endl;
  }
  else
  {
    os << indent << "m_Algorithm=NULL" << std::endl;
  }  
}

} // end namespace
