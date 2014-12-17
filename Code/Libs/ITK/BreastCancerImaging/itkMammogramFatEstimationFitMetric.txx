/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatEstimationFitMetric_txx
#define __itkMammogramFatEstimationFitMetric_txx


#include "itkMammogramFatEstimationFitMetric.h"

#include <itkWriteImage.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>

#include <vnl/vnl_math.h>

namespace itk
{


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

MammogramFatEstimationFitMetric
::MammogramFatEstimationFitMetric()
{
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

MammogramFatEstimationFitMetric
::~MammogramFatEstimationFitMetric()
{
}


/* -----------------------------------------------------------------------
   GetValue()
   ----------------------------------------------------------------------- */

MammogramFatEstimationFitMetric::MeasureType 
MammogramFatEstimationFitMetric
::GetValue( const ParametersType &parameters ) const
{
  itkExceptionMacro( << "ERROR: GetValue() is an abstract class and cannot be called directly." );
  return std::numeric_limits<double>::max();
}


/* -----------------------------------------------------------------------
   WriteIntensityVsEdgeDistToFile()
   ----------------------------------------------------------------------- */

void
MammogramFatEstimationFitMetric
::WriteIntensityVsEdgeDistToFile( std::string fileOutputIntensityVsEdgeDist )
{
  itkExceptionMacro( << "ERROR: WriteIntensityVsEdgeDistToFile() is an abstract class and cannot be called directly." );
}


/* -----------------------------------------------------------------------
   WriteFitToFile()
   ----------------------------------------------------------------------- */

void
MammogramFatEstimationFitMetric
::WriteFitToFile( std::string fileOutputFit,
                  const ParametersType &parameters )
{
  itkExceptionMacro( << "ERROR: WriteFitToFile() is an abstract class and cannot be called directly." );
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

void
MammogramFatEstimationFitMetric
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
