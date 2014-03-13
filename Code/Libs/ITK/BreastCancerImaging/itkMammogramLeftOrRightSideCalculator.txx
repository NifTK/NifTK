/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramLeftOrRightSideCalculator_txx
#define __itkMammogramLeftOrRightSideCalculator_txx

#include "itkMammogramLeftOrRightSideCalculator.h"

#include <itkNumericTraits.h>
#include <itkMinimumMaximumImageCalculator.h>

namespace itk
{

// ---------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------

template< class TInputImage >
MammogramLeftOrRightSideCalculator< TInputImage >
::MammogramLeftOrRightSideCalculator()
{
  m_FlgVerbose = false;

  m_Image = 0;

  m_BreastSide = UNKNOWN_BREAST_SIDE;
}


// ---------------------------------------------------------------------
// Compute the breast side
// ---------------------------------------------------------------------

template< class TInputImage >
void
MammogramLeftOrRightSideCalculator< TInputImage >
::Compute(void) throw (ExceptionObject)
{
  unsigned int i;

  if ( ! m_Image )
  {
    itkExceptionMacro( << "ERROR: No input image to MammogramLeftOrRightSideCalculator specified" );
  }

  typename ImageType::RegionType region;
  typename ImageType::SizeType   size;
  typename ImageType::SizeType   scanSize;
  typename ImageType::IndexType  start;
  typename ImageType::IndexType  idx;

  region = m_Image->GetLargestPossibleRegion();      

  size = region.GetSize();

  // Determine if this is a left or right breast by calculating the CoM

  start[0] = size[0]/10;
  start[1] = 0;

  scanSize[0] = size[0]*8/10;
  scanSize[1] = size[1];

  region.SetSize(  scanSize  );
  region.SetIndex( start );

  std::cout << "Image size: " << size << std::endl;
  std::cout << "Region: " << region << std::endl;

  unsigned int iRow = 0;
  unsigned int nRows = 5;
  unsigned int rowSpacing = size[1]/( nRows + 1 );

  float xMoment = 0.;
  float xMomentSum = 0.;
  float intensitySum = 0.;

  LineIteratorType itLinear( m_Image, region );

  itLinear.SetDirection( 0 );

  while ( ! itLinear.IsAtEnd() )
  {
    // Skip initial set of rows

    iRow = 0;
    while ( ( ! itLinear.IsAtEnd() ) && ( iRow < rowSpacing ) )
    {
      iRow++;
      itLinear.NextLine();
    }

    // Add next row to moment calculation

    while ( ! itLinear.IsAtEndOfLine() )
    {
      idx = itLinear.GetIndex();

      intensitySum += itLinear.Get();

      xMoment = idx[0]*itLinear.Get();
      xMomentSum += xMoment;

      ++itLinear;
    }
  }

  xMoment = xMomentSum/intensitySum;

  std::cout << "Center of mass in x: " << xMoment << std::endl;


  if ( xMoment > static_cast<float>(size[0])/2. )
  {
    m_BreastSide = RIGHT_BREAST_SIDE;
    std::cout << "RIGHT breast" << std::endl;
  }
  else 
  {
    m_BreastSide = LEFT_BREAST_SIDE;
    std::cout << "LEFT breast" << std::endl;
  }
}


// ---------------------------------------------------------------------
// PrintSelf()
// ---------------------------------------------------------------------

template< class TInputImage >
void
MammogramLeftOrRightSideCalculator< TInputImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Breast side: " << m_BreastSide << std::endl;
  os << indent << "Image: " << std::endl;
  m_Image->Print( os, indent.GetNextIndent() );
}

} // end namespace itk

#endif // __itkMammogramLeftOrRightSideCalculator_txx
