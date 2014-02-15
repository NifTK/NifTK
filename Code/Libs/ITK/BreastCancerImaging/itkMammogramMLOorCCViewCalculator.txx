/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramMLOorCCViewCalculator_txx
#define __itkMammogramMLOorCCViewCalculator_txx

#include "itkMammogramMLOorCCViewCalculator.h"

#include <itkNumericTraits.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageMomentsCalculator.h>

namespace itk
{

  static const size_t nDicomViewTags = 5;
  const std::string dicomViewTags[nDicomViewTags] = {
    "0018|5101",
    "0018|0015",
    "0018|1400",
    "07a1|1040",
    "0045|101B"
  };
    


// ---------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------

template< class TInputImage >
MammogramMLOorCCViewCalculator< TInputImage >
::MammogramMLOorCCViewCalculator()
{
  m_FlgVerbose = false;

  m_Score = 0.;

  m_Image = 0;

  m_MammogramView = UNKNOWN_MAMMO_VIEW;
}


/* -----------------------------------------------------------------------
   SetImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramMLOorCCViewCalculator<TInputImage>
::SetImage( const InputImageType *imInput )
{
  m_Image = const_cast< InputImageType *>( imInput );

  // Allocate the upper and lower images

  m_ImRegion  = m_Image->GetLargestPossibleRegion();
  m_ImSpacing = m_Image->GetSpacing();
  m_ImOrigin  = m_Image->GetOrigin();

  m_ImSize    = m_ImRegion.GetSize();
  m_ImStart   = m_ImRegion.GetIndex();

  m_ImSizeInMM[0] = m_ImSize[0]*m_ImSpacing[0];
  m_ImSizeInMM[1] = m_ImSize[1]*m_ImSpacing[1];

  this->Modified();
}


// ---------------------------------------------------------------------
// Compute the mammogram view
// ---------------------------------------------------------------------

template< class TInputImage >
void
MammogramMLOorCCViewCalculator< TInputImage >
::Compute(void) throw (ExceptionObject)
{
  unsigned int i;

  if ( ! m_Image )
  {
    itkExceptionMacro( << "ERROR: An input image to MammogramMLOorCCViewCalculator must be specified" );
  }

  
  // Check the dictionary for the view tag set to 'MLO' or 'CC'
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( i=0; i<nDicomViewTags; i++ )
  {
    
    std::string tagViewID = dicomViewTags[i];
    std::string tagViewValue;
    
    DictionaryType::ConstIterator tagItr = m_Dictionary.Find( tagViewID );
    DictionaryType::ConstIterator end    = m_Dictionary.End();
    
    if( tagItr != end )
    {
      MetaDataStringType::ConstPointer entryvalue = 
        dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
      
      if ( entryvalue )
      {
        tagViewValue = entryvalue->GetMetaDataObjectValue();
        
        if ( ( tagViewValue.find( "CC" ) != std::string::npos ) || 
             ( tagViewValue.find( "cc" ) != std::string::npos ) )
        {
          m_MammogramView = CC_MAMMO_VIEW;
          std::cout << "Mammogram is CC (from DICOM tag: " << tagViewID << ")" << std::endl;
          m_Score = 0;
          return;
        }
        
        else if ( ( tagViewValue.find( "MLO" ) != std::string::npos ) || 
                  ( tagViewValue.find( "mlo" ) != std::string::npos ) )
        {
          m_MammogramView = MLO_MAMMO_VIEW;
          std::cout << "Mammogram is MLO (from DICOM tag: " << tagViewID << ")" << std::endl;
          m_Score = 1;
          return;
        }
      }
    }
  }

  
  // If a file name has been specified then check whether this contains the strings 'MLO' or 'CC'
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( m_ImageFileName.length() )
  {

    // 'CC' found and 'MLO' not found

    if ( ( ( m_ImageFileName.find( "CC" )    != std::string::npos ) || 
           ( m_ImageFileName.find( "-cc-" )  != std::string::npos ) || 
           ( m_ImageFileName.find( ".cc." )  != std::string::npos ) || 
           ( m_ImageFileName.find( "_cc_" )  != std::string::npos ) )
         &&
         ( m_ImageFileName.find( "MLO" )   == std::string::npos ) && 
         ( m_ImageFileName.find( "-mlo-" ) == std::string::npos ) && 
         ( m_ImageFileName.find( ".mlo." ) == std::string::npos ) && 
         ( m_ImageFileName.find( "_mlo_" ) == std::string::npos ) )
    {
      m_MammogramView = CC_MAMMO_VIEW;
      std::cout << "Mammogram is CC (from file name: '" << m_ImageFileName << "')" << std::endl;
      m_Score = 0;
      return;
    }

    // 'MLO' found and 'CC' not found

    else if ( ( ( m_ImageFileName.find( "MLO" )   != std::string::npos ) || 
                ( m_ImageFileName.find( "-mlo-" ) != std::string::npos ) || 
                ( m_ImageFileName.find( ".mlo." ) != std::string::npos ) || 
                ( m_ImageFileName.find( "_mlo_" ) != std::string::npos ) )
              &&
              ( m_ImageFileName.find( "CC" )    == std::string::npos ) && 
              ( m_ImageFileName.find( "-cc-" )  == std::string::npos ) && 
              ( m_ImageFileName.find( ".cc." )  == std::string::npos ) && 
              ( m_ImageFileName.find( "_cc_" )  == std::string::npos ) )
    {
      m_MammogramView = MLO_MAMMO_VIEW;
      std::cout << "Mammogram is MLO (from file name: '" << m_ImageFileName << "')" << std::endl;
      m_Score = 1;
      return;
    }
  }

  
  // See if the image goes all the way to the top
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if 1

  InputImageRegionType region = m_Image->GetLargestPossibleRegion();

  InputImageSizeType size = region.GetSize();

  double nRows = size[1];
  double nRowToTest = size[1]/10;

  double cummulativeFraction = 0.;

  // Sum the image pixels

  IteratorType itImage( m_Image, region );

  double sumOfImageIntensities = 0.;

  for ( itImage.GoToBegin();
        ! itImage.IsAtEnd(); 
        ++itImage )
  {
    sumOfImageIntensities += itImage.Get();
  }

  std::cout << "Sum of image intensities: " << sumOfImageIntensities << std::endl;

  if ( sumOfImageIntensities == 0 )
  {
    std::cout << "ERROR: No pixels detected to calculate MLO or CC view" << std::endl;
    return;
  }

  // Sum the image intensities in the first few rows

  LineIteratorType itImageLinear( m_Image, region );

  itImageLinear.SetDirection( 0 );

  double iRow = 0;

  itImageLinear.GoToBegin();

  double sumOfInitialRowIntensities = 0.;

  while ( iRow < nRowToTest )
  {
    itImageLinear.GoToBeginOfLine();

    while ( ! itImageLinear.IsAtEndOfLine() )
    {
      sumOfInitialRowIntensities += itImageLinear.Get();
      ++itImageLinear;
    }

    itImageLinear.NextLine();
    iRow++;
  }
    
  sumOfInitialRowIntensities /= sumOfImageIntensities;

  // If sum is less than 5% of total then this should be a CC view

  if ( sumOfInitialRowIntensities < 0.053 )
  {
    m_MammogramView = CC_MAMMO_VIEW;
    std::cout << "Mammogram is CC (from image intensities)" << std::endl;
  }
  else
  {
    m_MammogramView = MLO_MAMMO_VIEW;
    std::cout << "Mammogram is MLO (from image intensities)" << std::endl;
  }

  std::cout << "Fraction of intensities in first rows: " 
            << 100.*sumOfInitialRowIntensities << "%" << std::endl;

  m_Score = sumOfInitialRowIntensities;


#else


  // Otherwise correlate the upper and lower regions of the image and detect a mismatch
  // (assumes that a CC mammogram will be well centered in the field of view).
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  // Create the upper and lower regions

  InputImageRegionType region = m_Image->GetLargestPossibleRegion();      
  InputImageSizeType size = region.GetSize();

  // Find the row with the largest sum of intensities
  // (it should correspond roughly to the nipple position)

  LineIteratorType itRegion( m_Image, region );

  itRegion.SetDirection( 0 );

  unsigned int iRow = 0;
  unsigned int iRowMax = 0;

  unsigned int nRows = size[1];
  unsigned int nRowsStart = size[1]/10;
  unsigned int nRowsEnd = 9*size[1]/10;

  double rowSum;
  double maxRowSum = 0.;

  itRegion.GoToBegin();
    
  while ( iRow < nRowsStart )
  {
    itRegion.NextLine();
    iRow++;
  }

  while ( iRow < nRowsEnd )
  {
    rowSum = 0.;

    itRegion.GoToBeginOfLine();
    while ( ! itRegion.IsAtEndOfLine() )
    {
      rowSum += itRegion.Get();
      ++itRegion;
    }

    if ( rowSum > maxRowSum )
    {
      maxRowSum = rowSum;
      iRowMax = iRow;
    }

    itRegion.NextLine();
    iRow++;
  }


  // Set the lower region

  InputImageRegionType regionLower;
  InputImageSizeType   sizeLower;

  regionLower  = region;

  sizeLower    = regionLower.GetSize();
  sizeLower[1] = iRowMax + 1;

  regionLower.SetSize( sizeLower );

  // Set the upper region

  InputImageRegionType regionUpper;
  InputImageIndexType  startUpper;
  InputImageSizeType   sizeUpper;

  regionUpper = region;

  startUpper    = regionUpper.GetIndex();
  startUpper[1] = iRowMax;

  sizeUpper     = regionUpper.GetSize();
  sizeUpper[1]  = sizeUpper[1] - iRowMax;

  regionUpper.SetIndex( startUpper );
  regionUpper.SetSize( sizeUpper );

  if ( this->GetDebug() )
  {
    std::cout << "Lower region: " << regionLower << std::endl
              << "Upper region: " << regionUpper << std::endl;
  }

  LineIteratorType itLowerRegion( m_Image, regionLower );
  LineIteratorType itUpperRegion( m_Image, regionUpper );

  itLowerRegion.SetDirection( 0 );
  itUpperRegion.SetDirection( 0 );


  // Iterate through the upper and lower regions calculating the mean

  double nPixels = 0.;
  double meanLower = 0.;
  double meanUpper = 0.;

  for (     itLowerRegion.GoToReverseBegin(),       itUpperRegion.GoToBegin();
            ( ! itLowerRegion.IsAtReverseEnd() ) && ( ! itUpperRegion.IsAtEnd() ); 
            itLowerRegion.PreviousLine(),           itUpperRegion.NextLine() )
  {
    itLowerRegion.GoToBeginOfLine();
    itUpperRegion.GoToBeginOfLine();

    while (    ( ! itLowerRegion.IsAtEndOfLine() ) 
               && ( ! itUpperRegion.IsAtEndOfLine() ) )
    {
      nPixels++;
      meanLower += itLowerRegion.Get();
      meanUpper += itUpperRegion.Get();
      
      ++itLowerRegion;
      ++itUpperRegion;
    }
  }

  for (     ;
            ( ! itLowerRegion.IsAtReverseEnd() ); 
            itLowerRegion.PreviousLine() )
  {
    itLowerRegion.GoToBeginOfLine();

    while ( ! itLowerRegion.IsAtEndOfLine() )
    {
      nPixels++;
      meanLower += itLowerRegion.Get();
      
      ++itLowerRegion;
    }
  }

  for (     ;
            ( ! itUpperRegion.IsAtEnd() ); 
            itUpperRegion.NextLine() )
  {
    itUpperRegion.GoToBeginOfLine();

    while ( ! itUpperRegion.IsAtEndOfLine() )
    {
      nPixels++;
      meanUpper += itUpperRegion.Get();

      ++itUpperRegion;
    }
  }

  meanLower /= nPixels;
  meanUpper /= nPixels;

  if ( this->GetDebug() )
  {
    std::cout << "No. of pixels: " << nPixels << std::endl
              << "Lower mean: " << meanLower << std::endl
              << "Upper mean: " << meanUpper << std::endl;
  }


  // Iterate through the upper and lower regions calculating the standard deviation

  double valueLower;
  double valueUpper;

  double stdDevLower = 0.;
  double stdDevUpper = 0.;

  for (     itLowerRegion.GoToReverseBegin(),       itUpperRegion.GoToBegin();
            ( ! itLowerRegion.IsAtReverseEnd() ) && ( ! itUpperRegion.IsAtEnd() ); 
            itLowerRegion.PreviousLine(),           itUpperRegion.NextLine() )
  {
    itLowerRegion.GoToBeginOfLine();
    itUpperRegion.GoToBeginOfLine();

    while (    ( ! itLowerRegion.IsAtEndOfLine() ) 
               && ( ! itUpperRegion.IsAtEndOfLine() ) )
    {
      valueLower = itLowerRegion.Get();
      stdDevLower += valueLower*valueLower;

      valueUpper = itUpperRegion.Get();
      stdDevUpper += valueUpper*valueUpper;
      
      ++itLowerRegion;
      ++itUpperRegion;
    }
  }

  for (     ;
            ( ! itLowerRegion.IsAtReverseEnd() ); 
            itLowerRegion.PreviousLine() )
  {
    itLowerRegion.GoToBeginOfLine();

    while ( ! itLowerRegion.IsAtEndOfLine() )
    {
      valueLower = itLowerRegion.Get();
      stdDevLower += valueLower*valueLower;
      
      ++itLowerRegion;
    }
  }

  for (     ;
            ( ! itUpperRegion.IsAtEnd() ); 
            itUpperRegion.NextLine() )
  {
    itUpperRegion.GoToBeginOfLine();

    while ( ! itUpperRegion.IsAtEndOfLine() )
    {
      valueUpper = itUpperRegion.Get();
      stdDevUpper += valueUpper*valueUpper;

      ++itUpperRegion;
    }
  }

  stdDevLower = sqrt( stdDevLower/nPixels );
  stdDevUpper = sqrt( stdDevUpper/nPixels );

  if ( this->GetDebug() )
  {
    std::cout << "Lower std. dev.: " << stdDevLower << std::endl
              << "Upper std. dev.: " << stdDevUpper << std::endl;
  }


  // Iterate through the upper and lower regions calculating the Score

  m_Score = 0.;

  for (     itLowerRegion.GoToReverseBegin(),       itUpperRegion.GoToBegin();
            ( ! itLowerRegion.IsAtReverseEnd() ) && ( ! itUpperRegion.IsAtEnd() ); 
            itLowerRegion.PreviousLine(),           itUpperRegion.NextLine() )
  {
    itLowerRegion.GoToBeginOfLine();
    itUpperRegion.GoToBeginOfLine();

    while (    ( ! itLowerRegion.IsAtEndOfLine() ) 
               && ( ! itUpperRegion.IsAtEndOfLine() ) )
    {
      m_Score += 
        ( static_cast<double>( itLowerRegion.Get() ) - meanLower )
        *( static_cast<double>( itUpperRegion.Get() ) - meanUpper );
      
      ++itLowerRegion;
      ++itUpperRegion;
    }
  }

  for (     ;
            ( ! itLowerRegion.IsAtReverseEnd() ); 
            itLowerRegion.PreviousLine() )
  {
    itLowerRegion.GoToBeginOfLine();

    while ( ! itLowerRegion.IsAtEndOfLine() )
    {
      m_Score += 
        ( static_cast<double>( itLowerRegion.Get() ) - meanLower )
        *(  - meanUpper );
      
      ++itLowerRegion;
    }
  }

  for (     ;
            ( ! itUpperRegion.IsAtEnd() ); 
            itUpperRegion.NextLine() )
  {
    itUpperRegion.GoToBeginOfLine();

    while ( ! itUpperRegion.IsAtEndOfLine() )
    {
      m_Score += 
        (  - meanLower )
        *( static_cast<double>( itUpperRegion.Get() ) - meanUpper );

      ++itUpperRegion;
    }
  }

  m_Score /= nPixels*stdDevLower*stdDevUpper;
  

  if ( this->GetDebug() )
  {
    std::cout << "Score: " << m_Score << std::endl;
  }

  if ( m_Score > 0.488 )
  {
    m_MammogramView = CC_MAMMO_VIEW;
    std::cout << "Mammogram is CC" << std::endl;
  }
  else
  {
    m_MammogramView = MLO_MAMMO_VIEW;
    std::cout << "Mammogram is MLO" << std::endl;
  }

#endif

}


// ---------------------------------------------------------------------
// PrintSelf()
// ---------------------------------------------------------------------

template< class TInputImage >
void
MammogramMLOorCCViewCalculator< TInputImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Mammogram view: " << m_MammogramView << std::endl;
  os << indent << "Image: " << std::endl;
  m_Image->Print( os, indent.GetNextIndent() );
}

} // end namespace itk

#endif // __itkMammogramMLOorCCViewCalculator_txx
