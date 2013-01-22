/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkBreastMaskSegmForModelling.h"


namespace itk
{

// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
BreastMaskSegmForModelling< ImageDimension, PixelType >
::BreastMaskSegmForModelling()
{

};


// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
BreastMaskSegmForModelling< ImageDimension, PixelType >
::~BreastMaskSegmForModelling()
{

};


// --------------------------------------------------------------------------
// Execute()
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
void
BreastMaskSegmForModelling< ImageDimension, PixelType >
::Execute( void )
{
  // Initialise the segmentation
  this->Initialise();
  this->SmoothTheInputImages();

  // Calculate the Maximum Image
  this->CalculateTheMaximumImage();

  // Segment the backgound using the maximum image histogram
  this->SegmentBackground();

  // Find the nipple and mid-sternum landmarks
  this->FindBreastLandmarks();

  // Segment the Pectoral Muscle
  InternalImageType::SizeType 
    maxSize = imStructural->GetLargestPossibleRegion().GetSize();

  PointSetType::Pointer pecPointSet = 
    this->SegmentThePectoralMuscle( static_cast<RealType>( maxSize[1] ) );

  MaskThePectoralMuscleOnly( pecPointSet );

  // Discard anything not within a fitted surface (switch -cropfit)
  if ( bCropWithFittedSurface )
    MaskWithBSplineBreastSurface();

  // OR: for prone-supine scheme: clip at a distance of 40mm 
  //     posterior to the mid sternum point
  else
    MaskAtAtFixedDistancePosteriorToMidSternum();
  
  // Finally smooth the mask and threshold to round corners etc.
  SmoothMask();

}


// --------------------------------------------------------------------------
// Mask the pectoral muscle using a B-Spline surface
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
void
BreastMaskSegmForModelling< ImageDimension, PixelType >
::MaskThePectoralMuscleOnly( PointSetType::Pointer &pecPointSet )
{


  // Fit the B-Spline surface to the pectoral surface
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InternalImageType::Pointer imFittedPectoralis;

  // We require smaller kernel support for the prone-supine case 

  imFittedPectoralis = 
    MaskImageFromBSplineFittedSurface( pecPointSet, 
				       imStructural->GetLargestPossibleRegion(), 
				       imStructural->GetOrigin(), 
				       imStructural->GetSpacing(), 
				       imStructural->GetDirection(), 
				       rYHeightOffset,
				       3, 8, 3 );

  // Write the fitted surface to file

  WriteImageToFile( fileOutputPectoralSurfaceMask, 
                    "fitted pectoral surface with offset", 
                    imFittedPectoralis, flgLeft, flgRight );

  // Write the chest surface points to a file?

  WriteBinaryImageToUCharFile( fileOutputChestPoints, 
                               "chest surface points", 
                               imChestSurfaceVoxels, flgLeft, flgRight );


  // Discard anything within the pectoral mask (i.e. below the surface fit)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  IteratorType itSeg    = IteratorType( imSegmented,        
					imStructural->GetLargestPossibleRegion() );

  IteratorType itFitPec = IteratorType( imFittedPectoralis, 
					imStructural->GetLargestPossibleRegion() );

  if ( flgVerbose ) 
    std::cout << "Discarding segmentation posterior to pectoralis mask. " 
	      << std::endl;

  for ( itSeg.GoToBegin(), itFitPec.GoToBegin(); 
        ( ! itSeg.IsAtEnd() ) && ( ! itFitPec.IsAtEnd() ) ; 
        ++itSeg, ++itFitPec )
  {
    if ( itSeg.Get() )
      if ( itFitPec.Get() )
        itSeg.Set( 0 );
  }
}

 


} // namespace itk
