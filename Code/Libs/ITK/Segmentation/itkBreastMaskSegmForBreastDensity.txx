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

#include "itkBreastMaskSegmForBreastDensity.h"


namespace itk
{

// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
BreastMaskSegmForBreastDensity< ImageDimension, PixelType >
::BreastMaskSegmForBreastDensity()
{

};


// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
BreastMaskSegmForBreastDensity< ImageDimension, PixelType >
::~BreastMaskSegmForBreastDensity()
{

};


// --------------------------------------------------------------------------
// Execute()
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
void
BreastMaskSegmForBreastDensity< ImageDimension, PixelType >
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
  PointSetType::Pointer pecPointSet = 
    this->SegmentThePectoralMuscle( static_cast<RealType>( 0.0 ) );

  MaskThePectoralMuscleAndLateralChestSkinSurface( pecPointSet );

  // Discard anything not within a fitted surface (switch -cropfit)
  if ( bCropWithFittedSurface )
    MaskWithBSplineBreastSurface();

  // OR Discard anything not within a certain radius of the breast center
  else 
    MaskBreastWithSphere();
  
  // Finally smooth the mask and threshold to round corners etc.
  SmoothMask();


}


// --------------------------------------------------------------------------
// Mask the pectoral muscle using a B-Spline surface
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class PixelType>
void
BreastMaskSegmForBreastDensity< ImageDimension, PixelType >
::MaskThePectoralMuscleAndLateralChestSkinSurface( PointSetType::Pointer &pecPointSet )
{
       
  // Extract the skin surface voxels
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ConnectedSurfaceVoxelFilterType::Pointer 
    connectedSurfacePoints = ConnectedSurfaceVoxelFilterType::New();

  connectedSurfacePoints->SetInput( imSegmented );
  connectedSurfacePoints->SetLower( 1000  );
  connectedSurfacePoints->SetUpper( 1000 );
  connectedSurfacePoints->SetReplaceValue( 1000 );
  connectedSurfacePoints->SetSeed( idxMidSternum  );
  connectedSurfacePoints->AddSeed( idxNippleLeft  );
  connectedSurfacePoints->AddSeed( idxNippleRight );

  try
  { 
    std::cout << "Region-growing the skin surface" << std::endl;
    connectedSurfacePoints->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }
  
  imChestSurfaceVoxels = connectedSurfacePoints->GetOutput();
  imChestSurfaceVoxels->DisconnectPipeline();

  // Extract the coordinates of the chest surface voxels -
  // i.e. posterior to the sternum
  
  InternalImageType::SizeType sizeChestSurfaceRegion;
  const InternalImageType::SpacingType& sp = imChestSurfaceVoxels->GetSpacing();

  start[0] = 0;
  start[1] = idxMidSternum[1];
  start[2] = 0;

  region = imChestSurfaceVoxels->GetLargestPossibleRegion();

  size = region.GetSize();
  sizeChestSurfaceRegion = size;

  sizeChestSurfaceRegion[1] = 60./sp[1];		// 60mm only

  if ( start[1] + sizeChestSurfaceRegion[1] > size[1] )
    sizeChestSurfaceRegion[1] = size[1] - start[1] - 1;

  region.SetSize( sizeChestSurfaceRegion );
  region.SetIndex( start );


  if ( flgVerbose )
    std::cout << "Collating chest surface points in region: "
	      << region << std::endl;

  IteratorType itSegPosteriorBreast( imChestSurfaceVoxels, region );
  

  for ( itSegPosteriorBreast.GoToBegin(); 
        ! itSegPosteriorBreast.IsAtEnd() ; 
        ++itSegPosteriorBreast )
  {
    if ( itSegPosteriorBreast.Get() ) {
      
      idx = itSegPosteriorBreast.GetIndex();
      
      // The 'height' of the chest surface
      pecHeight[0] = static_cast<RealType>( idx[1] );
      
      // Location of this surface point
      point[0] = static_cast<RealType>( idx[0] );
      point[1] = static_cast<RealType>( idx[2] );
      
      pecPointSet->SetPoint( iPointPec, point );
      pecPointSet->SetPointData( iPointPec, pecHeight );
      
      iPointPec++;
    }
  }
  

  // Fit the B-Spline surface with offset
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InternalImageType::Pointer imFittedPectoralis;

  imFittedPectoralis = 
    MaskImageFromBSplineFittedSurface( pecPointSet, 
				       imStructural->GetLargestPossibleRegion(), 
				       imStructural->GetOrigin(), 
				       imStructural->GetSpacing(), 
				       imStructural->GetDirection(), 
				       rYHeightOffset,
				       3, 5, 3 );

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
