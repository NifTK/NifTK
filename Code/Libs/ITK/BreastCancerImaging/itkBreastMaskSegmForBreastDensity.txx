/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include "itkBreastMaskSegmForBreastDensity.h"


namespace itk
{

// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class InputPixelType>
BreastMaskSegmForBreastDensity< ImageDimension, InputPixelType >
::BreastMaskSegmForBreastDensity()
{

};


// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class InputPixelType>
BreastMaskSegmForBreastDensity< ImageDimension, InputPixelType >
::~BreastMaskSegmForBreastDensity()
{

};


// --------------------------------------------------------------------------
// Execute()
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class InputPixelType>
void
BreastMaskSegmForBreastDensity< ImageDimension, InputPixelType >
::Execute( void )
{
  unsigned long iPointPec = 0;

  // Initialise the segmentation
  this->Initialise();
  this->SmoothTheInputImages();
  this->GreyScaleClosing();

  // Calculate the Maximum Image
  this->CalculateTheMaximumImage();

  // Segment the backgound using itkForegroundFromBackgroundImageThresholdCalculator
  this->SegmentForegroundFromBackground();

  // Find the nipple and mid-sternum landmarks
  this->FindBreastLandmarks();

  // Compute a 2D map of the height of the patient's anterior skin
  // surface and use it to remove the arms
  this->ComputeElevationOfAnteriorSurface();

  // Segment the Pectoral Muscle
  RealType rYHeightOffset = static_cast<RealType>( 0.0 );

  typename PointSetType::Pointer pecPointSet = 
    this->SegmentThePectoralMuscle( rYHeightOffset, iPointPec );

  MaskThePectoralMuscleAndLateralChestSkinSurface( rYHeightOffset, 
						   pecPointSet,
						   iPointPec );

  // Discard anything not within a fitted surface (switch -cropfit)
  if ( this->flgCropWithFittedSurface )
    this->MaskWithBSplineBreastSurface();

  // OR Discard anything not within a certain radius of the breast center
  else 
    this->MaskBreastWithSphere();
  
  // Smooth the mask and threshold to round corners etc.
  this->SmoothMask();

  // Extract the largest object
  this->ExtractLargestObject( this->LEFT_BREAST );

  std::string fileOutputLeft( "LabelledLeftImage.nii" );
  this->WriteImageToFile( fileOutputLeft, "largest objects", 
                          this->imSegmented );      

  this->ExtractLargestObject( this->RIGHT_BREAST );

  std::string fileOutputRight( "LabelledRightImage.nii" );
  this->WriteImageToFile( fileOutputRight, "largest objects", 
                          this->imSegmented );      

}


// --------------------------------------------------------------------------
// Mask the pectoral muscle using a B-Spline surface
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class InputPixelType>
void
BreastMaskSegmForBreastDensity< ImageDimension, InputPixelType >
::MaskThePectoralMuscleAndLateralChestSkinSurface( RealType rYHeightOffset, 
						   typename PointSetType::Pointer &pecPointSet,
						   unsigned long &iPointPec )
{
  typename InternalImageType::RegionType region;
  typename InternalImageType::SizeType size;
  typename InternalImageType::IndexType start;

       
  // Extract the skin surface voxels
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename ConnectedSurfaceVoxelFilterType::Pointer 
    connectedSurfacePoints = ConnectedSurfaceVoxelFilterType::New();

  connectedSurfacePoints->SetInput( this->imSegmented );
  connectedSurfacePoints->SetLower( 1000  );
  connectedSurfacePoints->SetUpper( 1000 );
  connectedSurfacePoints->SetReplaceValue( 1000 );
  connectedSurfacePoints->SetSeed( this->idxMidSternum  );

  connectedSurfacePoints->AddSeed( this->idxNippleLeft  );
  connectedSurfacePoints->AddSeed( this->idxNippleRight );

  connectedSurfacePoints->AddSeed( this->idxAreolarLeft[0] );
  connectedSurfacePoints->AddSeed( this->idxAreolarLeft[1] );
  connectedSurfacePoints->AddSeed( this->idxAreolarLeft[2] );
  connectedSurfacePoints->AddSeed( this->idxAreolarLeft[3] );

  connectedSurfacePoints->AddSeed( this->idxAreolarRight[0] );
  connectedSurfacePoints->AddSeed( this->idxAreolarRight[1] );
  connectedSurfacePoints->AddSeed( this->idxAreolarRight[2] );
  connectedSurfacePoints->AddSeed( this->idxAreolarRight[3] );

  std::cout << "Region-growing the skin surface" << std::endl;
  connectedSurfacePoints->Update();
  
  this->imChestSurfaceVoxels = connectedSurfacePoints->GetOutput();
  this->imChestSurfaceVoxels->DisconnectPipeline();

  // Extract the coordinates of the chest surface voxels -
  // i.e. posterior to the sternum
  
  typename InternalImageType::SizeType sizeChestSurfaceRegion;
  const typename InternalImageType::SpacingType& sp = this->imChestSurfaceVoxels->GetSpacing();

  start[0] = 0;
  start[1] = this->idxMidSternum[1];
  start[2] = 0;

  region = this->imChestSurfaceVoxels->GetLargestPossibleRegion();

  size = region.GetSize();
  sizeChestSurfaceRegion = size;

  sizeChestSurfaceRegion[1] = static_cast<typename InternalImageType::SizeValueType>(60./sp[1]);		// 60mm only

  if ( start[1] + sizeChestSurfaceRegion[1] > size[1] )
    sizeChestSurfaceRegion[1] = size[1] - start[1] - 1;

  region.SetSize( sizeChestSurfaceRegion );
  region.SetIndex( start );


  if ( this->flgVerbose )
    std::cout << "Collating chest surface points in region: "
	      << region << std::endl;

  IteratorType itSegPosteriorBreast( this->imChestSurfaceVoxels, region );
  
  VectorType pecHeight;
  typename InternalImageType::IndexType idx;
  typename PointSetType::PointType point;

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

 typename  InternalImageType::Pointer imFittedPectoralis;

  imFittedPectoralis = 
    this->MaskImageFromBSplineFittedSurface( pecPointSet, 
					     this->imStructural->GetLargestPossibleRegion(), 
					     this->imStructural->GetOrigin(), 
					     this->imStructural->GetSpacing(), 
					     this->imStructural->GetDirection(), 
					     rYHeightOffset,
					     3, 5, 3, false );

  // Write the fitted surface to file

  this->WriteImageToFile( this->fileOutputPectoralSurfaceMask, 
			  "fitted pectoral surface with offset", 
			  imFittedPectoralis, this->flgLeft, this->flgRight );
  
  // Write the chest surface points to a file?

  this->WriteBinaryImageToUCharFile( this->fileOutputChestPoints, 
				     "chest surface points", 
				     this->imChestSurfaceVoxels, this->flgLeft, this->flgRight );


 // Discard anything within the pectoral mask (i.e. below the surface fit)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  IteratorType itSeg    = IteratorType( this->imSegmented,        
					this->imStructural->GetLargestPossibleRegion() );

  IteratorType itFitPec = IteratorType( imFittedPectoralis, 
					this->imStructural->GetLargestPossibleRegion() );

  if ( this->flgVerbose ) 
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


  // Discard anything below the pectoral mask
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
  if ( this->flgVerbose ) 
    std::cout << "Discarding segmentation posteriior to pectoralis mask. " << std::endl;

  region = this->imPectoralVoxels->GetLargestPossibleRegion();
 
  start[0] = start[1] = start[2] = 0;
  region.SetIndex( start );
 
  LineIteratorType itSegLinear2( this->imSegmented, region );
  LineIteratorType itPecVoxelsLinear( this->imPectoralVoxels, region );
 
  itPecVoxelsLinear.SetDirection( 1 );
  itSegLinear2.SetDirection( 1 );
     
  for ( itPecVoxelsLinear.GoToBegin(), itSegLinear2.GoToBegin(); 
	! itPecVoxelsLinear.IsAtEnd(); 
	itPecVoxelsLinear.NextLine(), itSegLinear2.NextLine() )
  {
    itPecVoxelsLinear.GoToBeginOfLine();
    itSegLinear2.GoToBeginOfLine();
       
    // Find the first pectoral voxel for this column of voxels
 
    while ( ! itPecVoxelsLinear.IsAtEndOfLine() )
    {
      if ( itPecVoxelsLinear.Get() > 0 ) 
      {
	break;
      }
  
      ++itPecVoxelsLinear;
      ++itSegLinear2;
    }
 
    // and then set all remaining voxles in the segmented image to zero
 
    while ( ! itPecVoxelsLinear.IsAtEndOfLine() )
    {
      itSegLinear2.Set( 0 );
           
      ++itPecVoxelsLinear;
      ++itSegLinear2;
    }      
  }

  this->imPectoralVoxels = 0;
}
 


} // namespace itk
