/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include "itkBreastMaskSegmForBreastDensity.h"

#include <itkSignedMaurerDistanceMapImageFilter.h>

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
  if ( this->bgndThresholdProb )
  {
    this->SegmentBackground();
  }
  else
  {
    this->SegmentForegroundFromBackground();
  }

  // Find the nipple and mid-sternum landmarks
  this->FindBreastLandmarks();

  // Compute a 2D map of the height of the patient's anterior skin
  // surface and use it to remove the arms
  this->ComputeElevationOfAnteriorSurface( true );

  // Segment the Pectoral Muscle

#if 0

  typename InternalImageType::SizeType 
    maxSize = this->imStructural->GetLargestPossibleRegion().GetSize();

  RealType rYHeightOffset = static_cast< RealType >( maxSize[1] );

  typename PointSetType::Pointer pecPointSet = 
    this->SegmentThePectoralMuscle( rYHeightOffset, iPointPec, true );

#else

  typename PointSetType::Pointer pecPointSet = 
    this->SegmentThePectoralMuscle( 0., iPointPec );


  // Calculate the most posterior pectoral point

  RealType rYHeightOffset = 0.;
  PointDataIterator pointDataIt;

  for ( pointDataIt = pecPointSet->GetPointData()->Begin();
        pointDataIt != pecPointSet->GetPointData()->End();
        ++pointDataIt )
  {    
    if ( pointDataIt.Value()[0] > rYHeightOffset )
    {
      rYHeightOffset = pointDataIt.Value()[0];
    }
  }

  std::cout << "Most posterior pectoral muscle height: " << rYHeightOffset << std::endl;

  for ( pointDataIt = pecPointSet->GetPointData()->Begin();
        pointDataIt != pecPointSet->GetPointData()->End();
        ++pointDataIt )
  {    
    pointDataIt.Value()[0] -= rYHeightOffset;
  }
#endif

  // Mask the pectoral muscle

  MaskThePectoralMuscleAndLateralChestSkinSurface( false,
                                                   rYHeightOffset, 
						   pecPointSet,
						   iPointPec );

  // Discard anything not within a fitted surface (switch -cropfit)
  if ( this->flgCropWithFittedSurface )
    this->MaskWithBSplineBreastSurface( rYHeightOffset );

  // Discard anything not within the skin elevation mask  
  if ( this->imSkinElevationMap )
    this->CropTheMaskAccordingToEstimateOfCoilExtentInCoronalPlane();
    
  // Smooth the mask and threshold to round corners etc.
  this->SmoothMask();

  // Shrink the mask by a few millimeters to eliminiate the skin

  typedef SignedMaurerDistanceMapImageFilter<InternalImageType, InternalImageType> InputDistanceMapFilterType;

  typename InputDistanceMapFilterType::Pointer inputDistFilter = InputDistanceMapFilterType::New();

  inputDistFilter->SetInput( this->imSegmented );
  inputDistFilter->SetUseImageSpacing(true);
  inputDistFilter->InsideIsPositiveOn();

  std::cout << "Computing distance transform for skin estimation" << std::endl;
  inputDistFilter->Update();


  typename ThresholdingFilterType::Pointer thresholder = ThresholdingFilterType::New();
  
  thresholder->SetLowerThreshold( 2. ); // Remove 2mm
  thresholder->SetUpperThreshold( 100000 );

  thresholder->SetOutsideValue(  0  );
  thresholder->SetInsideValue( 1000 );

  thresholder->SetInput( inputDistFilter->GetOutput() );

  thresholder->Update();
  
  this->imSegmented = thresholder->GetOutput();

  // Extract the largest object
  this->ExtractLargestObject( this->LEFT_BREAST );

#if 0
  std::string fileOutputLeft( "LabelledLeftImage.nii" );
  this->WriteImageToFile( fileOutputLeft, "largest objects", 
                          this->imSegmented );      
#endif

  this->ExtractLargestObject( this->RIGHT_BREAST );

#if 0
  std::string fileOutputRight( "LabelledRightImage.nii" );
  this->WriteImageToFile( fileOutputRight, "largest objects", 
                          this->imSegmented );      
#endif
}


// --------------------------------------------------------------------------
// Mask the pectoral muscle using a B-Spline surface
// --------------------------------------------------------------------------

template <const unsigned int ImageDimension, class InputPixelType>
void
BreastMaskSegmForBreastDensity< ImageDimension, InputPixelType >
::MaskThePectoralMuscleAndLateralChestSkinSurface( bool flgIncludeChestSkinSurface,
                                                   RealType rYHeightOffset, 
						   typename PointSetType::Pointer &pecPointSet,
						   unsigned long &iPointPec )
{
  typename InternalImageType::IndexType start;

  typename InternalImageType::RegionType
    region = this->imSegmented->GetLargestPossibleRegion();

       
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
  
  if ( flgIncludeChestSkinSurface )
  {

    typename InternalImageType::SizeType sizeChestSurfaceRegion;
    const typename InternalImageType::SpacingType& sp = this->imChestSurfaceVoxels->GetSpacing();

    start[0] = 0;
    start[1] = this->idxMidSternum[1];
    start[2] = 0;
    
    typename InternalImageType::SizeType size = region.GetSize();
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
  }
  

  // Write the chest surface points to a file?

  this->WriteBinaryImageToUCharFile( this->fileOutputChestPoints, 
                                     "chest surface points", 
                                     this->imChestSurfaceVoxels, 
                                     this->flgLeft, this->flgRight );


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
					     3, this->pecControlPointSpacing, 3, true );
  
  // Write the fitted surface to file

  this->WriteImageToFile( this->fileOutputPectoralSurfaceMask, 
			  "fitted pectoral surface with offset", 
			  imFittedPectoralis, 
                          this->flgLeft, this->flgRight );


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
