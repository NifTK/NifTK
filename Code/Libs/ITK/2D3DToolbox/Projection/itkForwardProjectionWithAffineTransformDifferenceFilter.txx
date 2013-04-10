/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkForwardProjectionWithAffineTransformDifferenceFilter_txx
#define __itkForwardProjectionWithAffineTransformDifferenceFilter_txx

#include "itkForwardProjectionWithAffineTransformDifferenceFilter.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::ForwardProjectionWithAffineTransformDifferenceFilter()
{
  m_FlagPipelineInitialised = false;

  this->SetNumberOfRequiredInputs( 3 );
	
  this->SetNumberOfRequiredOutputs( 2 );

  m_NumberOfProjections = 0;

  // Create the forward projector
  m_ForwardProjectorOne = CreateForwardBackwardProjectionMatrixType::New();

  // Create the subtraction filter
  m_SubtractProjectionFromEstimateOne = Subtract2DImageFromVolumeSliceFilterType::New();

  // Create the back projector
  m_BackProjectorOne = BackwardImageProjector2Dto3DType::New();

  // Set the projection geometry (currently only one option)
  m_ProjectionGeometry = 0;
  
}


/* -----------------------------------------------------------------------
   SetInputVolume
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::SetInputVolume( InputVolumeType *im3D )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<InputVolumeType *>( im3D ));
}


/* -----------------------------------------------------------------------
   SetInputProjectionVolumeOne
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::SetInputProjectionVolumeOne( InputProjectionVolumeType *im2D )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(1, const_cast<InputProjectionVolumeType *>( im2D ));
}

/* -----------------------------------------------------------------------
   SetInputProjectionVolumeTwo
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::SetInputProjectionVolumeTwo( InputProjectionVolumeType *im2D )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(2, const_cast<InputProjectionVolumeType *>( im2D ));
}


/* -----------------------------------------------------------------------
   GetPointerToInputVolume()
   ----------------------------------------------------------------------- */

template <class IntensityType>
typename ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>::InputVolumePointer 
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::GetPointerToInputVolume( void )
{
  return dynamic_cast<InputVolumeType *>(ProcessObject::GetInput(0));
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if (m_FlagPipelineInitialised)
    os << indent << "Pipeline initialized" << std::endl;
  else
    os << indent << "Pipeline uninitialized" << std::endl;

  os << indent << "Number of projections: " << m_NumberOfProjections << std::endl;

  if (! m_ForwardProjectorOne.IsNull()) {
    os << indent << "Forward Projector: " << std::endl;
    m_ForwardProjectorOne.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Forward Projector: NULL" << std::endl;

  if (! m_SubtractProjectionFromEstimateOne.IsNull()) {
    os << indent << "Subtract 2D Image from Volume Slice Filter: " << std::endl;
    m_SubtractProjectionFromEstimateOne.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Subtract 2D Image from Volume Slice Filter: NULL" << std::endl;

  if (! m_BackProjectorOne.IsNull()) {
    os << indent << "Back Projector: " << std::endl;
    m_BackProjectorOne.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Back Projector: NULL" << std::endl;

  if (! m_ProjectionGeometry.IsNull()) {
    os << indent << "Projection Geometry: " << std::endl;
    m_ProjectionGeometry.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Projection Geometry: NULL" << std::endl;
}


/* -----------------------------------------------------------------------
   GenerateInputRequestedRegion()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::GenerateInputRequestedRegion()
{
  // generate everything in the region of interest
  InputVolumePointer inputPtr = const_cast<InputVolumeType *> (this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();
}


/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion(DataObject *)
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  // call the superclass' implementation of this method
  Superclass::EnlargeOutputRequestedRegion(output);
  
  // generate everything in the region of interest
  this->GetOutput()->SetRequestedRegionToLargestPossibleRegion();
}


/* -----------------------------------------------------------------------
   Initialise()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::Initialise(void)
{
  if (! m_FlagPipelineInitialised) {

    InputVolumePointer pInputVolume
      = dynamic_cast<InputVolumeType *>(ProcessObject::GetInput(0));
    InputVolumeConstPointer pInputConstVolume
      = dynamic_cast<const InputVolumeType *>(ProcessObject::GetInput(0));

    InputProjectionVolumePointer pInputProjectionsOne
      = dynamic_cast<InputProjectionVolumeType *>(ProcessObject::GetInput(1));

    InputProjectionVolumeSizeType    projSize3DOne    = pInputProjectionsOne->GetLargestPossibleRegion().GetSize();
    InputProjectionVolumeSpacingType projSpacing3DOne = pInputProjectionsOne->GetSpacing();
    InputProjectionVolumePointType   projOrigin3DOne  = pInputProjectionsOne->GetOrigin();

    InputProjectionVolumePointer pInputProjectionsTwo
      = dynamic_cast<InputProjectionVolumeType *>(ProcessObject::GetInput(2));

    InputProjectionVolumeSizeType    projSize3DTwo    = pInputProjectionsTwo->GetLargestPossibleRegion().GetSize();
    InputProjectionVolumeSpacingType projSpacing3DTwo = pInputProjectionsTwo->GetSpacing();
    InputProjectionVolumePointType   projOrigin3DTwo  = pInputProjectionsTwo->GetOrigin();

    assert((projSize3DOne[0]==projSize3DTwo[0])&&(projSize3DOne[1]==projSize3DTwo[1])&&(projSize3DOne[2]==projSize3DTwo[2])&&
	   (projSpacing3DOne[0]==projSpacing3DTwo[0])&&(projSpacing3DOne[0]==projSpacing3DTwo[0])&&(projSpacing3DOne[0]==projSpacing3DTwo[0])&&
	   (projOrigin3DOne[0]==projOrigin3DTwo[0])&&(projOrigin3DOne[0]==projOrigin3DTwo[0])&&(projOrigin3DOne[0]==projOrigin3DTwo[0]));

    // Set-up the forward projector one
    
    m_ForwardProjectorOne->SetInput( pInputVolume );

    typename ForwardProjectorOutputImageType::PointType fwdProjOrigin2DOne;
    fwdProjOrigin2DOne[0] = projOrigin3DOne[0];
    fwdProjOrigin2DOne[1] = projOrigin3DOne[1];
    m_ForwardProjectorOne->SetProjectedImageOrigin( fwdProjOrigin2DOne );

    typename ForwardProjectorOutputImageType::SizeType fwdprojSize2DOne;
    fwdprojSize2DOne[0] = projSize3DOne[0];
    fwdprojSize2DOne[1] = projSize3DOne[1];
    m_ForwardProjectorOne->SetProjectedImageSize( fwdprojSize2DOne );

    typename ForwardProjectorOutputImageType::SpacingType fwdprojSpacing2DOne;
    fwdprojSpacing2DOne[0] = projSpacing3DOne[0];
    fwdprojSpacing2DOne[1] = projSpacing3DOne[1];
    m_ForwardProjectorOne->SetProjectedImageSpacing( fwdprojSpacing2DOne );

    // Set-up the forward projector two
    
    m_ForwardProjectorTwo->SetInput( pInputVolume ); // Should change into the affine transformation of the original volume

    typename ForwardProjectorOutputImageType::PointType fwdProjOrigin2DTwo;
    fwdProjOrigin2DTwo[0] = projOrigin3DTwo[0];
    fwdProjOrigin2DTwo[1] = projOrigin3DTwo[1];
    m_ForwardProjectorTwo->SetProjectedImageOrigin( fwdProjOrigin2DTwo );

    typename ForwardProjectorOutputImageType::SizeType fwdprojSize2DTwo;
    fwdprojSize2DTwo[0] = projSize3DTwo[0];
    fwdprojSize2DTwo[1] = projSize3DTwo[1];
    m_ForwardProjectorTwo->SetProjectedImageSize( fwdprojSize2DTwo );

    typename ForwardProjectorOutputImageType::SpacingType fwdprojSpacing2DTwo;
    fwdprojSpacing2DTwo[0] = projSpacing3DTwo[0];
    fwdprojSpacing2DTwo[1] = projSpacing3DTwo[1];
    m_ForwardProjectorTwo->SetProjectedImageSpacing( fwdprojSpacing2DTwo );
    

    // Set-up the subtraction image filter one

    m_SubtractProjectionFromEstimateOne->SetInputImage2D( m_ForwardProjectorOne->GetOutput() );
    m_SubtractProjectionFromEstimateOne->SetInputVolume3D( pInputProjectionsOne );

    // Set-up the subtraction image filter two

    m_SubtractProjectionFromEstimateTwo->SetInputImage2D( m_ForwardProjectorTwo->GetOutput() );
    m_SubtractProjectionFromEstimateTwo->SetInputVolume3D( pInputProjectionsTwo );


    // Set-up the back-projection filter one

    m_BackProjectorOne->SetInput( m_SubtractProjectionFromEstimateOne->GetOutput() );

    typename BackwardImageProjector2Dto3DType::OutputImageSizeType backProjectedSizeOne 
      = pInputVolume->GetLargestPossibleRegion().GetSize();
    m_BackProjectorOne->SetBackProjectedImageSize(backProjectedSizeOne);

    typename BackwardImageProjector2Dto3DType::OutputImageSpacingType backProjectedSpacingOne 
      = pInputVolume->GetSpacing();
    m_BackProjectorOne->SetBackProjectedImageSpacing(backProjectedSpacingOne);

    typename BackwardImageProjector2Dto3DType::OutputImagePointType backProjectedOriginOne 
      = pInputVolume->GetOrigin();
    m_BackProjectorOne->SetBackProjectedImageOrigin(backProjectedOriginOne);

    // Set-up the back-projection filter two

    m_BackProjectorTwo->SetInput( m_SubtractProjectionFromEstimateTwo->GetOutput() );

    typename BackwardImageProjector2Dto3DType::OutputImageSizeType backProjectedSizeTwo 
      = pInputVolume->GetLargestPossibleRegion().GetSize();
    m_BackProjectorTwo->SetBackProjectedImageSize(backProjectedSizeTwo);

    typename BackwardImageProjector2Dto3DType::OutputImageSpacingType backProjectedSpacingTwo 
      = pInputVolume->GetSpacing();
    m_BackProjectorTwo->SetBackProjectedImageSpacing(backProjectedSpacingTwo);

    typename BackwardImageProjector2Dto3DType::OutputImagePointType backProjectedOriginTwo 
      = pInputVolume->GetOrigin();
    m_BackProjectorTwo->SetBackProjectedImageOrigin(backProjectedOriginTwo);


    // Set-up the tomosythesis geometry

    typename ProjectionGeometryType::ProjectionSizeType projSize2DOne;
    projSize2DOne[0] = projSize3DOne[0];
    projSize2DOne[1] = projSize3DOne[1];
    m_NumberOfProjections = projSize3DOne[2];

    typename ProjectionGeometryType::ProjectionSpacingType projSpacing2DOne;
    projSpacing2DOne[0] = projSpacing3DOne[0];
    projSpacing2DOne[1] = projSpacing3DOne[1];

    m_ProjectionGeometry->SetProjectionSize( projSize2DOne );
    m_ProjectionGeometry->SetProjectionSpacing( projSpacing2DOne );

    m_ProjectionGeometry->SetVolumeSize( pInputConstVolume->GetLargestPossibleRegion().GetSize() );
    m_ProjectionGeometry->SetVolumeSpacing( pInputConstVolume->GetSpacing() );
    
    m_FlagPipelineInitialised = true;
  }
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType>
::GenerateData(void)
{

  unsigned int iProjection;		// Set the projection iteration number

  EulerAffineTransformPointer affineTransform;
  PerspectiveProjectionTransformPointer perspTransform;

  // Initialise parameters
  // ~~~~~~~~~~~~~~~~~~~~~

  Initialise();


  // Execute the forward and back projection
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for (iProjection=0; iProjection<m_NumberOfProjections; iProjection++) {

    niftkitkInfoMacro(<< "Performing forward and back projections: " << iProjection);

    // Perform the set one
    perspTransform = m_ProjectionGeometry->GetPerspectiveTransform( iProjection );
    affineTransform = m_ProjectionGeometry->GetAffineTransform( iProjection );

    m_ForwardProjectorOne->SetPerspectiveTransform( perspTransform );
    m_ForwardProjectorOne->SetAffineTransform( affineTransform );

    m_SubtractProjectionFromEstimateOne->SetSliceNumber( iProjection );

    m_BackProjectorOne->SetPerspectiveTransform( perspTransform );
    m_BackProjectorOne->SetAffineTransform( affineTransform );

    m_BackProjectorOne->GraftOutput( this->GetOutput() );
    m_BackProjectorOne->Update();

    // Perform the set two
    m_ForwardProjectorTwo->SetPerspectiveTransform( perspTransform );
    m_ForwardProjectorTwo->SetAffineTransform( affineTransform );

    m_SubtractProjectionFromEstimateTwo->SetSliceNumber( iProjection );

    m_BackProjectorTwo->SetPerspectiveTransform( perspTransform );
    m_BackProjectorTwo->SetAffineTransform( affineTransform );

    m_BackProjectorTwo->GraftOutput( this->GetOutput() );
    m_BackProjectorTwo->Update();

    niftkitkDebugMacro(<< "Finished forward and back projections: " << iProjection << endl);

  }

  this->SetNthOutput(0, m_BackProjectorOne->GetOutput() );
  this->SetNthOutput(1, m_BackProjectorTwo->GetOutput() );
}


} // end namespace itk


#endif
