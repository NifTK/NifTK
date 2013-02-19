/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkForwardAndBackProjectionDifferenceFilter_txx
#define __itkForwardAndBackProjectionDifferenceFilter_txx

#include "itkForwardAndBackProjectionDifferenceFilter.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
ForwardAndBackProjectionDifferenceFilter<IntensityType>
::ForwardAndBackProjectionDifferenceFilter()
{
  m_FlagPipelineInitialised = false;

  this->SetNumberOfRequiredInputs( 2 );

  m_NumberOfProjections = 0;

  // Create the forward projector
  m_ForwardProjector = ForwardImageProjector3Dto2DType::New();

  // Create the subtraction filter
  m_SubtractProjectionFromEstimate = Subtract2DImageFromVolumeSliceFilterType::New();

  // Create the back projector
  m_BackProjector = BackwardImageProjector2Dto3DType::New();

  // Set the projection geometry (currently only one option)
  m_ProjectionGeometry = 0;
  
}


/* -----------------------------------------------------------------------
   SetInputVolume
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardAndBackProjectionDifferenceFilter<IntensityType>
::SetInputVolume( InputVolumeType *im3D )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<InputVolumeType *>( im3D ));
}


/* -----------------------------------------------------------------------
   SetInputProjectionVolume
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardAndBackProjectionDifferenceFilter<IntensityType>
::SetInputProjectionVolume( InputProjectionVolumeType *im2D )
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(1, const_cast<InputProjectionVolumeType *>( im2D ));
}


/* -----------------------------------------------------------------------
   GetPointerToInputVolume()
   ----------------------------------------------------------------------- */

template <class IntensityType>
typename ForwardAndBackProjectionDifferenceFilter<IntensityType>::InputVolumePointer 
ForwardAndBackProjectionDifferenceFilter<IntensityType>
::GetPointerToInputVolume( void )
{
  return dynamic_cast<InputVolumeType *>(ProcessObject::GetInput(0));
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardAndBackProjectionDifferenceFilter<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if (m_FlagPipelineInitialised)
    os << indent << "Pipeline initialized" << std::endl;
  else
    os << indent << "Pipeline uninitialized" << std::endl;

  os << indent << "Number of projections: " << m_NumberOfProjections << std::endl;

  if (! m_ForwardProjector.IsNull()) {
    os << indent << "Forward Projector: " << std::endl;
    m_ForwardProjector.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Forward Projector: NULL" << std::endl;

  if (! m_SubtractProjectionFromEstimate.IsNull()) {
    os << indent << "Subtract 2D Image from Volume Slice Filter: " << std::endl;
    m_SubtractProjectionFromEstimate.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Subtract 2D Image from Volume Slice Filter: NULL" << std::endl;

  if (! m_BackProjector.IsNull()) {
    os << indent << "Back Projector: " << std::endl;
    m_BackProjector.GetPointer()->Print(os, indent.GetNextIndent());
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
ForwardAndBackProjectionDifferenceFilter<IntensityType>
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
ForwardAndBackProjectionDifferenceFilter<IntensityType>
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
ForwardAndBackProjectionDifferenceFilter<IntensityType>
::Initialise(void)
{
  if (! m_FlagPipelineInitialised) {

    InputVolumePointer pInputVolume
      = dynamic_cast<InputVolumeType *>(ProcessObject::GetInput(0));
    InputVolumeConstPointer pInputConstVolume
      = dynamic_cast<const InputVolumeType *>(ProcessObject::GetInput(0));

    InputProjectionVolumePointer pInputProjections
      = dynamic_cast<InputProjectionVolumeType *>(ProcessObject::GetInput(1));

    InputProjectionVolumeSizeType    projSize3D    = pInputProjections->GetLargestPossibleRegion().GetSize();
    InputProjectionVolumeSpacingType projSpacing3D = pInputProjections->GetSpacing();
    InputProjectionVolumePointType   projOrigin3D  = pInputProjections->GetOrigin();



    // Set-up the forward projector
    
    m_ForwardProjector->SetInput( pInputVolume );

    typename ForwardProjectorOutputImageType::PointType fwdProjOrigin2D;
    fwdProjOrigin2D[0] = projOrigin3D[0];
    fwdProjOrigin2D[1] = projOrigin3D[1];
    m_ForwardProjector->SetProjectedImageOrigin( fwdProjOrigin2D );

    typename ForwardProjectorOutputImageType::SizeType fwdProjSize2D;
    fwdProjSize2D[0] = projSize3D[0];
    fwdProjSize2D[1] = projSize3D[1];
    m_ForwardProjector->SetProjectedImageSize( fwdProjSize2D );

    typename ForwardProjectorOutputImageType::SpacingType fwdProjSpacing2D;
    fwdProjSpacing2D[0] = projSpacing3D[0];
    fwdProjSpacing2D[1] = projSpacing3D[1];
    m_ForwardProjector->SetProjectedImageSpacing( fwdProjSpacing2D );
    

    // Set-up the subtraction image filter

    m_SubtractProjectionFromEstimate->SetInputImage2D( m_ForwardProjector->GetOutput() );
    m_SubtractProjectionFromEstimate->SetInputVolume3D( pInputProjections );


    // Set-up the back-projection filter

    m_BackProjector->SetInput( m_SubtractProjectionFromEstimate->GetOutput() );

    typename BackwardImageProjector2Dto3DType::OutputImageSizeType backProjectedSize 
      = pInputVolume->GetLargestPossibleRegion().GetSize();
    m_BackProjector->SetBackProjectedImageSize(backProjectedSize);

    typename BackwardImageProjector2Dto3DType::OutputImageSpacingType backProjectedSpacing 
      = pInputVolume->GetSpacing();
    m_BackProjector->SetBackProjectedImageSpacing(backProjectedSpacing);

    typename BackwardImageProjector2Dto3DType::OutputImagePointType backProjectedOrigin 
      = pInputVolume->GetOrigin();
    m_BackProjector->SetBackProjectedImageOrigin(backProjectedOrigin);


    // Set-up the tomosythesis geometry

    typename ProjectionGeometryType::ProjectionSizeType projSize2D;
    projSize2D[0] = projSize3D[0];
    projSize2D[1] = projSize3D[1];
    m_NumberOfProjections = projSize3D[2];

    typename ProjectionGeometryType::ProjectionSpacingType projSpacing2D;
    projSpacing2D[0] = projSpacing3D[0];
    projSpacing2D[1] = projSpacing3D[1];

    m_ProjectionGeometry->SetProjectionSize( projSize2D );
    m_ProjectionGeometry->SetProjectionSpacing( projSpacing2D );

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
ForwardAndBackProjectionDifferenceFilter<IntensityType>
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

    perspTransform = m_ProjectionGeometry->GetPerspectiveTransform( iProjection );
    affineTransform = m_ProjectionGeometry->GetAffineTransform( iProjection );

    m_ForwardProjector->SetPerspectiveTransform( perspTransform );
    m_ForwardProjector->SetAffineTransform( affineTransform );

    m_SubtractProjectionFromEstimate->SetSliceNumber( iProjection );

    m_BackProjector->SetPerspectiveTransform( perspTransform );
    m_BackProjector->SetAffineTransform( affineTransform );

    m_BackProjector->GraftOutput( this->GetOutput() );
    m_BackProjector->Update();

    niftkitkDebugMacro(<< "Finished forward and back projections: " << iProjection << endl);

  }

  this->GraftOutput( m_BackProjector->GetOutput() );
}


} // end namespace itk


#endif
