/*=============================================================================

  NifTK: An image processing toolkit jointly developed by the
  Dementia Research Centre, and the Centre For Medical Image Computing
  at University College London.

  See:        http://dementia.ion.ucl.ac.uk/
  http://cmic.cs.ucl.ac.uk/
  http://www.ucl.ac.uk/

  Last Changed      : $Date: 2012-08-13 13:00:32 +0100 (Mon, 13 Aug 2012) $
  Revision          : $Revision: 9470 $
  Last modified by  : $Author: jhh $

  Original author   : j.hipwell@ucl.ac.uk

  Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

  ============================================================================*/

#include <QTimer>
#include <QMessageBox>

#include "RegistrationExecution.h"

#include "mitkImageToNifti.h"
#include "niftiImageToMitk.h"

#include "mitkSurface.h"

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkVertex.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridGeometryFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkSphereSource.h>


// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

RegistrationExecution::RegistrationExecution( void *param )
{

  userData = static_cast<QmitkNiftyRegView*>( param );

}


// ---------------------------------------------------------------------------
// run()
// ---------------------------------------------------------------------------

void RegistrationExecution::run()
{
  
  QTimer::singleShot(0, this, SLOT( ExecuteRegistration() ));
  exec();

}


// ---------------------------------------------------------------------------
// ExecuteRegistration()
// ---------------------------------------------------------------------------

void RegistrationExecution::ExecuteRegistration()
{

  // Get the source and target MITK images from the data manager

  std::string name;

  QString sourceName = userData->m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = userData->m_Controls.m_TargetImageComboBox->currentText();

  QString targetMaskName = userData->m_Controls.m_TargetMaskImageComboBox->currentText();

  mitk::Image::Pointer mitkSourceImage = 0;
  mitk::Image::Pointer mitkTransformedImage = 0;
  mitk::Image::Pointer mitkTargetImage = 0;

  mitk::Image::Pointer mitkTargetMaskImage = 0;


  mitk::DataStorage::SetOfObjects::ConstPointer nodes = userData->GetNodes();

  mitk::DataNode::Pointer nodeSource = 0;
  mitk::DataNode::Pointer nodeTarget = 0;
  
  if ( nodes ) 
  {
    if (nodes->size() > 0) 
    {
      for (unsigned int i=0; i<nodes->size(); i++) 
      {
	(*nodes)[i]->GetStringProperty("name", name);

	if ( ( QString(name.c_str()) == sourceName ) && ( ! mitkSourceImage ) ) 
	{
	  mitkSourceImage = dynamic_cast<mitk::Image*>((*nodes)[i]->GetData());
	  nodeSource = (*nodes)[i];
	}

	if ( ( QString(name.c_str()) == targetName ) && ( ! mitkTargetImage ) ) 
	{
	  mitkTargetImage = dynamic_cast<mitk::Image*>((*nodes)[i]->GetData());
	  nodeTarget = (*nodes)[i];
	}

	if ( ( QString(name.c_str()) == targetMaskName ) && ( ! mitkTargetMaskImage ) )
	  mitkTargetMaskImage = dynamic_cast<mitk::Image*>((*nodes)[i]->GetData());
      }
    }
  }


  // Ensure the progress bar is scaled appropriately

  if (    userData->m_RegParameters.m_FlagDoInitialRigidReg 
	  && userData->m_RegParameters.m_FlagDoNonRigidReg ) 
    userData->m_ProgressBarRange = 50.;
  else
    userData->m_ProgressBarRange = 100.;

  userData->m_ProgressBarOffset = 0.;

  
  // Create and run the Aladin registration?

  if ( userData->m_RegParameters.m_FlagDoInitialRigidReg ) 
  {

    userData->m_RegAladin = 
      userData->m_RegParameters.CreateAladinRegistrationObject( mitkSourceImage, 
								mitkTargetImage, 
								mitkTargetMaskImage );
  
    userData->m_RegAladin->SetProgressCallbackFunction( &UpdateProgressBar, userData );

    userData->m_RegAladin->Run();

    // Transform the source image...

#if 0

    nifti_image *transformedFloatingImage = 
      nifti_copy_nim_info( userData->m_RegParameters.m_FloatingImage );

    transformedFloatingImage->data = (void *) malloc( transformedFloatingImage->nvox 
						      * transformedFloatingImage->nbyper );

    memcpy( transformedFloatingImage->data, userData->m_RegParameters.m_FloatingImage->data,
	    transformedFloatingImage->nvox * transformedFloatingImage->nbyper );

    mat44 *affineTransformation = userData->m_RegAladin->GetTransformationMatrix();
    mat44 invAffineTransformation = nifti_mat44_inverse( *affineTransformation );

    // ...by updating the sform matrix

    if ( transformedFloatingImage->sform_code > 0 )
    {
      transformedFloatingImage->sto_xyz = reg_mat44_mul( &invAffineTransformation, 
							 &(transformedFloatingImage->sto_xyz) );
    }
    else
    {
      transformedFloatingImage->sform_code = 1;
      transformedFloatingImage->sto_xyz = reg_mat44_mul( &invAffineTransformation, 
							 &(transformedFloatingImage->qto_xyz) );
    }
     
    transformedFloatingImage->sto_ijk = nifti_mat44_inverse( transformedFloatingImage->sto_xyz );

#else

    // ...or getting the resampled volume.

    nifti_image *transformedFloatingImage = userData->m_RegAladin->GetFinalWarpedImage();

#endif

    mitkSourceImage = ConvertNiftiImageToMitk( transformedFloatingImage );
    nifti_image_free( transformedFloatingImage );


    // Add this result to the data manager

    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();

    std::string nameOfResultImage;
    if ( userData->m_RegParameters.m_AladinParameters.regnType == RIGID_ONLY )
      nameOfResultImage = "RigidRegnOf_";
    else
      nameOfResultImage = "AffineRegnOf_";
    nameOfResultImage.append( nodeSource->GetName() );
    nameOfResultImage.append( "_To_" );
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkSourceImage );

    userData->GetDataStorage()->Add( resultNode );

    UpdateProgressBar( 100., userData );

    if ( userData->m_RegParameters.m_FlagDoNonRigidReg ) 
      userData->m_ProgressBarOffset = 50.;

    userData->m_RegParameters.m_AladinParameters.outputResultName = QString( nameOfResultImage.c_str() ); 
    userData->m_RegParameters.m_AladinParameters.outputResultFlag = true;

    sourceName = userData->m_RegParameters.m_AladinParameters.outputResultName;
  }


  // Create and run the F3D registration
  
  if ( userData->m_RegParameters.m_FlagDoNonRigidReg ) 
  {
    
    userData->m_RegNonRigid = 
      userData->m_RegParameters.CreateNonRigidRegistrationObject( mitkSourceImage, 
								  mitkTargetImage, 
								  mitkTargetMaskImage );  
    
    userData->m_RegNonRigid->SetProgressCallbackFunction( &UpdateProgressBar, 
							  userData );

    userData->m_RegNonRigid->Run_f3d();

    mitkTransformedImage = ConvertNiftiImageToMitk( userData->m_RegNonRigid->GetWarpedImage()[0] );

    // Add this result to the data manager
    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();

    std::string nameOfResultImage( "NonRigidRegnOf_" );
    nameOfResultImage.append( nodeSource->GetName() );
    nameOfResultImage.append( "_To_" );
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkTransformedImage );

    userData->GetDataStorage()->Add( resultNode );


    // Create VTK polydata to illustrate the deformation field

    nifti_image *controlPointGrid =
      userData->m_RegNonRigid->GetControlPointPositionImage();

    CreateControlPointVisualisation( controlPointGrid );
    CreateVectorFieldVisualisation( controlPointGrid );

    reg_bspline_refineControlPointGrid( userData->m_RegParameters.m_ReferenceImage,
					controlPointGrid );

    nifti_image *referenceImage = niftk::AllocateReferenceImageGivenControlPointGrid( controlPointGrid );

    nifti_image *deformationFieldImage = niftk::AllocateDeformationGivenReferenceImage( referenceImage );
    reg_getDeformationFromDisplacement( deformationFieldImage );

    reg_spline_getDeformationField( controlPointGrid, referenceImage, deformationFieldImage,
				    NULL, true, true );

    CreateDeformationVisualisationSurface( niftk::PLANE_XY, deformationFieldImage, 1, 1, 2 );
    CreateDeformationVisualisationSurface( niftk::PLANE_XZ, deformationFieldImage, 1, 2, 1 );
    CreateDeformationVisualisationSurface( niftk::PLANE_YZ, deformationFieldImage, 2, 1, 1 );

    if ( controlPointGrid      != NULL ) nifti_image_free( controlPointGrid );
    if ( referenceImage        != NULL ) nifti_image_free( referenceImage );
    if ( deformationFieldImage != NULL ) nifti_image_free( deformationFieldImage );


    UpdateProgressBar( 100., userData );
  }


  userData->m_Modified = false;
  userData->m_Controls.m_ExecutePushButton->setEnabled( false );
}


// ---------------------------------------------------------------------------
// CreateControlPointVisualisation();
// --------------------------------------------------------------------------- 

void RegistrationExecution::CreateControlPointVisualisation( nifti_image *controlPointGrid )
{
  if ( ! userData->m_RegNonRigid )
  {
    QMessageBox msgBox;
    msgBox.setText("No registration data to create VTK deformation visualisation.");
    msgBox.exec();
    
    return;
  }

  QString sourceName = userData->m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = userData->m_Controls.m_TargetImageComboBox->currentText();

  int nControlPoints = controlPointGrid->nx*controlPointGrid->ny*controlPointGrid->nz;

  std::cout << "Number of control points: " 
	    << nControlPoints << std::endl
	    << "Control point grid dimensions: " 
	    << controlPointGrid->nx << " x " 
	    << controlPointGrid->ny << " x " 
	    << controlPointGrid->nz << std::endl
	    << "Control point grid spacing: " 
	    << controlPointGrid->dx << " x " 
	    << controlPointGrid->dy << " x " 
	    << controlPointGrid->dz << std::endl;


  vtkSmartPointer<vtkPolyData> vtkControlPoints = vtkSmartPointer<vtkPolyData>::New();
  
  vtkControlPoints = niftk::F3DControlGridToVTKPolyDataPoints( controlPointGrid );
  
  mitk::Surface::Pointer mitkControlPoints = mitk::Surface::New();

  mitkControlPoints->SetVtkPolyData( vtkControlPoints );

  mitk::DataNode::Pointer mitkControlPointsNode = mitk::DataNode::New();

  std::string nameOfControlPoints( "ControlPointsFor_" );
  nameOfControlPoints.append( sourceName.toStdString() );
  nameOfControlPoints.append( "_To_" );
  nameOfControlPoints.append( targetName.toStdString() );
  
  mitkControlPointsNode->SetProperty("name", mitk::StringProperty::New(nameOfControlPoints) );

  mitkControlPointsNode->SetData( mitkControlPoints );
  mitkControlPointsNode->SetColor( 1., 0.808, 0.220 );

  userData->GetDataStorage()->Add( mitkControlPointsNode );
}


// ---------------------------------------------------------------------------
// CreateControlPointSphereVisualisation();
// --------------------------------------------------------------------------- 

void RegistrationExecution::CreateControlPointSphereVisualisation( nifti_image *controlPointGrid )
{
  if ( ! userData->m_RegNonRigid )
  {
    QMessageBox msgBox;
    msgBox.setText("No registration data to create VTK deformation visualisation.");
    msgBox.exec();
    
    return;
  }

  QString sourceName = userData->m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = userData->m_Controls.m_TargetImageComboBox->currentText();


 // Get the target image
  // ~~~~~~~~~~~~~~~~~~~~

  float radius = 1.;
  nifti_image *referenceImage = userData->m_RegParameters.m_ReferenceImage;

  if ( referenceImage )
    radius = vcl_sqrt( referenceImage->dx*referenceImage->dx +
                       referenceImage->dy*referenceImage->dy +
                       referenceImage->dz*referenceImage->dz );


  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

  polyData = niftk::F3DControlGridToVTKPolyDataSpheres( controlPointGrid, radius );

  mitk::Surface::Pointer mitkControlPoints = mitk::Surface::New();

  mitkControlPoints->SetVtkPolyData( polyData );

  mitk::DataNode::Pointer mitkControlPointsNode = mitk::DataNode::New();

  std::string nameOfControlPoints( "ControlPointSpheresFor_" );

  nameOfControlPoints.append( sourceName.toStdString() );
  nameOfControlPoints.append( "_To_" );
  nameOfControlPoints.append( targetName.toStdString() );
  
  mitkControlPointsNode->SetProperty("name", mitk::StringProperty::New(nameOfControlPoints) );

  mitkControlPointsNode->SetData( mitkControlPoints );
  mitkControlPointsNode->SetColor( 1., 0.808, 0.220 );

  userData->GetDataStorage()->Add( mitkControlPointsNode );
}


// ---------------------------------------------------------------------------
// CreateVectorFieldVisualisation();
// --------------------------------------------------------------------------- 

void RegistrationExecution::CreateVectorFieldVisualisation( nifti_image *controlPointGrid )
{
  if ( ! userData->m_RegNonRigid )
  {
    QMessageBox msgBox;
    msgBox.setText("No registration data to create VTK deformation visualisation.");
    msgBox.exec();
    
    return;
  }

  QString sourceName = userData->m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = userData->m_Controls.m_TargetImageComboBox->currentText();

  vtkSmartPointer<vtkPolyData> vtkVectorField = vtkSmartPointer<vtkPolyData>::New();
  
  vtkVectorField = niftk::F3DControlGridToVTKPolyDataVectorField( controlPointGrid, 1, 1, 1 );
  
  mitk::Surface::Pointer mitkVectorField = mitk::Surface::New();

  mitkVectorField->SetVtkPolyData( vtkVectorField );

  mitk::DataNode::Pointer mitkVectorFieldNode = mitk::DataNode::New();

  std::string nameOfVectorField( "VectorFieldFor_" );
  nameOfVectorField.append( sourceName.toStdString() );
  nameOfVectorField.append( "_To_" );
  nameOfVectorField.append( targetName.toStdString() );
  
  mitkVectorFieldNode->SetProperty("name", mitk::StringProperty::New(nameOfVectorField) );

  mitkVectorFieldNode->SetData( mitkVectorField );
  mitkVectorFieldNode->SetColor( 1., 0.808, 0.220 );

  userData->GetDataStorage()->Add( mitkVectorFieldNode );
}


// ---------------------------------------------------------------------------
// CreateDeformationVisualisationSurface();
// --------------------------------------------------------------------------- 

void RegistrationExecution::CreateDeformationVisualisationSurface( niftk::PlaneType plane,
								   nifti_image *controlPointGrid,
								   int xSkip,
								   int ySkip,
								   int zSkip )
{
  if ( ! userData->m_RegNonRigid )
  {
    QMessageBox msgBox;
    msgBox.setText("No registration data to create VTK deformation visualisation.");
    msgBox.exec();
    
    return;
  }

  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

  polyData = niftk::F3DDeformationToVTKPolyDataSurface( plane, controlPointGrid, 
							xSkip, ySkip, zSkip);


  QString sourceName = userData->m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = userData->m_Controls.m_TargetImageComboBox->currentText();

  // Add the deformation plane to the DataManager
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  float rgb[3];
  std::string nameOfDeformation;

  mitk::DataNode::Pointer mitkDeformationNode = mitk::DataNode::New();
  mitk::DataNode::Pointer mitkControlPointsNode = mitk::DataNode::New();

  switch (plane )
  {
  case niftk::PLANE_XY:
  {
    rgb[0] = 0.5; 		// Set colour to dark red
    rgb[1] = 0.247;
    rgb[2] = 0.247;

    nameOfDeformation = std::string( "DeformationXY_" );    

    break;
  }

  case niftk::PLANE_XZ:
  {
    rgb[0] = 0.247;		// Set colour to dark green
    rgb[1] = 0.247;
    rgb[2] = 0.5;

    nameOfDeformation = std::string( "DeformationXZ_" );    
    
    break;
  }

  case niftk::PLANE_YZ:
  {
    rgb[0] = 0.247;		// Set colour to dark blue
    rgb[1] = 0.5;
    rgb[2] = 0.247;

    nameOfDeformation = std::string( "DeformationYZ_" );    

    break;
  }
  }

  mitk::Surface::Pointer mitkDeformation = mitk::Surface::New();

  mitkDeformation->SetVtkPolyData( polyData );

  nameOfDeformation.append( sourceName.toStdString() );
  nameOfDeformation.append( "_To_" );
  nameOfDeformation.append( targetName.toStdString() );
  
  mitkDeformationNode->SetProperty("name", mitk::StringProperty::New( nameOfDeformation ) );
  mitkDeformationNode->SetColor( rgb );

  mitkDeformationNode->SetData( mitkDeformation );

  userData->GetDataStorage()->Add( mitkDeformationNode );
}
