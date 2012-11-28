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
    CreateDeformationVisualisationSurface( controlPointGrid, PLANE_XY );
    //CreateDeformationVisualisationSurface( controlPointGrid, PLANE_YZ );
    //CreateDeformationVisualisationSurface( controlPointGrid, PLANE_XZ );

    if ( controlPointGrid != NULL )
      nifti_image_free( controlPointGrid );

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

  int x, y, z, index;
  float xInit, yInit, zInit;

  PrecisionTYPE *controlPointPtrX, *controlPointPtrY, *controlPointPtrZ;

  mat44 *splineMatrix;

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


  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkIdType id;

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

  vtkSmartPointer<vtkPolyData> vtkControlPoints = vtkSmartPointer<vtkPolyData>::New();
  
  controlPointPtrX = static_cast< PrecisionTYPE * >( controlPointGrid->data );
  controlPointPtrY = &controlPointPtrX[ nControlPoints ];
  controlPointPtrZ = &controlPointPtrY[ nControlPoints ];

  if ( controlPointGrid->sform_code > 0 ) 
    splineMatrix = &( controlPointGrid->sto_xyz );
  else 
    splineMatrix = &( controlPointGrid->qto_xyz );

  for (z=0; z<controlPointGrid->nz; z++)
  {
    index = z*controlPointGrid->nx*controlPointGrid->ny;

    for (y=0; y<controlPointGrid->ny; y++)
    {
      for (x=0; x<controlPointGrid->nx; x++)
      {
	  
	// The initial control point position
	xInit =
	  splineMatrix->m[0][0]*static_cast<float>(x) +
	  splineMatrix->m[0][1]*static_cast<float>(y) +
	  splineMatrix->m[0][2]*static_cast<float>(z) +
	  splineMatrix->m[0][3];
	yInit =
	  splineMatrix->m[1][0]*static_cast<float>(x) +
	  splineMatrix->m[1][1]*static_cast<float>(y) +
	  splineMatrix->m[1][2]*static_cast<float>(z) +
	  splineMatrix->m[1][3];
	zInit =
	  splineMatrix->m[2][0]*static_cast<float>(x) +
	  splineMatrix->m[2][1]*static_cast<float>(y) +
	  splineMatrix->m[2][2]*static_cast<float>(z) +
	  splineMatrix->m[2][3];

	// The final control point position:
	//    controlPointPtrX[index], controlPointPtrY[index], controlPointPtrZ[index];

#if 1
	id = points->InsertNextPoint( -controlPointPtrX[index], 
				      -controlPointPtrY[index],
				       controlPointPtrZ[index] );


#else
	id = points->InsertNextPoint( -xInit, -yInit, zInit );
#endif 
	cells->InsertNextCell( 1 );
	cells->InsertCellPoint( id );

	index++;
      }
    }
  }

  vtkControlPoints->SetPoints( points );
  vtkControlPoints->SetVerts( cells );

  vtkControlPoints->Print( std::cout );
  
  // Add the control points to the DataManager

  mitk::Surface::Pointer mitkControlPoints = mitk::Surface::New();

  mitkControlPoints->SetVtkPolyData( vtkControlPoints );

  mitk::DataNode::Pointer mitkControlPointsNode = mitk::DataNode::New();

  std::string nameOfControlPoints( "ControlPointsFor_" );
  nameOfControlPoints.append( sourceName.toStdString() );
  nameOfControlPoints.append( "_To_" );
  nameOfControlPoints.append( targetName.toStdString() );
  
  mitkControlPointsNode->SetProperty("name", mitk::StringProperty::New(nameOfControlPoints) );

  mitkControlPointsNode->SetData( mitkControlPoints );

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

  int x, y, z, index;
  float xInit, yInit, zInit;

  PrecisionTYPE *controlPointPtrX, *controlPointPtrY, *controlPointPtrZ;

  mat44 *splineMatrix;

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

  vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

  vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();

  sphere->SetThetaResolution(10);
  sphere->SetPhiResolution(5);
  sphere->SetRadius( radius );


  controlPointPtrX = static_cast< PrecisionTYPE * >( controlPointGrid->data );
  controlPointPtrY = &controlPointPtrX[ nControlPoints ];
  controlPointPtrZ = &controlPointPtrY[ nControlPoints ];

  if ( controlPointGrid->sform_code > 0 ) 
    splineMatrix = &( controlPointGrid->sto_xyz );
  else 
    splineMatrix = &( controlPointGrid->qto_xyz );

  for (z=0; z<controlPointGrid->nz; z++)
  {
    index = z*controlPointGrid->nx*controlPointGrid->ny;

    for (y=0; y<controlPointGrid->ny; y++)
    {
      for (x=0; x<controlPointGrid->nx; x++)
      {
	  
	// The initial control point position
	xInit =
	  splineMatrix->m[0][0]*static_cast<float>(x) +
	  splineMatrix->m[0][1]*static_cast<float>(y) +
	  splineMatrix->m[0][2]*static_cast<float>(z) +
	  splineMatrix->m[0][3];
	yInit =
	  splineMatrix->m[1][0]*static_cast<float>(x) +
	  splineMatrix->m[1][1]*static_cast<float>(y) +
	  splineMatrix->m[1][2]*static_cast<float>(z) +
	  splineMatrix->m[1][3];
	zInit =
	  splineMatrix->m[2][0]*static_cast<float>(x) +
	  splineMatrix->m[2][1]*static_cast<float>(y) +
	  splineMatrix->m[2][2]*static_cast<float>(z) +
	  splineMatrix->m[2][3];

	// The final control point position:
	//    controlPointPtrX[index], controlPointPtrY[index], controlPointPtrZ[index];

	sphere->SetCenter( -controlPointPtrX[index], 
			   -controlPointPtrY[index],
			   controlPointPtrZ[index] );

	sphere->Update();

	vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
	polydataCopy->DeepCopy( sphere->GetOutput() );
	
	appendFilter->AddInput( polydataCopy );
	appendFilter->Update();

	index++;
      }
    }
  }

  // Add the control points to the DataManager

  mitk::Surface::Pointer mitkControlPoints = mitk::Surface::New();

  mitkControlPoints->SetVtkPolyData( appendFilter->GetOutput() );

  mitk::DataNode::Pointer mitkControlPointsNode = mitk::DataNode::New();

  std::string nameOfControlPoints( "ControlPointSpheresFor_" );

  nameOfControlPoints.append( sourceName.toStdString() );
  nameOfControlPoints.append( "_To_" );
  nameOfControlPoints.append( targetName.toStdString() );
  
  mitkControlPointsNode->SetProperty("name", mitk::StringProperty::New(nameOfControlPoints) );

  mitkControlPointsNode->SetData( mitkControlPoints );

  userData->GetDataStorage()->Add( mitkControlPointsNode );
}


// ---------------------------------------------------------------------------
// reg_bspline_refineControlPointGrid2D();
// --------------------------------------------------------------------------- 

void reg_bspline_refineControlPointGrid2D(  nifti_image *targetImage,
                                          nifti_image *splineControlPoint)
{
    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
    if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[1]=splineControlPoint->dim[1];
    oldDim[2]=splineControlPoint->dim[2];
    oldDim[3]=1;

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = 1.0f;

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)reg_floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+5;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)reg_floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+5;
    //    splineControlPoint->dim[1]=splineControlPoint->nx=(int)reg_ceil(targetImage->nx*targetImage->dx/splineControlPoint->dx)+4;
    //    splineControlPoint->dim[2]=splineControlPoint->ny=(int)reg_ceil(targetImage->ny*targetImage->dy/splineControlPoint->dy)+4;
    splineControlPoint->dim[3]=1;

    splineControlPoint->nvox=splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz*splineControlPoint->nt*splineControlPoint->nu;
    splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);

    gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]];

    for(int y=0; y<oldDim[2]; y++){
        int Y=2*y-1;
        if(Y<splineControlPoint->ny){
            for(int x=0; x<oldDim[1]; x++){
                int X=2*x-1;
                if(X<splineControlPoint->nx){

                    /* X Axis */
                    // 0 0
                    SetValue(gridPtrX, splineControlPoint->dim, X, Y, 0,
                             (GetValue(oldGridPtrX,oldDim,x-1,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrX,oldDim,x-1,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                              + 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
                                        GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) )
                              + 36.0f * GetValue(oldGridPtrX,oldDim,x,y,0) ) / 64.0f);
                    // 1 0
                    SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, 0,
                             (GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) ) ) / 16.0f);
                    // 0 1
                    SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, 0,
                             (GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x-1,y+1,0) +
                              GetValue(oldGridPtrX,oldDim,x+1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) ) ) / 16.0f);
                    // 1 1
                    SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, 0,
                             (GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
                              GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0) ) / 4.0f);

                    /* Y Axis */
                    // 0 0
                    SetValue(gridPtrY, splineControlPoint->dim, X, Y, 0,
                             (GetValue(oldGridPtrY,oldDim,x-1,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrY,oldDim,x-1,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                              + 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
                                        GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) )
                              + 36.0f * GetValue(oldGridPtrY,oldDim,x,y,0) ) / 64.0f);
                    // 1 0
                    SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, 0,
                             (GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
                              GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) ) ) / 16.0f);
                    // 0 1
                    SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, 0,
                             (GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x-1,y+1,0) +
                              GetValue(oldGridPtrY,oldDim,x+1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                              + 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) ) ) / 16.0f);
                    // 1 1
                    SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, 0,
                             (GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
                              GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0) ) / 4.0f);

                }
            }
        }
    }

    free(oldGrid);
}


// ---------------------------------------------------------------------------
// reg_bspline_refineControlPointGrid3D();
// --------------------------------------------------------------------------- 

void reg_bspline_refineControlPointGrid3D(nifti_image *targetImage,
                                          nifti_image *splineControlPoint)
{

    // The input grid is first saved
    SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
    SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
    if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
    int oldDim[4];
    oldDim[0]=splineControlPoint->dim[0];
    oldDim[1]=splineControlPoint->dim[1];
    oldDim[2]=splineControlPoint->dim[2];
    oldDim[3]=splineControlPoint->dim[3];

    splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
    splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
    splineControlPoint->dz = splineControlPoint->pixdim[3] = splineControlPoint->dz / 2.0f;

    //    splineControlPoint->dim[1]=splineControlPoint->nx=(int)reg_ceil(targetImage->nx*targetImage->dx/splineControlPoint->dx)+4;
    //    splineControlPoint->dim[2]=splineControlPoint->ny=(int)reg_ceil(targetImage->ny*targetImage->dy/splineControlPoint->dy)+4;
    //    splineControlPoint->dim[3]=splineControlPoint->nz=(int)reg_ceil(targetImage->nz*targetImage->dz/splineControlPoint->dz)+4;

    splineControlPoint->dim[1]=splineControlPoint->nx=(int)reg_floor(targetImage->nx*targetImage->dx/splineControlPoint->dx)+5;
    splineControlPoint->dim[2]=splineControlPoint->ny=(int)reg_floor(targetImage->ny*targetImage->dy/splineControlPoint->dy)+5;
    splineControlPoint->dim[3]=splineControlPoint->nz=(int)reg_floor(targetImage->nz*targetImage->dz/splineControlPoint->dz)+5;

    splineControlPoint->nvox=splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz*splineControlPoint->nt*splineControlPoint->nu;
    splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);
    
    gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
    SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *gridPtrZ = &gridPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
    SplineTYPE *oldGridPtrX = &oldGrid[0];
    SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]*oldDim[3]];
    SplineTYPE *oldGridPtrZ = &oldGridPtrY[oldDim[1]*oldDim[2]*oldDim[3]];


    for(int z=0; z<oldDim[3]; z++){
        int Z=2*z-1;
        if(Z<splineControlPoint->nz){
            for(int y=0; y<oldDim[2]; y++){
                int Y=2*y-1;
                if(Y<splineControlPoint->ny){
                    for(int x=0; x<oldDim[1]; x++){
                        int X=2*x-1;
                        if(X<splineControlPoint->nx){

                            /* X Axis */
                            // 0 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrX,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1)+
                                      GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)
                                      + 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                                GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                                GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                                GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                                GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) )
                                      + 36.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                                 GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                                 GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) )
                                      + 216.0f * GetValue(oldGridPtrX,oldDim,x,y,z) ) / 512.0f);

                            // 1 0 0
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z,
                                     ( GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1)) +
                                      36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z)) ) / 128.0f);

                            // 0 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z,
                                     ( GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrX,oldDim,x-1,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z)) ) / 128.0f);

                            // 1 1 0
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z,
                                     (GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) ) ) / 32.0f);

                            // 0 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z+1,
                                     ( GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                              GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y,z+1)) ) / 128.0f);

                            // 1 0 1
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z+1,
                                     (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) ) ) / 32.0f);

                            // 0 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z+1,
                                     (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x+1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) ) ) / 32.0f);

                            // 1 1 1
                            SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z+1,
                                     (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)) / 8.0f);
                            

                            /* Y Axis */
                            // 0 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrY,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1)+
                                      GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)
                                      + 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                                GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                                GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                                GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                                GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) )
                                      + 36.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                                 GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                                 GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) )
                                      + 216.0f * GetValue(oldGridPtrY,oldDim,x,y,z) ) / 512.0f);

                            // 1 0 0
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z,
                                     ( GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1)) +
                                      36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z)) ) / 128.0f);

                            // 0 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z,
                                     ( GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrY,oldDim,x-1,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z)) ) / 128.0f);

                            // 1 1 0
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z,
                                     (GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) ) ) / 32.0f);

                            // 0 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z+1,
                                     ( GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                              GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y,z+1)) ) / 128.0f);

                            // 1 0 1
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z+1,
                                     (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) ) ) / 32.0f);

                            // 0 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z+1,
                                     (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x+1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) ) ) / 32.0f);

                            // 1 1 1
                            SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z+1,
                                     (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)) / 8.0f);

                            /* Z Axis */
                            // 0 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z,
                                     (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1)+
                                      GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)
                                      + 6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                                GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                                GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                                GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                                GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) )
                                      + 36.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                                 GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                                 GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) )
                                      + 216.0f * GetValue(oldGridPtrZ,oldDim,x,y,z) ) / 512.0f);
                            
                            // 1 0 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z,
                                     ( GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1)) +
                                      36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z)) ) / 128.0f);
                            
                            // 0 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z,
                                     ( GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                              GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z)) ) / 128.0f);
                            
                            // 1 1 0
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z,
                                     (GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) ) ) / 32.0f);
                            
                            // 0 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z+1,
                                     ( GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                              GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                      36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y,z+1)) ) / 128.0f);
                            
                            // 1 0 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z+1,
                                     (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) ) ) / 32.0f);
                            
                            // 0 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z+1,
                                     (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                      6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                              GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) ) ) / 32.0f);
                            
                            // 1 1 1
                            SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z+1,
                                     (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                      GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                      GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)) / 8.0f);
                        }
                    }
                }
            }
        }
    }

    free(oldGrid);
}
/* *************************************************************** */
extern "C++"
void reg_bspline_refineControlPointGrid(nifti_image *referenceImage,
                                        nifti_image *controlPointGrid)
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] Starting the refine the control point grid\n");
#endif
    if(controlPointGrid->nz==1){
        switch(controlPointGrid->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_refineControlPointGrid2D<float>(referenceImage,controlPointGrid);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_refineControlPointGrid2D<double>(referenceImage,controlPointGrid);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
            exit(1);
        }
    }else{
        switch(controlPointGrid->datatype){
        case NIFTI_TYPE_FLOAT32:
            reg_bspline_refineControlPointGrid3D<float>(referenceImage,controlPointGrid);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_bspline_refineControlPointGrid3D<double>(referenceImage,controlPointGrid);
            break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] Only single or double precision is implemented for the bending energy gradient\n");
            fprintf(stderr,"[NiftyReg ERROR] The bending energy gradient has not computed\n");
            exit(1);
        }
    }
    // Compute the new control point header
    // The qform (and sform) are set for the control point position image
    controlPointGrid->quatern_b=referenceImage->quatern_b;
    controlPointGrid->quatern_c=referenceImage->quatern_c;
    controlPointGrid->quatern_d=referenceImage->quatern_d;
    controlPointGrid->qoffset_x=referenceImage->qoffset_x;
    controlPointGrid->qoffset_y=referenceImage->qoffset_y;
    controlPointGrid->qoffset_z=referenceImage->qoffset_z;
    controlPointGrid->qfac=referenceImage->qfac;
    controlPointGrid->qto_xyz = nifti_quatern_to_mat44(controlPointGrid->quatern_b,
                                                       controlPointGrid->quatern_c,
                                                       controlPointGrid->quatern_d,
                                                       controlPointGrid->qoffset_x,
                                                       controlPointGrid->qoffset_y,
                                                       controlPointGrid->qoffset_z,
                                                       controlPointGrid->dx,
                                                       controlPointGrid->dy,
                                                       controlPointGrid->dz,
                                                       controlPointGrid->qfac);

    // Origin is shifted from 1 control point in the qform
    float originIndex[3];
    float originReal[3];
    originIndex[0] = -1.0f;
    originIndex[1] = -1.0f;
    originIndex[2] = 0.0f;
    if(referenceImage->nz>1) originIndex[2] = -1.0f;
    reg_mat44_mul(&(controlPointGrid->qto_xyz), originIndex, originReal);
    if(controlPointGrid->qform_code==0) controlPointGrid->qform_code=1;
    controlPointGrid->qto_xyz.m[0][3] = controlPointGrid->qoffset_x = originReal[0];
    controlPointGrid->qto_xyz.m[1][3] = controlPointGrid->qoffset_y = originReal[1];
    controlPointGrid->qto_xyz.m[2][3] = controlPointGrid->qoffset_z = originReal[2];

    controlPointGrid->qto_ijk = nifti_mat44_inverse(controlPointGrid->qto_xyz);

    if(controlPointGrid->sform_code>0){
        float scalingRatio[3];
        scalingRatio[0]= controlPointGrid->dx / referenceImage->dx;
        scalingRatio[1]= controlPointGrid->dy / referenceImage->dy;
        scalingRatio[2]= controlPointGrid->dz / referenceImage->dz;

        controlPointGrid->sto_xyz.m[0][0]=referenceImage->sto_xyz.m[0][0] * scalingRatio[0];
        controlPointGrid->sto_xyz.m[1][0]=referenceImage->sto_xyz.m[1][0] * scalingRatio[0];
        controlPointGrid->sto_xyz.m[2][0]=referenceImage->sto_xyz.m[2][0] * scalingRatio[0];
        controlPointGrid->sto_xyz.m[3][0]=0.f;
        controlPointGrid->sto_xyz.m[0][1]=referenceImage->sto_xyz.m[0][1] * scalingRatio[1];
        controlPointGrid->sto_xyz.m[1][1]=referenceImage->sto_xyz.m[1][1] * scalingRatio[1];
        controlPointGrid->sto_xyz.m[2][1]=referenceImage->sto_xyz.m[2][1] * scalingRatio[1];
        controlPointGrid->sto_xyz.m[3][1]=0.f;
        controlPointGrid->sto_xyz.m[0][2]=referenceImage->sto_xyz.m[0][2] * scalingRatio[2];
        controlPointGrid->sto_xyz.m[1][2]=referenceImage->sto_xyz.m[1][2] * scalingRatio[2];
        controlPointGrid->sto_xyz.m[2][2]=referenceImage->sto_xyz.m[2][2] * scalingRatio[2];
        controlPointGrid->sto_xyz.m[3][2]=0.f;
        controlPointGrid->sto_xyz.m[0][3]=referenceImage->sto_xyz.m[0][3];
        controlPointGrid->sto_xyz.m[1][3]=referenceImage->sto_xyz.m[1][3];
        controlPointGrid->sto_xyz.m[2][3]=referenceImage->sto_xyz.m[2][3];
        controlPointGrid->sto_xyz.m[3][3]=1.f;

        // The origin is shifted by one compare to the reference image
        float originIndex[3];originIndex[0]=originIndex[1]=originIndex[2]=-1;
        if(referenceImage->nz<=1) originIndex[2]=0;
        reg_mat44_mul(&(controlPointGrid->sto_xyz), originIndex, originReal);
        controlPointGrid->sto_xyz.m[0][3] = originReal[0];
        controlPointGrid->sto_xyz.m[1][3] = originReal[1];
        controlPointGrid->sto_xyz.m[2][3] = originReal[2];
        controlPointGrid->sto_ijk = nifti_mat44_inverse(controlPointGrid->sto_xyz);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] The control point grid has been refined\n");
#endif
    return;
}


// ---------------------------------------------------------------------------
// CreateDeformationVisualisationSurface();
// --------------------------------------------------------------------------- 

void RegistrationExecution::CreateDeformationVisualisationSurface( nifti_image *controlPointGrid,
								   PlaneType plane )
{
  if ( ! userData->m_RegNonRigid )
  {
    QMessageBox msgBox;
    msgBox.setText("No registration data to create VTK deformation visualisation.");
    msgBox.exec();
    
    return;
  }

  int i, j, k;
  int x, y, z, index;

  PrecisionTYPE *deformationFieldPtrX, *deformationFieldPtrY, *deformationFieldPtrZ;

  nifti_image *deformationField;
  nifti_image *roi;

  QString sourceName = userData->m_Controls.m_SourceImageComboBox->currentText();
  QString targetName = userData->m_Controls.m_TargetImageComboBox->currentText();

  nifti_image *referenceImage = userData->m_RegParameters.m_ReferenceImage;

  std::cout << "Control point grid: " << std::endl;
  nifti_image_infodump( controlPointGrid );
  reg_io_WriteImageFile(controlPointGrid, "controlPointGrid.nii.gz" );

  std::cout << "Reference image: " << std::endl;
  nifti_image_infodump( referenceImage );


  // Define the region of interest
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  float xDownSample = 1.;
  float yDownSample = 1.;
  float zDownSample = 1.;

#if 0
  roi = nifti_copy_nim_info( referenceImage );

  switch (plane )
  {
  case PLANE_XY: 
  {
    roi->dim[3]    = roi->nz = controlPointGrid->nz;
    roi->pixdim[3] = roi->dz = controlPointGrid->dz;

    zDownSample = roi->dz / referenceImage->dz;

    break;
  }

  case PLANE_YZ: 
  {
    roi->dim[1]    = roi->nx = controlPointGrid->nx;
    roi->pixdim[1] = roi->dx = controlPointGrid->dx;
    
    xDownSample = roi->dx / referenceImage->dx;

    break;
  }

  case PLANE_XZ: 
  {
    roi->dim[2]    = roi->ny = controlPointGrid->ny;
    roi->pixdim[2] = roi->dy = controlPointGrid->dy;
    
    yDownSample = roi->dy / referenceImage->dy;

    break;
  }

  }
#else
  roi = nifti_copy_nim_info( controlPointGrid );

  roi->dim[0] = roi->ndim = 3;

  roi->dim[4]    = roi->nt = 1;
  roi->pixdim[4] = roi->dt = 1.0;

  roi->dim[5]    = roi->nu = 1;
  roi->pixdim[5] = roi->du = 1.0;

  roi->dim[6]    = roi->nv = 1;
  roi->pixdim[6] = roi->dv = 1.0;
  
  roi->dim[7]    = roi->nw = 1;
  roi->pixdim[7] = roi->dw = 1.0;

  roi->intent_code = 0;
  roi->intent_name[0] = '\0';
#endif

  // update the qform matrix
  roi->qto_xyz = nifti_quatern_to_mat44(roi->quatern_b,
					roi->quatern_c,
					roi->quatern_d,
					roi->qoffset_x,
					roi->qoffset_y,
					roi->qoffset_z,
					roi->dx,
					roi->dy,
					roi->dz,
					roi->qfac);

  roi->qto_ijk = nifti_mat44_inverse(roi->qto_xyz);

  // update the sform matrix
  roi->sto_xyz.m[0][0] *= xDownSample;
  roi->sto_xyz.m[1][0] *= xDownSample;
  roi->sto_xyz.m[2][0] *= xDownSample;

  roi->sto_xyz.m[0][1] *= yDownSample;
  roi->sto_xyz.m[1][1] *= yDownSample;
  roi->sto_xyz.m[2][1] *= yDownSample;

  roi->sto_xyz.m[0][2] *= zDownSample;
  roi->sto_xyz.m[1][2] *= zDownSample;
  roi->sto_xyz.m[2][2] *= zDownSample;

  float origin_sform[3] = { roi->sto_xyz.m[0][3], 
			    roi->sto_xyz.m[1][3], 
			    roi->sto_xyz.m[2][3] };

  roi->sto_xyz.m[0][3] = origin_sform[0];
  roi->sto_xyz.m[1][3] = origin_sform[1];
  roi->sto_xyz.m[2][3] = origin_sform[2];

  roi->sto_ijk = nifti_mat44_inverse(roi->sto_xyz);


  roi->nvox = roi->dim[1] * roi->dim[2] * roi->dim[3] * roi->dim[4];

  roi->data = (void *)calloc(roi->nvox, roi->nbyper);

  std::cout << "ROI: " << std::endl;
  nifti_image_infodump( roi );
  reg_io_WriteImageFile(roi, "roi.nii.gz" );


  // Compute the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  deformationField = nifti_copy_nim_info( roi );

  deformationField->dim[0] = deformationField->ndim = 5;
  deformationField->dim[1] = deformationField->nx = roi->nx;
  deformationField->dim[2] = deformationField->ny = roi->ny;
  deformationField->dim[3] = deformationField->nz = roi->nz;
  deformationField->dim[4] = deformationField->nt = 1;

  deformationField->pixdim[4] = deformationField->dt = 1.0;

  if ( roi->nz > 1) 
    deformationField->dim[5] = deformationField->nu = 3;
  else 
    deformationField->dim[5] = deformationField->nu = 2;

  deformationField->pixdim[5] = deformationField->du = 1.0;

  deformationField->dim[6]    = deformationField->nv = 1;
  deformationField->pixdim[6] = deformationField->dv = 1.0;
  
  deformationField->dim[7]    = deformationField->nw = 1;
  deformationField->pixdim[7] = deformationField->dw = 1.0;

  deformationField->nvox = 
    deformationField->nx*
    deformationField->ny*
    deformationField->nz*
    deformationField->nt*
    deformationField->nu;

  if (sizeof(PrecisionTYPE) == 8) 
    deformationField->datatype = NIFTI_TYPE_FLOAT64;
  else 
    deformationField->datatype = NIFTI_TYPE_FLOAT32;
    
  deformationField->nbyper = sizeof(PrecisionTYPE);

  deformationField->intent_code = NIFTI_INTENT_VECTOR;

  deformationField->data = (void *) calloc( deformationField->nvox, 
					    deformationField->nbyper );

  reg_spline_getDeformationField( controlPointGrid,
				  roi,
				  deformationField,
				  NULL, false, true );

  std::cout << "Deformation field: " << std::endl;
  nifti_image_infodump( deformationField );
  reg_io_WriteImageFile(deformationField, "deformationField.nii.gz" );


  // Create a VTK polydata object and add everything to it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkIdType id;

  vtkStructuredGrid *sgrid = vtkStructuredGrid::New();
  sgrid->SetDimensions(deformationField->nx, deformationField->ny, deformationField->nz);

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkPolyData> vtkControlPoints = vtkSmartPointer<vtkPolyData>::New();

  int nPoints = deformationField->nx*deformationField->ny*deformationField->nz;

  deformationFieldPtrX = static_cast< PrecisionTYPE * >( deformationField->data );
  deformationFieldPtrY = &deformationFieldPtrX[ nPoints ];
  deformationFieldPtrZ = &deformationFieldPtrY[ nPoints ];

  for (z=0; z<deformationField->nz; z++)
  {
    index = z*deformationField->nx*deformationField->ny;

    for (y=0; y<deformationField->ny; y++)
    {
      for (x=0; x<deformationField->nx; x++)
      {

	// The final control point position:
	//    deformationFieldPtrX[index], deformationFieldPtrY[index], deformationFieldPtrZ[index];

	id = points->InsertNextPoint( -deformationFieldPtrX[index], 
				      -deformationFieldPtrY[index],
				       deformationFieldPtrZ[index] );

	cells->InsertNextCell( 1 );
	cells->InsertCellPoint( id );

	index++;
      }
    }
  }

  sgrid->SetPoints( points );
  
  vtkControlPoints->SetPoints( points );
  vtkControlPoints->SetVerts( cells );


  // Add the deformation planes to the DataManager
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

  vtkSmartPointer<vtkStructuredGridGeometryFilter> structuredGridFilter = vtkSmartPointer<vtkStructuredGridGeometryFilter>::New();

  structuredGridFilter->SetInput( sgrid );

  std::string nameOfDeformation;

  switch (plane )
  {
  case PLANE_XY:
  {
    nameOfDeformation = std::string( "DeformationXY_" );    

    for (k=0; k<deformationField->nz; k++) 
    {
      vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
      
      structuredGridFilter->SetExtent( 0, deformationField->nx, 
				       0, deformationField->ny, 
				       k, k );
      structuredGridFilter->Update();
      polydataCopy->DeepCopy( structuredGridFilter->GetOutput() );
      
      appendFilter->AddInput( polydataCopy );
      appendFilter->Update();
    }
    
    break;
  }

  case PLANE_YZ:
  {
    nameOfDeformation = std::string( "DeformationYZ_" );    

    for (j=0; j<deformationField->ny; j++) 
    {
      vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
      
      structuredGridFilter->SetExtent( 0, deformationField->nx, 
				       j, j,
				       0, deformationField->nz );
      structuredGridFilter->Update();
      polydataCopy->DeepCopy( structuredGridFilter->GetOutput() );
      
      appendFilter->AddInput( polydataCopy );
      appendFilter->Update();
    }
    
    break;
  }

  case PLANE_XZ:
  {
    nameOfDeformation = std::string( "DeformationXZ_" );    

    for (i=0; i<deformationField->nx; i++) 
    {
      vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
      
      structuredGridFilter->SetExtent( i, i,
				       0, deformationField->ny, 
				       0, deformationField->nz );
      structuredGridFilter->Update();
      polydataCopy->DeepCopy( structuredGridFilter->GetOutput() );
      
      appendFilter->AddInput( polydataCopy );
      appendFilter->Update();
    }
  }
  }

  mitk::Surface::Pointer mitkDeformation = mitk::Surface::New();

  mitkDeformation->SetVtkPolyData( appendFilter->GetOutput() );

  mitk::DataNode::Pointer mitkDeformationNode = mitk::DataNode::New();

  nameOfDeformation.append( sourceName.toStdString() );
  nameOfDeformation.append( "_To_" );
  nameOfDeformation.append( targetName.toStdString() );
  
  mitkDeformationNode->SetProperty("name", mitk::StringProperty::New( nameOfDeformation ) );

  mitkDeformationNode->SetData( mitkDeformation );

  userData->GetDataStorage()->Add( mitkDeformationNode );
  

  // Add the deformation points to the DataManager

  mitk::Surface::Pointer mitkControlPoints = mitk::Surface::New();

  mitkControlPoints->SetVtkPolyData( vtkControlPoints );

  mitk::DataNode::Pointer mitkControlPointsNode = mitk::DataNode::New();

  std::string nameOfControlPoints( "DeformationPointsFor_" );
  nameOfControlPoints.append( sourceName.toStdString() );
  nameOfControlPoints.append( "_To_" );
  nameOfControlPoints.append( targetName.toStdString() );
  
  mitkControlPointsNode->SetProperty("name", mitk::StringProperty::New(nameOfControlPoints) );

  mitkControlPointsNode->SetData( mitkControlPoints );

  userData->GetDataStorage()->Add( mitkControlPointsNode );


  if ( roi != NULL )
    nifti_image_free( roi );

  if ( deformationField != NULL )
    nifti_image_free( deformationField );
}
