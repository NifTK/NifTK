/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <QTimer>

#include "RegistrationExecution.h"

#include "mitkImageToNifti.h"
#include "niftiImageToMitk.h"


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

  if ( userData->m_RegParameters.m_FlagDoInitialRigidReg && userData->m_RegParameters.m_FlagDoNonRigidReg ) 
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

    mitkSourceImage = ConvertNiftiImageToMitk( userData->m_RegAladin->GetFinalWarpedImage() );

    // Add this result to the data manager
    mitk::DataNode::Pointer resultNode = mitk::DataNode::New();

    std::string nameOfResultImage;
    if ( userData->m_RegParameters.m_AladinParameters.regnType == RIGID_ONLY )
      nameOfResultImage = "rigid registration to ";
    else
      nameOfResultImage = "affine_registration_to_";
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkSourceImage );

    userData->GetDataStorage()->Add( resultNode, nodeSource );

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

    std::string nameOfResultImage( "non-rigid_registration_to_" );
    nameOfResultImage.append( nodeTarget->GetName() );

    resultNode->SetProperty("name", mitk::StringProperty::New(nameOfResultImage) );
    resultNode->SetData( mitkTransformedImage );

    userData->GetDataStorage()->Add( resultNode, nodeSource );

    UpdateProgressBar( 100., userData );
   }


  userData->m_Modified = false;
  userData->m_Controls.m_ExecutePushButton->setEnabled( false );
}
