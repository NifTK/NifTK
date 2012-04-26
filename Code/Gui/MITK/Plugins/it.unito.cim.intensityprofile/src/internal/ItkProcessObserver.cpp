/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ItkProcessObserver.h"

#include <mitkProgressBar.h>
#include <mitkStatusBar.h>

#include <itkCommand.h>
#include <itkProcessObject.h>

ItkProcessObserver::ItkProcessObserver(itk::ProcessObject* itkProcess, const char* statusBarMessage) {
	m_ItkProcess = itkProcess;
	m_StatusBarMessage = statusBarMessage;

	// Not a nice thing to initialize static data members from constructors,
	// but this way we can be sure that these functions return a valid value.
	// (They are initialized at that time.)
	m_StatusBar = mitk::StatusBar::GetInstance();
	m_ProgressBar = mitk::ProgressBar::GetInstance();

	// This can be an arbitrary number. ITK processes report their progress as
	// a real number between 0 and 1. The MITK progress bar, however, expects integers,
	// so the ITK progress has to be scaled to the [0 ; m_StepsToDo] interval.
	m_StepsToDo = 100;
	m_StepsDone = 0;

	m_ProgressBar->AddStepsToDo(m_StepsToDo);

	typedef itk::SimpleMemberCommand<ItkProcessObserver> MemberCommand;

	MemberCommand::Pointer startCommand = MemberCommand::New();
	startCommand->SetCallbackFunction(this, &ItkProcessObserver::onStartEvent);
	m_ItkProcess->AddObserver(itk::StartEvent(), startCommand);

	MemberCommand::Pointer endCommand = MemberCommand::New();
	endCommand->SetCallbackFunction(this, &ItkProcessObserver::onEndEvent);
	m_ItkProcess->AddObserver(itk::EndEvent(), endCommand);

	MemberCommand::Pointer progressCommand = MemberCommand::New();
	progressCommand->SetCallbackFunction(this, &ItkProcessObserver::onProgressEvent);
	m_ItkProcess->AddObserver(itk::ProgressEvent(), progressCommand);
}

ItkProcessObserver::~ItkProcessObserver() {
}

void
ItkProcessObserver::onStartEvent() {
//	m_ProgressBar->AddStepsToDo(1);
//	m_ProgressBar->Progress();
	m_StatusBar->DisplayText(m_StatusBarMessage);
//	m_StatusBar->DisplayGreyValueText(m_ProcessName.c_str());
}

void
ItkProcessObserver::onEndEvent() {
	m_StatusBar->DisplayText("");
//	m_StatusBar->DisplayGreyValueText("");
	// We make the progress bar reach 100%.
	m_ProgressBar->Progress(m_StepsToDo - m_StepsDone);
}

void
ItkProcessObserver::onProgressEvent() {
	double progress = m_ItkProcess->GetProgress();
	double stepsDone = static_cast<int>(progress * m_StepsToDo);
	if (stepsDone > m_StepsDone) {
		m_ProgressBar->Progress(stepsDone - m_StepsDone);
		m_StepsDone = stepsDone;
	}
}
