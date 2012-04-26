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

#ifndef ITKPROCESSOBSERVER_H_
#define ITKPROCESSOBSERVER_H_

namespace itk {
class ProcessObject;
}

namespace mitk {
class StatusBar;
class ProgressBar;
}

class ItkProcessObserver {

	itk::ProcessObject* m_ItkProcess;
	const char* m_StatusBarMessage;
	int m_StepsToDo;
	int m_StepsDone;

	mitk::StatusBar* m_StatusBar;
	mitk::ProgressBar* m_ProgressBar;

public:
	ItkProcessObserver(itk::ProcessObject* itkProcess, const char* statusBarMessage);
  virtual ~ItkProcessObserver();

private:
	void onStartEvent();
	void onEndEvent();
	void onProgressEvent();

};

#endif /* ITKPROCESSOBSERVER_H_ */
