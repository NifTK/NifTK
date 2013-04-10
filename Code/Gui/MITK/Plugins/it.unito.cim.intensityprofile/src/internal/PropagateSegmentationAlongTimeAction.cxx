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

#include "PropagateSegmentationAlongTimeAction.h"

#include "mitkRenderingManager.h"
#include "mitkBaseRenderer.h"
#include "mitkImage.h"
#include "mitkImageDataItem.h"
#include "mitkSliceNavigationController.h"

#include <mitkImageAccessByItk.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageLinearIteratorWithIndex.h>

#include <QMessageBox>

void PropagateAlongTime(itk::Image<unsigned char, 4>* itkImage, unsigned timeStep) {
	typedef itk::Image<unsigned char, 4> Image4D;

	Image4D::RegionType wholeRegion = itkImage->GetLargestPossibleRegion();

	Image4D::RegionType selectedTimeStepRegion = wholeRegion;
	selectedTimeStepRegion.SetIndex(3, timeStep);
	selectedTimeStepRegion.SetSize(3, 1);

	itk::ImageRegionConstIterator<Image4D> it3D(itkImage, selectedTimeStepRegion);
	itk::ImageLinearIteratorWithIndex<Image4D> it4D(itkImage, wholeRegion);
	it4D.SetDirection(3);
	it3D.GoToBegin();
	it4D.GoToBegin();
	while (!it3D.IsAtEnd()) {
		unsigned char maskValue = it3D.Get();
		if (maskValue) {
			it4D.GoToBeginOfLine();
			while (!it4D.IsAtEndOfLine()) {
				it4D.Set(maskValue);
				++it4D;
			}
		}
		++it3D;
		it4D.NextLine();
	}
}

PropagateSegmentationAlongTimeAction::PropagateSegmentationAlongTimeAction()
{ 
}

PropagateSegmentationAlongTimeAction::~PropagateSegmentationAlongTimeAction()
{
}

void PropagateSegmentationAlongTimeAction::Run(const QList<mitk::DataNode::Pointer>& selectedNodes)
{
	mitk::DataNode* segmentationNode = selectedNodes.at(0);

	mitk::Image* segmentationImage =
			dynamic_cast<mitk::Image*>(segmentationNode->GetData());

	mitk::SliceNavigationController* timeNavigationController =
			mitk::RenderingManager::GetInstance()->GetTimeNavigationController();
	mitk::Stepper* timeStepper = timeNavigationController->GetTime();
	unsigned timeStep = timeStepper->GetPos();

	try {
	  AccessFixedTypeByItk_1(segmentationImage, PropagateAlongTime, (unsigned char), (4), timeStep);
	}
	catch (mitk::AccessByItkException& exception) {
	  QMessageBox msgBox;

    msgBox.setWindowTitle("Error");
    msgBox.setText(tr("Error occurred during propagating the segmentation\n"
            "for the other time steps."));
    msgBox.setDetailedText(tr(exception.what()));
    msgBox.exec();
	}
}

void PropagateSegmentationAlongTimeAction::SetDataStorage(mitk::DataStorage* dataStorage) {
	m_DataStorage = dataStorage;
}

void PropagateSegmentationAlongTimeAction::SetStdMultiWidget(QmitkStdMultiWidget *)
{
  // not needed
}

void PropagateSegmentationAlongTimeAction::SetSmoothed(bool /*smoothed*/)
{
  //not needed
}

void PropagateSegmentationAlongTimeAction::SetDecimated(bool /*decimated*/)
{
  //not needed
}

void PropagateSegmentationAlongTimeAction::SetFunctionality(berry::QtViewPart* /*functionality*/)
{
  //not needed
}
