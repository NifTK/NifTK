/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "SurfaceReconView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "SurfaceReconViewActivator.h"
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkNodePredicateDataType.h>
#include <QFileDialog>
#include <mitkCoordinateAxesData.h>
#include "SurfaceReconViewPreferencePage.h"
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <QtConcurrentRun>
#include <boost/bind.hpp>


const std::string SurfaceReconView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacerecon";


//-----------------------------------------------------------------------------
SurfaceReconView::SurfaceReconView()
{
  m_SurfaceReconstruction = niftk::SurfaceReconstruction::New();

  bool ok = false;
  ok = connect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
}


//-----------------------------------------------------------------------------
SurfaceReconView::~SurfaceReconView()
{
  bool ok = false;
  ok = disconnect(DoItButton, SIGNAL(clicked()), this, SLOT(DoSurfaceReconstruction()));
  assert(ok);

  ok = disconnect(LeftIntrinsicBrowseButton, SIGNAL(clicked()), this, SLOT(LeftBrowseButtonClicked()));
  assert(ok);
  ok = disconnect(RightIntrinsicBrowseButton, SIGNAL(clicked()), this, SLOT(RightBrowseButtonClicked()));
  assert(ok);
  ok = disconnect(StereoRigTransformBrowseButton, SIGNAL(clicked()), this, SLOT(StereoRigBrowseButtonClicked()));
  assert(ok);

  ok = disconnect(LeftChannelNodeNameComboBox,  SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)));
  assert(ok);
  ok = disconnect(RightChannelNodeNameComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)));
  assert(ok);

  // wait for it to finish first and then disconnect?
  // or the other way around?
  // i'd say disconnect first then wait because at that time we no longer care about the result
  // and the finished-handler might access some half-destroyed objects.
  ok = disconnect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
  m_BackgroundProcessWatcher.waitForFinished();
}


//-----------------------------------------------------------------------------
std::string SurfaceReconView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void SurfaceReconView::LeftBrowseButtonClicked()
{
  // FIXME: this blocks timer delivery?
  QString   file = QFileDialog::getOpenFileName(GetParent(), "Intrinsic Camera Calibration", m_LastFile);
  if (!file.isEmpty())
  {
    LeftIntrinsicPathLineEdit->setText(file);
    m_LastFile = file;
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::RightBrowseButtonClicked()
{
  // FIXME: this blocks timer delivery?
  QString   file = QFileDialog::getOpenFileName(GetParent(), "Intrinsic Camera Calibration", m_LastFile);
  if (!file.isEmpty())
  {
    RightIntrinsicPathLineEdit->setText(file);
    m_LastFile = file;
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::StereoRigBrowseButtonClicked()
{
  // FIXME: this blocks timer delivery?
  QString   file = QFileDialog::getOpenFileName(GetParent(), "Stereo Rig Calibration", m_LastFile);
  if (!file.isEmpty())
  {
    StereoRigTransformationPathLineEdit->setText(file);
    m_LastFile = file;
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::CreateQtPartControl( QWidget *parent )
{
  setupUi(parent);
  bool ok = false;
  ok = connect(DoItButton, SIGNAL(clicked()), this, SLOT(DoSurfaceReconstruction()));
  assert(ok);

  ok = connect(LeftIntrinsicBrowseButton, SIGNAL(clicked()), this, SLOT(LeftBrowseButtonClicked()));
  assert(ok);
  ok = connect(RightIntrinsicBrowseButton, SIGNAL(clicked()), this, SLOT(RightBrowseButtonClicked()));
  assert(ok);
  ok = connect(StereoRigTransformBrowseButton, SIGNAL(clicked()), this, SLOT(StereoRigBrowseButtonClicked()));
  assert(ok);

  ok = connect(LeftChannelNodeNameComboBox,  SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)), Qt::QueuedConnection);
  assert(ok);
  ok = connect(RightChannelNodeNameComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexChanged(int)), Qt::QueuedConnection);
  assert(ok);

  ctkServiceReference ref = mitk::SurfaceReconViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::SurfaceReconViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
  this->RetrievePreferenceValues();

  this->LeftChannelNodeNameComboBox->SetDataStorage(this->GetDataStorage());
  this->RightChannelNodeNameComboBox->SetDataStorage(this->GetDataStorage());
  CameraNodeComboBox->SetDataStorage(GetDataStorage());

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  this->LeftChannelNodeNameComboBox->SetAutoSelectNewItems(false);
  this->LeftChannelNodeNameComboBox->SetPredicate(isImage);
  this->RightChannelNodeNameComboBox->SetAutoSelectNewItems(false);
  this->RightChannelNodeNameComboBox->SetPredicate(isImage);
  mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isCoords = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
  CameraNodeComboBox->SetAutoSelectNewItems(false);
  CameraNodeComboBox->SetPredicate(isCoords);

  UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
  berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(SurfaceReconViewPreferencePage::s_PrefsNodeName)).Cast<berry::IBerryPreferences>();
  assert(prefs);

  m_MaxTriangulationErrorThresholdSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultTriangulationErrorPrefsName, 0.1f));

  m_MinDepthRangeSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultMinDepthRangePrefsName, 1.0f));
  m_MaxDepthRangeSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultMaxDepthRangePrefsName, 1000.0f));

  bool  useUndistortDefaultPath = prefs->GetBool(SurfaceReconViewPreferencePage::s_UseUndistortionDefaultPathPrefsName, true);
  if (useUndistortDefaultPath)
  {
    // FIXME: hard-coded prefs node names, etc.
    //        how to access header files in another plugin?
    //        see https://cmicdev.cs.ucl.ac.uk/trac/ticket/2505
    berry::IBerryPreferences::Pointer undistortPrefs = (prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.igiundistort")).Cast<berry::IBerryPreferences>();
    if (undistortPrefs.IsNotNull())
    {
      m_LastFile = QString::fromStdString(undistortPrefs->Get("default calib file path", ""));
    }
  }
  else
  {
    m_LastFile = QString::fromStdString(prefs->Get(SurfaceReconViewPreferencePage::s_DefaultCalibrationFilePathPrefsName, ""));
  }

}


//-----------------------------------------------------------------------------
void SurfaceReconView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnUpdate(const ctkEvent& event)
{
  // Optional. This gets called everytime the data sources are updated.
  // If the surface reconstruction was as fast as the GUI update, we could trigger it here.

  // we call this all the time to update the has-calib-property for the node comboboxes.
  UpdateNodeNameComboBox();

  if (m_AutomaticUpdateRadioButton->isChecked())
  {
    if (!m_BackgroundProcess.isRunning())
    {
      DoSurfaceReconstruction();
    }
  }
}


//-----------------------------------------------------------------------------
template <typename T>
static bool HasCalibProp(const typename T::Pointer& n)
{
  mitk::BaseProperty::Pointer  bp = n->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
  if (bp.IsNull())
  {
    return false;
  }
  return true;
}


//-----------------------------------------------------------------------------
static bool NeedsToLoadCalib(const QString& filename, const mitk::Image::Pointer& image)
{
  bool  needs2load = false;
  // filename overrides any existing properties
  if (!filename.isEmpty())
  {
    needs2load = true;
  }
  else
  {
    // no filename? check if there's a suitable property.
    // if not then invent some stuff.
    if (HasCalibProp<mitk::Image>(image))
    {
      needs2load = true;
    }
  }
  return needs2load;
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnComboBoxIndexChanged(int index)
{
  UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::UpdateNodeNameComboBox()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    QString leftText  = LeftChannelNodeNameComboBox->currentText();
    QString rightText = RightChannelNodeNameComboBox->currentText();

    mitk::DataNode::Pointer   leftNode  = storage->GetNamedNode(leftText.toStdString());
    mitk::DataNode::Pointer   rightNode = storage->GetNamedNode(rightText.toStdString());

    // either node or attached image has to have calibration property
    if (leftNode.IsNotNull())
    {
      bool    leftHasProp = HasCalibProp<mitk::DataNode>(leftNode);
      if (!leftHasProp)
      {
        // note: our comboboxes should have nodes only with image data!
        mitk::Image::Pointer img = dynamic_cast<mitk::Image*>(leftNode->GetData());
        assert(img.IsNotNull());
        leftHasProp = HasCalibProp<mitk::Image>(img);
      }

      if (leftHasProp)
      {
        LeftChannelNodeNameComboBox->lineEdit()->setStyleSheet("background-color: rgb(200, 255, 200);");
      }
      else
      {
        LeftChannelNodeNameComboBox->lineEdit()->setStyleSheet("background-color: rgb(255, 200, 200);");
      }
    }

    if (rightNode.IsNotNull())
    {
      bool    rightHasProp = HasCalibProp<mitk::DataNode>(rightNode);
      if (!rightNode)
      {
        // note: our comboboxes should have nodes only with image data!
        mitk::Image::Pointer img = dynamic_cast<mitk::Image*>(rightNode->GetData());
        assert(img.IsNotNull());
        rightHasProp = HasCalibProp<mitk::Image>(img);
      }

      if (rightHasProp)
      {
        RightChannelNodeNameComboBox->lineEdit()->setStyleSheet("background-color: rgb(200, 255, 200);");
      }
      else
      {
        RightChannelNodeNameComboBox->lineEdit()->setStyleSheet("background-color: rgb(255, 200, 200);");
      }
    }
  }
}


//-----------------------------------------------------------------------------
template <typename PropType>
static void CopyProp(const mitk::DataNode::Pointer source, mitk::Image::Pointer target, const char* name)
{
  mitk::BaseProperty::Pointer baseProp = target->GetProperty(name);
  if (baseProp.IsNull())
  {
    // none there yet, try to pull it from the datanode
    baseProp = source->GetProperty(name);
    if (baseProp.IsNotNull())
    {
      // check that it's the correct type
      typename PropType::Pointer   prop = dynamic_cast<PropType*>(baseProp.GetPointer());
      if (prop.IsNotNull())
      {
        // FIXME: copy? or simply ref the same object?
        target->SetProperty(name, prop);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::CopyImagePropsIfNecessary(const mitk::DataNode::Pointer source, mitk::Image::Pointer target)
{
  // we copy known meta-data properties to the images, but only if they dont exist yet.
  // we'll not ever change the value of an existing property!

  // calibration data
  CopyProp<mitk::CameraIntrinsicsProperty>(source, target, niftk::Undistortion::s_CameraCalibrationPropertyName);

  // has the image been rectified?
  CopyProp<mitk::BoolProperty>(source, target, niftk::SurfaceReconstruction::s_ImageIsRectifiedPropertyName);

  // has the image been undistorted?
  CopyProp<mitk::BoolProperty>(source, target, niftk::Undistortion::s_ImageIsUndistortedPropertyName);
}


//-----------------------------------------------------------------------------
void SurfaceReconView::LoadStereoRig(const std::string& filename, mitk::Image::Pointer img)
{
  assert(img.IsNotNull());

  itk::Matrix<float, 4, 4>    txf;
  txf.SetIdentity();

  if (!filename.empty())
  {
    std::ifstream   file(filename.c_str());
    if (!file.good())
    {
      throw std::runtime_error("Cannot open stereo-rig file " + filename);
    }
    float   values[3 * 4];
    for (unsigned int i = 0; i < (sizeof(values) / sizeof(values[0])); ++i)
    {
      if (!file.good())
      {
        throw std::runtime_error("Cannot read enough data from stereo-rig file " + filename);
      }
      file >> values[i];
    }
    file.close();

    // set rotation
    for (int i = 0; i < 9; ++i)
    {
      txf.GetVnlMatrix()(i / 3, i % 3) = values[i];
    }

    // set translation
    for (int i = 0; i < 3; ++i)
    {
      txf.GetVnlMatrix()(i, 3) = values[9 + i];
    }
  }
  else
  {
    // no idea what to invent here...
  }

  niftk::MatrixProperty::Pointer  prop = niftk::MatrixProperty::New(txf);
  img->SetProperty(niftk::SurfaceReconstruction::s_StereoRigTransformationPropertyName, prop);
}


//-----------------------------------------------------------------------------
void SurfaceReconView::DoSurfaceReconstruction()
{
  // buttons and other stuff should have been disabled to prevent this function from being called
  // whenever we are already running one instance in the background.
  assert(!m_BackgroundProcess.isRunning());

  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    std::string leftText  = LeftChannelNodeNameComboBox->currentText().toStdString();
    std::string rightText = RightChannelNodeNameComboBox->currentText().toStdString();

    const mitk::DataNode::Pointer leftNode  = storage->GetNamedNode(leftText);
    const mitk::DataNode::Pointer rightNode = storage->GetNamedNode(rightText);

    if (leftNode.IsNotNull() && rightNode.IsNotNull())
    {
      mitk::BaseData::Pointer leftData  = leftNode->GetData();
      mitk::BaseData::Pointer rightData = rightNode->GetData();

      if (leftData.IsNotNull() && rightData.IsNotNull())
      {
        mitk::Image::Pointer leftImage  = dynamic_cast<mitk::Image*>(leftData.GetPointer());
        mitk::Image::Pointer rightImage = dynamic_cast<mitk::Image*>(rightData.GetPointer());

        if (leftImage.IsNotNull() && rightImage.IsNotNull())
        {
          // if our output node exists already then we recycle it, of course.
          // it may not be tagged as "derived" from the correct source nodes
          // but that shouldn't be a problem here.

          std::string outputName = OutputNodeNameLineEdit->text().toStdString();
          m_BackgroundOutputNode = storage->GetNamedNode(outputName);
          if (m_BackgroundOutputNode.IsNull())
          {
            m_BackgroundOutputNode = mitk::DataNode::New();
            m_BackgroundOutputNode->SetName(outputName);

            mitk::DataStorage::SetOfObjects::Pointer   nodeParents = mitk::DataStorage::SetOfObjects::New();
            nodeParents->push_back(leftNode);
            nodeParents->push_back(rightNode);

            storage->Add(m_BackgroundOutputNode, nodeParents);
          }


          bool    needToLoadLeftCalib  = NeedsToLoadCalib(LeftIntrinsicPathLineEdit->text(),  leftImage);
          bool    needToLoadRightCalib = NeedsToLoadCalib(RightIntrinsicPathLineEdit->text(), rightImage);

          if (needToLoadLeftCalib)
          {
            niftk::Undistortion::LoadCalibration(LeftIntrinsicPathLineEdit->text().toStdString(), leftImage);
          }
          if (needToLoadRightCalib)
          {
            niftk::Undistortion::LoadCalibration(RightIntrinsicPathLineEdit->text().toStdString(), rightImage);
          }
          LoadStereoRig(StereoRigTransformationPathLineEdit->text().toStdString(), rightImage);

          CopyImagePropsIfNecessary(leftNode,  leftImage);
          CopyImagePropsIfNecessary(rightNode, rightImage);

          niftk::SurfaceReconstruction::OutputType  outputtype = niftk::SurfaceReconstruction::POINT_CLOUD;
          if (GenerateDisparityImageRadioBox->isChecked())
          {
            assert(!GeneratePointCloudRadioBox->isChecked());
            outputtype = niftk::SurfaceReconstruction::DISPARITY_IMAGE;
          }
          if (GeneratePointCloudRadioBox->isChecked())
          {
            assert(!GenerateDisparityImageRadioBox->isChecked());
            outputtype = niftk::SurfaceReconstruction::POINT_CLOUD;
          }

          // where to place the point cloud in 3d space
          mitk::DataNode::Pointer camNode;
          std::string             camNodeName  = CameraNodeComboBox->currentText().toStdString();
          if (!camNodeName.empty())
          {
            // is ok if node doesnt exist, SurfaceReconstruction will deal with that.
            camNode = storage->GetNamedNode(camNodeName);
          }

          niftk::SurfaceReconstruction::Method  method = (niftk::SurfaceReconstruction::Method) MethodComboBox->currentIndex();

          float maxTriError = (float) m_MaxTriangulationErrorThresholdSpinBox->value();
          float minDepth    = (float) m_MinDepthRangeSpinBox->value();
          float maxDepth    = (float) m_MaxDepthRangeSpinBox->value();

          try
          {
            // dont allow clicking on it until we are done with the current one.
            DoItButton->setEnabled(false);

            niftk::SurfaceReconstruction::ParamPacket   params;
            params.image1 = leftImage;
            params.image2 = rightImage;
            params.method = method;
            params.outputtype = outputtype;
            params.camnode = camNode;
            params.maxTriangulationError = maxTriError;
            params.minDepth = minDepth;
            params.maxDepth = maxDepth;

            m_BackgroundProcess = QtConcurrent::run(m_SurfaceReconstruction.GetPointer(), &niftk::SurfaceReconstruction::Run, params);
            m_BackgroundProcessWatcher.setFuture(m_BackgroundProcess);
            // Then delagate everything to class outside of plugin, so we can unit test it.
            //m_SurfaceReconstruction->Run(storage, outputNode, leftImage, rightImage, method, outputtype, camNode, maxTriError, minDepth, maxDepth);
          }
          catch (const std::exception& e)
          {
            std::cerr << "Whoops... something went wrong with surface reconstruction: " << e.what() << std::endl;
            // FIXME: show an error message on the plugin panel somewhere?
          }
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnBackgroundProcessFinished()
{
  m_BackgroundOutputNode->SetData(m_BackgroundProcessWatcher.result());

  DoItButton->setEnabled(true);
}

