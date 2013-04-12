/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "TagTrackerView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <mitkImage.h>
#include <mitkNodePredicateDataType.h>
#include <mitkPointSet.h>
#include "TagTrackerViewActivator.h"
#include "TagTrackerViewPreferencePage.h"
#include "mitkMonoTagExtractor.h"
#include "mitkStereoTagExtractor.h"

const std::string TagTrackerView::VIEW_ID = "uk.ac.ucl.cmic.igitagtracker";
const std::string TagTrackerView::NODE_ID = "Tag Locations";

//-----------------------------------------------------------------------------
TagTrackerView::TagTrackerView()
: m_Controls(NULL)
, m_LeftNode(NULL)
, m_RightNode(NULL)
, m_LeftIntrinsicFileName("")
, m_RightIntrinsicFileName("")
, m_RightToLeftRotationFileName("")
, m_RightToLeftTranslationFileName("")
, m_LeftIntrinsicMatrix(NULL)
, m_RightIntrinsicMatrix(NULL)
, m_RightToLeftRotationVector(NULL)
, m_RightToLeftTranslationVector(NULL)
, m_ListenToEventBusPulse(true)
, m_MinSize(0.01)
, m_MaxSize(0.0125)
{
}


//-----------------------------------------------------------------------------
TagTrackerView::~TagTrackerView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }

  if (m_LeftIntrinsicMatrix != NULL)
  {
    cvReleaseMat(&m_LeftIntrinsicMatrix);
  }
  if (m_RightIntrinsicMatrix != NULL)
  {
    cvReleaseMat(&m_RightIntrinsicMatrix);
  }
  if (m_RightToLeftRotationVector != NULL)
  {
    cvReleaseMat(&m_RightToLeftRotationVector);
  }
  if (m_RightToLeftTranslationVector != NULL)
  {
    cvReleaseMat(&m_RightToLeftTranslationVector);
  }
}


//-----------------------------------------------------------------------------
std::string TagTrackerView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void TagTrackerView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    mitk::TNodePredicateDataType<mitk::Image>::Pointer leftIsImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    mitk::TNodePredicateDataType<mitk::Image>::Pointer rightIsImage = mitk::TNodePredicateDataType<mitk::Image>::New();

    m_Controls = new Ui::TagTrackerViewControls();
    m_Controls->setupUi(parent);
    m_Controls->m_LeftComboBox->SetDataStorage(dataStorage);
    m_Controls->m_LeftComboBox->SetAutoSelectNewItems(false);
    m_Controls->m_LeftComboBox->SetPredicate(leftIsImage);
    m_Controls->m_RightComboBox->SetDataStorage(dataStorage);
    m_Controls->m_RightComboBox->SetAutoSelectNewItems(false);
    m_Controls->m_RightComboBox->SetPredicate(rightIsImage);

    connect(m_Controls->m_UpdateButton, SIGNAL(pressed()), this, SLOT(OnManualUpdate()));

    ctkServiceReference ref = mitk::TagTrackerViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::TagTrackerViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
    }
    this->RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TagTrackerView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    m_ListenToEventBusPulse = prefs->GetBool(TagTrackerViewPreferencePage::LISTEN_TO_EVENT_BUS_NAME, TagTrackerViewPreferencePage::LISTEN_TO_EVENT_BUS);
    m_MinSize = static_cast<float>(prefs->GetDouble(TagTrackerViewPreferencePage::MIN_SIZE_NAME, TagTrackerViewPreferencePage::MIN_SIZE));
    m_MaxSize = static_cast<float>(prefs->GetDouble(TagTrackerViewPreferencePage::MAX_SIZE_NAME, TagTrackerViewPreferencePage::MAX_SIZE));
  }

  if (m_ListenToEventBusPulse)
  {
    m_Controls->m_UpdateButton->setEnabled(false);
  }
  else
  {
    m_Controls->m_UpdateButton->setEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void TagTrackerView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void TagTrackerView::LoadMatrix(const QString& fileName, CvMat *matrixToWriteTo)
{
  if (matrixToWriteTo != NULL)
  {
    cvReleaseMat(&matrixToWriteTo);
  }

  matrixToWriteTo = (CvMat*)cvLoad(fileName.toStdString().c_str());
  if (matrixToWriteTo == NULL)
  {
    std::string message = std::string("Failed to load matrix ") + fileName.toStdString();
    throw std::logic_error(message);
  }
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnUpdate(const ctkEvent& event)
{
  if (m_ListenToEventBusPulse)
  {
    this->UpdateTags();
  }
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnManualUpdate()
{
  this->UpdateTags();
}


//-----------------------------------------------------------------------------
void TagTrackerView::UpdateTags()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  assert(dataStorage);

  if (m_LeftNode.IsNotNull() || m_RightNode.IsNotNull())
  {
    // Make sure all specified matrices are loaded.
    if (m_LeftIntrinsicFileName.size() > 0 && m_LeftIntrinsicMatrix == NULL)
    {
      this->LoadMatrix(m_LeftIntrinsicFileName, m_LeftIntrinsicMatrix);
    }
    if (m_RightIntrinsicFileName.size() > 0 && m_RightIntrinsicMatrix == NULL)
    {
      this->LoadMatrix(m_RightIntrinsicFileName, m_RightIntrinsicMatrix);
    }
    if (m_RightToLeftRotationFileName.size() > 0 && m_RightToLeftRotationVector == NULL)
    {
      this->LoadMatrix(m_RightToLeftRotationFileName, m_RightToLeftTranslationVector);
    }
    if (m_RightToLeftTranslationFileName.size() > 0 && m_RightToLeftTranslationVector == NULL)
    {
      this->LoadMatrix(m_RightToLeftTranslationFileName, m_RightToLeftTranslationVector);
    }

    // Retrieve the node from data storage, or create it if it does not exist.
    mitk::PointSet::Pointer pointSet;
    mitk::DataNode::Pointer pointSetNode = dataStorage->GetNamedNode(NODE_ID);

    if (pointSetNode.IsNull())
    {
      pointSet = mitk::PointSet::New();
      pointSetNode = mitk::DataNode::New();
      pointSetNode->SetData( pointSet );
      pointSetNode->SetProperty( "name", mitk::StringProperty::New(NODE_ID));
      pointSetNode->SetProperty( "opacity", mitk::FloatProperty::New(1));
      pointSetNode->SetProperty( "point line width", mitk::IntProperty::New(1));
      pointSetNode->SetProperty( "point 2D size", mitk::IntProperty::New(5));
      pointSetNode->SetBoolProperty("helper object", false);
      pointSetNode->SetBoolProperty("show distant lines", false);
      pointSetNode->SetBoolProperty("show distant points", false);
      pointSetNode->SetBoolProperty("show distances", false);
      pointSetNode->SetProperty("layer", mitk::IntProperty::New(99));
      pointSetNode->SetColor( 1.0, 0, 0 );
      dataStorage->Add(pointSetNode);
    }
    else
    {
      pointSet = static_cast<mitk::PointSet*>(pointSetNode->GetData());
      if (pointSet.IsNull())
      {
        // Give up, as the node has the wrong data.
        MITK_ERROR << "TagTrackerView::OnUpdate node " << NODE_ID << " does not contain an mitk::PointSet" << std::endl;
        return;
      }
    }

    // Now use the data to extract points, and update the point set.

    if ((m_LeftNode.IsNotNull() && m_RightNode.IsNull())
        || (m_LeftNode.IsNull() && m_RightNode.IsNotNull())
        )
    {
      mitk::Image::Pointer image;
      CvMat *intrinsics;

      if (m_LeftNode.IsNotNull())
      {
        image = static_cast<mitk::Image*>(m_LeftNode->GetData());
        intrinsics = m_LeftIntrinsicMatrix;
      }
      else
      {
        image = static_cast<mitk::Image*>(m_RightNode->GetData());
        intrinsics = m_RightIntrinsicMatrix;
      }

      if (image.IsNull())
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, mono case, image node is NULL" << std::endl;
        return;
      }
      if (intrinsics == NULL)
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, mono case, camera intrinsic matrix is NULL" << std::endl;
        return;
      }

      // Mono Case.
      mitk::MonoTagExtractor::Pointer extractor = mitk::MonoTagExtractor::New();
      extractor->ExtractPoints(
          image,
          m_MinSize,
          m_MaxSize,
          *intrinsics,
          pointSet
          );
    }
    else
    {
      mitk::Image::Pointer leftImage = static_cast<mitk::Image*>(m_LeftNode->GetData());
      mitk::Image::Pointer rightImage = static_cast<mitk::Image*>(m_RightNode->GetData());

      if (leftImage.IsNull())
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, left image is NULL" << std::endl;
        return;
      }
      if (rightImage.IsNull())
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, right image is NULL" << std::endl;
        return;
      }
      if (m_LeftIntrinsicMatrix == NULL)
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, left camera intrinsic matrix is NULL" << std::endl;
        return;
      }
      if (m_RightIntrinsicMatrix == NULL)
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, right camera intrinsic matrix is NULL" << std::endl;
        return;
      }
      if (m_RightToLeftRotationVector == NULL)
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, right to left rotation vector is NULL" << std::endl;
        return;
      }
      if (m_RightToLeftTranslationVector == NULL)
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, right to left translation vector is NULL" << std::endl;
        return;
      }

      // Stereo Case.
      mitk::StereoTagExtractor::Pointer extractor = mitk::StereoTagExtractor::New();
      extractor->ExtractPoints(
          leftImage,
          rightImage,
          m_MinSize,
          m_MaxSize,
          *m_LeftIntrinsicMatrix,
          *m_RightIntrinsicMatrix,
          *m_RightToLeftRotationVector,
          *m_RightToLeftTranslationVector,
          pointSet
          );
    } // end if mono/stereo

    int numberOfTrackedPoints = pointSet->GetSize();
    m_Controls->m_NumberOfTagsLabel->setText(QString("tags:") + QString(numberOfTrackedPoints));

  } // end if we have at least one node specified
}
