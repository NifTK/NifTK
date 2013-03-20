/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGITrackerTool.h"
#include <QFile>
#include <mitkDataNode.h>
#include <mitkBaseData.h>
#include <mitkRenderingManager.h>
#include <mitkBaseRenderer.h>

#include "mitkIGITestDataUtils.h"
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"
#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkTransform.h"
#include "vtkMatrix4x4.h"

NIFTK_IGISOURCE_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGITrackerTool, "IGI Tracker Tool");

//-----------------------------------------------------------------------------
QmitkIGITrackerTool::QmitkIGITrackerTool()
: m_UseICP(false)
, m_PointSetsInitialized(false)
, m_LinkCamera(false)
, m_ImageFiducialsDataNode(NULL)
, m_ImageFiducialsPointSet(NULL)
, m_TrackerFiducialsDataNode(NULL)
, m_TrackerFiducialsPointSet(NULL)
, m_FiducialRegistrationFilter(NULL)
, m_PermanentRegistrationFilter(NULL)
, m_focalPoint(-2000.0)
, m_ClipNear(5.0)
, m_ClipFar(6000.0)
, m_TransformTrackerToMITKCoords(false)
{
  m_FiducialRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();
  m_PermanentRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();
  this->InitPreMatrix();
}


//-----------------------------------------------------------------------------
QmitkIGITrackerTool::QmitkIGITrackerTool(NiftyLinkSocketObject * socket)
: QmitkIGINiftyLinkDataSource(socket)
, m_UseICP(false)
, m_PointSetsInitialized(false)
, m_LinkCamera(false)
, m_ImageFiducialsDataNode(NULL)
, m_ImageFiducialsPointSet(NULL)
, m_TrackerFiducialsDataNode(NULL)
, m_TrackerFiducialsPointSet(NULL)
, m_FiducialRegistrationFilter(NULL)
, m_PermanentRegistrationFilter(NULL)
, m_focalPoint(-2000.0)
, m_ClipNear(5.0)
, m_ClipFar(6000.0)
, m_TransformTrackerToMITKCoords(false)
{
  m_FiducialRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();
  m_PermanentRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();
  this->InitPreMatrix();
}


//-----------------------------------------------------------------------------
QmitkIGITrackerTool::~QmitkIGITrackerTool()
{
  this->RemoveFiducialsFromDataStorage();

  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (dataStorage != NULL)
  {
    mitk::DataNode::Pointer toolRep;
    foreach (toolRep, m_ToolRepresentations)
    {
      dataStorage->Remove(toolRep);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::InterpretMessage(NiftyLinkMessage::Pointer msg)
{
  if (msg->GetMessageType() == QString("STRING"))
  {
    QString str = static_cast<NiftyLinkStringMessage::Pointer>(msg)->GetString();
    if (str.isEmpty() || str.isNull())
    {
      return;
    }
    this->ProcessInitString(str);
  }
  else if (msg.data() != NULL &&
      (msg->GetMessageType() == QString("TRANSFORM") || msg->GetMessageType() == QString("TDATA"))
     )
  {
    //Check the tool name
    NiftyLinkTrackingDataMessage::Pointer trMsg;
    trMsg = static_cast<NiftyLinkTrackingDataMessage::Pointer>(msg);

    QString messageToolName = trMsg->GetTrackerToolName();
    QString sourceToolName = QString::fromStdString(this->GetDescription());
    if ( messageToolName == sourceToolName ) 
    {
      QmitkIGINiftyLinkDataType::Pointer wrapper = QmitkIGINiftyLinkDataType::New();
      wrapper->SetMessage(msg.data());
      wrapper->SetDataSource("QmitkIGITrackerTool");
      wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(msg->GetTimeCreated()));
      wrapper->SetDuration(1000000000); // nanoseconds

      this->AddData(wrapper.GetPointer());
      this->SetStatus("Receiving");
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::ProcessInitString(QString str)
{
  QString type = XMLBuilderBase::ParseDescriptorType(str);
  if (type == QString("TrackerClientDescriptor"))
  {
    m_InitString = str;
    ClientDescriptorXMLBuilder* clientInfo = new TrackerClientDescriptor();
    clientInfo->SetXMLString(str);

    if (!clientInfo->IsMessageValid())
    {
      delete clientInfo;
      return;
    }
    //A single source can have multiple tracked tools. 
    QStringList trackerTools = dynamic_cast<TrackerClientDescriptor*>(clientInfo)->GetTrackerTools();
    QString tool;
    this->SetNumberOfTools(trackerTools.length());
    std::list<std::string> StringList;


    foreach (tool , trackerTools)
    {
      std::string String;
      String = tool.toStdString();
      StringList.push_back(String);
      
    }
    if ( StringList.size() > 0 ) 
    {
      this->SetToolStringList(StringList);
    }
    this->ProcessClientInfo(clientInfo);
  }
  else
  {
    // error?
  }
}


//-----------------------------------------------------------------------------
QString QmitkIGITrackerTool::GetInitString()
{
  return m_InitString;
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerTool::CanHandleData(mitk::IGIDataType* data) const
{
  bool canHandle = false;
  std::string className = data->GetNameOfClass();

  if (data != NULL && className == std::string("QmitkIGINiftyLinkDataType"))
  {
    QmitkIGINiftyLinkDataType::Pointer dataType = dynamic_cast<QmitkIGINiftyLinkDataType*>(data);
    if (dataType.IsNotNull())
    {
      NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
      if (pointerToMessage != NULL
          && (pointerToMessage->GetMessageType() == QString("TRANSFORM")
              || pointerToMessage->GetMessageType() == QString("TDATA"))
          )
      {
        canHandle = true;
      }
    }
  }

  return canHandle;
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerTool::Update(mitk::IGIDataType* data)
{
  bool result = false;

  QmitkIGINiftyLinkDataType::Pointer dataType = static_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNotNull())
  {

    NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
    if (pointerToMessage != NULL)
    {
      this->HandleTrackerData(pointerToMessage);
      this->DisplayTrackerData(pointerToMessage);
      result = true;
    }
    else
    {
      MITK_ERROR << "QmitkIGITrackerTool::Update is receiving messages with no data ... this is wrong!" << std::endl;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::HandleTrackerData(NiftyLinkMessage* msg)
{
  if (msg->GetMessageType() == QString("TDATA"))
  {
    NiftyLinkTrackingDataMessage* trMsg;
    trMsg = static_cast<NiftyLinkTrackingDataMessage*>(msg);

    QString toolName = trMsg->GetTrackerToolName();

    mitk::DataStorage* ds = this->GetDataStorage();
    if (ds == NULL)
    {
      QString message("ERROR: QmitkIGITrackerTool, DataStorage Access Error: Could not access DataStorage!");
      emit StatusUpdate(message);
      return;
    }

    float   inputTransformMat[4][4];
    trMsg->GetMatrix(inputTransformMat);

    if ( m_LinkCamera )
    {
      // find the active camera via the rendering manager
      mitk::RenderingManager::RenderWindowVector RenderWindows;
      RenderWindows = mitk::RenderingManager::GetInstance()->GetAllRegisteredRenderWindows();

      int windowsFound = RenderWindows.size() ;
      for ( int i = 0 ; i < windowsFound ; i ++ )
      {
        vtkCamera * Camera;
        vtkRenderWindow * thisWindow;
        thisWindow = RenderWindows.at(i);

        //this performs the important function of rotating the logo, it looks really natty.
        /*
        vtkRendererCollection * Renderers;
        Renderers = thisWindow->GetRenderers();
        vtkRenderer * Renderer = NULL;
        Renderers->InitTraversal();
        for ( int i = 0 ; i <  Renderers->GetNumberOfItems() ; i ++ )
          Renderer = Renderers->GetNextItem();
        Camera = Renderer->GetActiveCamera();
        Camera->Azimuth( 1);*/
        Camera = mitk::BaseRenderer::GetInstance(thisWindow)->GetVtkRenderer()->GetActiveCamera();
        Camera->SetPosition(inputTransformMat[0][3],inputTransformMat[1][3],inputTransformMat[2][3]);
        //manually sort out the focal point, there is presumably an intelegent way to to this
        //camerausertransform seems pretty useless in this application
        float fxi=0;
        float fyi=0;
        float fzi=m_focalPoint; 
        float fx = inputTransformMat[0][0] * fxi + inputTransformMat[0][1] * fyi 
          + inputTransformMat[0][2] * fzi + inputTransformMat[0][3];
        float fy = inputTransformMat[1][0] * fxi + inputTransformMat[1][1] * fyi
          + inputTransformMat[1][2] * fzi + inputTransformMat[1][3];
        float fz = inputTransformMat[2][0] * fxi + inputTransformMat[2][1] * fyi
          + inputTransformMat[2][2] * fzi + inputTransformMat[2][3];
        Camera->SetFocalPoint(fx,fy,fz);
        double vuxi=0;
        double vuyi=1.0e9;
        double vuzi=0;
        double vux = inputTransformMat[0][0] * vuxi + inputTransformMat[0][1] * vuyi 
          + inputTransformMat[0][2] * vuzi + inputTransformMat[0][3];
        double vuy = inputTransformMat[1][0] * vuxi + inputTransformMat[1][1] * vuyi
          + inputTransformMat[1][2] * vuzi + inputTransformMat[1][3];
        double vuz = inputTransformMat[2][0] * vuxi + inputTransformMat[2][1] * vuyi
          + inputTransformMat[2][2] * vuzi + inputTransformMat[2][3];
        Camera->SetViewUp(vux,vuy,vuz);
        Camera->SetClippingRange(m_ClipNear, m_ClipFar);
      }
    }

    mitk::DataNode::Pointer tempNode = ds->GetNamedNode(toolName.toStdString().c_str());

    foreach ( tempNode, m_AssociatedTools.values(toolName))
    {

     if (tempNode.IsNull())
     {
       QString message = QObject::tr("ERROR: QmitkIGITrackerTool, could not find node %1").arg(toolName);
       emit StatusUpdate(message);
       return;
     }
    
     // Get the transform from data
     mitk::BaseData * data = tempNode->GetData();

     if ( m_TransformTrackerToMITKCoords ) 
     {
       mitk::AffineTransform3D::Pointer affineTransform = data->GetGeometry()->GetIndexToWorldTransform();
        
       if (affineTransform.IsNull())
       {
         QString message("ERROR: QmitkIGITrackerTool, AffineTransform IndexToWorldTransform not initialized!");
         emit StatusUpdate(message);
         return;
       }


       mitk::NavigationData::Pointer nd_in  = mitk::NavigationData::New();
       mitk::NavigationData::Pointer nd_out = mitk::NavigationData::New();
       mitk::NavigationData::PositionType p;

       mitk::FillVector3D(p, inputTransformMat[0][3], inputTransformMat[1][3], inputTransformMat[2][3]);
       nd_in->SetPosition(p);

       float * quats = new float[4];
       igtl::MatrixToQuaternion(inputTransformMat, quats);

       mitk::Quaternion mitkQuats(quats[0], quats[1], quats[2], quats[3]);
       nd_in->SetOrientation(mitkQuats);
       nd_in->SetDataValid(true);

       //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
       m_FiducialRegistrationFilter->SetInput(nd_in);
       m_FiducialRegistrationFilter->UpdateOutputData(0);
       nd_out = m_FiducialRegistrationFilter->GetOutput();
       nd_out->SetDataValid(true);

      //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
       //store the current scaling to set it after transformation
       mitk::Vector3D spacing = data->GetUpdatedTimeSlicedGeometry()->GetSpacing();
       //clear spacing of data to be able to set it again afterwards
       float scale[] = {1.0, 1.0, 1.0};
       data->GetGeometry()->SetSpacing(scale);

       /*now bring quaternion to affineTransform by using vnl_Quaternion*/
       affineTransform->SetIdentity();

       //calculate the transform from the quaternions
       static itk::QuaternionRigidTransform<double>::Pointer quatTransform = itk::QuaternionRigidTransform<double>::New();

       mitk::NavigationData::OrientationType orientation = nd_out->GetOrientation();
       // convert mitk::ScalarType quaternion to double quaternion because of itk bug
       vnl_quaternion<double> doubleQuaternion(orientation.x(), orientation.y(), orientation.z(), orientation.r());
       quatTransform->SetIdentity();
       quatTransform->SetRotation(doubleQuaternion);
       quatTransform->Modified();

       /* because of an itk bug, the transform can not be calculated with float data type.
       To use it in the mitk geometry classes, it has to be transfered to mitk::ScalarType which is float */
       static mitk::AffineTransform3D::MatrixType m;
       mitk::TransferMatrix(quatTransform->GetMatrix(), m);
       affineTransform->SetMatrix(m);

       ///*set the offset by convert from itkPoint to itkVector and setting offset of transform*/
       mitk::Vector3D pos;
       pos.Set_vnl_vector(nd_out->GetPosition().Get_vnl_vector());
       affineTransform->SetOffset(pos);
       affineTransform->Modified();

       //set the transform to data
       data->GetGeometry()->SetIndexToWorldTransform(affineTransform);
       //set the original spacing to keep scaling of the geometrical object
       data->GetGeometry()->SetSpacing(spacing);
      } 
     else
     {
       static itk::Matrix<float,3,3> m;
       for ( int row = 0 ; row < 3 ; row ++ ) 
       {
         for ( int col = 0 ; col < 3 ; col ++ ) 
         {
           m[row][col]=inputTransformMat[row][col];
         }
       }

       mitk::AffineTransform3D::Pointer affineTransform =mitk::AffineTransform3D::New(); 
       affineTransform->SetMatrix(m);
       mitk::Vector3D pos;
       for ( int i = 0 ; i < 3 ; i ++ )
       {
         pos[i] = inputTransformMat[i][3];
       }
       affineTransform->SetOffset(pos);
       affineTransform->Modified();
       data->GetGeometry()->SetIndexToWorldTransform(affineTransform);
     }//if m_TransformTrackerToMITKCoords 

     data->GetGeometry()->TransferItkToVtkTransform(); // update VTK Transform for rendering too
     data->GetGeometry()->Modified();
     data->Modified();

    } // foreach AssociatedTool
    ///---
    foreach ( tempNode, m_PreMatrixAssociatedTools.values(toolName))
    {

     if (tempNode.IsNull())
     {
       QString message = QObject::tr("ERROR: QmitkIGITrackerTool, could not find node %1").arg(toolName);
       emit StatusUpdate(message);
       return;
     }
    
     // Get the transform from data
     mitk::BaseData * data = tempNode->GetData();

     //In world coordinates X is up, in lap Y is up, so rotate -90 around z and
     //put it 400 in front of the camera
     itk::Matrix<double,4,4> InMatrix;
     for ( int row = 0 ; row < 4 ; row ++ ) 
     {
       for ( int col = 0 ; col < 4 ; col ++ )
       {
         InMatrix[row][col]=inputTransformMat[row][col];
       }
     }
     itk::Matrix<double,4,4> AfterMatrix =InMatrix *  m_PreMatrix ;
   
     static itk::Matrix<float,3,3> m;
     for ( int row = 0 ; row < 3 ; row ++ ) 
     {
       for ( int col = 0 ; col < 3 ; col ++ ) 
       {
         m[row][col]=AfterMatrix[row][col];
       }
     }

     mitk::AffineTransform3D::Pointer affineTransform =mitk::AffineTransform3D::New(); 
     affineTransform->SetMatrix(m);
     mitk::Vector3D pos;
     for ( int i = 0 ; i < 3 ; i ++ )
     {
       pos[i] = AfterMatrix[i][3];
     }
     affineTransform->SetOffset(pos);
     affineTransform->Modified();
     data->GetGeometry()->SetIndexToWorldTransform(affineTransform);

     data->GetGeometry()->TransferItkToVtkTransform(); // update VTK Transform for rendering too
     data->GetGeometry()->Modified();
     data->Modified();

    } // foreach CameraAssociated node
  } // if transform data
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::DisplayTrackerData(NiftyLinkMessage* msg)
{
  if (msg->GetMessageType() == QString("TRANSFORM"))
  {
    NiftyLinkTransformMessage* trMsg;
    trMsg = static_cast<NiftyLinkTransformMessage*>(msg);

    if (trMsg != NULL)
    {
      // Print stuff
      QString header;
      header.append("Message from: ");
      header.append(trMsg->GetHostName());
      header.append(", messageId=");
      header.append(QString::number(trMsg->GetId()));
      header.append("\n");

      QString matrix = trMsg->GetMatrixAsString();
      matrix.append("\n");

      QString message = header + matrix;

      emit StatusUpdate(message);
    }
  }
  else if (msg->GetMessageType() == QString("TDATA"))
  {
    NiftyLinkTrackingDataMessage* trMsg;
    trMsg = static_cast<NiftyLinkTrackingDataMessage*>(msg);

    if (trMsg != NULL)
    {
      // Print stuff
      QString header;
      header.append("Message from: ");
      header.append(trMsg->GetHostName());
      header.append(", messageId=");
      header.append(QString::number(trMsg->GetId()));
      header.append(", toolId=");
      header.append(trMsg->GetTrackerToolName());
      header.append("\n");

      QString matrix = trMsg->GetMatrixAsString();
      matrix.append("\n");

      QString message = header + matrix;

      emit StatusUpdate(message);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::EnableTool(const QString &toolName, const bool& enable)
{
  m_EnabledTools.insert(toolName, enable);

//  // Send command to enable the given tool.
//  CommandDescriptorXMLBuilder attachToolCmd;
//  attachToolCmd.setCommandName("AttachTool");
//  attachToolCmd.addParameter("ToolName", "QString", toolName);
//  attachToolCmd.addParameter("Enabled", "bool", QString::number(enable));
//
//  NiftyLinkStringMessage::Pointer cmdMsg(new NiftyLinkStringMessage());
//  cmdMsg->setString(attachToolCmd.GetXMLAsString());
//
//  qDebug() << "TODO: send message " << attachToolCmd.GetXMLAsString();

  QString statusMessage = QString("STATUS: tool ") + toolName + QString(", set to enabled=") + QString::number(enable) + QString("\n");
  emit StatusUpdate(statusMessage);
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::GetToolPosition(const QString &toolName)
{
  if (m_EnabledTools.contains(toolName) && m_EnabledTools.value(toolName))
  {
//    NiftyLinkMessage::Pointer getPos;
//    getPos.reset();
//
//    NiftyLinkTrackingDataMessage::Create_GET(getPos);
//
//    qDebug() << "TODO: send get current position message " << toolName;

    QString statusMessage = QString("STATUS: Requested position from tool ") + toolName + QString("\n");
    emit StatusUpdate(statusMessage);
  }
}

//---------------------------------------------------------------------------
void QmitkIGITrackerTool::GetCurrentTipPosition()
{
  igtl::TimeStamp::Pointer timeNow = igtl::TimeStamp::New();

  igtlUint64 idNow = GetTimeInNanoSeconds(timeNow);
  mitk::IGIDataType* data = this->RequestData(idNow);

  if (data != NULL)
  {
    QmitkIGINiftyLinkDataType::Pointer dataType = static_cast<QmitkIGINiftyLinkDataType*>(data);
     if (dataType.IsNotNull())
     {

        NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
        if (pointerToMessage != NULL)
        {
          
          NiftyLinkTrackingDataMessage* trMsg;
          trMsg = static_cast<NiftyLinkTrackingDataMessage*>(pointerToMessage);
          float inputTransformMat[4][4];
          trMsg->GetMatrix(inputTransformMat);
          qDebug() << inputTransformMat[0][3] << " " << inputTransformMat[1][3] << " " << inputTransformMat[2][3];
          mitk::PointSet::PointType point;
          point[0]=inputTransformMat[0][3];
          point[1]=inputTransformMat[1][3];
          point[2]=inputTransformMat[2][3];
          int Size=m_TrackerFiducialsPointSet->GetSize();
          m_TrackerFiducialsPointSet->InsertPoint(  Size, point );

        }
     }
  }
}


//---------------------------------------------------------------------------
bool QmitkIGITrackerTool::AddDataNode(const QString toolName, mitk::DataNode::Pointer dataNode)
{
  QList<mitk::DataNode::Pointer> AlreadyAddedNodes = m_AssociatedTools.values(toolName);
  if ( AlreadyAddedNodes.contains(dataNode) )
  {
    return false;
  }
  else
  {
    m_AssociatedTools.insertMulti(toolName,dataNode);
    return true;
  }
}


//---------------------------------------------------------------------------
bool QmitkIGITrackerTool::RemoveDataNode(const QString toolName, mitk::DataNode::Pointer dataNode)
{
  QList<mitk::DataNode::Pointer> AlreadyAddedNodes = m_AssociatedTools.values(toolName);
  int RemovedNodes = AlreadyAddedNodes.removeAll(dataNode);

  m_AssociatedTools.remove(toolName);
  mitk::DataNode::Pointer tempNode = mitk::DataNode::New();
  foreach ( tempNode, AlreadyAddedNodes ) 
  {
    m_AssociatedTools.insertMulti(toolName,tempNode);
  }
  if ( RemovedNodes != 1 ) 
  {
    return false;
  }
  else
  {
    return true;
  }
}


//---------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> QmitkIGITrackerTool::GetDataNode(const QString toolName)
{
  return m_AssociatedTools.values(toolName);
}


//---------------------------------------------------------------------------
bool QmitkIGITrackerTool::AddPreMatrixDataNode(const QString toolName, mitk::DataNode::Pointer dataNode)
{
  QList<mitk::DataNode::Pointer> AlreadyAddedNodes = m_PreMatrixAssociatedTools.values(toolName);
  if ( AlreadyAddedNodes.contains(dataNode) )
  {
    return false;
  }
  else
  {
    m_PreMatrixAssociatedTools.insertMulti(toolName,dataNode);
    return true;
  }
}


//---------------------------------------------------------------------------
bool QmitkIGITrackerTool::RemovePreMatrixDataNode(const QString toolName, mitk::DataNode::Pointer dataNode)
{
  QList<mitk::DataNode::Pointer> AlreadyAddedNodes = m_PreMatrixAssociatedTools.values(toolName);
  int RemovedNodes = AlreadyAddedNodes.removeAll(dataNode);

  m_PreMatrixAssociatedTools.remove(toolName);
  mitk::DataNode::Pointer tempNode = mitk::DataNode::New();
  foreach ( tempNode, AlreadyAddedNodes ) 
  {
    m_PreMatrixAssociatedTools.insertMulti(toolName,tempNode);
  }
  if ( RemovedNodes != 1 ) 
  {
    return false;
  }
  else
  {
    return true;
  }
}


//---------------------------------------------------------------------------
QList<mitk::DataNode::Pointer> QmitkIGITrackerTool::GetPreMatrixDataNode(const QString toolName)
{
  return m_PreMatrixAssociatedTools.values(toolName);
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkIGITrackerTool::GetToolRepresentation(const QString toolName)
{

  mitk::DataNode::Pointer result = NULL;

  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (dataStorage != NULL)
  {
    mitk::DataNode::Pointer tempNode = dataStorage->GetNamedNode(toolName.toStdString());
    if (tempNode.IsNull())
    {
      mitk::Vector3D cp;
      cp[0] = 0;
      cp[1] = 0;
      cp[2] = 7.5;

      tempNode = mitk::CreateConeRepresentation(toolName.toStdString().c_str(), cp);
      tempNode->SetColor(0.4,0.70,0.85);

      dataStorage->Add(tempNode);
      m_ToolRepresentations.insert(toolName, tempNode);
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::InitializeFiducials()
{

  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (dataStorage == NULL || m_PointSetsInitialized)
  {
    return;
  }

  m_ImageFiducialsPointSet = mitk::PointSet::New();
  m_ImageFiducialsDataNode = mitk::DataNode::New();
  m_ImageFiducialsDataNode->SetData(m_ImageFiducialsPointSet);

  mitk::Color color;
  color.Set(1.0f, 0.0f, 0.0f);
  m_ImageFiducialsDataNode->SetName("Registration_ImageFiducials");
  m_ImageFiducialsDataNode->SetColor(color);
  m_ImageFiducialsDataNode->SetBoolProperty( "updateDataOnRender", false );


  m_TrackerFiducialsPointSet = mitk::PointSet::New();

  m_TrackerFiducialsDataNode = mitk::DataNode::New();
  m_TrackerFiducialsDataNode->SetData(m_TrackerFiducialsPointSet);

  color.Set(0.0f, 1.0f, 0.0f);
  m_TrackerFiducialsDataNode->SetName("Registration_TrackingFiducials");
  m_TrackerFiducialsDataNode->SetColor(color);
  m_TrackerFiducialsDataNode->SetBoolProperty( "updateDataOnRender", false );

  dataStorage->Add(m_ImageFiducialsDataNode);
  dataStorage->Add(m_TrackerFiducialsDataNode);

  m_PointSetsInitialized = true;
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkIGITrackerTool::GetImageFiducialsNode() const
{
  return m_ImageFiducialsDataNode;
}


//-----------------------------------------------------------------------------
mitk::DataNode* QmitkIGITrackerTool::GetTrackerFiducialsNode() const
{
  return m_TrackerFiducialsDataNode;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::ClearFiducials()
{
  m_ImageFiducialsPointSet->Clear();
  m_TrackerFiducialsPointSet->Clear();
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::AddFiducialsToDataStorage()
{
  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (dataStorage != NULL)
  {
    if (!dataStorage->Exists(m_ImageFiducialsDataNode))
    {
      dataStorage->Add(m_ImageFiducialsDataNode);
    }
    if (!dataStorage->Exists(m_TrackerFiducialsDataNode))
    {
      dataStorage->Add(m_TrackerFiducialsDataNode);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::RemoveFiducialsFromDataStorage()
{
  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (dataStorage != NULL)
  {
    if (dataStorage->Exists(m_ImageFiducialsDataNode))
    {
      dataStorage->Remove(m_ImageFiducialsDataNode);
    }
    if (dataStorage->Exists(m_TrackerFiducialsDataNode))
    {
      dataStorage->Remove(m_TrackerFiducialsDataNode);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::RegisterFiducials()
{
  m_ImageFiducialsPointSet = dynamic_cast<mitk::PointSet*>(m_ImageFiducialsDataNode->GetData());
  m_TrackerFiducialsPointSet = dynamic_cast<mitk::PointSet*>(m_TrackerFiducialsDataNode->GetData());


  if (m_ImageFiducialsPointSet.IsNull() || m_TrackerFiducialsPointSet.IsNull())
  {
    QString warning = QString("WARNING: Registration not possible\n") +
      "Fiducial data objects not found.\n" +
      "Please set 3 or more fiducials in the image and with the tracking system.\n";
    emit StatusUpdate(warning);
    return;
  }

  unsigned int minFiducialCount = 3; // \Todo: move to configurable parameter.

  if ((m_ImageFiducialsPointSet->GetSize() < (int)minFiducialCount)
    || (m_TrackerFiducialsPointSet->GetSize() < (int)minFiducialCount)
    || (m_ImageFiducialsPointSet->GetSize() != m_TrackerFiducialsPointSet->GetSize()))
  {
    QString warning = QString("WARNING: Registration not possible\n") +
        QString("Not enough fiducial pairs found. At least %1 fiducial must ").arg(minFiducialCount) +
        QString("exist for the image and the tracking system respectively.\n") +
        QString("Currently, %1 fiducials exist for the image, %2 fiducials exist for the tracking system").arg(m_ImageFiducialsPointSet->GetSize()).arg(m_TrackerFiducialsPointSet->GetSize());
    emit StatusUpdate(warning);
    return;
  }

  /* now we have two PointSets with enough points to perform a landmark based transform */
  m_FiducialRegistrationFilter->SetUseICPInitialization(m_UseICP);
  
  if ( m_TransformTrackerToMITKCoords ) 
  {
    m_FiducialRegistrationFilter->SetSourceLandmarks(m_TrackerFiducialsPointSet);
    m_FiducialRegistrationFilter->SetTargetLandmarks(m_ImageFiducialsPointSet);
  }
  else
  {
    m_FiducialRegistrationFilter->SetSourceLandmarks(m_ImageFiducialsPointSet);
    m_FiducialRegistrationFilter->SetTargetLandmarks(m_TrackerFiducialsPointSet);
  }
  m_FiducialRegistrationFilter->Update();
  QString registrationQuality = QString("%0: FRE is %1mm (Std.Dev. %2), \n"
    "RMS error is %3mm,\n"
    "Minimum registration error (best fitting landmark) is  %4mm,\n"
    "Maximum registration error (worst fitting landmark) is %5mm.")
    .arg("Fiducial Registration")
    .arg(m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetFREStdDev(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetMinError(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetMaxError(), 3, 'f', 3);

  QString updateMessage = QString("Fiducial Registration complete, FRE: %0, RMS: %1")
    .arg(m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3);

  QString statusUpdate = registrationQuality + "\n" + updateMessage + "\n";
  emit StatusUpdate(statusUpdate);
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::ApplyFiducialTransform(mitk::DataNode::Pointer dataNode)
{
  /* Transform the image data to tracker space */
  mitk::NavigationData::Pointer nd_in = mitk::NavigationData::New();
  //nd_in->SetPosition(p);

 // float * quats = new float[4];

  mitk::Quaternion mitkQuats(0.0 , 0.0 , 0.0 , 1.0);
  nd_in->SetOrientation(mitkQuats);
  nd_in->SetDataValid(true);


  m_FiducialRegistrationFilter->SetInput(nd_in);
  m_FiducialRegistrationFilter->UpdateOutputData(0);
  mitk::NavigationData::Pointer nd_out = m_FiducialRegistrationFilter->GetOutput();
  nd_out->SetDataValid(true);

  mitk::NavigationData::OrientationType orientation = nd_out->GetOrientation();
  // convert mitk::ScalarType quaternion to double quaternion because of itk bug
  vnl_quaternion<double> doubleQuaternion(orientation.x(), orientation.y(), orientation.z(), orientation.r());
  //calculate the transform from the quaternions
  static itk::QuaternionRigidTransform<double>::Pointer quatTransform = itk::QuaternionRigidTransform<double>::New();
  quatTransform->SetIdentity();
  quatTransform->SetRotation(doubleQuaternion);
  quatTransform->Modified();

  /* because of an itk bug, the transform can not be calculated with float data type.
     To use it in the mitk geometry classes, it has to be transfered to mitk::ScalarType which is float */
  static mitk::AffineTransform3D::MatrixType m;
  mitk::TransferMatrix(quatTransform->GetMatrix(), m);
  mitk::AffineTransform3D::Pointer affineTransform =mitk::AffineTransform3D::New(); 
  affineTransform->SetMatrix(m);

  ///*set the offset by convert from itkPoint to itkVector and setting offset of transform*/
  mitk::Vector3D pos;
  pos.Set_vnl_vector(nd_out->GetPosition().Get_vnl_vector());
  affineTransform->SetOffset(pos);
  affineTransform->Modified();

  //set the transform to data
  dataNode->GetData()->GetGeometry()->SetIndexToWorldTransform(affineTransform);
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerTool::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  bool success = false;
  outputFileName = "";

  QmitkIGINiftyLinkDataType::Pointer dataType = static_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNotNull())
  {
    NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
    if (pointerToMessage != NULL)
    {
      NiftyLinkTrackingDataMessage* trMsg = static_cast<NiftyLinkTrackingDataMessage*>(pointerToMessage);
      if (trMsg != NULL)
      {
        QString directoryPath = QString::fromStdString(this->GetSavePrefix()) + QDir::separator() + QString("QmitkIGITrackerTool") + QDir::separator() + QString::fromStdString(this->GetDescription());
        QDir directory(directoryPath);
        if (directory.mkpath(directoryPath))
        {
          QString fileName =  directoryPath + QDir::separator() + tr("%1.txt").arg(data->GetTimeStampInNanoSeconds());

          float matrix[4][4];
          trMsg->GetMatrix(matrix);

          QFile matrixFile(fileName);
          matrixFile.open(QIODevice::WriteOnly | QIODevice::Text);

          if (!matrixFile.error())
          {
            QTextStream matout(&matrixFile);
            matout.setRealNumberPrecision(10);
            matout.setRealNumberNotation(QTextStream::FixedNotation);

            matout << matrix[0][0] << " " << matrix[0][1] << " " << matrix[0][2] << " " << matrix[0][3]  << "\n";
            matout << matrix[1][0] << " " << matrix[1][1] << " " << matrix[1][2] << " " << matrix[1][3]  << "\n";
            matout << matrix[2][0] << " " << matrix[2][1] << " " << matrix[2][2] << " " << matrix[2][3]  << "\n";
            matout << matrix[3][0] << " " << matrix[3][1] << " " << matrix[3][2] << " " << matrix[3][3]  << "\n";

            matrixFile.close();

            outputFileName = fileName.toStdString();
            success = true;
          }
        }
      }
    }
  }
  return success;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerTool::SetCameraLink(bool LinkCamera)
{
   m_LinkCamera = LinkCamera;
  
   if ( m_LinkCamera )
   {
     mitk::RenderingManager::RenderWindowVector RenderWindows;
     RenderWindows = mitk::RenderingManager::GetInstance()->GetAllRegisteredRenderWindows();

     int windowsFound = RenderWindows.size() ;
     for ( int i = 0 ; i < windowsFound ; i ++ )
     {
       vtkCamera * Camera;
       vtkRenderWindow * thisWindow;
       thisWindow = RenderWindows.at(i);
        
       mitk::BaseRenderer::GetInstance(thisWindow)->GetVtkRenderer()->InteractiveOff();
       mitk::BaseRenderer::GetInstance(thisWindow)->GetVtkRenderer()->LightFollowCameraOn();

       Camera = mitk::BaseRenderer::GetInstance(thisWindow)->GetVtkRenderer()->GetActiveCamera();

       Camera->SetPosition(0,0,0);
       Camera->SetFocalPoint(0,0,m_focalPoint);
       Camera->SetViewUp(0,10000,0);
       Camera->SetClippingRange(m_ClipNear, m_ClipFar);
     }
   }
   else
   {
     mitk::RenderingManager::RenderWindowVector RenderWindows;
     RenderWindows = mitk::RenderingManager::GetInstance()->GetAllRegisteredRenderWindows();

     int windowsFound = RenderWindows.size() ;
     for ( int i = 0 ; i < windowsFound ; i ++ )
     {
       vtkCamera * Camera;
       vtkRenderWindow * thisWindow;
       thisWindow = RenderWindows.at(i);

       Camera = mitk::BaseRenderer::GetInstance(thisWindow)->GetVtkRenderer()->GetActiveCamera();
       
       mitk::BaseRenderer::GetInstance(thisWindow)->GetVtkRenderer()->InteractiveOn();
       mitk::BaseRenderer::GetInstance(thisWindow)->GetVtkRenderer()->LightFollowCameraOff();

       Camera->SetPosition(0,0,0);
       Camera->SetFocalPoint(0,0,m_focalPoint);
       Camera->SetViewUp(1,0,0);
       vtkMatrix4x4 * viewMatrix = vtkMatrix4x4::New();
       vtkTransform * Transform = vtkTransform::New();
       Transform->SetMatrix(viewMatrix);
       Camera->SetUserViewTransform(Transform);

       Camera->SetClippingRange(m_ClipNear, m_ClipFar);
     }
   }
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerTool::GetCameraLink()
{
   return m_LinkCamera;
}


//----------------------------------------------------------------------------
void QmitkIGITrackerTool::InitPreMatrix()
{
     m_PreMatrix[0][0]=1.0;
     m_PreMatrix[0][1]=0.0;
     m_PreMatrix[0][2]=0.0;
     m_PreMatrix[0][3]=0.0;

     m_PreMatrix[1][0]=0.0;
     m_PreMatrix[1][1]=1.0;
     m_PreMatrix[1][2]=0.0;
     m_PreMatrix[1][3]=0.0;

     m_PreMatrix[2][0]=0.0;
     m_PreMatrix[2][1]=0.0;
     m_PreMatrix[2][2]=1.0;
     m_PreMatrix[2][3]=-380.0;

     m_PreMatrix[3][0]=0.0;
     m_PreMatrix[3][1]=0.0;
     m_PreMatrix[3][2]=0.0;
     m_PreMatrix[3][3]=1.0;
}


//------------------------------------------------------------------------------
void QmitkIGITrackerTool::SetUpPositioning(QString toolName , mitk::DataNode::Pointer dataNode)
{
  itk::Matrix<double,4,4> Tracking;
  itk::Matrix<double,4,4> AssociatedNode;
  itk::Matrix<double,4,4> Out;

  itk::Vector<double,4> ux;
  itk::Vector<double,4> uy;
  itk::Vector<double,4> uz;
  
  ux[0] = 1;
  ux[1] = 0;
  ux[2] = 0;
  ux[3] = 1;
  
  uy[0] = 0;
  uy[1] = 1;
  uy[2] = 0;
  uy[3] = 1;
 
  uz[0] = 0;
  uz[1] = 0;
  uz[2] = 1;
  uz[3] = 1;

  mitk::AffineTransform3D::Pointer AssociatedNodeTransform =mitk::AffineTransform3D::New(); 
  AssociatedNodeTransform = dataNode->GetData()->GetGeometry()->GetIndexToWorldTransform();

  igtl::TimeStamp::Pointer timeNow = igtl::TimeStamp::New();

  igtlUint64 idNow = GetTimeInNanoSeconds(timeNow);
  mitk::IGIDataType* data = this->RequestData(idNow);

  float inputTransformMat[4][4];
  if (data != NULL)
  {
    QmitkIGINiftyLinkDataType::Pointer dataType = static_cast<QmitkIGINiftyLinkDataType*>(data);
     if (dataType.IsNotNull())
     {

        NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
        if (pointerToMessage != NULL)
        {
          
          NiftyLinkTrackingDataMessage* trMsg;
          trMsg = static_cast<NiftyLinkTrackingDataMessage*>(pointerToMessage);
          trMsg->GetMatrix(inputTransformMat);
        }
     }
  }

  mitk::AffineTransform3D::MatrixType m1;
  m1=AssociatedNodeTransform->GetMatrix();
  mitk::Vector3D pos = AssociatedNodeTransform->GetOffset();
  for ( int row = 0 ; row < 4 ; row ++ ) 
  {
     for ( int col = 0 ; col < 4 ; col ++ ) 
     {
       Tracking[row][col]=inputTransformMat[row][col];
       if ( (row < 3) && (col < 3) )
       {
         AssociatedNode[row][col] = m1[row][col];
       }
       if ( row == 3 )
       {
         AssociatedNode[row][col] = inputTransformMat[row][col];
       }
       if ( col == 3 ) 
       {
         if ( row < 3 ) 
         {
            AssociatedNode[row][col] = pos[row];
         }
       }
     }
  }
  
  Out=AssociatedNode*Tracking.GetInverse();
  qDebug() << "Associated Node Matrix";
  for ( int row = 0 ; row < 4 ; row ++ ) 
  {
    qDebug() << AssociatedNode[row][0] << " " <<  AssociatedNode[row][1] << " " <<  AssociatedNode[row][2] << " " <<  AssociatedNode[row][3];
  }
  qDebug() << "Tracking Matrix";
  for ( int row = 0 ; row < 4 ; row ++ ) 
  {
    qDebug() << Tracking[row][0] << " " <<  Tracking[row][1] << " " <<  Tracking[row][2] << " " <<  Tracking[row][3];
  }
  qDebug() << "Out Matrix";
  for ( int row = 0 ; row < 4 ; row ++ ) 
  {
    qDebug() << Out[row][0] << " " <<  Out[row][1] << " " <<  Out[row][2] << " " <<  Out[row][3];
  }
  itk::Vector<double,4> ux1;
  itk::Vector<double,4> uy1;
  itk::Vector<double,4> uz1;
  ux1 = Out * ux;
  uy1 = Out * uy;
  uz1 = Out * uz;

  mitk::PointSet::Pointer SourcePointSet = mitk::PointSet::New();
  mitk::PointSet::Pointer TargetPointSet = mitk::PointSet::New();
  
  mitk::PointSet::PointType point11;
  mitk::PointSet::PointType point21;
  mitk::PointSet::PointType point31;
  mitk::PointSet::PointType point12;
  mitk::PointSet::PointType point22;
  mitk::PointSet::PointType point32;
  for ( int i = 0 ; i < 3 ; i ++ )
  {
    point11[i] = ux[i];
    point21[i] = uy[i];
    point31[i] = uz[i];
    point12[i] = ux1[i];
    point22[i] = uy1[i];
    point32[i] = uz1[i];
  }
  
  SourcePointSet->InsertPoint(0,point11);
  SourcePointSet->InsertPoint(1,point21);
  SourcePointSet->InsertPoint(2,point31);
  TargetPointSet->InsertPoint(0,point12);
  TargetPointSet->InsertPoint(1,point22);
  TargetPointSet->InsertPoint(2,point32);
 /* SourcePointSet->InsertPoint(0,point12);
  SourcePointSet->InsertPoint(1,point22);
  SourcePointSet->InsertPoint(2,point32);
  TargetPointSet->InsertPoint(0,point11);
  TargetPointSet->InsertPoint(1,point21);
  TargetPointSet->InsertPoint(2,point31);*/

  qDebug() << point11[0] << " " << point11[1] << " " << point11[2];
  qDebug() << point21[0] << " " << point21[1] << " " << point21[2];
  qDebug() << point31[0] << " " << point31[1] << " " << point31[2];
  qDebug() << point12[0] << " " << point12[1] << " " << point12[2];
  qDebug() << point22[0] << " " << point22[1] << " " << point22[2];
  qDebug() << point32[0] << " " << point32[1] << " " << point32[2];

  m_FiducialRegistrationFilter->SetUseICPInitialization(m_UseICP);
  m_FiducialRegistrationFilter->SetSourceLandmarks(SourcePointSet);
  m_FiducialRegistrationFilter->SetTargetLandmarks(TargetPointSet);
  
  m_FiducialRegistrationFilter->Update();
  QString registrationQuality = QString("%0: FRE is %1mm (Std.Dev. %2), \n"
    "RMS error is %3mm,\n"
    "Minimum registration error (best fitting landmark) is  %4mm,\n"
    "Maximum registration error (worst fitting landmark) is %5mm.")
    .arg("Fiducial Registration")
    .arg(m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetFREStdDev(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetMinError(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetMaxError(), 3, 'f', 3);

  QString updateMessage = QString("Fiducial Registration complete, FRE: %0, RMS: %1")
    .arg(m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
    .arg(m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3);

  QString statusUpdate = registrationQuality + "\n" + updateMessage + "\n";
  qDebug() <<  statusUpdate;

  dataNode->GetData()->GetGeometry()->GetBoundingBox();
}

