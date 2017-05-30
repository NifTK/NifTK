/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysTracker.h"
#include <mitkLogMacros.h>
#include <mitkExceptionMacro.h>
#include <mitkSurface.h>
#include <vtkMath.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <ftkInterface.h>
#include <helpers.hpp>
#include <geometryHelper.hpp>

namespace niftk
{

//-----------------------------------------------------------------------------
class AtracsysTrackerPrivate
{

public:

  AtracsysTrackerPrivate(const AtracsysTracker* q,
                         const std::vector<std::string>& toolGeometryFileNames);
  ~AtracsysTrackerPrivate();

  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > GetTrackingData();
  std::vector<mitk::Point3D> GetBallPositions();

private:

  void CheckError(ftkLibrary lib);

  const AtracsysTracker                      *m_Container;
  const std::vector<std::string>              m_GeometryFiles;
  uint64                                      m_SerialNumber;
  ftkLibrary                                  m_Lib;
  ftkFrameQuery                              *m_Frame;
};


//-----------------------------------------------------------------------------
AtracsysTrackerPrivate::AtracsysTrackerPrivate(const AtracsysTracker* t,
                                               const std::vector<std::string>& toolGeometryFileNames
                                              )
: m_Container(t)
, m_GeometryFiles(toolGeometryFileNames)
, m_SerialNumber(0)
, m_Lib(nullptr)
, m_Frame(nullptr)
{

  m_Lib = ftkInit();
  if ( m_Lib == nullptr )
  {
    mitkThrow() << "Cannot initialize Atracsys driver.";
  }

  DeviceData device;
  device.SerialNumber = 0uLL;

  ftkError err = ftkEnumerateDevices( m_Lib, fusionTrackEnumerator, &device );
  if ( err != FTK_OK )
  {
    this->CheckError(m_Lib);
  }

  if ( device.SerialNumber == 0uLL )
  {
    mitkThrow() << "No Atracsys device connected.";
  }

  m_SerialNumber = device.SerialNumber;
  MITK_INFO << "Connected to Atracsys SN:" << std::hex << m_SerialNumber;
  
  // load all geometries if any.
  for (int i = 0; i < m_GeometryFiles.size(); i++)
  {
    ftkGeometry geom;
    if (loadGeometry( m_Lib, m_SerialNumber, m_GeometryFiles[i], geom) == 2)
    {
      mitkThrow() << "Failed to load geometry file:" << m_GeometryFiles[i];
    }
    err = ftkSetGeometry( m_Lib, m_SerialNumber, &geom );
    if ( err != FTK_OK )
    {
      this->CheckError(m_Lib);
    }
  }
  
  m_Frame = ftkCreateFrame();
  if ( m_Frame == nullptr )
  {
    MITK_ERROR << "Cannot create frame instance";
    this->CheckError(m_Lib);
  }


  err = ftkSetFrameOptions( false, false, 0u, 0u,
                            100u, 16u, m_Frame );
  if ( err != FTK_OK )
  {
    ftkDeleteFrame( m_Frame );
    MITK_ERROR << "Cannot initialise frame";
    this->CheckError(m_Lib);
  }
}


//-----------------------------------------------------------------------------
AtracsysTrackerPrivate::~AtracsysTrackerPrivate()
{
  ftkDeleteFrame( m_Frame );

  if (m_Lib != nullptr)
  {
    ftkClose( &m_Lib );
  }
}


//-----------------------------------------------------------------------------
void AtracsysTrackerPrivate::CheckError(ftkLibrary lib)
{
  ftkErrorExt ext;
  ftkError err = ftkGetLastError( lib, &ext );
  if ( err == FTK_OK ) // means we successfully retrieved the error message.
  {
    std::string message;
    if ( ext.isError() )
    {
      ext.errorString( message );
      MITK_ERROR << "AtracsysTrackerPrivate:" << message;
      mitkThrow() << message;
    }
    if ( ext.isWarning() )
    {
      ext.warningString( message );
      MITK_WARN << "AtracsysTrackerPrivate:" << message;
    }
    ext.messageStack( message );
    if ( message.size() > 0u )
    {
      MITK_INFO << "AtracsysTrackerPrivate:Stack:\n" << message;
    }
  }
}


//-----------------------------------------------------------------------------
std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > AtracsysTrackerPrivate::GetTrackingData()
{
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > results;
  std::string message;
  mitk::Point4D rotationQuaternion;
  rotationQuaternion.Fill(0);
  mitk::Vector3D translation;
  translation.Fill(0);
  double q[4];

  ftkGetLastFrame( m_Lib, m_SerialNumber, m_Frame, 0 );
  
  ftkErrorExt extError;
  ftkGetLastError( m_Lib, &extError );

  if ( extError.isOk() || extError.isWarning( FTK_WAR_TEMP_HIGH ) ||
       extError.isWarning( FTK_WAR_TEMP_LOW ) )
  {
    bool isError(true);

    switch ( m_Frame->markersStat )
    {
      case QS_WAR_SKIPPED:
        message = "AtracsysTrackerPrivate:marker fields in the frame are not set correctly";
        break;

      case QS_ERR_INVALID_RESERVED_SIZE:
        message = "AtracsysTrackerPrivate:frame -> markersVersionSize is invalid";
        break;

      case QS_REPROCESS:
        message = "AtracsysTrackerPrivate:Frame needs reprocessing.";
        break;
        
      case QS_OK:
        isError = false;
        break;

      default:
        MITK_WARN << "AtracsysTrackerPrivate::GetBallPositions() = Invalid status.";
        isError = false;
        break;

    } // end switch
    
    if (isError)
    {
      MITK_ERROR << message;
      this->CheckError(m_Lib);
      mitkThrow() << message;
    }

    if (m_Frame->markersStat == QS_OK)
    {
      for (int i = 0u; i < m_Frame->markersCount; ++i )
      {
        double rotation[3][3];

        translation[0] = m_Frame->markers[i].translationMM[0];
        translation[1] = m_Frame->markers[i].translationMM[1];
        translation[2] = m_Frame->markers[i].translationMM[2];

        for (int r = 0; r < 3; r++)
        {
          for (int c = 0; c < 3; c++)
          {
            rotation[r][c] = m_Frame->markers[i].rotation[r][c];
          }
        }

        vtkMath::Matrix3x3ToQuaternion(rotation, q);

        rotationQuaternion[0] = q[0];
        rotationQuaternion[1] = q[1];
        rotationQuaternion[2] = q[2];
        rotationQuaternion[3] = q[3];

        std::pair<mitk::Point4D, mitk::Vector3D> transform(rotationQuaternion, translation);
        results.insert(std::pair<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >(
                       std::to_string(m_Frame->markers[i].geometryId), transform));
      }
    }
  }
  return results;
}


//-----------------------------------------------------------------------------
std::vector<mitk::Point3D> AtracsysTrackerPrivate::GetBallPositions()
{
  std::vector<mitk::Point3D> results;
  mitk::Point3D point;
  std::string message;

  ftkGetLastFrame( m_Lib, m_SerialNumber, m_Frame, 0  );

  ftkErrorExt extError;
  ftkGetLastError( m_Lib, &extError );

  if ( extError.isOk() || extError.isWarning( FTK_WAR_TEMP_HIGH ) ||
       extError.isWarning( FTK_WAR_TEMP_LOW ) )
  {
    bool isError(true);

    switch ( m_Frame->threeDFiducialsStat )
    {
      case QS_WAR_SKIPPED:
        message = "AtracsysTrackerPrivate:3D status fields in the frame is not set correctly.";
        break;

      case QS_ERR_INVALID_RESERVED_SIZE:
        message = "AtracsysTrackerPrivate:frame -> threeDFiducialsVersionSize is invalid.";
        break;

      case QS_ERR_OVERFLOW:
        message = "AtracsysTrackerPrivate:Buffer size too small.";
        break;

      case QS_REPROCESS:
        message = "AtracsysTrackerPrivate:Frame needs reprocessing.";
        break;
        
      case QS_OK:
        isError = false;
        break;

      default:
        MITK_WARN << "AtracsysTrackerPrivate::GetBallPositions() = Invalid status.";
        isError = false;
        break;

    } // end switch

    if (isError)
    {
      MITK_ERROR << message;
      this->CheckError(m_Lib);
      mitkThrow() << message;
    }

    if (m_Frame->threeDFiducialsStat == QS_OK)
    {
      for ( uint32 m = 0; m < m_Frame->threeDFiducialsCount; m++ )
      {
        if (   m_Frame->threeDFiducials[ m ].epipolarErrorPixels < 1
            && m_Frame->threeDFiducials[ m ].triangulationErrorMM < 0.2
            && m_Frame->threeDFiducials[ m ].probability > 0.8
            && m_Frame->threeDFiducials[ m ].positionMM.z > 700   // minimum range
            && m_Frame->threeDFiducials[ m ].positionMM.z < 2400  // maximum range for 0.11mm accuracy.
           )
        {
          point[0] = m_Frame->threeDFiducials[ m ].positionMM.x;
          point[1] = m_Frame->threeDFiducials[ m ].positionMM.y;
          point[2] = m_Frame->threeDFiducials[ m ].positionMM.z;
          results.push_back(point);
        }
      }
    }
  }
  return results;
}


//-----------------------------------------------------------------------------
AtracsysTracker::AtracsysTracker(mitk::DataStorage::Pointer dataStorage,
                                 std::string toolConfigFileName)
: niftk::IGITracker(dataStorage, toolConfigFileName, 330)
, m_Tracker(nullptr)
{
  MITK_INFO << "Creating AtracsysTracker";

  // Base class deserialises the MITK IGTToolStorage format.
  // We just need the filenames to send to the Atracsys API.

  std::vector<std::string> fileNames;
  for (int i = 0; i < m_NavigationToolStorage->GetToolCount(); i++)
  {
    mitk::NavigationTool::Pointer tool = m_NavigationToolStorage->GetTool(i);
    fileNames.push_back(tool->GetCalibrationFile());
  }

  m_Tracker.reset(new AtracsysTrackerPrivate(this, fileNames));

  // Manually construct a tracking volume.
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->Allocate(12);
  points->SetPoint(0,  -113.5, -171.5, 700);
  points->SetPoint(1,   113.5, -171.5, 700);
  points->SetPoint(2,   113.5,  171.5, 700);
  points->SetPoint(3,  -113.5,  171.5, 700);
  points->SetPoint(4,  -663.5, -488.0, 2000);
  points->SetPoint(5,   663.5, -488.0, 2000);
  points->SetPoint(6,   663.5,  488.0, 2000);
  points->SetPoint(7,  -663.5,  488.0, 2000);
  points->SetPoint(8,  -928.5, -683.0, 2800);
  points->SetPoint(9,   928.5, -683.0, 2800);
  points->SetPoint(10,  928.5,  683.0, 2800);
  points->SetPoint(11, -928.5,  683.0, 2800);

  vtkIdType id;
  vtkIdType ids[4];

  vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();

  id = 0;
  ids[0] = 0;
  ids[1] = 1;
  ids[2] = 2;
  ids[3] = 3;
  polys->InsertNextCell(id, ids);

  id = 1;
  ids[0] = 4;
  ids[1] = 5;
  ids[2] = 6;
  ids[3] = 7;
  polys->InsertNextCell(id, ids);

  id = 2;
  ids[0] = 8;
  ids[1] = 9;
  ids[2] = 10;
  ids[3] = 11;
  polys->InsertNextCell(id, ids);

  id = 3;
  ids[0] = 0;
  ids[1] = 4;
  ids[2] = 5;
  ids[3] = 1;
  polys->InsertNextCell(id, ids);

  id = 4;
  ids[0] = 1;
  ids[1] = 5;
  ids[2] = 6;
  ids[3] = 2;
  polys->InsertNextCell(id, ids);

  id = 5;
  ids[0] = 2;
  ids[1] = 6;
  ids[2] = 7;
  ids[3] = 3;
  polys->InsertNextCell(id, ids);

  id = 6;
  ids[0] = 4;
  ids[1] = 0;
  ids[2] = 3;
  ids[3] = 7;
  polys->InsertNextCell(id, ids);

  id = 7;
  ids[0] = 8;
  ids[1] = 9;
  ids[2] = 5;
  ids[3] = 4;
  polys->InsertNextCell(id, ids);

  id = 8;
  ids[0] = 5;
  ids[1] = 9;
  ids[2] = 10;
  ids[3] = 6;
  polys->InsertNextCell(id, ids);

  id = 9;
  ids[0] = 6;
  ids[1] = 10;
  ids[2] = 11;
  ids[3] = 7;
  polys->InsertNextCell(id, ids);

  id = 10;
  ids[0] = 8;
  ids[1] = 4;
  ids[2] = 7;
  ids[3] = 11;
  polys->InsertNextCell(id, ids);

  vtkPolyData *polyData = vtkPolyData::New();
  polyData->SetPoints(points);
  polyData->SetPolys(polys);
  
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(polyData);

  mitk::Color red;
  red.SetRed(1);
  red.SetBlue(0);
  red.SetGreen(0);

  mitk::DataNode::Pointer node = mitk::DataNode::New();
  node->SetData(surface);
  node->SetColor(red);
  node->SetOpacity(0.25);
  
  m_TrackingVolumeNode = node;

  this->SetVisibilityOfTrackingVolume(true);
  m_DataStorage->Add(m_TrackingVolumeNode);
}


//-----------------------------------------------------------------------------
AtracsysTracker::~AtracsysTracker()
{
  MITK_INFO << "Destroying AtracsysTracker";
}


//-----------------------------------------------------------------------------
std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > AtracsysTracker::GetTrackingData()
{
  return m_Tracker->GetTrackingData();
}


//-----------------------------------------------------------------------------
std::vector<mitk::Point3D> AtracsysTracker::GetBallPositions()
{
  return m_Tracker->GetBallPositions();
}

} // end namespace
