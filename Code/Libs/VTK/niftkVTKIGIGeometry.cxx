/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkVTKIGIGeometry.h"
#include "niftkVTKFunctions.h"

#include <vtkCubeSource.h>
#include <vtkSphereSource.h>
#include <vtkPlane.h>
#include <vtkPlaneCollection.h>
#include <vtkClipClosedSurface.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkAppendPolyData.h>
#include <vtkCylinderSource.h>
#include <vtkLineSource.h>
#include <vtkArcSource.h>
#include <vtkVersion.h>

#include <sstream>
#include <cassert>

namespace niftk
{ 
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeLaparoscope ( std::string RigidBodyFilename,
    std::string LeftHandeyeFilename , std::string RightHandeyeFilename, 
    std::string CentreHandeyeFilename, bool AddCrossHairs , float TrackerMarkerRadius, float LensAngle, float BodyLength ) 
{
  float channelBodyLength = BodyLength/4.0;
  std::vector < std::vector <float> > positions = this->ReadRigidBodyDefinitionFile(RigidBodyFilename);
  vtkSmartPointer<vtkMatrix4x4> lefthandeye = LoadMatrix4x4FromFile(LeftHandeyeFilename, false);
  vtkSmartPointer<vtkTransform> lefttransform = vtkSmartPointer<vtkTransform>::New();
  lefttransform->SetMatrix(lefthandeye);

  vtkSmartPointer<vtkPolyData> leftLensCowl = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkCylinderSource> leftLensCyl = vtkSmartPointer<vtkCylinderSource>::New();
  leftLensCyl->SetRadius(2.0);
  leftLensCyl->SetHeight(channelBodyLength+20.0);
  leftLensCyl->SetCenter(0.0,0.0,0.0);
  leftLensCyl->SetResolution(40);
  leftLensCyl->CappingOff();
  
  leftLensCowl=leftLensCyl->GetOutput();

  vtkSmartPointer<vtkTransform> leftTipTransform = vtkSmartPointer<vtkTransform>::New();
  leftTipTransform->RotateX(180 + (90 + LensAngle) );
  leftTipTransform->Translate(0,channelBodyLength/2,0);

  TranslatePolyData(leftLensCowl,leftTipTransform);
  TranslatePolyData(leftLensCowl,lefttransform);
 
  vtkSmartPointer<vtkMatrix4x4> righthandeye = LoadMatrix4x4FromFile(RightHandeyeFilename, false);
  vtkSmartPointer<vtkTransform> righttransform = vtkSmartPointer<vtkTransform>::New();
  righttransform->SetMatrix(righthandeye);

  vtkSmartPointer<vtkPolyData> rightLensCowl = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkCylinderSource> rightLensCyl = vtkSmartPointer<vtkCylinderSource>::New();
  rightLensCyl->SetRadius(2.0);
  rightLensCyl->SetHeight(channelBodyLength+20.0);
  rightLensCyl->SetCenter(0.0,0.0,0.0);
  rightLensCyl->SetResolution(40);
  rightLensCyl->CappingOff();
  
  rightLensCowl=rightLensCyl->GetOutput();

  vtkSmartPointer<vtkTransform> rightTipTransform = vtkSmartPointer<vtkTransform>::New();
  rightTipTransform->RotateX(180 + (90 + LensAngle) );
  rightTipTransform->Translate(0,channelBodyLength/2,0);
  
  TranslatePolyData(rightLensCowl,rightTipTransform);
  TranslatePolyData(rightLensCowl,righttransform);
 
  vtkSmartPointer<vtkMatrix4x4> centrehandeye = LoadMatrix4x4FromFile(CentreHandeyeFilename, false);
  vtkSmartPointer<vtkTransform> centretransform = vtkSmartPointer<vtkTransform>::New();
  centretransform->SetMatrix(centrehandeye);

  vtkSmartPointer<vtkPolyData> centreLensCowl = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkCylinderSource> centreLensCyl = vtkSmartPointer<vtkCylinderSource>::New();
  centreLensCyl->SetRadius(5.0);
  centreLensCyl->SetHeight(BodyLength+20.0);
  centreLensCyl->SetCenter(0.0,0.0,0.0);
  centreLensCyl->SetResolution(40);
  centreLensCyl->CappingOff();
  
  centreLensCowl=centreLensCyl->GetOutput();

  vtkSmartPointer<vtkTransform> centreTipTransform = vtkSmartPointer<vtkTransform>::New();
  centreTipTransform->RotateX(180 + (90 + LensAngle) );
  centreTipTransform->Translate(0,BodyLength/2,0);

  TranslatePolyData(centreLensCowl,centreTipTransform);
  TranslatePolyData(centreLensCowl,centretransform);
 
  vtkSmartPointer<vtkPolyData> ireds = this->MakeIREDs(positions , TrackerMarkerRadius );

  std::vector < float > lensOrigin; 
  lensOrigin.push_back(centrehandeye->GetElement(0,3));
  lensOrigin.push_back(centrehandeye->GetElement(1,3));
  lensOrigin.push_back(centrehandeye->GetElement(2,3));

  std::vector< std::vector < float > > axis; 
  axis.push_back(this->Centroid(positions));
  axis.push_back(lensOrigin);

  vtkSmartPointer<vtkAppendPolyData> LapAppenderer = vtkSmartPointer<vtkAppendPolyData>::New();

  LapAppenderer->AddInputData(leftLensCowl);
  LapAppenderer->AddInputData(rightLensCowl);
  LapAppenderer->AddInputData(centreLensCowl);

  vtkSmartPointer<vtkPlane> lensClippingPlane = vtkSmartPointer<vtkPlane>::New();
  vtkSmartPointer<vtkPlaneCollection> planeCollection = vtkSmartPointer<vtkPlaneCollection>::New();
  vtkSmartPointer<vtkClipClosedSurface> clipper = vtkSmartPointer<vtkClipClosedSurface>::New();
  lensClippingPlane->SetOrigin(lensOrigin[0],lensOrigin[1],lensOrigin[2]);
  float *normal = new float [4];
  normal [0] = 0.0 ; normal [1] = 0.0 ; normal [2] = -1.0 ; normal [3] = 0.0;
  float  *movedNormal = new float [4];
  centrehandeye->MultiplyPoint(normal, movedNormal);
  lensClippingPlane->SetNormal(movedNormal[0],movedNormal[1],movedNormal[2]);

  planeCollection->AddItem (lensClippingPlane);

  clipper->SetClippingPlanes(planeCollection);
  clipper->SetGenerateOutline(1);
  clipper->SetGenerateFaces(1);
  clipper->SetInputData(LapAppenderer->GetOutput());

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();
  appenderer->AddInputData(clipper->GetOutput());
  appenderer->AddInputData(ireds);
  appenderer->AddInputData(this->ConnectIREDs(axis));

  if ( AddCrossHairs ) 
  {
    vtkSmartPointer<vtkArcSource> leftTopArc = vtkSmartPointer<vtkArcSource>::New();
    vtkSmartPointer<vtkArcSource> leftBottomArc = vtkSmartPointer<vtkArcSource>::New();
    vtkSmartPointer<vtkArcSource> rightTopArc = vtkSmartPointer<vtkArcSource>::New();
    vtkSmartPointer<vtkArcSource> rightBottomArc = vtkSmartPointer<vtkArcSource>::New();
    
    vtkSmartPointer<vtkLineSource> leftX = vtkSmartPointer<vtkLineSource>::New();
    vtkSmartPointer<vtkLineSource> leftY = vtkSmartPointer<vtkLineSource>::New();
    vtkSmartPointer<vtkLineSource> rightX = vtkSmartPointer<vtkLineSource>::New();
    vtkSmartPointer<vtkLineSource> rightY = vtkSmartPointer<vtkLineSource>::New();

    leftTopArc->SetPoint1(0.0, 25, 75);
    leftTopArc->SetPoint2(-25, 0.0, 75);
    leftBottomArc->SetPoint1(0.0, -25, 75);
    leftBottomArc->SetPoint2(25, 0.0, 75);
   
    leftTopArc->SetResolution(50);
    leftBottomArc->SetResolution(50);
    rightTopArc->SetResolution(50);
    rightBottomArc->SetResolution(50);
    rightTopArc->SetPoint1(0.0, 25, 75);
    rightTopArc->SetPoint2(25, 0.0, 75);
    rightBottomArc->SetPoint1(0.0, -25, 75);
    rightBottomArc->SetPoint2(-25, 0.0, 75);

    leftTopArc->SetCenter (0.0, 0.0 , 75);
    leftBottomArc->SetCenter (0.0, 0.0 , 75);
    rightTopArc->SetCenter (0.0, 0.0 , 75);
    rightBottomArc->SetCenter (0.0, 0.0 , 75);

    leftX->SetPoint1(-25.0,0.0,75);
    leftX->SetPoint2(25.0,0.0,75);
    leftY->SetPoint1(0.0,-25.0,75);
    leftY->SetPoint2(0.0,25.0,75);

    rightX->SetPoint1(-25.0,0.0,75);
    rightX->SetPoint2(25.0,0.0,75);
    rightY->SetPoint1(0.0,-25.0,75);
    rightY->SetPoint2(0.0,25.0,75);

    vtkSmartPointer<vtkAppendPolyData> leftCrossApp = vtkSmartPointer<vtkAppendPolyData>::New();
    vtkSmartPointer<vtkAppendPolyData> rightCrossApp = vtkSmartPointer<vtkAppendPolyData>::New();
    vtkSmartPointer<vtkPolyData> leftCross = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyData> rightCross = vtkSmartPointer<vtkPolyData>::New();
    
    leftCrossApp->AddInputData(leftTopArc->GetOutput());
    leftCrossApp->AddInputData(leftBottomArc->GetOutput());
    leftCrossApp->AddInputData(leftX->GetOutput());
    leftCrossApp->AddInputData(leftY->GetOutput());
     
    rightCrossApp->AddInputData(rightTopArc->GetOutput());
    rightCrossApp->AddInputData(rightBottomArc->GetOutput());
    rightCrossApp->AddInputData(rightX->GetOutput());
    rightCrossApp->AddInputData(rightY->GetOutput());

    leftCross=leftCrossApp->GetOutput();
    rightCross=rightCrossApp->GetOutput();
    TranslatePolyData(leftCross,lefttransform);
    TranslatePolyData(rightCross,righttransform);
    appenderer->AddInputData(leftCross);
    appenderer->AddInputData(rightCross);
    
  }

  //get the lens position
  appenderer->Update();
  return appenderer->GetOutput();
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakePointer ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  std::vector < std::vector <float> > positions = this->ReadRigidBodyDefinitionFile(rigidBodyFilename);
  vtkSmartPointer<vtkMatrix4x4> handeye = LoadMatrix4x4FromFile(handeyeFilename, false);
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  transform->SetMatrix(handeye);

  vtkSmartPointer<vtkPolyData> ireds = this->MakeIREDs(positions);

  std::vector < float > tip; 
  tip.push_back(handeye->GetElement(0,3));
  tip.push_back(handeye->GetElement(1,3));
  tip.push_back(handeye->GetElement(2,3));

  std::vector< std::vector < float > > axis; 
  axis.push_back(tip);
  vtkSmartPointer<vtkPolyData> tipBall = this->MakeIREDs(axis,1.5);

  axis.push_back(this->Centroid(positions));

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

#if VTK_MAJOR_VERSION <= 5
  appenderer->AddInput(ireds);
  appenderer->AddInput(this->ConnectIREDs(positions, true));
  appenderer->AddInput(tipBall);
  appenderer->AddInput(this->ConnectIREDs(axis));
#else
  appenderer->AddInputData(ireds);
  appenderer->AddInputData(this->ConnectIREDs(positions, true));
  appenderer->AddInputData(tipBall);
  appenderer->AddInputData(this->ConnectIREDs(axis));
#endif

  //get the lens position
  appenderer->Update();
  return appenderer->GetOutput();

}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeReference ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  return this->MakePointer(rigidBodyFilename, handeyeFilename);
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeReferencePolaris ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  std::vector < std::vector <float> > positions = this->ReadRigidBodyDefinitionFile(rigidBodyFilename);
  vtkSmartPointer<vtkMatrix4x4> handeye = LoadMatrix4x4FromFile(handeyeFilename, false);
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  transform->SetMatrix(handeye);

  vtkSmartPointer<vtkPolyData> ireds = this->MakeIREDs(positions, 7.5);

  std::vector < float > tip; 
  tip.push_back(handeye->GetElement(0,3));
  tip.push_back(handeye->GetElement(1,3));
  tip.push_back(handeye->GetElement(2,3));

  std::vector< std::vector < float > > axis; 
  axis.push_back(tip);
  vtkSmartPointer<vtkPolyData> tipBall = this->MakeIREDs(axis,1.5);

  axis.push_back(this->Centroid(positions));

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

#if VTK_MAJOR_VERSION <= 5
  appenderer->AddInput(ireds);
  appenderer->AddInput(tipBall);
  appenderer->AddInput(this->ConnectIREDs(axis));
#else
  appenderer->AddInputData(ireds);
  appenderer->AddInputData(tipBall);
  appenderer->AddInputData(this->ConnectIREDs(axis));
#endif

  appenderer->Update();
  return appenderer->GetOutput();

}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeAWall ( const int& whichwall, const float& size, 
   const float& xOffset,  const float& yOffset,  const float& zOffset , 
   const float& thickness ) 
{
  vtkSmartPointer<vtkCubeSource> wall =  vtkSmartPointer<vtkCubeSource>::New();

  switch ( whichwall )
  {
    case 0: //the back wall
    {
      wall->SetXLength(size);
      wall->SetYLength(size);
      wall->SetZLength(thickness);
      wall->SetCenter(size * xOffset, size * yOffset, 
          size * zOffset + size * 0.5 + thickness * 0.5);
      break;
    }
    case 1: //the left wall
    {
      wall->SetXLength(size);
      wall->SetYLength(thickness);
      wall->SetZLength(size);
      wall->SetCenter(size * xOffset,
          size * yOffset + size * 0.5 + thickness * 0.5, size * zOffset) ;
      break;
    }
    case 2: //the front wall
    {
      wall->SetXLength(size);
      wall->SetYLength(size);
      wall->SetZLength(thickness);
      wall->SetCenter(size * xOffset, size * yOffset, 
          size * zOffset - size * 0.5 - thickness * 0.5);
      break;
    }
    case 3: //the right wall
    {
      wall->SetXLength(size);
      wall->SetYLength(thickness);
      wall->SetZLength(size);
      wall->SetCenter(size * xOffset,
          size * yOffset - size * 0.5 - thickness * 0.5, size * zOffset) ;
      break;
    }
    case 4: //the ceiling
    {
      wall->SetXLength(thickness);
      wall->SetYLength(size);
      wall->SetZLength(size);
      wall->SetCenter(size * xOffset + size * 0.5 + thickness * 0.5,
          size * yOffset, size * zOffset) ;
      break;
    }
    case 5: //the floor
    {
      wall->SetXLength(thickness);
      wall->SetYLength(size);
      wall->SetZLength(size);
      wall->SetCenter(size * xOffset - size * 0.5 - thickness * 0.5,
          size * yOffset, size * zOffset) ;
      break;
    }
    default: //a mistake
    {
      std::cerr << "Passed a bad number to MakeAWall : " << whichwall;
      return NULL;
    }
  }
  wall->Update();
  return wall->GetOutput();

}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeXAxes( const float& length , const bool& symmetric)
{
  vtkSmartPointer<vtkLineSource> Axis = vtkSmartPointer<vtkLineSource>::New();

  if ( symmetric )
  {
    Axis->SetPoint1(-length,0,0);
  }
  else
  {
    Axis->SetPoint1(0,0,0);
  }

  Axis->SetPoint2(length,0,0);

  Axis->Update();
  return Axis->GetOutput();
}
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeYAxes( const float& length , const bool& symmetric)
{
  vtkSmartPointer<vtkLineSource> Axis = vtkSmartPointer<vtkLineSource>::New();

  if ( symmetric )
  {
    Axis->SetPoint1(0,-length,0);
  }
  else
  {
    Axis->SetPoint1(0,0,0);
  }

  Axis->SetPoint2(0,length,0);
  
  Axis->Update();
  return Axis->GetOutput();
}
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeZAxes( const float& length , const bool& symmetric)
{
  vtkSmartPointer<vtkLineSource> Axis = vtkSmartPointer<vtkLineSource>::New();

  if ( symmetric )
  {
    Axis->SetPoint1(0,0,-length);
  }
  else
  {
    Axis->SetPoint1(0,0,0);
  }

  Axis->SetPoint2(0,0,length);
  
  Axis->Update();
  return Axis->GetOutput();
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeLapLensAxes()
{
  vtkSmartPointer<vtkLineSource> ZAxis = vtkSmartPointer<vtkLineSource>::New();
  vtkSmartPointer<vtkLineSource> XAxisLHC = vtkSmartPointer<vtkLineSource>::New();
  vtkSmartPointer<vtkLineSource> XAxisRHC = vtkSmartPointer<vtkLineSource>::New();
  vtkSmartPointer<vtkLineSource> YAxisLHC = vtkSmartPointer<vtkLineSource>::New();
  vtkSmartPointer<vtkLineSource> YAxisRHC = vtkSmartPointer<vtkLineSource>::New();

  ZAxis->SetPoint1(0,0,-2000);
  ZAxis->SetPoint2(0,0,2000);
  ZAxis->Update();

  XAxisLHC->SetPoint1(0,0,-10);
  XAxisLHC->SetPoint2(20,0,-10);
  XAxisLHC->Update();

  XAxisRHC->SetPoint1(0,0,10);
  XAxisRHC->SetPoint2(20,0,10);
  XAxisRHC->Update();

  YAxisLHC->SetPoint1(0,0,-10);
  YAxisLHC->SetPoint2(0,20,-10);
  YAxisLHC->Update();

  YAxisRHC->SetPoint1(0,0,10);
  YAxisRHC->SetPoint2(0,20,10);
  YAxisRHC->Update();

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();
#if VTK_MAJOR_VERSION <= 5
  appenderer->AddInput(ZAxis->GetOutput());
  appenderer->AddInput(XAxisLHC->GetOutput());
  appenderer->AddInput(XAxisRHC->GetOutput());
  appenderer->AddInput(YAxisLHC->GetOutput());
  appenderer->AddInput(YAxisRHC->GetOutput());
#else
  appenderer->AddInputData(ZAxis->GetOutput());
  appenderer->AddInputData(XAxisLHC->GetOutput());
  appenderer->AddInputData(XAxisRHC->GetOutput());
  appenderer->AddInputData(YAxisLHC->GetOutput());
  appenderer->AddInputData(YAxisRHC->GetOutput());
#endif
  appenderer->Update();
  return appenderer->GetOutput();
}
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeOptotrak( const float & width, bool Polaris)
{
  vtkSmartPointer<vtkCylinderSource> topBar1 = vtkSmartPointer<vtkCylinderSource>::New();
  vtkSmartPointer<vtkCylinderSource> topBar2 = vtkSmartPointer<vtkCylinderSource>::New();
  vtkSmartPointer<vtkCylinderSource> neck = vtkSmartPointer<vtkCylinderSource>::New();
  vtkSmartPointer<vtkCylinderSource> eye = vtkSmartPointer<vtkCylinderSource>::New();
  vtkSmartPointer<vtkSphereSource> leftEye = vtkSmartPointer<vtkSphereSource>::New();
  vtkSmartPointer<vtkSphereSource> rightEye = vtkSmartPointer<vtkSphereSource>::New();

  topBar1->SetRadius(50);
  topBar1->SetHeight(width);
  topBar1->CappingOn();
  topBar1->SetResolution(20);
  topBar1->Update();

  topBar2->SetRadius(50);
  topBar2->SetHeight(width);
  topBar2->CappingOn();
  topBar2->SetResolution(20);
  topBar2->Update();

  neck->SetRadius(50);
  neck->SetHeight(200);
  neck->CappingOn();
  neck->SetResolution(20);
  neck->Update();

  eye->SetRadius(60);
  eye->SetHeight(100);
  eye->CappingOff();
  eye->SetResolution(20);
  eye->Update();

  leftEye->SetRadius(60);
  leftEye->SetThetaResolution(8);
  leftEye->SetPhiResolution(8);
  leftEye->SetCenter(0, -width,-20);
  leftEye->Update();

  rightEye->SetRadius(60);
  rightEye->SetThetaResolution(8);
  rightEye->SetPhiResolution(8);
  rightEye->SetCenter(0, width,-20);
  rightEye->Update();

  vtkSmartPointer<vtkTransform> topBar1Transform = vtkSmartPointer<vtkTransform>::New();
  vtkSmartPointer<vtkTransform> topBar2Transform = vtkSmartPointer<vtkTransform>::New();
  vtkSmartPointer<vtkTransform> neckTransform = vtkSmartPointer<vtkTransform>::New();
  vtkSmartPointer<vtkTransform> eyeTransform = vtkSmartPointer<vtkTransform>::New();

  topBar1Transform->Translate(0, (width/2 + 50),0);
  topBar2Transform->Translate(0,-(width/2 + 50),0);
  if ( Polaris )
  {
    neckTransform->RotateZ(-90);
  }
  else
  {
    neckTransform->RotateZ(90);
  }
  neckTransform->Translate(0,150,0);
  eyeTransform->RotateX(90);
  
  TranslatePolyData(topBar1->GetOutput(),topBar1Transform);
  TranslatePolyData(topBar2->GetOutput(),topBar2Transform);
  TranslatePolyData(neck->GetOutput(),neckTransform);
  TranslatePolyData(eye->GetOutput(),eyeTransform);

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

#if VTK_MAJOR_VERSION <= 5
  appenderer->AddInput(topBar1->GetOutput());
  appenderer->AddInput(topBar2->GetOutput());
  appenderer->AddInput(neck->GetOutput());
  appenderer->AddInput(eye->GetOutput());
  appenderer->AddInput(leftEye->GetOutput());
  appenderer->AddInput(rightEye->GetOutput());
#else
  appenderer->AddInputData(topBar1->GetOutput());
  appenderer->AddInputData(topBar2->GetOutput());
  appenderer->AddInputData(neck->GetOutput());
  appenderer->AddInputData(eye->GetOutput());
  appenderer->AddInputData(leftEye->GetOutput());
  appenderer->AddInputData(rightEye->GetOutput());
#endif

  appenderer->Update();
  return appenderer->GetOutput();

}
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeTransrectalUSProbe(std::string handeyeFilename )
{
  vtkSmartPointer<vtkCylinderSource> body = vtkSmartPointer<vtkCylinderSource>::New();
  vtkSmartPointer<vtkCylinderSource> cowl = vtkSmartPointer<vtkCylinderSource>::New();
  vtkSmartPointer<vtkSphereSource> transducer = vtkSmartPointer<vtkSphereSource>::New();
  vtkSmartPointer<vtkLineSource> projection1 = vtkSmartPointer<vtkLineSource>::New();
  vtkSmartPointer<vtkLineSource> projection2 = vtkSmartPointer<vtkLineSource>::New();
  vtkSmartPointer<vtkArcSource> projection3 = vtkSmartPointer<vtkArcSource>::New();

  body->SetRadius(8.0);
  body->SetHeight(100.0);
  body->SetCenter(51.696,-60,0);
  body->SetResolution(40);
  body->CappingOn();
  body->Update();
  
  cowl->SetRadius(16.17);
  cowl->SetHeight(20.0);
  cowl->SetCenter(45.696,-8.085, 0.0);
  cowl->SetResolution(40);
  cowl->CappingOn();
  cowl->Update();

  transducer->SetRadius(16.170);
  transducer->SetCenter(45.696, 0.0, 0.0);
  transducer->SetThetaResolution(40);
  transducer->SetPhiResolution(40);
  transducer->Update();

  projection1->SetPoint1(45.696, 0.0, 0.0);
  projection1->SetPoint2(0.0, 45.696, 0.0);
  projection1->Update();
  
  projection2->SetPoint1(45.696, 0.0, 0.0);
  projection2->SetPoint2(91.392, 45.696, 0.0);
  projection2->Update();

  projection3->SetPoint1(0.0,45.696, 0.0);
  projection3->SetPoint2(91.392, 45.696, 0.0);
  projection3->SetCenter(45.696,0.0,0.0);
  projection3->SetResolution(40);
  projection3->Update();

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

#if VTK_MAJOR_VERSION <= 5
  appenderer->AddInput(body->GetOutput());
  appenderer->AddInput(cowl->GetOutput());
  appenderer->AddInput(transducer->GetOutput());
  appenderer->AddInput(projection1->GetOutput());
  appenderer->AddInput(projection2->GetOutput());
  appenderer->AddInput(projection3->GetOutput());
#else
  appenderer->AddInputData(body->GetOutput());
  appenderer->AddInputData(cowl->GetOutput());
  appenderer->AddInputData(transducer->GetOutput());
  appenderer->AddInputData(projection1->GetOutput());
  appenderer->AddInputData(projection2->GetOutput());
  appenderer->AddInputData(projection3->GetOutput());
#endif
  appenderer->Update();

  vtkSmartPointer<vtkMatrix4x4> handeye = LoadMatrix4x4FromFile(handeyeFilename, false);
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  transform->SetMatrix(handeye);

  TranslatePolyData(appenderer->GetOutput(),transform);

  appenderer->Update();
  return appenderer->GetOutput();

}
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeMonitor()
{
  vtkSmartPointer<vtkCubeSource> stick = vtkSmartPointer<vtkCubeSource>::New();
  vtkSmartPointer<vtkCylinderSource> base = vtkSmartPointer<vtkCylinderSource>::New();

  vtkSmartPointer<vtkCubeSource> screen =  vtkSmartPointer<vtkCubeSource>::New();

  stick->SetXLength(100);
  stick->SetYLength(340);
  stick->SetZLength(20);
  stick->SetCenter(0,-160,-15); 
  
  base->SetRadius(160);
  base->SetHeight(20.0);
  base->SetCenter(0.0,-340,-15);
  base->SetResolution(40);
  base->CappingOn();

  screen->SetXLength(640);
  screen->SetYLength(480);
  screen->SetZLength(10);
  screen->SetCenter(0,0,-5.0); 

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

  appenderer->AddInputData(stick->GetOutput());
  appenderer->AddInputData(base->GetOutput());
  appenderer->AddInputData(screen->GetOutput());

  return appenderer->GetOutput();

}

//-----------------------------------------------------------------------------
std::vector<std::vector <float > > VTKIGIGeometry::ReadRigidBodyDefinitionFile(std::string rigidBodyFilename)
{
  std::vector < std::vector <float > > returnVector;
  ifstream fin;
  fin.open(rigidBodyFilename.c_str());
  if ( ! fin ) 
  {
    std::cerr << "Failed to open " << rigidBodyFilename;
    return returnVector;
  }
  std::string line;
  std::vector <float> position;
  for ( int i = 0 ; i < 3 ; i++ )
  {
    position.push_back(0.0);
  }
  unsigned int counter;
  int views;
  while ( getline(fin,line) ) 
  {
    std::stringstream linestream(line);
    bool parseSuccess;
    parseSuccess = linestream >> counter >> position[0] >> position[1] >> position[2] >> views;
    if ( parseSuccess )
    {
      returnVector.push_back(position);
      if ( counter != returnVector.size() )
      {
        std::cerr << "Error reading " << rigidBodyFilename;
        return returnVector;
      }
    }
  }
  fin.close();
  return returnVector;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeIREDs(std::vector < std::vector <float> > IREDPositions, float Radius, int ThetaRes, int PhiRes )
{
  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();
  for ( int i = 0 ; i < IREDPositions.size() ; i ++ ) 
  {
    assert ( IREDPositions[i].size() == 3 );
    vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
    sphere->SetRadius(Radius);
    sphere->SetThetaResolution(ThetaRes);
    sphere->SetPhiResolution(PhiRes);
    sphere->SetCenter(IREDPositions[i][0],IREDPositions[i][1],IREDPositions[i][2]);
    sphere->Update();
#if VTK_MAJOR_VERSION <= 5
    appenderer->AddInput(sphere->GetOutput());
#else
    appenderer->AddInputData(sphere->GetOutput());
#endif
  }
  appenderer->Update();
  return appenderer->GetOutput();
}

//-----------------------------------------------------------------------------
std::vector <float>  VTKIGIGeometry::Centroid(std::vector < std::vector <float> > positions )
{
  assert ( positions.size() != 0 );

  unsigned int dimension = positions[0].size();
  std::vector <float> centroid;
  for ( unsigned int i = 0 ; i < dimension ; i ++ ) 
  {
    centroid.push_back(0.0);
  }

  for ( unsigned int d = 0 ; d < dimension ; d ++ ) 
  {
    for ( unsigned int i = 0 ; i < positions.size() ; i ++ ) 
    {
      centroid[d] += positions[i][d];
    }
    
    centroid[d] /= static_cast<float> (positions.size());
  }

  return centroid;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData>  VTKIGIGeometry::ConnectIREDs(std::vector < std::vector <float> > IREDPositions, bool isPointer )
{
  vtkSmartPointer<vtkPolyData> polyOut = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();
  assert ( IREDPositions.size() > 1 );
  assert ( IREDPositions[0].size() == 3 );
  if ( ! isPointer ) 
  {
    for ( unsigned int i = 0 ; i < IREDPositions.size () - 1 ; i ++ ) 
    {
      vtkSmartPointer<vtkLineSource> join = vtkSmartPointer<vtkLineSource>::New();
      join->SetPoint1 ( IREDPositions[i][0], IREDPositions[i][1], IREDPositions[i][2]);
      join->SetPoint2 ( IREDPositions[i+1][0], IREDPositions[i+1][1], IREDPositions[i+1][2]);
      join->Update();
#if VTK_MAJOR_VERSION <= 5
      appenderer->AddInput(join->GetOutput());
#else
      appenderer->AddInputData(join->GetOutput());
#endif
    }
  }
  else
  {
    assert ( IREDPositions.size() > 5 );
    //special case of pointer or reference
    vtkSmartPointer<vtkLineSource> join1 = vtkSmartPointer<vtkLineSource>::New();
    vtkSmartPointer<vtkLineSource> join2 = vtkSmartPointer<vtkLineSource>::New();
    vtkSmartPointer<vtkLineSource> join3 = vtkSmartPointer<vtkLineSource>::New();
    vtkSmartPointer<vtkLineSource> join4 = vtkSmartPointer<vtkLineSource>::New();

    join1->SetPoint1 ( IREDPositions[0][0], IREDPositions[0][1], IREDPositions[0][2]);
    join1->SetPoint2 ( IREDPositions[1][0], IREDPositions[1][1], IREDPositions[1][2]);
    join1->Update();

    join2->SetPoint1 ( IREDPositions[3][0], IREDPositions[3][1], IREDPositions[3][2]);
    join2->SetPoint2 ( IREDPositions[4][0], IREDPositions[4][1], IREDPositions[4][2]);
    join2->Update();
    
    join3->SetPoint1 ( IREDPositions[0][0], IREDPositions[0][1], IREDPositions[0][2]);
    join3->SetPoint2 ( IREDPositions[4][0], IREDPositions[4][1], IREDPositions[4][2]);
    join3->Update();
    
    join4->SetPoint1 ( IREDPositions[1][0], IREDPositions[1][1], IREDPositions[1][2]);
    join4->SetPoint2 ( IREDPositions[3][0], IREDPositions[3][1], IREDPositions[3][2]);
    join4->Update();
      
#if VTK_MAJOR_VERSION <= 5
    appenderer->AddInput(join1->GetOutput());
    appenderer->AddInput(join2->GetOutput());
    appenderer->AddInput(join3->GetOutput());
    appenderer->AddInput(join4->GetOutput());
#else
    appenderer->AddInputData(join1->GetOutput());
    appenderer->AddInputData(join2->GetOutput());
    appenderer->AddInputData(join3->GetOutput());
    appenderer->AddInputData(join4->GetOutput());
#endif
  }
  appenderer->Update();
  return appenderer->GetOutput();
} 
} //end namespace niftk
