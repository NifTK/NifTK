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
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkAppendPolyData.h>
#include <vtkCylinderSource.h>
#include <vtkLineSource.h>
#include <vtkArcSource.h>

#include <sstream>
#include <cassert>
namespace niftk
{ 
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeLaparoscope ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  std::vector < std::vector <float> > positions = this->ReadRigidBodyDefinitionFile(rigidBodyFilename);
  vtkSmartPointer<vtkMatrix4x4> handeye = LoadMatrix4x4FromFile(handeyeFilename, false);
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  transform->SetMatrix(handeye);


  vtkSmartPointer<vtkPolyData> lensCowl = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkCylinderSource> lensCyl = vtkSmartPointer<vtkCylinderSource>::New();
  lensCyl->SetRadius(5.0);
  lensCyl->SetHeight(20.0);
  lensCyl->SetCenter(0.0,0.0,0.0);
  lensCyl->SetResolution(40);
  
  lensCowl=lensCyl->GetOutput();

  vtkSmartPointer<vtkTransform> tipTransform = vtkSmartPointer<vtkTransform>::New();
  tipTransform->RotateX(90.0);
  tipTransform->Translate(0,10,0);

  TranslatePolyData(lensCowl,tipTransform);
  TranslatePolyData(lensCowl,transform);
 
  vtkSmartPointer<vtkPolyData> ireds = this->MakeIREDs(positions);

  std::vector < float > lensOrigin; 
  lensOrigin.push_back(handeye->GetElement(0,3));
  lensOrigin.push_back(handeye->GetElement(1,3));
  lensOrigin.push_back(handeye->GetElement(2,3));

  std::vector< std::vector < float > > axis; 
  axis.push_back(this->Centroid(positions));
  axis.push_back(lensOrigin);


  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

  appenderer->AddInput(ireds);
  appenderer->AddInput(lensCowl);
  appenderer->AddInput(this->ConnectIREDs(axis));

  //get the lens position
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

  appenderer->AddInput(ireds);
  appenderer->AddInput(this->ConnectIREDs(positions, true));
  appenderer->AddInput(tipBall);
  appenderer->AddInput(this->ConnectIREDs(axis));

  //get the lens position
  return appenderer->GetOutput();

}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeReference ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  return this->MakePointer(rigidBodyFilename, handeyeFilename);
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

  XAxisLHC->SetPoint1(0,0,-10);
  XAxisLHC->SetPoint2(20,0,-10);

  XAxisRHC->SetPoint1(0,0,10);
  XAxisRHC->SetPoint2(20,0,10);

  YAxisLHC->SetPoint1(0,0,-10);
  YAxisLHC->SetPoint2(0,20,-10);

  YAxisRHC->SetPoint1(0,0,10);
  YAxisRHC->SetPoint2(0,20,10);

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();
  appenderer->AddInput(ZAxis->GetOutput());
  appenderer->AddInput(XAxisLHC->GetOutput());
  appenderer->AddInput(XAxisRHC->GetOutput());
  appenderer->AddInput(YAxisLHC->GetOutput());
  appenderer->AddInput(YAxisRHC->GetOutput());
  
  return appenderer->GetOutput();
}
//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeOptotrak( const float & width)
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

  topBar2->SetRadius(50);
  topBar2->SetHeight(width);
  topBar2->CappingOn();
  topBar2->SetResolution(20);

  neck->SetRadius(50);
  neck->SetHeight(200);
  neck->CappingOn();
  neck->SetResolution(20);

  eye->SetRadius(60);
  eye->SetHeight(100);
  eye->CappingOff();
  eye->SetResolution(20);

  leftEye->SetRadius(60);
  leftEye->SetThetaResolution(8);
  leftEye->SetPhiResolution(8);
  leftEye->SetCenter(0, -width,-20);

  rightEye->SetRadius(60);
  rightEye->SetThetaResolution(8);
  rightEye->SetPhiResolution(8);
  rightEye->SetCenter(0, width,-20);

  vtkSmartPointer<vtkTransform> topBar1Transform = vtkSmartPointer<vtkTransform>::New();
  vtkSmartPointer<vtkTransform> topBar2Transform = vtkSmartPointer<vtkTransform>::New();
  vtkSmartPointer<vtkTransform> neckTransform = vtkSmartPointer<vtkTransform>::New();
  vtkSmartPointer<vtkTransform> eyeTransform = vtkSmartPointer<vtkTransform>::New();

  topBar1Transform->Translate(0,300,0);
  topBar2Transform->Translate(0,-300,0);
  neckTransform->RotateZ(90);
  neckTransform->Translate(0,150,0);
  eyeTransform->RotateX(90);
  
  TranslatePolyData(topBar1->GetOutput(),topBar1Transform);
  TranslatePolyData(topBar2->GetOutput(),topBar2Transform);
  TranslatePolyData(neck->GetOutput(),neckTransform);
  TranslatePolyData(eye->GetOutput(),eyeTransform);

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

  appenderer->AddInput(topBar1->GetOutput());
  appenderer->AddInput(topBar2->GetOutput());
  appenderer->AddInput(neck->GetOutput());
  appenderer->AddInput(eye->GetOutput());
  appenderer->AddInput(leftEye->GetOutput());
  appenderer->AddInput(rightEye->GetOutput());

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
  
  cowl->SetRadius(16.17);
  cowl->SetHeight(20.0);
  cowl->SetCenter(45.696,-8.085, 0.0);
  cowl->SetResolution(40);
  cowl->CappingOn();

  transducer->SetRadius(16.170);
  transducer->SetCenter(45.696, 0.0, 0.0);
  transducer->SetThetaResolution(40);
  transducer->SetPhiResolution(40);

  projection1->SetPoint1(45.696, 0.0, 0.0);
  projection1->SetPoint2(0.0, 45.696, 0.0);
  
  projection2->SetPoint1(45.696, 0.0, 0.0);
  projection2->SetPoint2(91.392, 45.696, 0.0);

  projection3->SetPoint1(0.0,45.696, 0.0);
  projection3->SetPoint2(91.392, 45.696, 0.0);
  projection3->SetCenter(45.696,0.0,0.0);
  projection3->SetResolution(40);

  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

  appenderer->AddInput(body->GetOutput());
  appenderer->AddInput(cowl->GetOutput());
  appenderer->AddInput(transducer->GetOutput());
  appenderer->AddInput(projection1->GetOutput());
  appenderer->AddInput(projection2->GetOutput());
  appenderer->AddInput(projection3->GetOutput());

  vtkSmartPointer<vtkMatrix4x4> handeye = LoadMatrix4x4FromFile(handeyeFilename, false);
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  transform->SetMatrix(handeye);

  TranslatePolyData(appenderer->GetOutput(),transform);

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
    appenderer->AddInput(sphere->GetOutput());
  }
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
      appenderer->AddInput(join->GetOutput());
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

    join2->SetPoint1 ( IREDPositions[3][0], IREDPositions[3][1], IREDPositions[3][2]);
    join2->SetPoint2 ( IREDPositions[4][0], IREDPositions[4][1], IREDPositions[4][2]);
    
    join3->SetPoint1 ( IREDPositions[0][0], IREDPositions[0][1], IREDPositions[0][2]);
    join3->SetPoint2 ( IREDPositions[4][0], IREDPositions[4][1], IREDPositions[4][2]);
    
    join4->SetPoint1 ( IREDPositions[1][0], IREDPositions[1][1], IREDPositions[1][2]);
    join4->SetPoint2 ( IREDPositions[3][0], IREDPositions[3][1], IREDPositions[3][2]);
      
    appenderer->AddInput(join1->GetOutput());
    appenderer->AddInput(join2->GetOutput());
    appenderer->AddInput(join3->GetOutput());
    appenderer->AddInput(join4->GetOutput());
  }
  return appenderer->GetOutput();
} 
} //end namespace niftk
