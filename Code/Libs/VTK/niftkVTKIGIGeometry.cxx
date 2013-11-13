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
  lensCyl->SetRadius(20.0);
  lensCyl->SetHeight(80.0);
  lensCyl->SetCenter(0.0,-40.0,0.0);
  lensCyl->SetResolution(6);
  lensCowl=lensCyl->GetOutput();
  TranslatePolyData(lensCowl,transform);
 
  vtkSmartPointer<vtkPolyData> ireds = this->MakeIREDs(positions);
 
  vtkSmartPointer<vtkAppendPolyData> appenderer = vtkSmartPointer<vtkAppendPolyData>::New();

  appenderer->AddInput(ireds);
  appenderer->AddInput(lensCowl);

  //get the lens position
  return appenderer->GetOutput();
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakePointer ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> VTKIGIGeometry::MakeReference ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{}

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
    
} //end namespace niftk
