/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMakeGeometry.h"
#include <mitkFileIOUtils.h>
#include <mitkIOUtil.h>

#include <vtkCubeSource.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeLaparoscope ( QString& rigidBodyFilename, const vtkMatrix4x4& handeye ) 
{}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakePointer ( QString& rigidBodyFilename, const vtkMatrix4x4& handeye ) 
{}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeReference ( QString& rigidBodyFilename, const vtkMatrix4x4& handeye ) 
{}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeAWall ( const int& whichwall, const float& size, 
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
      MITK_WARN << "Passed a bad number to MakeAWall : " << whichwall;
      return NULL;
    }
  }
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(wall->GetOutput());
  mitk::IOUtil::SaveSurface (surface,"/dev/shm/output.vtp");
  return surface;

}

//-----------------------------------------------------------------------------
std::vector<std::vector <float > > ReadRigidBodyDefinitionFile(std::string rigidBodyFilename)
{
  std::vector < std::vector <float > > returnVector;
  ifstream fin;
  fin.open(rigidBodyFilename.c_str());
  if ( ! fin ) 
  {
    MITK_ERROR << "Failed to open " << rigidBodyFilename;
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
        MITK_ERROR << "Error reading " << rigidBodyFilename;
        return returnVector;
      }
    }
  }
  fin.close();
  return returnVector;
}
    

