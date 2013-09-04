/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkFileIOUtils.h>
#include <niftkVTKFunctions.h>
#include <iostream>

namespace mitk {

bool LoadDoublesFromFile(const std::string& fileName, std::vector<double>& output)
{
  bool isSuccessful = false;

  if(fileName.size() > 0)
  {
    ifstream myfile(fileName.c_str());
    if (myfile.is_open())
    {
      bool finished = false;
      double value;
      output.clear();

      do
      {
        myfile >> value;
        if (!myfile.bad() && !myfile.eof() && !myfile.fail())
        {
          output.push_back(value);
        }
        else
        {
          finished = true;
        }
      } while (!finished);

      myfile.close();
      isSuccessful = true;
    }
  }
  return isSuccessful;
}


//-----------------------------------------------------------------------------
bool Load2DPointFromFile(const std::string& fileName, mitk::Point2D& point)
{
  bool isSuccessful = false;

  std::vector<double> pointData;
  isSuccessful = LoadDoublesFromFile(fileName, pointData);

  if (isSuccessful)
  {
    if (pointData.size() == 2)
    {
      point[0] = pointData[0];
      point[1] = pointData[1];
      isSuccessful = true;
    }
    else
    {
      isSuccessful = false;
    }
  }

  return isSuccessful;
}


//-----------------------------------------------------------------------------
bool Load3DPointFromFile(const std::string& fileName, mitk::Point3D& point)
{
  bool isSuccessful = false;

  std::vector<double> pointData;
  isSuccessful = LoadDoublesFromFile(fileName, pointData);

  if (isSuccessful)
  {
    if (pointData.size() == 3)
    {
      point[0] = pointData[0];
      point[1] = pointData[1];
      point[2] = pointData[2];
      isSuccessful = true;
    }
    else
    {
      isSuccessful = false;
    }
  }

  return isSuccessful;
}


//-----------------------------------------------------------------------------
vtkMatrix4x4* LoadVtkMatrix4x4FromFile(const std::string& fileName)
{
  return niftk::LoadMatrix4x4FromFile(fileName, true);
}


//-----------------------------------------------------------------------------
bool SaveVtkMatrix4x4ToFile (const std::string& fileName, const vtkMatrix4x4& matrix)
{
  bool isSuccessful = false;
  if (fileName.length() > 0)
  {
    isSuccessful = niftk::SaveMatrix4x4ToFile(fileName, matrix);
  }
  return isSuccessful;
}


} // end namespace
