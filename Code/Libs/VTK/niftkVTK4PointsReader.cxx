/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkSimplePointsReader.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "niftkvtk4PointsReader.h"

#include <vtkCellArray.h>
#include <vtkObjectFactory.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <sstream>

vtkStandardNewMacro(niftkvtk4PointsReader);

//----------------------------------------------------------------------------
niftkvtk4PointsReader::niftkvtk4PointsReader()
:m_ReadWeights(true)
{
  this->FileName = 0;
  this->SetNumberOfInputPorts(0);
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    m_Clipping[i] = false;
  }
}

//----------------------------------------------------------------------------
niftkvtk4PointsReader::~niftkvtk4PointsReader()
{
  this->SetFileName(0);
}

//----------------------------------------------------------------------------
void niftkvtk4PointsReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  os << indent << "FileName: "
     << (this->FileName ? this->FileName : "(none)") << "\n";

}

//----------------------------------------------------------------------------
int niftkvtk4PointsReader::RequestData(vtkInformation*,
                                       vtkInformationVector**,
                                       vtkInformationVector* outputVector)
{
  // Make sure we have a file to read.
  if(!this->FileName)
  {
    vtkErrorMacro("A FileName must be specified.");
    return 0;
  }

  // Open the input file.
  ifstream fin(this->FileName);
  if(!fin)
  {
    vtkErrorMacro("Error opening file " << this->FileName);
    return 0;
  }

  // Allocate objects to hold points and vertex cells.
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> verts = vtkSmartPointer<vtkCellArray>::New();

  // Read points from the file.
  vtkDebugMacro("Reading points from file " << this->FileName);
  double x[3];
  double weight;
  std::string line;
  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      bool parseSuccess;
      if ( m_ReadWeights ) 
      {
        parseSuccess = linestream >> x[0] >> x[1] >> x[2] >> weight;
      }
      else
      {
        parseSuccess = linestream >> x[0] >> x[1] >> x[2];
      }
      if ( parseSuccess ) 
      {
        bool ok=true;
        for ( int i = 0 ; i < 3 ; i++)
        {
          if ( m_Clipping[i] && ( ( x[i] < m_Min[i] ) || ( x[i] > m_Max[i] )) )
          {
           ok = false;
          }
        }
        if ( m_ReadWeights )
        {
          if ( m_Clipping[3] && ( ( weight < m_Min[3] ) || ( weight > m_Max[3] )) )
          {
            ok = false;
          }
        }

        if ( ok )
        {
          vtkIdType id = points->InsertNextPoint(x);
          verts->InsertNextCell(1, &id);
        }
      }
    }
    else
    {
      std::cerr << "Skippinng Comments" << line << std::endl;
    }
  }
  vtkDebugMacro("Read " << points->GetNumberOfPoints() << " points.");

  // Store the points and cells in the output data object.
  vtkPolyData* output = vtkPolyData::GetData(outputVector);
  output->SetPoints(points);
  output->SetVerts(verts);

  return 1;
}
//-------------------------------------------------------------------------------
void niftkvtk4PointsReader::SetClippingOn (int direction, double min, double max)
{
  m_Clipping[direction] = true;
  m_Min[direction] = min;
  m_Max[direction] = max;
}
//-------------------------------------------------------------------------------
void niftkvtk4PointsReader::SetClippingOff (int direction)
{
  m_Clipping[direction]=false;
}
