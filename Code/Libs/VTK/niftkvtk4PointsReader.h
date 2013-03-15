/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkvtk4PointsReader.h
 * \page niftkvtk4PointsReader
 * \section niftkvtk4PointsReaderSummary Reads points to a polydata object
 *
 * Reads a text file where
 * each line represents a point. The first 3 columns being the 
 * coordinates and the 4th column being a weighting. At present the
 * weighting is discarded.
 */

#ifndef __niftkvtk4PointsReader_h
#define __niftkvtk4PointsReader_h

#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"
#include "niftkVTKWin32ExportHeader.h"

#include "vtkPolyDataAlgorithm.h"

class NIFTKVTK_WINEXPORT niftkvtk4PointsReader : public vtkPolyDataAlgorithm
{
public:
  static niftkvtk4PointsReader* New();
  vtkTypeMacro(niftkvtk4PointsReader,vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/Get the name of the file from which to read points.
  vtkSetStringMacro(FileName);
  vtkGetStringMacro(FileName);

protected:
  niftkvtk4PointsReader();
  ~niftkvtk4PointsReader();

  char* FileName;

  int RequestData(vtkInformation*,
                  vtkInformationVector**,
                  vtkInformationVector*);
};

#endif
