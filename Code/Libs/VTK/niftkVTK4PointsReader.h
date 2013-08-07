/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVTK4PointsReader_h
#define niftkVTK4PointsReader_h

#include "niftkVTKWin32ExportHeader.h"
#include <NifTKConfigure.h>
#include <vtkPolyDataAlgorithm.h>

namespace niftk
{

/**
 * \class VTK4PointsReader
 * \brief Reads points to a polydata object
 *
 * Reads a text file where
 * each line represents a point. The first 3 columns being the
 * coordinates and the 4th column being a weighting. At present the
 * weighting is discarded, though may be used to clip points. Now has
 * option to not read weights, so should work for a file with 3 columns
 * defining the point data.
 */
class NIFTKVTK_WINEXPORT VTK4PointsReader : public vtkPolyDataAlgorithm
{
public:

  static VTK4PointsReader* New();
  vtkTypeMacro(VTK4PointsReader, vtkPolyDataAlgorithm);

  void PrintSelf(ostream& os, vtkIndent indent);

  /**
   * \brief Set/Get the name of the file from which to read points.
   */
  vtkSetStringMacro(FileName);
  vtkGetStringMacro(FileName);

  /**
   * \brief turn on range clipping (discards points that fall outside the set range)
   * \param direction to clip in (0 = x, 1 = y , 2 = z, 3 = weight)
   * \param min The minimum clipping value
   * \param max The maximum clipping value
   */
  void SetClippingOn ( int direction, double min, double max );

  /**
   * \brief turn off range clipping (discards points that fall outside the set range)
   * \param direction Direction to clip in (0 = x, 1 = y , 2 = z, 3 = weight)
   */
  void SetClippingOff ( int direction );

  /**
   * \brief Set/Get whether to read the weights (4th) column. Set to false if the file only has three columns
   */
  vtkSetMacro(m_ReadWeights,bool);
  vtkGetMacro(m_ReadWeights,bool);

protected:

  VTK4PointsReader();
  ~VTK4PointsReader();

  char* FileName;

  int RequestData(vtkInformation*,
                  vtkInformationVector**,
                  vtkInformationVector*);
  
  bool          m_Clipping[4];
  double        m_Min[4];
  double        m_Max[4];
  bool          m_ReadWeights;
};

} // end namespace

#endif
