/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef UPDATEMIDASVIEWINGCONTROLSINFO_H
#define UPDATEMIDASVIEWINGCONTROLSINFO_H

/**
 * \class UpdateMIDASViewingControlsInfo
 * \brief Basically a struct with a constructor simply to pass the current slice,
 * magnification and orientation information to QmitkMIDASMultiViewEditor.
 */
class UpdateMIDASViewingControlsInfo
{
public:

  // current values
  int currentTime;
  int currentSlice;
  int currentMagnification;
  bool isAxial;
  bool isSagittal;
  bool isCoronal;

  // the ranges (so sliders have the correct range).
  int minTime;
  int maxTime;
  int minSlice;
  int maxSlice;
  int minMagnification;
  int maxMagnification;

  UpdateMIDASViewingControlsInfo()
  : currentTime(0)
  , currentSlice(0)
  , currentMagnification(0)
  , isAxial(false)
  , isSagittal(false)
  , isCoronal(false)
  , minTime(0)
  , maxTime(0)
  , minSlice(0)
  , maxSlice(0)
  , minMagnification(0)
  , maxMagnification(0)
  {
  }
};

#endif // UPDATEMIDASVIEWINGCONTROLSINFO_H
