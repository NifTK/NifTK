/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MIDASMORPHOLOGICALSEGMENTORCONTROLSIMPL_H
#define MIDASMORPHOLOGICALSEGMENTORCONTROLSIMPL_H

#include "ui_MIDASMorphologicalSegmentorViewControls.h"
#include "MorphologicalSegmentorPipelineParams.h"

class QAbstractButton;

/**
 * \class MIDASMorphologicalSegmentorViewControlsImpl
 * \brief Implements a few Qt specific things that are of no interest to the MITK view class.
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 */
class MIDASMorphologicalSegmentorViewControlsImpl : public QWidget, public Ui_MIDASMorphologicalSegmentorViewControls
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /// \brief Constructor.
  MIDASMorphologicalSegmentorViewControlsImpl();

  /// \brief Destructor.
  ~MIDASMorphologicalSegmentorViewControlsImpl();

  /// \brief Creates the GUI, initialising everything to off.
  void setupUi(QWidget*);

  /// \brief Get the current tab number.
  int GetTabNumber();

  /// \brief Set the current tab number, which enables and disables widgets appropriately.
  void SetTabNumber(int i);

  /// \brief Enables/disables all controls.
  void EnableControls(bool b);

  /// \brief Set the dialog according to relevant image data.
  void SetControlsByImageData(double lowestValue, double highestValue, int numberAxialSlices);

  /// \brief Set the dialog according to current parameter values
  void SetControlsByParameterValues(MorphologicalSegmentorPipelineParams &params);

signals:

  void ThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSlicerNumber);
  void ErosionsValuesChanged(double upperThreshold, int numberOfErosions);
  void DilationValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations);
  void RethresholdingValuesChanged(int boxSize);
  void TabChanged(int tabNumber);
  void OKButtonClicked();
  void CancelButtonClicked();
  void ClearButtonClicked();

protected slots:

  void OnThresholdLowerValueChanged(double);
  void OnThresholdUpperValueChanged(double);
  void OnAxialCuttoffSliderChanged(int);
  void OnAxialCuttoffSpinBoxChanged(int);
  void OnThresholdAcceptButtonClicked();
  void OnErosionsSliderChanged(int);
  void OnErosionsSliderMoved(int);
  void OnErosionsAcceptButtonClicked();
  void OnErosionsResetButtonClicked();
  void OnDilationsSliderChanged(int);
  void OnDilationsSliderMoved(int);
  void OnDilationsAcceptButtonClicked();
  void OnDilationsResetButtonClicked();
  void OnRethresholdingSliderChanged(int);
  void OnRethresholdingSliderMoved(int);
  void OnRethresholdingResetButtonClicked();
  void OnOKCancelClicked(QAbstractButton*);
  void OnClearButtonClicked();

protected:

private:

  void EnableTab1Thresholding(bool enable);
  void EnableTab2Erosions(bool enable);
  void EnableTab3Dilations(bool enable);
  void EnableTab4ReThresholding(bool enable);
  void EnableOKButton(bool enable);
  void EnableCancelButton(bool enable);
  void EnableResetButton(bool enable);
  void EnableByTabNumber(int i);

  void EmitThresholdingValues();
  void EmitErosionValues();
  void EmitDilationValues();
  void EmitRethresholdingValues();
};

#endif
