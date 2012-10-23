/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _IMAGELOOKUPTABLESVIEW_H_INCLUDED
#define _IMAGELOOKUPTABLESVIEW_H_INCLUDED

#include "ui_ImageLookupTablesViewControls.h"

#include "berryQtViewPart.h"
#include "berryIBerryPreferences.h"
#include "QmitkAbstractView.h"
#include "mitkLevelWindowManager.h"
#include "mitkDataNode.h"

class QWidget;
class QSlider;
class LookupTableManager;

/**
 * \class ImageLookupTablesView
 * \brief Provides a simple GUI enabling the selection of window, level, min, max and a choice of lookup tables.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 */
class ImageLookupTablesView : public QmitkAbstractView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  explicit ImageLookupTablesView();
  ImageLookupTablesView(const ImageLookupTablesView& other);
  virtual ~ImageLookupTablesView();

  /// \brief Each view for a plugin has its own globally unique ID.
  static const std::string VIEW_ID;

  /// \brief Used to store the property name "data min"
  static const std::string DATA_MIN;

  /// \brief Used to store the property name "data max"
  static const std::string DATA_MAX;

  /// \brief Used to store the property name "data mean"
  static const std::string DATA_MEAN;

  /// \brief Used to store the property name "data std dev"
  static const std::string DATA_STDDEV;

  /// \brief Called by framework, this method creates all the controls for this view.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief called by QmitkAbstractView when DataManager's selection has changed.
  virtual void OnSelectionChanged( berry::IWorkbenchPart::Pointer source,
                                   const QList<mitk::DataNode::Pointer>& nodes );

protected:

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

  /// \brief We listen to the Level/Window property on the registered image, so this callback
  /// updates this GUI when the watched property changes.
  virtual void OnPropertyChanged(const itk::EventObject&);

  /// \brief Actually updates the GUI when property changes.
  virtual void OnPropertyChanged();

protected slots:

  /// \brief Qt slot called when a different image is selected or the min/max intensity limits has changed,
  /// which affects the actual minimum and maximum values that the sliders can reach.
  void OnRangeChanged();

  /// \brief Qt slot called when the min/max slider has changed.
  void OnWindowBoundsChanged();

  /// \brief Qt slot called when the window/level slider has changed.
  void OnLevelWindowChanged();

  /// \brief Qt slot called when the lookup table combo box value changes, trigger the lookup table to be swapped.
  void OnLookupTableComboBoxChanged(int comboBoxIndex);

  /// \brief Qt slot called when the reset button is pressed which will recalculate the minimum and maximum intensity values, and set the minimum and maximum value on the sliders.
  void OnResetButtonPressed();

private:

  /// \brief Creation of the connections of widgets to slots.
  void CreateConnections();

  /// \brief Checks the GUI selection.
  bool IsSelectionValid(const QList<mitk::DataNode::Pointer>& nodes);

  /// \brief Registers the given node, as the one we are tracking.
  void Register(const mitk::DataNode::Pointer node);

  /// \brief Unregisters the given node.
  void Unregister();

  /// \brief Called when a node is successfully registered, to initialize/reset the controls to appropriate ranges.
  void DifferentImageSelected();

  /// \brief Makes the GUI controls reflect the values on the level window property.
  void UpdateGuiFromLevelWindow();

  /// \brief Blocks all signals.
  void BlockSignals(bool b);

  /// \brief Enables/Disables controls.
  void EnableControls(bool b);

  /// \brief Retrieve's the pref values from preference service, and stored in member variables.
  void RetrievePreferenceValues();

  /// \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Gets the stats using itkStatisticsImageFilter.
  template<typename TPixel, unsigned int VImageDimension>
  void ITKGetStatistics(
      itk::Image<TPixel, VImageDimension> *itkImage,
      float& min,
      float& max,
      float &mean,
      float &stdDev);

  /// \brief All the controls for the main view part.
  Ui::ImageLookupTablesViewControls* m_Controls;

  /// \brief We contain a LookupTableManager containing vtkLookupTables loaded from the resource system.
  LookupTableManager *m_LookupTableManager;

  /// \brief Tracks the currently selected node.
  mitk::DataNode::Pointer m_CurrentNode;

  /// \brief Tracks the currently selected image.
  mitk::Image::Pointer m_CurrentImage;

  /// \brief We provide several different ways to initialise the first window/level that an image has.
  std::string m_InitialisationMethod;

  /// \brief Stores the percentage of the maximum intensity range to initialise to.
  double m_PercentageOfRange;

  /// \brief Stores the precision, as you could have float images, with intensity range between 0 and 1.
  int m_Precision;

  /// \brief To stop re-entering the same piece of code recursively.
  bool m_InUpdate;

  /// \brief Used to affect the step size on the sliders/spin boxes.
  int m_ThresholdForIntegerBehaviour;

  /// \brief To store the
  unsigned long int m_PropertyObserverTag;

};
#endif // _IMAGELOOKUPTABLESVIEW_H_INCLUDED
