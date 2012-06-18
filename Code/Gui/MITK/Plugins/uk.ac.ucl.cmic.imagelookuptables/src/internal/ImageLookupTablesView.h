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
 * \brief Provides a simple GUI enabling the selection of window, level, min, max
 * and a choice of many lookup tables.
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

  static const std::string DATA_MIN;
  static const std::string DATA_MAX;
  static const std::string DATA_MEAN;
  static const std::string DATA_STDDEV;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

protected slots:

  /// \brief Called when the min/max slider has changed, and updates the LevelWindow.
  void OnWindowBoundsChanged();

  /// \brief Called when the window/level slider has changed, and updates the LevelWindow.
  void OnLevelWindowChanged();

  /// \brief Called when the min/max intensity limits has changed, which affects the actual minimum and maximum values that the sliders can reach.
  void OnRangeChanged();

  /// \brief Called when the lookup table combo box value changes, trigger the lookup table to be swapped.
  void OnLookupTableComboBoxChanged(int comboBoxIndex);

  /// \brief Called when the reset button is pressed which will recalculate the minimum and maximum intensity values, and set the minimum and maximum value on the sliders.
  void OnResetButtonPressed();

private:

  /// \brief Creation of the connections of widgets to slots.
  void CreateConnections();

  /// \brief Called from OnNodeChanged to indicate that the window/level has changed, so this View needs to update.
  void LevelWindowChanged();

  /// \brief Called from OnNodeChanged to indicate a new image, and hence this View needs to update accordingly.
  void DifferentImageSelected(const mitk::DataNode* node, mitk::Image* image);

  /// \brief Makes the GUI controls reflect the values on the level window manager.
  void UpdateGuiFromLevelWindowManager();

  /// \brief Blocks signals from min/max, window/level widgets.
  void BlockMinMaxSignals(bool b);

  /// \brief Enables/Disables controls.
  void EnableControls(bool b);

  /// \brief Retrieve's the pref values from preference service, and stored in member variables.
  void RetrievePreferenceValues();

  /// \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Called when the data store thinks a node has changed (for whatever reason).
  virtual void NodeChanged(const mitk::DataNode* node);

  /// \brief Retrieve's the corresponding node for the image
  mitk::DataNode* FindNodeForImage(mitk::Image* image);

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

  /// \brief We contain a LookupTableManager containing vtkLookupTables loaded from resource system.
  LookupTableManager *m_LookupTableManager;

  /// \brief We create a mitkLevelWindowManager to correctly hook into the DataStorage for window/level.
  mitk::LevelWindowManager::Pointer m_LevelWindowManager;

  // Preferences.
  std::string m_InitialisationMethod;
  double m_PercentageOfRange;
  int m_Precision;

  // To Track the current image.
  mitk::DataNode::Pointer m_CurrentNode;
  mitk::Image::Pointer m_CurrentImage;
  mitk::LevelWindow m_CurrentLevelWindow;
  bool m_InUpdate;

  // Store a reference to the parent widget of this view.
  QWidget *m_Parent;
};
#endif // _IMAGELOOKUPTABLESVIEW_H_INCLUDED
