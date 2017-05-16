/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCaffeSegPreferenesWidget_h
#define niftkCaffeSegPreferenesWidget_h

#include <niftkCaffeGuiExports.h>
#include <QWidget>

// Forward declarations
class QPushButton;
class ctkPathLineEdit;
class QCheckBox;
class QLineEdit;
class QSpinBox;
class QLayout;
class QVBoxLayout;


namespace niftk
{

class NIFTKCAFFEGUI_EXPORT CaffePreferencesWidget : public QWidget
{
  Q_OBJECT
public:

  CaffePreferencesWidget(QWidget *parent=0);
  virtual ~CaffePreferencesWidget();

  QLayout * GetUILayout();

  /**
   * \brief Stores the name of the preference node that contains the name of the network description file.
   */
  static const QString NETWORK_DESCRIPTION_FILE_NAME;

  static const QString DEFAULT_NETWORK_DESCRIPTION_FILE;

  /**
   * \brief Stores the name of the preference node that contains the name of the network weights file.
   */
  static const QString NETWORK_WEIGHTS_FILE_NAME;

  static const QString DEFAULT_NETWORK_WEIGHTS_FILE;

  /**
   * \brief Stores the name of the preference node that contains whether to transpose data.
   */
  static const QString DO_TRANSPOSE_NAME;

  static const bool DEFAULT_DO_TRANSPOSE;

  /**
   * \brief Stores the name of the preference node that contains the name of the input MemoryData layer.
   */
  static const QString INPUT_LAYER_NAME;

  static const QString DEFAULT_INPUT_LAYER;

  /**
   * \brief Stores the name of the preference node that contains the name of the output blob.
   */
  static const QString OUTPUT_BLOB_NAME;

  static const QString DEFAULT_OUTPUT_BLOB;

  /**
   * \brief Stores the name of the preference node that contains the integer ID of the GPU device.
   */
  static const QString GPU_DEVICE_NAME;

  static const int DEFAULT_GPU_DEVICE;

  QString GetNetworkDescriptionFileName() const;
  void SetNetworkDescriptionFileName(const QString & path);

  QString GetNetworkWeightsFileName() const;
  void SetNetworkWeightsFileName(const QString & path);

  bool GetDoTranspose() const;
  void SetDoTranspose(bool t);

  QString GetMemoryLayerName() const;
  void SetMemoryLayerName(const QString & text);

  QString GetOutputBlobName() const;
  void SetOutputBlobName(const QString & text);

  int GetGPUDevice() const;
  void SetGPUDevice(int device);

protected:
  ctkPathLineEdit*  m_NetworkDescriptionFileName;
  ctkPathLineEdit*  m_NetworkWeightsFileName;
  QCheckBox*        m_DoTranspose;
  QLineEdit*        m_NameMemoryLayer;
  QLineEdit*        m_NameOutputBlob;
  QSpinBox*         m_GPUDevice;
  QVBoxLayout*      m_UiLayout;

private:
  CaffePreferencesWidget(const CaffePreferencesWidget &) = delete;
  void operator=(const CaffePreferencesWidget&) = delete;
};

} // end namespace

#endif
