#include "niftkCaffePreferencesWidget.h"
#include <QCheckBox>
#include <QLineEdit>
#include <QSpinBox>
#include <QLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLabel>
#include <ctkPathLineEdit.h>

namespace niftk
{

const QString CaffePreferencesWidget::NETWORK_DESCRIPTION_FILE_NAME("network description");
const QString CaffePreferencesWidget::DEFAULT_NETWORK_DESCRIPTION_FILE("");
const QString CaffePreferencesWidget::NETWORK_WEIGHTS_FILE_NAME("network weights");
const QString CaffePreferencesWidget::DEFAULT_NETWORK_WEIGHTS_FILE("");
const QString CaffePreferencesWidget::DO_TRANSPOSE_NAME("do transpose");
const bool    CaffePreferencesWidget::DEFAULT_DO_TRANSPOSE(true);
const QString CaffePreferencesWidget::INPUT_LAYER_NAME("input layer");
const QString CaffePreferencesWidget::DEFAULT_INPUT_LAYER("data");
const QString CaffePreferencesWidget::OUTPUT_BLOB_NAME("output blob");
const QString CaffePreferencesWidget::DEFAULT_OUTPUT_BLOB("prediction");
const QString CaffePreferencesWidget::GPU_DEVICE_NAME("GPU device");
const int     CaffePreferencesWidget::DEFAULT_GPU_DEVICE(-1);

//-----------------------------------------------------------------------------
CaffePreferencesWidget::CaffePreferencesWidget(QWidget * parent)
: QWidget(parent)
{
  m_UiLayout = new QVBoxLayout();  
  QFormLayout * formLayout = new QFormLayout();
  m_NetworkDescriptionFileName = new ctkPathLineEdit();
  formLayout->addRow("Network description file name", m_NetworkDescriptionFileName);

  m_NetworkWeightsFileName = new ctkPathLineEdit();
  formLayout->addRow("Network weights file name", m_NetworkWeightsFileName);

  m_DoTranspose = new QCheckBox();
  m_DoTranspose->setChecked(true);
  formLayout->addRow("Transpose input/output data", m_DoTranspose);

  m_NameMemoryLayer = new QLineEdit();
  formLayout->addRow("Input layer name", m_NameMemoryLayer);

  m_NameOutputBlob = new QLineEdit();
  formLayout->addRow("Output blob name", m_NameOutputBlob);

  m_GPUDevice = new QSpinBox();
  m_GPUDevice->setMinimum(-1);
  m_GPUDevice->setMaximum(10);
  formLayout->addRow("GPU device", m_GPUDevice);

  m_UiLayout->addLayout(formLayout);

  m_UiLayout->addStretch();
}


//-----------------------------------------------------------------------------
CaffePreferencesWidget::~CaffePreferencesWidget()
{}


//-----------------------------------------------------------------------------
QLayout * CaffePreferencesWidget::GetUILayout()
{
  return m_UiLayout;
}


//-----------------------------------------------------------------------------
QString CaffePreferencesWidget::GetNetworkDescriptionFileName() const
{
  return m_NetworkDescriptionFileName->currentPath();
}


//-----------------------------------------------------------------------------
void CaffePreferencesWidget::SetNetworkDescriptionFileName(const QString & path)
{
  m_NetworkDescriptionFileName->setCurrentPath(path);
}


//-----------------------------------------------------------------------------
QString CaffePreferencesWidget::GetNetworkWeightsFileName() const
{
  return m_NetworkWeightsFileName->currentPath();
}


//-----------------------------------------------------------------------------
void CaffePreferencesWidget::SetNetworkWeightsFileName(const QString & path)
{
  m_NetworkWeightsFileName->setCurrentPath(path);
}


//-----------------------------------------------------------------------------
bool CaffePreferencesWidget::GetDoTranspose() const
{
  return m_DoTranspose->isChecked();
}


//-----------------------------------------------------------------------------
void CaffePreferencesWidget::SetDoTranspose(bool t)
{
  m_DoTranspose->setChecked(t);
}


//-----------------------------------------------------------------------------
QString CaffePreferencesWidget::GetMemoryLayerName() const
{
  return m_NameMemoryLayer->text();
}


//-----------------------------------------------------------------------------
void CaffePreferencesWidget::SetMemoryLayerName(const QString & text)
{
  return m_NameMemoryLayer->setText(text);
}


//-----------------------------------------------------------------------------
QString CaffePreferencesWidget::GetOutputBlobName() const
{
  return m_NameOutputBlob->text();
}


//-----------------------------------------------------------------------------
void CaffePreferencesWidget::SetOutputBlobName(const QString & text)
{
  m_NameOutputBlob->setText(text);
}


//-----------------------------------------------------------------------------
int CaffePreferencesWidget::GetGPUDevice() const
{
  return m_GPUDevice->value();
}


//-----------------------------------------------------------------------------
void CaffePreferencesWidget::SetGPUDevice(int device)
{
  m_GPUDevice->setValue(device);
}

}
