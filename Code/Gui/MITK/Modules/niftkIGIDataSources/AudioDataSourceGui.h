/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef AudioDataSourceGui_h

#include "niftkIGIDataSourcesExports.h"
#include <QmitkIGIDataSourceGui.h>
#include "ui_AudioDataSourceGui.h"

class QWidget;


class NIFTKIGIDATASOURCES_EXPORT AudioDataSourceGui : public QmitkIGIDataSourceGui, public Ui_AudioDataSourceGui
{
  Q_OBJECT

public:
  mitkClassMacro(AudioDataSourceGui, QmitkIGIDataSourceGui);
  itkNewMacro(AudioDataSourceGui);


  /** @see QmitkIGIDataSourceGui::Update() */
  virtual void Update();

  /** @see QmitkIGIDataSourceGui::Initialize(QWidget*) */
  virtual void Initialize(QWidget* parent);


protected:
  AudioDataSourceGui();
  virtual ~AudioDataSourceGui();


protected slots:
  /** @see QComboBox::currentIndexChanged(const QString&) */
  void OnCurrentDeviceIndexChanged(const QString& text);
  /** @see QComboBox::currentIndexChanged(int) */
  void OnCurrentFormatIndexChanged(int index);


private:
  AudioDataSourceGui(const AudioDataSourceGui& copyme);
  AudioDataSourceGui& operator=(const AudioDataSourceGui& assignme);


};

#endif // AudioDataSourceGui_h
