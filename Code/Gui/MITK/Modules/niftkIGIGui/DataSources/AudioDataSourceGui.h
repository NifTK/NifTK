/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef AudioDataSourceGui_h

#include "niftkIGIGuiExports.h"
#include <DataSources/QmitkIGIDataSourceGui.h>
#include "ui_AudioDataSourceGui.h"

class QWidget;


class NIFTKIGIGUI_EXPORT AudioDataSourceGui : public QmitkIGIDataSourceGui, public Ui_AudioDataSourceGui
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


private:
  AudioDataSourceGui(const AudioDataSourceGui& copyme);
  AudioDataSourceGui& operator=(const AudioDataSourceGui& assignme);


};

#endif // AudioDataSourceGui_h
