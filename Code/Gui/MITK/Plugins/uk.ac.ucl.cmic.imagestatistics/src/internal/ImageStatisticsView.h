/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : $Author$

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 


#ifndef ImageStatisticsView_h
#define ImageStatisticsView_h

#include <berryISelectionListener.h>

#include <QmitkAbstractView.h>

#include "ui_ImageStatisticsViewControls.h"


/*!
 * \class ImageStatisticsView
 * \brief Provides simple image statistics over an image, or a region of interest.
 *
 * This class will refresh, either by directly hitting the update button, or if
 * the preference to auto-update is set to true, then stats are automatically updated
 * when a valid selection is made in the DataManager. A valid selection is 1 or 2 images. If only 1 image is selected,
 * the stats are computed over the whole image. If 2 images are selected, the first is assumed to be
 * the reference data, and the second is assumed to be a mask.  The mask does not have to be binary.
 * If the mask is non-binary, the statistics are computed for every non-zero label type.
 * If the preference "assume a binary mask", then regardless of how many values are in the mask image,
 * anything that is not the background value (default 0) is assumed to be foreground.
 *
 * \sa QmitkAbstractView
 * \ingroup uk_ac_ucl_cmic_imagestatistics_internal
*/
class ImageStatisticsView : public QmitkAbstractView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
  public:  

    berryObjectMacro(ImageStatisticsView);
    ImageStatisticsView();
    virtual ~ImageStatisticsView();

    static const std::string VIEW_ID;

    virtual void CreateQtPartControl(QWidget *parent);

    /// \brief called by QmitkAbstractView when DataManager's selection has changed
    virtual void OnSelectionChanged( berry::IWorkbenchPart::Pointer source,
                                     const QList<mitk::DataNode::Pointer>& nodes );

    /// \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
    virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  protected slots:
  
    /// \brief Checks to see if there is a valid selection, and if so, triggers Update with the currently selected nodes.
    void TryUpdate();

  protected:
  
    virtual void SetFocus();

  private:

    /// \brief Retrieves the preferences, and sets the private member variables accordingly.
    void RetrievePreferenceValues();

    /// \brief Enables/Disables all widget controls in m_Controls.
    void EnableControls(bool enabled);

    /// \brief Checks the GUI selection.
    bool IsSelectionValid(const QList<mitk::DataNode::Pointer>& nodes);

    /// \brief Used to clear the table, and create appropriate headers.
    void InitializeTable();

    /// \brief Called when the user clicks the GUI "update" button, or when the selection changed.
    void Update(const QList<mitk::DataNode::Pointer>& nodes);

    /// \brief Used to add a single row.
    template <typename PixelType>
    void AddTableRow(
        int row,
        QString &value,
        PixelType &min,
        PixelType &max,
        double &mean,
        double &stdDev,
        unsigned long int &count,
        double &volume);

    /// Gets a set of labels from a mask image.
    template <typename PixelType, unsigned int VImageDimension>
    void GetLabelValues(
        itk::Image<PixelType, VImageDimension>* itkImage,
        std::set<PixelType> &labels);


    /// \brief Calculates the voxel volume.
    template <typename PixelType, unsigned int VImageDimension>
    void GetVoxelVolume(
        itk::Image<PixelType, VImageDimension>* itkImage,
        double &volume
        );

    /// \brief Used to check value against min, max etc.
    template <typename TPixel>
    void TestMinAndMax(
        TPixel &imageValue,
        TPixel &min,
        TPixel &max
        );

    /// \brief Used to accumulate, mean, and s_0, s_1, s_2
    /// \see http://en.wikipedia.org/wiki/Standard_deviation
    template <typename TPixel>
    void AccumulateData(
        TPixel &imageValue,
        double &mean,
        double &s0,
        double &s1,
        double &s2,
        unsigned long int &counter
        );

    /// \brief Used to set the values to initial values such as zero.
    template <typename TPixel>
    void InitializeData(
        TPixel &min,
        TPixel &max,
        double &mean,
        double &s0,
        double &s1,
        double &s2,
        double &stdDev,
        unsigned long int &counter
        );

    /// \brief Does final calculation of mean and stddev.
    void CalculateMeanAndStdDev(
        double &mean,
        double &s0,
        double &s1,
        double &s2,
        double &stdDev,
        unsigned long int &counter
        );

    /// \brief Used to check value against min, max etc.
    template <typename TPixel>
    void AccumulateValue(
        TPixel &imageValue,
        TPixel &min,
        TPixel &max,
        double &mean,
        double &s0,
        double &s1,
        double &s2,
        unsigned long int &counter
        );

    /// \brief Used to check value against min, max etc.
    template <typename TPixel1, typename TPixel2, typename LabelType>
    void AccumulateValue(
        bool    &invert,
        LabelType &valueToCompareMaskAgainst,
        TPixel1 &imageValue,
        TPixel2 &maskValue,
        TPixel1 &min,
        TPixel1 &max,
        double  &mean,
        double  &s0,
        double  &s1,
        double  &s2,
        unsigned long int &counter
        );

    /// See: http://docs.mitk.org/nightly-qt4/group__Adaptor.html
    /// Specifically: http://docs.mitk.org/nightly-qt4/group__Adaptor.html#gaf4672e81ea40d0683dfcf996e788ca98
    /// \brief Updates the stats in the table.
    template <typename TPixel, unsigned int VImageDimension>
    void UpdateTable(
        itk::Image<TPixel, VImageDimension>* itkImage
        );

    /// See: http://docs.mitk.org/nightly-qt4/group__Adaptor.html
    /// Specifically: http://docs.mitk.org/nightly-qt4/group__Adaptor.html#ga0f12c9f206cd8e385bfaaff8afeb73c7
    /// \brief Updates the stats in the table.
    template <typename TPixel1, unsigned int VImageDimension1, typename TPixel2, unsigned int VImageDimension2>
    void UpdateTableWithMask(
        itk::Image<TPixel1, VImageDimension1>* itkImage,
        itk::Image<TPixel2, VImageDimension2>* itkMask
        );

    Ui::ImageStatisticsViewControls m_Controls;
    bool                            m_AutoUpdate;
    bool                            m_RequireSameSizeImage;
    bool                            m_AssumeBinary;
    int                             m_BackgroundValue;
    mitk::DataNode::Pointer         m_MaskNode;
    mitk::DataNode::Pointer         m_ImageNode;

};

#endif // ImageStatisticsView_h

