#!/usr/bin/env python

import os.path
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File)


class InterSliceCorrelationPlotInputSpec(BaseInterfaceInputSpec):    
    in_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input image")
    bval_file = File(argstr="%s", exists=True, mandatory=False,
                        desc="Input bval file")
    
class InterSliceCorrelationPlotOutputSpec(TraitedSpec):
    out_file   = File(exists=False, genfile = True,
                      desc="Interslice correlation plot")

class InterSliceCorrelationPlot(BaseInterface):
    input_spec = InterSliceCorrelationPlotInputSpec
    output_spec = InterSliceCorrelationPlotOutputSpec
    _suffix = "_interslice_ncc"
    
    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.pdf'
        return os.path.abspath(outfile)
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs
    
    def _run_interface(self, runtime):
        # Load the original image
        nib_image = nib.load(self.inputs.in_file)
        data = nib_image.get_data()
        dim = data.shape
        vol_number = 1;
        if len(dim)>3:
            vol_number=dim[3]
        if self.inputs.bval_file:
            bvalues=np.loadtxt(self.inputs.bval_file)
            if len(bvalues) != vol_number:
                exit
        cc_values = np.zeros((vol_number,len(range(1,dim[2]-1))))
        b0_values = np.zeros((vol_number,len(range(1,dim[2]-1))))
        for v in range(vol_number):
            if vol_number>1:
                volume=data[:,:,:,v]
            else:
                volume=data[:,:,:]
            current_b0=False
            if self.inputs.bval_file:
                if bvalues[v]<20:
                    current_b0=True
            temp_array1=volume[:,:,0];
            voxel_number = temp_array1.size
            temp_array1=np.reshape(temp_array1,voxel_number)
            temp_array1 = (temp_array1 - np.mean(temp_array1)) / np.std(temp_array1)
            for z in range(1,dim[2]-1):
                temp_array2=np.reshape(volume[:,:,z],voxel_number)
                temp_array2 = (temp_array2 - np.mean(temp_array2)) / np.std(temp_array2)
                if current_b0==True:
                    cc_values[v,z-1]=np.nan
                    b0_values[v,z-1]=(np.correlate(temp_array1,temp_array2) / np.double(voxel_number))
                else:
                    b0_values[v,z-1]=np.nan
                    cc_values[v,z-1]=(np.correlate(temp_array1,temp_array2) / np.double(voxel_number))
                temp_array1=np.copy(temp_array2)
        fig = plt.figure(figsize = (vol_number/4,6))
        mask_cc_values = np.ma.masked_array(cc_values,np.isnan(cc_values))
        mask_b0_values = np.ma.masked_array(b0_values,np.isnan(b0_values))
        mean_cc_values=np.mean(mask_cc_values[:,:],axis=0)
        mean_b0_values=np.mean(mask_b0_values[:,:],axis=0)
        std_cc_values=np.std(mask_cc_values[:,:],axis=0)
        std_b0_values=np.std(mask_b0_values[:,:],axis=0)
        x_axis = np.array(range(1,dim[2]-1))
        mpl.rcParams['text.latex.unicode']=True
        plt.plot(x_axis , mean_cc_values , 'b-',
                         label='Mean non B0 $(\pm 3.5 \sigma)$')
        plt.plot(x_axis , mean_b0_values , 'r-',
                         label='Mean B0 $(\pm 3.5 \sigma)$')
        sigma_mul=3.5
        plt.fill_between(x_axis,
                         mean_cc_values - sigma_mul*std_cc_values,
                         mean_cc_values + sigma_mul*std_cc_values,
                         facecolor='b',
                         linestyle='dashed',
                         alpha=0.2)
        plt.fill_between(x_axis,
                         mean_b0_values - sigma_mul*std_b0_values,
                         mean_b0_values + sigma_mul*std_b0_values,
                         facecolor='r',
                         linestyle='dashed',
                         alpha=0.2)       
        for v in range(vol_number):
            current_b0=False
            current_label=False
            if self.inputs.bval_file:
                if bvalues[v]<20:
                    current_b0=True
            if current_b0==True:
                for z in range(1,dim[2]-1):
                    if b0_values[v,z-1] < mean_b0_values[z-1]-sigma_mul*std_b0_values[z-1]:
                        if current_label:
                            plt.plot(z, b0_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]))
                        else:
                            current_label=True
                            plt.plot(z, b0_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]),
                                     label='Volume '+str(v))
            else:
                for z in range(1,dim[2]-1):
                    if cc_values[v,z-1] < mean_cc_values[z-1]-sigma_mul*std_cc_values[z-1]:
                        if current_label:
                            plt.plot(z, cc_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]))
                        else:
                            current_label=True
                            plt.plot(z, cc_values[v,z-1], '.',
                                     color=(plt.cm.jet(v*255/vol_number)[0:3]),
                                     label='Volume '+str(v))
        plt.ylabel('Normalised cross-corelation')
        plt.xlabel('Slice')
        plt.title('Inter-slice normalised cross-correlation\n' + \
                  'Scan: '+os.path.basename(self.inputs.in_file))
        plt.xticks(np.arange(0, dim[2], 2.0))
        plt.ylim([0,1])
        plt.legend(loc='best', numpoints=1, fontsize='small')
        self.out_file = self._gen_output_filename(self.inputs.in_file)
        fig.savefig(self.out_file, format='PDF')
        plt.close()
        return runtime
        