import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import csv


def read_stats_csv(filename, no_lines, no_cols):
    stats = np.zeros((no_lines,no_cols))
    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        row_index = 0
        for row in reader:
            for i in range(no_cols):
                stats[row_index,i] = row[i]
            row_index = row_index + 1
        #print stats
    return stats
    
# the base directory of the results sinked from nipype
results_base_dir = 'results/hellhole/'
# How many iterations of the pipeline were performed
no_itr = 2
no_dir = 68
no_labels = 49
row_no = 3
col_no = 2

# How many variants, i.e. runtypes, do we have?
no_variants = 4

labels = ['genu', 'body', 'splenium', 'fornix', 'cingulum (l)', 'cingulum (r)']
rois = [3,4,5,6,36,35]

base_res = 'dti_likelihood_study'
variant_name_list = []

dwi_summary = np.zeros((no_variants,len(rois),no_itr*no_dir))
proc_residual_summary = np.zeros((no_variants,len(rois),no_itr*no_dir))
tensor_summary = np.zeros((no_variants,len(rois),no_itr))
trans_summary = np.zeros((no_variants,no_itr*no_dir))
variant_index = 0
for i in ['_log','']:
    for j in ['_LIN','_CUB']:
        # This gives us the variant we want to test, we need to load the statistics from each of the
        # iterations, and compile them together, make some empty numpy arrays
        
        dwi_res = np.zeros((no_dir*no_itr*no_labels,4))
        proc_residual_res = np.zeros((no_dir*no_itr*no_labels,4))
        trans_res = np.zeros((no_dir*no_itr,1))
        tensor_res = np.zeros((no_itr*no_labels,4))
        for itr in range(no_itr):
            results_dir = results_base_dir+base_res+i+j+'_'+str(itr)
            print results_dir
            
            dwi_res[range(no_dir*itr*no_labels, no_dir*(itr+1)*no_labels),:] = read_stats_csv(results_dir+'/dwi_stats.csv',no_dir*no_labels,4)
            proc_residual_res[range(no_dir*itr*no_labels, no_dir*(itr+1)*no_labels),:] = read_stats_csv(results_dir+'/proc_residual_stats.csv',no_dir*no_labels,4)
            trans_res[range(no_dir*itr, no_dir*(itr+1)),:] = read_stats_csv(results_dir+'/affine_stats.csv',no_dir,1)
            tensor_res[range(itr*no_labels, (itr+1)*no_labels),:] = read_stats_csv(results_dir+'/tensor_stats.csv',no_labels,4)
        # Once we've accumulated all the results into an array, find the indicies of the rois of index
        roi_index = 0
        for roi in rois:
            indicies = dwi_res[:,0] == roi
            dwi_summary[variant_index,roi_index,:] = dwi_res[indicies,1]/dwi_res[indicies,3]
            proc_residual_summary[variant_index,roi_index,:] = proc_residual_res[indicies,1]/proc_residual_res[indicies,3]
            indicies = tensor_res[:,0] == roi
            tensor_summary[variant_index,roi_index,:] = tensor_res[indicies,1]/tensor_res[indicies,3]
            roi_index = roi_index + 1
        trans_summary[variant_index, :] = trans_res[:,0]
        #print dwi_res[range(no_dir*itr*no_labels, no_dir*(itr+1)*no_labels),:]
        
        variant_name_list.append(i+j)
        variant_index = variant_index + 1
        
plt.figure()
for i in range(len(rois)):
    ax = plt.subplot( row_no, col_no,(i % (row_no*col_no))+1)
    ax.boxplot(dwi_summary[:,i,:].transpose())
    plt.setp(ax, xticklabels=(variant_name_list))
    plt.title('MSE of forward model '+labels[i])
    
plt.figure()
for i in range(len(rois)):
    ax = plt.subplot( row_no, col_no,(i % (row_no*col_no))+1)
    ax.boxplot(proc_residual_summary[:,i,:].transpose())
    plt.setp(ax, xticklabels=(variant_name_list))
    plt.title('MSE of forward model '+labels[i])

plt.figure()
ax = plt.subplot(111)
ax.boxplot(trans_summary[range(4),:].transpose())
plt.setp(ax, xticklabels=(variant_name_list[0:4]))
plt.title('log trans l2 '+labels[i])

plt.figure()
for i in range(len(rois)):
    ax = plt.subplot( row_no, col_no,(i % (row_no*col_no))+1)
    ax.boxplot(tensor_summary[:,i,:].transpose())
    plt.setp(ax, xticklabels=(variant_name_list))
    plt.title('log l2 tensor '+labels[i])
    
plt.figure()
for i in range(len(rois)):
    ax = plt.subplot( row_no, col_no,(i % (row_no*col_no))+1)
    for j in range(no_itr):
        # Use black for the original data MSE and red for the processed data MSE
        plt.plot(tensor_summary[:,i,j], np.mean(dwi_summary[:,i,range(j*no_dir, (j+1)*no_dir)],1), 'xk', markersize=10)
        plt.plot(tensor_summary[:,i,j], np.mean(proc_residual_summary[:,i,range(j*no_dir, (j+1)*no_dir)],1), 'xr', markersize=10)
        plt.xlabel('Tensor distance (L2 norm log tensor)')
        plt.ylabel('MSE')
        plt.tight_layout()
        
# Do the same plot again, but don't separate out by different ROIs (easier to see trends)
plt.figure()
x_total = np.zeros((no_variants*len(rois)*no_itr))
y1_total = np.zeros((no_variants*len(rois)*no_itr))
y2_total = np.zeros((no_variants*len(rois)*no_itr))

ind = 0
for i in range(len(rois)):
    for j in range(no_itr):
        # Use black for the original data MSE and red for the processed data MSE
        plt.plot(tensor_summary[:,i,j], np.mean(dwi_summary[:,i,range(j*no_dir, (j+1)*no_dir)],1), 'xk', markersize=10)
        plt.plot(tensor_summary[:,i,j], np.mean(proc_residual_summary[:,i,range(j*no_dir, (j+1)*no_dir)],1), 'xr', markersize=10)
        tmp1 = np.mean(dwi_summary[:,i,range(j*no_dir, (j+1)*no_dir)],1)
        tmp2 = np.mean(proc_residual_summary[:,i,range(j*no_dir, (j+1)*no_dir)],1)        
        for k in range(no_variants):
            x_total[ind] = tensor_summary[k,i,j]
            y1_total[ind] = tmp1[k]
            y2_total[ind] = tmp2[k]
            ind = ind +1
        plt.xlabel('Tensor error (L2 norm log tensor)')
        plt.ylabel('MSE of forward model')
        plt.legend(['Original data', 'Processed data'])
        plt.tight_layout()
# Run a linear regression of the error of the tensor with respect to the original data MSE
lr1 = ss.linregress(x_total[np.isnan(x_total)==False],y1_total[np.isnan(x_total)==False])
# and the processed data
lr2 = ss.linregress(x_total[np.isnan(x_total)==False],y2_total[np.isnan(x_total)==False])
print "The slope of the observed data MSE with respect to the tensor error is "+str(lr1[0])+" with a p-value of "+str(lr1[3])
print "The slope of the processed data MSE with respect to the tensor error is "+str(lr2[0])+" with a p-value of "+str(lr2[3])
        

plt.figure()
x_total = np.zeros((no_variants*no_dir*no_itr))
y1_total = np.zeros((no_variants*no_dir*no_itr))
y2_total = np.zeros((no_variants*no_dir*no_itr))

ind = 0
# Plot the transformation error against the data likelihoods
for j in range(no_itr):
    x = trans_summary[:,range(j*no_dir, (j+1)*no_dir)].flatten()
    y1 = np.mean(dwi_summary[:,:,range(j*no_dir, (j+1)*no_dir)],1).flatten()
    y2 = np.mean(proc_residual_summary[:,:,range(j*no_dir, (j+1)*no_dir)],1).flatten()
    plt.plot( x,y1, 'xk', markersize=10)
    plt.plot(x, y2, 'xr', markersize=10)
    for k in range(no_variants*no_dir):
        x_total[ind] = x[k]
        y1_total[ind] = y1[k]
        y2_total[ind] = y2[k]
        ind = ind + 1
    plt.xlabel('Transformation error (L2 norm log matrix)')
    plt.ylabel('MSE of the forward model')
    plt.legend(['Original data', 'Processed data'])


# Run a linear regression of the error of the tensor with respect to the original data MSE
lr1 = ss.linregress(x_total[np.isnan(x_total)==False],y1_total[np.isnan(x_total)==False])
# and the processed data
lr2 = ss.linregress(x_total[np.isnan(x_total)==False],y2_total[np.isnan(x_total)==False])
print "The slope of the observed data MSE with respect to the transformation error is "+str(lr1[0])+" with a p-value of "+str(lr1[3])
print "The slope of the processed data MSE with respect to the transformation error is "+str(lr2[0])+" with a p-value of "+str(lr2[3])

plt.show()