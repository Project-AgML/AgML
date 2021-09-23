#%%
from MaskRCNNTrainingPipeline import GrapeBunchDataset
import MaskRCNNTrainingPipeline as mrc
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import argparse
import torch
import torchvision
from scipy import stats
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

num_classes=2
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, rpn_nms_thresh=0.45, box_nms_thresh=0.25)
in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
    # and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
model.load_state_dict(torch.load('albu_state.pth'), strict=False)
model.eval()
dataset_test= GrapeBunchDataset('test', mrc.get_transform(train=False))
device=torch.device('cuda')
model.to(device)

x_count=[]
y_count=[]

x_area=[] # GT
y_area=[]   # predicted
#%%
for x in range(len(dataset_test)):
    im= dataset_test[x][0]
    annotations=dataset_test[x][1]
    gt_masks=annotations['masks'].numpy()
    #print(gt_masks.shape[0])
    x_count.append(gt_masks.shape[0])
    gt_area=0
    for i in range(gt_masks.shape[0]):
        gt_area=gt_area+np.sum(gt_masks[i])
    x_area.append(gt_area)
    #print(annotations)
    #masked_im=im
    results = model([im.to(device)])
    r = results[0]
    mask = r['masks'].cpu()
    #data = np.load('tr/masks/GH010078000144.npz')
    #mask = data['arr_0']
    #mask = np.moveaxis((mask == 1), (0,1,2), (1,2,0))
    scores = r['scores'].cpu().detach().numpy()
    #np.save(IMAGE_DIR+ims_names[x][:-4], mask)
    #colors = random_colors(mask.shape[0])
    #masked_im = dataset_test[x][0]

    #print(mask[0,0,:,:])
    #print(mask[0].shape)
    print(mask.shape)
    pred_area=0
    pred_counts=0
    for i in range(mask.shape[0]):
        npmask=mask[i,0,:,:].detach().numpy()
        #npmask=mask[i]
        #npmask[npmask!=1] = 0
        if scores[i]>.5:
            pred_counts=pred_counts+1
            pred_area=np.sum(npmask)+pred_area
            #masked_im = apply_mask(masked_im,npmask,colors[i])
    y_area.append(pred_area)
    y_count.append(pred_counts)
    #plt.imsave('new_final/'+str(x)+'_masked.png',masked_im)

# %%
plt.scatter(x_count,y_count)

slope, intercept, r_value, p_value, std_err = stats.linregress(x_count,y_count)

slope=np.round(slope,2)
intercept=np.round(intercept,2)

r_squared= r_value ** 2

plt.title('# of Ground Truth vs Predicted Grape Bunch Instances')

plt.xlabel('# of Ground Truth Instances')
plt.ylabel('# of Predicted Instances')

plt.text(16, 14, 'R-squared = %0.2f' % r_squared)
plt.text(12, 10, 'y='+str(slope)+'x + '+str(intercept))

x_plot = np.linspace(np.min(x_count),np.max(x_count),100)
plt.plot(x_plot,x_plot*slope + intercept,'k--')

plt.savefig('instances.png')

#%%


plt.scatter(x_area, y_area)

slope, intercept, r_value, p_value, std_err = stats.linregress(x_area,y_area)

slope=np.round(slope,2)
intercept=np.round(intercept,2)

r_squared= r_value ** 2

plt.title('Annotated Area vs Predicted Area')

plt.xlabel('Annotated Area')
plt.ylabel('Predicted Area')
plt.text(400000, 350000, 'R-squared = %0.2f' % r_squared)

plt.text(350000, 300000, 'y='+str(slope)+'x + '+str(intercept))

x_plot = np.linspace(np.min(x_area),np.max(x_area),100)
plt.plot(x_plot,x_plot*slope + intercept,'k--')
plt.tight_layout()

plt.savefig('area.png')


# %%
torch.save(model.state_dict(), 'albu_state.pth')

# %%
