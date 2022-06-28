import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

#from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
 
BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d #torch.nn.SyncBatchNorm
BatchNorm3d = torch.nn.BatchNorm3d

relu_inplace = True    

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BNReLU3D(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm3d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
'''

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale

        #self.apply(self._initialize_weights)

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

    
class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

        #self.apply(self._initialize_weights)

    def forward(self, x, proxy):
        #print('debug point alpha. proxy shape is :{}'.format(proxy.shape)) torch.Size([44, 256, 17, 1])
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)


    
class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)
        

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super().__init__()
        #self.object_context_block = ObjectAttentionBlock2D(in_channels, 
        #                                                   key_channels, 
        #                                                   scale, 
        #                                                   bn_type)
        self.object_context_block = _ObjectAttentionBlock(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

        #self.apply(self._initialize_weights)

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)
    
    

class SpatialGather_Module_3D(SpatialGather_Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super().__init__(cls_num, scale)

    def forward(self, feats, probs):
        batch_size, c, d, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3), probs.size(4)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x dhw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x dhw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock_3D(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W ->
        N X C X D X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W ->
        N X C X D X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock_3D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool3d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU3D(self.key_channels, bn_type=bn_type),
            nn.Conv3d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU3D(self.key_channels, bn_type=bn_type),
        )
        # f_object and f_down has 2D input, which is key input [11, 32, 17, 1]
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv3d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU3D(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, d, h, w = x.size(0), x.size(2), x.size(3), x.size(4)
        #print('debug point 1. x shape is: {}'.format(x.shape))
        #print('debug point 1. x shape is: {}'.format(self.f_pixel(x).shape))
        #x shape is: torch.Size([11, 32, 64, 64, 64])
        #print('debug point 1. proxy shape is: {}'.format(proxy.shape)) # torch.Size([11, 32, 17, 1])
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1) # [11, 32, 64*64*64]
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context 

class SpatialOCR_Module_3D(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super().__init__()
        self.object_context_block = _ObjectAttentionBlock_3D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv3d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU3D(out_channels, bn_type=bn_type),
            nn.Dropout3d(dropout)
        )

    def forward(self, feats, proxy_feats):
        #print('debug point 4. proxy_feats is : {}'.format(proxy_feats.shape))
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output
    


    
class OCR_2D_Module(nn.Module):

    # def _initialize_weights(self, m):
    #     classname = m.__class__.__name__
    #     # for every Linear layer in a model..
    #     if classname.find('Linear') != -1:
    #         # apply a uniform distribution to the weights and a bias=0
    #         m.weight.data.uniform_(0.0, 1.0)
    #         m.bias.data.fill_(0)

    def __init__(self,
                 last_inp_channels,
                 ocr_mid_channels,
                 ocr_key_channels,
                 ocr_dropout,
                 num_joints):
        super().__init__()
        
        self.conv3x3_ocr = nn.Sequential(
                    nn.Conv2d(last_inp_channels, ocr_mid_channels,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(ocr_mid_channels),
                    nn.ReLU(inplace=relu_inplace)
        )
        #originally,
        #nn.SyncBatchNorm(ocr_mid_channels) - yk.
        
        
        self.ocr_gather_head = SpatialGather_Module(num_joints) #oniginally, config.DATASET.NUM_CLASSES
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=ocr_dropout,
                                                 )
        
        
        #self._initialize_weights()
        #self.apply(self._initialize_weights)
        #self.conv3x3_ocr.apply(self._initialize_weights)
        #self.ocr_gather_head.apply(self._initialize_weights)
        #self.ocr_distri_head.apply(self._initialize_weights)
    
    def forward(self, features, heatmaps, return_context=False):
        
        features = self.conv3x3_ocr(features)
        context = self.ocr_gather_head(features, heatmaps)
        features = self.ocr_distri_head(features, context)
        
        if return_context:
            return features, context
        
        return features

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_normal_(m.weight)
    #             # nn.init.normal_(m.weight, 0, 0.001)
    #             #nn.init.constant_(m.bias, 0)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)



class OCR_3D_Module(nn.Module):
    def __init__(self,
                 last_inp_channels,
                 ocr_mid_channels,
                 ocr_key_channels,
                 ocr_dropout,
                 num_joints):
        super().__init__()
        
        self.conv3x3_ocr = nn.Sequential(
                    nn.Conv3d(last_inp_channels, ocr_mid_channels,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(ocr_mid_channels),
                    nn.ReLU(inplace=relu_inplace)
        )
        #originally,
        #nn.SyncBatchNorm(ocr_mid_channels) - yk.
        
        self.ocr_gather_head = SpatialGather_Module_3D(num_joints) #oniginally, config.DATASET.NUM_CLASSES
        self.ocr_distri_head = SpatialOCR_Module_3D(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=ocr_dropout,
                                                 )
        
        
        #self._initialize_weights()
        #self.apply(self._initialize_weights)
        #self.conv3x3_ocr.apply(self._initialize_weights)
        #self.ocr_gather_head.apply(self._initialize_weights)
        #self.ocr_distri_head.apply(self._initialize_weights)
    
    def forward(self, volumes, heatmap_volumes, return_context=False):

        #print(volumes.shape) torch.Size([11, 32, 64, 64, 64])
        #print(heatmap_volumes.shape) torch.Size([11, 17, 64, 64, 64]) 

        volumes = self.conv3x3_ocr(volumes)
        #print(volumes.shape) torch.Size([11, 32, 64, 64, 64])
        context = self.ocr_gather_head(volumes, heatmap_volumes)
        volumes = self.ocr_distri_head(volumes, context)
        
        if return_context:
            return volumes, context

        return volumes
