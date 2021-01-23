import torch
from torch.autograd import Function
from torch.nn import functional as F
import torch.nn as nn


torch.set_default_tensor_type('torch.cuda.FloatTensor')

class RoIOffsetPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, roi, offset, feature_size):

        # try:
        proposal_bbox = roi.clone()
        _, n, c, h, w = feature_size.tolist()
        ctx.feature_size = feature_size

        # print(proposal_bbox)
        start_x = max(min(round(proposal_bbox[1].item() * ctx.spatial_scale), w-1), 0)
        start_y = max(min(round(proposal_bbox[2].item() * ctx.spatial_scale), h-1), 0)
        end_x = max(min(round(proposal_bbox[3].item() * ctx.spatial_scale), w-1), 1)
        end_y = max(min(round(proposal_bbox[4].item() * ctx.spatial_scale), h-1), 1)

        try:
            roi_feature_map = features[..., start_y:end_y, start_x:end_x]
        except:
            print(start_x, start_y, end_x, end_y)

        roi_width = max(end_x - start_x +1, 1)
        roi_height = max(end_y - start_y + 1, 1)
        bin_width = roi_width / ctx.pooled_width
        bin_height = roi_height / ctx.pooled_height

        # Offsets
        # Create meshgrid of pooled location coordinates (H X W) and repeat along the number of time steps as well
        coord_array_x = torch.arange(start=start_x, end=end_x+1, step=1)
        coord_array_y = torch.arange(start=start_y, end=end_y+1, step=1)
        x_t = coord_array_x.repeat(roi_height).view(roi_height, roi_width).cuda().repeat(n, 1, 1).unsqueeze(1)
        y_t = coord_array_y.repeat(roi_width, 1).t().contiguous().view(roi_height, roi_width).cuda().repeat(n, 1, 1).unsqueeze(1)

        # Clamp offsets within feature space
        # d_oy = torch.clamp(offset[0, proposal_ind,:,:,:,0] * h, 0, h-1) # (7, 7)
        # d_ox = torch.clamp(offset[0, proposal_ind,:,:,:,1] * w, 0, w-1) # (7, 7)
        d_oy = torch.clamp(offset[:, :, : ,0] * h, 0, h-1) # (7, 7)
        d_ox = torch.clamp(offset[:, :, :, 1] * w, 0, w-1) # (7, 7)

        # Spread d or contract d depending on whether ROI is greater than or smaller than pooling size
        # If ROI is small than contract d
        # If ROI is big then expand d
        try:
            if roi_width == 1:
                d_x = torch.Tensor(0).long()
            else:
                d_x = torch.linspace(start=0, end=ctx.pooled_width-1, steps=roi_width).long()
            if roi_height == 1:
                d_y = torch.Tensor(0).long()
            else:
                d_y = torch.linspace(start=0, end=ctx.pooled_height-1, steps=roi_height).long()
        except RuntimeError:
            print(roi_width)
            print(roi_height)
            exit()
        # Add offsets to the meshgrid
        if roi_height == 1:
            d_ox = d_ox[:,d_y, :].unsqueeze(1)
            d_oy = d_oy[:,d_y, :].unsqueeze(1)
            if roi_width == 1:
                d_ox = d_ox[:,:, d_x].unsqueeze(2)
                d_oy = d_oy[:,:, d_x].unsqueeze(2)
            else:
                d_ox = d_ox[:, :, d_x]
                d_oy = d_oy[:, :, d_x]
        elif roi_width == 1:
            d_ox = d_ox[:, d_y, :]
            d_oy = d_oy[:, d_y, :]
            d_ox = d_ox[:, :, d_x].unsqueeze(2)
            d_oy = d_oy[:, :, d_x].unsqueeze(2)
        else:
            d_ox = d_ox[:, d_y,:][:, :, d_x] #(Y,X)
            d_oy = d_oy[:, d_y,:][:, :, d_x] #(Y,X)
#
        d_ox = d_ox.unsqueeze(1)
        d_oy = d_oy.unsqueeze(1)
#
        d_ox_ind = d_x.repeat(roi_height).view(roi_height, roi_width).cuda().repeat(n, 1, 1).unsqueeze(1)
        d_oy_ind = d_y.repeat(roi_width, 1).t().contiguous().view(roi_height, roi_width).cuda().repeat(n, 1, 1).unsqueeze(1)
        d_ind = d_ox_ind + ctx.pooled_height * d_oy_ind
#
#         # Expand or contract offsets for each bin to the size of the ROI in feature map pixels
#         # d_ox, d_ox_ind = F.adaptive_avg_pool2d(d_ox.unsqueeze(1), output_size=(roi_height, roi_width), return_indices=True)
#         # d_oy, d_oy_ind = F.adaptive_avg_pool2d(d_oy.unsqueeze(1), output_size=(roi_height, roi_width), return_indices=True)
#
        # Add the offsets to the base index of each feature map pixel
        d_ox = (d_ox + x_t).repeat(1, c, 1, 1)
        d_oy = (d_oy + y_t).repeat(1, c, 1, 1)

        # Convert these offsets into linear indices
        fx = torch.floor(d_ox)
        fy = torch.floor(d_oy)
        cx = torch.ceil(d_ox)
        cy = torch.ceil(d_oy)

        index00 = (fy * h + fx).long()
        index01 = (fy * h + cx).long()
        index10 = (cy * h + fx).long()
        index11 = (cy * h + cx).long()

        # Base offsets to add for channel and time dimensions
        channel_offsets = torch.arange(start=0, end=c, step=1).unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(n,1,roi_height,roi_width).long().cuda()
        time_offsets = torch.arange(start=0, end=n, step=1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,c,roi_height,roi_width).long().cuda()
        index00_linear = time_offsets * c * h * w + \
                         channel_offsets * h * w + \
                         index00
        index01_linear = time_offsets * c * h * w + \
                         channel_offsets * h * w + \
                         index01
        index10_linear = time_offsets * c * h * w + \
                         channel_offsets * h * w + \
                         index10
        index11_linear = time_offsets * c * h * w + \
                         channel_offsets * h * w + \
                         index11
# #
# #         # Pool along the time dimensions as well
# #         time_indices = torch.arange(start=0, end=n, step=1).long().repeat(roi_height*roi_width,1)
# #         time_indices = time_indices.permute((1,0))
# #         time_indices = time_indices.contiguous().view(-1)
# #
# #         out =  features[:, time_indices, :, fy, fx]
# #         out += features[:, time_indices, :, fy, cx]
# #         out += features[:, time_indices, :, cy, fx]
# #         out += features[:, time_indices, :, cy, cx]
# # #
# #         out = out / 4
# #         out = out.squeeze() # [N X H X W, Channels]
# #         out = out.view(n, roi_height * roi_width, c) # [N, HxW, Channels]
# #         out = out.permute((0,2,1)).view(n, c, roi_height, roi_width) # Permute is needed to preserve spatial order [N, Channels, HxW]
# #
# #         out = out.contiguous().view(n,c,roi_height, roi_width)
#
        # Flatten features and index using generated indices
        out =  features[0].view(-1)[index00_linear] * (1-(d_oy - fy)) * (1-(d_ox - fx))
        out += features[0].view(-1)[index01_linear] * (1-(d_oy - fy)) * (1-(cx - d_ox))
        out += features[0].view(-1)[index10_linear] * (1-(cy - d_oy)) * (1-(d_ox - fx))
        out += features[0].view(-1)[index11_linear] * (1-(cy - d_oy)) * (1-(cx - d_ox))
        out /= torch.pow(2, (fx==cx)*1 + (fy==cy)*1).float().cuda()
        # Reshape back to original dimension
        out = out.view(n,c,roi_height, roi_width)

        before_pool_size = torch.tensor(out.size())
        max_pooled_val, max_pooled_ind = F.adaptive_max_pool2d(out, ctx.pooled_height, return_indices=True) # (1, 512*3, 7, 7)

        # sanity_check_for_maxpool = adaptive_maxunpool2d(max_pooled_val, max_pooled_ind, before_pool_size)
        # sanity_check_for_maxpool = F.adaptive_max_pool2d(sanity_check_for_maxpool, ctx.pooled_height)
        # sanity_check_for_maxpool = torch.mean(sanity_check_for_maxpool - max_pooled_val)
        # try:
        #     assert(sanity_check_for_maxpool == 0.0)
        # except AssertionError:
        #     print(sanity_check_for_maxpool)

        ctx.save_for_backward(max_pooled_ind, before_pool_size, index00, index01, index10, index11, d_ox, d_oy, features, d_ind)

        time_pooled = max_pooled_val
        # except:
        #     pass


        return time_pooled


    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)

        batch_size, num_time_steps, num_channels, data_height, data_width = ctx.feature_size.tolist()
        grad_input = grad_output.new(num_time_steps, num_channels, data_height, data_width).zero_()
        grad_offset = grad_output.new(num_time_steps, ctx.pooled_height, ctx.pooled_width, 2).zero_()

        max_pooled_ind, before_pool_size, index00, index01, index10, index11, d_ox, d_oy, features, d_ind = ctx.saved_tensors

        fx = torch.floor(d_ox)
        fy = torch.floor(d_oy)
        cx = torch.ceil(d_ox)
        cy = torch.ceil(d_oy)

        # Get gradient for adaptive max pooling step
        grad_max_pool = adaptive_maxunpool2d(grad_output, max_pooled_ind, before_pool_size, accumulate=True)

        # Get gradient for features
        grad_feature00 = adaptive_maxunpool2d(grad_max_pool, index00, ctx.feature_size[1:], accumulate=True)
        grad_feature01 = adaptive_maxunpool2d(grad_max_pool, index01, ctx.feature_size[1:], accumulate=True)
        grad_feature10 = adaptive_maxunpool2d(grad_max_pool, index10, ctx.feature_size[1:], accumulate=True)
        grad_feature11 = adaptive_maxunpool2d(grad_max_pool, index11, ctx.feature_size[1:], accumulate=True)

        scale_factor =  torch.pow(2, (fx == cx) * 1 + (fy == cy) * 1).float().cuda()

        # Scale them by the weights of bilinear indexing
        offset_weights00 = (1-(d_oy - fy)) * (1-(d_ox - fx)) / scale_factor
        offset_weights01 = (1-(d_oy - fy)) * (1-(cx - d_ox)) / scale_factor
        offset_weights10 = (1-(cy - d_oy)) * (1-(d_ox - fx)) / scale_factor
        offset_weights11 = (1-(cy - d_oy)) * (1-(cx - d_ox)) / scale_factor

        offset_weights00_unpooled = adaptive_maxunpool2d(offset_weights00, index00, ctx.feature_size[1:], accumulate=True)
        offset_weights01_unpooled = adaptive_maxunpool2d(offset_weights01, index01, ctx.feature_size[1:], accumulate=True)
        offset_weights10_unpooled = adaptive_maxunpool2d(offset_weights10, index10, ctx.feature_size[1:], accumulate=True)
        offset_weights11_unpooled = adaptive_maxunpool2d(offset_weights11, index11, ctx.feature_size[1:], accumulate=True)

        grad_input[0] += grad_feature00 * offset_weights00_unpooled
        grad_input[0] += grad_feature01 * offset_weights01_unpooled
        grad_input[0] += grad_feature10 * offset_weights10_unpooled
        grad_input[0] += grad_feature11 * offset_weights11_unpooled
        # Get gradient for offset fully connected layer

        channel_offsets = torch.arange(start=0, end=num_channels, step=1).unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(num_time_steps,1,before_pool_size[-2],before_pool_size[-1]).long().cuda()
        time_offsets = torch.arange(start=0, end=num_time_steps, step=1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,num_channels,before_pool_size[-2],before_pool_size[-1]).long().cuda()
        index00_linear = time_offsets * num_channels * data_height * data_width + \
                         channel_offsets * data_height * data_width + \
                         index00
        index01_linear = time_offsets * num_channels * data_height * data_width + \
                         channel_offsets * data_height * data_width + \
                         index01
        index10_linear = time_offsets * num_channels * data_height * data_width + \
                         channel_offsets * data_height * data_width + \
                         index10
        index11_linear = time_offsets * num_channels * data_height * data_width + \
                         channel_offsets * data_height * data_width + \
                         index11

        # Gather gradients from the feature gradients that were touched by these offsets (This 2D feature map is equal to the size of the ROI pixels)
        feature_offset_x = grad_feature00.view(-1)[index00_linear]  * (1-(d_oy - fy)) * torch_sign_fn(1-(d_ox - fx)) * features[0].view(-1)[index00_linear]
        feature_offset_x += grad_feature01.view(-1)[index01_linear] * (1-(d_oy - fy)) * torch_sign_fn(1-(cx - d_ox)) * features[0].view(-1)[index01_linear]
        feature_offset_x += grad_feature10.view(-1)[index10_linear] * (1-(cy - d_oy)) * torch_sign_fn(1-(d_ox - fx)) * features[0].view(-1)[index10_linear]
        feature_offset_x += grad_feature11.view(-1)[index11_linear] * (1-(cy - d_oy)) * torch_sign_fn(1-(cx - d_ox)) * features[0].view(-1)[index11_linear]
        feature_offset_x /= scale_factor

        feature_offset_y = grad_feature00.view(-1)[index00_linear]   * (1-(d_ox - fx)) * torch_sign_fn(1-(d_oy - fy)) * features[0].view(-1)[index00_linear]
        feature_offset_y += grad_feature01.view(-1)[index01_linear]  * (1-(cx - d_ox)) * torch_sign_fn(1-(d_oy - fy)) * features[0].view(-1)[index01_linear]
        feature_offset_y += grad_feature10.view(-1)[index10_linear]  * (1-(d_ox - fx)) * torch_sign_fn(1-(cy - d_oy)) * features[0].view(-1)[index10_linear]
        feature_offset_y += grad_feature11.view(-1)[index11_linear]  * (1-(cx - d_ox)) * torch_sign_fn(1-(cy - d_oy)) * features[0].view(-1)[index11_linear]
        feature_offset_y /= scale_factor

        feature_offset_y = torch.sum(feature_offset_y, dim=1).unsqueeze(1)
        feature_offset_x = torch.sum(feature_offset_x, dim=1).unsqueeze(1)

        grad_offset_size = torch.Tensor([num_time_steps, 1, ctx.pooled_height, ctx.pooled_width]).long().cuda()

        # Expand or contract back to the offset indices
        feature_offset_x = adaptive_maxunpool2d(feature_offset_x, d_ind, grad_offset_size, accumulate=True)
        feature_offset_y = adaptive_maxunpool2d(feature_offset_y, d_ind, grad_offset_size, accumulate=True)

        grad_offset[:,:,:,0] = feature_offset_x.squeeze(1)
        grad_offset[:,:,:,1] = feature_offset_y.squeeze(1)

        return grad_input, None, grad_offset, None

def torch_sign_fn(x):
    x[x > 0] = 1
    x[x == 0] = 0
    x[x < 0] = -1

    return x

def adaptive_maxunpool2d(input, indices, output_size, accumulate=False):
    '''

    :param input:           Input to unpool (N,C,Hin,Win)
    :param indices:         Indices for max element (N,C,Hin,Win)
    :param output_size:     Output size to match (Torch.Tensor) 4 [N, C, Hout, Wout]
    :param accumulate:      This switch determines whether repeated indices should be added to the final output (True
                            for gradient computation)
    :return:
            unpool2d:       Unpooled output of dim [N, C, Hout, Wout]
    '''
    if len(input.size()) < 4:
        raise AttributeError

    unpool2d = input.new(torch.Size(output_size.tolist())).zero_().view(-1)     # Create output tensor of output size

    f_before_pooling = output_size[-1] * output_size[-2]                        # Size of 2D feature maps before pooling (H x W)
    last_ind = output_size[0] * output_size[1] * f_before_pooling               # Number of elements in output
    pooled_length = indices.size(-1) * indices.size(-2)                         # Number of elements pooled in each fmap

    # Generate channel offsets for each channel
    base_ind = torch.arange(start=0, end=last_ind, step=f_before_pooling).repeat(pooled_length, 1).\
        permute(1, 0).contiguous().view(-1).long().cuda()

    indices = indices.view(-1)                                                  # Flatten indices
    final_ind = base_ind + indices                                              # Add channel offset to indices

    # unpool2d[final_ind] += input.view(-1)                # Flatten then populate values from output
    unpool2d.put_(final_ind, input.view(-1), accumulate=accumulate)
    return unpool2d.view(torch.Size(output_size.tolist()))

# def adaptive_avgunpool2d(input, output_size, accumulate=False):
#     '''
#
#     :param input:           Input to unpool (N,C,Hin,Win)
#     :param indices:         Indices for max element (N,C,Hin,Win)
#     :param output_size:     Output size to match (Torch.Tensor) 4 [N, C, Hout, Wout]
#     :param accumulate:      This switch determines whether repeated indices should be added to the final output (True
#                             for gradient computation)
#     :return:
#             unpool2d:       Unpooled output of dim [N, C, Hout, Wout]
#     '''
#     if len(input.size()) < 4:
#         raise AttributeError
#
#     unpool2d = input.new(torch.Size(output_size.tolist())).zero_().view(-1)     # Create output tensor of output size
#
#     f_before_pooling = output_size[-1] * output_size[-2]                        # Size of 2D feature maps before pooling (H x W)
#     last_ind = output_size[0] * output_size[1] * f_before_pooling               # Number of elements in output
#     pooled_length = input.size(-1) * input.size(-2)                         # Number of elements pooled in each fmap
#
#     # Generate channel offsets for each channel
#     base_ind = torch.arange(start=0, end=last_ind, step=f_before_pooling).repeat(pooled_length, 1).\
#         permute(1, 0).contiguous().view(-1).long().cuda()
#
#     final_ind = base_ind                                            # Add channel offset to indices
#
#     # unpool2d[final_ind] += input.view(-1)                # Flatten then populate values from output
#     unpool2d.put_(final_ind, input.view(-1), accumulate=accumulate)
#     return unpool2d.view(torch.Size(output_size.tolist()))

class maxPool2DFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, roi, offset, feature_size):

        # try:
        proposal_bbox = roi.clone()
        _, n, c, h, w = feature_size.tolist()
        ctx.feature_size = feature_size

        # print(proposal_bbox)
        start_x = max(min(round(proposal_bbox[1].item() * ctx.spatial_scale), w-1), 0)
        start_y = max(min(round(proposal_bbox[2].item() * ctx.spatial_scale), h-1), 0)
        end_x = max(min(round(proposal_bbox[3].item() * ctx.spatial_scale), w-1), 1)
        end_y = max(min(round(proposal_bbox[4].item() * ctx.spatial_scale), h-1), 1)

        try:
            roi_feature_map = features[..., start_y:end_y, start_x:end_x]
        except:
            print(start_x, start_y, end_x, end_y)
        max_pooled_val, max_pooled_ind = F.adaptive_max_pool2d(roi_feature_map, ctx.pooled_height,return_indices=True)  # (1, 512*3, 7, 7)
        before_pool_size = torch.tensor(roi_feature_map.size())
        ctx.save_for_backward(max_pooled_ind, before_pool_size, proposal_bbox)
        return max_pooled_val
        return time_pooled


    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)

        batch_size, num_time_steps, num_channels, data_height, data_width = ctx.feature_size.tolist()
        grad_input = grad_output.new(num_time_steps, num_channels, data_height, data_width).zero_()
        grad_offset = grad_output.new(num_time_steps, ctx.pooled_height, ctx.pooled_width, 2).zero_()

        max_pooled_ind, before_pool_size, proposal_bbox = ctx.saved_tensors

        start_x = max(min(round(proposal_bbox[1].item() * ctx.spatial_scale), data_width-1), 0)
        start_y = max(min(round(proposal_bbox[2].item() * ctx.spatial_scale), data_height-1), 0)
        end_x = max(min(round(proposal_bbox[3].item() * ctx.spatial_scale), data_width-1), 1)
        end_y = max(min(round(proposal_bbox[4].item() * ctx.spatial_scale), data_height-1), 1)

        grad_max_pool = adaptive_maxunpool2d(grad_output, max_pooled_ind, before_pool_size, accumulate=True)

        grad_input[..., start_y:end_y, start_x:end_x] = grad_max_pool

        return grad_input, None, grad_offset, None


def test_grad_maxPool():
    from torch.autograd import gradcheck
    features = torch.randn((3, 32, 19, 19), dtype=torch.double, requires_grad=True)
    roi = torch.randn((5), dtype=torch.double, requires_grad=False)
    offset = torch.randn((3, 3, 3, 2), dtype=torch.double, requires_grad=True)
    feature_size = torch.tensor([1, 3, 32, 19, 19]).long()
    input = (features, roi, offset, feature_size)

    fn = maxPool2DFunction(3,3,1.0/16.0)

    test = gradcheck(fn, input, eps=1e-6, atol=1e-4)

    print(test)

def test_grad_roioffset():
    from torch.autograd import gradcheck
    features = torch.randn((3, 32, 19, 19), dtype=torch.double, requires_grad=True)
    roi = torch.randn((5), dtype=torch.double, requires_grad=False)
    offset = torch.randn((3, 3, 3, 2), dtype=torch.double, requires_grad=True)
    feature_size = torch.tensor([1, 3, 32, 19, 19]).long()
    input = (features, roi, offset, feature_size)

    fn = RoIOffsetPoolFunction(3,3,1.0/16.0)

    test = gradcheck(fn, input, eps=1e-6, atol=1e-4, raise_exception=True)

    print(test)


if __name__ == '__main__':
    test_grad_roioffset()
