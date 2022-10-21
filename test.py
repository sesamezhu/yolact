import time
import os
import glob
import torch
import cv2
import random
from utils import timer
import torch.backends.cudnn as cudnn
from yolact import Yolact
import numpy as np
import torch.nn.functional as F
from utils.augmentations import FastBaseTransform
from utils.functions import SavePath
from collections import defaultdict
from layers.box_utils import crop, sanitize_coordinates
from data import COLORS
from data import cfg, set_cfg, mask_type, MEANS, STD, activation_func
from PIL import Image
from pylab import imshow
from pylab import array
# from pylab import plot
import matplotlib.pyplot as plt
from pylab import title
from pylab import figure

os.environ['CUDA_VISIBLE_DEVICE'] = '0'


class mask_rcnn(object):
    def __init__(self, gpu_id, trained_model):
        self.gpu_id = gpu_id
        self.trained_model = trained_model
        self.model_path = SavePath.from_str(self.trained_model)
        self.configA = self.model_path.model_name + '_config'
        set_cfg(self.configA)
        # cudnn.fastest = True
        cudnn.benchmark = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('Loading model...', end='')
        self.device = torch.device("cuda:{}".format(self.gpu_id) if torch.cuda.is_available() else "cpu")
        self.net = Yolact()
        self.net.load_weights(self.trained_model)
        self.net.eval()
        self.net = self.net.to(self.device)
        print("模型网络在：{}中".format(next(self.net.parameters()).device))
        print(' Done.')
        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        self.color_cache = defaultdict(lambda: {})
        self.init_seed(0)

    def init_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
        if seed == 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def inference(self, image):
        with torch.no_grad():
            frame = torch.from_numpy(image).to(self.device).float()
            # frame = torch.from_numpy(image).cuda(non_blocking=True).float()
            frame = frame.unsqueeze(0)
            batch = FastBaseTransform()(frame)  # unsqueeze增加维度
            torch.cuda.synchronize()
            T_1 = time.perf_counter()
            torch.cuda.set_device('cuda:{}'.format(self.gpu_id))
            preds = self.net(batch)[0]
            torch.cuda.synchronize()
            T_2 = time.perf_counter()
            # print("所用时间：{}".format(T_2 - T_1))
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            outputs = self.postprocess(preds, image, visualize_lincomb=False,
                                       crop_masks=True,
                                       score_threshold=0.8)
            cfg.rescore_bbox = save
            # if outputs["pred_classes"].shape[0] == 2:
            #     print('识别成功')
            return outputs

    def postprocess(self, det_output, image, interpolation_mode='bilinear', visualize_lincomb=False, crop_masks=True,
                    score_threshold=0.15):
        height, width, _ = image.shape
        net = det_output['net']
        dets = det_output['detection']

        if dets is None:
            return {"pred_boxes": torch.Tensor(), "scores": torch.Tensor(), "pred_classes": torch.Tensor(),
                    "pred_masks": torch.Tensor(), "image_size": image.shape}

        if score_threshold > 0:
            keep = dets['score'] > score_threshold

            for k in dets:
                if k != 'proto':
                    dets[k] = dets[k][keep]

            if dets['score'].size(0) == 0:
                return {"pred_boxes": torch.Tensor(), "scores": torch.Tensor(), "pred_classes": torch.Tensor(),
                        "pred_masks": torch.Tensor(), "image_size": image.shape}

        classes = dets['class']
        boxes = dets['box']
        scores = dets['score']
        masks = dets['mask']
        # print('数据转换时mask在：',masks.device)

        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:  # True
            proto_data = dets['proto']

            if cfg.mask_proto_debug:  # False
                np.save('scripts/proto.npy', proto_data.cpu().numpy())

            if visualize_lincomb:  # False
                self.display_lincomb(proto_data, masks)

            masks = proto_data @ masks.t()
            masks = cfg.mask_proto_mask_activation(masks)

            if crop_masks:  # True
                masks = crop(masks, boxes)

            masks = masks.permute(2, 0, 1).contiguous()

            if cfg.use_maskiou:  # True
                with timer.env('maskiou_net'):
                    with torch.no_grad():
                        maskiou_p = net.maskiou_net(masks.unsqueeze(1))
                        maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)
                        if cfg.rescore_mask:
                            if cfg.rescore_bbox:
                                scores = scores * maskiou_p
                            else:
                                scores = [scores, scores * maskiou_p]

            masks = F.interpolate(masks.unsqueeze(0), (height, width), mode=interpolation_mode,
                                  align_corners=False).squeeze(0)

            masks.gt_(0.5)

        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], width, cast=False)
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], height, cast=False)
        boxes = boxes.long()

        if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
            full_masks = torch.zeros(masks.size(0), height, width)

            for jdx in range(masks.size(0)):
                x1, y1, x2, y2 = boxes[jdx, :]

                mask_w = x2 - x1
                mask_h = y2 - y1

                if mask_w * mask_h <= 0 or mask_w < 0:
                    continue

                mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
                mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
                mask = mask.gt(0.5).float()
                full_masks[jdx, y1:y2, x1:x2] = mask

            masks = full_masks
        return {"pred_boxes": boxes, "scores": scores, "pred_classes": classes, "pred_masks": masks,
                "image_size": image.shape}
        # return classes, scores, boxes, masks, image.shape

    def prep_display(self, dets_out, frame, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        img = torch.from_numpy(frame).cuda().float()
        if undo_transform:
            img_numpy = self.undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = (dets_out['pred_classes'], dets_out['scores'], dets_out['pred_boxes'], dets_out['pred_masks'])
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:15]

            if cfg.eval_mask_branch:
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(15, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < 0.5:
                num_dets_to_consider = j
                break

        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
                return self.color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    self.color_cache[on_gpu][color_idx] = color
                return color

        if cfg.eval_mask_branch and num_dets_to_consider > 0:
            masks = masks[:num_dets_to_consider, :, :, None]
            colors = torch.cat(
                [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],
                dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
            inv_alph_masks = masks * (-mask_alpha) + 1
            masks_color_summand = masks_color[0]

            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if num_dets_to_consider == 0:
            return img_numpy

        # if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            # if args.display_bboxes:
            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            # if args.display_text:
            _class = cfg.dataset.class_names[classes[j]]
            text_str = '%s: %.2f' % (_class, score) if 1 else _class

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

            text_pt = (x1, y2 - 3)
            text_color = [255, 255, 255]

            # cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return img_numpy

    def undo_image_transformation(self, img, w, h):
        img_numpy = img.permute(1, 2, 0).cpu().numpy()
        img_numpy = img_numpy[:, :, (2, 1, 0)]  # to BRG

        if cfg.backbone.transform.normalize:
            img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
        elif cfg.backbone.transform.subtract_means:
            img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)

        img_numpy = img_numpy[:, :, (2, 1, 0)]  # to RGB
        img_numpy = np.clip(img_numpy, 0, 1)

        return cv2.resize(img_numpy, (w, h))

    def display_lincomb(self, proto_data, masks):
        out_masks = torch.matmul(proto_data, masks.t())
        # out_masks = cfg.mask_proto_mask_activation(out_masks)

        for kdx in range(1):
            jdx = kdx + 0
            coeffs = masks[jdx, :].cpu().numpy()
            idx = np.argsort(-np.abs(coeffs))
            # plt.bar(list(range(idx.shape[0])), coeffs[idx])
            # plt.show()

            coeffs_sort = coeffs[idx]
            arr_h, arr_w = (4, 8)
            proto_h, proto_w, _ = proto_data.size()
            arr_img = np.zeros([proto_h * arr_h, proto_w * arr_w])
            arr_run = np.zeros([proto_h * arr_h, proto_w * arr_w])
            test = torch.sum(proto_data, -1).cpu().numpy()

            for y in range(arr_h):
                for x in range(arr_w):
                    i = arr_w * y + x

                    if i == 0:
                        running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                    else:
                        running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                    running_total_nonlin = running_total
                    if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                        running_total_nonlin = (1 / (1 + np.exp(-running_total_nonlin)))

                    arr_img[y * proto_h:(y + 1) * proto_h, x * proto_w:(x + 1) * proto_w] = (proto_data[:, :,
                                                                                             idx[i]] / torch.max(
                        proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                    arr_run[y * proto_h:(y + 1) * proto_h, x * proto_w:(x + 1) * proto_w] = (
                                running_total_nonlin > 0.5).astype(np.float)
            plt.imshow(arr_img)
            plt.show()
            # plt.imshow(arr_run)
            # plt.show()
            # plt.imshow(test)
            # plt.show()
            plt.imshow(out_masks[:, :, jdx].cpu().numpy())
            plt.show()


if __name__ == "__main__":
    img_path = 'TK_attach.png'
    im = cv2.imread(img_path)
    mr = mask_rcnn(0, "yolact_plus_resnet50_205_40000.pth")
    outputs_test = mr.inference(im)
    h, w, _ = im.shape
    vis = mr.prep_display(outputs_test, im, 0, 0, False)
    figure(figsize=(15,15))
    imshow(vis)
    title(img_path)
    plt.show()
