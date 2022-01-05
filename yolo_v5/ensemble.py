import glob
import os
import argparse
import pandas as pd
import numpy as np


def GeneralEnsemble(dets, iou_thresh, weights):
    assert(type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert(len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        # Same class
                        iou = computeIOU(box, obox)
                        if iou > bestiou:
                            bestiou = iou
                            bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[0] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w * b[1]
                    yc += w * b[2]
                    bw += w * b[3]
                    bh += w * b[4]
                    conf += w * b[0]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [conf, xc, yc, bw * 0.975, bh * 0.975]
                out.append(new_box)
    return out


def getCoords(box):
    x1 = float(box[1]) - float(box[3]) / 2
    x2 = float(box[1]) + float(box[3]) / 2
    y1 = float(box[2]) - float(box[4]) / 2
    y2 = float(box[2]) + float(box[4]) / 2
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./ensemble", type=str)
    parser.add_argument("--conf_thr", default=0.1, type=float)
    parser.add_argument("--ens_thr", default=0.7, type=float)
    parser.add_argument("--iou_thr", default=0.01, type=float)
    parser.add_argument("--weights", default=None, nargs='+', type=float)
    args = parser.parse_args()
    print(
        f'conf_thr: {args.conf_thr}, ens_thr: {args.ens_thr}, iou_thr: {args.iou_thr}')
    file_list = glob.glob(os.path.join(args.folder, "*.csv"))

    # weights
    if(args.weights == None):
        weights = []
        wei = 1 / len(file_list)
        for i in range(len(file_list)):
            weights.append(wei)
    else:
        weights = args.weights
    # data preprocessing
    total_data = []
    for ind, path in enumerate(file_list):
        print(f'ensemble: {path}, weight: {weights[ind]}')
        data = pd.read_csv(path)
        for index, dec in enumerate(data['PredictionString']):
            if type(dec) == float:
                dec = []
            else:
                dec = dec.split(" ")
                if(len(dec) % 5 == 1):
                    dec = dec[:-1]
                dec = [float(item) for item in dec]
                dec_num = len(dec) // 5
                dec = np.array(dec, dtype=float).reshape(dec_num, 5)
                data['PredictionString'].iloc[index] = dec
        total_data.append(list(data['PredictionString']))

    # ensemble
    result = {
        'patientId': [key for key in data['patientId']],
        'PredictionString': [''] * len(data['patientId'])
    }
    result = pd.DataFrame(result)

    positive = 0
    for index in range(len(total_data[0])):
        dets = []
        no_det = 0
        for num in range(len(file_list)):
            if(type(total_data[num][index]) == float):
                dets.append([])
                no_det += 1
            else:
                det = total_data[num][index].tolist()
                dets.append(det)
        # too much nan
        if(no_det >= int(len(file_list) * args.ens_thr)):
            continue
        else:
            ens = GeneralEnsemble(dets, args.iou_thr, weights=weights)
            eff_ens = []
            for ins in ens:
                if(ins[0] >= args.conf_thr):
                    eff_ens.append(ins)
            if(len(eff_ens) != 0):
                positive += 1

            eff_ens = [str(item) for sublist in eff_ens for item in sublist]
            eff_ens = " ".join(eff_ens)
            result['PredictionString'].iloc[index] = eff_ens

    print(f"test positive: {positive}")
    result.to_csv('./ensemble.csv', index=False)
    print("finish!!!")
