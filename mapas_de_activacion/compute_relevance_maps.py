import argparse
import logging
import pickle as pkl
import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

def find_obj(obj_folder, sample, side):
    for root, folders, files in os.walk(obj_folder):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".obj":
                file_sample, file_side, _ = file.split("_")
                if sample == file_sample and side == file_side:
                    return os.path.join(os.path.abspath(root), file)

def heatmap_on_image(heatmap, image):
    plt.figure(figsize=(108,36),dpi=1)
    hmax = sns.heatmap(heatmap, vmin=-1, vmax=1,
                    cmap="bwr",
                    xticklabels=False, yticklabels=False, cbar=False, 
                    alpha=1, zorder=1)
    # plt.savefig("os.path.join(relevance_maps_path, "heatmap.png"))
    hmax.imshow(image,
                cmap="gray",
                alpha=0.5,
                aspect = hmax.get_aspect(),
                extent = hmax.get_xlim() + hmax.get_ylim(),
                zorder = 2)
    plt.tight_layout()
    # plt.savefig("os.path.join(relevance_maps_path, "heatmap_background.png"))

    ax = plt.gca()
    canvas = ax.figure.canvas
    canvas.draw()
    hoi = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    hoi = np.reshape(hoi, (36,108,3))
    hoi = cv.cvtColor(hoi,cv.COLOR_BGR2RGBA)
    plt.close()
    return hoi

relevance_maps = {}

def main(maps_path, data_path, obj_paths):
    i = 0
    for root, folders, files in os.walk(maps_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".pkl":
                file_path = os.path.join(root, file)

                sample, side, axis, _ = file.split("_")

                print(sample, side, axis, file_path)

                with open(file_path, 'rb') as f:
                    data = pkl.load(f)

                img = np.array(data).T
                img = np.where(img >= 0, img, 0)

                img = cv.resize(img, (108, 36))
                img_max = np.max(img)
                print(np.min(img), np.max(img))

                mask = cv.imread(os.path.join(data_path, "{}_{}".format(sample,side), "{}_{}_0_panorama_ext_{}_gray_mask_input.png".format(sample,side,axis)), cv.IMREAD_GRAYSCALE)

                mask = mask / 255.0
                img = img * mask
                img = img.astype('float32')

                cv.imwrite(os.path.join(root,"relevance_{}_{}_{}.png".format(sample,side,axis)), cv.resize(img, (0, 0), fx = 5, fy = 5)*255)

                img_izq = img[:,:72]
                relevance_maps_izq = os.path.join(os.path.abspath(root),"relevance_map_{}_{}_{}_izq.png".format(sample,side,axis))
                cv.imwrite(relevance_maps_izq, img_izq*255)

                img_dch = img[:,36:]
                relevance_maps_dch = os.path.join(os.path.abspath(root),"relevance_map_{}_{}_{}_dch.png".format(sample,side,axis))
                cv.imwrite(relevance_maps_dch, img_dch*255)

                panorama = cv.imread(os.path.join(data_path, "{}_{}".format(sample,side), "{}_{}_0_panorama_ext_{}_gray_input.png".format(sample,side,axis)), cv.IMREAD_GRAYSCALE)
                heatmap = heatmap_on_image(img, panorama)
                heatmap = cv.resize(heatmap, (540, 180))

                cv.imwrite(os.path.join(root,"heatmap_{}_{}_{}.png".format(sample,side,axis)), heatmap)
                cv.imwrite(os.path.join(root,"heatmap_{}_{}_{}_izq.png.png".format(sample,side,axis)), heatmap[:,:360])
                cv.imwrite(os.path.join(root,"heatmap_{}_{}_{}_dch.png.png".format(sample,side,axis)), heatmap[:,180:])


                if "resnet" in file_path:
                    net = "Resnet"
                else:
                    net = "Panorama"

                obj = find_obj(obj_paths, sample, side)

                relevance_maps[i] = {
                    "net": net,
                    "sample" : sample,
                    "side" : side,
                    "axis" : axis,
                    "obj" : obj,
                    "izq" : relevance_maps_izq,
                    "dch" : relevance_maps_dch,
                    "max" : img_max

                }
                i += 1

    relevance_maps_df = pd.DataFrame.from_dict(relevance_maps).T
    relevance_maps_df.to_csv("relevance_maps.csv", sep=";", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--maps",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-obj",
        "--obj_paths",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-v", 
        "--verbose", 
        type=int, 
        required=False, 
        default=0
    )

    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose == 2:
        log_level = logging.DEBUG
    else:
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    main(args.maps, args.data, args.obj_paths)
