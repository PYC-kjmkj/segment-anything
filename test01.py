import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def main():
    image = cv2.imread(
        r'D:\WorkingRange\NIO\ExteInspect\001-ExteInspect_Segment_engine\ropes\data\image\2.png')  # 路径不能有中文
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()

    sam_checkpoint = \
        r"D:\WorkingRange\NIO\ExteInspect\001-ExteInspect_Segment_engine\ropes\data\SAM-model\sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)

    # 图像标点-用于分割
    input_point = np.array(
        [[1130, 1041], [289, 1056], [2145, 1159], [358, 786], [374, 274], [2100, 317], [785, 312]])
    input_label = np.array([1, 1, 1, 1, 1, 1, 0])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    print(masks.shape)  # (number_of_masks) x H x W
    print(scores)

    # 批量显示mask后的图片
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        plt.imshow(mask)
        if i == 1:
            # Convert the mask to uint8
            mask = mask.astype(np.uint8)
            normalized_img = cv2.normalize(mask, None, alpha=255, norm_type=cv2.NORM_MINMAX)
            # Write the mask to a file
            cv2.imwrite(r'D:\WorkingRange\NIO\ExteInspect\001-ExteInspect_Segment_engine\ropes\data\image\222.png',
                        normalized_img)
        # show_mask(mask, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
