# @Author: Jinyu Zhang
# @Time: 2022/4/30 15:07
# @E-mail: JinyuZ1996@outlook.com
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# default_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 14}
# tag_y = ["$α$=1", "$α$=2", "$α$=3", "$α$=4", "$α$=5"]
# tag_x = ["$β$=1", "$β$=2", "$β$=3", "$β$=4", "$β$=5"]

# HR_5 = np.array([[75.22, 76.34, 75.31, 78.03, 76.57],
#                  [80.52, 82.93, 81.33, 83.97, 83.41],
#                  [78.70, 80.41, 79.12, 82.91, 81.44],
#                  [80.04, 82.66, 81.03, 83.87, 83.28],
#                  [78.12, 79.26, 79.21, 80.14, 80.52]])

# fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
# plt.subplots_adjust(top=0.94, bottom=0.22, left=0.15, right=0.99, hspace=0,
#                     wspace=0)

# ax.set_xticks(np.arange(0, 5, 1))
# ax.set_yticks(np.arange(len(tag_y)))
# ax.set_xticklabels(tag_x)
# ax.set_yticklabels(tag_y)
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# for i in range(len(tag_y)):
#     for j in range(len(tag_x)):
#         text = ax.text(j, i, HR_5[i, j],
#                        ha="center", va="center", color="black", fontweight="bold")

# plt.imshow(HR_5, cmap='coolwarm', origin='upper', aspect="auto")
# plt.colorbar()
# plt.xlabel('Values of the parameter $β$\n(a) HR@5 (%)\n', default_font)
# plt.ylabel('Values of the parameter $α$', default_font)
# ax.set_title("Fig. 1: Happer-parameter test of $α$ and $β$ on A dataset.", default_font)

# plt.show()

# plt.savefig('attn_1.jpg',bbox_inches='tight')  #保存成jpg格式


def get_heatmap(attn):
    default_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 14}
    tag_y = [f"$w_{i}$" for i in range(len(attn))]
    tag_x = [f"$w_{i}$" for i in range(len(attn[0]))]


    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    plt.subplots_adjust(top=0.94, bottom=0.22, left=0.15, right=0.99, hspace=0,
                    wspace=0)

    # ax.set_xticks(np.arange(len(tag_x)))
    # ax.set_yticks(np.arange(len(tag_y)))
    # ax.set_xticklabels(tag_x)
    # ax.set_yticklabels(tag_y)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # for i in range(len(tag_y)):
    #     for j in range(len(tag_x)):
    #         text = ax.text(j, i, attn[i, j],
    #                        ha="center", va="center", color="black", fontweight="bold")

    plt.imshow(attn, cmap='coolwarm', origin='upper', aspect="auto")
    plt.colorbar()
    plt.xlabel('Values of the parameter $β$\n(a) HR@5 (%)\n', default_font)
    plt.ylabel('Values of the parameter $α$', default_font)
    ax.set_title("Fig. 1: Happer-parameter test of $α$ and $β$ on A dataset.", default_font)

    plt.savefig('attn_1.jpg',bbox_inches='tight')  #保存成jpg格式

if __name__ == "__main__":
    import torch
    attn = torch.rand(256,256)
    get_heatmap(attn)