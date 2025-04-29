import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
from matplot2tikz import save as tikz_save
import seaborn as sns

if __name__ == "__main__":


    # Raw results as a multi-line string
    raw_data = """Layer Merged | Mask IoU | Boundary IoU | Throughput (im/s) | FLOPs/Image
    0 | 0.9162 | 0.8644 | 4.3749 | 1484.6100
    1 | 0.9129 | 0.8613 | 4.3390 | 1484.6100
    2 | 0.9156 | 0.8640 | 4.3495 | 1484.6100
    3 | 0.9159 | 0.8644 | 4.3663 | 1484.6100
    4 | 0.9162 | 0.8646 | 4.3644 | 1484.6100
    5 | 0.9158 | 0.8635 | 4.4671 | 1461.6347
    6 | 0.9154 | 0.8630 | 4.3641 | 1484.6100
    7 | 0.9156 | 0.8637 | 4.3635 | 1484.6100
    8 | 0.9159 | 0.8642 | 4.3638 | 1484.6100
    9 | 0.9156 | 0.8635 | 4.3631 | 1484.6100
    10 | 0.9162 | 0.8643 | 4.3643 | 1484.6100
    11 | 0.9140 | 0.8608 | 4.4663 | 1461.6347
    12 | 0.9159 | 0.8639 | 4.3601 | 1484.6100
    13 | 0.9161 | 0.8642 | 4.3598 | 1484.6100
    14 | 0.9161 | 0.8642 | 4.3617 | 1484.6100
    15 | 0.9160 | 0.8639 | 4.3633 | 1484.6100
    16 | 0.9163 | 0.8645 | 4.3622 | 1484.6100
    17 | 0.9158 | 0.8637 | 4.4631 | 1461.6347
    18 | 0.9161 | 0.8642 | 4.3623 | 1484.6100
    19 | 0.9163 | 0.8643 | 4.3617 | 1484.6100
    20 | 0.9162 | 0.8643 | 4.3639 | 1484.6100
    21 | 0.9157 | 0.8637 | 4.3594 | 1484.6100
    22 | 0.9162 | 0.8644 | 4.3605 | 1484.6100
    23 | 0.9160 | 0.8644 | 4.3893 | 1461.6347
    """

    # Load and clean data
    df = pd.read_csv(StringIO(raw_data), sep='|')
    df.columns = [c.strip().replace(' (im/s)', '').replace('/Image', '') for c in df.columns]
    df['Layer'] = df['Layer Merged'].astype(str).str.strip()
    df = df[df['Layer'] != 'None']
    df['Layer'] = df['Layer'].astype(int)
    df['Mask IoU'] = df['Mask IoU'].astype(float)
    df['FLOPs'] = df['FLOPs'].astype(float)

    # Baseline metrics
    baseline_flops = 1493.8470
    baseline_miou = 0.9163

    # Compute score
    epsilon = 1e-6
    df['delta_flops'] = baseline_flops - df['FLOPs']
    df['delta_miou'] = baseline_miou - df['Mask IoU']
    df['score'] = df['delta_flops'] / (df['delta_miou'] + epsilon)

    # Log-transform the score to compress dynamic range
    df['log_score'] = np.log10(df['score'] + 1)

    print(df['log_score'])

    # Highlighted layers
    best_layers = [0, 4, 5, 10, 16, 17, 19, 20, 22, 23]


    raw_data_sam2 = """Layer|JF|FLOPs
    0 | 0.897 | 808.51
    1 | 0.898 | 808.51
    2 | 0.896 | 806.43
    3 | 0.899 | 808.44
    4 | 0.900 | 808.44
    5 | 0.899 | 808.44
    6 | 0.900 | 808.44
    7 | 0.890 | 808.44
    8 | 0.830 | 806.40
    9 | 0.893 | 808.45
    10 | 0.894 | 808.45
    11 | 0.900 | 808.45
    12 | 0.900 | 808.45
    13 | 0.901 | 808.45
    14 | 0.901 | 808.45
    15 | 0.900 | 808.45
    16 | 0.900 | 808.45
    17 | 0.900 | 808.45
    18 | 0.885 | 808.45
    19 | 0.900 | 808.45
    20 | 0.900 | 808.45
    21 | 0.899 | 808.45
    22 | 0.891 | 808.45
    23 | 0.886 | 808.73
    24 | 0.890 | 808.45
    25 | 0.900 | 808.45
    26 | 0.901 | 808.45
    27 | 0.888 | 808.45
    28 | 0.900 | 808.45
    29 | 0.890 | 808.45
    30 | 0.901 | 808.45
    31 | 0.902 | 808.45
    32 | 0.902 | 808.45
    33 | 0.901 | 808.73
    34 | 0.902 | 808.45
    35 | 0.904 | 808.45
    36 | 0.901 | 808.45
    37 | 0.903 | 808.45
    38 | 0.903 | 808.45
    39 | 0.901 | 808.45
    40 | 0.900 | 808.45
    41 | 0.894 | 808.45
    42 | 0.900 | 808.45
    43 | 0.901 | 808.73
    44 | 0.901 | 806.40
    45 | 0.901 | 808.43
    46 | 0.901 | 808.43
    47 | 0.901 | 808.43
    """

    # Load and clean data
    df2 = pd.read_csv(StringIO(raw_data_sam2), sep='|')
    # df.columns = [c.strip().replace(' (im/s)', '').replace('/Image', '') for c in df.columns]
    # df2['Layer'] = df2['Layer Merged'].astype(str).str.strip()
    # df2 = df2[df2['Layer'] != 'None']
    df2['Layer'] = df2['Layer'].astype(int)
    df2['JF'] = df2['JF'].astype(float)
    df2['FLOPs'] = df2['FLOPs'].astype(float)

    # Baseline metrics
    baseline_flops_sam2 = 810.5
    baseline_JF_sam2 = 0.904

    # Compute score
    epsilon = 1e-6
    df2['delta_flops_sam2'] = baseline_flops_sam2 - df2['FLOPs']
    df2['delta_jf_sam2'] = baseline_JF_sam2 - df2['JF']
    df2['score'] = df2['delta_flops_sam2'] / (df2['delta_jf_sam2'] + epsilon)

    # Log-transform the score to compress dynamic range
    df2['log_score'] = np.log10(df2['score'] + 1)

    print(df2['log_score'])
    best_layers_sam2 = [31,32,34,35,37,38,44,45,46,47]
    # best = df2.nlargest(10, 'score')
    # print(best)

    # styling
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.2)

    # 4:3 aspect ratio
    fig, ax = plt.subplots(figsize=(4, 3))

    # plot
    ax.plot(
        df['Layer'], df['log_score'],
        marker='o', linestyle='-',
        color='#FFA500',  # light orange
        markersize=4,
        zorder=1,
    )

    ax.plot(
        df2['Layer'], df2['log_score'],
        marker='o', linestyle='-',
        color='#89CFF0',
        markersize=4,
        zorder=3,
    )

    # highlight the selected layers
    highlight_df = df[df['Layer'].isin(best_layers)]
    ax.scatter(
        highlight_df['Layer'], highlight_df['log_score'],
        marker='D',
        s=40,
        color='#b86e14',  # dark orange
        linewidth=0.5,
        zorder=10,
        label='SAM - Selected layers'
    )

    highlight_df2 = df2[df2['Layer'].isin(best_layers_sam2)]
    ax.scatter(
        highlight_df2['Layer'], highlight_df2['log_score'],
        marker='D',
        s=40,
        color='#4682B4',  # dark blue
        linewidth=0.5,
        zorder=10,
        label='SAM 2 - Selected layers'
    )

    # 5) annotate each diamond
    #for x, y in zip(highlight_df['Layer'], highlight_df['log_score']):
     #   ax.text(x, y, str(x),
    #            fontsize=6, ha='center', va='center',
     #           zorder=11)

    # 6) x‐ticks every 2 layers
    ax.set_xticks(np.arange(2,
                            df2['Layer'].max() + 1, 4))

    # labels, legend, despine, layout
    # shrink the axis labels
    ax.set_xlabel('Layer', fontsize=10, fontweight='bold')  # was default ~12–14
    ax.set_ylabel('Score', fontsize=10, fontweight='bold')
    plt.xlim([-0.1, 48])
    # shrink the tick labels
    ax.tick_params(axis='both', which='major', labelsize=8)

    # shrink the legend text
    ax.legend(loc='upper left')
    ax.legend(fontsize=8, frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()

    # save it
    # fig.savefig("token_merging_score.svg")  # scalable vector
    # fig.savefig("token_merging_score.pdf")  # if you prefer PDF
    # fig.savefig("token_merging_score.png", dpi=300)  # high-res PNG

    # optional TikZ export:
    # tikz_save("token_merging_score.tex")
    plt.show()