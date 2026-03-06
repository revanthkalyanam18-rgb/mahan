import splitfolders

splitfolders.ratio(
    "cell_images",
    output="data",
    seed=42,
    ratio=(0.7,0.15,0.15)
)