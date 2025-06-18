# Image Guides

## Dealing with large scale datasets

When performing `input validation` on large-scale image datasets, it is important to consider the memory requirements of your metrics. To prevent out-of-memory errors, it is recommended to use batch loading—processing the dataset in smaller chunks rather than loading all images at once.

You can achieve this by using the `build_img_dataloader` function provided in the `pymdma.image.data` module, which creates a dataloader that loads images in batches. Alternatively, you can use your own custom dataloader, as long as it yields batches in a compatible format for the metric computation.

```python
from pathlib import Path
from pymdma.image.data import build_img_dataloader
from pymdma.image.measures.input_val import Brightness

# obtain image file paths
image_files = list(Path("path-to-large-img-dataset").iterdir())

dataloader = build_img_dataloader(
    file_paths=image_files,
    batch_size=10,
    num_workers=5,
)

# compute brightness results and save them
brightness = Brightness()
brightness_results = []
for imgs, _, _ in dataloader:
    brightness_results += brightness.compute(imgs).value[1]

print("Brightness results:", brightness_results)
```
