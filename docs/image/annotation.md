# Annotation Validation

At this moment pyMDMA only supports the COCO [1] dataset annotation format.

If you wish to test this on other formats such as YOLO [2] or PASCAL VOC [3] you can convert these annotations to the COCO format and execute the metrics over the parsed annotations. Tools such as [voc2coco](https://github.com/yukkyo/voc2coco) and [YOLO-to-COCO-format-converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter) can be used for this purpose.

## COCO Dataset Metrics

### Validity

::: pymdma.image.measures.input_val.annotation.coco.DatasetCompletness
::: pymdma.image.measures.input_val.annotation.coco.AnnotationCorrectness
::: pymdma.image.measures.input_val.annotation.coco.AnnotationUniqueness

______________________________________________________________________

## References

[1] T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” arXiv.org, May 01, 2014. Available: https://arxiv.org/abs/1405.0312v3.

[2] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” arXiv.org, Jun. 08, 2015. Available: https://arxiv.org/abs/1506.02640v5.

[3] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The Pascal Visual Object Classes (VOC) Challenge,” Int J Comput Vis, vol. 88, no. 2, pp. 303–338, Jun. 2010, doi: 10.1007/s11263-009-0275-4
