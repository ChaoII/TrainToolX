def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
            sample_bbox[2] <= object_bbox[0] or \
            sample_bbox[1] >= object_bbox[3] or \
            sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap