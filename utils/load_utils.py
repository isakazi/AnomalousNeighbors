import torch


def edge_tvt_split(ei):
    """
    Splits edges into 85:5:10 train val test partition
    (Following route of VGRNN paper)
    :param ei:
    :return:
    """
    if isinstance(ei, tuple):
        ne = ei[0].size(1)
    else:
        ne = ei.size(1)
    val = int(ne * 0.85)
    te = int(ne * 0.90)
    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)

    masks[0, rnd[:val]] = True
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True

    return masks[0], masks[1], masks[2]


def edge_tv_split(ei, v_size=0.05):
    """
    Splits edges into train-validation sets (in this case we assume that the
    :param ei:
    :param v_size:
    :return:
    """
    ne = ei.size(1)
    val = int(ne * v_size)

    masks = torch.zeros(2, ne).bool()
    rnd = torch.randperm(ne)
    masks[1, rnd[:val]] = True
    masks[0, rnd[val:]] = True

    return masks[0], masks[1]
