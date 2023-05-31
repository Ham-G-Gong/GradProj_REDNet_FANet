import json
from lpcvc.loader.lpcvc_loader import LPCVCLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
	"lpcvc": LPCVCLoader,
    }[name]
