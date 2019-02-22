import numpy as np
import time
from types import SimpleNamespace

from caproto.threading.client import Context
from caproto import ChannelType


def get_pv_container(ctx, attr_to_pvname):
    pvs = ctx.get_pvs(*attr_to_pvname.values())
    return SimpleNamespace(**dict(zip(attr_to_pvname.keys(), pvs)))


def save(prefix='13SIM1:', cam='cam1:', image='image1:'):
    sim_detector_pvs = {
        'acquire': f'{prefix}{cam}Acquire',
        'image_mode': f'{prefix}{cam}ImageMode',
        'image_source': f'{prefix}{image}NDArrayPort',

        'enabled': f'{prefix}{image}EnableCallbacks',

        'unique_id': f'{prefix}{image}UniqueId_RBV',
        'array_size0': f'{prefix}{image}ArraySize0_RBV',
        'array_size1': f'{prefix}{image}ArraySize1_RBV',
        'array_size2': f'{prefix}{image}ArraySize2_RBV',
        'bayer_pattern': f'{prefix}{image}BayerPattern_RBV',
        'array_data': f'{prefix}{image}ArrayData',

        'color_mode': f'{prefix}{cam}ColorMode',
        'color_mode_rbv': f'{prefix}{cam}ColorMode_RBV',
    }

    with Context() as ctx:
        ns = get_pv_container(ctx, sim_detector_pvs)
        for attr in sim_detector_pvs:
            print(attr, '=', getattr(ns, attr).read().data)

        ns.image_source.write('SIM1')
        ns.image_mode.write('Single', data_type=ChannelType.STRING)

        color_modes = [
            enum_string.decode('ascii') for enum_string in
            ns.color_mode_rbv.read(data_type='control').metadata.enum_strings
        ]

        for color_mode in color_modes:
            ns.color_mode.write(color_mode, data_type=ChannelType.STRING,
                                wait=True)
            time.sleep(0.1)
            ns.acquire.write(1, wait=True)
            image = ns.array_data.read().data
            array_size = [size.read().data[0]
                          for size in (ns.array_size0, ns.array_size1,
                                       ns.array_size2)
                          ]
            print(color_mode, array_size, image.shape, image.dtype, image)

            with open(f'{color_mode}.{image.dtype.name}.npz', 'wb') as f:
                np.savez_compressed(f, array_size=array_size, image=image,
                                    color_mode=color_mode)


if __name__ == '__main__':
    save()
