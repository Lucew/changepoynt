import typing
import json

import changepoynt.simulation.base as simbase


def from_json(json_str: str) -> typing.Union[simbase.SignalPartCollection, simbase.SignalPart]:

    # try to parse the json from the string so we get an error if invalid
    json_dict = json.loads(json_str)
    json_dict = simbase.decompress_compressed_json(json_dict)

    # check that we received the dict for a single signal
    assert len(json_dict) == 1, 'The dict contains more than one object.'
    object_key = list(json_dict.keys())[0]

    # check whether it is of class SignalPart
    if object_key in simbase.SignalPart.get_registered_signal_parts():

        # construct the thing
        return simbase.SignalPart.from_json(json_str)

    # check whether it is of class SignalPartCollection
    if object_key in simbase.SignalPartCollection.get_registered_signal_parts():

        # construct the thing
        return simbase.SignalPartCollection.from_json(json_str)

    raise ValueError(f'The object {object_key} is not a registered class of type SignalPart or SignalPartCollection.')


def to_json(input_obj: typing.Union[simbase.SignalPartCollection, simbase.SignalPart],
            compress: bool = False) -> str:
    if not isinstance(input_obj, simbase.SignalPart) and not isinstance(input_obj, simbase.SignalPartCollection):
        raise TypeError(f'Object hast to be subclass of SignalPart or SignalPartCollection. It is: {type(input_obj)}.')
    return input_obj.to_json(compress=compress)
