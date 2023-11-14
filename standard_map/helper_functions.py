import six
import numpy as np
import torch
import torch.optim as optim


class HelperClass:

    def set_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _preprocess_thetas(self):
        return self.thetas / np.pi - 1

    def _postprocess_thetas(self):
        return (self.thetas + 1) * np.pi


def type_name(expected_type):
    messages = {
        six.string_types: 'string',
        np.ndarray: 'numpy array',
        list: 'list',
        bool: 'bool',
        int: 'int',
        dict: 'dictionary',
        str: 'str',
    }
    return messages[expected_type]


def validate_data_type(input_object, expected_type, error_prefix='Input object'):
    if not isinstance(input_object, expected_type):
        error_message = '{0}: {1} \nis not of type {2}'.format(error_prefix, str(input_object), type_name(expected_type))
        raise AssertionError(error_message)
