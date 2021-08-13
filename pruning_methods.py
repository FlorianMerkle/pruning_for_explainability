from torch import nn
import torch.nn.utils.prune as prune


class _BasePruning:
    """ Base Pruning Class for Pruning Methods

    This class provides substantial functionality for different pruning methods.
    """

    def __init__(self):
        self.model = nn.Module
        self.pruning_percentage = 0.0
        self.pruning_method = ''

    def __call__(self, model, pruning_percentage=0.5):
        self.model = self.model_checker(model)
        self.pruning_percentage = self.pruning_percentage_checker(pruning_percentage)
        self.pruning_printer()
        return self.pruning()

    def pruning_printer(self):
        print('-' * 10)
        print('---Pruning - {}---'.format(self.pruning_method))

    @staticmethod
    def model_checker(model):
        if not isinstance(model, nn.Module):
            raise TypeError('Model is not of type nn.Module')
        return model

    @staticmethod
    def pruning_percentage_checker(pruning_percentage):
        if not isinstance(pruning_percentage, float):
            raise TypeError('Pruning percentage is not a float')
        return pruning_percentage

    def pruning(self):
        raise NotImplementedError()


class LocalMagnitudeUnstructured(_BasePruning):
    """ A Class for Local Magnitude Unstructured Pruning

    This class embeds the pytorch prune.l1_unstructred function to prune CNN layers.
    """

    def __init__(self):
        super().__init__()
        self.pruning_method = 'local_magnitude_unstructured'

    def pruning(self):
        for name, module in self.model.named_modules():
            # do Conv2d pruning: relevant for GradCAM
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_percentage)
                print('Conv2d Module {} pruned with {} by {}'.format(name, self.pruning_method, self.pruning_percentage))
        return self.pruning_method


class LocalRandomUnstructured(_BasePruning):
    """ A Class for Local Random Unstructured Pruning

    This class embeds the pytorch prune.random_unstructred function to prune CNN layers.
    """

    def __init__(self):
        super().__init__()
        self.pruning_method = 'local_random_unstructured'

    def pruning(self):
        for name, module in self.model.named_modules():
            # do Conv2d pruning
            if isinstance(module, nn.Conv2d):
                prune.random_unstructured(module, name='weight', amount=self.pruning_percentage)
                print('Conv2d Module {} pruned with {} by {}'.format(name, self.pruning_method, self.pruning_percentage))
        return self.pruning_method


class LocalMagnitudeStructured(_BasePruning):
    def __init__(self):
        super().__init__()
        self.pruning_method = 'local_magnitude_structured'

    def pruning(self):
        # prune.ln_structured(n=1) # L1
        raise NotImplementedError()


class LocalRandomStructured(_BasePruning):
    def __init__(self):
        super().__init__()
        self.pruning_method = 'local_random_structured'

    def pruning(self):
        # prune.random_structured()
        raise NotImplementedError()


class GlobalMagnitudeUnstructured(_BasePruning):
    def __init__(self):
        super().__init__()
        self.pruning_method = 'global_magnitude_unstructured'

    def pruning(self):
        # prune.global_unstructured(pruning_method=prune.L1Unstructured)
        raise NotImplementedError()
