import future, sys, os, datetime, argparse
from typing import List, Tuple
import numpy as np
import matplotlib

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch, torch.nn
from torch import nn
from torch.nn import Sequential, Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

import copy

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

# from Optimization.BayesianGradients.src.DeterministicLayers import GradBatch_Linear as Linear


def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    '''
        Make params regular Tensors instead of nn.Parameter
    '''
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def _set_nested_attr(obj: Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def load_weights(mod: Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


def compute_jacobian(model, x, y, loss_fn):
    '''
    Adapted from: https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240/7
    @param model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
    @param x: input since any gradients requires some input
    @return: either store jac directly in parameters or store them differently

    we'll be working on a copy of the model because we don't want to interfere with the optimizers and other functionality
    '''

    jac_model = copy.deepcopy(model) # because we're messing around with parameters (deleting, reinstating etc)
    all_params, all_names = extract_weights(jac_model) # "deparameterize weights"
    load_weights(jac_model, all_names, all_params) # reinstate all weights as plain tensors

    def param_as_input_func(model, x, y, loss_fn, param):
        load_weights(model, [name], [param]) # name is from the outer scope
        out = model(x)
        loss_out = loss_fn(out, y.to(torch.float))
        return loss_out

    jac_list = []
    for i, (name, param) in enumerate(zip(all_names, all_params)):
        jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, x, y, loss_fn, param), param,
                                                 strict=True if i==0 else False, vectorize=False if i==0 else True)
        jac_list.append(torch.flatten(jac))

    flattened_jacobian = torch.cat(jac_list)

    del jac_model # cleaning up
    return flattened_jacobian


def GraNd_score(model, x, y, loss_fn):
    flattened_jacobian = compute_jacobian(model, x, y, loss_fn)
    score = torch.linalg.norm(flattened_jacobian)
    return score


def EL2N_score(model, x, y, loss_fn):
    with torch.no_grad():
        out = model(x)
        normalized_out = F.softmax(out, dim=-1)
        # torch.linalg.norm is Frobenius norm, which is 2 norm by default
        score = torch.linalg.norm(normalized_out - y.to(torch.float))
        return score


class Net(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()

        self.nn = Sequential(Linear(in_features=input_features, out_features=hidden_features, bias=True),
                     torch.nn.Tanh(),
                     Linear(in_features=hidden_features, out_features=output_features, bias=True),)

    def forward(self, x):
        return self.nn(x)


def main():
    batch_size = 5
    input_features = 4
    hidden_features = 7
    output_features = 3
    net = Net(input_features, hidden_features, output_features)
    x = torch.randn(batch_size, input_features).requires_grad_()
    y = F.one_hot(torch.arange(0, batch_size) % output_features)
    loss_fn = nn.CrossEntropyLoss()
    # compute_jacobian(net, x, y, loss_fn)
    grand_score = GraNd_score(net, x, y, loss_fn)
    el2n_score = EL2N_score(net, x, y, loss_fn)
    print(f"{grand_score=}")
    print(f"{el2n_score=}")


if __name__=="__main__":
    main()
