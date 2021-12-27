import torch
import numpy as np
import random

def neuron_to_cover(layer_neuron_activated_dic, cover_ratio):
    not_covered_all = [(layer_name, index) for (layer_name, index), v in layer_neuron_activated_dic.items() if not v]
    cover_size = (int) (len(not_covered_all) * cover_ratio)
    if not_covered_all:
        if cover_ratio == 1.0:
            neurons_2be_covered = not_covered_all
        else:
            neurons_2be_covered = random.sample(not_covered_all, cover_size)
    else:
        neurons_2be_covered = [] # All neurons are already get activated
    return neurons_2be_covered, not_covered_all

def cal_neurons_cov_loss(layers_output_dict, neurons_2be_covered):
    if len(neurons_2be_covered) == 0:
        return 0.0
    cov_loss = 0
    for (layer_name, neuron_idx) in neurons_2be_covered:
        # cov_loss += torch.mean(layers_output_dict[layer_name][:,neuron_idx,...])
        cov = torch.mean(scale(layers_output_dict[layer_name])[:,neuron_idx,...])
        cov_loss += cov
    return cov_loss

def update_coverage_v2(layers_output_dict, threshold, layer_neuron_activated_dic):
    total_activated_count = count_activated_neurons(layer_neuron_activated_dic)
    total_neuron_count = 0

    activated_cur_update = 0
    for layer_name in layers_output_dict.keys():
        # B X C_out X H X W
        layer_output = layers_output_dict[layer_name]
        activated_count = 0
        for neuron_idx in range(layer_output.shape[1]):
            neuron_output = layer_output[:,neuron_idx,...]
            scaled = scale(neuron_output)
            if (layer_name, neuron_idx) not in layer_neuron_activated_dic.keys():
                layer_neuron_activated_dic[(layer_name, neuron_idx)] = False

            activated = torch.mean(scaled).item() > threshold
            if activated and not layer_neuron_activated_dic[(layer_name, neuron_idx)]:
                layer_neuron_activated_dic[(layer_name, neuron_idx)] = True
                activated_count += 1

        activated_cur_update += activated_count
        total_neuron_count += layer_output.shape[1]
    total_activated_count += activated_cur_update
    return total_activated_count, total_neuron_count

def count_activated_neurons(layer_neuron_activated_dic):
    return np.sum(list(map(lambda x: x == True, layer_neuron_activated_dic.values())))

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled
