import torch
import numpy as np
import pandas as pd
import copy


def get_random_problems(batch_size, customer_size, aug_type=None):
    depot_size = 1
    node_size = customer_size * 2 + depot_size
    depot_x_y = torch.rand(size=(batch_size, depot_size, 2))
    # shape: (batch, depot, 2)
    pick_x_y = torch.rand(size=(batch_size, customer_size, 2))
    delivery_x_y = torch.rand(size=(batch_size, customer_size, 2))
    # shape: (batch, customer, 2)
    depot_x_y, pick_x_y, delivery_x_y, aug_number = aug(
        aug_type=aug_type,
        depot_x_y=depot_x_y,
        pick_x_y=pick_x_y,
        delivery_x_y=delivery_x_y
    )
    return depot_x_y, pick_x_y, delivery_x_y, aug_number


def get_1_random_problem(batch_size, customer_size, aug_type=None):
    depot_size = 1
    depot_x_y = torch.rand(size=(depot_size, 2)).repeat(batch_size, 1, 1)
    pick_x_y = torch.rand(size=(customer_size, 2)).repeat(batch_size, 1, 1)
    delivery_x_y = torch.rand(size=(customer_size, 2)).repeat(batch_size, 1, 1)
    depot_x_y, pick_x_y, delivery_x_y, aug_number = aug(
        aug_type=aug_type,
        depot_x_y=depot_x_y,
        pick_x_y=pick_x_y,
        delivery_x_y=delivery_x_y
    )
    return depot_x_y, pick_x_y, delivery_x_y, aug_number


def get_dataset_problem(load_path, batch_size, aug_type='8'):
    filename = load_path
    data = pd.read_csv(filename, sep=',', header=None)
    data = data.to_numpy()
    depot_size = 1
    customer_size = int(data[0][0])
    scale = int(data[0][1])
    depot_xy = data[1:depot_size + 1] / scale
    pick_xy = data[depot_size + 1:depot_size + customer_size + 1] / scale
    delivery_xy = data[depot_size + customer_size + 1:depot_size + 2 * customer_size + 1] / scale
    full_node = data[1:depot_size + 2 * customer_size + 1] / scale

    depot_x_y = torch.FloatTensor(depot_xy[0][0:2]).unsqueeze(0)
    for i in range(len(2 * pick_xy)):
        pick_x_y = torch.FloatTensor(pick_xy[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [pick_x_y, torch.FloatTensor(pick_xy[i][0:2]).unsqueeze(0)], dim=0)
        delivery_x_y = torch.FloatTensor(delivery_xy[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [delivery_x_y, torch.FloatTensor(delivery_xy[i][0:2]).unsqueeze(0)], dim=0)
    depot_x_y = depot_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    pick_x_y = pick_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    delivery_x_y = delivery_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    depot_x_y, pick_x_y, delivery_x_y, aug_number = aug(
        aug_type=aug_type, depot_x_y=depot_x_y, pick_x_y=pick_x_y, delivery_x_y=delivery_x_y)
    data = {'depot_x_y': depot_x_y.numpy().tolist(), 'pick_x_y': pick_x_y.numpy().tolist(),
            'delivery_x_y': delivery_x_y.numpy().tolist(),
            'full_node': full_node, 'scale': scale, 'aug_number': aug_number}
    return depot_x_y, pick_x_y, delivery_x_y, customer_size, data, aug_number


def aug(aug_type, depot_x_y, pick_x_y, delivery_x_y):
    aug_number = 1
    if aug_type == '8':
        aug_number = 8
        depot_x_y = augment_x_y_by_8(depot_x_y)
        pick_x_y = augment_x_y_by_8(pick_x_y)
        delivery_x_y = augment_x_y_by_8(delivery_x_y)
    elif aug_type == '9':
        aug_number = 9
        pick_x_y_2 = delivery_x_y.clone()
        delivery_x_y_2 = pick_x_y.clone()
        depot_x_y_1 = augment_x_y_by_8(depot_x_y)
        depot_x_y_2 = depot_x_y.clone()
        pick_x_y_1 = augment_x_y_by_8(pick_x_y)
        delivery_x_y_1 = augment_x_y_by_8(delivery_x_y)

        depot_x_y = torch.cat((depot_x_y_1, depot_x_y_2), dim=0)
        pick_x_y = torch.cat((pick_x_y_1, pick_x_y_2), dim=0)
        delivery_x_y = torch.cat((delivery_x_y_1, delivery_x_y_2), dim=0)
    return depot_x_y, pick_x_y, delivery_x_y, aug_number


def augment_x_y_by_8(x_y):
    # shape: (batch, N, 2)
    x = x_y[:, :, [0]]
    y = x_y[:, :, [1]]
    # shape: (batch, N, 1)

    data1 = torch.cat((x, y), dim=2)
    data2 = torch.cat((1 - x, y), dim=2)
    data3 = torch.cat((x, 1 - y), dim=2)
    data4 = torch.cat((1 - x, 1 - y), dim=2)
    data5 = torch.cat((y, x), dim=2)
    data6 = torch.cat((1 - y, x), dim=2)
    data7 = torch.cat((y, 1 - x), dim=2)
    data8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_x_y = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
    # shape: (8 * batch_size, N, 2)
    return aug_x_y
