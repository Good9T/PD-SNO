import copy

import torch
from dataclasses import dataclass
from Problem import get_random_problems, get_1_random_problem, get_dataset_problem
from copy import deepcopy

@dataclass
class Reset_State:
    depot_x_y: torch.Tensor = None
    # shape: (batch, 1, 2)
    pick_x_y: torch.Tensor = None
    # shape: (batch, customer, 2)
    delivery_x_y: torch.Tensor = None
    # shape: (batch, customer, 2)
    customer_size: torch.int = None
    mt_size: torch.int = None

@dataclass
class Backup_State:
    depot_x_y: torch.Tensor = None
    # shape: (batch, 1, 2)
    pick_x_y: torch.Tensor = None
    # shape: (batch, customer, 2)
    delivery_x_y: torch.Tensor = None
    # shape: (batch, customer, 2)
    customer_size: torch.int = None
    mt_size: torch.int = None

@dataclass
class Step_State:
    batch_idx: torch.Tensor = None
    mt_idx: torch.Tensor = None
    selected_count: int = None
    current_node: torch.Tensor = None
    finished: torch.Tensor = None
    # shape: (batch, mt)
    mask: torch.Tensor = None
    # shape: (batch, mt, node)




class EnvEval:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.customer_size = None
        self.mt_size = None
        self.node_size = None
        self.random_dataset = self.env_params['random_dataset']

        if self.random_dataset:
            self.customer_size = self.env_params['customer_size']
            self.mt_size = self.env_params['mt_size']

        else:
            self.load_path = self.env_params['load_path']

        self.batch_idx = None
        self.mt_idx = None
        # shape: (batch, mt)
        self.batch_size = None
        self.all_node_x_y = None
        # shape: (batch, node, 2)

        self.backup_all_node_x_y = None

        self.selected_count = None
        self.current_node = None
        # shape: (batch, mt)
        self.selected_node_list = None
        # shape: (batch, mt, 0~)

        self.at_the_pick = None
        self.finished = None
        # shape: (batch, mt)
        self.visited_flag = None
        self.mask = None
        self.lock = None
        # shape: (batch, mt, node)

        self.reset_state = Reset_State()
        self.step_state = Step_State()
        self.backup_state = Backup_State()

    def load_random_problems(self, batch_size, sample, aug_type='8'):
        self.batch_size = batch_size
        if sample:
            depot_x_y, pick_x_y, delivery_x_y, aug_factor = get_1_random_problem(
                batch_size=batch_size,
                customer_size=self.customer_size,
                aug_type=aug_type
            )
        else:
            depot_x_y, pick_x_y, delivery_x_y, aug_factor = get_random_problems(
                batch_size=batch_size,
                customer_size=self.customer_size,
                aug_type=aug_type
            )

        self.node_size = self.customer_size * 2 + 1
        self.batch_size = batch_size * aug_factor
        depot_x_y = depot_x_y.to('cuda:0')
        pick_x_y = pick_x_y.to('cuda:0')
        delivery_x_y = delivery_x_y.to('cuda:0')
        self.all_node_x_y = torch.cat((depot_x_y, pick_x_y, delivery_x_y), dim=1)
        # shape: (batch, node, 2)
        self.batch_idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.mt_size)
        self.mt_idx = torch.arange(self.mt_size)[None, :].expand(self.batch_size, self.mt_size)

        self.reset_state.depot_x_y = depot_x_y
        self.reset_state.pick_x_y = pick_x_y
        self.reset_state.delivery_x_y = delivery_x_y
        self.reset_state.customer_size = pick_x_y.size(1)

        self.step_state.batch_idx = self.batch_idx
        self.step_state.mt_idx = self.mt_idx

    def load_dataset_problem(self, batch_size, sample, aug_type='8'):
        if sample:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        depot_x_y, pick_x_y, delivery_x_y, customer_size, data, aug_number = get_dataset_problem(
            load_path=self.load_path,
            batch_size=self.batch_size,
            aug_type=aug_type
        )
        depot_x_y = depot_x_y.to('cuda:0')
        pick_x_y = pick_x_y.to('cuda:0')
        delivery_x_y = delivery_x_y.to('cuda:0')
        self.customer_size = customer_size
        self.mt_size = customer_size
        self.batch_size = self.batch_size * aug_number

        self.all_node_x_y = torch.cat((depot_x_y, pick_x_y, delivery_x_y), dim=1)
        # shape: (batch, node, 2)
        self.batch_idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.mt_size)
        self.mt_idx = torch.arange(self.mt_size)[None, :].expand(self.batch_size, self.mt_size)

        self.reset_state.depot_x_y = depot_x_y
        self.reset_state.pick_x_y = pick_x_y
        self.reset_state.delivery_x_y = delivery_x_y
        self.reset_state.customer_size = pick_x_y.size(1)

        self.step_state.batch_idx = self.batch_idx
        self.step_state.mt_idx = self.mt_idx
        return data

    def backup_problem(self):
        self.backup_state.depot_x_y = deepcopy(self.reset_state.depot_x_y)
        self.backup_state.pick_x_y = deepcopy(self.reset_state.pick_x_y)
        self.backup_state.delivery_x_y = deepcopy(self.reset_state.delivery_x_y)
        self.backup_state.customer_size = deepcopy(self.customer_size)
        self.backup_state.mt_size = deepcopy(self.mt_size)

    def load_backup_problem(self):
        self.reset_state.depot_x_y = deepcopy(self.backup_state.depot_x_y)
        self.reset_state.pick_x_y = deepcopy(self.backup_state.pick_x_y)
        self.reset_state.delivery_x_y = deepcopy(self.backup_state.delivery_x_y)
        self.reset_state.customer_size = deepcopy(self.backup_state.customer_size)
        self.reset_state.mt_size = deepcopy(self.backup_state.mt_size)
        self.batch_idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.mt_size)
        self.mt_idx = torch.arange(self.mt_size)[None, :].expand(self.batch_size, self.mt_size)
        self.step_state.batch_idx = self.batch_idx
        self.step_state.mt_idx = self.mt_idx

    def reset(self):
        self.node_size = self.customer_size * 2 + 1
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.mt_size, 0), dtype=torch.long)
        self.at_the_pick = torch.zeros(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        self.visited_flag = torch.zeros(size=(self.batch_size, self.mt_size, self.node_size))
        self.lock = torch.zeros(size=(self.batch_size, self.mt_size, self.node_size))
        self.lock[:, :, 1+self.customer_size:] = float('-inf')
        self.mask = torch.zeros(size=(self.batch_size, self.mt_size, self.node_size))
        # lock all delivery node at initial state
        self.finished = torch.zeros(size=(self.batch_size, self.mt_size), dtype=torch.bool)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.mask = self.mask
        self.step_state.finished = self.finished
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        self.at_the_pick = (selected < 1 + self.customer_size) * (selected > 0)
        self.visited_flag[self.batch_idx, self.mt_idx, selected] = float('-inf')
        unlock = selected.clone()
        unlock[self.at_the_pick] += self.customer_size
        self.lock[self.batch_idx, self.mt_idx, unlock] = 0
        self.mask = self.visited_flag.clone() + self.lock.clone()

        self.mask = self.mask.clone()

        new_finished = (self.visited_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + new_finished

        self.mask[:, :, 0][self.finished] = 0
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.mask = self.mask
        self.step_state.finished = self.finished
        done = self.finished.all()
        if done:
            reward = -self.get_travel_distance()
        else:
            reward = None

        return self.step_state, reward, done

    def get_travel_distance(self):
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_t_x_y = self.all_node_x_y[:, None, :, :].expand(-1, self.mt_size, -1, -1)
        seq_ordered = all_t_x_y.gather(dim=2, index=index_to_gather)
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        # shape : (batch, mt)
        return travel_distances





