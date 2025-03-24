from transformers.utils import is_sagemaker_mp_enabled
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from collections.abc import Mapping
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
from copy import deepcopy
from packaging import version
from trainer.base import FinetuneTrainer

from transformers.trainer_pt_utils import (
    nested_detach,
)


from transformers.utils import (
    is_sagemaker_mp_enabled,
)

from accelerate.utils import (
    is_deepspeed_available,
)

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_only,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_deepspeed_available():
    import deepspeed




from trainer.unlearn.base import UnlearnTrainer
import numpy as np


def print_metrics(step, metrics):
    """
    Print current training metrics in a formatted way.

    Args:
        epoch (int): Current epoch or iteration number.
        metrics (dict): A dictionary containing metric names as keys and their current values.
    """
    # Prepare the formatted string
    metrics_string = ', '.join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    print(f"Step {step}: {metrics_string}")

class GRU(UnlearnTrainer):
    def __init__(self,  gamma_gru=0.8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma_gru = gamma_gru
        self.gradient_accumulation_steps = kwargs["args"].gradient_accumulation_steps

        self.dotp_retain = None
        self.flattened_gradient = 0.0
        self.flattened_memory = 0.0
        self.flattened_memory_old = 0.0
        self.flattened_memory_accumulation = 0.0
        self.structure_map = None

        self.steps = 0

        self.gradient_accum = {}
                        
        self.memory_grad = {}

    def orthogonal_component(self, g, g1):

        g1g1 = self.compute_total_gradient_dot_product(g1, self.structure_map, g1, self.structure_map)
        gg1 = self.dotp_retain
        print(gg1/g1g1)
        projection = gg1/g1g1* g1
        orthogonal = g - projection

        return orthogonal
    
    def store_grads(self, model, loss=None, typ=None):
        """
        Accumulates gradients of specified layers, preserving their original shapes.
        Optionally, adjusts which layers are trainable just before computing gradients.

        Args:
            model (torch.nn.Module): The model from which to store gradients.
            loss (torch.Tensor, optional): The loss tensor to perform backward operation. If provided, will compute gradients.

        Returns:
            None: Modifies internal tensors to store accumulated gradients.
        """

        # Perform backward pass if a loss tensor is provided
        if loss:
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

        # Loop through parameters and accumulate gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                # Choose the correct dictionary based on 'typ'
                if typ == "objective":
                    target_dict = self.gradient_accum
                elif typ == "retain":
                    target_dict = self.memory_grad
                else:
                    raise ValueError("Invalid type specified for gradient storage")

                # Initialize the dictionary key if it doesn't exist
                if name not in target_dict:
                    target_dict[name] = torch.zeros_like(param.grad, device=param.grad.device)  # Initialize on the same device

                # Accumulate the gradients
                target_dict[name] += param.grad.detach()

        if loss:
            model.zero_grad()

    def flatten_and_store_grads(self):
        """
        Flattens accumulated gradients from different gradient dictionaries, moves them to CPU,
        and stores them along with a structure map for each type of gradient.
        """

        # Helper function to flatten gradients, move to CPU, and record their original structure
        def flatten_to_cpu_and_record_structure(gradient_dict):
            flattened_grads = []
            structure_map = []
            for name, grad in gradient_dict.items():
                if grad is not None:
                    grad_flat = grad.view(-1)
                    flattened_grads.append(grad_flat)
                    structure_map.append((name, grad.shape))

            if flattened_grads:
                return torch.cat(flattened_grads).to('cpu'), structure_map
            else:
                return torch.tensor([], dtype=torch.float32).to('cpu'), []

  
        self.flattened_gradient, self.structure_map = flatten_to_cpu_and_record_structure(self.gradient_accum)
      
        self.flattened_memory_accumulation, _ = flatten_to_cpu_and_record_structure(self.memory_grad)

    def compute_total_gradient_dot_product(self, flattened_grads1, structure_map1, flattened_grads2, structure_map2):
        """
        Computes the total dot product between gradients from two sets of flattened gradients and their respective structure maps.

        Args:
            flattened_grads1 (torch.Tensor): The first flattened gradient tensor.
            structure_map1 (list): A list of tuples containing parameter names and their corresponding shapes for the first set of gradients.
            flattened_grads2 (torch.Tensor): The second flattened gradient tensor.
            structure_map2 (list): A list of tuples containing parameter names and their corresponding shapes for the second set of gradients.

        Returns:
            float: The total dot product summed across all matching layers.
        """
        #assert len(structure_map1) == len(structure_map2), "Both gradient structures must contain the same number of elements."

        total_dot_product = 0.0
        index = 0

        # Ensure both gradient tensors are on the same device
        flattened_grads1 = flattened_grads1.to('cuda')
        flattened_grads2 = flattened_grads2.to('cuda')

        # for ((name1, shape1), (name2, shape2)) in zip(structure_map1, structure_map2):
        #     assert name1 == name2 and shape1 == shape2, f"Gradient mismatch: {name1} vs {name2} or {shape1} vs {shape2}"

        for ((name1, shape1), (name2, shape2)) in zip(structure_map1, structure_map2):
            assert shape1 == shape2, f"Gradient mismatch: {name1} vs {name2} or {shape1} vs {shape2}"

            size = np.prod(shape1)  # Total number of elements in this layer's gradient
            grad_slice1 = flattened_grads1[index:index + size].view(shape1)
            grad_slice2 = flattened_grads2[index:index + size].view(shape2)

            # Compute the dot product of the two gradient slices
            dot_product = (grad_slice1 * grad_slice2).sum()
            total_dot_product += dot_product.item()

            index += size

        return total_dot_product

    def restore_gradients_from_flat(self, model):
        """
        Restores gradients to the model's parameters directly from a flattened gradient tensor.

        Args:
            model (torch.nn.Module): The model to which the gradients will be restored.
            flattened_grads (torch.Tensor): The flattened gradient tensor.
            structure_map (list): A list of tuples containing parameter names and their corresponding shapes.
        """

        index = 0  # Index to track position in the flattened gradient tensor

        for name, shape in self.structure_map:
            size = np.prod(shape)  # Total number of elements in this gradient
            if size == 0:  # Skip layers with no parameters
                continue

            # Extract the relevant slice from the flattened gradient tensor
            grad_slice = self.flattened_gradient[index:index + size].view(shape)

            # Find the corresponding parameter in the model
            param = next((p for n, p in model.named_parameters() if n == name), None)
            if param.requires_grad:
                # Check if the shape of the extracted gradient matches the parameter's shape
                if grad_slice.shape != param.shape:
                    raise ValueError(f"Gradient shape mismatch for {name}: expected {param.shape}, got {grad_slice.shape}")

                # Set the parameter's gradient to the extracted slice
                param.grad = grad_slice.to(param.device)

            index += size  # Update index to the start of the next gradient slice

        if index != self.flattened_gradient.numel():
            raise ValueError("Total number of gradient elements does not match the length of the flattened gradient tensor.")
       
    def pipeline(self):
        if self.dotp_retain<0:
            print("dotp_retain:",self.dotp_retain)
            self.flattened_gradient = self.orthogonal_component(self.flattened_gradient, self.flattened_memory)
            torch.cuda.empty_cache()

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        retain_loss += retain_outputs.loss
        return retain_loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to train.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.

            Return:
                `torch.Tensor`: The tensor with training loss on this batch.
            """
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available(min_version="2.0"):
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # GRU overwrite
                #self.accelerator.backward(loss, **kwargs)

                torch.cuda.empty_cache()

                if self.steps % self.gradient_accumulation_steps == 0:

                    # Flatten and move accumulated gradients to CPU before clearing
                    self.flatten_and_store_grads()
                    self.gradient_accum = {}
                    self.memory_grad = {}

                    self.flattened_memory = self.gamma_gru * self.flattened_memory_accumulation + (1 - self.gamma_gru) * self.flattened_memory_old
                    self.flattened_memory_old = self.flattened_memory
                    self.dotp_retain = self.compute_total_gradient_dot_product(self.flattened_gradient, self.structure_map, 
                                                                        self.flattened_memory, self.structure_map)
                    self.pipeline()

                    self.restore_gradients_from_flat(model)
                    self.flattened_memory_accumulation = 0
                    torch.cuda.empty_cache()

            return loss.detach() / self.args.gradient_accumulation_steps
   
    def compute_loss(self, model, inputs, return_outputs=False):

        
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss
        del forget_outputs
        self.store_grads(model, loss=forget_loss, typ = "objective")

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        self.store_grads(model, loss=retain_loss, typ = "retain")

        loss = forget_loss
        self.steps +=1


        metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
                'forget_loss': forget_loss
            }
        print_metrics(self.steps, metrics)

        return (loss, forget_outputs) if return_outputs else loss