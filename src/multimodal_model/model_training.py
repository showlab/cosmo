from transformers import Trainer, TrainingArguments
from transformers.trainer import *
from typing import Dict, Any
import tqdm
from tqdm import tqdm
import random
from typing import Dict
import torch
from collections import defaultdict
import numpy as np
from torch.utils.data import DistributedSampler
import torch.nn as nn
from typing import Union
import transformers
from transformers.trainer import is_apex_available
from .utils import next_token_predict_accuracy, patch_torch_save, patch_torch_distributed_new_group
try:
    from azfuse import File
except Exception as e:
    print("azfuse not installed, use torch.save instead of azfuse.File.open")
try:
    import deepspeed
except Exception as e:
    print("deepspeed not installed")
if is_apex_available():
    from apex import amp

class CustomTrainer(Trainer):
    def __init__(self, 
                 model, 
                 args: TrainingArguments, 
                 train_dataloader=None, 
                 eval_dataloader=None,
                 model_params: Dict[str, Any] = None,
                 training_params: Dict[str, Any] = None,
                 compute_metrics=None,
                 custom_logger=None,
                 wandb_agent=None,
                 upload_model_to_blob=False,
                 custom_dist_init_group_timeout=None,
                 ):
        super().__init__(model=model,
                         args=args,
                         )
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model_params = model_params
        self.training_params = training_params
        self.custom_logger = custom_logger
        self.compute_metrics = compute_metrics
        self.wandb_agent = wandb_agent
        # for data resampler (tsv file), we do not need write it for wds since wds already implement it
        self.data_resampling = self.training_params['data_resampling']
        if self.data_resampling:
            print("Data resampling is enabled!")
            self.old_epoch = -1.
            self.new_epoch = 0.
        # Initialize moment values for the losses
        self.exception_handling = self.training_params["exception_handling"]
        if self.exception_handling:
            self.moment_lm_loss = 0
            self.moment_cl_loss = 0
            self.moment_beta = 0.9  # Adjust this value based on your preference. 0.9 gives more weight to historical values.
            self.lm_exception_bound = .7  # If the loss is x larger than the moment value, scale down the loss
            self.cl_exception_bound = .7  # If the loss is 
            self.update_flag = False
        # for upload model to blob
        self.upload_model_to_blob = upload_model_to_blob
        if self.upload_model_to_blob:
            self._replace_torch_save_by_azfuse_save()
        self.custom_dist_init_group_timeout = custom_dist_init_group_timeout
        if self.custom_dist_init_group_timeout:
            self._replace_torch_distributed_new_group()
        # # HF will save model in format .safetensors by default, but sometimes we want "pytorch_model.bin"
        # # Comment this line if you want to save safetensors like "model.safetensors", it's faster and smaller
        self.args.save_safetensors = False
        
    def _replace_torch_save_by_azfuse_save(self):
        """
        Since write large model to blob is very slow, we replace torch.save by azfuse.save
        """
        torch.save = patch_torch_save()

    def _replace_torch_distributed_new_group(self):
        """
        Since read data is very slow for remote clusers, we replace torch.distributed.new_group by custom timeout
        """
        torch.distributed.new_group = patch_torch_distributed_new_group()

    def set_epoch_for_distributed_resampler(self):
        """
        Do not call callback of huggingface Trainer since callback_handler.on_epoch_begin has a bug
        """
        if self.args.local_rank == 0:  # Only call resampler.set_epoch() on device 0
            for dataloader in self.train_dataloader.dataloaders:
                if hasattr(dataloader, "dist_sampler") and isinstance(dataloader.dist_sampler, DistributedSampler):
                    if self.new_epoch == 0:
                        dataloader.dist_sampler.set_epoch(0)
                    else:
                        dataloader.increase_dist_sampler_epoch(stop_type='epoch_stop')
            self.old_epoch = self.new_epoch

    
    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)

    def get_train_dataloader(self, train_dataset=None):
        """
        override this method since huggingface trainer only support Pytorch BaseDataset now
        """
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        """
        do not remove eval_dataset=None, since huggingface trainer will pass eval_dataset to this method
        """
        return self.eval_dataloader
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override this method to use your own custom loss function.
        Do not remove return_outputs since huggingface trainer will pass return_outputs in evaluate method
        See https://huggingface.co/docs/transformers/main/model_doc/llama for output details.
        Follow OPT Training, we do:
            1. If parted batch size, (e.g, the last interation in each dataloader) skip this batch (ing...)
            2. If NaN or Inf value in loss, skip this batch
            3. If the loss is too large compared with moment value, skip this batch
        More to do:
            1. How to deal with hardware error (e.g., one node may be down)
        Args:
            model (:obj:`torch.nn.Module`):
                The model that is being trained. If :obj:`model_init` is provided, this must be `None`.
            inputs (:obj:`Dict[str, torch.Tensor]`):
                inputs["visual"]: image/video in shape (B, T_img, F, C, H, W)
                inputs["input_ids"]: text["input_ids"]
                inputs["attention_mask"]: text["attention_mask"]
                inputs["labels"]: text["labels"]
        Returns:
            :custom_loss:`torch.FloatTensor`: The loss value.
        """
        # Retrieve batch size from inputs
        B = inputs["input_ids"].shape[0]
        type_name = inputs["type_name"]
        data_type = inputs["data_type"]
        self.update_flag = True

        if self.data_resampling:
            self.new_epoch = self.state.epoch
            # for new epoch, set the resampler seed
            if int(self.new_epoch) > int(self.old_epoch):
                self.set_epoch_for_distributed_resampler()
                # Determine the batch size based on the type_name
                # for wds
            if data_type == "wds":
                # Assuming you have access to the dataloaders dictionary mapping type names to DataLoader instances
                specific_dataloader = [dataloader for dataloader in self.train_dataloader.dataloaders if dataloader.type_name == type_name][0]
                batch_size_from_dataset = specific_dataloader.dataset.batch_size
            else:
                specific_dataloader = [dataloader for dataloader in self.train_dataloader.dataloaders if dataloader.type_name == type_name][0]
                batch_size_from_dataset = specific_dataloader.batch_size
        else:
            # Determine the batch size based on the type_name
            specific_dataloader = next(d for d in self.train_dataloader.dataloaders if d.type_name == type_name)
            batch_size_from_dataset = getattr(specific_dataloader.dataset, 'batch_size', specific_dataloader.batch_size)
        if batch_size_from_dataset is None:
            raise ValueError(f"batch_size_from_dataset is None for type {type_name}")

        # Pass inputs through the model
        output, contrastive_loss, top1_ce_acc, top5_ce_acc = model(
            vision_x=inputs["visual"],
            lang_x=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            labels=inputs["labels"]
        )
        lm_loss = output[0]
        
        # Step 1: Skip the batch if it's the last one in the dataloader
        if B < batch_size_from_dataset:
            print(f"Warning: Skip this batch for type {type_name} since the batch size {B} is smaller than {batch_size_from_dataset}")
            zero_loss = 0.0 * lm_loss
            self.optimizer.zero_grad(set_to_none=True)
            return zero_loss if not return_outputs else (zero_loss, output)

        # Step1.2: Check if some input["labels"] are the same within the batch due to wds sampling strategy
        if torch.unique(inputs["labels"]).size(0) < B:
            print(f"Warning: Setting contrastive loss to 0 because some text labels are the same!")
            contrastive_loss *= 0.0

        if self.exception_handling:
            if self.moment_lm_loss == 0:
                self.moment_lm_loss = lm_loss.item()
            if self.moment_cl_loss == 0:
                self.moment_cl_loss = contrastive_loss.item()
            # Step 2: Handle potential NaN or Inf values in losses
            lm_loss = self._check_loss_validity(lm_loss, "lm_loss")
            contrastive_loss = self._check_loss_validity(contrastive_loss, "contrastive_loss")
        else:
            # Add some checks for NaN and Inf values in the loss. This is to handle LLAMA's error about :
            # Exception: Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.
            if torch.isnan(lm_loss).any() or torch.isinf(lm_loss).any() or torch.isnan(contrastive_loss).any() or torch.isinf(contrastive_loss).any():
                # self.custom_logger.info("Warning: NaN or Inf value in loss detected. Give zero loss to skip this batch.")
                print("Warning: NaN or Inf value in loss detected.")
                # print("Warning: NaN or Inf value in loss detected. Give zero loss to skip this batch.")
                # zero_loss = 0.0 * lm_loss + 0.0 * contrastive_loss
                # self.optimizer.zero_grad(set_to_none=True)
                # return zero_loss if not return_outputs else (zero_loss, output)
        # Compute the custom loss
        custom_loss = self._calculate_custom_loss(self.model_params, lm_loss, contrastive_loss, inputs["type_name"])

        # Step 3: Weighted loss
        if type_name in self.training_params:
            custom_loss *= self.training_params[type_name]

        acc1, acc5 = next_token_predict_accuracy(output.logits, inputs["labels"], -100, topk=(1, 5))
        if self.exception_handling:
            if self.update_flag:
                self._log_metrics(lm_loss, contrastive_loss, output, inputs, acc1, acc5, top1_ce_acc, top5_ce_acc)       
            else:
                zero_loss = 0.0 * custom_loss
                self.optimizer.zero_grad(set_to_none=True)
                return zero_loss if not return_outputs else (zero_loss, output)
        else:
            self._log_metrics(lm_loss, contrastive_loss, output, inputs, acc1, acc5, top1_ce_acc, top5_ce_acc)
        return custom_loss if not return_outputs else (custom_loss, output)


    def _check_loss_validity(self, loss, loss_name):
        """Helper function to check and handle NaN or Inf values in the loss."""
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN or Inf value in {loss_name} detected.")
            self.update_flag = False
            return loss
        if loss_name == "lm_loss" and loss.item() > self.lm_exception_bound + self.moment_lm_loss:
            print(f"Warning: {loss_name} value {loss.item()} is too large compared with moment value {self.moment_lm_loss}. ")
            self._update_moment_losses(loss, loss_name)
            loss *= (self.cl_exception_bound + self.moment_cl_loss) / loss.item()
            # self.update_flag = False
            return loss
        if loss_name == "contrastive_loss" and loss.item() > self.cl_exception_bound + self.moment_cl_loss:
            print(f"Warning: {loss_name} value {loss.item()} is too large compared with moment value {self.moment_cl_loss}.")
            self._update_moment_losses(loss, loss_name)
            # scale down the loss
            loss *= (self.cl_exception_bound + self.moment_cl_loss) / loss.item()
            # self.update_flag = False
            return loss
        self._update_moment_losses(loss, loss_name)
        return loss



    def _update_moment_losses(self, loss, loss_name):
        """Helper function to update moment values for the losses."""
        # Update moment values for the losses only if the loss is not zero
        if loss.item() != 0 and loss_name == "lm_loss":
            self.moment_lm_loss = self.moment_beta * self.moment_lm_loss + (1 - self.moment_beta) * loss.item()
        if loss.item() != 0 and loss_name == "contrastive_loss":
            self.moment_cl_loss = self.moment_beta * self.moment_cl_loss + (1 - self.moment_beta) * loss.item()


    def _calculate_custom_loss(self, model_params, lm_loss, contrastive_loss, type_name):
        """
        Helper function to calculate custom loss.
        Do not compute contrastive loss for interlevel data since this task is quite simple for such data.
        """
        # if model_params['multimodality_model']['use_contrastive_loss'] and type_name != "inter_img_txt":
        #     return lm_loss + model_params['multimodality_model']['contrastive_loss_weight'] * contrastive_loss
        # if model_params['multimodality_model']['use_contrastive_loss'] and type_name in ["img_txt", "vid_txt"]: # "inter_img_txt"
        if model_params['multimodality_model']['use_contrastive_loss'] and "inter" not in type_name:
        # if model_params['multimodality_model']['use_contrastive_loss']:
            return lm_loss + model_params['multimodality_model']['contrastive_loss_weight'] * contrastive_loss
        return lm_loss # + model_params['multimodality_model']['contrastive_loss_weight'] * contrastive_loss


    def _log_metrics(self, lm_loss, contrastive_loss, output, inputs, acc1, acc5, top1_ce_acc, top5_ce_acc):
        """
        Helper function to log metrics. 
        ! Notice Huggingface Trainer show optimization steps rather than forward steps.
        ! If use gradient accumulation, the number of forward steps = optimization steps x gradient accumulation steps. 
        ! If show train Dataloader type and length: [('img_txt', 125), ('inter_img_txt', 1000), ('vid_txt_tsv', 75)]
        ! means the dataloader will iterate over for the 75th iteration, then the 125th iteration, then the 1000th iteration.
        """
        if random.random() < 0.001:
            message = {
                "type": inputs["type_name"],
                "contrastive_loss timely": contrastive_loss.item(),
                "lm_loss timely": lm_loss.item(),
                "acc1 timely": acc1.item(),
                "acc5 timely": acc5.item(),
                "top1_ce_acc timely": top1_ce_acc.item(),
                "top5_ce_acc timely": top5_ce_acc.item(),
            }
            if self.exception_handling:
                message["contrastive_loss moment"] = self.moment_cl_loss
                message["lm_loss moment"] = self.moment_lm_loss

            self.custom_logger.info(message)
        metrics_to_log = {
            inputs["type_name"]: {
                'lm_loss': lm_loss.item(),
                'cl_loss': contrastive_loss.item() if self.model_params['multimodality_model']['use_contrastive_loss'] else None
            }
        }
        # sometimes the wandb may have self.stream.flush() OSError: [Errno 5] Input/output error for long time training
        if self.wandb_agent:
            try:
                self.wandb_agent.log(metrics_to_log)
            except Exception as e:
                print(f"Error while logging to wandb: {e}")

    def custom_move_inputs_to_device(self, inputs: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        inputs["visual"] = inputs["visual"].to(device, dtype=dtype)
        inputs["input_ids"] = inputs["input_ids"].to(device, dtype=torch.long)
        inputs["attention_mask"] = inputs["attention_mask"].to(device, dtype=dtype)
        inputs["labels"] = inputs["labels"].to(device)
        return inputs

    def custom_eval_step(self, model, inputs, return_outputs=False):
        output, contrastive_loss, top1_ce_acc, top5_ce_acc = model(
            vision_x=inputs["visual"],
            lang_x=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            labels=inputs["labels"],
        )
        lm_loss = output[0]
        logits = output.logits #logits: 128 * 32 * 50267, labels: 128 * 32
        acc1, acc5 = next_token_predict_accuracy(logits, inputs["labels"], -100, topk=(1, 5))
        return ([contrastive_loss.item(), lm_loss.item()], acc1, acc5, top1_ce_acc, top5_ce_acc) if not return_outputs else ([contrastive_loss.item(), lm_loss.item()], output, acc1, acc5, top1_ce_acc, top5_ce_acc)


    def evaluate(self, eval_dataset=None, ignore_keys=None):
        """
        override this method to use your own custom evaluation function
        show the evaluation results (both loss and accuracy) for each type of data
        """
        eval_dataloader = self.get_eval_dataloader()
        max_eval_batches = min(len(eval_dataloader), self.training_params['max_eval_batches'])
        self.model.eval()
        device = next(self.model.parameters()).device
        self.custom_logger.info(f"Evaluating {len(eval_dataloader)} batches")
        self.cast_dtype = next(self.model.parameters()).dtype

        total_acc1 = defaultdict(int)
        total_acc5 = defaultdict(int)
        total_ce_acc1 = defaultdict(int)
        total_ce_acc5 = defaultdict(int)
        total_samples = defaultdict(int)
        total_losses = defaultdict(lambda: np.array([0.0, 0.0]))

        # To calculate averages over all types
        total_acc1_all = 0
        total_acc5_all = 0
        total_ce_acc1_all = 0
        total_ce_acc5_all = 0
        total_samples_all = 0
        total_losses_all = np.array([0.0, 0.0])


        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
                if step > max_eval_batches:
                    break
                inputs = self.custom_move_inputs_to_device(inputs, device, self.cast_dtype)
                losses, acc1, acc5, top1_ce_acc, top5_ce_acc = self.custom_eval_step(self.model, inputs, return_outputs=False)
                type_name = inputs["type_name"]
                num_samples = inputs["input_ids"].shape[0]
                total_acc1[type_name] += acc1.item() * num_samples  # acc1 per batch * batch size
                total_acc5[type_name] += acc5.item() * num_samples  # acc5 per batch * batch size
                total_ce_acc1[type_name] += top1_ce_acc.item() * num_samples  # acc1 per batch * batch size
                total_ce_acc5[type_name] += top5_ce_acc.item() * num_samples  # acc5 per batch * batch size
                total_samples[type_name] += num_samples  # total number of samples
                total_losses[type_name] += np.array(losses) * num_samples  # sum of losses * batch size

                # Update total counts
                total_acc1_all += acc1.item() * num_samples
                total_acc5_all += acc5.item() * num_samples
                total_ce_acc1_all += top1_ce_acc.item() * num_samples
                total_ce_acc5_all += top5_ce_acc.item() * num_samples
                total_samples_all += num_samples
                total_losses_all += np.array(losses) * num_samples
                
        for type_name in total_samples.keys():
            average_acc1 = total_acc1[type_name] / total_samples[type_name]
            average_acc5 = total_acc5[type_name] / total_samples[type_name]
            average_ce_acc1 = total_ce_acc1[type_name] / total_samples[type_name]
            average_ce_acc5 = total_ce_acc5[type_name] / total_samples[type_name]
            average_losses = total_losses[type_name] / total_samples[type_name]
            
            self.custom_logger.info({
                "type": type_name, 
                "average_acc1": average_acc1, 
                "average_acc5": average_acc5, 
                "average_ce_acc1": average_ce_acc1,
                "average_ce_acc5": average_ce_acc5,
                "average_contrastive_loss": average_losses[0], 
                "average_lm_loss": average_losses[1]
            })
            
            if self.wandb_agent is not None:
                try:
                    self.wandb_agent.log({
                        f"val_{type_name}_average_acc1": average_acc1, 
                        f"val_{type_name}_average_acc5": average_acc5, 
                        f"val_{type_name}_average_ce_acc1": average_ce_acc1,
                        f"val_{type_name}_average_ce_acc5": average_ce_acc5,
                        f"val_{type_name}_average_contrastive_loss": average_losses[0], 
                        f"val_{type_name}_average_lm_loss": average_losses[1]
                    }, commit=False)
                except Exception as e:
                    print(f"Error while logging to wandb: {e}")
        
        # Calculate and log averages over all types
        average_acc1_all = total_acc1_all / total_samples_all
        average_acc5_all = total_acc5_all / total_samples_all
        average_ce_acc1_all = total_ce_acc1_all / total_samples_all
        average_ce_acc5_all = total_ce_acc5_all / total_samples_all
        average_losses_all = total_losses_all / total_samples_all

        self.custom_logger.info({
            "type": "all", 
            "average_acc1": average_acc1_all, 
            "average_acc5": average_acc5_all, 
            "average_ce_acc1": average_ce_acc1_all,
            "average_ce_acc5": average_ce_acc5_all,
            "average_contrastive_loss": average_losses_all[0], 
            "average_lm_loss": average_losses_all[1]
        })

        if self.wandb_agent is not None:
            try:
                self.wandb_agent.log({
                    f"val_all_average_acc1": average_acc1_all, 
                    f"val_all_average_acc5": average_acc5_all, 
                    f"val_all_average_ce_acc1": average_ce_acc1_all,
                    f"val_all_average_ce_acc5": average_ce_acc5_all,
                    f"val_all_average_contrastive_loss": average_losses_all[0], 
                    f"val_all_average_lm_loss": average_losses_all[1]
                }, commit=True)
            except Exception as e:
                print(f"Error while logging to wandb: {e}")

        return {'lm_top1': average_acc1_all, 'lm_top5': average_acc5_all, 'ce_top1': average_ce_acc1_all, 'ce_top5': average_ce_acc5_all, 'eval_loss': average_losses_all[1]}
        # return average_losses

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        For transformer version <= 4.29.2, the model save will be fast.
        For transofrmer version >= 4.30.0, state_dict = self.accelerator.get_state_dict(self.deepspeed) [line 2828 in https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/trainer.py] is very slow even more than 30 minutes. 
        We override this method to speed up.
        """
        transformers_version = transformers.__version__
        if transformers_version <= "4.29.2":
            super().save_model(output_dir, _internal_call)
        else:
            if output_dir is None:
                output_dir = self.args.output_dir
            if is_torch_tpu_available():
                self._save_tpu(output_dir)
            elif is_sagemaker_mp_enabled():
                # Calling the state_dict needs to be done on the wrapped model and on all processes.
                os.makedirs(output_dir, exist_ok=True)
                state_dict = self.model_wrapped.state_dict()
                if self.args.should_save:
                    self._save_new(output_dir, state_dict=state_dict)
                if IS_SAGEMAKER_MP_POST_1_10:
                    # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                    Path(os.path.join(output_dir, "user_content.pt")).touch()
            elif version.parse(transformers_version) > version.parse("4.35.2") and self.is_fsdp_enabled:
                if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                    version.parse(accelerate_version) > version.parse("0.24.1")
                ):
                    state_dict = self.accelerator.get_state_dict(self.model)
                    if self.args.should_save:
                        self._save(output_dir, state_dict=state_dict)
            elif version.parse(transformers_version) <= version.parse("4.35.2") and self.fsdp is not None or self.is_fsdp_enabled:
                state_dict = self.model.state_dict() if not self.is_fsdp_enabled else {}
                if self.args.should_save:
                    self._save_new(output_dir, state_dict=state_dict)
                if self.is_fsdp_enabled:
                    # remove the dummy state_dict
                    remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                    save_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir)

            elif self.is_deepspeed_enabled:
                # this takes care of everything as long as we aren't under zero3
                if version.parse(accelerate_version) <= version.parse("0.20.3"):
                    raise ValueError("Install Accelerate from main branch")
                try:
                    # state_dict = self.model_wrapped.state_dict()
                    # state_dict = self.accelerator.get_state_dict(self.deepspeed)
                    state_dict = self.accelerator.unwrap_model(self.deepspeed).state_dict()
                    if self.args.should_save:
                        # self._save_new(output_dir)
                        # torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                        self._save_new(output_dir, state_dict=state_dict)
                        # self.accelerate.save_state(state_dict, output_dir)
                except ValueError:
                    logger.warning(
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    if self.args.should_save:
                        self._save_new(output_dir, state_dict={})
                    # remove the dummy state_dict
                    remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                    self.model_wrapped.save_checkpoint(output_dir)

            elif self.args.should_save:
                self._save_new(output_dir)

            # Push to the Hub when `save_model` is called by the user.
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        

    def _save_new(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        if self.args.save_safetensors:
            safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
        else:
            torch.save(unwrap_model(self.model).state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
            # torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save_checkpoint_new(self, model, trial, metrics=None):
        """
        This function is from hf directly.
        But we replace it with older transformers version (v4.35.2) to  handle following error for newer transformers:
            os.rename(staging_output_dir, output_dir) FileNotFoundError: [Errno 2] No such file or directory: '/tmp-checkpoint-5000' -> '/checkpoint-5000'
        """
       # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.model_wrapped.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.fsdp or self.is_fsdp_enabled:
            if self.is_fsdp_enabled:
                save_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
                )
            else:
                # FSDP has a different interface for saving optimizer states.
                # Needs to be called on all ranks to gather all states.
                # full_optim_state_dict will be deprecated after Pytorch 2.2!
                full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)
                torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif self.args.should_save and not self.is_deepspeed_enabled and not (self.fsdp or self.is_fsdp_enabled):
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_tpu_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        if is_torch_npu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["npu"] = torch.npu.random.get_rng_state_all()
            else:
                rng_states["npu"] = torch.npu.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


    # def _save_checkpoint_new(self, model, trial, metrics=None):
    #     """
    #     Handle error:     os.rename(staging_output_dir, output_dir) FileNotFoundError: [Errno 2] No such file or directory: '/tmp-checkpoint-5000' -> '/checkpoint-5000'

    #     """
    #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #     if self.hp_search_backend is None and trial is None:
    #         self.store_flos()

    #     run_dir = self._get_output_dir(trial=trial)
    #     output_dir = os.path.join(run_dir, checkpoint_folder)
    #     if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    #         logger.warning(
    #             f"Checkpoint destination directory {output_dir} already exists and is non-empty."
    #             "Saving will proceed but saved results may be invalid."
    #         )
    #         staging_output_dir = output_dir
    #     else:
    #         staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
    #     self.save_model(staging_output_dir, _internal_call=True)

    #     if not self.args.save_only_model:
    #         # Save optimizer and scheduler
    #         self._save_optimizer_and_scheduler(staging_output_dir)
    #         # Save RNG state
    #         self._save_rng_state(staging_output_dir)

    #     # Determine the new best metric / best model checkpoint
    #     if metrics is not None and self.args.metric_for_best_model is not None:
    #         metric_to_check = self.args.metric_for_best_model
    #         if not metric_to_check.startswith("eval_"):
    #             metric_to_check = f"eval_{metric_to_check}"
    #         metric_value = metrics[metric_to_check]

    #         operator = np.greater if self.args.greater_is_better else np.less
    #         if (
    #             self.state.best_metric is None
    #             or self.state.best_model_checkpoint is None
    #             or operator(metric_value, self.state.best_metric)
    #         ):
    #             self.state.best_metric = metric_value
    #             self.state.best_model_checkpoint = output_dir

    #     # Save the Trainer state
    #     if self.args.should_save:
    #         self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

    #     if self.args.push_to_hub:
    #         self._push_from_checkpoint(staging_output_dir)

    #     # Place checkpoint in final location after all saving is finished.
    #     # First wait for everyone to finish writing
    #     self.args.distributed_state.wait_for_everyone()
    #     # Then go through the rewriting process starting on process 0
    #     if staging_output_dir != output_dir:
    #         with self.args.main_process_first(desc="Renaming model checkpoint folder to true location"):
    #             if os.path.exists(staging_output_dir):
    #                 try:
    #                     os.rename(staging_output_dir, output_dir)
    #                 except FileNotFoundError:
    #                     # Sometimes the rename fails, we suspect an NFS issue but have not been able to fully
    #                     # understand when it happens. We ignore it and only raise a warning.
    #                     logger.warning(
    #                         f"Error renaming {staging_output_dir} to {output_dir}. "
    #                         "You can find the checkpoint as well in the former."
    #                     )
    #     # Maybe delete some older checkpoints.
    #     if self.args.should_save:
    #         self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _maybe_log_save_evaluate_new(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint_new(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # def _save_new(self, output_dir: Optional[str] = None, state_dict=None):
    #     # If we are executing this function, we are the process zero, so we don't check for that.
    #     output_dir = output_dir if output_dir is not None else self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     logger.info(f"Saving model checkpoint to {output_dir}")
    #     # Save a trained model and configuration using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     if not isinstance(self.model, PreTrainedModel):
    #         if state_dict is None:
    #             state_dict = self.model.state_dict()

    #         if isinstance(unwrap_model(self.model), PreTrainedModel):
    #             unwrap_model(self.model).save_pretrained(
    #                 output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
    #             )
    #         else:
    #             logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
    #             if self.args.save_safetensors:
    #                 safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
    #             else:
    #                 torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    #     else:
    #         self.model.save_pretrained(
    #             output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
    #         )

    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)

    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        If deepspeed version is less than 0.8.3, use _inner_training_loop_old.
        Otherwise, use _inner_training_loop_new.
        """
        deepspeed_version = deepspeed.__version__
        if deepspeed_version <= "0.8.3":
            self._inner_training_loop_old(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
        else:
            self._inner_training_loop_new(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

    def _inner_training_loop_old(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        HF's implementation, modify:
            1. change log to tar 
        """
        skip_first_batches = None # add this line to avoid error
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa
        try:
            delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
            )
        except AttributeError:
             delay_optimizer_creation = is_sagemaker_mp_enabled() or self.fsdp is not None or self.is_fsdp_enabled
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            print("load pretrained model from deepspeed!")
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        print("***** Running training *****")
        print(f"  Num examples = {num_examples:,}")
        print(f"  Num Epochs = {num_train_epochs:,}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_steps:,}")
        print(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print(f"  Continuing training from epoch {epochs_trained}")
            print(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    print(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    print(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    (total_batched_samples % args.gradient_accumulation_steps != 0)
                    and args.parallel_mode == ParallelMode.DISTRIBUTED
                    and args._no_sync_in_gradient_accumulation
                    and hasattr(model, "no_sync")
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)


                # training_step
                # # ============ add this part to skip the batch ============
                # if self.exception_handling:
                #     # Check if the current process has a zero tensor.
                #     skip_batch = tr_loss_step.item() == 0.0

                #     # Represent the decision as a tensor. 
                #     # If skip_batch is True, decision_tensor is 0, else 1.
                #     decision_tensor = torch.tensor(0. if skip_batch else 1.).to(args.device)
                    
                #     # Use all_reduce to sum up decision tensors from all processes.
                #     torch.distributed.all_reduce(decision_tensor, op=torch.distributed.ReduceOp.SUM)
                    
                #     # Synchronize all processes
                #     dist.barrier()

                #     # If the sum is less than the total number of processes, then at least one process has a zero tensor.
                #     total_processes = torch.distributed.get_world_size()
                #     if decision_tensor.item() < total_processes:
                #         print(f"Skip this batch in training loop since at least one process encountered a zero tensor.")
                #         continue


                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))


                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                self.deepspeed.step()
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    # if save large ckpt, this would be very slow, how to speed up
                    if self.upload_model_to_blob:
                        with File.async_upload(enabled=True, shm_as_tmp=True):
                            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            if self.upload_model_to_blob:
                with File.async_upload(enabled=True, shm_as_tmp=True):
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            else:
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        print("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            # this line may report RuntimeError
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    print(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _inner_training_loop_new(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa
        # <=4.35.2
        if version.parse(transformers.__version__) <= version.parse("4.35.2"):              
            delay_optimizer_creation = is_sagemaker_mp_enabled() or self.fsdp is not None or self.is_fsdp_enabled
        else:
            delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        print("***** Running training *****")
        print(f"  Num examples = {num_examples:,}")
        print(f"  Num Epochs = {num_train_epochs:,}")
        print(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            print(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_steps:,}")
        print(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print(f"  Continuing training from epoch {epochs_trained}")
            print(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                print(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    if self.upload_model_to_blob:
                        with File.async_upload(enabled=True, shm_as_tmp=True):
                            self._maybe_log_save_evaluate_new(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self._maybe_log_save_evaluate_new(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            if self.upload_model_to_blob:
                with File.async_upload(enabled=True, shm_as_tmp=True):
                    self._maybe_log_save_evaluate_new(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            else:
                self._maybe_log_save_evaluate_new(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        print("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    print(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)