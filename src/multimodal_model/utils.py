import torch
import datetime
try:
    from azfuse import File
except Exception as e:
    print("azfuse not installed, use torch.save instead of azfuse.File.open")

def next_token_predict_accuracy(output, target, padding, topk=(1,)):
  """
  Next koen prediction accuracy, so we need to shift the target to the right by one.
  Computes the accuracy over the k top predictions for the specified values of k
  """
  output = output[:, :-1]
  target = target[:, 1:]
  with torch.no_grad():
    maxk = max(topk)
    if output.shape[-1] < maxk:
      print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

    maxk = min(maxk, output.shape[-1])
    batch_size = target.size(0)

    # Take topk along the last dimension.
    _, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

    mask = (target != padding).type(target.dtype)
    target_expand = target[..., None].expand_as(pred)
    correct = pred.eq(target_expand)
    correct = correct * mask[..., None].expand_as(correct)

    res = []
    for k in topk:
      correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / mask.sum()))
    return res


def torch_save_patch(origin_save, obj, f, *args, **kwargs):
    if isinstance(f, str):
        with File.open(f, 'wb') as fp:
            result = origin_save(obj, fp, *args, **kwargs)
    else:
        result = torch.save(obj, f, *args, **kwargs)
    return result


def patch_torch_save():
    old_save = torch.save
    torch.save = lambda *args, **kwargs: torch_save_patch(old_save, *args, **kwargs)
    return old_save


def patch_torch_distributed_new_group(timeout=5400):
    old_new_group = torch.distributed.new_group
    torch.distributed.new_group = lambda *args, **kwargs: old_new_group(*args, **kwargs, timeout=datetime.timedelta(seconds=timeout))
    return old_new_group