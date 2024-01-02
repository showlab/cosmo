from .eval_tasks import eval_vqa, eval_captioning, eval_classification, eval_retrieval, eval_video_vqa, eval_video_captioning, eval_zs_classification, eval_video_mc
from .models import eval_base_model

class BaseEvaluator:
    def __init__(self, config: dict, eval_model: eval_base_model.BaseEvalModel, task:str, dataset_name:str, shot:int):
        self.config = config
        self.task = task
        self.dataset_name = dataset_name
        self.shot = shot # if shot is -1, we use all data to training
        self.eval_model = eval_model
    
    
    def get_params(self):
        common_params = {
            "config": self.config,
            "dataset_name": self.dataset_name,
            "eval_model": self.eval_model,
            "num_shots": self.shot
        }
        for task in self.config['tasks']:
                if task['name'] == self.task:
                    if task['params'] is not None:
                        task_params = {k: v for k, v in task['params'].items()}
                        common_params.update(task_params) # Update the common_params with task_params
                    return common_params

        # If we've gone through all tasks and haven't found the right one, raise an error
        raise ValueError(f"No task named '{self.task}' found in the config.")
    
    def evaluate(self):
        raise NotImplementedError

class VQAEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        return eval_vqa.evaluate_vqa(**params)

class CaptioningEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        print(params)
        return eval_captioning.evaluate_captioning(**params)

class ClassificationEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        return eval_classification.evaluate_classification(**params)

class RetrievalEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        return eval_retrieval.evaluate_retrieval(**params)

class VideoVQAEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        return eval_video_vqa.evaluate_video_vqa(**params)

class VideoCaptioningEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        return eval_video_captioning.evaluate_video_captioning(**params)

class ZSClassificationEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        return eval_zs_classification.evaluate_zs_classification(**params)

class VideoMCEvaluator(BaseEvaluator):
    def evaluate(self):
        params = self.get_params()
        return eval_video_mc.evaluate_video_mc(**params)

EVALUATORS = {
    "vqa": VQAEvaluator,
    "captioning": CaptioningEvaluator,
    "imageclassification": ClassificationEvaluator,
    "retrieval": RetrievalEvaluator,
    "video_vqa": VideoVQAEvaluator,
    "video_captioning": VideoCaptioningEvaluator,
    "zs_classification": ZSClassificationEvaluator,
    "video_mc": VideoMCEvaluator
}

def evaluate_model(config: dict, eval_model: eval_base_model.BaseEvalModel, task:str, dataset_name:str, shot:int):
    evaluator_cls = EVALUATORS.get(task)
    if evaluator_cls is None:
        raise ValueError(f"Unsupported task: {task}")
    evaluator = evaluator_cls(config, eval_model, task, dataset_name, shot)
    return evaluator.evaluate()
