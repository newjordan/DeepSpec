from deepspec.data import CacheCollator
from deepspec.modeling.dspark.deepseek2 import DeepseekV2DSparkModel
from deepspec.modeling.dspark.deepseek2.config import (
    build_draft_config as build_deepseek2_draft_config,
)
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer
from transformers import AutoConfig, AutoTokenizer


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


class DeepseekV2DSparkTrainer(Qwen3DSparkTrainer):
    def build_models(self):
        model_args = self.args.model
        target_config_path = getattr(
            model_args,
            "target_config_name_or_path",
            model_args.target_model_name_or_path,
        )
        tokenizer_path = getattr(
            model_args,
            "target_tokenizer_name_or_path",
            target_config_path,
        )
        tokenizer = None
        if bool(getattr(model_args, "load_tokenizer", False)):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        target_config = AutoConfig.from_pretrained(target_config_path)
        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)
        init_mode = str(getattr(model_args, "embedding_init", "random"))
        assert init_mode == "random", (
            "DeepseekV2DSparkTrainer currently supports embedding_init='random' "
            "only. Add a lightweight staged tensor initializer before enabling "
            "target-weight initialization."
        )
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_deepseek2_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return DeepseekV2DSparkModel(draft_config)
