import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def _validate_required_fields(target_config) -> None:
    required_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_bias",
        "attention_dropout",
        "hidden_act",
        "initializer_range",
        "max_position_embeddings",
        "mlp_bias",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "rms_norm_eps",
        "rope_parameters",
        "v_head_dim",
        "kv_lora_rank",
    )
    for field in required_fields:
        assert hasattr(target_config, field), (
            f"target_config.{field} must be provided."
        )


def _get_optional_int(model_args, name: str):
    if name not in model_args:
        return None
    value = getattr(model_args, name)
    if value is None:
        return None
    return int(value)


def build_draft_config(target_config, model_args):
    assert str(target_config.model_type) == "deepseek_v2", (
        "DeepseekV2 DSpark expects a deepseek_v2 target config, "
        f"got model_type={target_config.model_type!r}."
    )
    _validate_required_fields(target_config)

    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = copy.deepcopy(target_config)
    draft_intermediate_size = _get_optional_int(model_args, "draft_intermediate_size")
    if draft_intermediate_size is not None:
        assert draft_intermediate_size > 0
        draft_config.intermediate_size = draft_intermediate_size

    draft_config.architectures = ["DeepseekV2DSparkModel"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
]
