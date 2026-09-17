"""Vision towers for Gemma3, Llama4, Mistral3, Qwen2-VL, and Qwen2.5-VL."""

from __future__ import annotations

import torch

from engine.detect import detect_missing, detect_vision_family
from engine.vision_mm import (
    gemma3_vision_expected_shapes,
    gemma3_vision_forward,
    llama4_vision_expected_shapes,
    llama4_vision_forward,
    mistral3_vision_expected_shapes,
    mistral3_vision_forward,
    multimodal_embeddings,
    qwen25vl_vision_expected_shapes,
    qwen25vl_vision_forward,
    qwen2vl_vision_expected_shapes,
    qwen2vl_vision_forward,
    validate_vision_weights,
)


def test_vision_families_are_detected() -> None:
    assert (
        detect_vision_family(
            {
                "model_type": "qwen2_vl",
                "vision_config": {"hidden_size": 16},
            }
        )
        == "qwen2_vl"
    )
    assert (
        detect_vision_family(
            {
                "model_type": "qwen2_5_vl",
                "vision_config": {"hidden_size": 16},
            }
        )
        == "qwen2_5_vl"
    )
    assert (
        detect_vision_family(
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "vision_config": {"hidden_size": 16},
            }
        )
        == "gemma3"
    )
    assert (
        detect_vision_family(
            {
                "model_type": "llama4",
                "architectures": ["Llama4ForConditionalGeneration"],
                "vision_config": {"hidden_size": 16},
            }
        )
        == "llama4"
    )
    assert (
        detect_vision_family(
            {
                "model_type": "mistral3",
                "architectures": ["Mistral3ForConditionalGeneration"],
                "vision_config": {"hidden_size": 16},
            }
        )
        == "mistral3"
    )
    assert (
        detect_vision_family(
            {
                "model_type": "internvl",
                "text_config": {"model_type": "qwen2"},
                "vision_config": {"hidden_size": 16},
            }
        )
        is None
    )
    assert "vision" in detect_missing(
        {
            "model_type": "internvl",
            "text_config": {"model_type": "qwen2"},
            "vision_config": {"hidden_size": 16},
        },
        "internvl",
    )


def test_qwen2vl_tiny_parity() -> None:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel,
    )

    torch.manual_seed(4)
    cfg = Qwen2VLVisionConfig(
        depth=2,
        embed_dim=16,
        hidden_size=32,
        hidden_act="quick_gelu",
        mlp_ratio=2,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
    )
    cfg._attn_implementation = "eager"
    hf = Qwen2VisionTransformerPretrainedModel(cfg).eval()
    weights = {f"visual.{name}": tensor.detach() for name, tensor in hf.state_dict().items()}
    validate_vision_weights("qwen2_vl", weights, {"vision_config": cfg.to_dict()})
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(16, 3 * 1 * 2 * 2)
    with torch.inference_mode():
        expected = hf(pixels, grid_thw=grid, return_dict=True).pooler_output
        actual = qwen2vl_vision_forward(pixels, grid, weights, cfg.to_dict())
    assert actual.shape == (4, 32)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_qwen25vl_tiny_parity() -> None:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLVisionConfig,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    torch.manual_seed(6)
    cfg = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        hidden_act="silu",
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=32,
        window_size=4,
        fullatt_block_indexes=(1,),
    )
    cfg._attn_implementation = "eager"
    hf = Qwen2_5_VisionTransformerPretrainedModel(cfg).eval()
    weights = {f"visual.{name}": tensor.detach() for name, tensor in hf.state_dict().items()}
    validate_vision_weights("qwen2_5_vl", weights, {"vision_config": cfg.to_dict()})
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(16, 3 * 1 * 2 * 2)
    with torch.inference_mode():
        expected = hf(pixels, grid_thw=grid, return_dict=True).pooler_output
        actual = qwen25vl_vision_forward(pixels, grid, weights, cfg.to_dict())
    assert actual.shape == (4, 32)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_gemma3_vision_tiny_parity() -> None:
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    torch.manual_seed(2)
    vcfg = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=8,
        patch_size=4,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        vision_use_head=False,
    )
    vcfg._attn_implementation = "eager"
    hf = SiglipVisionModel(vcfg).eval()
    raw = {
        "image_token_id": 99,
        "mm_tokens_per_image": 4,
        "hidden_size": 32,
        "vision_config": vcfg.to_dict(),
    }
    weights = {
        f"vision_tower.{name}": tensor.detach() for name, tensor in hf.state_dict().items()
    }
    weights["multi_modal_projector.mm_input_projection_weight"] = torch.randn(16, 32)
    weights["multi_modal_projector.mm_soft_emb_norm.weight"] = torch.zeros(16)
    validate_vision_weights("gemma3", weights, raw)
    pixels = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        hf_hidden = hf(pixel_values=pixels, return_dict=True).last_hidden_state
        actual = gemma3_vision_forward(pixels, weights, vcfg.to_dict(), raw)
    pooled = torch.nn.functional.avg_pool2d(
        hf_hidden.transpose(1, 2).reshape(1, 16, 2, 2), kernel_size=1, stride=1
    )
    # 8/4=2 patches per side; mm_tokens=4 → tokens_per_side=2, kernel=1
    assert actual.shape == (4, 32)
    assert gemma3_vision_expected_shapes(vcfg.to_dict(), 32)[
        "vision_tower.embeddings.patch_embedding.weight"
    ] == (16, 3, 4, 4)
    assert hf_hidden.shape[-1] == 16
    assert pooled.numel() == 64


def test_gemma3_embeds_scatter() -> None:
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    torch.manual_seed(3)
    vcfg = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
        vision_use_head=False,
    )
    vcfg._attn_implementation = "eager"
    hf = SiglipVisionModel(vcfg).eval()
    raw = {
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "image_token_id": 7,
        "mm_tokens_per_image": 4,
        "hidden_size": 32,
        "vision_config": vcfg.to_dict(),
    }
    weights = {
        f"vision_tower.{name}": tensor.detach() for name, tensor in hf.state_dict().items()
    }
    weights["multi_modal_projector.mm_input_projection_weight"] = torch.randn(16, 32)
    weights["multi_modal_projector.mm_soft_emb_norm.weight"] = torch.zeros(16)
    input_ids = torch.tensor([[1, 7, 7, 7, 7, 2]])
    text = torch.randn(1, 6, 32)
    pixels = torch.randn(1, 3, 8, 8)
    embeds, positions, delta = multimodal_embeddings(
        raw, input_ids, text, weights, pixel_values=pixels, recipe_id="gemma3"
    )
    assert embeds.shape == text.shape
    assert positions is None and delta is None
    assert not torch.equal(embeds[:, 1:5], text[:, 1:5])
    assert torch.equal(embeds[:, 0], text[:, 0])


def test_llama4_vision_shapes_and_forward() -> None:
    from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig
    from transformers.models.llama4.modeling_llama4 import Llama4VisionModel

    torch.manual_seed(8)
    cfg = Llama4VisionConfig(
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_channels=3,
        image_size=8,
        patch_size=4,
        pixel_shuffle_ratio=0.5,
        projector_input_dim=32,
        projector_output_dim=32,
        vision_output_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    cfg._attn_implementation = "eager"
    hf = Llama4VisionModel(cfg).eval()
    mapped = {f"vision_model.{name}": tensor.detach() for name, tensor in hf.state_dict().items()}
    mapped["multi_modal_projector.linear_1.weight"] = torch.randn(16, 32)
    raw = {"hidden_size": 16, "vision_config": cfg.to_dict(), "image_token_id": 5}
    validate_vision_weights("llama4", mapped, raw)
    pixels = torch.randn(1, 3, 8, 8)
    features = llama4_vision_forward(pixels, mapped, cfg.to_dict())
    # 8/4=2 patches; +cls then drop; pixel shuffle 0.5 → 1 token
    assert features.shape[-1] == 16
    assert llama4_vision_expected_shapes(cfg.to_dict(), 16)[
        "vision_model.patch_embedding.linear.weight"
    ] == (32, 3 * 4 * 4)


def test_mistral3_vision_tiny_parity() -> None:
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

    torch.manual_seed(7)
    vcfg = PixtralVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
        num_channels=3,
        hidden_act="silu",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    vcfg._attn_implementation = "eager"
    text = {
        "hidden_size": 16,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 32,
    }
    wrapper = Mistral3Config(
        vision_config=vcfg,
        text_config=text,
        image_token_index=9,
        spatial_merge_size=2,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
        vision_feature_layer=-1,
    )
    vision = PixtralVisionModel(vcfg).eval()
    projector = Mistral3MultiModalProjector(wrapper).eval()
    raw = wrapper.to_dict()
    raw["image_token_id"] = 9
    weights = {
        f"vision_tower.{name}": tensor.detach()
        for name, tensor in vision.state_dict().items()
    }
    weights.update(
        {
            f"multi_modal_projector.{name}": tensor.detach()
            for name, tensor in projector.state_dict().items()
        }
    )
    validate_vision_weights("mistral3", weights, raw)
    pixels = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        hf_hidden = vision(pixel_values=pixels, return_dict=True).last_hidden_state
        expected = projector(hf_hidden.squeeze(0), torch.tensor([[8, 8]]))
        actual = mistral3_vision_forward(
            pixels, weights, vcfg.to_dict(), raw, image_sizes=torch.tensor([[8, 8]])
        )
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert mistral3_vision_expected_shapes(vcfg.to_dict(), 16)[
        "vision_tower.patch_conv.weight"
    ] == (16, 3, 4, 4)


def test_qwen2vl_expected_shapes() -> None:
    shapes = qwen2vl_vision_expected_shapes(
        {
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 32,
            "mlp_ratio": 2,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 1,
        }
    )
    assert shapes["visual.patch_embed.proj.weight"] == (16, 3, 1, 2, 2)
    shapes = qwen25vl_vision_expected_shapes(
        {
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "out_hidden_size": 24,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 1,
        }
    )
    assert shapes["visual.merger.mlp.2.weight"] == (24, 64)
